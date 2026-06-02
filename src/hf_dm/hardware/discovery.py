"""Port discovery compatible with MM_PyBpod2 bringup notebooks."""

from __future__ import annotations

import gc
import time
from typing import Any

import serial
from serial.tools import list_ports

from hf_dm.hardware.bringup_constants import (
    DEFAULT_BAUDRATE,
    STREAMDIO_CONFIG_DISCOVERY,
    STREAMDIO_STREAM_DISCOVERY,
)


def list_com_ports() -> list[dict[str, str]]:
    ports: list[dict[str, str]] = []
    for port in list_ports.comports():
        device = port.device or ""
        if device.upper().startswith("COM"):
            ports.append(
                {
                    "device": device,
                    "description": port.description or "",
                    "manufacturer": port.manufacturer or "",
                    "hwid": port.hwid or "",
                }
            )
    return ports


def bpod_identity_probe(port: str, baud: int = DEFAULT_BAUDRATE, timeout: float = 0.35) -> dict[str, Any]:
    try:
        serial_obj = serial.Serial(port, baudrate=baud, timeout=timeout, rtscts=True)
        time.sleep(0.08)
        serial_obj.reset_input_buffer()
        serial_obj.reset_output_buffer()
        serial_obj.write(b"6")
        handshake = serial_obj.read(1)
        if handshake != b"5":
            serial_obj.close()
            return {"is_bpod": False, "why": "no_handshake", "handshake": handshake}

        serial_obj.reset_input_buffer()
        serial_obj.write(b"F")
        ident = serial_obj.read(4)
        try:
            serial_obj.write(b"Z")
            serial_obj.read(1)
        except Exception:
            pass
        serial_obj.close()

        if len(ident) != 4:
            return {"is_bpod": False, "why": "wrong_ident_length", "ident": ident}

        fw_major = int.from_bytes(ident[:2], "little")
        machine_type = int.from_bytes(ident[2:], "little")
        is_bpod = machine_type in {1, 2, 3, 4} and fw_major > 0
        return {
            "is_bpod": is_bpod,
            "why": "ok" if is_bpod else "bad_identity",
            "fw_major": fw_major,
            "machine_type": machine_type,
        }
    except Exception as exc:
        return {"is_bpod": False, "why": f"{type(exc).__name__}: {exc}"}


def resolve_bpod_port(port: str = "AUTO", baud: int = DEFAULT_BAUDRATE) -> str:
    if port != "AUTO":
        return port
    probes = {p["device"]: bpod_identity_probe(p["device"], baud=baud) for p in list_com_ports()}
    candidates = [p for p, probe in probes.items() if probe.get("is_bpod")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple Bpod candidates: {candidates}. Set bpod_port explicitly.")
    raise RuntimeError(f"No Bpod found. Probes: {probes}")


def probe_streamdio_discovery(port: str, baud: int, timeout_s: float = 0.45) -> dict[str, Any]:
    result = {"port": port, "config_hits": 0, "stream_hits": 0, "error": None}
    serial_obj: serial.Serial | None = None
    try:
        serial_obj = serial.Serial(port, baudrate=baud, timeout=0.03, write_timeout=0.03, rtscts=True)
        time.sleep(0.05)
        serial_obj.reset_input_buffer()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            chunk = serial_obj.read(max(1, serial_obj.in_waiting or 1))
            if chunk:
                result["config_hits"] += chunk.count(bytes([STREAMDIO_CONFIG_DISCOVERY]))
                result["stream_hits"] += chunk.count(bytes([STREAMDIO_STREAM_DISCOVERY]))
            time.sleep(0.005)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:
                pass
    return result


def classify_streamdio_probe(result: dict[str, Any]) -> str:
    if result["error"]:
        return "error"
    if result["config_hits"] > 0 and result["stream_hits"] == 0:
        return "config"
    if result["stream_hits"] > 0 and result["config_hits"] == 0:
        return "stream"
    if result["config_hits"] > 0 and result["stream_hits"] > 0:
        return "ambiguous"
    return "none"


def resolve_streamdio_ports(
    dio_config_port: str = "AUTO",
    dio_stream_port: str = "AUTO",
    *,
    excluded_port: str | None = None,
    baud: int = DEFAULT_BAUDRATE,
) -> tuple[str, str]:
    if dio_config_port != "AUTO" and dio_stream_port != "AUTO":
        return dio_config_port, dio_stream_port

    ports = [p["device"] for p in list_com_ports() if p["device"] != excluded_port]
    probes = {port: probe_streamdio_discovery(port, baud) for port in ports}

    config_candidates = [p for p, pr in probes.items() if classify_streamdio_probe(pr) == "config"]
    stream_candidates = [p for p, pr in probes.items() if classify_streamdio_probe(pr) == "stream"]

    selected_config = dio_config_port
    selected_stream = dio_stream_port

    if selected_config == "AUTO":
        if len(config_candidates) != 1:
            raise RuntimeError(f"Expected 1 StreamDIO config port, got {config_candidates} from {probes}")
        selected_config = config_candidates[0]

    if selected_stream == "AUTO":
        if len(stream_candidates) != 1:
            raise RuntimeError(f"Expected 1 StreamDIO stream port, got {stream_candidates} from {probes}")
        selected_stream = stream_candidates[0]

    return selected_config, selected_stream


def connect_bpod_core(port: str | None = None):
    """Connect using bpod-core discovery (preferred) with bringup fallback."""
    from bpod_core.bpod import Bpod, discover_bpod

    if port and port != "AUTO":
        return Bpod(port=port)

    try:
        info = next(discover_bpod())
        return Bpod(port=info.port)
    except StopIteration:
        resolved = resolve_bpod_port("AUTO")
        gc.collect()
        return Bpod(port=resolved)
