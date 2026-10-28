"""Bpod port discovery and bpod-core setup (StreamDIO finds its own ports)."""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

import serial
from serial.tools import list_ports

from hf_dm.hardware.bringup_constants import DEFAULT_BAUDRATE

logger = logging.getLogger(__name__)


def list_com_ports() -> list[dict[str, str]]:
    ports: list[dict[str, str]] = []
    for port in list_ports.comports():
        device = port.device or ""
        if device:
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


def configure_bpod_core_settings_path() -> Path:
    """Use an application-specific bpod-core settings directory."""
    import bpod_core.bpod as bpod_module
    import bpod_core.bpod.constants as bpod_constants

    configured = os.environ.get("HF_DM_BPOD_CORE_CONFIG_DIR")
    if configured:
        config_path = Path(configured).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parents[3]
        config_path = repo_root / "data" / "runtime" / "bpod-core"

    bpod_module.CONFIG_PATH = config_path
    bpod_constants.CONFIG_PATH = config_path
    return config_path


def ensure_bpod_core_settings_unlocked(config_path: Path, timeout_s: float = 1.0) -> None:
    """Fail fast if another task process is holding this app's bpod-core lock."""
    try:
        from filelock import FileLock, Timeout
    except Exception:
        return

    lock_path = (config_path / "settings.json").with_suffix(".lock")
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=max(0.0, float(timeout_s))):
            return
    except Timeout as exc:
        raise RuntimeError(
            "The bpod-hf-dm bpod-core settings lock is held by another task process, "
            "so Bpod startup would hang. Close the old hf-dm-run/python process and "
            f"retry. Lock file: {lock_path}"
        ) from exc


def connect_bpod_core(port: str | None = None):
    """Connect with bpod-core after bounded bringup-style port discovery."""
    from bpod_core.bpod import Bpod

    config_path = configure_bpod_core_settings_path()
    ensure_bpod_core_settings_unlocked(config_path)
    if port and port != "AUTO":
        resolved = port
    else:
        resolved = resolve_bpod_port("AUTO")
        gc.collect()

    bpod = Bpod(port=resolved)
    try:
        if not any(module.is_connected for module in bpod.modules.values()):
            logger.info("No Bpod modules detected at startup; refreshing module discovery once")
            bpod.update_modules()
    except BaseException:
        bpod.close()
        raise
    return bpod


def streamdio_serial_port(config_port: str, stream_port: str) -> str:
    """The StreamDIO's third USB serial interface (the Teensy's ``Serial``).

    Opening it at 134 baud reboots that board, and no other, into its bootloader:
    the way to flash it while the Bpod's Teensy is on the same host.
    """
    ports = {port.device: port for port in list_ports.comports()}
    serial_number = getattr(ports.get(config_port), "serial_number", None)
    if not serial_number:
        raise RuntimeError(f"No USB serial number for StreamDIO port {config_port}")
    siblings = [
        device
        for device, port in ports.items()
        if port.serial_number == serial_number and device not in {config_port, stream_port}
    ]
    if len(siblings) != 1:
        raise RuntimeError(
            f"Expected one more serial port of StreamDIO {serial_number}, found {siblings}"
        )
    return siblings[0]
