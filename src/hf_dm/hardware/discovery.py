"""Port discovery compatible with MM_PyBpod2 bringup notebooks."""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

import serial
from serial.tools import list_ports

from hf_dm.hardware.bringup_constants import (
    DEFAULT_BAUDRATE,
    STREAMDIO_CONFIG_DISCOVERY,
    STREAMDIO_STREAM_DISCOVERY,
)

logger = logging.getLogger(__name__)


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


def _port_info_by_device() -> dict[str, dict[str, str]]:
    return {p["device"]: p for p in list_com_ports()}


def _serial_from_hwid(hwid: str) -> str | None:
    for part in hwid.split():
        if part.startswith("SER="):
            return part[4:] or None
    return None


def _interface_from_hwid(hwid: str) -> int | None:
    marker = ":x."
    if marker not in hwid:
        return None
    tail = hwid.rsplit(marker, 1)[-1]
    digits = ""
    for char in tail:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


def _infer_config_sibling(stream_port: str, probes: dict[str, dict[str, Any]]) -> str | None:
    """Infer USB1/config from the detected USB2/stream sibling on the same Teensy."""
    info = _port_info_by_device()
    stream_info = info.get(stream_port)
    if not stream_info:
        return None

    serial_number = _serial_from_hwid(stream_info.get("hwid", ""))
    stream_interface = _interface_from_hwid(stream_info.get("hwid", ""))
    if not serial_number:
        return None

    siblings: list[tuple[int, str]] = []
    for port, port_info in info.items():
        if port == stream_port:
            continue
        if _serial_from_hwid(port_info.get("hwid", "")) != serial_number:
            continue
        interface = _interface_from_hwid(port_info.get("hwid", ""))
        if interface is None:
            continue
        probe = probes.get(port, {})
        if classify_streamdio_probe(probe) == "stream":
            continue
        siblings.append((interface, port))

    if stream_interface is not None:
        preferred = [port for interface, port in siblings if interface < stream_interface]
        if len(preferred) == 1:
            return preferred[0]

    if len(siblings) == 1:
        return siblings[0][1]
    return None


def _infer_streamdio_pair(probes: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    """Infer config/stream ports from a two-interface StreamDIO USB device."""
    info = _port_info_by_device()
    grouped: dict[str, list[tuple[int, str]]] = {}
    for port in probes:
        port_info = info.get(port)
        if not port_info:
            continue
        serial_number = _serial_from_hwid(port_info.get("hwid", ""))
        interface = _interface_from_hwid(port_info.get("hwid", ""))
        if serial_number is None or interface is None:
            continue
        if classify_streamdio_probe(probes[port]) == "error":
            continue
        grouped.setdefault(serial_number, []).append((interface, port))

    candidates: list[tuple[str, str]] = []
    for ports in grouped.values():
        unique = sorted(set(ports))
        if len(unique) < 2:
            continue
        config_port = unique[0][1]
        stream_port = unique[-1][1]
        candidates.append((config_port, stream_port))

    if len(candidates) == 1:
        return candidates[0]
    return None


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
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
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
    excluded_ports: set[str] | None = None,
    baud: int = DEFAULT_BAUDRATE,
) -> tuple[str, str]:
    if dio_config_port != "AUTO" and dio_stream_port != "AUTO":
        return dio_config_port, dio_stream_port

    excluded = {p for p in (excluded_ports or set()) if p}
    if excluded_port:
        excluded.add(excluded_port)
    ports = [p["device"] for p in list_com_ports() if p["device"] not in excluded]
    probes = {port: probe_streamdio_discovery(port, baud) for port in ports}

    config_candidates = [p for p, pr in probes.items() if classify_streamdio_probe(pr) == "config"]
    stream_candidates = [p for p, pr in probes.items() if classify_streamdio_probe(pr) == "stream"]
    inferred_pair = _infer_streamdio_pair(probes)

    selected_config = dio_config_port
    selected_stream = dio_stream_port

    if selected_config == "AUTO":
        if len(config_candidates) == 1:
            selected_config = config_candidates[0]
        elif len(stream_candidates) == 1:
            inferred = _infer_config_sibling(stream_candidates[0], probes)
            if inferred is None:
                raise RuntimeError(f"Expected 1 StreamDIO config port, got {config_candidates} from {probes}")
            selected_config = inferred
        elif inferred_pair is not None:
            selected_config = inferred_pair[0]
        else:
            raise RuntimeError(f"Expected 1 StreamDIO config port, got {config_candidates} from {probes}")

    if selected_stream == "AUTO":
        if len(stream_candidates) == 1:
            selected_stream = stream_candidates[0]
        elif inferred_pair is not None and selected_config == inferred_pair[0]:
            selected_stream = inferred_pair[1]
        else:
            raise RuntimeError(f"Expected 1 StreamDIO stream port, got {stream_candidates} from {probes}")

    return selected_config, selected_stream


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


def configure_bpod_event_allocation(bpod: Any) -> dict[str, Any]:
    """Assign firmware event slots to modules and USB SoftCodes.

    bpod-core compiles the event table locally but, as of 0.1.0a12, does not send
    the firmware ``%`` command that installs the same allocation on the Bpod.
    Without it, host SoftCodes and module events are silently ignored.
    """
    if bpod.is_running:
        raise RuntimeError("Cannot configure Bpod event allocation during a trial")

    hardware = bpod._hardware
    input_description = bytes(hardware.input_description)
    modules = list(bpod.modules.values())
    requested_module_counts = [int(module.n_events) for module in modules]
    n_usb_channels = input_description.count(b"X") + input_description.count(b"Z")
    max_serial_events = int(hardware.max_serial_events)
    if n_usb_channels <= 0:
        raise RuntimeError("Bpod reports no USB SoftCode input channel")
    n_serial_channels = len(modules) + n_usb_channels
    baseline = max_serial_events // n_serial_channels
    module_minimums = [
        requested if bool(getattr(module, "is_connected", False)) else 0
        for module, requested in zip(modules, requested_module_counts, strict=True)
    ]
    module_counts = [baseline] * len(modules)

    for module_index, minimum in enumerate(module_minimums):
        extra_needed = max(0, minimum - module_counts[module_index])
        module_counts[module_index] += extra_needed
        for donor_index in reversed(range(len(module_counts))):
            if extra_needed <= 0:
                break
            if donor_index == module_index:
                continue
            available = max(0, module_counts[donor_index] - module_minimums[donor_index])
            transfer = min(extra_needed, available)
            module_counts[donor_index] -= transfer
            extra_needed -= transfer
        if extra_needed:
            raise RuntimeError(
                "Bpod modules requested too many event slots: "
                f"{requested_module_counts} of {max_serial_events}"
            )

    remaining = max_serial_events - sum(module_counts)
    if remaining < n_usb_channels or remaining % n_usb_channels:
        raise RuntimeError(
            "Cannot assign all Bpod serial events without shifting the event decoder: "
            f"{module_counts} modules leave {remaining} events for {n_usb_channels} USB channels"
        )

    softcodes_per_usb = remaining // n_usb_channels
    for module, count in zip(modules, module_counts, strict=True):
        module.n_events = count
        if hasattr(module, "_define_event_names"):
            module._define_event_names()

    if hasattr(bpod, "_compile_input_events"):
        from bpod_core.bpod._threads import _build_event_lookup

        bpod._compile_input_events()
        bpod._event_lookup = _build_event_lookup(bpod._input_events, bpod._actions)
        bpod._hardware_hash = bpod._compute_hardware_hash()
    else:
        bpod._n_softcodes = softcodes_per_usb

    counts: list[int] = []
    module_index = 0
    for channel_type in input_description:
        if channel_type == ord("U"):
            counts.append(module_counts[module_index])
            module_index += 1
        elif channel_type in (ord("X"), ord("Z")):
            counts.append(softcodes_per_usb)

    expected_channels = len(module_counts) + n_usb_channels
    if (
        len(counts) != expected_channels
        or sum(counts) != max_serial_events
        or any(not 0 <= count <= 255 for count in counts)
    ):
        raise RuntimeError(
            "Invalid Bpod event allocation "
            f"{counts}; expected {expected_channels} serial channels"
        )

    bpod.serial0.write(bytes((ord("%"), *counts)))
    if not bpod.serial0.read_bool():
        raise RuntimeError(f"Bpod rejected event allocation {counts}")

    allocation = {
        "event_counts": counts,
        "requested_module_event_counts": requested_module_counts,
        "module_event_counts": module_counts,
        "softcodes_per_usb": softcodes_per_usb,
        "allocated_events": sum(counts),
        "max_serial_events": max_serial_events,
    }
    logger.info("Configured Bpod event allocation: %s", allocation)
    return allocation


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
        bpod.hf_dm_event_allocation = configure_bpod_event_allocation(bpod)
    except BaseException:
        bpod.close()
        raise
    return bpod
