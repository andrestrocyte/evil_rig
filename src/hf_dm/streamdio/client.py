"""StreamDIO client: the ``streamdio`` driver, speaking evil_rig's channel numbers."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

from streamdio import ModuleConfig, StreamDIO

from hf_dm.hardware.bringup_constants import (
    CAMERA_TRIGGER_PIN,
    CAP_TOUCH_INPUT_CHANNEL,
    CAP_TOUCH_INPUT_PIN,
    DEFAULT_CAMERA_EXPOSURE_US,
    DEFAULT_CAMERA_FREQUENCY_HZ,
    SPEAKER_GATE_PIN,
    STREAMDIO_TIMING_HEARTBEAT_PERIOD_US,
)
from hf_dm.streamdio.channel_map import (
    ANALOG_CHANNELS,
    CAMERA,
    INPUT_CHANNELS,
    LICK,
    OUTPUT_CHANNELS,
    SPEAKER,
)
from hf_dm.timing_heartbeat import TIMING_HEARTBEAT_EVENT_TYPE

logger = logging.getLogger(__name__)

# The in-RAM history is a debugging view: the session recorder keeps the stream.
DRIVER_HISTORY_EVENTS = 50_000
CAMERA_STOP_TIMEOUT_S = 0.5
# Detector state as the fixed-layout firmware reported its polarity.
_POLARITY_CODES = {None: 0, "rising": 1, "falling": 2}


def _json_safe(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _number_table(dio: StreamDIO, event_type: str, logical: dict[str, int]) -> dict[int, int]:
    """Module channel number -> evil_rig channel number, for the running map."""
    table: dict[int, int] = {}
    number = 1
    while (name := dio.channel_name(event_type, number)) is not None:
        if name in logical:
            table[number] = logical[name]
        number += 1
    return table


class StreamDIOClient:
    """The StreamDIO module, with events and outputs in evil_rig's channel numbers.

    Events reach the callback as the rest of evil_rig records them: ``dio_input``,
    ``dio_output`` and ``analog_input`` carry evil_rig channel numbers, each camera
    trigger is one ``camera`` event (its rising edge), and the module's delivery
    heartbeat is a ``timing_heartbeat`` event. Other events pass through unchanged.
    """

    def __init__(self, config_port: str = "AUTO", stream_port: str = "AUTO", *, baudrate: int):
        self.config_port = config_port
        self.stream_port = stream_port
        self.baudrate = baudrate  # USB serial: the module ignores it
        self._dio: StreamDIO | None = None
        self._callback: Callable[..., None] | None = None
        self._config_lock = threading.Lock()
        self._map: ModuleConfig | None = None
        self._inputs: dict[int, int] = {}
        self._outputs: dict[int, int] = {}
        self._module_outputs: dict[int, int] = {}
        self._analog: dict[int, int] = {}
        self._camera_output: int | None = None
        self._output_condition = threading.Condition()
        self._output_states: dict[int, int] = {}
        self._output_sequences: dict[int, int] = defaultdict(int)
        self._output_events: dict[int, tuple[int, int, float, float]] = {}
        self._output_event_history: dict[
            int,
            deque[tuple[int, int, int, float, float]],
        ] = defaultdict(lambda: deque(maxlen=64))
        self._last_buffer_stats: dict[str, int] = {}
        self._camera_events: list[tuple[int, float]] = []

    def open(self) -> None:
        """Connect to both USB interfaces; ``AUTO`` ports are found by listening."""
        config_port = None if self.config_port == "AUTO" else self.config_port
        stream_port = None if self.stream_port == "AUTO" else self.stream_port
        try:
            self._dio = StreamDIO(
                config_port=config_port,
                stream_port=stream_port,
                history_maxlen=DRIVER_HISTORY_EVENTS,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open StreamDIO config={self.config_port}, stream={self.stream_port}. "
                "If Windows reports 'Access is denied', close old hf-dm-run/python "
                "processes, serial monitors, or unplug/replug the StreamDIO USB device."
            ) from exc
        self.config_port = self._dio.config_port
        self.stream_port = self._dio.stream_port

    def apply_map(self, channel_map: ModuleConfig) -> dict[str, Any]:
        """Make ``channel_map`` the module's map; outputs start at their initial level.

        The driver checks it against the board, applies it and compares the hash it
        reads back. Any Bpod object created earlier must re-read the module
        (``bpod.update_modules()``): its event names and count follow the map.
        """
        dio = self._require_dio()
        with self._config_lock:
            dio.apply_config(channel_map)
            status = dio.config_status()
            dio.start_latency_heartbeat(STREAMDIO_TIMING_HEARTBEAT_PERIOD_US // 1000)
            self._map = channel_map
            self._inputs = _number_table(dio, "dio_input", INPUT_CHANNELS)
            self._outputs = _number_table(dio, "dio_output", OUTPUT_CHANNELS)
            self._module_outputs = {
                logical: number for number, logical in self._outputs.items()
            }
            self._analog = _number_table(dio, "analog_in", ANALOG_CHANNELS)
            self._camera_output = next(
                (
                    number
                    for number in range(1, len(channel_map) + 1)
                    if dio.channel_name("dio_output", number) == CAMERA
                ),
                None,
            )
            with self._output_condition:
                self._output_states = {
                    logical: 0 for logical in self._module_outputs
                }
                for channel in channel_map:
                    if channel.name in OUTPUT_CHANNELS and hasattr(channel, "initial"):
                        self._output_states[OUTPUT_CHANNELS[channel.name]] = int(
                            channel.initial
                        )
                self._output_sequences = defaultdict(int)
                self._output_events = {}
                self._output_event_history = defaultdict(lambda: deque(maxlen=64))
        return {str(key): _json_safe(value) for key, value in status.items()}

    @property
    def channel_map(self) -> ModuleConfig | None:
        return self._map

    def module_output_number(self, channel: int) -> int:
        """The module's number for evil_rig output ``channel``."""
        try:
            return self._module_outputs[int(channel)]
        except KeyError:
            raise RuntimeError(f"Output channel {channel} is not in the StreamDIO map") from None

    @property
    def module_info(self) -> dict[str, Any]:
        """The module's identity: what a rig certificate binds, stable across sessions.

        The firmware build (``build_id``) identifies the firmware exactly; the version
        stays 1 until StreamDIO's first release. What changes from session to session
        (the map, its hash, the boot ID) is in :meth:`map_info`.
        """
        if self._dio is None:
            return {}
        info = {
            key: _json_safe(self._dio.module_info.get(key))
            for key in ("name", "firmware_version", "build_id", "board_id")
        }
        info["board"] = getattr(self._dio.board, "name", None)
        info["speaker_pin"] = SPEAKER_GATE_PIN
        info["camera_pin"] = CAMERA_TRIGGER_PIN
        return info

    def map_info(self) -> dict[str, Any]:
        """The running map, as applied this session: channels, hash, boot ID."""
        if self._dio is None or self._map is None:
            return {}
        return {
            "channel_map": self._map.to_dict(),
            **{
                key: _json_safe(self._dio.module_info.get(key))
                for key in ("config_source", "config_hash", "boot_id", "n_bpod_events")
            },
        }

    def start_camera_session(
        self,
        frequency_hz: int = DEFAULT_CAMERA_FREQUENCY_HZ,
        exposure_us: int = DEFAULT_CAMERA_EXPOSURE_US,
    ) -> None:
        dio = self._require_dio()
        with self._config_lock:
            camera = dio.pulses[CAMERA]
            camera.start(hz=frequency_hz, width_us=exposure_us)
            status = camera.status()
        if (
            not status.get("running")
            or round(float(status.get("frequency_hz", -1)), 3) != round(float(frequency_hz), 3)
            or int(status.get("width_us", -1)) != int(exposure_us)
        ):
            raise RuntimeError(f"Unexpected camera trigger status after start: {status}")

    def stop_camera(self) -> None:
        if self._dio is None:
            return
        with self._config_lock:
            camera = self._dio.pulses[CAMERA]
            camera.stop()
            # A stop takes effect after the frame in progress.
            deadline = time.perf_counter() + CAMERA_STOP_TIMEOUT_S
            while (status := camera.status()).get("running"):
                if time.perf_counter() > deadline:
                    raise RuntimeError(f"Camera triggers remained enabled after stop: {status}")
                time.sleep(0.005)

    def start_streaming(self, callback: Callable[..., None] | None = None) -> None:
        dio = self._require_dio()
        if self._map is None:
            raise RuntimeError("Apply the StreamDIO channel map before streaming")
        dio.clear_event_history()
        self._camera_events = []
        self._callback = callback
        if not dio.start_streaming(callback=self._dispatch_event):
            raise RuntimeError("StreamDIO start_streaming failed")

    def _translate(
        self, event_type: str, channel: int | None, value: int | None
    ) -> tuple[str, int | None, int | None] | None:
        """A driver event in evil_rig's terms; None for events evil_rig drops."""
        if event_type == "dio_input" and channel is not None:
            return event_type, self._inputs.get(int(channel), int(channel)), value
        if event_type == "dio_output" and channel is not None:
            if int(channel) == self._camera_output:
                # One event per frame, at the trigger's rising edge.
                return ("camera", None, None) if value else None
            return event_type, self._outputs.get(int(channel), int(channel)), value
        if event_type == "analog_in" and channel is not None:
            return "analog_input", self._analog.get(int(channel), int(channel)), value
        if event_type == "latency_heartbeat":
            return TIMING_HEARTBEAT_EVENT_TYPE, None, None
        return event_type, channel, value

    def _dispatch_event(
        self,
        event_type: str,
        channel: int | None,
        value: int | None,
        timestamp_us: int,
        relative_time: float,
    ) -> None:
        host_mono_s = time.perf_counter()
        host_wall_s = time.time()
        translated = self._translate(event_type, channel, value)
        if translated is None:
            return
        event_type, channel, value = translated
        if event_type == "camera":
            self._camera_events.append((int(timestamp_us), float(relative_time)))
        if event_type == "dio_output" and channel is not None and value is not None:
            channel_i = int(channel)
            value_i = int(value)
            with self._output_condition:
                self._output_states[channel_i] = value_i
                self._output_sequences[channel_i] += 1
                sequence = self._output_sequences[channel_i]
                self._output_events[channel_i] = (
                    value_i,
                    int(timestamp_us),
                    float(relative_time),
                    host_mono_s,
                )
                self._output_event_history[channel_i].append(
                    (
                        sequence,
                        value_i,
                        int(timestamp_us),
                        float(relative_time),
                        host_mono_s,
                    )
                )
                self._output_condition.notify_all()
        if self._callback is not None:
            self._callback(
                event_type,
                channel,
                value,
                timestamp_us,
                relative_time,
                host_perf_counter_s=host_mono_s,
                host_wall_time_s=host_wall_s,
            )

    def set_output_channel(
        self,
        channel: int,
        state: bool | int,
        *,
        verify_event: bool = True,
        timeout_s: float = 0.15,
    ) -> dict[str, Any]:
        """Set an output through verified USB1 and await its physical pin-change event."""
        dio = self._require_dio()
        channel_i = int(channel)
        state_i = 1 if bool(state) else 0
        module_number = self.module_output_number(channel_i)
        with self._output_condition:
            prior_state = self._output_states.get(channel_i)
            prior_sequence = self._output_sequences[channel_i]

        with self._config_lock:
            command_start = time.perf_counter()
            acknowledged = bool(dio.manual_override_output(module_number, state_i))
            command_end = time.perf_counter()
        if not acknowledged:
            raise RuntimeError(
                f"StreamDIO rejected output channel {channel_i} state {state_i}"
            )

        result: dict[str, Any] = {
            "channel": channel_i,
            "state": state_i,
            "acknowledged": True,
            "command_start_mono_s": command_start,
            "command_end_mono_s": command_end,
            "changed": prior_state != state_i,
            "event_verified": False,
        }
        if not verify_event or not dio.streaming_active:
            with self._output_condition:
                self._output_states[channel_i] = state_i
            return result
        if prior_state == state_i:
            result["event_verified"] = True
            result["verification_reason"] = "already_in_requested_state"
            return result

        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        with self._output_condition:
            while True:
                if (
                    self._output_sequences[channel_i] > prior_sequence
                    and self._output_states.get(channel_i) == state_i
                ):
                    event = self._output_events[channel_i]
                    result.update(
                        {
                            "event_verified": True,
                            "module_timestamp_us": event[1],
                            "relative_time_s": event[2],
                            "event_host_perf_counter_s": event[3],
                        }
                    )
                    return result
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                self._output_condition.wait(timeout=remaining)
        raise RuntimeError(
            f"StreamDIO acknowledged output channel {channel_i} state {state_i}, "
            "but no matching physical output event arrived"
        )

    def output_sequence_snapshot(self, channels: tuple[int, ...]) -> dict[int, int]:
        """Return event cursors used to observe later Bpod-serial output changes."""

        with self._output_condition:
            return {
                int(channel): int(self._output_sequences[int(channel)])
                for channel in channels
            }

    def wait_for_output_event(
        self,
        channel: int,
        state: bool | int,
        *,
        after_sequence: int,
        timeout_s: float = 0.15,
    ) -> dict[str, Any]:
        """Observe a Bpod-issued physical output event without issuing a command."""

        channel_i = int(channel)
        state_i = 1 if bool(state) else 0
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        with self._output_condition:
            while True:
                history = self._output_event_history.get(channel_i, ())
                for sequence, value, timestamp_us, relative_s, host_mono_s in history:
                    if int(sequence) > int(after_sequence) and int(value) == state_i:
                        return {
                            "channel": channel_i,
                            "state": state_i,
                            "event_verified": True,
                            "actuator": "bpod_fsm_serial",
                            "sequence": int(sequence),
                            "module_timestamp_us": int(timestamp_us),
                            "relative_time_s": float(relative_s),
                            "event_host_perf_counter_s": float(host_mono_s),
                        }
                # A busy or faulting channel can advance beyond the bounded
                # history ring.  The latest physical event remains a valid
                # observation of the requested post-cursor state.
                latest_sequence = int(self._output_sequences[channel_i])
                latest = self._output_events.get(channel_i)
                if (
                    latest is not None
                    and latest_sequence > int(after_sequence)
                    and int(latest[0]) == state_i
                ):
                    return {
                        "channel": channel_i,
                        "state": state_i,
                        "event_verified": True,
                        "actuator": "bpod_fsm_serial",
                        "sequence": latest_sequence,
                        "module_timestamp_us": int(latest[1]),
                        "relative_time_s": float(latest[2]),
                        "event_host_perf_counter_s": float(latest[3]),
                        "verification_reason": "latest_physical_event",
                    }
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                self._output_condition.wait(timeout=remaining)
        raise RuntimeError(
            "No matching Bpod-serial physical output event arrived for "
            f"channel {channel_i} state {state_i} after sequence {after_sequence}"
        )

    def get_cap_touch_status(self) -> dict[str, Any]:
        """The module's cap-touch lick detector, in the fixed-layout firmware's terms."""
        dio = self._require_dio()
        if self._map is None or LICK not in {channel.name for channel in self._map}:
            return {"enabled": False, "channel": CAP_TOUCH_INPUT_CHANNEL, "pin": CAP_TOUCH_INPUT_PIN}
        with self._config_lock:
            status = dio.thresholds[LICK].status()
        active = status.get("active")
        return {
            "enabled": bool(status.get("enabled")),
            "channel": CAP_TOUCH_INPUT_CHANNEL,
            "pin": CAP_TOUCH_INPUT_PIN,
            "state": int(status.get("level", 0)),
            "polarity": _POLARITY_CODES.get(active, 0),
            "last_value": int(status.get("sample", 0)),
            "baseline": int(status.get("baseline", 0)),
            "rising_high_threshold": status.get("rising_on"),
            "rising_low_threshold": status.get("rising_off"),
            "falling_high_threshold": status.get("falling_on"),
            "falling_low_threshold": status.get("falling_off"),
            "blanking": bool(status.get("blanking")),
            "samples": int(status.get("samples", 0)),
        }

    def get_module_status(self) -> dict[str, Any]:
        """Return a one-shot StreamDIO status snapshot for cold-path reports."""
        dio = self._require_dio()
        status: dict[str, Any] = {
            "config_port": self.config_port,
            "stream_port": self.stream_port,
            "streaming_active": bool(dio.streaming_active),
            "module_info": self.module_info,
            "buffer_stats": self._buffer_stats(),
        }
        with self._config_lock:
            for name, read in (
                ("config_status", dio.config_status),
                ("analog_input_status", lambda: dio.analog["force"].status()),
                ("speaker_status", lambda: dio.outputs[SPEAKER].status()),
            ):
                try:
                    status[name] = {
                        str(key): _json_safe(value) for key, value in dict(read()).items()
                    }
                except Exception as exc:
                    status[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        if "speaker_status" in status:
            status["speaker_status"]["pin"] = SPEAKER_GATE_PIN
        return status

    def stop_streaming(self) -> None:
        if self._dio is not None and self._dio.streaming_active:
            self._dio.stop_streaming()
            self._last_buffer_stats = self._buffer_stats()

    def _buffer_stats(self) -> dict[str, int]:
        if self._dio is None:
            return {}
        return {
            str(key): int(value)
            for key, value in self._dio.get_buffer_stats().items()
            if isinstance(value, (int, float))
        }

    def get_buffer_stats(self) -> dict[str, int]:
        if self._dio is not None:
            return self._buffer_stats()
        return dict(self._last_buffer_stats)

    def get_camera_events(self) -> list[tuple[int, float]]:
        """Return camera TTLs without scanning/copying the 1 kHz analog history."""
        return list(self._camera_events)

    def close(self) -> None:
        if self._dio is None:
            return
        errors: list[str] = []
        dio = self._dio
        for stage, operation in (
            ("stop_streaming", self.stop_streaming),
            ("stop_camera", self.stop_camera),
            # Also drives every output low, as the module's emergency off.
            ("driver_close", dio.close),
        ):
            try:
                operation()
            except BaseException as exc:
                errors.append(f"{stage}: {type(exc).__name__}: {exc}")
        self._dio = None
        self._callback = None
        if errors:
            raise RuntimeError("StreamDIO close failed: " + "; ".join(errors))

    def _require_dio(self) -> StreamDIO:
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        return self._dio

    @property
    def raw(self) -> StreamDIO | None:
        return self._dio
