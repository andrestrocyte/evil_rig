"""StreamDIO wrapper: imports MM_PyBpod2 module (same as bringup notebooks)."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_DEFAULT_DELTA_HIGH,
    CAP_TOUCH_DEFAULT_DELTA_LOW,
    CAP_TOUCH_DEFAULT_MIN_HIGH_THRESHOLD,
    CAP_TOUCH_INPUT_CHANNEL,
    CAP_TOUCH_INPUT_PIN,
    DEFAULT_CAMERA_EXPOSURE_US,
    DEFAULT_CAMERA_FREQUENCY_HZ,
    DIRECTION_INPUT_CHANNEL,
    FORCE_ANALOG_CHANNEL,
    FORCE_ANALOG_PIN,
    FORCE_ANALOG_SAMPLE_INTERVAL_US,
    INPUT_REFRACTORY_PERIOD_US,
    LICK_INPUT_CHANNEL,
    PULSE_INPUT_CHANNEL,
    STREAMDIO_ANALOG_INPUT_EVENT_BASE,
    STREAMDIO_BPOD_STATE_EVENT_BASE,
    STREAMDIO_CAP_TOUCH_CONFIG_FIRMWARE_VERSION,
    STREAMDIO_CAP_TOUCH_DIGITIZER_FIRMWARE_VERSION,
    STREAMDIO_DIO_INPUT_EVENT_BASE,
    STREAMDIO_DIO_OUTPUT_EVENT_BASE,
    STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE,
    STREAMDIO_TIMING_HEARTBEAT_FIRMWARE_VERSION,
)
from hf_dm.hardware.discovery import resolve_streamdio_ports
from hf_dm.timing_heartbeat import TIMING_HEARTBEAT_EVENT_TYPE


class _TimingHeartbeatLogFilter(logging.Filter):
    """Suppress only MM_PyBpod2's per-heartbeat WARNING log line."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return f"Diagnostic event {STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE} " not in message


def _uint12(value: int) -> int:
    return max(0, min(4095, int(value)))


def _json_safe(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


class StreamDIOClient:
    """Thin facade over pybpod_streamDIO_module.StreamDIOProtocol."""

    def __init__(self, config_port: str = "AUTO", stream_port: str = "AUTO", *, baudrate: int):
        # The bringup driver logs every 1 kHz analog sample at INFO and pybpod's
        # ArCOM logger can emit byte-level DEBUG messages. Both streams can stall
        # the task/display loop even though experiment data are already retained
        # by StreamDIO and this session recorder.
        driver_logger = logging.getLogger(
            "pybpod_streamDIO_module.stream_dio_protocol"
        )
        driver_logger.setLevel(logging.WARNING)
        if not any(
            isinstance(item, _TimingHeartbeatLogFilter)
            for item in driver_logger.filters
        ):
            driver_logger.addFilter(_TimingHeartbeatLogFilter())
        logging.getLogger("pybpodapi.com.arcom").setLevel(logging.WARNING)
        try:
            from pybpod_streamDIO_module.stream_dio_protocol import StreamDIOProtocol
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pybpod_streamDIO_module is not installed. Install MM_PyBpod2 "
                "into this environment, or vendor the StreamDIO driver explicitly."
            ) from exc
        StreamDIOProtocol.DIO_INPUT_EVENT_BASE = STREAMDIO_DIO_INPUT_EVENT_BASE
        StreamDIOProtocol.DIO_OUTPUT_EVENT_BASE = STREAMDIO_DIO_OUTPUT_EVENT_BASE
        StreamDIOProtocol.ANALOG_INPUT_EVENT_BASE = STREAMDIO_ANALOG_INPUT_EVENT_BASE
        StreamDIOProtocol.BPOD_STATE_EVENT_BASE = STREAMDIO_BPOD_STATE_EVENT_BASE

        self._protocol_cls = StreamDIOProtocol
        self.config_port = config_port
        self.stream_port = stream_port
        self.baudrate = baudrate
        self._dio: Any = None
        self._callback: Callable[..., None] | None = None
        self._config_lock = threading.Lock()
        self._output_condition = threading.Condition()
        self._output_states: dict[int, int] = {}
        self._output_sequences: dict[int, int] = defaultdict(int)
        self._output_events: dict[int, tuple[int, int, float, float]] = {}
        self._output_event_history: dict[
            int,
            deque[tuple[int, int, int, float, float]],
        ] = defaultdict(lambda: deque(maxlen=64))
        self._last_buffer_stats: dict[str, int] = {}

    def open(
        self,
        excluded_bpod_port: str | None = None,
        excluded_bpod_ports: set[str] | None = None,
    ) -> None:
        cfg, stream = resolve_streamdio_ports(
            self.config_port,
            self.stream_port,
            excluded_port=excluded_bpod_port,
            excluded_ports=excluded_bpod_ports,
            baud=self.baudrate,
        )
        self.config_port = cfg
        self.stream_port = stream
        self._dio = self._protocol_cls(config_port=cfg, stream_port=stream, connect=False)
        try:
            self._dio.open()
            # Firmware initialization drives all module outputs low.
            n_outputs = int(self._dio.module_info["n_output_channels"])
            with self._output_condition:
                self._output_states = {channel: 0 for channel in range(1, n_outputs + 1)}
                self._output_sequences = defaultdict(int)
                self._output_events = {}
                self._output_event_history = defaultdict(lambda: deque(maxlen=64))
        except Exception as exc:
            self._dio = None
            raise RuntimeError(
                f"Failed to open StreamDIO config={cfg}, stream={stream}. "
                "If Windows reports 'Access is denied', close old hf-dm-run/python "
                "processes, PyBpod notebooks, serial monitors, or unplug/replug the "
                "StreamDIO USB device."
            ) from exc

    def configure_rotary_and_force(
        self,
        lick_channels: tuple[int, ...] = (LICK_INPUT_CHANNEL,),
        extra_input_channels: tuple[int, ...] = (),
    ) -> None:
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        with self._config_lock:
            firmware_version = int(self._dio.module_info.get("firmware_version", 0))
            if firmware_version < STREAMDIO_TIMING_HEARTBEAT_FIRMWARE_VERSION:
                raise RuntimeError(
                    "StreamDIO firmware does not emit the required 100 ms timing "
                    f"heartbeat. Found firmware version {firmware_version}; expected "
                    f"{STREAMDIO_TIMING_HEARTBEAT_FIRMWARE_VERSION} or newer. Flash "
                    "firmware/Module_DIO_streeaming_100us_cycle before running a task."
                )
            if LICK_INPUT_CHANNEL in {int(channel) for channel in lick_channels}:
                if firmware_version < STREAMDIO_CAP_TOUCH_DIGITIZER_FIRMWARE_VERSION:
                    raise RuntimeError(
                        "StreamDIO firmware does not include the cap-touch digitizer. "
                        f"Found firmware version {firmware_version}; expected "
                        f"{STREAMDIO_CAP_TOUCH_DIGITIZER_FIRMWARE_VERSION} or newer. "
                        "Flash firmware/Module_DIO_streeaming_100us_cycle before running "
                        "with cap-touch licks on channel 14."
                    )
            enable_vector = [False] * int(self._dio.module_info["n_input_channels"])
            enable_vector[PULSE_INPUT_CHANNEL - 1] = True
            enable_vector[DIRECTION_INPUT_CHANNEL - 1] = True
            for channel in tuple(lick_channels) + tuple(extra_input_channels):
                if 1 <= int(channel) <= len(enable_vector):
                    enable_vector[int(channel) - 1] = True
            ok = self._dio.configure_module(
                refractory_period=INPUT_REFRACTORY_PERIOD_US,
                input_channels_enabled=enable_vector,
                analog_input={
                    "channel": FORCE_ANALOG_CHANNEL,
                    "enabled": True,
                    "sample_interval_us": FORCE_ANALOG_SAMPLE_INTERVAL_US,
                },
            )
            if not ok:
                raise RuntimeError("StreamDIO configure_module failed")
            status = self._dio.get_analog_input_status()
        if not status.get("enabled"):
            raise RuntimeError(f"Analog force not enabled: {status}")
        if int(status.get("channel", -1)) != FORCE_ANALOG_CHANNEL:
            raise RuntimeError(f"Unexpected force channel: {status}")
        if int(status.get("pin", -1)) != FORCE_ANALOG_PIN:
            raise RuntimeError(f"Unexpected force pin: {status}")

    def configure_cap_touch_analog_diagnostic(
        self,
        *,
        sample_interval_us: int = FORCE_ANALOG_SAMPLE_INTERVAL_US,
    ) -> dict[str, Any]:
        """Temporarily stream cap-touch pin 25 as the only analog channel."""
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        with self._config_lock:
            enable_vector = [False] * int(self._dio.module_info["n_input_channels"])
            ok = self._dio.configure_module(
                refractory_period=INPUT_REFRACTORY_PERIOD_US,
                input_channels_enabled=enable_vector,
                analog_input={
                    "channel": CAP_TOUCH_INPUT_CHANNEL,
                    "enabled": True,
                    "sample_interval_us": int(sample_interval_us),
                },
            )
            if not ok:
                raise RuntimeError("StreamDIO cap-touch analog diagnostic config failed")
            status = self._dio.get_analog_input_status()
        if not status.get("enabled"):
            raise RuntimeError(f"Analog cap-touch not enabled: {status}")
        if int(status.get("channel", -1)) != CAP_TOUCH_INPUT_CHANNEL:
            raise RuntimeError(f"Unexpected cap-touch analog channel: {status}")
        if int(status.get("pin", -1)) != CAP_TOUCH_INPUT_PIN:
            raise RuntimeError(f"Unexpected cap-touch analog pin: {status}")
        return {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in dict(status).items()
        }

    def start_camera_session(
        self,
        frequency_hz: int = DEFAULT_CAMERA_FREQUENCY_HZ,
        exposure_us: int = DEFAULT_CAMERA_EXPOSURE_US,
    ) -> None:
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        with self._config_lock:
            if not self._dio.start_camera_triggers(frequency=frequency_hz, exposure_us=exposure_us):
                raise RuntimeError("Failed to start camera triggers on StreamDIO")

    def stop_camera(self) -> None:
        if self._dio is not None:
            with self._config_lock:
                self._dio.stop_camera_triggers()

    def start_streaming(self, callback: Callable[..., None] | None = None) -> None:
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        self._dio.clear_event_history()
        self._callback = callback

        # The bringup driver's start_streaming() queries the current timestamp on
        # the stream USB endpoint after streaming has begun. At 1 kHz that read
        # can consume event bytes instead of the timestamp reply and shift packet
        # alignment. Read it while the endpoint is idle, then let startup reuse it.
        timestamp_getter = getattr(self._dio, "_streamdio_get_current_timestamp", None)
        pre_stream_timestamp = timestamp_getter() if callable(timestamp_getter) else 0
        if callable(timestamp_getter):
            self._dio._streamdio_get_current_timestamp = lambda: pre_stream_timestamp
        try:
            started = self._dio.start_streaming(callback=self._dispatch_event)
        finally:
            if callable(timestamp_getter):
                self._dio._streamdio_get_current_timestamp = timestamp_getter
        if not started:
            raise RuntimeError("StreamDIO start_streaming failed")

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
        if (
            event_type == "diagnostic"
            and channel is not None
            and int(channel) == STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE
        ):
            event_type = TIMING_HEARTBEAT_EVENT_TYPE
            channel = None
            value = None
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
                history = getattr(self, "_output_event_history", None)
                if history is None:
                    history = defaultdict(lambda: deque(maxlen=64))
                    self._output_event_history = history
                history[channel_i].append(
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
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        channel_i = int(channel)
        state_i = 1 if bool(state) else 0
        with self._output_condition:
            prior_state = self._output_states.get(channel_i)
            prior_sequence = self._output_sequences[channel_i]

        with self._config_lock:
            command_start = time.perf_counter()
            acknowledged = bool(self._dio.manual_override_output(channel_i, state_i))
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
        if not verify_event or not getattr(self._dio, "streaming_active", False):
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
                history = getattr(self, "_output_event_history", {}).get(channel_i, ())
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

    def set_speaker_noise(
        self,
        enabled: bool,
        *,
        sample_interval_us: int,
    ) -> dict[str, Any]:
        """Control firmware-generated white noise and verify module status."""
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        with self._config_lock:
            command_start = time.perf_counter()
            acknowledged = bool(
                self._dio.configure_speaker_noise(
                    bool(enabled),
                    sample_interval_us=int(sample_interval_us),
                )
            )
            command_end = time.perf_counter()
            if not acknowledged:
                raise RuntimeError(f"StreamDIO rejected speaker noise enabled={bool(enabled)}")
            status = {
                str(key): value.item() if hasattr(value, "item") else value
                for key, value in dict(self._dio.get_speaker_status()).items()
            }
        if bool(status.get("enabled")) != bool(enabled):
            raise RuntimeError(f"Unexpected StreamDIO speaker status: {status}")
        return {
            "enabled": bool(enabled),
            "acknowledged": True,
            "command_start_mono_s": command_start,
            "command_end_mono_s": command_end,
            "status": status,
        }

    def get_cap_touch_status(self) -> dict[str, Any]:
        """Read the firmware-local cap-touch ADC digitizer status from USB1."""
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        config_arcom = getattr(self._dio, "_config_arcom", None)
        if config_arcom is None:
            raise RuntimeError("StreamDIO driver does not expose the config ArCOM port")
        with self._config_lock:
            config_arcom.write(ord("l"), "uint8")
            status = {
                "enabled": bool(config_arcom.read(1, "uint8", "builtin")),
                "channel": int(config_arcom.read(1, "uint8", "builtin")),
                "pin": int(config_arcom.read(1, "uint8", "builtin")),
                "state": int(config_arcom.read(1, "uint8", "builtin")),
                "polarity": int(config_arcom.read(1, "uint8", "builtin")),
                "last_value": int(config_arcom.read(1, "uint16", "builtin")),
                "baseline": int(config_arcom.read(1, "uint16", "builtin")),
                "rising_high_threshold": int(config_arcom.read(1, "uint16", "builtin")),
                "rising_low_threshold": int(config_arcom.read(1, "uint16", "builtin")),
                "falling_high_threshold": int(config_arcom.read(1, "uint16", "builtin")),
                "falling_low_threshold": int(config_arcom.read(1, "uint16", "builtin")),
            }
        return status

    def get_module_status(self) -> dict[str, Any]:
        """Return a one-shot StreamDIO status snapshot for cold-path reports."""
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        status: dict[str, Any] = {
            "config_port": self.config_port,
            "stream_port": self.stream_port,
            "streaming_active": bool(getattr(self._dio, "streaming_active", False)),
            "module_info": {
                str(key): _json_safe(value)
                for key, value in dict(getattr(self._dio, "module_info", {}) or {}).items()
            },
        }
        with self._config_lock:
            for name, getter_name in (
                ("analog_input_status", "get_analog_input_status"),
                ("speaker_status", "get_speaker_status"),
            ):
                getter = getattr(self._dio, getter_name, None)
                if not callable(getter):
                    continue
                try:
                    status[name] = {
                        str(key): _json_safe(value)
                        for key, value in dict(getter()).items()
                    }
                except Exception as exc:
                    status[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        return status

    def configure_cap_touch_digitizer(
        self,
        *,
        min_high_threshold: int | None = None,
        delta_high: int | None = None,
        delta_low: int | None = None,
    ) -> dict[str, Any]:
        """Configure firmware-local cap-touch ADC-to-digital thresholds."""
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        firmware_version = int(self._dio.module_info.get("firmware_version", 0))
        if firmware_version < STREAMDIO_CAP_TOUCH_CONFIG_FIRMWARE_VERSION:
            raise RuntimeError(
                "StreamDIO firmware does not support runtime cap-touch threshold "
                f"configuration. Found firmware version {firmware_version}; expected "
                f"{STREAMDIO_CAP_TOUCH_CONFIG_FIRMWARE_VERSION} or newer. Flash "
                "firmware/Module_DIO_streeaming_100us_cycle before using tuned "
                "cap-touch thresholds."
            )
        config_arcom = getattr(self._dio, "_config_arcom", None)
        if config_arcom is None:
            raise RuntimeError("StreamDIO driver does not expose the config ArCOM port")
        min_high = _uint12(
            CAP_TOUCH_DEFAULT_MIN_HIGH_THRESHOLD
            if min_high_threshold is None
            else min_high_threshold
        )
        high_delta = _uint12(
            CAP_TOUCH_DEFAULT_DELTA_HIGH if delta_high is None else delta_high
        )
        low_delta = _uint12(CAP_TOUCH_DEFAULT_DELTA_LOW if delta_low is None else delta_low)
        if low_delta > high_delta:
            low_delta = high_delta
        with self._config_lock:
            config_arcom.write(ord("t"), "uint8")
            config_arcom.write(min_high, "uint16")
            config_arcom.write(high_delta, "uint16")
            config_arcom.write(low_delta, "uint16")
            acknowledged = bool(config_arcom.read(1, "uint8", "builtin"))
        if not acknowledged:
            raise RuntimeError("StreamDIO rejected cap-touch threshold configuration")
        return self.get_cap_touch_status()

    def stop_streaming(self) -> None:
        if self._dio is not None and getattr(self._dio, "streaming_active", False):
            self._dio.stop_streaming()
            stats = getattr(self._dio, "_buffer_stats", None)
            if isinstance(stats, dict):
                self._last_buffer_stats = {
                    str(key): int(value)
                    for key, value in stats.items()
                    if isinstance(value, (int, float))
                }
            time.sleep(0.05)

    def get_buffer_stats(self) -> dict[str, int]:
        if self._dio is not None:
            stats = getattr(self._dio, "_buffer_stats", None)
            if isinstance(stats, dict):
                return {
                    str(key): int(value)
                    for key, value in stats.items()
                    if isinstance(value, (int, float))
                }
        return dict(self._last_buffer_stats)

    def get_camera_events(self) -> list[tuple[int, float]]:
        """Return (timestamp_us, relative_s) for camera TTL events in history."""
        if self._dio is None:
            return []
        events = self._dio.get_event_history(event_types=["camera"])
        return [(int(e[3]), float(e[-1])) for e in events]

    def close(self) -> None:
        if self._dio is None:
            return
        errors: list[str] = []
        dio = self._dio
        for stage, operation in (
            ("stop_streaming", self.stop_streaming),
            ("stop_camera", self.stop_camera),
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

    @property
    def raw(self) -> Any:
        return self._dio
