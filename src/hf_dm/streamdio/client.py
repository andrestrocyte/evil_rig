"""StreamDIO wrapper: imports MM_PyBpod2 module (same as bringup notebooks)."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable

from hf_dm.hardware.bringup_constants import (
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
    STREAMDIO_CAP_TOUCH_DIGITIZER_FIRMWARE_VERSION,
    STREAMDIO_DIO_INPUT_EVENT_BASE,
    STREAMDIO_DIO_OUTPUT_EVENT_BASE,
)
from hf_dm.hardware.discovery import resolve_streamdio_ports


class StreamDIOClient:
    """Thin facade over pybpod_streamDIO_module.StreamDIOProtocol."""

    def __init__(self, config_port: str = "AUTO", stream_port: str = "AUTO", *, baudrate: int):
        # The bringup driver logs every 1 kHz analog sample at INFO and pybpod's
        # ArCOM logger can emit byte-level DEBUG messages. Both streams can stall
        # the task/display loop even though experiment data are already retained
        # by StreamDIO and this session recorder.
        logging.getLogger("pybpod_streamDIO_module.stream_dio_protocol").setLevel(logging.WARNING)
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
        except Exception as exc:
            self._dio = None
            raise RuntimeError(
                f"Failed to open StreamDIO config={cfg}, stream={stream}. "
                "If Windows reports 'Access is denied', close old hf-dm-run/python "
                "processes, PyBpod notebooks, serial monitors, or unplug/replug the "
                "StreamDIO USB device."
            ) from exc

    def configure_rotary_and_force(self, lick_channels: tuple[int, ...] = (LICK_INPUT_CHANNEL,)) -> None:
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        with self._config_lock:
            if LICK_INPUT_CHANNEL in {int(channel) for channel in lick_channels}:
                firmware_version = int(self._dio.module_info.get("firmware_version", 0))
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
            for channel in lick_channels:
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
        if event_type == "dio_output" and channel is not None and value is not None:
            channel_i = int(channel)
            value_i = int(value)
            with self._output_condition:
                self._output_states[channel_i] = value_i
                self._output_sequences[channel_i] += 1
                self._output_events[channel_i] = (
                    value_i,
                    int(timestamp_us),
                    float(relative_time),
                    time.perf_counter(),
                )
                self._output_condition.notify_all()
        if self._callback is not None:
            self._callback(event_type, channel, value, timestamp_us, relative_time)

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

    def stop_streaming(self) -> None:
        if self._dio is not None and getattr(self._dio, "streaming_active", False):
            self._dio.stop_streaming()
            time.sleep(0.05)

    def get_camera_events(self) -> list[tuple[int, float]]:
        """Return (timestamp_us, relative_s) for camera TTL events in history."""
        if self._dio is None:
            return []
        events = self._dio.get_event_history(event_types=["camera"])
        return [(int(e[3]), float(e[-1])) for e in events]

    def close(self) -> None:
        if self._dio is None:
            return
        try:
            self.stop_streaming()
            self.stop_camera()
            self._dio.close()
        except Exception:
            pass
        self._dio = None
        self._callback = None

    @property
    def raw(self) -> Any:
        return self._dio
