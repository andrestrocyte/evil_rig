"""Live cap-touch diagnostic for the StreamDIO digitizer."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from hf_dm.config import SessionConfig
from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_INPUT_CHANNEL,
    CAP_TOUCH_INPUT_PIN,
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    CAP_TOUCH_POWER_PIN,
    FORCE_ANALOG_CHANNEL,
)
from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.outputs import StreamDIOOutputs


@dataclass
class CapTouchDiagnosticState:
    lick_channels: tuple[int, ...]
    force_channel: int = FORCE_ANALOG_CHANNEL
    lock: threading.Lock = field(default_factory=threading.Lock)
    edges: list[dict[str, Any]] = field(default_factory=list)
    statuses: list[dict[str, Any]] = field(default_factory=list)
    force_sample_count: int = 0
    force_min: int | None = None
    force_max: int | None = None
    output_events: list[dict[str, Any]] = field(default_factory=list)

    def handle_event(
        self,
        event_type: str,
        channel: int | None,
        value: int | None,
        timestamp_us: int,
        relative_time_s: float,
    ) -> None:
        channel_i = None if channel is None else int(channel)
        value_i = None if value is None else int(value)
        with self.lock:
            if event_type == "dio_input" and channel_i in self.lick_channels:
                self.edges.append(
                    {
                        "channel": channel_i,
                        "value": value_i,
                        "timestamp_us": int(timestamp_us),
                        "relative_time_s": float(relative_time_s),
                        "host_mono_s": time.perf_counter(),
                    }
                )
            elif event_type == "analog_input" and channel_i == self.force_channel:
                self.force_sample_count += 1
                if value_i is not None:
                    self.force_min = value_i if self.force_min is None else min(self.force_min, value_i)
                    self.force_max = value_i if self.force_max is None else max(self.force_max, value_i)
            elif event_type == "dio_output":
                self.output_events.append(
                    {
                        "channel": channel_i,
                        "value": value_i,
                        "timestamp_us": int(timestamp_us),
                        "relative_time_s": float(relative_time_s),
                    }
                )

    def add_status(self, status: dict[str, Any]) -> None:
        normalized = {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in dict(status).items()
        }
        with self.lock:
            self.statuses.append(normalized)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            high_edges = [edge for edge in self.edges if int(edge.get("value") or 0) == 1]
            low_edges = [edge for edge in self.edges if int(edge.get("value") or 0) == 0]
            values = [
                int(status["last_value"])
                for status in self.statuses
                if status.get("last_value") is not None
            ]
            baselines = [
                int(status["baseline"])
                for status in self.statuses
                if status.get("baseline") is not None
            ]
            final_status = dict(self.statuses[-1]) if self.statuses else {}
            return {
                "lick_channels": list(self.lick_channels),
                "cap_touch_input_channel": CAP_TOUCH_INPUT_CHANNEL,
                "cap_touch_input_pin": CAP_TOUCH_INPUT_PIN,
                "cap_touch_power_output_channel": CAP_TOUCH_POWER_OUTPUT_CHANNEL,
                "cap_touch_power_pin": CAP_TOUCH_POWER_PIN,
                "edge_count": len(self.edges),
                "high_edge_count": len(high_edges),
                "low_edge_count": len(low_edges),
                "recent_edges": list(self.edges[-8:]),
                "status_count": len(self.statuses),
                "adc_min": min(values) if values else None,
                "adc_max": max(values) if values else None,
                "adc_last": values[-1] if values else None,
                "baseline_last": baselines[-1] if baselines else None,
                "final_status": final_status,
                "force_sample_count": self.force_sample_count,
                "force_min": self.force_min,
                "force_max": self.force_max,
                "output_events": list(self.output_events[-8:]),
            }


def run_cap_touch_diagnostic(
    config: SessionConfig,
    *,
    duration_s: float = 15.0,
    poll_hz: float = 10.0,
    min_high_threshold: int | None = None,
    delta_high: int | None = None,
    delta_low: int | None = None,
) -> dict[str, Any]:
    """Run a short live StreamDIO-only cap-touch check and return a report."""
    ports = config.ports
    duration_s = max(0.1, float(duration_s))
    poll_hz = max(1.0, min(50.0, float(poll_hz)))
    state = CapTouchDiagnosticState(
        lick_channels=tuple(int(channel) for channel in config.licks.channels)
    )
    client = StreamDIOClient(
        config_port=ports.dio_config_port,
        stream_port=ports.dio_stream_port,
        baudrate=ports.baudrate,
    )
    outputs: StreamDIOOutputs | None = None
    started_mono_s = time.perf_counter()
    power_on: dict[str, Any] | None = None
    power_off: dict[str, Any] | None = None
    module_info: dict[str, Any] = {}
    listen_started_mono_s: float | None = None
    listen_finished_mono_s: float | None = None
    threshold_status: dict[str, Any] | None = None
    requested_thresholds = {
        "min_high_threshold": (
            config.licks.cap_touch_min_high_threshold
            if min_high_threshold is None
            else min_high_threshold
        ),
        "delta_high": (
            config.licks.cap_touch_delta_high if delta_high is None else delta_high
        ),
        "delta_low": config.licks.cap_touch_delta_low if delta_low is None else delta_low,
    }

    print(
        "Cap-touch diagnostic: reading input ch14/pin25, powering sensor via "
        "output ch13/pin15."
    )
    print(
        "Touch the lick detector during the window; high edges should increment "
        "immediately."
    )
    try:
        client.open()
        module_info = {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in dict(client.raw.module_info).items()
        }
        print(f"StreamDIO firmware: {module_info.get('firmware_version')}")
        client.configure_rotary_and_force(lick_channels=state.lick_channels)
        threshold_requested = any(value is not None for value in requested_thresholds.values())
        firmware_version = int(module_info.get("firmware_version", 0))
        if threshold_requested or firmware_version >= 4:
            threshold_status = client.configure_cap_touch_digitizer(
                min_high_threshold=requested_thresholds["min_high_threshold"],
                delta_high=requested_thresholds["delta_high"],
                delta_low=requested_thresholds["delta_low"],
            )
            print(
                "Cap-touch threshold config: "
                f"{json.dumps(threshold_status, sort_keys=True)}"
            )
        client.start_streaming(callback=state.handle_event)
        outputs = StreamDIOOutputs(
            client,
            cap_touch_power_channel=config.licks.power_output_channel,
            speaker_sample_interval_us=config.speaker.sample_interval_us,
        )
        power_on = outputs.cap_touch_power(True)
        print(f"Power ON verified: {json.dumps(power_on, sort_keys=True)}")

        listen_started_mono_s = time.perf_counter()
        deadline = listen_started_mono_s + duration_s
        poll_interval_s = 1.0 / poll_hz
        next_print_s = 0.0
        while time.perf_counter() < deadline:
            status = client.get_cap_touch_status()
            state.add_status(status)
            now = time.perf_counter()
            if now >= next_print_s:
                snap = state.snapshot()
                print(
                    "touch status: "
                    f"adc={snap['adc_last']} baseline={snap['baseline_last']} "
                    f"active={int(status.get('state', 0))} "
                    f"hi_edges={snap['high_edge_count']} "
                    f"lo_edges={snap['low_edge_count']} "
                    f"threshold={status.get('rising_high_threshold')}"
                )
                next_print_s = now + 0.5
            time.sleep(poll_interval_s)
        listen_finished_mono_s = time.perf_counter()
    finally:
        try:
            if outputs is not None:
                power_off = outputs.cap_touch_power(False)
                print(f"Power OFF verified: {json.dumps(power_off, sort_keys=True)}")
        finally:
            client.close()

    report = state.snapshot()
    total_duration_s = time.perf_counter() - started_mono_s
    listen_duration_s = (
        None
        if listen_started_mono_s is None or listen_finished_mono_s is None
        else listen_finished_mono_s - listen_started_mono_s
    )
    report.update(
        {
            "duration_s": total_duration_s,
            "listen_duration_s": listen_duration_s,
            "module_info": module_info,
            "power_on": power_on,
            "power_off": power_off,
            "requested_thresholds": requested_thresholds,
            "threshold_status": threshold_status,
            "detected_touch": report["high_edge_count"] > 0,
        }
    )
    print("Cap-touch diagnostic report:")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["detected_touch"]:
        print(
            "No high lick edges were detected. If the detector was touched during "
            "this run, check the cap-touch sensor wiring/power or lower the "
            "firmware threshold."
        )
    return report
