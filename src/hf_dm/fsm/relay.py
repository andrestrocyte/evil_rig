"""Serial relay setup on Bpod (breakout board bytes, same as bringup notebooks)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from bpod_core.bpod import Bpod

from hf_dm.hardware.bringup_constants import (
    CCW_FORCE_OUTPUT_CHANNEL,
    CW_FORCE_OUTPUT_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
    SERIAL_GATE_OFF_MESSAGE_ID,
    SERIAL_GATE_ON_MESSAGE_ID,
    SPEAKER_GATE_OPCODE,
    STREAMDIO_BPOD_STATE_OPCODE,
    STREAMDIO_STATE_MARKER_IDS_BY_MESSAGE_ID,
    STREAMDIO_STATE_MARKER_MESSAGE_IDS,
    STREAMDIO_STATE_MARKERS,
)


def wait_until_perf_counter(deadline_s: float, spin_window_s: float = 0.025) -> None:
    """Wait to a high-resolution monotonic deadline."""
    while True:
        remaining_s = float(deadline_s) - time.perf_counter()
        if remaining_s <= 0.0:
            return
        if remaining_s > spin_window_s:
            time.sleep(remaining_s - spin_window_s)


@dataclass(frozen=True)
class RelayMessageIds:
    pump_on: int = 1
    pump_off: int = 2
    cw_force_on: int = 3
    cw_force_off: int = 4
    ccw_force_on: int = 5
    ccw_force_off: int = 6
    speaker_gate_on: int = SERIAL_GATE_ON_MESSAGE_ID
    speaker_gate_off: int = SERIAL_GATE_OFF_MESSAGE_ID


def _serial_module_name(bpod: Bpod, serial_channel_index: int) -> str:
    for name, module in bpod.modules.items():
        if module.is_connected and module.index == serial_channel_index:
            return name

    if serial_channel_index != 0:
        raise RuntimeError(
            f"No connected Bpod serial module on module port {serial_channel_index + 1}"
        )

    for name, module in bpod.modules.items():
        if module.is_connected:
            return name
    connected = [n for n, m in bpod.modules.items() if m.is_connected]
    if not connected:
        raise RuntimeError("No connected Bpod serial module for breakout relay")
    return connected[0]


class SerialRelay:
    """Preload outbound serial messages without relaying module bytes to Bpod USB."""

    def __init__(
        self,
        bpod: Bpod,
        serial_channel_index: int = 0,
        *,
        speaker_gate_active_high: bool = True,
    ):
        self.bpod = bpod
        self.module_name = _serial_module_name(bpod, serial_channel_index)
        self.module = bpod.modules[self.module_name]
        # Bpod's module relay forwards incoming module bytes onto the same USB
        # stream used for FSM timestamps/events. Outbound messages work with it
        # disabled, and enabling it lets StreamDIO replies corrupt that stream.
        self.module.set_relay(False)
        self.ids = RelayMessageIds()
        self._serial_channel_index = serial_channel_index
        self.speaker_gate_active_high = bool(speaker_gate_active_high)
        self._load_messages()

    def _load_messages(self) -> None:
        m = self.module
        pairs = [
            (self.ids.pump_on, [PUMP_OUTPUT_CHANNEL, 1]),
            (self.ids.pump_off, [PUMP_OUTPUT_CHANNEL, 0]),
            (self.ids.cw_force_on, [CW_FORCE_OUTPUT_CHANNEL, 1]),
            (self.ids.cw_force_off, [CW_FORCE_OUTPUT_CHANNEL, 0]),
            (self.ids.ccw_force_on, [CCW_FORCE_OUTPUT_CHANNEL, 1]),
            (self.ids.ccw_force_off, [CCW_FORCE_OUTPUT_CHANNEL, 0]),
            (
                self.ids.speaker_gate_on,
                [SPEAKER_GATE_OPCODE, 1 if self.speaker_gate_active_high else 0],
            ),
            (
                self.ids.speaker_gate_off,
                [SPEAKER_GATE_OPCODE, 0 if self.speaker_gate_active_high else 1],
            ),
        ]
        pairs.extend(
            (
                STREAMDIO_STATE_MARKER_MESSAGE_IDS[state_name],
                [STREAMDIO_BPOD_STATE_OPCODE, int(state_id)],
            )
            for state_name, state_id in STREAMDIO_STATE_MARKERS
        )
        for msg_id, payload in pairs:
            m.load_serial_message(msg_id, bytes(payload))

    @property
    def action_prefix(self) -> str:
        return self.module_name

    def act(self, message_id: int) -> dict[str, int]:
        return {self.module_name: message_id}

    def bpod_state_marker_act(self, state_name: str) -> dict[str, int]:
        return self.act(STREAMDIO_STATE_MARKER_MESSAGE_IDS[str(state_name)])

    @staticmethod
    def bpod_state_id_for_message(message_id: int) -> int | None:
        return STREAMDIO_STATE_MARKER_IDS_BY_MESSAGE_ID.get(int(message_id))

    def send_bytes(self, byte_values: list[int], inter_byte_s: float = 0.002) -> None:
        """Send raw bytes to the selected Bpod module port immediately."""
        payload = bytes(byte_values)
        self.module.set_relay(False)
        # Firmware 'T' takes a one-based module port, payload length, and bytes.
        # Output.override uses 'O', which only controls physical outputs and
        # silently does nothing for a serial module channel.
        port = int(self.module.index) + 1
        for value in payload:
            self.bpod.serial0.write(bytes((ord("T"), port, 1, value)))
            time.sleep(inter_byte_s)

    def prepare_bpod_run(self) -> int:
        """Disable incoming relay and clear idle USB bytes before an FSM starts."""
        self.module.set_relay(False)
        serial = self.bpod.serial0
        pending = max(0, int(getattr(serial, "in_waiting", 0) or 0))
        serial.reset_input_buffer()
        return pending

    def send_input_marker(self, port: int, state: int) -> None:
        """Forward one behavior-port edge to StreamDIO without changing the FSM.

        The existing StreamDIO ['I', 'P', port, state] command emits a timestamped
        port_input event. Send the whole packet in one write; do not issue an
        acknowledgement-producing command while the FSM USB reader is active.
        """
        if not 1 <= port <= 5 or state not in (0, 1):
            raise ValueError("Optical marker needs behavior port 1..5 and state 0/1")
        payload = bytes((ord("I"), ord("P"), port, state))
        packet = bytes((ord("T"), int(self.module.index) + 1, len(payload))) + payload
        written = self.bpod.serial0.write(packet)
        if written is not None and written != len(packet):
            raise RuntimeError("Incomplete Bpod optical-marker USB write")

    def set_channel_state(self, channel: int, state: bool) -> None:
        self.send_bytes([int(channel), 1 if state else 0])

    def set_force_direction(self, direction: str | None) -> None:
        if direction == "cw":
            self.set_channel_state(CCW_FORCE_OUTPUT_CHANNEL, False)
            self.set_channel_state(CW_FORCE_OUTPUT_CHANNEL, True)
        elif direction == "ccw":
            self.set_channel_state(CW_FORCE_OUTPUT_CHANNEL, False)
            self.set_channel_state(CCW_FORCE_OUTPUT_CHANNEL, True)
        else:
            self.force_off()

    def force_off(self) -> None:
        self.set_channel_state(CW_FORCE_OUTPUT_CHANNEL, False)
        self.set_channel_state(CCW_FORCE_OUTPUT_CHANNEL, False)

    def pump_off(self) -> None:
        self.set_channel_state(PUMP_OUTPUT_CHANNEL, False)

    def pump_on(self) -> None:
        self.set_channel_state(PUMP_OUTPUT_CHANNEL, True)

    def pulse_pump(self, duration_s: float) -> None:
        self.pump_on()
        wait_until_perf_counter(time.perf_counter() + max(0.0, float(duration_s)))
        self.pump_off()

    def speaker_gate(self, enabled: bool) -> None:
        on_level = 1 if self.speaker_gate_active_high else 0
        self.send_bytes([SPEAKER_GATE_OPCODE, on_level if enabled else 1 - on_level])

    def pulse_speaker_gate(self, duration_s: float) -> None:
        self.speaker_gate(True)
        wait_until_perf_counter(time.perf_counter() + max(0.0, float(duration_s)))
        self.speaker_gate(False)

    def all_off(self) -> None:
        self.pump_off()
        self.force_off()
        self.speaker_gate(False)
