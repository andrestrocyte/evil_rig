"""Serial relay setup on Bpod (breakout board bytes — same as bringup notebooks)."""

from __future__ import annotations

from dataclasses import dataclass

from bpod_core.bpod import Bpod

from hf_dm.hardware.bringup_constants import (
    CCW_FORCE_OUTPUT_CHANNEL,
    CW_FORCE_OUTPUT_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
    SERIAL_GATE_OFF_MESSAGE_ID,
    SERIAL_GATE_ON_MESSAGE_ID,
    SPEAKER_GATE_OPCODE,
)


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


def _serial_module_name(bpod: Bpod) -> str:
    for name, module in bpod.modules.items():
        if module.is_connected and name.startswith("Serial"):
            return name
    connected = [n for n, m in bpod.modules.items() if m.is_connected]
    if not connected:
        raise RuntimeError("No connected Bpod serial module for breakout relay")
    return connected[0]


class SerialRelay:
    """Preload serial messages and expose FSM action names for bpod-core."""

    def __init__(self, bpod: Bpod, serial_channel_index: int = 0):
        self.bpod = bpod
        self.module_name = _serial_module_name(bpod)
        self.module = bpod.modules[self.module_name]
        self.module.set_relay(True)
        self.ids = RelayMessageIds()
        self._serial_channel_index = serial_channel_index
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
            (self.ids.speaker_gate_on, [SPEAKER_GATE_OPCODE, 1]),
            (self.ids.speaker_gate_off, [SPEAKER_GATE_OPCODE, 0]),
        ]
        for msg_id, payload in pairs:
            m.load_serial_message(msg_id, bytes(payload))

    @property
    def action_prefix(self) -> str:
        return self.module_name

    def act(self, message_id: int) -> dict[str, int]:
        return {self.module_name: message_id}
