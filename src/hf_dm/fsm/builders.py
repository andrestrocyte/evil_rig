"""bpod-core FSM templates for HF decision-making (timed hardware only)."""

from __future__ import annotations

from typing import Literal

from bpod_core.fsm import StateMachine

from hf_dm.fsm.relay import SerialRelay
from hf_dm.fsm import softcodes as sc

Direction = Literal["cw", "ccw"]


def build_choice_epoch_fsm(
    relay: SerialRelay,
    force_direction: Direction,
    choice_timeout_s: float,
) -> StateMachine:
    """
    Apply effort cue, wait up to choice_timeout_s for host softcode choice or timeout.

    Host supervisor reads StreamDIO wheel and sends sc.CHOICE_CW / sc.CHOICE_CCW.
  On Tup (timeout), transitions to white-noise punishment state.
    """
    fsm = StateMachine(name=f"ChoiceEpoch_{force_direction}")
    force_on = relay.ids.cw_force_on if force_direction == "cw" else relay.ids.ccw_force_on
    force_off_other = relay.ids.ccw_force_off if force_direction == "cw" else relay.ids.cw_force_off

    fsm.add_state(
        name="ApplyForce",
        timer=0.05,
        transitions={"Tup": "ChoiceWindow"},
        actions={
            **relay.act(force_off_other),
            **relay.act(force_on),
        },
    )
    fsm.add_state(
        name="ChoiceWindow",
        timer=choice_timeout_s,
        transitions={
            "SoftCode1": "ChoiceRegistered",
            "SoftCode2": "ChoiceRegistered",
            "Tup": "TimeoutPunish",
        },
    )
    fsm.add_state(
        name="ChoiceRegistered",
        timer=0.01,
        transitions={"Tup": ">exit"},
        actions={
            **relay.act(relay.ids.cw_force_off),
            **relay.act(relay.ids.ccw_force_off),
        },
    )
    fsm.add_state(
        name="TimeoutPunish",
        timer=0.01,
        transitions={"Tup": "PlayWhiteNoise"},
        actions={
            **relay.act(relay.ids.cw_force_off),
            **relay.act(relay.ids.ccw_force_off),
        },
    )
    fsm.add_state(
        name="PlayWhiteNoise",
        timer=2.0,
        transitions={"Tup": "SpeakerGateOff"},
        actions=relay.act(relay.ids.speaker_gate_on),
        comment="Gate white noise on pin 27 (speaker bringup)",
    )
    fsm.add_state(
        name="SpeakerGateOff",
        timer=0.05,
        transitions={"Tup": ">exit"},
        actions=relay.act(relay.ids.speaker_gate_off),
    )
    return fsm


def build_reward_fsm(relay: SerialRelay, pump_duration_s: float) -> StateMachine:
    fsm = StateMachine(name=f"Reward_{pump_duration_s:.3f}s")
    fsm.add_state(
        name="PumpOn",
        timer=pump_duration_s,
        transitions={"Tup": "PumpOff"},
        actions=relay.act(relay.ids.pump_on),
    )
    fsm.add_state(
        name="PumpOff",
        timer=0.05,
        transitions={"Tup": ">exit"},
        actions=relay.act(relay.ids.pump_off),
    )
    return fsm


def build_iti_fsm(min_duration_s: float) -> StateMachine:
    fsm = StateMachine(name=f"ITI_{min_duration_s:.1f}s")
    fsm.add_state(
        name="InterTrial",
        timer=min_duration_s,
        transitions={"Tup": ">exit"},
    )
    return fsm


def build_centering_reward_fsm(relay: SerialRelay, pump_duration_s: float) -> StateMachine:
    """Short pump when host sends QUIESCENCE_OK after centering."""
    fsm = StateMachine(name="CenteringReward")
    fsm.add_state(
        name="WaitCentered",
        timer=120.0,
        transitions={"SoftCode0": "PumpOn"},
        comment="Host sends softcode 0 when wheel centered",
    )
    fsm.add_state(
        name="PumpOn",
        timer=pump_duration_s,
        transitions={"Tup": "PumpOff"},
        actions=relay.act(relay.ids.pump_on),
    )
    fsm.add_state(
        name="PumpOff",
        timer=0.05,
        transitions={"Tup": ">exit"},
        actions=relay.act(relay.ids.pump_off),
    )
    return fsm
