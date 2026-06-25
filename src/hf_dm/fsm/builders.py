"""Optional bpod-core FSM templates for relay timing experiments."""

from __future__ import annotations

from typing import Literal

from bpod_core.fsm import StateMachine

from hf_dm.fsm import softcodes as sc
from hf_dm.fsm.relay import SerialRelay

Direction = Literal["cw", "ccw"]

SERIAL_STEP_S = 0.01
STATE_MARKER_S = 0.001


def _softcode_action(value: int, extra: dict[str, int] | None = None) -> dict[str, int]:
    actions = {"SoftCode": int(value)}
    if extra:
        actions.update(extra)
    return actions


def _marker_name(state_name: str) -> str:
    return f"{state_name}_Pre"


def _add_state_marker(
    fsm: StateMachine,
    relay: SerialRelay,
    *,
    state_name: str,
    abort_transition: dict[str, str] | None = None,
) -> str:
    marker_name = _marker_name(state_name)
    transitions = {"Tup": state_name}
    if abort_transition:
        transitions.update(abort_transition)
    fsm.add_state(
        name=marker_name,
        timer=STATE_MARKER_S,
        transitions=transitions,
        actions=relay.bpod_state_marker_act(state_name),
        comment=f"StreamDIO timestamp marker for {state_name}.",
    )
    return marker_name


def build_host_supervised_decision_fsm(
    speaker_relay: SerialRelay,
    *,
    quiescence_s: float,
    choice_timeout_s: float,
    choice_blank_s: float,
    high_reward_s: float,
    low_reward_s: float,
    post_reward_s: float,
    timeout_noise_s: float,
    post_timeout_s: float,
    high_delay_s: float | None = None,
    low_delay_s: float | None = None,
) -> StateMachine:
    """Build the production FSM while Python supervises wheel/display continuously."""
    fsm = StateMachine(name="HostSupervisedDecision")
    abort_transition = {f"SoftCode{sc.SESSION_ABORT}": _marker_name("Abort")}
    high_delay = choice_blank_s if high_delay_s is None else float(high_delay_s)
    low_delay = choice_blank_s if low_delay_s is None else float(low_delay_s)

    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="Quiescence",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="Quiescence",
        timer=quiescence_s,
        transitions={
            f"SoftCode{sc.QUIESCENCE_MOVEMENT}": "QuiescenceReset",
            "Tup": _marker_name("ChoiceWindow"),
            **abort_transition,
        },
        actions=_softcode_action(sc.STATE_QUIESCENCE),
        comment="Any host-reported wheel motion restarts the Bpod-owned stillness timer.",
    )
    fsm.add_state(
        name="QuiescenceReset",
        timer=0.001,
        transitions={"Tup": _marker_name("Quiescence"), **abort_transition},
        comment="A distinct state is required because a self-target is not a Bpod transition.",
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="ChoiceWindow",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="ChoiceWindow",
        timer=choice_timeout_s,
        transitions={
            f"SoftCode{sc.CHOICE_HIGH}": _marker_name("ChoiceBlankHigh"),
            f"SoftCode{sc.CHOICE_LOW}": _marker_name("ChoiceBlankLow"),
            "Tup": _marker_name("TimeoutNoise"),
            **abort_transition,
        },
        actions=_softcode_action(sc.STATE_CHOICE),
        comment="Python renders at 60 Hz; Bpod owns the 20 s Tup timeout.",
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="ChoiceBlankHigh",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="ChoiceBlankHigh",
        timer=high_delay,
        transitions={"Tup": _marker_name("RewardHigh"), **abort_transition},
        actions=_softcode_action(sc.STATE_BLANK),
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="ChoiceBlankLow",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="ChoiceBlankLow",
        timer=low_delay,
        transitions={"Tup": _marker_name("RewardLow"), **abort_transition},
        actions=_softcode_action(sc.STATE_BLANK),
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="RewardHigh",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="RewardHigh",
        timer=high_reward_s,
        transitions={"Tup": _marker_name("PostRewardHigh"), **abort_transition},
        actions=_softcode_action(sc.STATE_PUMP_ON),
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="RewardLow",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="RewardLow",
        timer=low_reward_s,
        transitions={"Tup": _marker_name("PostRewardLow"), **abort_transition},
        actions=_softcode_action(sc.STATE_PUMP_ON),
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="PostRewardHigh",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="PostRewardHigh",
        timer=post_reward_s,
        transitions={"Tup": ">exit", **abort_transition},
        actions=_softcode_action(sc.STATE_PUMP_OFF),
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="PostRewardLow",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="PostRewardLow",
        timer=post_reward_s,
        transitions={"Tup": ">exit", **abort_transition},
        actions=_softcode_action(sc.STATE_PUMP_OFF),
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="TimeoutNoise",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="TimeoutNoise",
        timer=timeout_noise_s,
        transitions={"Tup": _marker_name("PostTimeout"), **abort_transition},
        actions=_softcode_action(
            sc.STATE_TIMEOUT_ON,
            speaker_relay.act(speaker_relay.ids.speaker_gate_on),
        ),
        comment="Bpod directly gates the already-playing PC white noise on pin 27.",
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="PostTimeout",
        abort_transition=abort_transition,
    )
    fsm.add_state(
        name="PostTimeout",
        timer=post_timeout_s,
        transitions={"Tup": ">exit", **abort_transition},
        actions=_softcode_action(
            sc.STATE_TIMEOUT_OFF,
            speaker_relay.act(speaker_relay.ids.speaker_gate_off),
        ),
    )
    _add_state_marker(fsm, speaker_relay, state_name="Abort")
    fsm.add_state(
        name="Abort",
        timer=0.001,
        transitions={"Tup": ">exit"},
        actions=_softcode_action(
            sc.STATE_ABORT,
            speaker_relay.act(speaker_relay.ids.speaker_gate_off),
        ),
    )
    return fsm


def _add_force_off_sequence(
    fsm: StateMachine,
    relay: SerialRelay,
    *,
    prefix: str,
    next_state: str,
) -> None:
    fsm.add_state(
        name=f"{prefix}ForceOffCW",
        timer=SERIAL_STEP_S,
        transitions={"Tup": f"{prefix}ForceOffCCW"},
        actions=relay.act(relay.ids.cw_force_off),
    )
    fsm.add_state(
        name=f"{prefix}ForceOffCCW",
        timer=SERIAL_STEP_S,
        transitions={"Tup": next_state},
        actions=relay.act(relay.ids.ccw_force_off),
    )


def _add_reward_sequence(
    fsm: StateMachine,
    relay: SerialRelay,
    *,
    prefix: str,
    pump_duration_s: float,
) -> None:
    fsm.add_state(
        name=f"{prefix}PumpOn",
        timer=pump_duration_s,
        transitions={"Tup": f"{prefix}PumpOff"},
        actions=relay.act(relay.ids.pump_on),
    )
    fsm.add_state(
        name=f"{prefix}PumpOff",
        timer=SERIAL_STEP_S,
        transitions={"Tup": ">exit"},
        actions=relay.act(relay.ids.pump_off),
    )


def build_all_off_fsm(relay: SerialRelay) -> StateMachine:
    """Put breakout outputs into a quiet baseline: pump/force/speaker all off."""
    fsm = StateMachine(name="AllOff")
    fsm.add_state(
        name="PumpOff",
        timer=SERIAL_STEP_S,
        transitions={"Tup": "ForceOffCW"},
        actions=relay.act(relay.ids.pump_off),
    )
    fsm.add_state(
        name="ForceOffCW",
        timer=SERIAL_STEP_S,
        transitions={"Tup": "ForceOffCCW"},
        actions=relay.act(relay.ids.cw_force_off),
    )
    fsm.add_state(
        name="ForceOffCCW",
        timer=SERIAL_STEP_S,
        transitions={"Tup": "SpeakerGateOff"},
        actions=relay.act(relay.ids.ccw_force_off),
    )
    fsm.add_state(
        name="SpeakerGateOff",
        timer=SERIAL_STEP_S,
        transitions={"Tup": ">exit"},
        actions=relay.act(relay.ids.speaker_gate_off),
    )
    return fsm


def build_decision_trial_fsm(
    relay: SerialRelay,
    force_direction: Direction,
    choice_timeout_s: float,
    matched_reward_s: float,
    opposite_reward_s: float,
    timeout_noise_s: float = 2.0,
) -> StateMachine:
    """
    Reference FSM: force cue, host-reported wheel choice, reward, timeout noise.

    The active task avoids this template during live choices so wheel threshold
    crossings cannot be delayed by Bpod wait/softcode handling.
    """
    fsm = StateMachine(name=f"DecisionTrial_{force_direction}")
    force_on = relay.ids.cw_force_on if force_direction == "cw" else relay.ids.ccw_force_on

    cw_target = "MatchedForceOffCW" if force_direction == "cw" else "OppositeForceOffCW"
    ccw_target = "MatchedForceOffCW" if force_direction == "ccw" else "OppositeForceOffCW"

    _add_force_off_sequence(fsm, relay, prefix="Pre", next_state="ApplyForce")
    fsm.add_state(
        name="ApplyForce",
        timer=SERIAL_STEP_S,
        transitions={"Tup": "ChoiceWindow"},
        actions=relay.act(force_on),
    )
    fsm.add_state(
        name="ChoiceWindow",
        timer=choice_timeout_s,
        transitions={
            "SoftCode0": cw_target,
            "SoftCode1": ccw_target,
            "Tup": "TimeoutForceOffCW",
        },
    )

    _add_force_off_sequence(fsm, relay, prefix="Matched", next_state="MatchedPumpOn")
    _add_reward_sequence(
        fsm,
        relay,
        prefix="Matched",
        pump_duration_s=matched_reward_s,
    )

    _add_force_off_sequence(fsm, relay, prefix="Opposite", next_state="OppositePumpOn")
    _add_reward_sequence(
        fsm,
        relay,
        prefix="Opposite",
        pump_duration_s=opposite_reward_s,
    )

    _add_force_off_sequence(fsm, relay, prefix="Timeout", next_state="PlayWhiteNoise")
    fsm.add_state(
        name="PlayWhiteNoise",
        timer=timeout_noise_s,
        transitions={"Tup": "SpeakerGateOff"},
        actions=relay.act(relay.ids.speaker_gate_on),
        comment="Gate continuous white noise on pin 27 during timeout only",
    )
    fsm.add_state(
        name="SpeakerGateOff",
        timer=SERIAL_STEP_S,
        transitions={"Tup": ">exit"},
        actions=relay.act(relay.ids.speaker_gate_off),
    )
    return fsm


def build_choice_epoch_fsm(
    relay: SerialRelay,
    force_direction: Direction,
    choice_timeout_s: float,
    timeout_noise_s: float = 2.0,
) -> StateMachine:
    """
    Reference FSM for graph validation, not the active live choice loop.

    On Tup (timeout), transitions to white-noise punishment state.
    """
    fsm = StateMachine(name=f"ChoiceEpoch_{force_direction}")
    force_on = relay.ids.cw_force_on if force_direction == "cw" else relay.ids.ccw_force_on

    _add_force_off_sequence(fsm, relay, prefix="Pre", next_state="ApplyForce")
    fsm.add_state(
        name="ApplyForce",
        timer=SERIAL_STEP_S,
        transitions={"Tup": "ChoiceWindow"},
        actions=relay.act(force_on),
    )
    fsm.add_state(
        name="ChoiceWindow",
        timer=choice_timeout_s,
        transitions={
            "SoftCode0": "ChoiceRegisteredForceOffCW",
            "SoftCode1": "ChoiceRegisteredForceOffCW",
            "Tup": "TimeoutForceOffCW",
        },
    )
    _add_force_off_sequence(fsm, relay, prefix="ChoiceRegistered", next_state=">exit")
    _add_force_off_sequence(fsm, relay, prefix="Timeout", next_state="PlayWhiteNoise")
    fsm.add_state(
        name="PlayWhiteNoise",
        timer=timeout_noise_s,
        transitions={"Tup": "SpeakerGateOff"},
        actions=relay.act(relay.ids.speaker_gate_on),
        comment="Gate white noise on pin 27 (speaker bringup)",
    )
    fsm.add_state(
        name="SpeakerGateOff",
        timer=SERIAL_STEP_S,
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
        timer=SERIAL_STEP_S,
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
    """Reference centering FSM for optional experiments."""
    fsm = StateMachine(name="CenteringReward")
    fsm.add_state(
        name="WaitCentered",
        timer=120.0,
        transitions={"SoftCode0": "PumpOn"},
        comment="Reference softcode transition when wheel is centered",
    )
    fsm.add_state(
        name="PumpOn",
        timer=pump_duration_s,
        transitions={"Tup": "PumpOff"},
        actions=relay.act(relay.ids.pump_on),
    )
    fsm.add_state(
        name="PumpOff",
        timer=SERIAL_STEP_S,
        transitions={"Tup": ">exit"},
        actions=relay.act(relay.ids.pump_off),
    )
    return fsm
