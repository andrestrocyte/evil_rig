"""Optional bpod-core FSM templates for relay timing experiments."""

from __future__ import annotations

from bpod_core.fsm import StateMachine

from hf_dm.fsm import softcodes as sc
from hf_dm.fsm.reference.builders import (
    SERIAL_STEP_S,
    Direction,
    build_all_off_fsm,
    build_centering_reward_fsm,
    build_choice_epoch_fsm,
    build_decision_trial_fsm,
    build_iti_fsm,
    build_reward_fsm,
)
from hf_dm.fsm.relay import SerialRelay
STATE_MARKER_S = 0.001

__all__ = [
    "SERIAL_STEP_S",
    "Direction",
    "build_all_off_fsm",
    "build_centering_reward_fsm",
    "build_choice_epoch_fsm",
    "build_decision_trial_fsm",
    "build_host_supervised_decision_fsm",
    "build_iti_fsm",
    "build_reward_fsm",
]


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
    force_direction: Direction | None = None,
    quiescence_s: float,
    choice_timeout_s: float,
    choice_blank_s: float,
    high_reward_s: float,
    low_reward_s: float,
    high_reward_enabled: bool = True,
    low_reward_enabled: bool = True,
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
    force_on_action: dict[str, int] | None = None
    force_off_action: dict[str, int] | None = None
    if force_direction == "cw":
        force_on_action = speaker_relay.act(speaker_relay.ids.cw_force_on)
        force_off_action = speaker_relay.act(speaker_relay.ids.cw_force_off)
    elif force_direction == "ccw":
        force_on_action = speaker_relay.act(speaker_relay.ids.ccw_force_on)
        force_off_action = speaker_relay.act(speaker_relay.ids.ccw_force_off)
    timeout_target = "TimeoutForceOff" if force_off_action is not None else _marker_name("TimeoutNoise")

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
            "Tup": timeout_target,
            **abort_transition,
        },
        actions=_softcode_action(sc.STATE_CHOICE, force_on_action),
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
        actions=_softcode_action(sc.STATE_BLANK, force_off_action),
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
        actions=_softcode_action(sc.STATE_BLANK, force_off_action),
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
        actions=_softcode_action(
            sc.STATE_PUMP_ON,
            (
                speaker_relay.act(speaker_relay.ids.pump_on)
                if high_reward_enabled
                else None
            ),
        ),
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
        actions=_softcode_action(
            sc.STATE_PUMP_ON,
            (
                speaker_relay.act(speaker_relay.ids.pump_on)
                if low_reward_enabled
                else None
            ),
        ),
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
        actions=_softcode_action(
            sc.STATE_PUMP_OFF,
            speaker_relay.act(speaker_relay.ids.pump_off),
        ),
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
        actions=_softcode_action(
            sc.STATE_PUMP_OFF,
            speaker_relay.act(speaker_relay.ids.pump_off),
        ),
    )
    _add_state_marker(
        fsm,
        speaker_relay,
        state_name="TimeoutNoise",
        abort_transition=abort_transition,
    )
    if force_off_action is not None:
        fsm.add_state(
            name="TimeoutForceOff",
            timer=0.001,
            transitions={"Tup": _marker_name("TimeoutNoise"), **abort_transition},
            actions=force_off_action,
            comment="Bpod turns active force off before host-side verification.",
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
