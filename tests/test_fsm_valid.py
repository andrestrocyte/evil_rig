"""Validate FSM templates compile on mock hardware (no device)."""

from unittest.mock import MagicMock

import pytest

from hf_dm.fsm import softcodes as sc
from hf_dm.fsm.builders import build_host_supervised_decision_fsm
from hf_dm.fsm.reference import (
    build_all_off_fsm,
    build_choice_epoch_fsm,
    build_decision_trial_fsm,
    build_iti_fsm,
    build_reward_fsm,
)
from hf_dm.fsm.relay import RelayMessageIds, SerialRelay
from hf_dm.data.recorder import SessionRecorder
from hf_dm.hardware.bringup_constants import (
    STREAMDIO_BPOD_STATE_OPCODE,
    STREAMDIO_STATE_MARKER_MESSAGE_IDS,
)


@pytest.fixture
def mock_relay():
    bpod = MagicMock()
    module = MagicMock()
    module.is_connected = True
    module.load_serial_message.return_value = True
    bpod.modules = {"Serial1": module}
    relay = SerialRelay(bpod)
    return relay


def test_choice_fsm_validates(mock_relay):
    from bpod_core.fsm import StateMachine

    fsm = build_choice_epoch_fsm(mock_relay, "cw", choice_timeout_s=15.0)
    assert isinstance(fsm, StateMachine)
    fsm.check()


def test_all_off_fsm_validates(mock_relay):
    fsm = build_all_off_fsm(mock_relay)
    fsm.check()
    assert list(fsm.states) == ["PumpOff", "ForceOffCW", "ForceOffCCW", "SpeakerGateOff"]


def test_decision_trial_fsm_branches_reward_and_timeout(mock_relay):
    from bpod_core.fsm import StateMachine

    fsm = build_decision_trial_fsm(
        mock_relay,
        "cw",
        choice_timeout_s=15.0,
        matched_reward_s=0.08,
        opposite_reward_s=0.04,
        timeout_noise_s=2.0,
    )
    assert isinstance(fsm, StateMachine)
    fsm.check()
    assert fsm.states["ChoiceWindow"].transitions["SoftCode0"] == "MatchedForceOffCW"
    assert fsm.states["ChoiceWindow"].transitions["SoftCode1"] == "OppositeForceOffCW"
    assert fsm.states["ChoiceWindow"].transitions["Tup"] == "TimeoutForceOffCW"
    assert fsm.states["MatchedPumpOn"].timer == 0.08
    assert fsm.states["OppositePumpOn"].timer == 0.04
    assert fsm.states["PlayWhiteNoise"].timer == 2.0


def test_production_fsm_owns_quiescence_choice_reward_and_timeout(mock_relay):
    fsm = build_host_supervised_decision_fsm(
        mock_relay,
        quiescence_s=3.0,
        choice_timeout_s=20.0,
        choice_blank_s=1.0,
        high_reward_s=0.08,
        low_reward_s=0.04,
        post_reward_s=1.0,
        timeout_noise_s=2.0,
        post_timeout_s=1.0,
    )

    fsm.check()
    assert next(iter(fsm.states)) == "Quiescence_Pre"
    assert fsm.states["Quiescence_Pre"].timer == 0.001
    assert (
        fsm.states["Quiescence_Pre"].actions[mock_relay.module_name]
        == STREAMDIO_STATE_MARKER_MESSAGE_IDS["Quiescence"]
    )
    assert fsm.states["Quiescence"].transitions["SoftCode2"] == "QuiescenceReset"
    assert fsm.states["Quiescence"].transitions["Tup"] == "ChoiceWindow_Pre"
    assert fsm.states["QuiescenceReset"].transitions["Tup"] == "Quiescence_Pre"
    assert fsm.states["Quiescence"].timer == 3.0
    assert fsm.states["ChoiceWindow"].timer == 20.0
    assert fsm.states["ChoiceWindow"].transitions["SoftCode0"] == "ChoiceBlankHigh_Pre"
    assert fsm.states["ChoiceWindow"].transitions["SoftCode1"] == "ChoiceBlankLow_Pre"
    assert fsm.states["ChoiceWindow"].transitions["Tup"] == "TimeoutNoise_Pre"
    assert fsm.states["ChoiceBlankHigh"].timer == 1.0
    assert fsm.states["ChoiceBlankHigh"].transitions["Tup"] == "RewardHigh_Pre"
    assert fsm.states["RewardHigh"].timer == 0.08
    assert fsm.states["RewardLow"].timer == 0.04
    assert fsm.states["RewardHigh"].actions[mock_relay.module_name] == mock_relay.ids.pump_on
    assert fsm.states["RewardLow"].actions[mock_relay.module_name] == mock_relay.ids.pump_on
    assert fsm.states["PostRewardHigh"].timer == 1.0
    assert (
        fsm.states["PostRewardHigh"].actions[mock_relay.module_name]
        == mock_relay.ids.pump_off
    )
    assert (
        fsm.states["TimeoutNoise_Pre"].actions[mock_relay.module_name]
        == STREAMDIO_STATE_MARKER_MESSAGE_IDS["TimeoutNoise"]
    )
    assert fsm.states["TimeoutNoise"].actions[mock_relay.module_name] == 210
    assert fsm.states["PostTimeout"].actions[mock_relay.module_name] == 211


def test_production_fsm_directly_times_active_force_transitions(mock_relay):
    fsm = build_host_supervised_decision_fsm(
        mock_relay,
        force_direction="cw",
        quiescence_s=3.0,
        choice_timeout_s=20.0,
        choice_blank_s=1.0,
        high_reward_s=0.08,
        low_reward_s=0.04,
        post_reward_s=1.0,
        timeout_noise_s=2.0,
        post_timeout_s=1.0,
    )

    fsm.check()
    assert fsm.states["ChoiceWindow"].actions[mock_relay.module_name] == mock_relay.ids.cw_force_on
    assert (
        fsm.states["ChoiceBlankHigh"].actions[mock_relay.module_name]
        == mock_relay.ids.cw_force_off
    )
    assert (
        fsm.states["ChoiceBlankLow"].actions[mock_relay.module_name]
        == mock_relay.ids.cw_force_off
    )
    assert fsm.states["ChoiceWindow"].transitions["Tup"] == "TimeoutForceOff"
    assert (
        fsm.states["TimeoutForceOff"].actions[mock_relay.module_name]
        == mock_relay.ids.cw_force_off
    )
    assert fsm.states["TimeoutForceOff"].transitions["Tup"] == "TimeoutNoise_Pre"


def test_production_fsm_never_directly_opens_pump_for_omitted_reward(mock_relay):
    fsm = build_host_supervised_decision_fsm(
        mock_relay,
        quiescence_s=3.0,
        choice_timeout_s=20.0,
        choice_blank_s=1.0,
        high_reward_s=0.001,
        low_reward_s=0.04,
        high_reward_enabled=False,
        post_reward_s=1.0,
        timeout_noise_s=2.0,
        post_timeout_s=1.0,
    )

    fsm.check()
    assert mock_relay.module_name not in fsm.states["RewardHigh"].actions
    assert fsm.states["RewardLow"].actions[mock_relay.module_name] == mock_relay.ids.pump_on


def test_production_fsm_supports_shape_specific_reward_delays(mock_relay):
    fsm = build_host_supervised_decision_fsm(
        mock_relay,
        quiescence_s=3.0,
        choice_timeout_s=20.0,
        choice_blank_s=0.1,
        high_reward_s=0.08,
        low_reward_s=0.04,
        post_reward_s=1.0,
        timeout_noise_s=2.0,
        post_timeout_s=1.0,
        high_delay_s=2.1,
        low_delay_s=0.1,
    )

    fsm.check()
    assert fsm.states["ChoiceBlankHigh"].timer == 2.1
    assert fsm.states["ChoiceBlankLow"].timer == 0.1


def test_decision_trial_reverses_reward_mapping_for_ccw_force(mock_relay):
    fsm = build_decision_trial_fsm(
        mock_relay,
        "ccw",
        choice_timeout_s=15.0,
        matched_reward_s=0.08,
        opposite_reward_s=0.04,
    )
    assert fsm.states["ChoiceWindow"].transitions["SoftCode0"] == "OppositeForceOffCW"
    assert fsm.states["ChoiceWindow"].transitions["SoftCode1"] == "MatchedForceOffCW"


def test_reward_fsm_validates(mock_relay):
    from bpod_core.fsm import StateMachine

    fsm = build_reward_fsm(mock_relay, 0.08)
    assert isinstance(fsm, StateMachine)
    fsm.check()


def test_iti_fsm_validates():
    fsm = build_iti_fsm(3.0)
    fsm.check()


def test_message_ids():
    ids = RelayMessageIds()
    assert ids.speaker_gate_on == 210


def test_serial_relay_preloads_streamdio_state_markers():
    bpod = MagicMock()
    module = MagicMock()
    module.is_connected = True
    module.index = 0
    module.load_serial_message.return_value = True
    bpod.modules = {"Serial1": module}

    SerialRelay(bpod, serial_channel_index=0)

    module.set_relay.assert_called_once_with(False)
    loaded = {
        call.args[0]: list(call.args[1])
        for call in module.load_serial_message.call_args_list
    }
    assert loaded[STREAMDIO_STATE_MARKER_MESSAGE_IDS["Quiescence"]] == [
        STREAMDIO_BPOD_STATE_OPCODE,
        1,
    ]
    assert loaded[STREAMDIO_STATE_MARKER_MESSAGE_IDS["TimeoutNoise"]] == [
        STREAMDIO_BPOD_STATE_OPCODE,
        9,
    ]


def test_serial_relay_sends_immediate_bytes_on_configured_output():
    bpod = MagicMock()
    module = MagicMock()
    module.is_connected = True
    module.index = 0
    module.load_serial_message.return_value = True
    serial_out = MagicMock()
    bpod.modules = {"Serial1": module}
    bpod.outputs = {"Serial1": serial_out}

    relay = SerialRelay(bpod, serial_channel_index=0)
    relay.send_bytes([5, 0, 3, 1], inter_byte_s=0)

    assert module.set_relay.call_count == 2
    assert all(call.args == (False,) for call in module.set_relay.call_args_list)
    assert [call.args[0] for call in serial_out.override.call_args_list] == [5, 0, 3, 1]


def test_serial_relay_clears_pending_input_before_bpod_run():
    bpod = MagicMock()
    module = MagicMock()
    module.is_connected = True
    module.index = 0
    module.load_serial_message.return_value = True
    bpod.modules = {"Serial1": module}
    bpod.serial0.in_waiting = 7

    relay = SerialRelay(bpod, serial_channel_index=0)
    discarded = relay.prepare_bpod_run()

    assert discarded == 7
    assert all(call.args == (False,) for call in module.set_relay.call_args_list)
    bpod.serial0.reset_input_buffer.assert_called_once_with()


def test_choice_softcodes_are_zero_based_for_bpod_core():
    assert sc.CHOICE_HIGH == 0
    assert sc.CHOICE_LOW == 1
    assert sc.QUIESCENCE_MOVEMENT == 2


def test_recorder_stores_fsm_json(mock_relay, tmp_path):
    rec = SessionRecorder(out_dir=tmp_path, subject_id="S1", phase="dm")
    fsm = build_choice_epoch_fsm(mock_relay, "cw", choice_timeout_s=15.0)

    rec.log_fsm_definition("choice_epoch_cw", fsm)
    assert "choice_epoch_cw" in rec.fsm_definitions
    assert isinstance(rec.fsm_definitions["choice_epoch_cw"], str)
    assert len(rec.fsm_definitions["choice_epoch_cw"]) > 10
