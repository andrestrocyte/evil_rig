"""Validate FSM templates compile on mock hardware (no device)."""

from unittest.mock import MagicMock

import pytest

from hf_dm.fsm import softcodes as sc
from hf_dm.fsm.builders import (
    build_all_off_fsm,
    build_choice_epoch_fsm,
    build_decision_trial_fsm,
    build_host_supervised_decision_fsm,
    build_iti_fsm,
    build_reward_fsm,
)
from hf_dm.fsm.relay import RelayMessageIds, SerialRelay
from hf_dm.data.recorder import SessionRecorder


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
    assert fsm.states["Quiescence"].transitions["SoftCode2"] == "QuiescenceReset"
    assert fsm.states["Quiescence"].timer == 3.0
    assert fsm.states["ChoiceWindow"].timer == 20.0
    assert fsm.states["ChoiceWindow"].transitions["Tup"] == "TimeoutNoise"
    assert fsm.states["ChoiceBlankHigh"].timer == 1.0
    assert fsm.states["RewardHigh"].timer == 0.08
    assert fsm.states["RewardLow"].timer == 0.04
    assert fsm.states["PostRewardHigh"].timer == 1.0
    assert fsm.states["TimeoutNoise"].actions[mock_relay.module_name] == 210
    assert fsm.states["PostTimeout"].actions[mock_relay.module_name] == 211


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
    from bpod_core.fsm import StateMachine

    fsm = build_iti_fsm(3.0)
    fsm.check()


def test_message_ids():
    ids = RelayMessageIds()
    assert ids.speaker_gate_on == 210


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

    assert [call.args[0] for call in serial_out.override.call_args_list] == [5, 0, 3, 1]


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
