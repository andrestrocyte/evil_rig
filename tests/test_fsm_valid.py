"""Validate FSM templates compile on mock hardware (no device)."""

from unittest.mock import MagicMock

import pytest

from hf_dm.fsm.builders import build_choice_epoch_fsm, build_iti_fsm, build_reward_fsm
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


def test_recorder_stores_fsm_json(mock_relay, tmp_path):
    rec = SessionRecorder(out_dir=tmp_path, subject_id="S1", phase="dm")
    fsm = build_choice_epoch_fsm(mock_relay, "cw", choice_timeout_s=15.0)

    rec.log_fsm_definition("choice_epoch_cw", fsm)
    assert "choice_epoch_cw" in rec.fsm_definitions
    assert isinstance(rec.fsm_definitions["choice_epoch_cw"], str)
    assert len(rec.fsm_definitions["choice_epoch_cw"]) > 10
