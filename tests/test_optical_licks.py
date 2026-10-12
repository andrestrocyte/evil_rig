from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

from hf_dm.cli import _apply_session_setup, _setup_from_values
from hf_dm.fsm.relay import SerialRelay
from hf_dm.config import ConfigValidationError, SessionConfig, validate_session_config
from hf_dm.data.session_export import _flags, _write_licks_csv
from hf_dm.monitor.gui import OperatorMonitorState
from hf_dm.session.controller import HFDecisionSession
from hf_dm.session.optical_licks import BpodLickReader, BpodLickForwarder, BpodLickWorker
from hf_dm.session.simulated import (
    SimulatedHFDecisionSession,
    SimulationOptions,
    shorten_for_simulation,
)


@pytest.mark.parametrize("entry,release", [("In", "Out"), ("_High", "_Low")])
def test_optical_reader_keeps_hardware_times_without_duplicate_or_lost_final_edges(entry, release):
    epoch = datetime(2026, 1, 1)
    base_us = 1_767_225_600_000_000
    rows = [(f"Port1{entry}", 100), (f"Port2{entry}", 101)]
    bpod = SimpleNamespace(
        _time_reference=SimpleNamespace(reset_system_time_ns=base_us * 1000),
        _event_thread=object(),
    )

    def snapshot():
        return pl.DataFrame(
            {
                "time": [epoch + timedelta(microseconds=t) for _, t in rows],
                "trial": [7] * len(rows),
                "event": [name for name, _ in rows],
            }
        )

    bpod.peek_data = snapshot
    edges = []
    reader = BpodLickReader(bpod, 1, edges.append)
    reader.poll()
    reader.poll()
    assert len(edges) == 1
    assert edges[0]["timestamp_us"] == 100
    assert edges[0]["bpod_trial"] == 7
    # A late release after the FSM reader stops must still be captured.
    rows.append((f"Port1{release}", 150))
    reader.poll()
    assert [e["value"] for e in edges] == [1, 0]
    assert edges[-1]["relative_time_s"] == 0.00015
    assert edges[-1]["source"] == "bpod_optical"
    assert edges[-1]["clock"] == "bpod_session"
    # Next state machine gets a fresh event buffer, even if shorter.
    bpod._event_thread = object()
    rows[:] = [(f"Port1{entry}", 200)]
    reader.poll()
    reader.poll()
    assert [e["timestamp_us"] for e in edges] == [100, 150, 200]
    assert reader.edge_count == 3


def test_lick_worker_surfaces_failure_without_blocking_display():
    import threading
    entered, release = threading.Event(), threading.Event()
    def poll():
        entered.set()
        release.wait(1)
        raise ValueError("receipt missing")
    worker = BpodLickWorker(SimpleNamespace(poll=poll))
    worker.start()
    assert entered.wait(1)
    worker.raise_if_error()  # display continues while a receipt is outstanding
    release.set()
    worker.stop()
    with pytest.raises(RuntimeError, match="Optical lick acquisition"):
        worker.raise_if_error()


def test_lick_reader_drain_has_a_lock_timeout():
    bpod = SimpleNamespace(_time_reference=SimpleNamespace(reset_system_time_ns=0))
    reader = BpodLickReader(bpod, 1, lambda event: None)
    with reader._poll_lock:
        with pytest.raises(TimeoutError, match="previous events"):
            reader.poll(timeout_s=0.001)


def test_optical_selection_ignores_streamdio_touch_and_counts_only_beam_entries():
    config = SessionConfig()
    config.licks.source = "optical"
    session = HFDecisionSession(config, run_mode="bench")
    messages = []
    session._publish_monitor = lambda kind, payload: messages.append((kind, payload))
    session._handle_streamdio_event("dio_input", 14, 1, 100, 0.0001)
    assert session._lick_event_count() == 0
    for value in (1, 0):
        session._record_lick_edge(
            {
                "channel": 1,
                "pin": None,
                "value": value,
                "source": "bpod_optical",
                "timestamp_us": 200 + value,
                "relative_time_s": 0.0002,
            }
        )
    assert session._lick_event_count() == 1
    assert [kind for kind, _ in messages] == ["lick_edge", "lick", "lick_edge"]
    state = OperatorMonitorState()
    for kind, payload in messages:
        state.apply({"type": kind, "payload": payload})
    assert state.lick_count == 1
    assert state.last_lick_source == "optical"
    assert state.lick_input_states[1]["active"] is False
    assert session._lick_input_descriptors()[0]["label"] == "Bpod Port1 (optical)"


def test_lick_source_yaml_validation_and_session_setup_snapshot(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("licks:\n  source: optical\n  bpod_port: 5\n  channels: [14]\n")
    config = SessionConfig.from_yaml(path)
    validate_session_config(config)  # Port5 and StreamDIO DI05 are separate namespaces.
    assert config.licks.active_channels == (5,)
    assert config.licks.streamdio_channels == ()
    setup = _setup_from_values(mouse_dir=tmp_path, lick_source="optical", optical_clock="streamdio")
    _apply_session_setup(config, setup)
    import json

    info = json.loads((tmp_path / setup.session_name / "session_info.json").read_text())
    assert info["config"]["licks"]["source"] == "optical"
    assert info["setup"]["lick_source"] == "optical"
    assert info["config"]["licks"]["optical_clock"] == "streamdio"
    config.licks.source = "both"
    with pytest.raises(ConfigValidationError, match="licks.source"):
        validate_session_config(config)


def test_optical_iti_acquires_inputs_with_a_timed_output_free_fsm():
    config = SessionConfig()
    config.licks.source = "optical"
    session = HFDecisionSession(config, run_mode="bench")
    fsms = []

    def run(fsm, *, trial_number):
        fsm.check()
        fsms.append(fsm)

    session.bpod = SimpleNamespace(
        run=run,
        is_running=False,
        wait=lambda: None,
        get_data=lambda **kwargs: "captured optical events",
    )
    session.recorder = SimpleNamespace(log_bpod_trial=Mock())
    session._bpod_lick_reader = SimpleNamespace(poll=Mock())
    session._observe_licks_during_black(3.0)
    assert fsms[0].states["Observe"].timer == 3.0
    assert fsms[0].states["Observe"].actions == {}
    assert fsms[0].states["Observe"].transitions["SoftCode3"] == ">exit"
    session._bpod_lick_reader.poll.assert_called_once()
    session.recorder.log_bpod_trial.assert_called_once_with(0, "captured optical events")


def test_optical_csv_identifies_clock_and_input_state(tmp_path):
    import csv

    events = [
        {
            "event_type": "lick_edge",
            "payload": {
                "channel": 1,
                "value": 0,
                "timestamp_us": 120,
                "relative_time_s": 0.00012,
                "source": "bpod_optical",
                "clock": "bpod_session",
                "bpod_trial": 3,
                "event": "Port1Out",
                "counted": False,
            },
        }
    ]
    path = _write_licks_csv(tmp_path / "licks.csv", events)
    with path.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["clock"] == "bpod_session"
    assert row["source"] == "bpod_optical"
    assert row["state"] == "0"
    assert row["bpod_event"] == "Port1Out"
    assert (
        _flags(
            {
                "config": {
                    "licks": {
                        "source": "optical",
                        "power_required": True,
                    }
                }
            }
        )["cap_touch"]
        is False
    )


@pytest.mark.parametrize(
    "source,clock", [("capacitive", "bpod"), ("optical", "bpod"), ("optical", "streamdio")]
)
def test_selected_lick_source_is_recorded_in_simulated_trial(tmp_path, monkeypatch, source, clock):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    config.licks.source = source
    config.licks.optical_clock = clock
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config, SimulationOptions(choice_mode="star", frames_to_choice=4)
    )
    session.connect()
    summary = session.run()
    assert summary.n_trials_completed == 1
    assert session.recorder.trials[0].lick_count == 1
    expected_channel = 1 if source == "optical" else 14
    assert session.recorder.trials[0].lick_channels == [expected_channel]
    expected_source = "bpod_optical" if source == "optical" else "streamdio_capacitive"
    assert session._lick_events[0]["source"] == expected_source
    if source == "optical" and clock == "streamdio":
        lick = session._lick_events[0]
        assert lick["clock"] == "streamdio"
        assert lick["timestamp_origin"] == "streamdio_uart_receive"
        assert lick["timestamp_us"] != lick["bpod_timestamp_us"]
        assert any(row[0] == "port_input" for row in session.recorder.streamdio_events)
    power_ops = [op[1] for op in session.relay.ops if op[0] == "cap_touch_power"]
    assert (True in power_ops) is (source == "capacitive")


def test_forwarded_lick_uses_received_module_time_and_preserves_original_edge_time():
    event = {
        "channel": 1,
        "value": 1,
        "timestamp_us": 900,
        "relative_time_s": 0.0009,
        "clock": "bpod_session",
        "source": "bpod_optical",
    }

    def send(port, state):
        forwarder.accept_receipt(port, state, 12345, 0.012345, 42.0)

    forwarder = BpodLickForwarder(send)
    routed = forwarder.forward(event)
    assert routed["timestamp_us"] == 12345
    assert routed["relative_time_s"] == 0.012345
    assert routed["bpod_timestamp_us"] == 900
    assert routed["bpod_relative_time_s"] == 0.0009
    assert routed["timestamp_origin"] == "streamdio_uart_receive"
    assert routed["clock"] == "streamdio"


@pytest.mark.parametrize("failure", ["missing", "wrong_state", "duplicate", "write_error"])
def test_forwarder_never_substitutes_a_timestamp_when_receipt_fails(failure):
    def send(port, state):
        if failure == "write_error":
            raise OSError("USB disconnected")
        if failure == "wrong_state":
            forwarder.accept_receipt(port, 0, 1000, 0.001, 1.0)
        if failure == "duplicate":
            forwarder.accept_receipt(port, state, 1000, 0.001, 1.0)
            forwarder.accept_receipt(port, state, 1000, 0.001, 1.0)

    forwarder = BpodLickForwarder(send, timeout_s=0.001)
    with pytest.raises((OSError, RuntimeError, TimeoutError)):
        forwarder.forward(
            {"channel": 1, "value": 1, "timestamp_us": 1, "relative_time_s": 0.000001}
        )
    assert forwarder._pending is None


@pytest.mark.parametrize("module_index", [0, 1, 2])
def test_marker_write_is_one_atomic_no_ack_bpod_packet(module_index):
    relay = object.__new__(SerialRelay)
    relay.module = SimpleNamespace(index=module_index, set_relay=Mock())
    relay.bpod = SimpleNamespace(serial0=SimpleNamespace(write=Mock(return_value=7)))
    relay.send_input_marker(1, 1)
    relay.bpod.serial0.write.assert_called_once_with(
        bytes([ord("T"), module_index + 1, 4, ord("I"), ord("P"), 1, 1])
    )
    relay.module.set_relay.assert_not_called()
