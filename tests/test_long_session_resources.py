from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from hf_dm.streamdio.wheel import WheelAccumulator, _histogram_percentile
from hf_dm.session.run_control import RunControl


@pytest.mark.parametrize("count", [1, 2, 20, 10001])
def test_exact_force_histogram_matches_full_sample_percentile(count):
    rng = np.random.default_rng(41)
    values = rng.integers(0, 4096, size=count).tolist()
    histogram = Counter(float(value) for value in values)
    for percentile in (0, 5, 50, 95, 100):
        assert _histogram_percentile(histogram, count, percentile) == pytest.approx(np.percentile(values, percentile))
    wheel = WheelAccumulator()
    for index, value in enumerate(values):
        wheel.handle_event('analog_input', 13, value, index, index / 1000)
    assert wheel.snapshot()['force_p95'] == pytest.approx(np.percentile(values, 95))
    assert len(wheel._force_histogram) <= 4096
    assert wheel.snapshot(include_percentile=False)['force_p95'] is None
    wheel.reset_trial()
    assert not wheel._force_histogram
    assert wheel.snapshot()['force_p95'] is None


def test_percentile_calculation_releases_acquisition_lock(monkeypatch):
    from hf_dm.streamdio import wheel as module
    wheel = WheelAccumulator()
    wheel.handle_event('analog_input', 13, 10, 0, 0)
    def percentile(*args):
        assert wheel.lock.acquire(blocking=False)
        wheel.lock.release()
        return 10
    monkeypatch.setattr(module, '_histogram_percentile', percentile)
    assert wheel.snapshot()['force_p95'] == 10


def test_storage_guard_checks_destination_before_hardware_and_preserves_reserve(tmp_path, monkeypatch):
    from hf_dm.session import run_control
    calls = []
    free = [5000]
    def usage(path):
        calls.append(path)
        return SimpleNamespace(free=free[0])
    monkeypatch.setattr(run_control.shutil, 'disk_usage', usage)
    guard = RunControl(storage_paths=[tmp_path/'new/session', tmp_path/'videos'], reserve_bytes=1000)
    guard.check()
    assert len(calls) == 1  # both directories share the same volume
    guard.check()
    assert len(calls) == 1  # no disk I/O in each display frame
    free[0] = 900
    guard._next_storage_check = 0
    with pytest.raises(OSError, match='Low disk space'):
        guard.check()


def test_camera_history_survives_driver_ring_without_analog_scan():
    from hf_dm.streamdio.client import StreamDIOClient

    client = StreamDIOClient(baudrate=0)
    client._camera_output = 6  # the camera trigger's module output number
    client._analog = {1: 13}
    for index in range(60000):
        client._dispatch_event("analog_in", 1, 1, index, index / 1000)
        if index % 30 == 0:
            client._dispatch_event("dio_output", 6, 1, index, index / 1000)
            client._dispatch_event("dio_output", 6, 0, index + 1, index / 1000)
    # One camera event per frame (the trigger's rising edge), none for the analog.
    assert len(client.get_camera_events()) == 2000
    assert client.get_camera_events()[0] == (0, 0)
    copy = client.get_camera_events()
    copy.clear()
    assert len(client.get_camera_events()) == 2000
