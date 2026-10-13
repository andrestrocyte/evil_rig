from collections import Counter
from types import SimpleNamespace
import threading

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
    class Driver:
        _event_lock = threading.Lock()
        _all_events_history = []
        _event_history = []
        def clear_event_history(self):
            self._all_events_history.clear()
        def start_streaming(self, callback):
            self.callback = callback
            return True
        def get_event_history(self, **kwargs):
            pytest.fail('camera access must not scan the analog history')
    client = StreamDIOClient.__new__(StreamDIOClient)
    client._dio = Driver()
    client.start_streaming()
    raw = client._dio
    for index in range(60000):
        raw._all_events_history.append(('analog_input', 13, 1, index, index/1000))
        if index % 30 == 0:
            raw.callback('camera', 0, 0, index, index/1000)
    assert len(raw._all_events_history) == 50000
    assert len(client.get_camera_events()) == 2000
    assert client.get_camera_events()[0] == (0, 0)
    copy = client.get_camera_events()
    copy.clear()
    assert len(client.get_camera_events()) == 2000
