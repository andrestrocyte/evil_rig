"""Failure and rollback checks for camera settings, independent of the SDK."""

import ctypes
from functools import partial
from types import SimpleNamespace

import pytest

from hf_dm.camera.settings import apply_roi
from hf_dm.camera.spinnaker import HeadlessSpinnakerRecorder, _SpinnakerCRecorder


class Sensor:
    def __init__(self):
        self.values = {"Width": 1440, "Height": 1080, "OffsetX": 0, "OffsetY": 0}
        self.writes = []

    def write(self, name, value):
        self.writes.append((name, value))
        trial = {**self.values, name: value}
        if trial["Width"] + trial["OffsetX"] > 1440 or trial["Height"] + trial["OffsetY"] > 1080:
            raise RuntimeError("outside sensor")
        self.values.update(trial)

    def apply(self, requested):
        return apply_roi(requested, self.values.__getitem__, self.write)


def test_roi_keeps_inherited_crop_without_writes_and_expands_in_safe_order():
    sensor = Sensor()
    assert sensor.apply({}) == {"width": 1440, "height": 1080, "offset_x": 0, "offset_y": 0}
    assert sensor.writes == []
    crop = sensor.apply({"width": 640, "height": 480, "offset_x": 32, "offset_y": 24})
    sensor.writes.clear()
    assert sensor.apply({}) == crop
    assert sensor.writes == []
    full = sensor.apply({"width": 1440, "height": 1080, "offset_x": 0, "offset_y": 0})
    assert full["width"] == 1440
    assert sensor.writes[:2] == [("OffsetX", 0), ("OffsetY", 0)]


@pytest.mark.parametrize("failure", ["reject", "ignore", "clamp"])
def test_roi_failure_restores_original_crop_after_partial_write(failure):
    sensor = Sensor()
    original = sensor.apply({"width": 640, "height": 480, "offset_x": 32, "offset_y": 24})
    write = sensor.write

    def faulty(name, value):
        if name == "Height" and value == 600:
            if failure == "reject":
                raise RuntimeError("height rejected")
            if failure == "ignore":
                return
            value = 598
        write(name, value)

    sensor.write = faulty
    with pytest.raises(RuntimeError, match="original ROI restored"):
        sensor.apply({"width": 800, "height": 600})
    assert sensor.apply({}) == original


def test_roi_restoration_failure_is_explicit():
    sensor = Sensor()
    sensor.write = lambda *args: (_ for _ in ()).throw(RuntimeError("device lost"))
    with pytest.raises(RuntimeError, match="restoration also failed: device lost"):
        sensor.apply({"width": 800})


@pytest.mark.parametrize("backend", ["c", "pyspin"])
def test_required_enum_detects_successful_write_with_wrong_readback(backend, tmp_path):
    if backend == "c":
        r = _SpinnakerCRecorder(tmp_path, serial_number=None, fps=30, exposure_us=10000,
                               pixel_format="Mono8", gain_db=None, gamma=None, acquisition_timeout_ms=10)
        r._get_node = lambda *args: ctypes.c_void_p(1)

        def entry(node, name, target):
            ctypes.cast(target, ctypes.POINTER(ctypes.c_void_p)).contents.value = 2
            return 0

        r._fn = {"spinEnumerationGetEntryByName": entry,
                 "spinEnumerationEntryGetIntValue": lambda *args: 0,
                 "spinEnumerationSetIntValue": lambda *args: 0}
        r._read_enum = lambda *args: "Off"
        call = partial(r._set_enum, None, "TriggerMode", "On", required=True)
    else:
        r = HeadlessSpinnakerRecorder(tmp_path, "test", "test")
        node = SimpleNamespace(GetEntryByName=lambda name: SimpleNamespace(GetValue=lambda: 1),
                               SetIntValue=lambda value: None)
        r._pyspin = SimpleNamespace(CEnumerationPtr=lambda node: node, IsAvailable=lambda node: True,
                                    IsWritable=lambda node: True, IsReadable=lambda node: True)
        r._read_pyspin_value = lambda *args: "Off"
        call = partial(r._set_pyspin_enum, SimpleNamespace(GetNode=lambda name: node),
                       "TriggerMode", "On", required=True)
    with pytest.raises(RuntimeError, match="readback mismatch"):
        call()
    assert "TriggerMode" not in r._settings_readback


@pytest.mark.parametrize("kind,value,actual", [("Integer", 640, 638), ("Boolean", False, True)])
@pytest.mark.parametrize("backend", ["c", "pyspin"])
def test_explicit_numeric_and_bool_settings_cannot_silently_fail(backend, kind, value, actual, tmp_path):
    if backend == "c":
        r = _SpinnakerCRecorder(tmp_path, serial_number=None, fps=30, exposure_us=10000,
                               pixel_format="Mono8", gain_db=None, gamma=None, acquisition_timeout_ms=10)
        r._get_node = lambda *args: ctypes.c_void_p(1)
        r._fn = {f"spin{kind}SetValue": lambda *args: 0}
        r._read_value = lambda *args: actual
        call = partial(r._set_int, None, "Width", value) if kind == "Integer" else partial(
            r._set_bool, None, "AcquisitionFrameRateEnable", value, required=True)
    else:
        r = HeadlessSpinnakerRecorder(tmp_path, "test", "test")
        node = SimpleNamespace(SetValue=lambda value: None, GetValue=lambda: actual)
        r._pyspin = SimpleNamespace(**{f"C{kind}Ptr": lambda node: node},
                                    IsAvailable=lambda node: True, IsWritable=lambda node: True,
                                    IsReadable=lambda node: True)
        call = partial(r._set_pyspin_value, SimpleNamespace(GetNode=lambda name: node), "Setting", value, kind)
    with pytest.raises(RuntimeError, match="readback mismatch"):
        call()


@pytest.mark.parametrize("backend", ["c", "pyspin"])
def test_explicit_float_missing_node_fails_and_quantized_value_is_recorded(backend, tmp_path):
    if backend == "c":
        r = _SpinnakerCRecorder(tmp_path, serial_number=None, fps=30, exposure_us=10000,
                               pixel_format="Mono8", gain_db=None, gamma=None, acquisition_timeout_ms=10)
        r._get_node = lambda *args: None
        call = partial(r._set_float, None, "ExposureTime", 10000)
        with pytest.raises(RuntimeError, match="unavailable"):
            call()
        r._get_node = lambda *args: ctypes.c_void_p(1)
        r._fn = {"spinFloatSetValue": lambda *args: 0}
        r._read_value = lambda *args: 9999.0
    else:
        r = HeadlessSpinnakerRecorder(tmp_path, "test", "test")
        node = SimpleNamespace(SetValue=lambda value: None, GetValue=lambda: 9999.0)
        r._pyspin = SimpleNamespace(CFloatPtr=lambda node: node, IsAvailable=lambda node: False,
                                    IsWritable=lambda node: True, IsReadable=lambda node: True)
        call = partial(r._set_pyspin_value, SimpleNamespace(GetNode=lambda name: node), "ExposureTime", 10000, "Float")
        with pytest.raises(RuntimeError, match="unavailable"):
            call()
        r._pyspin.IsAvailable = lambda node: True
    call()
    assert r._settings_readback["ExposureTime"] == 9999.0
