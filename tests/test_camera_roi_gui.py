import sys
import types

import numpy as np

from hf_dm.camera.roi_gui import _normalize_roi, _snap_down, select_spinnaker_roi


def test_roi_values_snap_to_camera_increments_and_sensor_bounds():
    roi = _normalize_roi(
        x=13,
        y=9,
        width=101,
        height=77,
        width_spec={"min": 8, "max": 1440, "inc": 8},
        height_spec={"min": 4, "max": 1080, "inc": 4},
        offset_x_spec={"min": 0, "max": 0, "inc": 8},
        offset_y_spec={"min": 0, "max": 0, "inc": 4},
    )

    assert roi == {"width": 96, "height": 76, "offset_x": 8, "offset_y": 8}
    assert _snap_down(2000, 8, 1440, 8) == 1440


def test_visual_roi_acquires_full_frame_and_cleans_up(monkeypatch):
    class Entry:
        def __init__(self, value):
            self.value = value

        def GetValue(self):
            return self.value

    class Node:
        def __init__(self, *, minimum=0, maximum=1, increment=1):
            self.minimum = minimum
            self.maximum = maximum
            self.increment = increment
            self.value = minimum

        def GetEntryByName(self, name):
            return Entry(name)

        def SetIntValue(self, value):
            self.value = value

        def SetValue(self, value):
            self.value = value

        def GetMin(self):
            return self.minimum

        def GetMax(self):
            return self.maximum

        def GetInc(self):
            return self.increment

    nodes = {
        "TriggerMode": Node(),
        "AcquisitionMode": Node(),
        "ExposureAuto": Node(),
        "ExposureMode": Node(),
        "ExposureTime": Node(minimum=10, maximum=100_000),
        "GainAuto": Node(),
        "Gain": Node(minimum=0, maximum=30),
        "PixelFormat": Node(),
        "OffsetX": Node(minimum=0, maximum=1432, increment=8),
        "OffsetY": Node(minimum=0, maximum=1076, increment=4),
        "Width": Node(minimum=8, maximum=1440, increment=8),
        "Height": Node(minimum=4, maximum=1080, increment=4),
    }

    class NodeMap:
        def GetNode(self, name):
            return nodes[name]

    class Image:
        released = False

        def IsIncomplete(self):
            return False

        def GetNDArray(self):
            return np.zeros((1080, 1440), dtype=np.uint8)

        def Release(self):
            self.released = True

    image = Image()

    class Camera:
        initialized = False
        acquiring = False
        deinitialized = False

        def Init(self):
            self.initialized = True

        def DeInit(self):
            self.deinitialized = True

        def GetNodeMap(self):
            return NodeMap()

        def BeginAcquisition(self):
            self.acquiring = True

        def EndAcquisition(self):
            self.acquiring = False

        def GetNextImage(self, _timeout):
            return image

    camera = Camera()

    class CameraList:
        cleared = False

        def GetSize(self):
            return 1

        def GetByIndex(self, _index):
            return camera

        def Clear(self):
            self.cleared = True

    camera_list = CameraList()

    class SystemInstance:
        released = False

        def GetCameras(self):
            return camera_list

        def ReleaseInstance(self):
            self.released = True

    system = SystemInstance()
    fake_pyspin = types.SimpleNamespace(
        System=types.SimpleNamespace(GetInstance=lambda: system),
        CEnumerationPtr=lambda node: node,
        CFloatPtr=lambda node: node,
        CIntegerPtr=lambda node: node,
        IsAvailable=lambda _node: True,
        IsWritable=lambda _node: True,
        IsReadable=lambda _node: True,
    )
    fake_cv2 = types.SimpleNamespace(
        selectROI=lambda _name, frame, **_kwargs: (13, 9, 101, 77)
        if frame.shape == (1080, 1440)
        else (0, 0, 0, 0),
        destroyWindow=lambda _name: None,
        destroyAllWindows=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "PySpin", fake_pyspin)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    roi = select_spinnaker_roi(exposure_us=12_000, gain_db=4.0)

    assert roi == {"width": 96, "height": 76, "offset_x": 8, "offset_y": 8}
    assert camera.initialized
    assert camera.deinitialized
    assert not camera.acquiring
    assert image.released
    assert camera_list.cleared
    assert system.released
    assert nodes["Width"].value == 1440
    assert nodes["Height"].value == 1080
