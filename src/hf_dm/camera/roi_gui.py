"""Visual full-frame ROI selection for FLIR Spinnaker cameras."""

from __future__ import annotations

from typing import Any


def select_spinnaker_roi(
    *,
    serial_number: str | None = None,
    exposure_us: int = 10_000,
    gain_db: float | None = None,
    pixel_format: str = "Mono8",
    window_name: str = "HF-DM FLIR ROI",
) -> dict[str, int] | None:
    """Acquire a full-sensor frame and return an OpenCV-selected camera ROI."""
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "OpenCV is required for visual ROI setup. Install bpod-hf-dm[video]."
        ) from exc
    try:
        import PySpin  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "PySpin is required for visual ROI setup. Install the FLIR Spinnaker SDK bindings."
        ) from exc

    system: Any = None
    camera_list: Any = None
    camera: Any = None
    image: Any = None
    initialized = False
    acquiring = False
    list_cleared = False
    try:
        system = PySpin.System.GetInstance()
        camera_list = system.GetCameras()
        if int(camera_list.GetSize()) == 0:
            raise RuntimeError("No Spinnaker cameras found")
        camera = _select_camera(camera_list, serial_number)
        camera_list.Clear()
        list_cleared = True

        camera.Init()
        initialized = True
        nodemap = camera.GetNodeMap()
        _set_enum(PySpin, nodemap, "TriggerMode", "Off")
        _set_enum(PySpin, nodemap, "AcquisitionMode", "Continuous")
        _set_enum(PySpin, nodemap, "ExposureAuto", "Off")
        _set_enum(PySpin, nodemap, "ExposureMode", "Timed")
        _set_float(PySpin, nodemap, "ExposureTime", float(exposure_us))
        if gain_db is not None:
            _set_enum(PySpin, nodemap, "GainAuto", "Off")
            _set_float(PySpin, nodemap, "Gain", float(gain_db))
        _set_enum(PySpin, nodemap, "PixelFormat", str(pixel_format))
        _reset_full_frame(PySpin, nodemap)

        width_spec = _integer_spec(PySpin, nodemap, "Width")
        height_spec = _integer_spec(PySpin, nodemap, "Height")
        offset_x_spec = _integer_spec(PySpin, nodemap, "OffsetX")
        offset_y_spec = _integer_spec(PySpin, nodemap, "OffsetY")

        camera.BeginAcquisition()
        acquiring = True
        frame = None
        for _ in range(10):
            image = camera.GetNextImage(1000)
            try:
                if image.IsIncomplete():
                    continue
                array = image.GetNDArray()
                frame = array.copy() if hasattr(array, "copy") else array
                break
            finally:
                image.Release()
                image = None
        if frame is None:
            raise RuntimeError("Could not acquire a complete Spinnaker frame for ROI selection")

        camera.EndAcquisition()
        acquiring = False
        selection = cv2.selectROI(
            window_name,
            frame,
            showCrosshair=True,
            fromCenter=False,
        )
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            cv2.destroyAllWindows()
        x, y, width, height = (int(value) for value in selection)
        if width <= 0 or height <= 0:
            return None
        return _normalize_roi(
            x=x,
            y=y,
            width=width,
            height=height,
            width_spec=width_spec,
            height_spec=height_spec,
            offset_x_spec=offset_x_spec,
            offset_y_spec=offset_y_spec,
        )
    finally:
        if image is not None:
            try:
                image.Release()
            except Exception:
                pass
        if acquiring and camera is not None:
            try:
                camera.EndAcquisition()
            except Exception:
                pass
        if initialized and camera is not None:
            try:
                camera.DeInit()
            except Exception:
                pass
        if camera_list is not None and not list_cleared:
            try:
                camera_list.Clear()
            except Exception:
                pass
        camera = None
        if system is not None:
            try:
                system.ReleaseInstance()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _select_camera(camera_list: Any, serial_number: str | None) -> Any:
    if serial_number in (None, ""):
        return camera_list.GetByIndex(0)
    for index in range(int(camera_list.GetSize())):
        camera = camera_list.GetByIndex(index)
        if str(camera.TLDevice.DeviceSerialNumber.GetValue()) == str(serial_number):
            return camera
    raise RuntimeError(f"Spinnaker camera serial not found: {serial_number}")


def _set_enum(PySpin: Any, nodemap: Any, node_name: str, entry_name: str) -> None:
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        return
    entry = node.GetEntryByName(entry_name)
    if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
        node.SetIntValue(entry.GetValue())


def _set_float(PySpin: Any, nodemap: Any, node_name: str, value: float) -> None:
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        return
    node.SetValue(min(float(node.GetMax()), max(float(node.GetMin()), float(value))))


def _integer_node(PySpin: Any, nodemap: Any, node_name: str) -> Any | None:
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        return None
    return node


def _integer_spec(PySpin: Any, nodemap: Any, node_name: str) -> dict[str, int]:
    node = _integer_node(PySpin, nodemap, node_name)
    if node is None:
        raise RuntimeError(f"Spinnaker ROI node is unavailable: {node_name}")
    increment = int(node.GetInc()) if hasattr(node, "GetInc") else 1
    return {
        "min": int(node.GetMin()),
        "max": int(node.GetMax()),
        "inc": max(1, increment),
    }


def _set_integer(PySpin: Any, nodemap: Any, node_name: str, value: int) -> None:
    node = _integer_node(PySpin, nodemap, node_name)
    if node is None or not PySpin.IsWritable(node):
        return
    minimum = int(node.GetMin())
    maximum = int(node.GetMax())
    increment = max(1, int(node.GetInc()) if hasattr(node, "GetInc") else 1)
    node.SetValue(_snap_down(int(value), minimum, maximum, increment))


def _reset_full_frame(PySpin: Any, nodemap: Any) -> None:
    offset_x = _integer_node(PySpin, nodemap, "OffsetX")
    offset_y = _integer_node(PySpin, nodemap, "OffsetY")
    width = _integer_node(PySpin, nodemap, "Width")
    height = _integer_node(PySpin, nodemap, "Height")
    if offset_x is not None:
        _set_integer(PySpin, nodemap, "OffsetX", int(offset_x.GetMin()))
    if offset_y is not None:
        _set_integer(PySpin, nodemap, "OffsetY", int(offset_y.GetMin()))
    if width is not None:
        _set_integer(PySpin, nodemap, "Width", int(width.GetMax()))
    if height is not None:
        _set_integer(PySpin, nodemap, "Height", int(height.GetMax()))


def _snap_down(value: int, minimum: int, maximum: int, increment: int) -> int:
    value = min(int(maximum), max(int(minimum), int(value)))
    return int(minimum) + ((value - int(minimum)) // max(1, int(increment))) * max(
        1, int(increment)
    )


def _normalize_roi(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    width_spec: dict[str, int],
    height_spec: dict[str, int],
    offset_x_spec: dict[str, int],
    offset_y_spec: dict[str, int],
) -> dict[str, int]:
    normalized_width = _snap_down(
        width,
        width_spec["min"],
        width_spec["max"],
        width_spec["inc"],
    )
    normalized_height = _snap_down(
        height,
        height_spec["min"],
        height_spec["max"],
        height_spec["inc"],
    )
    max_x = max(offset_x_spec["min"], width_spec["max"] - normalized_width)
    max_y = max(offset_y_spec["min"], height_spec["max"] - normalized_height)
    return {
        "width": normalized_width,
        "height": normalized_height,
        "offset_x": _snap_down(
            x,
            offset_x_spec["min"],
            max_x,
            offset_x_spec["inc"],
        ),
        "offset_y": _snap_down(
            y,
            offset_y_spec["min"],
            max_y,
            offset_y_spec["inc"],
        ),
    }
