"""Verified ROI updates shared by both Spinnaker backends."""

from collections.abc import Callable, Mapping

ROI_NODES = {"width": "Width", "height": "Height", "offset_x": "OffsetX", "offset_y": "OffsetY"}


def apply_roi(
    requested: Mapping[str, int | None],
    read: Callable[[str], int],
    write: Callable[[str, int], None],
) -> dict[str, int]:
    """Retain unspecified values and restore the original crop if any write fails.

    Offsets must be zero before expanding dimensions; dimensions must be reduced
    before moving offsets. Read even an inherited ROI so metadata is reproducible.
    """
    def snapshot() -> dict[str, int]:
        return {key: int(read(node)) for key, node in ROI_NODES.items()}

    original = snapshot()
    target = dict(original)
    for key in ROI_NODES:
        value = requested.get(key)
        if value is not None:
            target[key] = int(value)
    if target == original:
        return original
    if target["width"] <= 0 or target["height"] <= 0 or min(target.values()) < 0:
        raise ValueError(f"Invalid camera ROI: {target}")

    def update(values: dict[str, int]) -> dict[str, int]:
        write("OffsetX", 0)
        write("OffsetY", 0)
        for key, node in ROI_NODES.items():
            write(node, values[key])
        actual = snapshot()
        if actual != values:
            raise RuntimeError(f"Camera ROI readback mismatch: requested {values}, got {actual}")
        return actual

    try:
        return update(target)
    except Exception as exc:
        try:
            update(original)
        except Exception as rollback:
            raise RuntimeError(f"Camera ROI failed ({exc}); restoration also failed: {rollback}") from exc
        raise RuntimeError(f"Camera ROI failed; original ROI restored: {exc}") from exc
