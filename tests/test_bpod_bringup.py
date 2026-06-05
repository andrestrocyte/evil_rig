from types import SimpleNamespace

import pytest

from hf_dm.hardware.discovery import configure_bpod_event_allocation


class FakeSerial:
    def __init__(self, acknowledged: bool = True) -> None:
        self.acknowledged = acknowledged
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.writes.append(bytes(payload))
        return len(payload)

    def read_bool(self) -> bool:
        return self.acknowledged


def make_bpod(*, acknowledged: bool = True):
    modules = {
        "DIO1": SimpleNamespace(n_events=28, is_connected=True),
        "Serial2": SimpleNamespace(n_events=15, is_connected=False),
        "Serial3": SimpleNamespace(n_events=15, is_connected=False),
    }
    return SimpleNamespace(
        is_running=False,
        _hardware=SimpleNamespace(
            input_description=b"UUUXZFFFFBBPPPPP",
            max_serial_events=75,
        ),
        modules=modules,
        _n_softcodes=8,
        serial0=FakeSerial(acknowledged),
    )


def test_bpod_bringup_installs_firmware_event_allocation():
    bpod = make_bpod()

    allocation = configure_bpod_event_allocation(bpod)

    assert allocation["requested_module_event_counts"] == [28, 15, 15]
    assert allocation["event_counts"] == [28, 15, 2, 15, 15]
    assert allocation["softcodes_per_usb"] == 15
    assert allocation["allocated_events"] == 75
    assert bpod._n_softcodes == 15
    assert bpod.serial0.writes == [b"%\x1c\x0f\x02\x0f\x0f"]


def test_bpod_bringup_requires_firmware_acknowledgement():
    bpod = make_bpod(acknowledged=False)

    with pytest.raises(RuntimeError, match="rejected event allocation"):
        configure_bpod_event_allocation(bpod)


def test_bpod_bringup_rejects_connected_modules_that_request_too_many_events():
    bpod = make_bpod()
    bpod.modules["Serial2"].is_connected = True
    bpod.modules["Serial2"].n_events = 40
    bpod.modules["Serial3"].is_connected = True
    bpod.modules["Serial3"].n_events = 40

    with pytest.raises(RuntimeError, match="requested too many event slots"):
        configure_bpod_event_allocation(bpod)
