from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hf_dm.hardware.discovery import configure_bpod_event_allocation
from hf_dm.hardware import discovery


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


@pytest.mark.parametrize("initially_connected", [False, True])
def test_connect_refreshes_missing_module_before_allocating_events(monkeypatch, initially_connected):
    import bpod_core.bpod as bpod_module

    bpod = make_bpod()
    if not initially_connected:
        bpod.modules = {
            f"Serial{index}": SimpleNamespace(n_events=15, is_connected=False)
            for index in range(1, 4)
        }

    def refresh():
        bpod.modules = make_bpod().modules

    bpod.update_modules = Mock(side_effect=refresh)
    bpod.close = Mock()
    monkeypatch.setattr(bpod_module, "Bpod", lambda **kwargs: bpod)
    monkeypatch.setattr(discovery, "configure_bpod_core_settings_path", lambda: None)
    monkeypatch.setattr(discovery, "ensure_bpod_core_settings_unlocked", lambda path: None)

    assert discovery.connect_bpod_core("COM_TEST") is bpod

    assert bpod.update_modules.call_count == (0 if initially_connected else 1)
    assert bpod.hf_dm_event_allocation["event_counts"] == [28, 15, 2, 15, 15]
    assert bpod.serial0.writes == [b"%\x1c\x0f\x02\x0f\x0f"]
    bpod.close.assert_not_called()


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
