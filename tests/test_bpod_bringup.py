from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hf_dm.hardware import discovery


def make_bpod():
    return SimpleNamespace(
        modules={
            "StreamDIO1": SimpleNamespace(n_events=24, is_connected=True),
            "Serial2": SimpleNamespace(n_events=15, is_connected=False),
            "Serial3": SimpleNamespace(n_events=6, is_connected=False),
        },
    )


@pytest.mark.parametrize("initially_connected", [False, True])
def test_connect_refreshes_missing_module_once(monkeypatch, initially_connected):
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
    bpod.close.assert_not_called()


def test_connect_closes_the_bpod_when_module_discovery_fails(monkeypatch):
    import bpod_core.bpod as bpod_module

    bpod = SimpleNamespace(
        modules={"Serial1": SimpleNamespace(is_connected=False)},
        update_modules=Mock(side_effect=RuntimeError("no module answered")),
        close=Mock(),
    )
    monkeypatch.setattr(bpod_module, "Bpod", lambda **kwargs: bpod)
    monkeypatch.setattr(discovery, "configure_bpod_core_settings_path", lambda: None)
    monkeypatch.setattr(discovery, "ensure_bpod_core_settings_unlocked", lambda path: None)

    with pytest.raises(RuntimeError, match="no module answered"):
        discovery.connect_bpod_core("COM_TEST")
    bpod.close.assert_called_once()
