# Working on evil_rig

This is the Python `hf_dm` Bpod/StreamDIO rig application. See README for operation
and `docs/QUALITY_AUDIT_2026_10.md` for measured hardware coverage and open work.

## Environment

- Python 3.12, dependencies pinned in `requirements.lock`.
- Windows installation: `scripts/install_rig.ps1`; this rig has its own `.venv`.
- Hardware drivers and Spinnaker SDK are optional for tests/simulation.
  Do not assume hardware is present in cloud CI, or absent on a physical rig.
- Data, machine settings and local audit artifacts under `data/` are ignored.
- Preserve the separate `bpod-hf-dm` checkout when using it as a reference.

## Verification

Use the checkout's virtualenv Python:

```text
python -m ruff check src tests
python -m mypy src/hf_dm/config.py src/hf_dm/data/recorder.py src/hf_dm/data/session_export.py
python -m pytest
```

For headless runs, set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.
CI runs these checks on Windows and Linux. Simulation needs no hardware:

```text
python -m hf_dm.cli --config configs/dm_50.yaml --simulate --simulate-headless --simulate-trials 3 --simulate-choice-mode alternate
```

Live checks can operate pump, force and audio outputs; use the user's authorized
bench scope. Preserve raw data and original timestamp semantics. Do not equate a
successful process exit or valid export with successful sensor detection. Keep
failures visible, waits finite, and hardware calibration distinct from software
event verification.
