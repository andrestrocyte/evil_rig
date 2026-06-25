# AGENTS.md

## Cursor Cloud specific instructions

This is a single Python package (`hf_dm`, the head-fixed decision-making Bpod task). It is a
CLI-launched rig application with no web server, database, or network ports. Everything runs in one
process and talks to hardware over USB/serial. See `README.md` for the full command reference.

### Environment
- Python 3.12 with a project virtualenv at `.venv` (the startup update script creates it and runs
  `pip install -e ".[dev]"`). Activate it with `source .venv/bin/activate` before running anything.
- The runtime dep `bpod-core` and dev deps (`pytest`, `mypy`) come from PyPI. The live-hardware
  drivers `pybpod_streamDIO_module` (from `MM_PyBpod2`), the Spinnaker SDK / `PySpin`, and the sound
  module are **not installed** and are **not needed** for tests or simulation — they are imported
  lazily only when connecting to real rig hardware.

### Running without hardware (the dev/run path here)
- There is no real rig in this environment, so use simulation mode:
  `python -m hf_dm.cli --config configs/dm_50.yaml --simulate --simulate-headless --simulate-trials 3 --simulate-choice-mode alternate`
- Run headless GUI/pygame code with `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy` exported. It is not
  strictly required (the session still completes without it) but it suppresses noisy ALSA/SDL audio
  errors since the VM has no audio/display device.
- Live/preflight/diagnostic subcommands (`--preflight`, `--hardware-self-test`,
  `--cap-touch-diagnostic`, `--frame2ttl-*`, `--bpod-opcode-diagnostic`) require physical hardware and
  will not work in this environment.
- Sessions write outputs under `data/sessions/` (gitignored).

### Test / type-check
- Tests: `pytest` (testpaths = `tests`); the suite is hardware-free and passes headless.
- `mypy src` runs (dev extra) but currently reports pre-existing type errors; it is not wired up as a
  required/configured lint gate, so treat its output as informational. There is no other linter.
