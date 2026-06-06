# bpod-hf-dm

Head-fixed **decision-making with effort** (wheel + force cue), implemented on **[bpod-core](https://github.com/int-brain-lab/bpod-core)** with StreamDIO hardware.

Design reference: [IBLRIG](https://int-brain-lab.github.io/iblrig/) (IBL two-alternative choice structure) plus the **force-discount wheel** paradigm from `MM_PyBpod2/examples/screen_shape_force_discount_task.py`.

## Hard Rule: Rig Contract

All pin/channel/timing defaults are defined in this repository, mainly in `src/hf_dm/hardware/bringup_constants.py` and `configs/*.yaml`. They match the hardware brought up with the MM_PyBpod2 notebooks:

| Notebook | What this repo uses |
|----------|---------------------|
| `rotary_encoder_bringup.ipynb` | DI01 pulse, DI03 direction, AI13 force (pin 14) |
| `screen_bringup_shape_force_discount.ipynb` | CW force ch3/pin21, CCW ch5/pin19, pump ch1/pin23 |
| `camera_bringup_pin32_ttl.ipynb` | StreamDIO camera 30 Hz, 10 ms exposure |
| `speaker_bringup_pin27_whitenoise.ipynb` | Continuous PC white noise, gated by Bpod serial action on pin 27 |
| `cap_touch_bringup_pin15_power_pin14_touch.ipynb` | Cap-touch power output ch13/pin15; lick edges on input ch14/pin25 |

The task does not read rig parameters from a sister checkout. MM_PyBpod2 is needed as an installed
driver package for `pybpod_streamDIO_module` unless that driver is deliberately vendored later.

Ports default to `AUTO` with the same discovery logic as the screen task.
Cap-touch power is held high for the whole session. Input channel 14 is enabled in the StreamDIO
digital stream; each configured high edge is stored both as a raw module-timestamped event and as
an explicit `lick` session event while force remains sampled on AI13.

## Architecture

```
Session supervisor (Python)
  - pygame display coupled to wheel position at display FPS
  - StreamDIO rotary, force, lick, and camera TTL streams
  - verified StreamDIO USB commands for force, pump, and cap-touch power
  - continuous PC white noise held behind the pin-27 hardware gate
  - required headless Spinnaker recording for the whole session
  - full raw StreamDIO NPZ stream plus JSON/NPZ trial and event records
  - wall timestamps plus high-resolution monotonic `perf_counter` timestamps

Bpod-core trial FSM
  - installs the full firmware module/USB event allocation during bringup
  - owns 3 s wheel quiescence, 20 s choice `Tup`, blank, reward, timeout, and exit
  - receives host SoftCodes only for wheel movement during quiescence and committed choices
  - triggers verified pump/force host actions with output SoftCodes
  - directly gates timeout white noise with its Serial-module pin-27 action
```

**Trial timeline (decision_making phase)**

1. Bpod requires **3 s wheel quiescence**; any streamed wheel movement sends a reset SoftCode.
2. Bpod enters a **20 s ChoiceWindow**. Its `Tup` is the authoritative timeout.
3. A star (high effort/high reward) and circle (low effort/low reward) begin on opposite lateral sides.
4. The live Python loop reads StreamDIO and redraws at 60 Hz. Turning right moves only the left option
   toward center; turning left moves only the right option toward center.
5. Star/circle side is randomized each trial. The masks contain exactly the same number of lit pixels.
6. On a centered choice, Python sends one high/low SoftCode. Bpod immediately enters the blank state,
   turns force off, waits `choice_blank_s`, enters the appropriate pump state, then holds a final
   `post_reward_s` state before exiting.
7. On timeout, Bpod enters the timeout state directly, gates audible white noise on pin 27, gates it
   off after `timeout_noise_s`, holds `post_timeout_s`, and exits.
8. A fresh Bpod FSM starts the next trial. There is no host-side choice wait or extra decision-task ITI.

Pump and effort outputs retain the low-latency, verified StreamDIO USB path and require both a USB
acknowledgement and a timestamped physical `dio_output` event. Bpod controls when those actions occur
through state-entry output SoftCodes.

## Install

```bash
cd /path/to/bpod-hf-dm
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e /path/to/bpod-core
pip install -e /path/to/MM_PyBpod2_andrestrocyte   # StreamDIO + optional sound module
pip install -e .
```

Requires Python >= 3.10, Windows rig (same as bringups), Bpod State Machine 2+.

### Headless Spinnaker recording (no SpinView)

Install the **Teledyne/FLIR Spinnaker SDK** and set `camera.backend: spinnaker`. The recorder uses
`PySpin` when a compatible binding is installed and automatically falls back to the installed
Spinnaker C API when it is not. Both paths configure the verified bringup trigger contract:
`FrameStart`, `Line0`, rising edge, and `TriggerWidth` exposure.

Example:

```yaml
camera:
  frequency_hz: 30
  exposure_us: 10000
  record_video: true
  required: true
  backend: spinnaker
  serial_number: null      # optional: choose a specific camera
  pixel_format: Mono8
  acquisition_timeout_ms: 1000
  save_dir: data/videos
```

The session writes a frame sequence, `frame_timestamps.csv`, and `recording_summary.json` under
`data/videos/<subject>_<phase>_<timestamp>_spinnaker/`. With `required: true`, the task refuses to
run if recording cannot start.

### Operator monitor GUI

The official configs enable a separate lightweight monitor on the computer/main screen
(`monitor.display_index: 0`), while the task stimulus remains on the behavior display
(`display.display_index: 1`). The monitor runs in its own process and receives only nonblocking,
drop-safe telemetry, so closing or slowing the GUI does not stall the Bpod/StreamDIO task loop.

It shows a low-rate camera preview, a compact left-rotated task screen copy, the active Bpod FSM
state, latest star/circle/timeout outcome squares, lick-edge/pump/noise event rails, raw cap-touch
pin status, a live force-effort ADC trace with a session-held y-axis maximum, and the wheel trace
reset every trial.
Use `--no-monitor` for command-line checks, or `--monitor` to force it on when a YAML file has it
disabled. `--simulate-headless` disables the monitor automatically.

Cap touch uses StreamDIO input channel 14 / physical pin 25 for the lick detector and output
channel 13 / physical pin 15 for detector power. The GUI shows both the power output and the raw
input channel value so detector transitions are visible even before trial-level counted licks.
Because the firmware can stream only one configured analog input at a time, force remains on analog
channel 13 and the vendored StreamDIO firmware in `firmware/` digitizes cap-touch pin 25 locally
into normal channel-14 high/low input events. Flash that firmware if the detector LED changes but no
lick events appear in the GUI. With firmware version 3 or newer, the operator monitor also polls the
firmware digitizer status at low rate so the cap-touch ADC value/baseline is visible without moving
force off analog channel 13.

## Run

```bash
# No-trial hardware preflight: Bpod/StreamDIO/Spinnaker streams and cleanup only.
python -m hf_dm.cli --config configs/dm_50.yaml --preflight --preflight-duration 2

# Controlled live check. WARNING: activates real force, pump, speaker, camera, and StreamDIO.
python -m hf_dm.cli --config configs/dm_50.yaml --hardware-self-test

# Cap-touch input check only. Touch the lick detector during the window.
python -m hf_dm.cli --config configs/dm_50.yaml --cap-touch-diagnostic --cap-touch-duration 15

# Stage 1 - centering shaping (bring a lateral cue into the center target)
python -m hf_dm.cli --config configs/centering_50.yaml

# Stage 2-4 - full DM (50 / 100 / 200 trials)
python -m hf_dm.cli --config configs/dm_50.yaml
python -m hf_dm.cli --config configs/dm_100.yaml
python -m hf_dm.cli --config configs/dm_200.yaml

# Same tasks without the operator GUI
python -m hf_dm.cli --config configs/dm_50.yaml --no-monitor
```

Outputs: `data/sessions/<subject>_<phase>_<timestamp>.npz` and `.json`.

On this Windows rig, `python -m hf_dm.cli ...` is the reliable launcher. An editable user-site
install may place `hf-dm-run.exe` under `%APPDATA%\Python\Python312\Scripts`, which is not always on
PowerShell's `PATH`.

### Hardware-free simulation

Use this before touching the live rig to verify the real Bpod-style state sequence, screen motion,
reward timing, timeout gate, lick logging, camera TTL bookkeeping, and cleanup/finalization:

```bash
python -m hf_dm.cli --config configs/dm_50.yaml --simulate --simulate-headless --simulate-trials 3 --simulate-choice-mode alternate
```

Drop `--simulate-headless` to open the pygame task window. Choice modes are `matched`, `opposite`,
`timeout`, `cw`, `ccw`, and `alternate`. In the decision task, matched means star/high-effort and
opposite means circle/low-effort.

### Live rig troubleshooting

This task uses its own bpod-core settings namespace under `data/runtime/bpod-core`, so unrelated
services using the global bpod-core settings lock do not block it. Override the location with
`HF_DM_BPOD_CORE_CONFIG_DIR`. If the task-specific lock is held, close the older `hf_dm.cli` task
process and rerun the preflight.

If StreamDIO reports `Access is denied` on COM7/COM8, another process owns the StreamDIO USB port.
Close old notebooks, serial monitors, or task processes, then unplug/replug StreamDIO if needed.

`--cap-touch-diagnostic` does not start Bpod trials. It opens StreamDIO, keeps force streaming on
analog channel 13, powers the cap-touch detector on output channel 13 / pin 15, polls the firmware
digitizer on input channel 14 / pin 25, and prints ADC value, baseline, threshold, and high/low lick
edge counts. If `hi_edges` increments while touching the detector, the task GUI should also show
licks. If ADC stays near zero or `hi_edges` never increments while the detector is touched, check
the cap-touch sensor wiring/power first, then the firmware threshold.

`--hardware-self-test` runs the production Bpod FSM with synthetic wheel choices but real rig
outputs. It fails unless both effort directions and pump transitions have firmware output events,
choice-to-pump timing matches the configured blank state, pump duration error is below 10 ms, force
ADC and camera TTLs are present, the Bpod timeout gate completes, Spinnaker saves frames, and there
are no choice-loop stalls or incorrectly decoded Bpod states. Licks and real wheel edges still
require touching/turning the physical inputs.
For an input check, run a longer `--preflight`, turn the wheel both ways, and touch the lick sensor;
the saved `preflight_end` event reports wheel pulse/direction edges and cap-touch input edges.

## Data logged

- Per-trial: star/circle sides, chosen shape/side, force direction, choice, reward duration, RT, pulses, force ADC stats, licks, wall/mono timestamps, camera TTL list
- Exact bpod-core state/event rows for every trial in JSON and NPZ `bpod_trials_json`
- Raw NPZ `streamdio_events`: every wheel/direction edge, force sample, camera TTL, lick, module timestamp, and high-resolution host timestamp
- Events: `session_start`, `trial_start`, `stimulus_on`, `force_on`, `choice_detected`, `feedback_on`, `pump_start`, `pump_end`, `output_actual`, `lick`, `white_noise_*`, cleanup, `trial_end`, `session_end`
- Video: every camera frame with camera and host timestamps in `frame_timestamps.csv`
- Session summary: end reason (`completed`, `idle_timeout`, `max_consecutive_failures`)

## Learning curricula

- **[docs/CURRICULUM_BPOD_CORE.md](docs/CURRICULUM_BPOD_CORE.md)** - become fluent in bpod-core using this repo
- **[docs/CURRICULUM_TRAINING.md](docs/CURRICULUM_TRAINING.md)** - mouse training stages (centering -> 50 -> 100 -> 200)

## License

MIT
