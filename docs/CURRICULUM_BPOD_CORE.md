# Curriculum: mastering bpod-core through `bpod-hf-dm`

Goal: understand bpod-core session control and StreamDIO host supervision well enough to extend this task without returning to ad-hoc `pybpodapi` scripts.

Estimated time: 2-3 weeks part-time on the rig.

## Week 1 - bpod-core foundations

### Day 1-2: Mental model

Read in order:

1. [bpod-core state machine docs](https://int-brain-lab.github.io/bpod-core/state_machines/index.html)
2. `bpod-core/examples/state_machines/two_choice.py` for canonical discrete choices
3. [IBLRIG overview](https://int-brain-lab.github.io/iblrig/) for how behavior, acquisition, and stimuli are separated

Exercise: Draw how a discrete port event maps to states in bpod-core, then contrast it with this task's continuous wheel loop.

### Day 3-4: API surface

Study `bpod-core/bpod_core/bpod/__init__.py` and the output APIs:

- `Bpod(...)` connection and session clock reset
- `bpod.outputs["Serial1"].override(byte)` for optional/reference relay experiments
- `Module.load_serial_message(...)` for optional FSM templates
- `Bpod.run(fsm)` starts the active trial without blocking the display loop
- `Bpod.send_softcode(...)` reports quiescence movement and committed choices
- `Bpod.set_softcode_handler(...)` lets state entries trigger verified host outputs
- `Bpod.get_data(...)` stores the exact completed trial record

Important: the active decision-making trial is a real bpod-core FSM. Bpod owns all state timers and
branches, while the Python loop remains free to render and read StreamDIO continuously.

### Day 5: Validation and compilation

Read `bpod_core/fsm.py` (`check`, `add_state`) and skim `bpod_core/bpod/_threads.py`.

Exercise: Run the FSM template tests and inspect the generated graph, but remember those templates are reference material. The live task path is in `hf_dm/session/controller.py`.

## Week 2 - This repository

### Day 1: Hardware layer

| File | Learn |
|------|-------|
| `hf_dm/hardware/bringup_constants.py` | Single source of pin truth |
| `hf_dm/hardware/discovery.py` | AUTO COM discovery and firmware event allocation from bringups |

Exercise: Run the MM bringup notebooks and confirm constants match printed channel numbers.

The bringup step installs the same complete module/USB event allocation in firmware and bpod-core.
This is required for host SoftCodes to trigger transitions and for completed Bpod state rows to
decode without an event-index shift.

### Day 2: StreamDIO adapter

| File | Learn |
|------|-------|
| `hf_dm/streamdio/client.py` | Wrapper over `StreamDIOProtocol` |
| `hf_dm/streamdio/wheel.py` | Pulse/direction to signed position |

The adapter also prevents a bringup-driver startup race by obtaining the stream timestamp before
streaming begins. It suppresses per-sample console logging while preserving every raw event in the
session NPZ.

Exercise: Log 10 seconds of events; verify `deg_per_pulse` against a hand turn and inspect the
`streamdio_events` NPZ array.

### Day 3: Verified StreamDIO outputs

| File | Learn |
|------|-------|
| `hf_dm/streamdio/outputs.py` | Active pump, effort, cap-touch power, and speaker control |
| `hf_dm/fsm/relay.py` | Bringup-compatible Bpod Serial-module messages, including pin-27 gate |
| `hf_dm/fsm/builders.py` | Active Bpod decision-trial FSM |
| `hf_dm/fsm/runtime.py` | Thread-safe bridge between Bpod states and the host loop |

Exercise: Run the hardware self-test and inspect `output_actual` events for pump, both effort
directions, and cap-touch power. Every changed physical output must have a USB acknowledgement and a
firmware `dio_output` event.

### Day 4: Session supervisor

| File | Learn |
|------|-------|
| `hf_dm/session/controller.py` | Full trial timeline and live choice loop |
| `hf_dm/data/recorder.py` | NPZ/JSON schema |

Exercise: Run `--simulate --simulate-headless --simulate-trials 3` and inspect `bpod_trials` in the
JSON. The rewarded branch should read `Quiescence`, `ChoiceWindow`, `ChoiceBlank*`, `Reward*`,
`PostReward*`; timeout should read `Quiescence`, `ChoiceWindow`, `TimeoutNoise`, `PostTimeout`.

Then run `--hardware-self-test` with the rig clear. It drives matched, opposite, and timeout choices
through the production loop while activating real outputs. Verify zero choice-loop stalls, precise
80/40 ms pump durations, timeout noise, all-off cleanup, and matching camera TTL/frame counts.

### Day 5: Display, audio, camera

| File | Learn |
|------|-------|
| `hf_dm/display/pygame_display.py` | Equal-pixel lateral star/circle options moving to center |
| `hf_dm/audio/white_noise.py` | Continuous soundcard white noise from the speaker bringup |
| Bpod Serial-module action | Hardware gate on Teensy pin 27 |
| `hf_dm/camera/spinnaker.py` | Headless PySpin/C API recorder with external-trigger configuration |

## Key design decision

The task is hybrid by design. Bpod-core owns the discrete trial contract: quiescence, choice timeout,
accepted high/low branch, blank state, pump state, timeout gate, and exit. Python never blocks on
`Bpod.wait()` while the trial is active; it keeps the screen and wheel loop at 60 Hz and sends only
small SoftCode events. State-entry SoftCodes trigger the verified direct StreamDIO pump/force path,
and Bpod's Serial action directly gates audible timeout noise.

StreamDIO owns wheel, force ADC, licks, camera TTL history, and physical pump/effort output events.
pygame owns the screen. The recorder timestamps and stores all three domains, including the exact
bpod-core event rows.

All host deadlines and monotonic timestamps use `time.perf_counter()`. On this Windows rig,
`time.monotonic()` resolves only every 15.625 ms and must not be used for reward timing.
