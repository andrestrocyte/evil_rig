# Curriculum: mastering bpod-core through `bpod-hf-dm`

Goal: understand **typed FSM design**, **hardware execution**, and **host supervision** well enough to extend this task without reverting to ad-hoc `pybpodapi` scripts.

Estimated time: **2–3 weeks** part-time on the rig.

---

## Week 1 — bpod-core foundations

### Day 1–2: Mental model

Read in order:

1. [bpod-core state machine docs](https://int-brain-lab.github.io/bpod-core/state_machines/index.html)
2. `bpod-core/examples/state_machines/two_choice.py` — canonical discrete choice
3. [IBLRIG overview](https://int-brain-lab.github.io/iblrig/) — how IBL maps behavior to FSM + Bonsai (we replace Bonsai with pygame + StreamDIO)

**Exercise:** Draw on paper how `two_choice.py` maps `Port1_High` / `Port2_High` to states. Compare to our softcode bridge (`hf_dm/fsm/softcodes.py`).

### Day 3–4: API surface

Study `bpod-core/bpod_core/bpod/__init__.py`:

- `Bpod.run(fsm)` / `Bpod.wait()` — non-blocking trial execution
- `send_softcode(n)` — host-injected events
- `Module.load_serial_message` — breakout byte protocol
- `get_data()` — Polars trial log with Bpod timestamps

**Exercise:** Run `examples/minimal_examples/run_global_timer.py` on hardware; inspect `get_data()` columns.

### Day 5: Validation & compilation

Read `bpod_core/fsm.py` (`check`, `add_state`) and skim `bpod_core/bpod/_threads.py`:

- Why `Tup` is special
- How `>exit` and `>back` work
- What `EventThread` reconstructs from opcode stream

**Exercise:** Intentionally break `hf_dm/fsm/builders.py` (invalid action name); run `fsm.check()` and read the error.

---

## Week 2 — This repository

### Day 1: Hardware layer

| File | Learn |
|------|-------|
| `hf_dm/hardware/bringup_constants.py` | Single source of pin truth |
| `hf_dm/hardware/discovery.py` | AUTO COM logic from bringups |

**Exercise:** Run MM bringup notebooks; confirm constants match printed channel numbers.

### Day 2: StreamDIO adapter

| File | Learn |
|------|-------|
| `hf_dm/streamdio/client.py` | Wrapper over `StreamDIOProtocol` |
| `hf_dm/streamdio/wheel.py` | Pulse/direction → signed position |

**Exercise:** Log 10 s of events; verify `deg_per_pulse` matches hand-turn.

### Day 3: FSM templates

| File | Learn |
|------|-------|
| `hf_dm/fsm/relay.py` | Serial message IDs = bringup bytes |
| `hf_dm/fsm/builders.py` | One FSM per phase (choice, reward, ITI, punish) |

**Exercise:** `fsm.to_file("trial.pdf")` after `build_choice_epoch_fsm`; visualize graph.

### Day 4: Session supervisor

| File | Learn |
|------|-------|
| `hf_dm/session/controller.py` | Full trial timeline |
| `hf_dm/data/recorder.py` | NPZ schema |

**Exercise:** Dry-run with `n_trials: 2` in YAML; inspect `.json` event list.

### Day 5: Display & audio

| File | Learn |
|------|-------|
| `hf_dm/display/pygame_display.py` | Stimulus only on host |
| `hf_dm/audio/white_noise.py` | PC noise + FSM gate |

---

## Week 3 — Integration & IBL comparison

### Compare paradigms

| IBL / IBLRIG | bpod-hf-dm |
|--------------|------------|
| Port left/right choice | Wheel position threshold (continuous) |
| Fixed reward time | Matched vs opposite pump (effort discount) |
| Bonsai stimuli | pygame + force cue on breakout |
| Bpod FSM only | Bpod FSM + StreamDIO parallel |

### Extension projects (ordered)

1. Add **biased block structure** (70/30 left/right) via trial generator in config
2. Log **Bpod `get_data()`** alongside NPZ for alignment with core timestamps
3. Add **condition** in FSM for hardware line triggering (advanced)
4. Port **stimulus side** to Bonsai while keeping FSM identical

### Mastery checklist

- [ ] Explain why wheel quiescence cannot live purely inside Bpod FSM without rotary module events
- [ ] Add a new FSM state (e.g. longer timeout noise) without touching StreamDIO code
- [ ] Change reward asymmetry in YAML only
- [ ] Diagnose COM port conflict using discovery helpers
- [ ] Align one session video frame to `camera_ttl_timestamps_us` in NPZ

---

## Key design decision (documented)

**Continuous wheel → discrete FSM** via softcodes is deliberate: StreamDIO runs on Teensy USB parallel to Bpod, matching your bringup stack. All **timed outputs** (pump, force, speaker, ITI) stay in bpod-core FSM for microsecond-scheduled hardware; all **position integration** stays on the host with logged timestamps — standard in IBL-style rigs that combine FSM + external acquisition.
