# bpod-hf-dm

Head-fixed **decision-making with effort** (wheel + force cue), implemented on **[bpod-core](https://github.com/int-brain-lab/bpod-core)** with **StreamDIO** hardware from [MM_PyBpod2](https://github.com/deviandr/MM_PyBpod2).

Design reference: [IBLRIG](https://int-brain-lab.github.io/iblrig/) (IBL two-alternative choice structure) plus the **force-discount wheel** paradigm from `MM_PyBpod2/examples/screen_shape_force_discount_task.py`.

## Hard rule: bringup compatibility

All pin/channel/COM defaults match the bringup notebooks in `MM_PyBpod2/examples/`:

| Notebook | What this repo uses |
|----------|---------------------|
| `rotary_encoder_bringup.ipynb` | DI01 pulse, DI03 direction, AI13 force (pin 14) |
| `screen_bringup_shape_force_discount.ipynb` | CW force ch3/pin21, CCW ch5/pin19, pump ch1/pin23 |
| `camera_bringup_pin32_ttl.ipynb` | StreamDIO camera 30 Hz, 10 ms exposure |
| `speaker_bringup_pin27_whitenoise.ipynb` | Gate pin 27, serial bytes `[89,1]` / `[89,0]`, msg 210/211 |

Ports default to `AUTO` with the same discovery logic as the screen task.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Session supervisor (Python)                                 │
│  - pygame display (cue / feedback)                           │
│  - StreamDIO wheel + force stream                            │
│  - engagement (idle 60 s, 10 failed trials)                  │
│  - NPZ + JSON logging                                        │
└───────────────┬─────────────────────────────┬───────────────┘
                │ softcodes                  │ camera TTL
                ▼                            ▼
┌───────────────────────────┐   ┌──────────────────────────┐
│  bpod-core FSM (timed I/O)  │   │  StreamDIO (Teensy)       │
│  - force direction outputs  │   │  - rotary + analog force    │
│  - pump reward duration     │   │  - camera triggers        │
│  - white-noise gate         │   │  - session-long streaming │
│  - ITI ≥ 3 s                │   └──────────────────────────┘
└───────────────────────────┘
```

**Trial timeline (decision_making phase)**

1. **2 s black screen** (display)
2. **≥ 2 s quiescence** (no wheel pulses; host)
3. **FSM: apply force cue** (CW or CCW resistance via breakout)
4. **FSM: 15 s choice window** — host reads wheel, sends softcode on threshold
5. **FSM: matched/opposite pump** (0.08 s vs 0.04 s — high effort = longer reward)
6. **Feedback** (display)
7. **FSM: ITI ≥ 3 s**
8. On timeout: **FSM plays white noise** (speaker gate, 2 s)

## Install

```bash
cd /path/to/bpod-hf-dm
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e /path/to/bpod-core
pip install -e /path/to/MM_PyBpod2   # StreamDIO + optional sound module
pip install -e .
```

Requires Python ≥ 3.10, Windows rig (same as bringups), Bpod State Machine 2+.

## Run

```bash
# Stage 1 — centering only (reward when wheel near center)
hf-dm-run --config configs/centering_50.yaml

# Stage 2–4 — full DM (50 / 100 / 200 trials)
hf-dm-run --config configs/dm_50.yaml
hf-dm-run --config configs/dm_100.yaml
hf-dm-run --config configs/dm_200.yaml
```

Outputs: `data/sessions/<subject>_<phase>_<timestamp>.npz` and `.json`.

## Data logged

- Per-trial: force direction, choice, matched, reward duration, RT, pulses, force ADC stats, wall/mono timestamps, camera TTL list
- Events: `session_start`, `trial_start`, `quiescence_ok`, `pre_trial_black_*`, `trial_end`, `session_end`
- Session summary: end reason (`completed`, `idle_timeout`, `max_consecutive_failures`)

## Learning curricula

- **[docs/CURRICULUM_BPOD_CORE.md](docs/CURRICULUM_BPOD_CORE.md)** — become fluent in bpod-core using this repo
- **[docs/CURRICULUM_TRAINING.md](docs/CURRICULUM_TRAINING.md)** — mouse training stages (centering → 50 → 100 → 200)

## License

MIT
