# Curriculum: training mice on the HF effort decision task

Aligned with IBL-style **staged training** ([IBLRIG](https://int-brain-lab.github.io/iblrig/)) and your force-discount logic (longer pump when choice matches effort cue).

---

## Stage 0 — Habituation (outside this repo)

- Head-fix, water restriction schedule, rig entry
- Let mouse explore wheel without contingencies
- Confirm pump delivers reliably (use `MM_PyBpod2/examples/pump_bringup.ipynb`)

---

## Stage 1 — Centering for reward (`centering_50.yaml`)

**Config:** `configs/centering_50.yaml`  
**Trials:** 50  
**Criterion:** mouse receives water when wheel position within **±8°** of center after 2 s quiescence.

**Success criteria (per session)**

- ≥ 60% centered trials with reward
- No signs of stress (grooming freeze, escape attempts)
- Engagement: session not ended by idle timeout

**If failing:** widen `centering_threshold_deg` to 12; increase `matched_pump_s` slightly.

---

## Stage 2 — Full task, short (`dm_50.yaml`)

**Trials:** 50  
**Full contingencies:**

- Random CW/CCW force cue (triangle orientation)
- Turn wheel to either side (±35°)
- **Matched direction → 0.08 s pump** (high effort / correct side)
- **Opposite → 0.04 s pump**
- Timeout → white noise (2 s gated)

**Success criteria**

- ≥ 50% responded trials
- ≥ 55% matched on responded trials (slightly above chance on effort task)
- Mean RT < 12 s

---

## Stage 3 — Standard volume (`dm_100.yaml`)

**Trials:** 100  

Track in NPZ:

- `matched` rate over blocks of 20 trials
- `force_mean` during cue epoch (effort proxy)
- `response_time_s` drift (fatigue)

**Intervention triggers**

- Matched rate drops > 15% across blocks → refresh centering session
- 10 consecutive timeouts → session auto-ends (engagement rule)

---

## Stage 4 — Full session (`dm_200.yaml`)

**Trials:** 200  

Use for stable psychometrics and video alignment.

**Analysis suggestions**

```python
import numpy as np
d = np.load("data/sessions/....npz", allow_pickle=True)
trials = d["trials"]
matched = trials["matched"][trials["responded"]]
print(matched.mean(), trials["response_time_s"].mean())
```

---

## Engagement rules (implemented)

| Rule | Parameter | Default |
|------|-----------|---------|
| No wheel movement | `engagement.idle_timeout_s` | 60 s |
| Consecutive timeouts | `engagement.max_consecutive_failures` | 10 |

Session `end_reason` in JSON records which rule fired.

---

## Daily rig checklist

1. Run rotary + camera bringup notebooks if any hardware change
2. `hf-dm-run --config configs/dm_50.yaml` (or stage-appropriate)
3. Verify `.npz` + `.json` in `data/sessions/`
4. Start external video recorder synced to camera TTL (see camera bringup)
5. Note `session_end` reason before unhead-fixing

---

## Suggested calendar

| Week | Sessions | Config |
|------|----------|--------|
| 1 | 3–5 | `centering_50.yaml` |
| 2 | 4–6 | `dm_50.yaml` |
| 3–4 | 6–10 | `dm_100.yaml` |
| 5+ | data collection | `dm_200.yaml` |

Adjust by lab welfare guidelines and individual mouse progress.
