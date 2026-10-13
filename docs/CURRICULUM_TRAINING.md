# Curriculum: training mice on the HF effort decision task

Aligned with staged IBL-style training and the force-discount logic: the star is always the
high-effort/high-reward option and the circle is always the low-effort/low-reward option.

## Stage 0 - Habituation

- Head-fix, water schedule, and rig entry
- Let the mouse explore the wheel without contingencies
- Confirm pump delivery with `MM_PyBpod2/examples/pump_bringup.ipynb`

## Stage 1 - Centering for reward (`centering_50.yaml`)

Trials: 50

Each trial starts the cue at a random lateral target. The arrow points toward the center target and
the screen follows the wheel continuously. Water is delivered immediately when the cue enters
`centering_threshold_deg`; a timeout produces the configured white-noise punishment.

Success criteria:

- At least 60% centered trials with reward
- No signs of stress
- Session not ended by idle timeout

If failing: widen `centering_threshold_deg` to 12 and consider a slightly longer `matched_pump_s`.

## Stage 2 - Full task, short (`dm_50.yaml`)

Trials: 50

Full contingencies:

- Star and circle start on opposite lateral sides; their side is randomized every trial
- Star and circle have exactly equal lit-pixel counts
- Turning right moves only the left option toward center; turning left moves only the right option
- Star choice: high effort and `0.08 s` pump by default
- Circle choice: low effort and `0.04 s` pump by default
- Bpod requires 3 s quiescence and owns the 20 s choice timeout
- Timeout: Bpod controls the StreamDIO pin-26 gate to pass continuous soundcard white noise for `2 s` by default

Success criteria:

- At least 50% responded trials
- At least 55% matched on responded trials
- Mean response time under 12 seconds

## Stage 3 - Standard volume (`dm_100.yaml`)

Trials: 100

Track in NPZ/JSON:

- `matched` rate over blocks of 20 trials
- `force_mean` during the cue epoch
- `response_time_s` drift as a fatigue marker
- `choice_loop_max_frame_s` and `choice_loop_stall_count` for display/task stalls

Intervention triggers:

- Matched rate drops by more than 15% across blocks: refresh centering
- 10 consecutive timeouts: session auto-ends by engagement rule

## Stage 4 - Full session (`dm_200.yaml`)

Trials: 200

Use for stable psychometrics and video alignment.

```python
import numpy as np

d = np.load("data/sessions/....npz", allow_pickle=True)
trials = d["trials"]
matched = trials["matched"][trials["responded"]]
print(matched.mean(), trials["response_time_s"].mean())
```

## Daily rig checklist

1. Record subject weight and the licensed water target:
   `hf-dm log-weight <subject> --weight <grams> --water-target <ml>`
2. Run no-trial hardware preflight:
   `python -m hf_dm.cli --config configs/dm_50.yaml --preflight --preflight-duration 2`
3. Confirm the typed preflight results have no FAIL entries, zero overruns, and camera TTLs.
4. After code/hardware changes, run the real-output diagnostic with the rig clear:
   `python -m hf_dm.cli --config configs/dm_50.yaml --hardware-self-test`
5. Confirm no old Python, PyBpod, notebook, or serial-monitor process owns COM7/COM8.
6. Run the stage config:
   `python -m hf_dm.cli --config configs/dm_50.yaml`
7. Verify `.npz`/`.json` in `data/sessions/` and video/timestamps in `data/videos/`.
8. Run `hf-dm session-qc <session>` and review every WARN/FAIL result.
9. Check calibrated water totals, licks, raw force/wheel events, camera TTLs, and `session_end`.

## Suggested calendar

| Week | Sessions | Config |
|------|----------|--------|
| 1 | 3-5 | `centering_50.yaml` |
| 2 | 4-6 | `dm_50.yaml` |
| 3-4 | 6-10 | `dm_100.yaml` |
| 5+ | data collection | `dm_200.yaml` |

Adjust by lab welfare guidelines and individual mouse progress.
