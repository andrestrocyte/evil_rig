# Rig audit, 12 October 2026

The initial assessment was about 7/10 overall: strong recording and failure
handling, but cumbersome daily operation and several timing/lifecycle bugs.
The Windows rig now has an independent installation and a desktop launcher with
session selection, on-demand diagnostics, cooperative Stop and saved results.
This audit does not establish that every hardware component is calibrated or
that the system is bug-free.

Local verification: full suite 343 passed; after the final camera accounting
correction, all 22 camera tests passed. Ruff and stable-module mypy also passed.

## Repairs

- Direct Mono8 PGM saving removes the slow SDK encoder from acquisition.
  The preview keeps current frames separately from lossless recording. Its target
  is now 15 fps, with cadence preserved under small scheduling jitter.
- Move optical polling and StreamDIO receipt waits out of the display thread.
  Preserve original Bpod edge timestamps and label StreamDIO receipt timestamps.
- Use precise frame pacing without integer-millisecond rounding or busy waiting.
  A logged initial bright/dark square establishes the detector baseline before
  barcodes. Decode pulse widths from StreamDIO timestamps, reject malformed pulses,
  and fail the launcher check when the recorded barcode result fails.
- Optical ITI/preflight state machines now respond to Stop. Driver/thread waits
  have deadlines; cleanup failure cannot be reported as a successful stop.
- Self-test counts/modality and capacitive-power expectations match the selected
  sensor. Optical certification requires both edges on the selected clock rather
  than requiring a capacitive touch test.
- Read the actual speaker pin before recording hardware identity. Suppress
  spurious heartbeat warnings and per-sample driver console logging, preserve
  partially opened resources for cleanup, and expose audio cleanup failures.
- Record the installed driver's pinned VCS revision rather than attributing a
  wheel inside `.venv` to the parent task repository's Git commit.
- Keep SDK acquisition timeouts separate from lost-frame counts. Waiting before
  or after camera triggers can time out without losing an image. The production
  timing check recorded one such timeout but matched all 804 triggers to images.

## Evidence on this PC

| Check | Observed result | Limit |
|---|---|---|
| Camera + GUI, 60 seconds | 1,798 saved frames = 1,798 recorded triggers; no incomplete frames, save errors or StreamDIO overruns | One minute, not an hours-long soak |
| Camera timing | 29.94 Hz by camera timestamps; maximum backlog growth 20.83 ms, ending -1.16 ms relative to start | Clock-relative growth, not absolute exposure latency |
| Preview | 901 previews; GUI frame age median 74 ms, p95 92 ms; no image-render failures | Host receipt/render timing; excludes calibrated display scanout |
| Output/camera self-test | Three real FSM trials pass, including both force outputs, reward timing and speaker gate; saved artifacts validate | Choices injected; physical force, water volume and SPL not measured |
| Frame2TTL | Six flashes yield six edges; all six representative barcodes match | Short diagnostic on this display |
| Production display timing | Three live FSM trials retain 3 s quiescence, normal barcodes, 60 Hz display and 30 Hz camera; all trial barcodes complete and match, 804 triggers match 804 images, output and data validation pass | Synthetic wheel choices; choice timeout shortened to 2 s |
| Stop during optical observation | Aborts and saves in 2.75 s; no cleanup error; interlock clear | One deliberately interrupted live input check |
| Windows installation | Own Python 3.12 environment, pinned driver, desktop shortcut; GUI opened and inspected | Spinnaker SDK already installed on this PC |
| Reference checkout | All 95 baseline source files unchanged | Compared against start-of-audit SHA-256 manifest |

Local evidence (recordings are intentionally excluded from Git):

- `data/runtime/audit/camera_latency_summary.json`
- `data/runtime/audit/production_timing/result.json`
- `data/runtime/operator/runs/20261012_212437_904677/result.json` (camera)
- `data/runtime/operator/runs/20261012_212901_829814/result.json` (complete rig check)
- `data/runtime/operator/runs/20261012_213348_212361/result.json` (clean Stop)
- `data/runtime/audit/stop_recovery.json` (verified recovery after the failed Stop)

The earlier rig check at `20261012_211459_030909` incorrectly reported overall
success despite only four of six barcode matches. The earlier Stop check at
`20261012_213045_158102` saved data but failed Bpod cleanup. Original logs remain
as failure evidence; use the later checks above for the repaired behavior.

## Remaining physical work

The unattended tests contained no physical optical-beam, capacitive-touch or
wheel gestures. The analog force input produced a near-zero unloaded trace;
its response to load is unverified. A synthetic optical marker already traversed
Bpod to StreamDIO, but physical beam-edge relay timing still needs a block/unblock
check. Software cannot establish calibrated force, reward volume or sound level
without measurements.

Use the launcher's 20-second input view when someone is at the rig. Retain the
animal-use commissioning/calibration gate until physical evidence is available.
This does not mean repeating the full diagnostics before each bench session.
