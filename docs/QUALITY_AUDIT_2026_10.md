# Rig audit, 12 October 2026

The initial assessment was about 7/10 overall: strong recording and failure
handling, but cumbersome daily operation and several timing/lifecycle bugs.
The Windows rig now has an independent installation and a desktop launcher with
session selection, on-demand diagnostics, cooperative Stop and saved results.
This audit does not establish that every hardware component is calibrated or
that the system is bug-free.

Local verification: full suite 534 passed. Ruff and type checks for config,
recorder, session export, raw I/O, journal I/O and array I/O passed. CI checks
Windows and Linux; exact-commit results are linked in PR #5.

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
- Skip the optical reader's expensive dataframe join when the Bpod event buffer
  has not changed. New edges, final events and replaced FSM buffers remain read.
- Keep exact force percentiles in an ADC-code histogram; the 60 Hz display reads
  a cheap snapshot, and percentile sorting runs outside the acquisition lock.
- Bound duplicate raw-event histories in the recorder and driver. WAL-only
  bounded mode retains all events in final exports and after crash recovery.
  Diagnostic counts and camera histories survive the rolling cache wrapping.
- Stop cleanly on low disk space. Reserve checks run before connecting and every
  two seconds, with the same data-finalization path as other acquisition errors.
- Compare saved images with actual StreamDIO triggers. Recorder lifetime includes
  startup and shutdown waiting, so multiplying it by target fps falsely reported
  missing triggers. Camera count mismatches now fail launcher result checks.
- Reuse commissioning across subject, trial-count, seed, profile-label and output
  path changes, including auto-resolved monitor telemetry paths. Keep reward caps
  checked per run; bind the actual installed driver source hash. Exact commit
  matching no longer requires a redundant Git tag. Certificates created before
  this update need regeneration because code and identity hashing changed.
- Stream raw NPZ serialization, CSV exports and NPZ validation in blocks rather
  than constructing several full Python copies. Preserve every value, column and
  dtype. Normal finalization rejects a missing, truncated or CRC-invalid WAL and
  retains recovery evidence instead of substituting a partial in-memory cache.
- Bound the optional chunk queue to two blocks. Backlog falls back to the complete
  WAL; a writer that fails to stop prevents completion/manifest publication.
- Separate the 30-second hardware-stop deadline from the five-minute export
  deadline, using a per-launch token and a verified-cleanup signal. Data validation
  also has a five-minute process timeout. Stop no longer kills a healthy large
  export merely because its hardware shutdown grace period expired.
- Verify required camera enums, booleans and explicit ROI by readback. Apply ROI
  in an order that permits shrinking/moving and expanding; restore the original
  crop after rejected writes. Record inherited and actual settings at startup and
  shutdown instead of reporting requested values as if they were applied.
- Give PySpin the same direct Mono8 PGM saving and preview-before-save behavior as
  the C backend. Retain post-save CSV clock semantics and add separate host receipt
  clocks. A PySpin save timeout now fails capture instead of being confused with
  an ordinary acquisition wait; a stopped camera wait no longer invents an error.
- Stream crash-recovery WAL/chunk reads and compare shared prefixes sequentially.
  Retain original chunks, append only the missing suffix in blocks, and keep the
  WAL/marker until the recovered bundle validates. Stop on divergent sources,
  corrupt chunks, missing indices or interior WAL CRC damage. Preserve damaged
  final WAL bytes in a manifest-hashed backup. Close readers on early failure and
  sort chunk numbers numerically beyond 999.
- Preserve camera resource ownership through startup and cleanup failures. Stop
  capture before releasing SDK objects; release failed images before their camera,
  clear/destroy camera lists and retain failed handles for a later cleanup attempt.
  Preserve the original startup error and historical cleanup errors after a retry.
  Close timestamp files even when flushing fails. Reject recorder-instance reuse
  before it can overwrite data. Camera cleanup errors now reach the session's
  cleanup interlock instead of being ignored in the recording summary.
- Stream session-journal scanning, repair and replay without full text/row copies.
  Retain a complete last row when only its newline is missing. Repair a malformed
  unterminated final record, including a torn UTF-8 character, but refuse malformed
  terminated lines, invalid schemas and sequence errors. Preserve exact original
  bytes, including line endings, before atomic repair. Distinct retry inputs get
  distinct backups; recovery and offline export verify recorded backup checksums.
  Moved sessions use their local evidence. Refuse completed sessions before any
  repair, and detect source-file changes during scanning or replay.
- Stream structured semantic events and JSON-array NPZ members in byte-bounded
  blocks, preserving their existing dtypes and values. Detect changed row counts
  or strings that grow after sizing instead of truncating them. Stream the JSON
  sidecar text and reuse its event dictionaries for cold exports. Validate full
  semantic-array contents and counts in blocks, including JSON decoding and ZIP
  checksums; reject truncated/trailing data and object/pickle arrays.
- Checkpoint a validated recovery before deleting its WAL or incomplete marker.
  If either deletion fails, retry verifies the completed bundle and finishes only
  cleanup, preserving the NPZ and journal bytes. Refuse changed or missing completed
  artifacts. Accept copied checkpoints with local evidence, and preserve valid
  older bundles whose WAL was already removed; refuse invalid older bundles with
  no WAL instead of rebuilding from incomplete sources.
- Declare an empty raw WAL at new-session startup and persist an independent
  raw-event count after periodic flushes, trial completion and final close.
  Serialize flush/checkpoint pairs without holding the incoming-sample lock during
  disk I/O. Before any recovery rewrite, compare WAL/chunk data against the latest
  count and refuse missing or shortened sources. Complete chunks can supply a
  missing WAL. Keep compatibility with older journals while reporting absent
  independent count evidence. Offline validation checks the same lower bound.
  Record unsupported WAL fsync, propagate checkpoint failures, and retain journal
  sequence progression after a completed line's fsync fails.
- Expose the existing calibration/certification window through **Rig setup** in
  the launcher, using the selected profile, sensor and data directory. Keep session
  actions disabled until setup closes, save setup logs/results, and distinguish a
  closed dialog from a passing certificate. Group cascading unreadable-certificate
  errors into one actionable setup item while retaining all checks in a full-report
  view. Cancel requests cooperative stop and keeps the setup window visible until
  the active certification process exits; later diagnostics are skipped. Prevent
  valve calibration from overlapping certification.

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
| Two-minute acquisition | 126,494 driver events = 126,494 exported rows; 3,609 triggers = 3,609 frames; recent raw cache 50,718 rows; zero overruns or lost frames | High-rate working history is bounded; not an hours-long soak |
| Memory / preview | Aggregate acquisition+GUI RSS about 264 MiB after minute one; preview age 75 ms median / 91 ms p95 | Transient camera backlog reached 186 ms then recovered; final growth -0.9 ms |
| Idle optical poll optimization | Subsequent 30-second run: 915 triggers = 915 frames; preview age 74 / 89 ms median/p95; aggregate CPU median 336% to 214% (100% = one CPU core) | Different run durations; host measurements, not calibrated exposure latency |
| Force snapshot benchmark | 60,000 samples: display snapshot mean 2.42 ms to 0.0011 ms; exact p95 tested against NumPy | Synthetic ADC values; physical force calibration still pending |
| Low-space fault injection | Stop/finalize in 1.07 s, 17 triggers = 17 frames, exports valid, clean cleanup/interlock | Raised reserve threshold without filling the disk; not a physical disk-full or power-loss test |
| Million-row export | Finalization peak RSS 725.7 to 69.6 MiB; 17.59 to 16.10 s. Validation 611.8 to 62.1 MiB; 22.92 to 9.99 s | Synthetic records, separate runs on this PC; not an animal session |
| Export equivalence | All 1,000,000 NPZ rows and dtype equal; raw/force/camera/Frame2TTL CSVs byte-identical to prior exporter | Metadata/timing fields intentionally vary by benchmark run |
| 3.6-million-row export | Finalize 58.04 s / 139.5 MiB peak RSS; validation 31.96 s / 103.8 MiB; complete bundle validates | Synthetic 1 kHz record volume for one hour; not a live one-hour hardware soak |
| Stop with block exports | 225 triggers = 225 images; clean shutdown, valid data and clear interlock 2.41 s after Stop | Short live optical observation; no physical beam gestures |
| Camera settings | Actual 1440 × 1080 Mono8; 640 × 480 crop at (32, 24), full-frame restoration and deliberately invalid width all behave correctly; clean SDK cleanup | Live C backend; alternate PySpin adapters exercised with fakes |
| Camera after readback changes | 914 triggers = 914 images; preview 77 / 87 ms median/p95; receipt-to-save 2.92 / 5.89 ms; all 25 artifacts validate | Thirty-second input observation; no incomplete/save errors, clean cleanup; not calibrated exposure-to-display timing |
| Compression experiment | Lossless level-1 compression retains about 24% of original image bytes; minimum tested PNG filtering takes about 22 ms/frame | Eight saved frames; insufficient margin to change the live default without further throughput testing |
| Million-row crash recovery | Peak RSS 353.6 to 68.2 MiB; recovery plus validation 25.13 to 27.23 s; complete bundle validates | Synthetic source records; more sequential verification trades some time for bounded raw-data memory |
| Recovery equivalence / failures | All one million raw NPZ rows and dtype equal; four raw CSVs byte-identical. Injected disk failure and failed validation retry successfully; original prefix chunks unchanged | Process/file fault injection, not storage hardware failure or sudden power loss |
| 3.6-million-row crash recovery | Recovery and validation complete in 97.44 s at 139.3 MiB peak RSS; all 3.6 million records retained, bundle valid | Synthetic one-hour volume at 1 kHz; semantic camera timeline still grows with record duration |
| Camera startup recovery | Missing serial and rejected ROI both release all native handles; subsequent recording saves 466 images for 466 triggers, all 25 artifacts validate, cleanup succeeds | Live C backend; PySpin startup and failed-release paths use unit fakes |
| Preview after cleanup changes | Host frame age 74 / 89 ms median/p95; receipt-to-save 2.31 / 3.70 ms; maximum backlog growth 0.92 ms, ending -0.50 ms relative to start | Fifteen-second GUI observation; excludes calibrated exposure/display latency |
| Journal boundary faults | Missing newline now yields a valid recovery; torn UTF-8 tail recovers; newline-terminated malformed JSON is refused. Missing or altered prior backups stop retries; altered backups cannot become a valid offline re-export | File/process fault injection; not physical storage failure |
| Journal scan / replay | 50,006 rows produce identical hashes; parser peak RSS 181.7 to 35.5 MiB, scan/consume time 1.18 / 1.13 s | Synthetic journal; this isolates parsing, not total recovery memory |
| Semantic-event recovery | All 50,000 events and payload values retained exactly; both full recovered bundles validate; total recovery/validation 6.67 / 6.48 s, peak RSS 351.7 / 348.7 MiB | Final semantic-event export dominates peak memory; no large overall memory reduction claimed |
| Semantic-array and JSON streaming | Same 50,000-event workload now peaks at 141.7 MiB versus 348.7 MiB; full recovery/validation 8.16 versus 6.48 s; every sample value matches the original hash | Synthetic workload; stronger validation and two-pass string sizing add work. Semantic models, timeline analysis and JSON validation still grow with session size |
| Live recording with semantic streaming | 465 camera triggers = 465 images; 17,302 raw records and 191 semantic events retained; all 25 artifacts validate; no incomplete/save errors, clean cleanup | Fifteen-second observation; one ordinary acquisition wait timeout, with no missing triggered frame |
| Interrupted recovery cleanup | Before: marker deletion failure followed by retry changed one recovered raw row to zero and still validated. After: retry retains the row and validates existing artifacts without rewriting | Synthetic temporary recording; regression tests cover both deletions, both layouts, checkpoint write failure, altered artifacts, relative paths, copied evidence and older bundles |
| Missing raw source / whole-record truncation | Before: 20 recorded rows became 0 or 10 and recovery reported valid. After: both attempts fail before journal changes and retain the incomplete marker | Synthetic temporary recordings. Checkpoints prove a lower bound; uncheckpointed tail loss remains unknowable from the count alone |
| Live raw checkpoints | 31 checkpoints; final count equals all 31,943 raw events. 915 camera triggers = 915 images, all 25 artifacts valid; no incomplete frames, acquisition timeouts, save errors or cleanup failure | Thirty-second input observation; physical gestures and output calibration not exercised |
| Preview with raw checkpoints | Host frame age 77 / 88 ms median/p95; receipt-to-save 2.24 / 3.72 ms; maximum backlog growth 0.51 ms, ending -0.81 ms | Host-relative measurements, not calibrated exposure-to-screen latency |
| Setup workflow | Real launcher opens the setup window with selected optical/StreamDIO settings and E: data path; task buttons disable while it is open and re-enable after Cancel. Missing-certificate consequences group into one action; full report remains available | UI inspection only; no hardware action or certificate creation |
| Setup cancellation | Actual Tk window remains visible while a simulated child observes Stop and exits; Continue stays disabled, cancellation is logged, and no certificate is created | Simulated child, not a physical driver hang; unit test verifies later diagnostics do not launch after Stop |

Local evidence (recordings are intentionally excluded from Git):

- `data/runtime/audit/camera_latency_summary.json`
- `data/runtime/audit/recovery_marker_failure_before.json` and `recovery_marker_failure_after.json` (interrupted cleanup retry)
- `data/runtime/audit/missing_raw_before.json` and `missing_raw_after.json` (missing and shortened raw sources)
- `data/runtime/audit/raw_checkpoint_live_result.json` (live checkpoint counts and camera timing)
- `data/runtime/audit/setup_gui_result.json`, `launcher_completion.png`, `setup_completion.png` (real Windows UI)
- `data/runtime/audit/setup_cancel_result.json` (cooperative cancellation through the Tk workflow)
- `data/runtime/audit/production_timing/result.json`
- `data/runtime/operator/runs/20261012_212437_904677/result.json` (camera)
- `data/runtime/operator/runs/20261012_212901_829814/result.json` (complete rig check)
- `data/runtime/operator/runs/20261012_213348_212361/result.json` (clean Stop)
- `data/runtime/audit/stop_recovery.json` (verified recovery after the failed Stop)
- `data/runtime/audit/sustained_comparison.json` (complete stream, camera and resource counts)
- `data/runtime/operator/runs/20261012_215523_338716/result.json` (two-minute acquisition)
- `data/runtime/operator/runs/20261012_220416_893759/result.json` (idle-poll optimization)
- `data/runtime/audit/storage_guard_result.json` (injected low-space threshold)
- `data/runtime/audit/wheel_snapshot_before.json` and `wheel_snapshot_after.json`
- `data/runtime/audit/export_before/benchmark.json` and `export_optimized/benchmark.json`
- `data/runtime/audit/export_equivalence.json` (complete raw values and CSV comparison)
- `data/runtime/audit/export_one_hour/benchmark.json` (3.6 million synthetic records)
- `data/runtime/operator/runs/20261012_223247_869969/stop_result.json` (live block-export Stop)
- `data/runtime/audit/camera_settings_readback.json` (live readback and ROI rollback)
- `data/runtime/audit/camera_receipt_result.json` (live counts, GUI age and receipt-to-save timing)
- `data/runtime/audit/png_compression.json` (read-only compression benchmark)
- `data/runtime/audit/recovery_before/benchmark.json` and `recovery_after/benchmark.json`
- `data/runtime/audit/recovery_equivalence.json` (one-million-row exact comparison)
- `data/runtime/audit/recovery_one_hour/benchmark.json` (3.6-million-row recovery)
- `data/runtime/audit/camera_start_failure_result.json` (startup failures, resource release and subsequent live recording)
- `data/runtime/audit/journal_boundaries_before.json` and `journal_boundaries_after.json`
- `data/runtime/audit/journal_scan_before.json` and `journal_scan_after.json`
- `data/runtime/audit/journal_recovery_before/benchmark.json` and `journal_recovery_after/benchmark.json`
- `data/runtime/audit/journal_recovery_streamed/benchmark.json` (NPZ streaming alone: 195.2 MiB)
- `data/runtime/audit/journal_recovery_streamed_json/benchmark.json` (NPZ plus JSON streaming: 141.7 MiB)
- `data/runtime/audit/event_export_result.json` (live recording and stronger array validation)

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

Normal finalization, validation and crash recovery stream high-rate raw data in blocks; the
lower-rate semantic timeline and heartbeat analysis still grow with session size.
Journal parsing/repair and semantic NPZ/text writing now stream. Reconstructed
semantic events, timeline analysis and JSON validation still grow in memory with
session size. Full-resolution PGM video
uses about 2.8 GB/min on this camera. A stalled optional writer is fault-injected in
tests; a physical storage-device hang has not been tested.
Windows reported unavailable directory fsync support: file flushes, atomic writes
and process-crash recovery are tested, but sudden-power-loss durability is not
established. These limits prevent a defensible "10/10, bug-free" claim.

Camera settings readback and ROI rollback now pass on the connected C backend.
The alternate PySpin path has unit coverage for setting failures, pixel fidelity,
preview-before-save order, timestamps and image release, but has not had live SDK
validation on this Python 3.12 installation (vendor bindings are unavailable).
