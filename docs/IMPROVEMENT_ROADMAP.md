# Improvement roadmap

This roadmap tracks non-breaking improvements derived from comparison with IBLRIG and the
MM_PyBpod2 force-discount reference task.

## Guardrails

- Existing configuration defaults preserve the established task behavior.
- New behavior is opt-in or restricted to setup, monitoring, validation, or export paths.
- The production 60 Hz choice loop and production host-supervised FSM behavior remain unchanged.
- Every completed item is covered by hardware-free regression tests.

## Status

| Item | Status | Completion |
| --- | --- | --- |
| 1. Opt-in side counterbalancing | Complete | `827129f` |
| 2. Session QC evaluator | Complete | `827129f` |
| 3. Start-time Git provenance | Complete | `827129f` |
| 4. Reward-volume annotation | Complete | `827129f` |
| 5. Valve calibration workflow | Complete | `827129f` |
| 6. Subject weight and water log | Complete | `827129f` |
| 7. Online psychometric monitor panel | Complete | `bd1213d` |
| 8. Fitted cross-clock alignment | Complete | `bd1213d` |
| 9. Extract production trial runners | Complete | `bd1213d` |
| 10. Quarantine legacy/reference paths | Complete for safe scope | `bd1213d` |
| 11. Typed validation and stricter CI | Complete | `827129f` |
| 12. ALF export adapter | Deferred | Out of scope until IBL integration is requested |
| 13. Alyx auto-registration | Deferred | Out of scope until IBL integration is requested |

## Completed remaining items

### 7. Online psychometric monitor panel

- Added session-running matched/responded rate, responded-trial mean RT, and signed side-bias
  readouts to the existing behavior panel.
- Statistics consume only existing `trial_result` telemetry and replace duplicate trial updates by
  `trial_index`.
- The monitor remains opt-in and the disabled-monitor path is unchanged.
- Covered by state, duplicate-update, disabled-path, and headless construction/render tests in
  `tests/test_operator_monitor.py`.

### 8. Fitted cross-clock alignment

- Added export-time affine fits for `perf_counter -> wall`, `StreamDIO -> perf_counter`, and
  `StreamDIO -> wall`.
- Each pair records slope, intercept, anchor count, maximum absolute residual, RMS residual, and a
  graceful `insufficient_anchors` status.
- Fits use the already-recorded Bpod state-marker and Frame2TTL anchors; acquisition is untouched.
- Covered by exact synthetic-fit, insufficient-anchor, and manifest tests in
  `tests/test_session_export.py`.

### 9. Extract production trial runners

- Moved decision-epoch, decision-trial, and centering-trial bodies into classes under
  `src/hf_dm/session/runners/`.
- `HFDecisionSession` keeps the same private method signatures and delegates directly to the runner
  classes.
- Extracted method bodies were compared byte-for-byte before delegation was installed.
- `tests/test_simulated_session.py` fixes the expected TrialRecord projection and ordered semantic
  event stream for matched, opposite, and timeout outcomes.

### 10. Quarantine reference and legacy paths

- Moved all non-live FSM templates into `src/hf_dm/fsm/reference/`, whose module documentation marks
  them as validation-only.
- Compatibility re-exports remain in `hf_dm.fsm.builders`; the production
  `build_host_supervised_decision_fsm` body is unchanged.
- `_run_dm_trial_legacy` and `_run_live_choice_epoch_legacy` were not deleted. A repository search
  found that `session/hardware_self_test.py` still overrides and invokes the live-choice legacy
  method, so deletion would violate the safety gate.

## Validation for `bd1213d`

- `161 passed` under headless SDL.
- Ruff passed for `src` and `tests`.
- The CI mypy gate passed for `config.py`, `data/recorder.py`, and `data/session_export.py`.
- Every tracked `configs/*.yaml` file is unchanged.
- ALF export and Alyx registration remain intentionally out of scope.
