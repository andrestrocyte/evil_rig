# Animal-ready release and rig certification

This document is the release checklist for using `hf_dm` with an animal. Passing software tests is
necessary but not sufficient. The startup gate deliberately rejects missing, stale, mismatched, or
warning-only evidence before Bpod or StreamDIO is opened.

## Release model

An animal-use certificate is bound to one exact combination of:

- clean Git commit and exact Git tag;
- `requirements.lock` and installed package versions;
- canonical session configuration hash;
- reward-calibration file hash and fit quality;
- hardware failsafe revision;
- Bpod, StreamDIO, and camera identities/firmware.

Changing any bound input invalidates the old certificate. Bench diagnostics and simulation use an
explicit non-animal run mode and do not weaken the animal gate.

## Before certification

1. Work from a clean release checkout and create a reproducible release tag. Do not tag a release
   `animal-ready` until physical acceptance is complete.
2. Put session data on a local volume that supports durable file flushes. Network and virtual
   filesystems that reject `fsync`/`FlushFileBuffers` cannot pass storage certification.
3. Run `hf-dm calibrate-valve` into a collection vessel. The calibration must cover both configured
   reward durations, meet the configured R-squared and maximum-residual limits, and be current.
4. Set `welfare.max_cumulative_reward_ul`, log a current subject weight/water record, and record the
   approved protocol, trial limit, ITI, engagement stops, and cumulative reward cap.
5. Set `animal_use.hardware_failsafe_revision` to the installed independent hardware watchdog or
   power-cut revision. A Python cleanup routine is not an independent hardware failsafe.
6. Copy `physical_certification_evidence.example.json`, replace every placeholder with measured,
   traceable evidence, and sign all four approvals.

## Unified no-animal certification

The command below actuates the pump, force outputs, speaker gate, cap-touch power, and camera trigger.
Run it only with no animal present, a collection vessel fitted, the force mechanism clear, safe
speaker exposure, and immediate emergency power control available.

```powershell
hf-dm certify-rig --config configs/dm_50.yaml `
  --physical-evidence certifications/rig1-physical-evidence.json `
  --confirm-no-animal --confirm-collection-vessel --confirm-force-clear `
  --confirm-speaker-safe --confirm-emergency-stop
```

The workflow executes the no-trial preflight, production-FSM hardware self-test, Frame2TTL square and
barcode diagnostics, Bpod opcode transport, and cap-touch diagnostic. It checks exact camera
TTL/frame equality, zero StreamDIO overruns, state ownership, and the expected physical output
sequences. Machine failures cannot be overridden by human evidence.

The signed evidence must establish:

- valve timing at 40 ms and 80 ms with an oscilloscope or equivalent instrument;
- speaker SPL below the protocol maximum;
- CW/CCW force direction and magnitude below the approved limit;
- wheel sign, radius, and gain matching the config;
- an independent hardware failsafe under stuck-high, USB unplug, frozen host, Bpod reset,
  StreamDIO reset, and emergency-stop cases;
- all-off de-energization times for the same failure cases;
- a soak at least twice the configured worst-case session duration, with zero overruns/drops and
  passing session validation/QC;
- approved welfare limits matching the config.

Every evidence item needs an operator, measurement time, instrument or artifact identifier where
applicable, and the SHA-256 of the immutable raw artifact. The JSON template is structural guidance;
its example values are not certification evidence.

## Starting an animal session

After `certify-rig` writes an all-PASS certificate at the configured path, run:

```powershell
python -m hf_dm.cli --config configs/dm_50.yaml --confirm-animal-use
```

The confirmation applies to one process invocation. The gate is evaluated once before hardware is
opened and again against connected device identities before streaming or output actuation.

## Failure response

Any crash or cleanup failure leaves `~/.hf_dm/hardware_cleanup_interlock.json` latched. Do not remove
it merely to make startup proceed. Physically disconnect or de-energize outputs, inspect the rig,
run failure-injection/all-off checks, repair the cause, and obtain a fresh certificate if hardware,
software, config, calibration, or failsafe revision changed.

Recover incomplete data with `hf-dm recover-session`. The append-only session journal and binary
StreamDIO WAL preserve complete records up to the last successful durable flush; they do not promise
survival of bytes still held in device, OS, or storage-controller caches during abrupt power loss.

## Audit verdict and comparison (2026-07-13)

The task implementation is unusually strong for a focused, single-rig neuroscience application.
The Cursor findings concerning WAL coverage, choice/timeout reconciliation, blocking verification,
timeout-gate simulation, deterministic barcode timestamps, barcode capacity, NPZ payload width, ITI
use, chunk retries, fsync caching, and fail-open cleanup are resolved. Host-clock simulation remains
an intentional limitation: it validates semantics, not Bpod clock drift, USB latency, display timing,
or loaded-rig deadlines. Those measurements belong in the physical certificate.

`bpod-core` is the correct low-level dependency: it is a lean, typed Bpod FSM interface whose own
project page lists configuration, calibration, data management, GUI, and module support as non-goals.
It remains an alpha and currently warns against production use. This repository supplies the missing
task/rig layers while keeping critical timing on Bpod. The pin remains at 0.1.0a12 because 0.1.0a13
changes remote-Bpod parity, serial struct helpers, and documentation rather than relevant local-rig
safety or task behavior.

MM_PyBpod2 is a broader PyBpod/Bpod2 adaptation and supplies the vendor StreamDIO driver. Its README
reports Windows/Python 3.10 as the tested environment. `hf_dm` is narrower but adds an exact Python
3.12 lock, task-specific recovery/QC, output-ownership checks, and a certificate-bound animal gate.

IBLRIG remains the stronger reference for ecosystem maturity: multi-rig deployment, Alyx/ONE,
standard experiment descriptions, distributed transfer, hardware validators, calibration, QC, and
years of production use. This repository should not claim equivalent institutional validation or
breadth. For this force/wheel/StreamDIO experiment, its WAL, cleanup interlock, exact edge-ownership
checks, and independent-failsafe evidence gate are unusually stringent.

References: [bpod-core](https://pypi.org/project/bpod-core/),
[MM_PyBpod2](https://github.com/MathiasMahn/MM_PyBpod2), and
[IBLRIG](https://int-brain-lab.github.io/iblrig/).

Recommendation: stop changing task logic unless a real measurement or pilot exposes a defect.
Continue only along the evidence path—independent failsafe, instrumented certification, failure
injection, full soak, exact release tag/CI, then a tightly supervised minimal pilot with immediate
data and welfare review.
