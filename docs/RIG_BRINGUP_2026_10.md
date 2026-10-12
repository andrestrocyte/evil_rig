# Rig bringup and integration, October 2026

Changes ported from the working `bpod-hf-dm` checkout into `evil_rig`:

- Keep CW force on pin 21 and CCW force on pin 19; invert wheel movement in the task display and matching choice/force logic through `display.invert_wheel_movement`.
- Move speaker gate from pin 27 to pin 26 in firmware, host constants, diagnostics and documentation. The amplifier uses external 12 V and a separate computer audio cable.
- Refresh module discovery when Bpod initially reports no module. Send immediate module commands with Bpod's `T` transport rather than physical-output overrides.
- Run the Teensy status LED from the foreground, outside the sampling interrupt.
- Choose capacitive (StreamDIO pin 25, power pin 15) or optical (Bpod behavior port 1) licks per session in the setup GUI or CLI. Show optical edges/counts in the same GUI, keep capacitive power off in optical sessions, and observe optical licks during ITI/preflight through output-free Bpod state machines.
- Export lick source, clock, state, Bpod event/trial and original Bpod timestamp; make hardware flags source-aware.
- Add optional StreamDIO receipt timestamps for optical licks using the existing Bpod-module link and existing StreamDIO firmware commands.
- Remove seconds of Mono8 C-API camera backlog by saving copied pixel buffers directly as lossless, full-resolution PGM files and generating previews from memory. Keep saved frame numbering and timestamp CSV format; do not switch acquisition to an overwriting buffer mode.

`evil_rig` retains its newer configuration validation, recorder/DAQ/camera health monitoring, durability and certification logic. Machine-specific COM settings, recordings, certificates and calibration data are not copied into Git.

## Optical clock selection and meaning

Keep the optical lickometer connected to Bpod behavior port 1 and StreamDIO connected to Bpod module port 1. No extra wire, level converter or firmware flash is needed for this relay.

Select optical licks and StreamDIO clock in session setup, or add:

```text
--lick-source optical --lick-port 1 --lick-clock streamdio
```

Shipped YAML files select `licks.optical_clock: streamdio`; older YAML files without that setting retain Bpod clock for compatibility. Capacitive licks always use StreamDIO.

Route: optical edge -> Bpod hardware event -> host reader -> Bpod module UART -> StreamDIO `IP` marker -> StreamDIO timestamped `port_input` receipt.

**The StreamDIO timestamp is marker receipt time, not the original physical beam-break time.** This route includes variable Bpod USB delivery, host polling/scheduling and UART latency. It gives a common clock domain but does not give simultaneous physical-edge sampling. For precise lick intervals retain `bpod_timestamp_us`; `timestamp_origin=streamdio_uart_receive` makes the receipt semantics explicit. `relay_receipt_wait_s` measures only the forwarding-to-host-receipt portion, not the complete physical-edge delay. Missing, mismatched or duplicate pending receipts fail the session rather than silently changing clock.

Hardware validation on 2026-10-12 confirmed a synthetic marker passed through Bpod during an active output-free FSM and returned a StreamDIO timestamp (about 0.62 ms forwarding-to-host receipt in that one probe). No physical beam transitions occurred during that relay check. Optical beam detection had been confirmed separately; the combined physical-edge relay still needs a block/unblock check. Simulation tests exercise both clocks, edge counts, ITI observation and export provenance.

## Camera delay evidence

The latest pre-fix rig session saved 950 frames in 58.3 s (16.3 fps) with 30 Hz requested triggers. Camera versus host timestamp deltas showed up to 2.77 s accumulated backlog. Preview publishing-to-GUI transit was much smaller: median 27 ms, 95th percentile 51 ms. This localizes the major delay to acquisition/recording rather than the GUI queue.

A standalone eight-second hardware comparison on the same Blackfly S camera found:

| Metric | Before | After |
|---|---:|---:|
| Saved frames | 115 | 239 |
| Saved rate | 14.36 fps | 29.86 fps |
| Maximum accumulated camera/host backlog growth | 3.056 s | 0.0093 s |
| Capture/save/cleanup errors | 0 | 0 |

The fix bypasses `spinImageSaveFromExt` for Mono8, copies pixels before SDK image release, preserves stride/padding correctly, and writes the same PGM format directly. Other pixel formats/older SDKs retain the SDK fallback. Camera setup now temporarily selects Timed exposure before configuring trigger nodes, then restores TriggerWidth after enabling FrameStart/Line0; the connected camera makes TriggerMode unavailable in TriggerWidth mode. This preserves the stricter required-setting checks in `evil_rig` rather than suppressing their errors. Preview queues continue to keep the newest frame. The GUI age indicator now uses host acquisition time when available, so queue transit is included; it is not a calibrated sensor-exposure-to-screen latency measurement. Camera timestamps and StreamDIO trigger timestamps remain the scientific timing sources.

A small visible preview delay remains from preview throttling, GUI refresh and screen scanout. The follow-up audit increased the preview target to 15 Hz and verified a one-minute GUI acquisition: 1,798 triggers matched 1,798 saved frames, maximum backlog growth was 20.83 ms and GUI frame age was 74 ms median / 92 ms p95. These are not calibrated exposure-to-display latency measurements. See [the audit](QUALITY_AUDIT_2026_10.md) for evidence and remaining physical checks. Previously missed/delayed video frames cannot be recovered by this change.
