# StreamDIO Firmware

`Module_DIO_streeaming_100us_cycle/Module_DIO_streeaming_100us_cycle.ino`
is the StreamDIO firmware used by the bringup notebooks, vendored here so the
task and firmware stay in sync.

This cap-touch digitizer and timing-heartbeat build reports firmware version
`5`. The Sanworks
`ArCOM` wrapper is vendored beside the sketch so the firmware can be compiled
without installing an extra Arduino library.

The current task streams force on analog channel 13 / Teensy pin 14. The
firmware's existing `N` command supports only one streamed analog channel at a
time, so cap-touch channel 14 / pin 25 cannot also be streamed as a second ADC
channel without changing firmware.

This version keeps force streaming on channel 13 and adds a local cap-touch
digitizer:

- samples cap-touch analog pin 25 in firmware
- tracks a slow baseline while not touched
- detects both high-going and low-going cap-touch signals with hysteresis
- emits normal StreamDIO `dio_input` events for channel 14 high/low when the
  sample crosses hysteretic thresholds
- exposes an `l` status command with the current ADC value, baseline,
  thresholds, active state, and polarity for low-rate GUI polling
- exposes a USB1 `t` command to tune the cap-touch minimum high threshold,
  touch delta, and release delta at runtime without moving force off analog
  channel 13
- uses non-overlapping stream event ranges for 14 inputs: DIO input events
  `1..28`, DIO output events `29..54`, and analog events from `55`
- leaves the Python task and GUI using the same `lick_edge` / counted `lick`
  path they already use for digital input events

It also emits diagnostic event `1011` every 100,000 us while streaming. Each
event carries the Teensy timestamp at emission; the Python callback records
the earliest host `perf_counter` and wall-clock timestamps for that same
event. Every session therefore contains an always-on clock/transport QC trace
that can distinguish Teensy-period errors, transient host delivery stalls,
missed heartbeats, and accumulated PC-versus-Teensy clock-rate drift. Firmware
older than version 5 is rejected at live StreamDIO configuration so a real
session cannot silently omit this QC stream.

After flashing this firmware, the operator GUI should show licks as input
`ch14/pin25` transitions while the force ADC plot continues to receive channel
13 samples. Its session-status panel also shows the current 100 ms heartbeat
interval and cumulative PC-minus-Teensy elapsed-time difference.

Compile or upload from the repo with:

```powershell
.\scripts\flash_streamdio_firmware.ps1
.\scripts\flash_streamdio_firmware.ps1 -Port usb:70000/0/0/4
```

Use the StreamDIO Teensy upload port, not the Bpod Teensy. On the rig where
this was prepared, the StreamDIO Triple Serial group was COM6/COM8/COM7 and the
upload port was `usb:70000/0/0/4`; the Bpod Teensy was the separate group
starting at COM5. Do not infer config/stream ordering from COM numbers; the
task's `AUTO` discovery identifies those interfaces from firmware discovery
bytes.
