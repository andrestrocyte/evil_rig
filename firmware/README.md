# StreamDIO Firmware

`Module_DIO_streeaming_100us_cycle/Module_DIO_streeaming_100us_cycle.ino`
is the StreamDIO firmware used by the bringup notebooks, vendored here so the
task and firmware stay in sync.

This cap-touch digitizer build reports firmware version `3`. The Sanworks
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
- uses non-overlapping stream event ranges for 14 inputs: DIO input events
  `1..28`, DIO output events `29..54`, and analog events from `55`
- leaves the Python task and GUI using the same `lick_edge` / counted `lick`
  path they already use for digital input events

After flashing this firmware, the operator GUI should show licks as input
`ch14/pin25` transitions while the force ADC plot continues to receive channel
13 samples.

Compile or upload from the repo with:

```powershell
.\scripts\flash_streamdio_firmware.ps1
.\scripts\flash_streamdio_firmware.ps1 -Port usb:70000/0/0/4
```

Use the StreamDIO Teensy upload port, not the Bpod Teensy. On the rig where
this was prepared, the StreamDIO Triple Serial group was COM6/COM8/COM7 and the
upload port was `usb:70000/0/0/4`; the Bpod Teensy was the separate group
starting at COM5.
