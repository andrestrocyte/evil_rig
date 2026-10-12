# StreamDIO Firmware

`Module_DIO_streeaming_100us_cycle/Module_DIO_streeaming_100us_cycle.ino`
is the StreamDIO firmware used by the bringup notebooks, vendored here so the
task and firmware stay in sync.

The speaker output is Teensy **pin 26**. Both the `Y` manual gate command and
the `K` firmware noise generator use this pin. Move the amplifier gate wire
from the original pin 27 to pin 26 with rig power disconnected. The task still
uses continuous PC audio with a separate amplifier gate; the audio cable is
required. Capacitive-touch input remains on pin 25.

The onboard status LED (Teensy pin 13) blinks once per second: 500 ms on,
500 ms off. This nonblocking indicator runs in the foreground, outside the
100 us acquisition interrupt. It shows that the firmware is running; it does
not certify the carrier board, wiring, or connected peripherals.

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
.\scripts\flash_streamdio_firmware.ps1 -Port <verified-StreamDIO-upload-port>
```

Identify the StreamDIO Teensy by its USB serial number before every upload;
USB paths can change when cables move. The replacement StreamDIO currently
has serial `15755980` (Teensy programmer serial `1575598`), with config COM10,
stream COM11, and primary COM12. Bpod has serial `17032730` (programmer serial
`1703273`), with primary COM5. Do not flash the Bpod Teensy with this sketch.
The task's `AUTO` discovery identifies config/stream interfaces from firmware
discovery bytes rather than assuming their COM ordering.
