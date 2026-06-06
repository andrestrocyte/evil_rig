# StreamDIO Firmware

`Module_DIO_streeaming_100us_cycle/Module_DIO_streeaming_100us_cycle.ino`
is the StreamDIO firmware used by the bringup notebooks, vendored here so the
task and firmware stay in sync.

The current task streams force on analog channel 13 / Teensy pin 14. The
firmware's existing `N` command supports only one streamed analog channel at a
time, so cap-touch channel 14 / pin 25 cannot also be streamed as a second ADC
channel without changing firmware.

This version keeps force streaming on channel 13 and adds a local cap-touch
digitizer:

- samples cap-touch analog pin 25 in firmware
- tracks a slow baseline while not touched
- emits normal StreamDIO `dio_input` events for channel 14 high/low when the
  sample crosses hysteretic thresholds
- leaves the Python task and GUI using the same `lick_edge` / counted `lick`
  path they already use for digital input events

After flashing this firmware, the operator GUI should show licks as input
`ch14/pin25` transitions while the force ADC plot continues to receive channel
13 samples.

This machine currently does not have `arduino-cli` or `teensy_loader_cli` on
`PATH`, so flashing must be done with the local Teensy/Arduino toolchain.
