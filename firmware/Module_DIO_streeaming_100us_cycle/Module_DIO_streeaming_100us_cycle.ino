/*
  ----------------------------------------------------------------------------

  This file is part of the MM_PyBpod2 repository
  and based on the DIO module in the Sanworks Bpod_Gen2 repository
  Copyright (C) Mathias Mahn 2025
  Modified to include camera frame trigger and equipment control

  ----------------------------------------------------------------------------

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  version 3.

  This program is distributed  WITHOUT ANY WARRANTY and without even the
  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/
// Custom DIO Module with 100 microsecond deterministic cycle time interfaces the Bpod state machine with TTL signals on Teensy digital pins.
// Uses IntervalTimer for precise timing and circular buffers for non-blocking operations
// Input channels: pins 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 14, 25
// Output channels: pins 23, 22, 21, 20, 19, 18, 17, 16, 41, 40, 39, 38, 15
// Special function pins:
//   Pin 32: Camera frame trigger (configurable frequency and exposure)
//   Pin 31: UCLA miniscope trigger (toggle high/low)
//   Pin 30: Fiber photometry (toggle high/low)
//   Pin 27: speaker white-noise output or manual TTL gate
//   Pin 14 / channel 13: optional analog force input on the direct USB streaming path

#include "ArCOM.h"  // Import Sanworks serial communication wrapper
//#include "TeensyTimerTool.h"
//using namespace TeensyTimerTool;

#if !defined(USB_TRIPLE_SERIAL)
  #error Error! You must choose 'Triple Serial' from the 'Tools' > 'USB Type' menu in the Arduino application before uploading this firmware.
#endif

// ========== CONSTANTS AND DEFINITIONS ==========

// Timing constants
#define CYCLE_TIME_MICROS 100
#define CYCLES_PER_MS 10

// Diagnostic event codes
#define DIAGNOSTIC_EVENT_BASE 1000
#define DIAG_SERIAL1_OVERFLOW 1001
#define DIAG_OUTPUT_QUEUE_OVERFLOW 1002
#define DIAG_CRITICAL_OVERFLOW 1003
#define DIAG_SERIAL1_PARSE_ERROR 1004
#define DIAG_USB1_TIMEOUT 1005
#define DIAG_BUFFER_EMERGENCY_RESET 1006
#define DIAG_STREAMING_OVERFLOW 1007
#define DIAG_SYSTEM_RESET 1008
#define DIAG_CYCLE_TIMING_VIOLATION 1009
#define DIAG_CYCLE_COUNT_VIOLATION 1010
#define DIAG_TIMING_HEARTBEAT 1011
// Debug-only diagnostic events (1100-1199)
#define DIAG_DEBUG_BASE 1100
#define DIAG_CYCLE_TIME 1101
#define DIAG_CYCLE_STATS 1102
#define DIAG_BUFFER_STATS 1105
#define DIAG_PERFORMANCE_STATS 1107
#define DIAG_TEENSY_USB_BUFFER_FREE 1109
#define DIAG_SYSTEM_HEALTH 1112 // implement
#define DIAG_COMMUNICATION_STATS 1113 // implement


// Event code definitions
#define DIO_INPUT_EVENT_BASE 1
#define DIO_OUTPUT_EVENT_BASE 29
#define ANALOG_INPUT_EVENT_BASE 55
#define CAMERA_EVENT 75
#define BPOD_STATE_EVENT_BASE 100
#define BPOD_BNC_OUT_EVENT_BASE 400
#define BPOD_VALVE_EVENT_BASE 404
#define BPOD_LED_EVENT_BASE 414
#define BPOD_BNC_IN_EVENT_BASE 700
#define BPOD_PORT_IN_EVENT_BASE 704

// Buffer sizes
#define SERIAL1_BUFFER_SIZE 256
#define OUTPUT_QUEUE_SIZE 64
#define STREAMING_BUFFER_SIZE 1024
#define CRITICAL_EVENT_BUFFER_SIZE 16

// Buffer pressure thresholds
#define BUFFER_PRESSURE_THRESHOLD 0.6f    // 60% full = pressure
#define BUFFER_CRITICAL_THRESHOLD 0.8f    // 80% full = critical
#define BUFFER_EMERGENCY_THRESHOLD 0.9f   // 90% full = emergency

// Channel counts
#define nInputChannels 14
#define nOutputChannels 13

// Analog sample packing: 12-bit ADC value in the upper bits, 52-bit timestamp below.
#define ANALOG_VALUE_SHIFT 52
#define ANALOG_VALUE_MASK 0x0FFF
#define ANALOG_TIMESTAMP_MASK ((1ULL << ANALOG_VALUE_SHIFT) - 1ULL)

// Forward declarations for structs used in function signatures.
// This avoids Arduino auto-prototype issues when functions below accept these types.
struct BufferStatus;
struct OutputCommand;
struct StreamingEvent;

// ========== GLOBAL VARIABLES ==========

// Module setup
ArCOM Serial1COM(Serial1);
ArCOM PC1(SerialUSB1);
ArCOM PC2(SerialUSB2);

char moduleName[] = "DIO_12Ch_Stream";
const char* eventNames[] = {"1_Hi", "1_Lo", "2_Hi", "2_Lo", "3_Hi", "3_Lo", "4_Hi", "4_Lo", "5_Hi", "5_Lo",
                      "6_Hi", "6_Lo", "7_Hi", "7_Lo", "8_Hi", "8_Lo", "9_Hi", "9_Lo", "10_Hi", "10_Lo",
                      "11_Hi", "11_Lo", "12_Hi", "12_Lo", "13_Hi", "13_Lo", "14_Hi", "14_Lo"};
const uint32_t FirmwareVersion = 5;
byte nEventNames = (sizeof(eventNames)/sizeof(char *));

// Pin definitions
byte inputPins[nInputChannels] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 14, 25};
byte outputPins[nOutputChannels] = {23, 22, 21, 20, 19, 18, 17, 16, 41, 40, 39, 38, 15};
const byte FORCE_ANALOG_CHANNEL = 13;  // Module channel 13 maps to Teensy pin 14 / A0.
const byte CAP_TOUCH_CHANNEL = 14;     // Module channel 14 maps to Teensy pin 25 / A11.
const byte CAP_TOUCH_POWER_OUTPUT_CHANNEL = 13; // Output channel 13 maps to Teensy pin 15.

// Special function pins
const byte CAMERA_PIN = 32;
const byte MINISCOPE_PIN = 31;
const byte PHOTOMETRY_PIN = 30;
const byte SPEAKER_PIN = 27;
const int LED_pin = 13;

// Timing and control variables
IntervalTimer cycleTimer;

volatile uint32_t cycleCount = 0;
volatile uint64_t sessionStartTime = 0;
volatile uint32_t lastCycleStartMicros = 0;
uint32_t refractoryPeriod = 300;

// Input/output state variables
byte inputChState[nInputChannels] = {0};
byte lastInputChState[nInputChannels] = {0};
byte inputsEnabled[nInputChannels] = {0};
uint64_t inputChSwitchTime[nInputChannels] = {0};
byte outputChState[nOutputChannels] = {0};
uint16_t allOutputStates = 0;

// Analog force input state
boolean analogInputEnabled = false;
byte analogInputChannel = FORCE_ANALOG_CHANNEL;
uint32_t analogSampleIntervalUs = 1000;
uint64_t lastAnalogSampleTime = 0;
uint16_t lastAnalogValue = 0;

// Cap-touch ADC-to-digital event state.
//
// The firmware streams one configured analog channel at a time through the
// existing 'N' command. The decision task uses that stream for force on channel
// 13, while the lick detector output on channel 14 is analog enough that normal
// digitalReadFast() edges may never cross the Teensy digital threshold. This
// local digitizer samples pin 25 in firmware and emits the normal channel-14
// DIO high/low events, so force ADC streaming can stay on channel 13.
boolean capTouchDigitizerEnabled = true;
uint32_t capTouchSampleIntervalUs = 1000;
uint64_t lastCapTouchSampleTime = 0;
uint16_t capTouchBaseline = 0;
uint16_t capTouchLastValue = 0;
byte capTouchDigitizedState = 0;
byte capTouchActivePolarity = 0; // 0 = idle, 1 = high-going touch, 2 = low-going touch.
boolean capTouchBaselineInitialized = false;
uint64_t capTouchIgnoreUntilTime = 0;
const uint16_t CAP_TOUCH_DEFAULT_MIN_HIGH_THRESHOLD = 700;
const uint16_t CAP_TOUCH_DEFAULT_DELTA_HIGH = 350;
const uint16_t CAP_TOUCH_DEFAULT_DELTA_LOW = 160;
const uint32_t CAP_TOUCH_POWER_SETTLE_US = 100000;
uint16_t capTouchMinHighThreshold = CAP_TOUCH_DEFAULT_MIN_HIGH_THRESHOLD;
uint16_t capTouchDeltaHigh = CAP_TOUCH_DEFAULT_DELTA_HIGH;
uint16_t capTouchDeltaLow = CAP_TOUCH_DEFAULT_DELTA_LOW;

// Speaker white-noise state
boolean speakerNoiseEnabled = false;
uint32_t speakerSampleIntervalUs = CYCLE_TIME_MICROS;
uint64_t lastSpeakerSampleTime = 0;
uint32_t speakerSamplesGenerated = 0;
uint32_t speakerNoiseLfsr = 0x13579BDF;
byte speakerCurrentState = 0;
byte speakerGateState = 0;

// Camera control variables
boolean cameraEnabled = false;
uint32_t cameraFrequency = 30;
uint32_t exposureTime = 10000;
uint32_t framePeriod = 33333;
uint64_t lastFrameTime = 0;
uint64_t frameStartTime = 0;
boolean frameActive = false;

// Equipment states
boolean miniscopeEnabled = false;
boolean photometryEnabled = false;
byte currentBpodState = 0;

// USB connection states
boolean PC1Connected = false;
boolean PC2Connected = false;
boolean streamingEnabled = false;
const byte configDiscoveryByte = 222;
const byte streamingDiscoveryByte = 223;
volatile bool needsUSB2Processing = false;

// Firmware-origin timing heartbeat. The host records its own receive/dispatch
// time for the same event, providing independent Teensy and PC clock samples.
const uint64_t TIMING_HEARTBEAT_PERIOD_US = 100000;
uint64_t nextTimingHeartbeatUs = 0;

// Timing control
const int cycle_pin = 33;
const int USB_send_pin = 34;
boolean cyle_state = false;
boolean LED_state = false;

inline void setStatusLed(boolean state) {
  if (LED_pin >= 0) {
    digitalWrite(LED_pin, state ? HIGH : LOW);
  }
}

// Buffer monitoring structure
struct BufferStatus {
  float serial1FillRatio;
  float criticalEventFillRatio;
  float streamingFillRatio;
  bool isUnderPressure;
  bool isOverflowing;
};

// // Dynamic processing limits with buffer awareness
// struct ProcessingLimits {
//   int maxReads = 16;
//   int maxSends = 8;
//   int maxProcessed = 32;

//   // Emergency limits for high-pressure situations
//   int emergencyMaxReads = 64;
//   int emergencyMaxSends = 32;
//   int emergencyMaxProcessed = 128;

//   // Minimum safe limits
//   int minReads = 8;
//   int minSends = 4;
//   int minProcessed = 16;

//   uint32_t lastAdjustTime = 0;
// } limits;

// Debug control
boolean debugMode = false;
boolean verboseDebug = false;

// Cycle timing statistics
struct CycleStats {
  uint32_t durations[100];
  uint32_t computeTimes[100];
  uint8_t index;
  uint32_t count;
  uint32_t lastReportTime;
} cycleStats = {.index = 0, .count = 0, .lastReportTime = 0};

// Additional debug counters
struct DebugCounters {
  uint32_t usb1Commands;
  uint32_t usb2Commands;
  uint32_t serial1Bytes;
  uint32_t bufferOverflows;
  uint32_t lastReportTime;
} debugCounters = {0};

// ========== DATA STRUCTURES ==========

// Serial1 incoming command buffer
struct Serial1Buffer {
  volatile byte buffer[SERIAL1_BUFFER_SIZE];
  volatile uint16_t head;
  volatile uint16_t tail;
  volatile uint16_t count;
} serial1Buf = {.head = 0, .tail = 0, .count = 0};

// Output command queue for deferred execution
struct OutputCommand {
  byte type; // 1=single channel, 2=all channels, 3=camera, 4=equipment
  byte channel;
  byte state;
  uint16_t data;
};

struct OutputQueue {
  OutputCommand commands[OUTPUT_QUEUE_SIZE];
  volatile uint16_t head;
  volatile uint16_t tail;
  volatile uint16_t count;
} outputQueue = {.head = 0, .tail = 0, .count = 0};

// Event streaming buffer
struct StreamingEvent {
  uint16_t eventCode;
  uint64_t timestamp;
};

struct StreamingBuffer {
  StreamingEvent events[STREAMING_BUFFER_SIZE];
  volatile uint16_t head;
  volatile uint16_t tail;
  volatile uint16_t count;
} streamBuf = {.head = 0, .tail = 0, .count = 0};

// Critical events to send to Bpod immediately
struct CriticalEventBuffer {
  volatile byte events[CRITICAL_EVENT_BUFFER_SIZE];
  volatile uint16_t head;
  volatile uint16_t tail;
  volatile uint16_t count;
} criticalEvents = {.head = 0, .tail = 0, .count = 0};

struct USBCommandState {
  bool inProgress;
  byte commandCode;
  byte bytesNeeded;
  byte bytesReceived;
  byte buffer[8]; // Max bytes for any command
  uint32_t startTime;
  uint32_t timeoutMs;
};
USBCommandState usb1CmdState = {false, 0, 0, 0, {}, 0, 20};
USBCommandState usb2CmdState = {false, 0, 0, 0, {}, 0, 20};


// ========== UTILITY FUNCTIONS ==========

static uint64_t microsOverflows = 0;
static uint32_t lastMicros = 0;

uint64_t getCurrentTimestamp() {
    uint32_t currentMicros = micros();

    // Handle overflow (happens every ~71 minutes)
    if (currentMicros < lastMicros) {
        microsOverflows++;
    }
    lastMicros = currentMicros;

    return (microsOverflows << 32ULL) + currentMicros - sessionStartTime;
}

// // Get current timestamp (safe to call from ISR)
// uint64_t getCurrentTimestamp() {
//   return sessionStartTime + ((uint64_t)cycleCount * CYCLE_TIME_MICROS);
// }

// ========== BUFFER OPERATIONS ==========

bool serial1BufferPut(byte data) {
  uint16_t next = (serial1Buf.head + 1) % SERIAL1_BUFFER_SIZE;
  if (next == serial1Buf.tail) return false;

  serial1Buf.buffer[serial1Buf.head] = data;
  serial1Buf.head = next;
  serial1Buf.count++;
  return true;
}

bool serial1BufferGet(byte* data) {
  if (serial1Buf.count == 0) return false;

  *data = serial1Buf.buffer[serial1Buf.tail];
  serial1Buf.tail = (serial1Buf.tail + 1) % SERIAL1_BUFFER_SIZE;
  serial1Buf.count--;
  return true;
}

bool outputQueuePut(byte type, byte channel, byte state, uint16_t data = 0) {
  uint16_t next = (outputQueue.head + 1) % OUTPUT_QUEUE_SIZE;
  if (next == outputQueue.tail) return false;

  outputQueue.commands[outputQueue.head] = {type, channel, state, data};
  outputQueue.head = next;
  outputQueue.count++;
  return true;
}

bool outputQueueGet(OutputCommand* cmd) {
  if (outputQueue.count == 0) return false;

  *cmd = outputQueue.commands[outputQueue.tail];
  outputQueue.tail = (outputQueue.tail + 1) % OUTPUT_QUEUE_SIZE;
  outputQueue.count--;
  return true;
}

bool streamingEventPut(uint16_t eventCode, uint64_t timestamp) {
  // if (!streamingEnabled && eventCode < DIAGNOSTIC_EVENT_BASE) {
  if (!streamingEnabled) {
    return true; // Ignore non-diagnostic events if not streaming
  }

  uint16_t next = (streamBuf.head + 1) % STREAMING_BUFFER_SIZE;
  if (next == streamBuf.tail) return false;

  streamBuf.events[streamBuf.head] = {eventCode, timestamp};
  streamBuf.head = next;
  streamBuf.count++;
  return true;
}

bool streamingEventGet(StreamingEvent* event) {
  if (streamBuf.count == 0) return false;

  *event = streamBuf.events[streamBuf.tail];
  streamBuf.tail = (streamBuf.tail + 1) % STREAMING_BUFFER_SIZE;
  streamBuf.count--;
  return true;
}

bool criticalEventPut(byte eventCode) {
  uint16_t next = (criticalEvents.head + 1) % CRITICAL_EVENT_BUFFER_SIZE;
  if (next == criticalEvents.tail) return false;

  criticalEvents.events[criticalEvents.head] = eventCode;
  criticalEvents.head = next;
  criticalEvents.count++;
  return true;
}

bool criticalEventGet(byte* eventCode) {
  if (criticalEvents.count == 0) return false;

  *eventCode = criticalEvents.events[criticalEvents.tail];
  criticalEvents.tail = (criticalEvents.tail + 1) % CRITICAL_EVENT_BUFFER_SIZE;
  criticalEvents.count--;
  return true;
}

// Calculate current buffer status
BufferStatus getBufferStatus() {
  BufferStatus status;

  // Calculate fill ratios (you'll need to expose these from your buffer structs)
  status.serial1FillRatio = (float)serial1Buf.count / SERIAL1_BUFFER_SIZE;
  status.criticalEventFillRatio = (float)criticalEvents.count / CRITICAL_EVENT_BUFFER_SIZE;
  status.streamingFillRatio = (float)streamBuf.count / STREAMING_BUFFER_SIZE;

  // Determine pressure state
  float maxFillRatio = max(status.serial1FillRatio,
                          max(status.criticalEventFillRatio, status.streamingFillRatio));

  status.isUnderPressure = maxFillRatio > BUFFER_PRESSURE_THRESHOLD;
  status.isOverflowing = maxFillRatio > BUFFER_EMERGENCY_THRESHOLD;

  return status;
}

// // Smart limit adjustment based on buffer pressure AND compute time
// void adjustProcessingLimits(uint32_t computeTime, const BufferStatus& bufferStatus) {
//   static uint32_t adjustCounter = 0;
//   static uint32_t lastPressureTime = 0;
//   static bool wasPressured = false;

//   adjustCounter++;
//   uint32_t currentTime = micros();

//   // Only adjust every 50 cycles (5ms) to avoid oscillation
//   if (adjustCounter % 50 != 0) return;

//   // Emergency mode: buffers are critically full
//   if (bufferStatus.isOverflowing) {
//     limits.maxReads = limits.emergencyMaxReads;
//     limits.maxSends = limits.emergencyMaxSends;
//     limits.maxProcessed = limits.emergencyMaxProcessed;
//     lastPressureTime = currentTime;
//     wasPressured = true;
//     return;
//   }

//   // High pressure mode: buffers are filling up
//   if (bufferStatus.isUnderPressure) {
//     lastPressureTime = currentTime;
//     wasPressured = true;

//     // Scale limits based on buffer pressure and available compute time
//     float pressureMultiplier = 1.0f;
//     if (bufferStatus.serial1FillRatio > BUFFER_PRESSURE_THRESHOLD) {
//       pressureMultiplier = max(pressureMultiplier,
//         1.0f + (bufferStatus.serial1FillRatio - BUFFER_PRESSURE_THRESHOLD) * 3.0f);
//     }
//     if (bufferStatus.criticalEventFillRatio > BUFFER_PRESSURE_THRESHOLD) {
//       pressureMultiplier = max(pressureMultiplier,
//         1.0f + (bufferStatus.criticalEventFillRatio - BUFFER_PRESSURE_THRESHOLD) * 4.0f);
//     }

//     // Only increase limits if we have compute time available
//     if (computeTime < 70) { // Less than 70µs used
//       limits.maxReads = min(limits.emergencyMaxReads,
//         (int)(limits.minReads * pressureMultiplier));
//       limits.maxSends = min(limits.emergencyMaxSends,
//         (int)(limits.minSends * pressureMultiplier));
//       limits.maxProcessed = min(limits.emergencyMaxProcessed,
//         (int)(limits.minProcessed * pressureMultiplier));
//     }
//     return;
//   }

//   // Gradual cooldown after pressure period
//   if (wasPressured && (currentTime - lastPressureTime) > 10000) { // 10ms cooldown
//     // Slowly ramp down limits back to normal
//     limits.maxReads = max(limits.minReads, limits.maxReads - 2);
//     limits.maxSends = max(limits.minSends, limits.maxSends - 1);
//     limits.maxProcessed = max(limits.minProcessed, limits.maxProcessed - 4);

//     // If we've reached minimum levels, clear pressure state
//     if (limits.maxReads <= 16 && limits.maxSends <= 4 && limits.maxProcessed <= 32) {
//       wasPressured = false;
//     }
//     return;
//   }

//   // Normal operation: only adjust if timing is consistently good/bad
//   static uint32_t goodTimingCount = 0;
//   static uint32_t badTimingCount = 0;

//   if (computeTime < 50) {
//     goodTimingCount++;
//     badTimingCount = 0;
//   } else if (computeTime > 80) {
//     badTimingCount++;
//     goodTimingCount = 0;
//   } else {
//     goodTimingCount = 0;
//     badTimingCount = 0;
//   }

//   // Only adjust after consistent performance (prevent oscillation)
//   if (goodTimingCount > 10 && !wasPressured) {
//     // Slightly increase limits for better responsiveness
//     limits.maxReads = min(limits.emergencyMaxReads, limits.maxReads + 1);
//     limits.maxSends = min(limits.emergencyMaxSends, limits.maxSends + 1);
//     limits.maxProcessed = min(limits.emergencyMaxProcessed, limits.maxProcessed + 2);
//     goodTimingCount = 0;
//   } else if (badTimingCount > 5) {
//     // Decrease limits to maintain timing
//     limits.maxReads = max(limits.minReads, limits.maxReads - 1);
//     limits.maxSends = max(limits.minSends, limits.maxSends - 1);
//     limits.maxProcessed = max(limits.minProcessed, limits.maxProcessed - 2);
//     badTimingCount = 0;
//   }
// }

byte getUSBCommandBytes(byte cmd) {
  switch (cmd) {
    case 'R': return 4; // refractory period
    case 'E': return 2; // enable input
    case 'N': return 6; // configure analog input
    case 't': return 6; // configure cap-touch ADC digitizer thresholds
    case 'K': return 5; // speaker white-noise enable + sample interval
    case 'Y': return 1; // speaker gate high / low on pin 27
    case 'A': return 2; // all outputs
    case 'O': return 2; // single output
    case 'C': return 1; // camera enable
    case 'Q': return 4; // camera frequency
    case 'X': return 4; // exposure time
    case 'M': return 1; // miniscope
    case 'P': return 1; // photometry
    case 'D': return 1; // debug mode
    default: return 0;  // immediate commands
  }
}

// ========== INTERRUPT SERVICE ROUTINE ==========
// moved to end, so that all functions are defined before ISR

// ========== CORE PROCESSING FUNCTIONS ==========

inline boolean isAnalogCapableChannel(byte channel) {
  return channel == FORCE_ANALOG_CHANNEL || channel == CAP_TOUCH_CHANNEL;
}

inline uint64_t packAnalogSample(uint64_t timestampUs, uint16_t analogValue) {
  return (((uint64_t)(analogValue & ANALOG_VALUE_MASK)) << ANALOG_VALUE_SHIFT) |
         (timestampUs & ANALOG_TIMESTAMP_MASK);
}

uint32_t nextSpeakerNoiseValue() {
  speakerNoiseLfsr ^= speakerNoiseLfsr << 13;
  speakerNoiseLfsr ^= speakerNoiseLfsr >> 17;
  speakerNoiseLfsr ^= speakerNoiseLfsr << 5;
  if (speakerNoiseLfsr == 0) {
    speakerNoiseLfsr = 0x13579BDF;
  }
  return speakerNoiseLfsr;
}

void setAnalogInputEnabled(byte channel, boolean enabled) {
  if (!isAnalogCapableChannel(channel)) {
    return;
  }

  byte channelIndex = channel - 1;
  analogInputChannel = channel;
  analogInputEnabled = enabled;
  lastAnalogSampleTime = 0;

  if (enabled) {
    inputsEnabled[channelIndex] = 0;
    pinMode(inputPins[channelIndex], INPUT_DISABLE);
    lastAnalogValue = analogRead(inputPins[channelIndex]) & ANALOG_VALUE_MASK;
  } else {
    if (channel == CAP_TOUCH_CHANNEL) {
      pinMode(inputPins[channelIndex], INPUT);
    } else {
      pinMode(inputPins[channelIndex], INPUT_PULLUP);
    }
    inputChState[channelIndex] = digitalReadFast(inputPins[channelIndex]);
    lastInputChState[channelIndex] = inputChState[channelIndex];
  }
}

void processInputs() {
  uint64_t currentTime = getCurrentTimestamp();
  byte thisEvent = 1;

  for (int i = 0; i < nInputChannels; i++) {
    if (capTouchDigitizerEnabled && i == (CAP_TOUCH_CHANNEL - 1)) {
      thisEvent += 2;
      continue;
    }
    if (inputsEnabled[i] == 1 && !(analogInputEnabled && i == (analogInputChannel - 1))) {
      byte newState = digitalReadFast(inputPins[i]);

      // Check refractory period
      if (currentTime - inputChSwitchTime[i] > refractoryPeriod) {
        if ((newState == 1) && (lastInputChState[i] == 0)) {
          // Rising edge - CRITICAL for Bpod state machine
          Serial1COM.writeByte(thisEvent);
          streamingEventPut(DIO_INPUT_EVENT_BASE + thisEvent - 1, currentTime);
          inputChSwitchTime[i] = currentTime;
          lastInputChState[i] = newState;
        } else if ((newState == 0) && (lastInputChState[i] == 1)) {
          // Falling edge - CRITICAL for Bpod state machine
          Serial1COM.writeByte(thisEvent + 1);
          streamingEventPut(DIO_INPUT_EVENT_BASE + thisEvent, currentTime);
          inputChSwitchTime[i] = currentTime;
          lastInputChState[i] = newState;
        }
      }
      inputChState[i] = newState;
    }
    thisEvent += 2;
  }
}

void emitCapTouchDigitizedEvent(byte newState, uint64_t timestampUs) {
  const byte channelIndex = CAP_TOUCH_CHANNEL - 1;
  const byte eventByte = 1 + channelIndex * 2 + (newState ? 0 : 1);
  Serial1COM.writeByte(eventByte);
  streamingEventPut(DIO_INPUT_EVENT_BASE + eventByte - 1, timestampUs);
  inputChSwitchTime[channelIndex] = timestampUs;
  inputChState[channelIndex] = newState;
  lastInputChState[channelIndex] = newState;
}

void resetCapTouchDigitizer(uint64_t timestampUs) {
  const byte channelIndex = CAP_TOUCH_CHANNEL - 1;
  capTouchBaseline = 0;
  capTouchLastValue = 0;
  capTouchDigitizedState = 0;
  capTouchActivePolarity = 0;
  capTouchBaselineInitialized = false;
  lastCapTouchSampleTime = 0;
  capTouchIgnoreUntilTime = timestampUs + CAP_TOUCH_POWER_SETTLE_US;
  inputChState[channelIndex] = 0;
  lastInputChState[channelIndex] = 0;
  inputChSwitchTime[channelIndex] = timestampUs;
}

uint16_t capTouchSaturatedAdd(uint16_t value, uint16_t delta) {
  uint32_t result = (uint32_t)value + delta;
  if (result > ANALOG_VALUE_MASK) {
    return ANALOG_VALUE_MASK;
  }
  return (uint16_t)result;
}

uint16_t capTouchSaturatedSubtract(uint16_t value, uint16_t delta) {
  if (value <= delta) {
    return 0;
  }
  return value - delta;
}

uint16_t capTouchRisingHighThreshold() {
  uint16_t threshold = capTouchSaturatedAdd(capTouchBaseline, capTouchDeltaHigh);
  if (threshold < capTouchMinHighThreshold) {
    threshold = capTouchMinHighThreshold;
  }
  return threshold;
}

uint16_t capTouchRisingLowThreshold(uint16_t highThreshold) {
  uint16_t threshold = capTouchSaturatedAdd(capTouchBaseline, capTouchDeltaLow);
  if (threshold > highThreshold) {
    threshold = highThreshold;
  }
  return threshold;
}

uint16_t capTouchFallingHighThreshold() {
  return capTouchSaturatedSubtract(capTouchBaseline, capTouchDeltaHigh);
}

uint16_t capTouchFallingLowThreshold() {
  return capTouchSaturatedSubtract(capTouchBaseline, capTouchDeltaLow);
}

void configureCapTouchDigitizer(uint16_t minHighThreshold, uint16_t deltaHigh, uint16_t deltaLow) {
  capTouchMinHighThreshold = min((uint16_t)ANALOG_VALUE_MASK, minHighThreshold);
  capTouchDeltaHigh = min((uint16_t)ANALOG_VALUE_MASK, deltaHigh);
  capTouchDeltaLow = min((uint16_t)ANALOG_VALUE_MASK, deltaLow);
  if (capTouchDeltaLow > capTouchDeltaHigh) {
    capTouchDeltaLow = capTouchDeltaHigh;
  }
  resetCapTouchDigitizer(getCurrentTimestamp());
}

void processCapTouchDigitizer() {
  if (!capTouchDigitizerEnabled) {
    return;
  }

  // If channel 14 is deliberately configured as the streamed analog channel,
  // let processAnalogInput() own it and avoid duplicate samples/events.
  if (analogInputEnabled && analogInputChannel == CAP_TOUCH_CHANNEL) {
    return;
  }

  uint64_t currentTime = getCurrentTimestamp();
  if (currentTime < capTouchIgnoreUntilTime) {
    lastCapTouchSampleTime = currentTime;
    return;
  }

  if (currentTime - lastCapTouchSampleTime < capTouchSampleIntervalUs) {
    return;
  }

  const byte channelIndex = CAP_TOUCH_CHANNEL - 1;
  uint16_t value = analogRead(inputPins[channelIndex]) & ANALOG_VALUE_MASK;
  capTouchLastValue = value;
  lastCapTouchSampleTime = currentTime;

  if (!capTouchBaselineInitialized) {
    capTouchBaseline = value;
    capTouchBaselineInitialized = true;
  }

  if (capTouchDigitizedState == 0) {
    // Track slow drift only while not touching. Keep this integer-only so the
    // 100 us cycle stays cheap.
    capTouchBaseline = (uint16_t)(((uint32_t)capTouchBaseline * 31 + value) / 32);
  }

  uint16_t risingHighThreshold = capTouchRisingHighThreshold();
  uint16_t risingLowThreshold = capTouchRisingLowThreshold(risingHighThreshold);
  uint16_t fallingHighThreshold = capTouchFallingHighThreshold();
  uint16_t fallingLowThreshold = capTouchFallingLowThreshold();

  if (
    capTouchDigitizedState == 0 &&
    value >= risingHighThreshold &&
    currentTime - inputChSwitchTime[channelIndex] > refractoryPeriod
  ) {
    capTouchDigitizedState = 1;
    capTouchActivePolarity = 1;
    emitCapTouchDigitizedEvent(1, currentTime);
  } else if (
    capTouchDigitizedState == 0 &&
    capTouchBaseline > capTouchDeltaHigh &&
    value <= fallingHighThreshold &&
    currentTime - inputChSwitchTime[channelIndex] > refractoryPeriod
  ) {
    capTouchDigitizedState = 1;
    capTouchActivePolarity = 2;
    emitCapTouchDigitizedEvent(1, currentTime);
  } else if (
    capTouchDigitizedState == 1 &&
    capTouchActivePolarity == 1 &&
    value <= risingLowThreshold &&
    currentTime - inputChSwitchTime[channelIndex] > refractoryPeriod
  ) {
    capTouchDigitizedState = 0;
    capTouchActivePolarity = 0;
    emitCapTouchDigitizedEvent(0, currentTime);
  } else if (
    capTouchDigitizedState == 1 &&
    capTouchActivePolarity == 2 &&
    value >= fallingLowThreshold &&
    currentTime - inputChSwitchTime[channelIndex] > refractoryPeriod
  ) {
    capTouchDigitizedState = 0;
    capTouchActivePolarity = 0;
    emitCapTouchDigitizedEvent(0, currentTime);
  }
}

void processAnalogInput() {
  if (!analogInputEnabled) {
    return;
  }

  uint64_t currentTime = getCurrentTimestamp();
  if (currentTime - lastAnalogSampleTime < analogSampleIntervalUs) {
    return;
  }

  byte channelIndex = analogInputChannel - 1;
  uint16_t analogValue = analogRead(inputPins[channelIndex]) & ANALOG_VALUE_MASK;
  lastAnalogValue = analogValue;
  lastAnalogSampleTime = currentTime;
  streamingEventPut(ANALOG_INPUT_EVENT_BASE + channelIndex, packAnalogSample(currentTime, analogValue));
}

void processSpeakerNoise() {
  if (!speakerNoiseEnabled) {
    return;
  }

  uint64_t currentTime = getCurrentTimestamp();
  if (currentTime - lastSpeakerSampleTime < speakerSampleIntervalUs) {
    return;
  }

  lastSpeakerSampleTime = currentTime;
  speakerCurrentState = nextSpeakerNoiseValue() & 0x01;
  digitalWriteFast(SPEAKER_PIN, speakerCurrentState);
  speakerSamplesGenerated++;
}

void processOutputs() {
  OutputCommand cmd;
  uint64_t currentTime = getCurrentTimestamp();

  // Process one command per cycle to maintain timing
  if (outputQueueGet(&cmd)) {
    switch (cmd.type) {
      case 1: // Single channel
        if (cmd.channel < nOutputChannels && outputChState[cmd.channel] != cmd.state) {
          digitalWriteFast(outputPins[cmd.channel], cmd.state);
          outputChState[cmd.channel] = cmd.state;

          // Update bit representation
          if (cmd.state) {
            allOutputStates |= (1 << cmd.channel);
          } else {
            allOutputStates &= ~(1 << cmd.channel);
          }

          if (cmd.channel == (CAP_TOUCH_POWER_OUTPUT_CHANNEL - 1)) {
            resetCapTouchDigitizer(currentTime);
          }

          uint16_t eventCode = DIO_OUTPUT_EVENT_BASE + cmd.channel * 2 + cmd.state;
          streamingEventPut(eventCode, currentTime);
        }
        break;

      case 2: // All channels
        {
          uint16_t newStates = cmd.data;
          for (int i = 0; i < nOutputChannels; i++) {
            byte newState = (newStates >> i) & 1;
            if (outputChState[i] != newState) {
              digitalWriteFast(outputPins[i], newState);
              outputChState[i] = newState;

              if (i == (CAP_TOUCH_POWER_OUTPUT_CHANNEL - 1)) {
                resetCapTouchDigitizer(currentTime);
              }

              uint16_t eventCode = DIO_OUTPUT_EVENT_BASE + i * 2 + newState;
              streamingEventPut(eventCode, currentTime);
            }
          }
          allOutputStates = newStates & 0x1FFF;
        }
        break;

      case 3: // Camera control
        cameraEnabled = (cmd.state == 1);
        if (!cameraEnabled) {
          digitalWriteFast(CAMERA_PIN, LOW);
          frameActive = false;
        } else {
          // Reset frame timing when enabling camera
          frameStartTime = currentTime;
          lastFrameTime = currentTime;
        }
        break;

      case 4: // Equipment control
        if (cmd.channel == 1) { // Miniscope
          miniscopeEnabled = (cmd.state == 1);
          digitalWriteFast(MINISCOPE_PIN, cmd.state);
        } else if (cmd.channel == 2) { // Photometry
          photometryEnabled = (cmd.state == 1);
          digitalWriteFast(PHOTOMETRY_PIN, cmd.state);
        }
        break;
    }
  }
}

void processCamera() {
  if (!cameraEnabled) return;

  uint64_t currentTime = getCurrentTimestamp();

  if (!frameActive) {
    // Check if it's time for next frame
    if (currentTime - lastFrameTime >= framePeriod) {
      digitalWriteFast(CAMERA_PIN, HIGH);
      frameActive = true;
      frameStartTime = currentTime;
      lastFrameTime = currentTime;
      streamingEventPut(CAMERA_EVENT,currentTime);
    }
  } else {
    // Check if exposure time has elapsed
    if (currentTime - frameStartTime >= exposureTime) {
      digitalWriteFast(CAMERA_PIN, LOW);
      frameActive = false;
    }
  }
}

void processTimingHeartbeat() {
  if (!streamingEnabled || nextTimingHeartbeatUs == 0) {
    return;
  }

  uint64_t currentTime = getCurrentTimestamp();
  if (currentTime < nextTimingHeartbeatUs) {
    return;
  }

  // Emit the actual Teensy time. If execution skipped more than one period,
  // advance to the next future deadline rather than creating a catch-up burst;
  // host QC will identify the missing period from the timestamp gap.
  streamingEventPut(DIAG_TIMING_HEARTBEAT, currentTime);
  do {
    nextTimingHeartbeatUs += TIMING_HEARTBEAT_PERIOD_US;
  } while (nextTimingHeartbeatUs <= currentTime);
}

// ========== SERIAL COMMUNICATION ==========

void readSerial1Data() {
  const int maxReadsPerCall = 64; // Constant limit benchmark this
  int readsRemaining = maxReadsPerCall;

  while (Serial1COM.available() && readsRemaining-- > 0) {
    if (debugMode) {
      debugCounters.serial1Bytes++;
    }

    byte data = Serial1COM.readByte();
    if (!serial1BufferPut(data)) {
      streamingEventPut(DIAG_SERIAL1_OVERFLOW, getCurrentTimestamp());
      if (debugMode) {
        debugCounters.bufferOverflows++;
      }
      break; // Buffer full
    }
  }
}

void executeCommand(byte opCode, byte* params, byte paramCount) {
  uint64_t currentTime = getCurrentTimestamp();

  switch (opCode) {
    case 255: // Module info
      returnModuleInfo(Serial1COM);
      break;

    case 1 ... 13: // Output channels
      if (paramCount >= 1) {
        outputQueuePut(1, opCode - 1, params[0]);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'E': // Enable input
      if (paramCount >= 2 && params[0] > 0 && params[0] <= nInputChannels) {
        inputsEnabled[params[0] - 1] = params[1];
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'A': // All outputs
      if (paramCount >= 2) {
        uint16_t newStates = params[0] | ((uint16_t)params[1] << 8);
        outputQueuePut(2, 0, 0, newStates);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'C': // Camera
      if (paramCount >= 1) {
        outputQueuePut(3, 0, params[0]);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'Q': // Camera frequency
      if (paramCount >= 4) {
        uint32_t freq = *(uint32_t*)params;
        cameraFrequency = freq;
        if (freq > 0) {
          framePeriod = 1000000 / freq;
        }
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'X': // Exposure time
      if (paramCount >= 4) {
        exposureTime = *(uint32_t*)params;
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'M': // Miniscope
      if (paramCount >= 1) {
        outputQueuePut(4, 1, params[0]);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'P': // Photometry
      if (paramCount >= 1) {
        outputQueuePut(4, 2, params[0]);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'K': // Speaker white noise
      if (paramCount >= 5) {
        byte enabled = params[0];
        uint32_t interval = *(uint32_t*)&params[1];
        speakerGateState = 0;
        speakerSampleIntervalUs = max((uint32_t)CYCLE_TIME_MICROS, interval);
        speakerNoiseEnabled = (enabled == 1);
        lastSpeakerSampleTime = 0;
        if (!speakerNoiseEnabled) {
          speakerCurrentState = 0;
          digitalWriteFast(SPEAKER_PIN, LOW);
        }
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'Y': // Manual speaker gate on pin 27
      if (paramCount >= 1) {
        speakerNoiseEnabled = false;
        speakerGateState = params[0] ? 1 : 0;
        speakerCurrentState = speakerGateState;
        digitalWriteFast(SPEAKER_PIN, speakerGateState ? HIGH : LOW);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'S': // Bpod state
      if (paramCount >= 1 && params[0] != currentBpodState) {
        currentBpodState = params[0];
        streamingEventPut(BPOD_STATE_EVENT_BASE + params[0], currentTime);
      } else if (paramCount < 1) {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'B': // BNC output
      if (paramCount >= 2 && params[0] >= 1 && params[0] <= 2) {
        uint16_t eventCode = BPOD_BNC_OUT_EVENT_BASE + (params[0] - 1) * 2 + params[1];
        streamingEventPut(eventCode, currentTime);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'V': // Valve
      if (paramCount >= 2 && params[0] >= 1 && params[0] <= 5) {
        uint16_t eventCode = BPOD_VALVE_EVENT_BASE + (params[0] - 1) * 2 + params[1];
        streamingEventPut(eventCode, currentTime);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'L': // LED PWM
      if (paramCount >= 2 && params[0] >= 1 && params[0] <= 5) {
        uint16_t eventCode = BPOD_LED_EVENT_BASE + (params[0] - 1) * 256 + params[1];
        streamingEventPut(eventCode, currentTime);
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    case 'I': // Input events
      if (paramCount >= 3) {
        if (params[0] == 'B' && params[1] >= 1 && params[1] <= 2) {
          uint16_t eventCode = BPOD_BNC_IN_EVENT_BASE + (params[1] - 1) * 2 + params[2];
          streamingEventPut(eventCode, currentTime);
        } else if (params[0] == 'P' && params[1] >= 1 && params[1] <= 5) {
          uint16_t eventCode = BPOD_PORT_IN_EVENT_BASE + (params[1] - 1) * 2 + params[2];
          streamingEventPut(eventCode, currentTime);
        } else {
          streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
        }
      } else {
        streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      }
      break;

    default:
      streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, currentTime);
      break;
  }
}


void processSerial1Commands() {
  static byte commandState = 0; // 0=waiting for opcode, 1-N=reading parameters
  static byte opCode = 0;
  static byte paramCount = 0;
  static byte params[8];
  static byte expectedParams = 0;

  const int maxProcessedPerCall = 64; // Constant limit
  int processedRemaining = maxProcessedPerCall;
  byte data;

  while (processedRemaining-- > 0 && serial1BufferGet(&data)) {
    if (commandState == 0) {
      opCode = data;
      paramCount = 0;
      expectedParams = getExpectedParams(opCode);

      if (expectedParams == 0) {
        executeCommand(opCode, params, paramCount);
        commandState = 0;
      } else {
        commandState = 1;
      }
    } else {
      params[paramCount++] = data;

      if (paramCount >= expectedParams) {
        executeCommand(opCode, params, paramCount);
        commandState = 0;
      }
    }
  }
}

// Fast lookup table for parameter counts
inline byte getExpectedParams(byte opCode) {
  // Handle channel opcodes (1 to nOutputChannels)
  if (opCode >= 1 && opCode <= nOutputChannels) {
    return 1;
  }

  // Handle specific command opcodes
  switch (opCode) {
    case 255: return 0;
    case 'E': return 2;
    case 'A': return 2;
    case 'C': return 1;
    case 'M': return 1;
    case 'P': return 1;
    case 'K': return 5;
    case 'Y': return 1;
    case 'Q': return 4;
    case 'X': return 4;
    case 'S': return 1;
    case 'B': return 2;
    case 'V': return 2;
    case 'L': return 2;
    case 'I': return 3;
    default: return 0; // Unknown command
  }
}

void sendCriticalEvents() {
  const int maxSendsPerCall = 32; // Constant limit
  int sendsRemaining = maxSendsPerCall;
  byte eventCode;

  while (sendsRemaining-- > 0 && criticalEventGet(&eventCode)) {
    Serial1COM.writeByte(eventCode);
  }
}

// ========== USB COMMUNICATION ==========

void handleUSB1Communication() {
  // Phase 1: Start new commands or continue existing ones
  if (!usb1CmdState.inProgress && PC1.available()) {
    byte cmd = PC1.readByte();
    byte bytesNeeded = getUSBCommandBytes(cmd);

    if (bytesNeeded == 0) {
      // Immediate command - execute right away
      executeImmediateUSB1Command(cmd);
    } else {
      // Multi-byte command - set up state machine
      usb1CmdState.inProgress = true;
      usb1CmdState.commandCode = cmd;
      usb1CmdState.bytesNeeded = bytesNeeded;
      usb1CmdState.bytesReceived = 0;
      usb1CmdState.startTime = millis();
    }
  }

  // Phase 2: Continue multi-byte commands
  if (usb1CmdState.inProgress) {
    processUSB1MultiByteCommand();
  }

  // Phase 3: Handle USB1 discovery when no active communication
  if (!usb1CmdState.inProgress && !PC1.available()) {
    handleUSB1Discovery();
  }
}

void handleUSB1Discovery() {
  static uint32_t PC1_discovery_counter = 0;
  if (!PC1Connected) {
    if (PC1_discovery_counter == CYCLES_PER_MS * 100) {
      PC1.writeByte(configDiscoveryByte);
      PC1_discovery_counter = 0;
    } else {
      PC1_discovery_counter++;
    }
  }
}

void processUSB1MultiByteCommand() {
  // Collect available bytes (non-blocking)
  while (PC1.available() && usb1CmdState.bytesReceived < usb1CmdState.bytesNeeded) {
    usb1CmdState.buffer[usb1CmdState.bytesReceived] = PC1.readByte();
    usb1CmdState.bytesReceived++;
  }

  // Check if command is complete
  if (usb1CmdState.bytesReceived >= usb1CmdState.bytesNeeded) {
    executeCompleteUSB1Command();
    usb1CmdState.inProgress = false; // Reset state
    return;
  }

  // Check for timeout
  if (millis() - usb1CmdState.startTime > usb1CmdState.timeoutMs) {
    streamingEventPut(DIAG_USB1_TIMEOUT, getCurrentTimestamp());
    PC1.writeByte(0); // Send failure response
    usb1CmdState.inProgress = false; // Reset state
  }
}

void executeImmediateUSB1Command(byte cmd) {
  switch (cmd) {
    case '6': // Handshake
      PC1Connected = true;
      PC1.writeByte(53);
      PC1.send_now();
      sessionStartTime = micros();
      microsOverflows = 0;
      break;

    case 'F': // Get firmware
      PC1.writeUint16(FirmwareVersion);
      break;

    case 'I': // Get module info
      returnModuleInfo(PC1);
      PC1.writeByte(nOutputChannels);
      break;

    case 'G': // Get camera/equipment status
      PC1.writeByte(cameraEnabled ? 1 : 0);
      PC1.writeUint32(cameraFrequency);
      PC1.writeUint32(exposureTime);
      PC1.writeByte(miniscopeEnabled ? 1 : 0);
      PC1.writeByte(photometryEnabled ? 1 : 0);
      break;

    case 'Z': // Disconnect
      PC1Connected = false;
      speakerNoiseEnabled = false;
      speakerCurrentState = 0;
      speakerGateState = 0;
      digitalWriteFast(SPEAKER_PIN, LOW);
      PC1.writeByte(1);
      break;

    case 'U': // Get debug status
      PC1.writeByte(debugMode ? 1 : 0);
      PC1.writeByte(verboseDebug ? 1 : 0);
      // Send current buffer utilization
      PC1.writeByte((serial1Buf.count * 100) / SERIAL1_BUFFER_SIZE);
      PC1.writeByte((outputQueue.count * 100) / OUTPUT_QUEUE_SIZE);
      PC1.writeByte((streamBuf.count * 100) / STREAMING_BUFFER_SIZE);
      break;

    case 'n': // Get analog input status
      PC1.writeByte(analogInputEnabled ? 1 : 0);
      PC1.writeByte(analogInputChannel);
      PC1.writeByte(inputPins[analogInputChannel - 1]);
      PC1.writeUint32(analogSampleIntervalUs);
      PC1.writeUint16(lastAnalogValue);
      break;

    case 'l': // Get cap-touch ADC digitizer status
      PC1.writeByte(capTouchDigitizerEnabled ? 1 : 0);
      PC1.writeByte(CAP_TOUCH_CHANNEL);
      PC1.writeByte(inputPins[CAP_TOUCH_CHANNEL - 1]);
      PC1.writeByte(capTouchDigitizedState);
      PC1.writeByte(capTouchActivePolarity);
      PC1.writeUint16(capTouchLastValue);
      PC1.writeUint16(capTouchBaseline);
      PC1.writeUint16(capTouchRisingHighThreshold());
      PC1.writeUint16(capTouchRisingLowThreshold(capTouchRisingHighThreshold()));
      PC1.writeUint16(capTouchFallingHighThreshold());
      PC1.writeUint16(capTouchFallingLowThreshold());
      break;

    case 'k': // Get speaker white-noise status
      PC1.writeByte(speakerNoiseEnabled ? 1 : 0);
      PC1.writeByte(SPEAKER_PIN);
      PC1.writeUint32(speakerSampleIntervalUs);
      PC1.writeUint32(speakerSamplesGenerated);
      PC1.writeByte(speakerNoiseEnabled ? speakerCurrentState : speakerGateState);
      break;

    default:
      // Unknown command - this shouldn't happen
      PC1.writeByte(0); // Send failure response
      streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, getCurrentTimestamp());
      break;

  }
}

void executeCompleteUSB1Command() {
  if (debugMode) {
    debugCounters.usb1Commands++;
  }

  switch (usb1CmdState.commandCode) {
    case 'R': // Set refractory period
      {
        uint32_t* value = (uint32_t*)usb1CmdState.buffer;
        refractoryPeriod = *value;
        PC1.writeByte(1); // Success acknowledgment
      }
      break;

    case 'E': // Enable/disable input channel
      {
        byte channel = usb1CmdState.buffer[0] - 1; // Convert to 0-based index
        byte state = usb1CmdState.buffer[1];
        if (channel < nInputChannels) {
          inputsEnabled[channel] = state;
          PC1.writeByte(1); // Success
        } else {
          PC1.writeByte(0); // Invalid channel
        }
      }
      break;

    case 'N': // Configure analog input
      {
        byte channel = usb1CmdState.buffer[0];
        byte enabled = usb1CmdState.buffer[1];
        uint32_t* interval = (uint32_t*)&usb1CmdState.buffer[2];

        if (!isAnalogCapableChannel(channel) || *interval == 0) {
          PC1.writeByte(0);
          break;
        }

        analogSampleIntervalUs = max((uint32_t)CYCLE_TIME_MICROS, *interval);
        setAnalogInputEnabled(channel, enabled == 1);
        PC1.writeByte(1);
      }
      break;

    case 't': // Configure cap-touch ADC digitizer thresholds
      {
        uint16_t* minHighThreshold = (uint16_t*)&usb1CmdState.buffer[0];
        uint16_t* deltaHigh = (uint16_t*)&usb1CmdState.buffer[2];
        uint16_t* deltaLow = (uint16_t*)&usb1CmdState.buffer[4];

        configureCapTouchDigitizer(*minHighThreshold, *deltaHigh, *deltaLow);
        PC1.writeByte(1);
      }
      break;

    case 'K': // Configure speaker white noise
      {
        byte enabled = usb1CmdState.buffer[0];
        uint32_t* interval = (uint32_t*)&usb1CmdState.buffer[1];

        speakerGateState = 0;
        speakerSampleIntervalUs = max((uint32_t)CYCLE_TIME_MICROS, *interval);
        speakerNoiseEnabled = (enabled == 1);
        lastSpeakerSampleTime = 0;
        if (!speakerNoiseEnabled) {
          speakerCurrentState = 0;
          digitalWriteFast(SPEAKER_PIN, LOW);
        }
        PC1.writeByte(1);
      }
      break;

    case 'A': // Set all outputs
      {
        byte byte1 = usb1CmdState.buffer[0];
        byte byte2 = usb1CmdState.buffer[1];
        uint16_t newOutputStates = byte1 | ((uint16_t)byte2 << 8);
        outputQueuePut(2, 0, 0, newOutputStates);
        PC1.writeByte(1); // Success
      }
      break;

    case 'O': // Set output channel
      {
        byte channel = usb1CmdState.buffer[0] - 1; // Convert to 0-based index
        byte state = usb1CmdState.buffer[1];
        if (channel < nOutputChannels) {
          outputQueuePut(1, channel, state);
          PC1.writeByte(1); // Success
        } else {
          PC1.writeByte(0); // Invalid channel
        }
      }
      break;

    case 'C': // Camera enable/disable
      {
        byte state = usb1CmdState.buffer[0];
        cameraEnabled = (state == 1);
        if (!cameraEnabled) {
          digitalWriteFast(CAMERA_PIN, LOW);
          frameActive = false;
        } else {
          // Reset frame timing when enabling camera
          frameStartTime = getCurrentTimestamp();
          lastFrameTime = getCurrentTimestamp();
        }
        PC1.writeByte(1); // Success acknowledgment
      }
      break;

    case 'Q': // Set camera frequency
      {
        uint32_t* freq = (uint32_t*)usb1CmdState.buffer;
        cameraFrequency = *freq;
        if (cameraFrequency > 0) {
          framePeriod = 1000000 / cameraFrequency;
        }
        PC1.writeByte(1); // Success acknowledgment
      }
      break;

    case 'X': // Set exposure time
      {
        uint32_t* exposure = (uint32_t*)usb1CmdState.buffer;
        exposureTime = *exposure;
        PC1.writeByte(1); // Success acknowledgment
      }
      break;

    case 'M': // Miniscope control
      {
        byte state = usb1CmdState.buffer[0];
        miniscopeEnabled = (state == 1);
        digitalWriteFast(MINISCOPE_PIN, state);
        PC1.writeByte(1); // Success acknowledgment
      }
      break;

    case 'P': // Photometry control
      {
        byte state = usb1CmdState.buffer[0];
        photometryEnabled = (state == 1);
        digitalWriteFast(PHOTOMETRY_PIN, state);
        PC1.writeByte(1); // Success acknowledgment
      }
      break;

    case 'D': // Debug mode control
      {
        byte mode = usb1CmdState.buffer[0];
        debugMode = (mode & 0x01) != 0;
        verboseDebug = (mode & 0x02) != 0;
        if (!debugMode) {
          digitalWriteFast(cycle_pin, 0);
        }
        PC1.writeByte(1); // Success acknowledgment
      }
      break;

    default:
      // Unknown multi-byte command - this shouldn't happen if getUSBCommandBytes is correct
      PC1.writeByte(0); // Send failure response
      streamingEventPut(DIAG_SERIAL1_PARSE_ERROR, getCurrentTimestamp());
      break;
  }
}

void handleUSB2Communication() {

  if (PC2.available()) {
    byte cmd = PC2.readByte();
    if (debugMode) {
      debugCounters.usb2Commands++;
    }

    switch (cmd) {
      case '6': // Handshake
        PC2Connected = true;
        PC2.writeByte(53);
        PC2.send_now();
        break;

      case 'S': // Start streaming
        // CLEAR ALL BUFFERS before starting stream
        streamBuf.head = 0;
        streamBuf.tail = 0;
        streamBuf.count = 0;

        // Clear the actual buffer memory to prevent corruption
        memset((void*)streamBuf.events, 0, sizeof(streamBuf.events));

        // Set streaming flag and schedule the first beat one full period after
        // the stream-start acknowledgement.
        streamingEnabled = true;
        nextTimingHeartbeatUs = getCurrentTimestamp() + TIMING_HEARTBEAT_PERIOD_US;
        // Send acknowledgment

        PC2.writeByte(1);
        break;

      case 'Y': // Stop streaming
        streamingEnabled = false;
        nextTimingHeartbeatUs = 0;
        PC1.writeByte(1); // write stop acknowledge to PC1 to not interfere with remaining PC2 streaming events.
        break;

      case 'T': // Get current timestamp
        {
          uint64_t currentTime = getCurrentTimestamp();
          PC2.writeByteArray((byte*)&currentTime, 8);
        }
        break;

      case 'Z': // Disconnect
        PC2Connected = false;
        streamingEnabled = false;
        nextTimingHeartbeatUs = 0;
        PC2.writeByte(1);
        break;

      default:
        // Unknown command, ignore for streaming interface
        break;
    }
  } else {
    // Handle USB2 discovery
    static uint32_t PC2_discovery_counter = 0;
    if (!PC2Connected){
       if (PC2_discovery_counter == CYCLES_PER_MS * 100) {
        PC2.writeByte(streamingDiscoveryByte);
        PC2_discovery_counter=0;
       }
      PC2_discovery_counter++;
    }
  }

  // Send streaming events
  if (streamingEnabled && PC2Connected) {
    sendStreamingData();
  }
}

void sendStreamingData() {
  // Transfer all complete messages from streamBuf that fit (with all bytes) into USB buffer
  const int bytesPerEvent = 10; // 2 bytes (uint16_t) + 8 bytes (uint64_t)
  const int maxEventsPerCall = PC2.availableForWrite()/ bytesPerEvent;
  int eventsSent = 0;
  StreamingEvent event;

  uint16_t eventsToSend = min(streamBuf.count, maxEventsPerCall);
  if (eventsToSend == 0) return;

  PC2.writeByte(eventsToSend);

  while (eventsSent < eventsToSend && streamingEventGet(&event)) { // important to first check if
    PC2.writeUint16(event.eventCode);
    PC2.writeByteArray((byte*)&event.timestamp, 8);
    eventsSent++;
  }
  PC2.send_now();
}

void handleUSBCommunication() {

  handleUSB1Communication();
  handleUSB2Communication();

}

void returnModuleInfo(ArCOM& serialPort) {
  serialPort.writeByte('A'); // Acknowledge
  serialPort.writeUint32(FirmwareVersion); // 4-byte firmware version
  serialPort.writeByte(sizeof(moduleName)-1);
  serialPort.writeCharArray(moduleName, sizeof(moduleName)-1); // Module name
  serialPort.writeByte(1); // 1 if more info follows, 0 if not
  serialPort.writeByte('#'); // Op code for: Number of behavior events this module can generate
  serialPort.writeByte(nInputChannels*2); // 2 states for each input channel
  serialPort.writeByte(1); // 1 if more info follows, 0 if not
  serialPort.writeByte('E'); // Op code for: Behavior event names
  serialPort.writeByte(nEventNames);
  for (int i = 0; i < nEventNames; i++) { // Once for each event name
    serialPort.writeByte(strlen(eventNames[i])); // Send event name length
    for (uint j = 0; j < strlen(eventNames[i]); j++) { // Once for each character in this event name
      serialPort.writeByte(*(eventNames[i]+j)); // Send the character
    }
  }
  serialPort.writeByte(0); // 1 if more info follows, 0 if not
}

void storeCycleStats(uint32_t cycleDuration) {
  cycleStats.durations[cycleStats.index] = cycleDuration;

  cycleStats.index = (cycleStats.index + 1) % 100;
  if (cycleStats.count < 100) {
    cycleStats.count++;
  }
}

void updateComputeTime(uint32_t computeTime) {
  if (cycleStats.count > 0) {
    uint8_t currentIndex = (cycleStats.index + 99) % 100; // Previous index
    cycleStats.computeTimes[currentIndex] = computeTime;
  }
}

void calculateCycleStatistics(uint32_t* meanDuration, uint32_t* minDuration, uint32_t* maxDuration,
                             uint32_t* meanCompute, uint32_t* minCompute, uint32_t* maxCompute) {
  if (cycleStats.count == 0) {
    *meanDuration = *minDuration = *maxDuration = 0;
    *meanCompute = *minCompute = *maxCompute = 0;
    return;
  }

  uint32_t sumDuration = 0, sumCompute = 0;
  *minDuration = *minCompute = 0xFFFFFFFF;
  *maxDuration = *maxCompute = 0;

  for (uint8_t i = 0; i < cycleStats.count; i++) {
    // Duration stats
    sumDuration += cycleStats.durations[i];
    if (cycleStats.durations[i] < *minDuration) *minDuration = cycleStats.durations[i];
    if (cycleStats.durations[i] > *maxDuration) *maxDuration = cycleStats.durations[i];

    // Compute time stats
    sumCompute += cycleStats.computeTimes[i];
    if (cycleStats.computeTimes[i] < *minCompute) *minCompute = cycleStats.computeTimes[i];
    if (cycleStats.computeTimes[i] > *maxCompute) *maxCompute = cycleStats.computeTimes[i];
  }

  *meanDuration = sumDuration / cycleStats.count;
  *meanCompute = sumCompute / cycleStats.count;
}

void processDebugReporting() {
  uint32_t currentTime = millis();

  // Report cycle statistics every 10ms
  if (currentTime - cycleStats.lastReportTime >= 10) {
    uint32_t meanDur, minDur, maxDur, meanComp, minComp, maxComp;
    calculateCycleStatistics(&meanDur, &minDur, &maxDur, &meanComp, &minComp, &maxComp);

    // Send as structured debug event
    sendDebugCycleStats(meanDur, minDur, maxDur, meanComp, minComp, maxComp);
    cycleStats.lastReportTime = currentTime;
  }

  // Report other statistics every 100ms
  if (verboseDebug && (currentTime - debugCounters.lastReportTime >= 10)) {
    sendDebugBufferStats();
    sendDebugPerformanceStats();
    debugCounters.lastReportTime = currentTime;
  }
}

void sendDebugCycleStats(uint32_t meanDur, uint32_t minDur, uint32_t maxDur,
                        uint32_t meanComp, uint32_t minComp, uint32_t maxComp) {
  // Create a structured debug event
  // You'll need to modify your streaming protocol to handle multi-part events
  streamingEventPut(DIAG_CYCLE_STATS, (uint64_t)meanDur << 32 | minDur);
  streamingEventPut(DIAG_CYCLE_STATS+1, (uint64_t)maxDur << 32 | meanComp);
  streamingEventPut(DIAG_CYCLE_STATS+2, (uint64_t)minComp << 32 | maxComp);
}

void sendDebugBufferStats() {
  uint16_t serial1Util = (serial1Buf.count * 1000) / SERIAL1_BUFFER_SIZE;
  uint16_t outputUtil = (outputQueue.count * 1000) / OUTPUT_QUEUE_SIZE;
  uint16_t streamUtil = ((streamBuf.count-3) * 1000) / STREAMING_BUFFER_SIZE;// subtract the just put 3 sendDebugCycleStats

  // Encode multiple values into event data
  streamingEventPut(DIAG_BUFFER_STATS , (uint64_t)serial1Util << 32 | outputUtil);
  streamingEventPut(DIAG_BUFFER_STATS + 1, (uint64_t)streamUtil << 32 | debugCounters.bufferOverflows);
}

void sendDebugPerformanceStats() {
  uint32_t currentTime = millis();
  uint32_t timeDiff = currentTime - debugCounters.lastReportTime;

  if (timeDiff > 0) {
    uint32_t usb1Rate = (debugCounters.usb1Commands * 1000) / timeDiff;
    uint32_t usb2Rate = (debugCounters.usb2Commands * 1000) / timeDiff;
    uint32_t serial1Rate = (debugCounters.serial1Bytes * 1000) / timeDiff;

    streamingEventPut(DIAG_PERFORMANCE_STATS, (uint64_t)usb1Rate << 32 | usb2Rate);
    streamingEventPut(DIAG_PERFORMANCE_STATS + 1, (uint64_t)serial1Rate << 32 | cycleCount);

    // Reset counters
    debugCounters.usb1Commands = 0;
    debugCounters.usb2Commands = 0;
    debugCounters.serial1Bytes = 0;
  }
}

// void adjustProcessingLimits(uint32_t computeTime) {
//   static uint32_t adjustCounter = 0;
//   adjustCounter++;

//   // Adjust every 100 cycles (10ms)
//   if (adjustCounter % 100 == 0) {
//     if (computeTime < 50) { // Under 50µs - increase limits
//       limits.maxReads = min(32, limits.maxReads + 2);
//       limits.maxSends = min(8, limits.maxSends + 1);
//       limits.maxProcessed = min(64, limits.maxProcessed + 4);
//     } else if (computeTime > 80) { // Over 80µs - decrease limits
//       limits.maxReads = max(8, limits.maxReads - 2);
//       limits.maxSends = max(2, limits.maxSends - 1);
//       limits.maxProcessed = max(16, limits.maxProcessed - 4);
//     }
//   }
// }

// ========== SETUP AND MAIN LOOP ==========

void setup() {
  Serial1.begin(1312500); // ArCOM does not wrap the begin function -> sart up directly
  analogReadResolution(12);
  analogReadAveraging(1);
  // Initialize session start time
  sessionStartTime = micros();

  // Send system reset diagnostic event
  streamingEventPut(DIAG_SYSTEM_RESET, (uint64_t)0);

  // Initialize pins
  for (int i = 0; i < nInputChannels; i++) {
    if ((i + 1) == CAP_TOUCH_CHANNEL) {
      pinMode(inputPins[i], INPUT);
    } else {
      pinMode(inputPins[i], INPUT_PULLUP);
    }
    inputsEnabled[i] = 1;
    inputChState[i] = digitalReadFast(inputPins[i]);
    lastInputChState[i] = inputChState[i];
  }
  for (int i = 0; i < nOutputChannels; i++) {
    pinMode(outputPins[i], OUTPUT);
    digitalWriteFast(outputPins[i], LOW);
    outputChState[i] = 0;
  }

  pinMode(CAMERA_PIN, OUTPUT);
  pinMode(MINISCOPE_PIN, OUTPUT);
  pinMode(PHOTOMETRY_PIN, OUTPUT);
  pinMode(SPEAKER_PIN, OUTPUT);
  if (LED_pin >= 0) {
    pinMode(LED_pin, OUTPUT);
  }
  pinMode(cycle_pin, OUTPUT);
  pinMode(USB_send_pin, OUTPUT);

  digitalWriteFast(CAMERA_PIN, LOW);
  digitalWriteFast(MINISCOPE_PIN, LOW);
  digitalWriteFast(PHOTOMETRY_PIN, LOW);
  digitalWriteFast(SPEAKER_PIN, LOW);
  setStatusLed(LOW);

  // Calculate initial frame period
  if (cameraFrequency > 0) {
    framePeriod = 1000000 / cameraFrequency;
  }

  // Start the precise cycle timer
  cycleTimer.begin(cycleTimerISR, CYCLE_TIME_MICROS);
  cycleTimer.priority(64);
}

void cycleTimerISR() {
  // noInterrupts()
  digitalWriteFast(cycle_pin, 1);
  cycleCount++;
  static uint64_t lastCycleTime = 0;// Static means, that the variable is initialized to 0 only once when the program starts
  uint64_t  cycleStartTime = getCurrentTimestamp();

    // Turn off LED state at the start of each cycle this will lead to lower brightness representing higher computational load
  if (LED_state) {
      setStatusLed(0);
  }

  uint32_t stream_available_for_write = PC2.availableForWrite();

  if (debugMode) {
    // digitalWriteFast(cycle_pin, 1);
    streamingEventPut(DIAG_DEBUG_BASE, cycleStartTime);
    streamingEventPut(DIAG_TEENSY_USB_BUFFER_FREE+1, (uint64_t)stream_available_for_write);
  } else {
    if (stream_available_for_write <= 2048){
      streamingEventPut(DIAG_TEENSY_USB_BUFFER_FREE+1, (uint64_t)stream_available_for_write);
    }
  }

  // Store cycle duration from previous cycle
  if (lastCycleTime != 0) {
    uint32_t cycleDuration = cycleStartTime - lastCycleTime;
    storeCycleStats(cycleDuration); // Will update compute time later

    // Check for timing violations
    if (cycleDuration > 100) { // 0.5 µs margin
      streamingEventPut(DIAG_CYCLE_TIMING_VIOLATION, getCurrentTimestamp());
    }
    // Check for cycle count increase violation
    static uint32_t cycleCount_last = cycleCount;
    if (cycleCount > cycleCount_last+1) { // check if increased by 1
      streamingEventPut(DIAG_CYCLE_COUNT_VIOLATION, getCurrentTimestamp());
    }
    cycleCount_last = cycleCount;
  }


  // === CRITICAL TIMING SECTION ===
  // These functions must complete within 100µs

  processInputs();     // ~15-25µs (includes immediate Serial1 transmission)
  processCapTouchDigitizer();
  processAnalogInput();
  processOutputs();    // ~5-15µs
  processCamera();     // ~2-5µs
  processTimingHeartbeat();
  processSpeakerNoise();
  readSerial1Data();
  processSerial1Commands();
  sendCriticalEvents(); // Backup for any failed immediate sends

  if (PC2.available() || !PC2Connected || (streamingEnabled && streamBuf.count > 0)) {
    needsUSB2Processing = true;
  }

  handleUSBCommunication();


  // Debug reporting (if enabled)
  if (debugMode) {
    processDebugReporting();
    // digitalWriteFast(cycle_pin, 0);
  }

  // Store cycle start time for next cycle
  lastCycleTime = cycleStartTime;
  // Update compute time for current cycle
  uint64_t computeEndTime = getCurrentTimestamp();
  uint64_t computeTime = computeEndTime - cycleStartTime;
  updateComputeTime(computeTime);

  // BufferStatus bufferStatus = getBufferStatus(); // check if this is still needed
  // adjustProcessingLimits(computeTime, bufferStatus);

  // interrupts();

  // Background tasks when cycle is not active
  //else {
  if (computeTime < 10) {
    static uint32_t backgroundCounter = 0;
    backgroundCounter++;
    // Every 1000 background loops, do some maintenance
    if (backgroundCounter % 1000 == 0) {
      // Check for buffer overflows and reset if necessary
      if (serial1Buf.count >= SERIAL1_BUFFER_SIZE - 10) {
        streamingEventPut(DIAG_BUFFER_EMERGENCY_RESET, getCurrentTimestamp());
        serial1Buf.head = 0;
        serial1Buf.tail = 0;
        serial1Buf.count = 0;
      }

      if (streamBuf.count >= STREAMING_BUFFER_SIZE - 100) {
        streamingEventPut(DIAG_STREAMING_OVERFLOW, getCurrentTimestamp());
        for (int i = 0; i < 50; i++) {
          StreamingEvent dummy;
          streamingEventGet(&dummy);
        }
      }
    }

    if (!LED_state) {
      setStatusLed(1);
    }
  }



  if (debugMode) {
    streamingEventPut(DIAG_CYCLE_TIME, getCurrentTimestamp());
    streamingEventPut(DIAG_TEENSY_USB_BUFFER_FREE, (uint64_t)PC2.availableForWrite());

  }
  digitalWriteFast(cycle_pin, 0);
}

void loop() {
}
