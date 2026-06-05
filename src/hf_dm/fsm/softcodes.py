"""SoftCode contract between the host task and the bpod-core trial FSM."""

# Host -> Bpod input events
CHOICE_HIGH = 0
CHOICE_LOW = 1
QUIESCENCE_MOVEMENT = 2
SESSION_ABORT = 3

# Bpod -> host output actions
STATE_QUIESCENCE = 0
STATE_CHOICE = 1
STATE_BLANK = 2
STATE_PUMP_ON = 3
STATE_PUMP_OFF = 4
STATE_TIMEOUT_ON = 5
STATE_TIMEOUT_OFF = 6
STATE_ABORT = 7

# Compatibility aliases used by older analysis/tests.
CHOICE_CW = CHOICE_HIGH
CHOICE_CCW = CHOICE_LOW
