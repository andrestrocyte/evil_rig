"""Host → Bpod softcode bridge (wheel / session supervisor → FSM)."""

# Trial supervision
QUIESCENCE_OK = 0
CHOICE_CW = 1
CHOICE_CCW = 2
REWARD_MATCHED = 3
REWARD_OPPOSITE = 4
SESSION_ABORT = 5
