from hf_dm.frame2ttl import (
    decode_frame2ttl_barcode_events,
    encode_frame2ttl_trial_code,
    frame2ttl_barcode_states,
)


def _events_from_states(states: list[bool], *, frame_period_s: float, inverted: bool = False):
    events = []
    previous: bool | None = True if inverted else False
    for index, state in enumerate(states):
        out_state = not state if inverted else state
        if previous is None or out_state != previous:
            t = index * frame_period_s
            events.append(
                {
                    "state": int(out_state),
                    "timestamp_us": int(t * 1_000_000),
                    "relative_time_s": t,
                    "host_perf_counter_s": t,
                }
            )
        previous = out_state
    final_state = False if not inverted else True
    if previous != final_state:
        t = len(states) * frame_period_s
        events.append(
            {
                "state": int(final_state),
                "timestamp_us": int(t * 1_000_000),
                "relative_time_s": t,
                "host_perf_counter_s": t,
            }
        )
    return events


def test_frame2ttl_trial_barcode_round_trips():
    code = encode_frame2ttl_trial_code(
        trial_index=37,
        star_side="right",
        trial_mode="forced",
        forced_shape="circle",
        modality="risk",
    )
    states = frame2ttl_barcode_states(code)
    events = _events_from_states(states, frame_period_s=1 / 60)

    decoded = decode_frame2ttl_barcode_events(events, frame_period_s=1 / 60)

    assert code.bits == "M2K6"
    assert decoded["decoded"]
    assert decoded["word"] == code.word
    assert decoded["checksum_ok"]
    assert decoded["pulse_groups"] == [2, 6]
    assert decoded["modality_pulse_count"] == 2
    assert decoded["kind_pulse_count"] == 6
    assert decoded["fields"]["star_side"] == "right"
    assert decoded["fields"]["trial_mode"] == "forced"
    assert decoded["fields"]["forced_shape"] == "circle"
    assert decoded["fields"]["modality"] == "risk"


def test_frame2ttl_decoder_accepts_inverted_ttl_polarity():
    code = encode_frame2ttl_trial_code(
        trial_index=5,
        star_side="left",
        trial_mode="free",
        forced_shape=None,
        modality="force",
    )
    states = frame2ttl_barcode_states(code)
    events = _events_from_states(states, frame_period_s=1 / 60, inverted=True)

    decoded = decode_frame2ttl_barcode_events(events, frame_period_s=1 / 60)

    assert decoded["decoded"]
    assert decoded["inverted"]
    assert decoded["word"] == code.word
    assert decoded["pulse_groups"] == [1, 1]


def test_frame2ttl_count_barcode_stays_short_and_covers_trial_kinds():
    cases = [
        ("force", "free", None, "left", [1, 1]),
        ("force", "free", None, "right", [1, 2]),
        ("risk", "forced", "star", "left", [2, 3]),
        ("risk", "forced", "star", "right", [2, 4]),
        ("delay", "forced", "circle", "left", [3, 5]),
        ("delay", "forced", "circle", "right", [3, 6]),
    ]
    for modality, trial_mode, forced_shape, star_side, expected_groups in cases:
        code = encode_frame2ttl_trial_code(
            trial_index=99,
            star_side=star_side,
            trial_mode=trial_mode,
            forced_shape=forced_shape,
            modality=modality,
        )
        states = frame2ttl_barcode_states(code)
        events = _events_from_states(states, frame_period_s=1 / 60)
        decoded = decode_frame2ttl_barcode_events(events, frame_period_s=1 / 60)

        assert len(states) <= 156
        assert decoded["decoded"]
        assert decoded["pulse_groups"] == expected_groups
        assert decoded["word"] == code.word


def test_frame2ttl_decoder_rejects_data_pulses_without_preamble():
    frame_period_s = 1 / 60
    # Two otherwise valid-looking pulse groups, but no 12-frame preamble.
    states = (
        [True] * 6
        + [False] * 9
        + [True] * 6
        + [False] * 6
        + [True] * 6
        + [False] * 9
    )
    events = _events_from_states(states, frame_period_s=frame_period_s)

    decoded = decode_frame2ttl_barcode_events(events, frame_period_s=frame_period_s)

    assert not decoded["decoded"]
    assert decoded["reason"] == "preamble_not_found"
