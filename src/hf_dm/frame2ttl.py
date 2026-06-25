"""Frame2TTL sync patch barcode helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PROTOCOL_VERSION = 2
TOTAL_BITS = 2  # Two pulse-count groups: modality, then trial kind.
TRIAL_INDEX_MASK = 0
TRIAL_TYPE_MASK = 0xF
CHECKSUM_MASK = 0

MODALITY_PULSE_COUNTS = {"force": 1, "risk": 2, "delay": 3, "other": 4}
MODALITY_BY_PULSE_COUNT = {value: key for key, value in MODALITY_PULSE_COUNTS.items()}
TRIAL_KIND_FIELDS = {
    1: {"trial_mode": "free", "star_side": "left", "forced_shape": None},
    2: {"trial_mode": "free", "star_side": "right", "forced_shape": None},
    3: {"trial_mode": "forced", "star_side": "left", "forced_shape": "star"},
    4: {"trial_mode": "forced", "star_side": "right", "forced_shape": "star"},
    5: {"trial_mode": "forced", "star_side": "left", "forced_shape": "circle"},
    6: {"trial_mode": "forced", "star_side": "right", "forced_shape": "circle"},
}


@dataclass(frozen=True)
class Frame2TTLTrialCode:
    protocol_version: int
    trial_index: int
    trial_index_mod: int
    trial_type_code: int
    checksum: int
    word: int
    bits: str
    fields: dict[str, Any]
    modality_pulse_count: int
    kind_pulse_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frame2ttl_trial_type_code(
    *,
    star_side: str | None,
    trial_mode: str,
    forced_shape: str | None,
    modality: str,
) -> int:
    """Return the trial-kind pulse count for the robust Frame2TTL barcode."""
    del modality
    side = "right" if star_side == "right" else "left"
    mode = "forced" if trial_mode == "forced" else "free"
    if mode == "free":
        return 2 if side == "right" else 1
    shape = "circle" if forced_shape == "circle" else "star"
    if shape == "star":
        return 4 if side == "right" else 3
    return 6 if side == "right" else 5


def decode_trial_type_code(code: int) -> dict[str, Any]:
    return dict(TRIAL_KIND_FIELDS.get(int(code), {}))


def frame2ttl_checksum(protocol_version: int, trial_index_mod: int, trial_type_code: int) -> int:
    del protocol_version, trial_index_mod, trial_type_code
    return 0


def frame2ttl_modality_pulse_count(modality: str) -> int:
    return MODALITY_PULSE_COUNTS.get(str(modality or "force").lower(), 4)


def encode_frame2ttl_trial_code(
    *,
    trial_index: int,
    star_side: str | None,
    trial_mode: str,
    forced_shape: str | None,
    modality: str,
) -> Frame2TTLTrialCode:
    trial_index_i = int(trial_index)
    trial_index_mod = 0
    kind_count = frame2ttl_trial_type_code(
        star_side=star_side,
        trial_mode=trial_mode,
        forced_shape=forced_shape,
        modality=modality,
    )
    modality_count = frame2ttl_modality_pulse_count(modality)
    checksum = 0
    word = (int(modality_count) << 4) | int(kind_count)
    bits = f"M{modality_count}K{kind_count}"
    fields = dict(decode_trial_type_code(kind_count))
    fields["modality"] = MODALITY_BY_PULSE_COUNT.get(modality_count, "other")
    return Frame2TTLTrialCode(
        protocol_version=PROTOCOL_VERSION,
        trial_index=trial_index_i,
        trial_index_mod=trial_index_mod,
        trial_type_code=kind_count,
        checksum=checksum,
        word=word,
        bits=bits,
        fields=fields,
        modality_pulse_count=modality_count,
        kind_pulse_count=kind_count,
    )


def frame2ttl_barcode_states(
    code: Frame2TTLTrialCode | str,
    *,
    preamble_high_frames: int = 12,
    preamble_low_frames: int = 12,
    one_high_frames: int = 6,
    zero_high_frames: int = 6,
    inter_bit_low_frames: int = 6,
) -> list[bool]:
    """Return one sync-patch state per video frame for a two-group count barcode."""
    del zero_high_frames
    groups = _barcode_groups(code)
    states: list[bool] = []
    states.extend([True] * max(1, int(preamble_high_frames)))
    states.extend([False] * max(1, int(preamble_low_frames)))
    for group_index, count in enumerate(groups):
        for pulse_index in range(max(0, int(count))):
            states.extend([True] * max(1, int(one_high_frames)))
            is_last_pulse = pulse_index == int(count) - 1
            gap_frames = (
                preamble_low_frames
                if is_last_pulse
                else inter_bit_low_frames
            )
            states.extend([False] * max(1, int(gap_frames)))
        if int(count) <= 0 or group_index == len(groups) - 1:
            states.extend([False] * max(1, int(preamble_low_frames)))
    return states


def _barcode_groups(code: Frame2TTLTrialCode | str) -> tuple[int, int]:
    if isinstance(code, Frame2TTLTrialCode):
        return int(code.modality_pulse_count), int(code.kind_pulse_count)
    text = str(code).replace("M", "").replace("K", ",").replace("-", ",")
    parts = [part for part in text.split(",") if part.strip()]
    if len(parts) >= 2:
        return max(1, int(parts[0])), max(1, int(parts[1]))
    value = int(str(code), 0)
    return max(1, (value >> 4) & 0xF), max(1, value & 0xF)


def parse_frame2ttl_word(word: int) -> dict[str, Any]:
    word_i = int(word)
    modality_count = (word_i >> 4) & 0xF
    trial_type = word_i & 0xF
    fields = dict(decode_trial_type_code(trial_type))
    fields["modality"] = MODALITY_BY_PULSE_COUNT.get(modality_count, "other")
    valid = modality_count in MODALITY_BY_PULSE_COUNT and trial_type in TRIAL_KIND_FIELDS
    return {
        "protocol_version": PROTOCOL_VERSION,
        "trial_index_mod": 0,
        "modality_pulse_count": modality_count,
        "kind_pulse_count": trial_type,
        "trial_type_code": trial_type,
        "checksum": 0,
        "expected_checksum": 0,
        "checksum_ok": valid,
        "fields": fields,
    }


def decode_frame2ttl_barcode_events(
    events: list[dict[str, Any]],
    *,
    frame_period_s: float,
    expected_bits: int = TOTAL_BITS,
    preamble_high_frames: int = 12,
    preamble_low_frames: int = 12,
    one_high_frames: int = 6,
    zero_high_frames: int = 6,
    inter_bit_low_frames: int = 6,
    tolerance_frames: float = 2.0,
) -> dict[str, Any]:
    """Decode a two-group pulse-count barcode from DI edge events.

    The decoder uses host receipt times when available because the GUI/session needs
    immediate coherence checks. Raw StreamDIO module timestamps remain in each event
    for offline alignment.
    """
    del expected_bits, zero_high_frames
    frame_period = max(1e-6, float(frame_period_s))
    ordered = sorted(events, key=_event_time_s)
    normal = _decode_with_polarity(
        ordered,
        frame_period_s=frame_period,
        preamble_high_frames=preamble_high_frames,
        preamble_low_frames=preamble_low_frames,
        one_high_frames=one_high_frames,
        inter_bit_low_frames=inter_bit_low_frames,
        tolerance_frames=tolerance_frames,
        inverted=False,
    )
    if normal.get("decoded"):
        return normal
    inverted = _decode_with_polarity(
        ordered,
        frame_period_s=frame_period,
        preamble_high_frames=preamble_high_frames,
        preamble_low_frames=preamble_low_frames,
        one_high_frames=one_high_frames,
        inter_bit_low_frames=inter_bit_low_frames,
        tolerance_frames=tolerance_frames,
        inverted=True,
    )
    if inverted.get("decoded"):
        return inverted
    return normal if normal.get("bits") else inverted


def _decode_with_polarity(
    events: list[dict[str, Any]],
    *,
    frame_period_s: float,
    preamble_high_frames: int,
    preamble_low_frames: int,
    one_high_frames: int,
    inter_bit_low_frames: int,
    tolerance_frames: float,
    inverted: bool,
) -> dict[str, Any]:
    pulses = _high_pulses(events, inverted=inverted)
    pulse_frames: list[float] = []
    preamble_index: int | None = None
    preamble_fallback = False
    for index, pulse in enumerate(pulses):
        frames = pulse["duration_s"] / frame_period_s
        if abs(frames - preamble_high_frames) <= tolerance_frames or (
            frames >= (float(preamble_high_frames) + float(one_high_frames)) / 2.0
        ):
            preamble_index = index
            break
    if preamble_index is None:
        if pulses:
            preamble_index = 0
            preamble_fallback = True
        else:
            return {
                "decoded": False,
                "reason": "preamble_not_found",
                "inverted": inverted,
                "edge_count": len(events),
                "bits": "",
                "pulse_frames": [],
            }
    if preamble_index is None:
        return {
            "decoded": False,
            "reason": "preamble_not_found",
            "inverted": inverted,
            "edge_count": len(events),
            "bits": "",
            "pulse_frames": [],
        }
    separator_threshold_frames = (
        float(preamble_low_frames) + float(inter_bit_low_frames)
    ) / 2.0
    groups: list[int] = []
    current_count = 0
    for pulse_index, pulse in enumerate(pulses[preamble_index + 1 :], start=preamble_index + 1):
        frames = pulse["duration_s"] / frame_period_s
        if abs(frames - float(one_high_frames)) > tolerance_frames:
            continue
        current_count += 1
        pulse_frames.append(frames)
        next_pulse = pulses[pulse_index + 1] if pulse_index + 1 < len(pulses) else None
        if next_pulse is None:
            groups.append(current_count)
            break
        gap_frames = (
            (next_pulse["start_time_s"] - pulse["end_time_s"]) / frame_period_s
        )
        if gap_frames >= separator_threshold_frames:
            groups.append(current_count)
            current_count = 0
            if len(groups) >= 2:
                break
    if current_count and len(groups) < 2:
        groups.append(current_count)
    if len(groups) < 2:
        return {
            "decoded": False,
            "reason": "not_enough_groups",
            "inverted": inverted,
            "edge_count": len(events),
            "bits": "",
            "pulse_groups": groups,
            "pulse_frames": pulse_frames,
            "preamble_pulse": pulses[preamble_index],
            "preamble_fallback": preamble_fallback,
        }
    modality_count, kind_count = int(groups[0]), int(groups[1])
    word = (modality_count << 4) | kind_count
    parsed = parse_frame2ttl_word(word)
    first_edge = events[0] if events else None
    return {
        "decoded": True,
        "inverted": inverted,
        "edge_count": len(events),
        "bits": f"M{modality_count}K{kind_count}",
        "word": word,
        "pulse_groups": [modality_count, kind_count],
        "pulse_frames": pulse_frames,
        "preamble_pulse": pulses[preamble_index],
        "preamble_fallback": preamble_fallback,
        "first_event": first_edge,
        **parsed,
    }


def _high_pulses(events: list[dict[str, Any]], *, inverted: bool) -> list[dict[str, Any]]:
    pulses: list[dict[str, Any]] = []
    high_start: dict[str, Any] | None = None
    for event in events:
        state = int(event.get("state", event.get("value", 0)))
        if inverted:
            state = 0 if state else 1
        if state and high_start is None:
            high_start = event
        elif not state and high_start is not None:
            start_t = _event_time_s(high_start)
            end_t = _event_time_s(event)
            if end_t > start_t:
                pulses.append(
                    {
                        "start_event": high_start,
                        "end_event": event,
                        "start_time_s": start_t,
                        "end_time_s": end_t,
                        "duration_s": end_t - start_t,
                    }
                )
            high_start = None
    return pulses


def _event_time_s(event: dict[str, Any]) -> float:
    for key in ("relative_time_s", "timestamp_us", "host_perf_counter_s", "monotonic_s"):
        if key in event and event[key] is not None:
            if key == "timestamp_us":
                return float(event[key]) / 1_000_000.0
            return float(event[key])
    return 0.0
