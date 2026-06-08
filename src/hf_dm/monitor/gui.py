"""Lightweight operator GUI fed by nonblocking telemetry messages."""

from __future__ import annotations

import ctypes
import logging
import math
import multiprocessing as mp
import queue
import signal
import time
from collections import Counter
from collections import deque
from dataclasses import dataclass
from typing import Any

from hf_dm.config import MonitorConfig

logger = logging.getLogger(__name__)

_FRAME_EVENTS = {"camera_frame", "task_frame"}
_CRITICAL_EVENTS = {
    "trial_start",
    "trial_result",
    "session_summary",
    "shutdown",
    "pump",
    "white_noise",
    "lick",
    "lick_edge",
    "cap_touch_power",
    "fsm_states",
    "fsm_state",
}


@dataclass(frozen=True)
class MonitorDisplayBounds:
    x: int
    y: int
    width: int
    height: int
    is_primary: bool


def _list_windows_displays() -> list[MonitorDisplayBounds]:
    try:
        user32 = ctypes.windll.user32
    except Exception:
        return []

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
            ("szDevice", ctypes.c_wchar * 32),
        ]

    displays: list[MonitorDisplayBounds] = []
    monitor_enum_proc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(RECT),
        ctypes.c_double,
    )

    def _callback(hmonitor, _hdc, _rect, _lparam):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            displays.append(
                MonitorDisplayBounds(
                    x=int(info.rcMonitor.left),
                    y=int(info.rcMonitor.top),
                    width=int(info.rcMonitor.right - info.rcMonitor.left),
                    height=int(info.rcMonitor.bottom - info.rcMonitor.top),
                    is_primary=bool(info.dwFlags & 1),
                )
            )
        return 1

    user32.EnumDisplayMonitors(0, 0, monitor_enum_proc(_callback), 0)
    displays.sort(key=lambda item: (item.x, item.y))
    return displays


def _pick_display_bounds(display_index: int | None) -> MonitorDisplayBounds | None:
    displays = _list_windows_displays()
    if not displays:
        return None
    if display_index is not None and 0 <= display_index < len(displays):
        return displays[display_index]
    for display in displays:
        if display.is_primary:
            return display
    return displays[0]


class OperatorMonitor:
    """Small process wrapper that drops stale GUI messages instead of blocking."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self._process: mp.Process | None = None
        self._event_queue: Any = None
        self._camera_frame_queue: Any = None
        self._task_frame_queue: Any = None
        self._status_queue: Any = None
        self._last_by_key: dict[str, float] = {}
        self._dropped_messages = 0
        self._dropped_frames = 0
        self._published_counts: Counter[str] = Counter()
        self._last_child_status: dict[str, Any] = {}

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def dropped_messages(self) -> int:
        return self._dropped_messages

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def published_counts(self) -> dict[str, int]:
        return dict(self._published_counts)

    @property
    def child_status(self) -> dict[str, Any]:
        return self._drain_status_queue()

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self.running:
            return
        try:
            ctx = mp.get_context("spawn")
            self._event_queue = ctx.Queue(maxsize=max(16, int(self.config.queue_size)))
            self._camera_frame_queue = ctx.Queue(maxsize=2)
            self._task_frame_queue = ctx.Queue(maxsize=2)
            self._status_queue = ctx.Queue(maxsize=16)
            self._process = ctx.Process(
                target=_run_monitor_process,
                args=(
                    self.config,
                    self._event_queue,
                    self._camera_frame_queue,
                    self._task_frame_queue,
                    self._status_queue,
                ),
                name="hf-dm-operator-monitor",
                daemon=True,
            )
            self._process.start()
        except Exception:
            logger.exception("Failed to start operator monitor")
            self._process = None
            self._event_queue = None
            self._camera_frame_queue = None
            self._task_frame_queue = None
            self._status_queue = None

    def _queue_for_event(self, event_type: str) -> Any:
        if event_type == "camera_frame":
            return self._camera_frame_queue
        if event_type == "task_frame":
            return self._task_frame_queue
        return self._event_queue

    def _publish_frame_latest(self, target_queue: Any, message: dict[str, Any]) -> bool:
        if target_queue is None:
            return False
        while True:
            try:
                target_queue.put_nowait(message)
                self._published_counts[str(message["type"])] += 1
                return True
            except queue.Full:
                self._dropped_frames += 1
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    return False
                except Exception:
                    return False
            except Exception:
                self._dropped_frames += 1
                return False

    def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> bool:
        if not self.config.enabled:
            return False
        message = {
            "type": str(event_type),
            "payload": payload or {},
            "sent_mono_s": time.perf_counter(),
            "sent_wall_s": time.time(),
        }
        target_queue = self._queue_for_event(event_type)
        if target_queue is None:
            return False
        if event_type in _FRAME_EVENTS:
            return self._publish_frame_latest(target_queue, message)
        try:
            target_queue.put_nowait(message)
            self._published_counts[event_type] += 1
            return True
        except queue.Full:
            self._dropped_messages += 1
            if event_type in _CRITICAL_EVENTS:
                try:
                    target_queue.get_nowait()
                    target_queue.put_nowait(message)
                    self._published_counts[event_type] += 1
                    return True
                except Exception:
                    return False
            return False
        except Exception:
            if event_type in _FRAME_EVENTS:
                self._dropped_frames += 1
            else:
                self._dropped_messages += 1
            return False

    def publish_throttled(
        self,
        key: str,
        max_hz: float,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        if max_hz <= 0:
            return False
        now = time.perf_counter()
        interval_s = 1.0 / float(max_hz)
        previous = self._last_by_key.get(key, 0.0)
        if now - previous < interval_s:
            return False
        self._last_by_key[key] = now
        return self.publish(event_type, payload)

    def stop(self, timeout_s: float = 1.0) -> None:
        if not self.config.enabled:
            return
        self.publish("shutdown", {})
        process = self._process
        if process is not None:
            process.join(timeout=max(0.1, float(timeout_s)))
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
        self._drain_status_queue()
        for q in (
            self._event_queue,
            self._camera_frame_queue,
            self._task_frame_queue,
            self._status_queue,
        ):
            if q is None:
                continue
            try:
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass
        self._process = None
        self._event_queue = None
        self._camera_frame_queue = None
        self._task_frame_queue = None
        self._status_queue = None

    def _drain_status_queue(self) -> dict[str, Any]:
        status = self._last_child_status
        if self._status_queue is None:
            return dict(status)
        drained = 0
        while drained < 128:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            drained += 1
            if isinstance(message, dict):
                status = dict(message)
        self._last_child_status = dict(status)
        return dict(status)


class OperatorMonitorState:
    """Pure dashboard model kept separate from Tk rendering for tests."""

    def __init__(self, *, history_s: float = 20.0, trial_history: int = 12):
        self.history_s = max(1.0, float(history_s))
        self.trial_history = max(1, int(trial_history))
        self.session_info: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}
        self.current_trial: int | None = None
        self.current_phase = "idle"
        self.current_force_direction: str | None = None
        self.current_star_side: str | None = None
        self.current_circle_side: str | None = None
        self.fsm_states: list[dict[str, Any]] = []
        self.active_fsm_state: str | None = None
        self.last_task_state: dict[str, Any] = {}
        self.latest_camera_frame: dict[str, Any] | None = None
        self.latest_task_frame: dict[str, Any] | None = None
        self.camera_frame_count = 0
        self.task_frame_count = 0
        self.last_camera_frame_mono_s: float | None = None
        self.last_task_frame_mono_s: float | None = None
        self.force_samples: deque[tuple[float, float]] = deque()
        self.force_session_max = 1.0
        self.wheel_samples: deque[tuple[float, float, int, int, int]] = deque()
        self.lick_events: deque[float] = deque()
        self.lick_edge_events: deque[float] = deque()
        self.pump_events: deque[tuple[float, bool]] = deque()
        self.noise_events: deque[tuple[float, bool]] = deque()
        self.outcomes: deque[dict[str, Any]] = deque(maxlen=self.trial_history)
        self.camera_ttl_count = 0
        self.lick_count = 0
        self.lick_edge_count = 0
        self.lick_inputs: list[dict[str, Any]] = []
        self.lick_input_states: dict[int, dict[str, Any]] = {}
        self.last_lick_value: int | None = None
        self.last_lick_edge_mono_s: float | None = None
        self.last_touch_sample_mono_s: float | None = None
        self.last_lick_source = "-"
        self.cap_touch_power: dict[str, Any] = {}
        self.cap_touch_power_active = False
        self.pump_active = False
        self.noise_active = False
        self.last_update_mono_s = time.perf_counter()

    def apply(self, message: dict[str, Any]) -> None:
        event_type = str(message.get("type", ""))
        payload = dict(message.get("payload") or {})
        now = time.perf_counter()
        self.last_update_mono_s = now

        if event_type == "session_info":
            self.session_info.update(payload)
            inputs = payload.get("lick_inputs", [])
            if isinstance(inputs, list):
                self.lick_inputs = [dict(item) for item in inputs if isinstance(item, dict)]
                for item in self.lick_inputs:
                    channel = _optional_int(item.get("channel"))
                    if channel is None:
                        continue
                    self.lick_input_states.setdefault(channel, dict(item))
            power = payload.get("cap_touch_power")
            if isinstance(power, dict):
                self.cap_touch_power.update(power)
        elif event_type == "session_summary":
            self.summary.update(payload)
            self.current_phase = "finished"
        elif event_type == "trial_start":
            self.current_trial = _optional_int(payload.get("trial_index"))
            self.current_phase = str(payload.get("phase", "trial"))
            self.current_force_direction = _optional_str(payload.get("force_direction"))
            self.current_star_side = _optional_str(payload.get("star_side"))
            self.current_circle_side = _optional_str(payload.get("circle_side"))
            self.wheel_samples.clear()
        elif event_type == "trial_phase":
            self.current_phase = str(payload.get("phase", self.current_phase))
        elif event_type == "fsm_states":
            states = payload.get("states", [])
            if isinstance(states, list):
                self.fsm_states = [dict(item) for item in states if isinstance(item, dict)]
        elif event_type == "fsm_state":
            self.active_fsm_state = _optional_str(
                payload.get("active_state", payload.get("state"))
            )
            self.current_phase = str(payload.get("phase", self.current_phase))
        elif event_type == "task_state":
            self.last_task_state = payload
            self.current_phase = str(payload.get("phase", self.current_phase))
            if payload.get("trial_index") is not None:
                self.current_trial = _optional_int(payload.get("trial_index"))
        elif event_type == "trial_result":
            self.outcomes.append(payload)
            self.current_phase = "iti"
        elif event_type == "force_sample":
            value = _optional_float(payload.get("value"))
            if value is not None:
                self.force_samples.append((now, value))
                self.force_session_max = max(self.force_session_max, value)
        elif event_type == "wheel_sample":
            self.wheel_samples.append(
                (
                    now,
                    float(payload.get("position_deg", 0.0)),
                    int(payload.get("net_pulses", 0)),
                    int(payload.get("cw_pulses", 0)),
                    int(payload.get("ccw_pulses", 0)),
                )
            )
        elif event_type == "lick_edge":
            self.lick_edge_events.append(now)
            self.lick_edge_count += 1
            self.last_lick_value = _optional_int(payload.get("value"))
            self.last_lick_edge_mono_s = now
            self.last_lick_source = "digital"
            self._update_lick_input_state(payload, now=now, source="digital")
        elif event_type == "cap_touch_sample":
            self.last_lick_value = _optional_int(payload.get("value"))
            self.last_touch_sample_mono_s = now
            self.last_lick_source = _touch_sample_source_label(payload.get("source"))
            self._update_lick_input_state(payload, now=now, source=self.last_lick_source)
        elif event_type == "lick":
            self.lick_events.append(now)
            self.lick_count += 1
        elif event_type == "cap_touch_power":
            self.cap_touch_power.update(payload)
            self.cap_touch_power_active = bool(payload.get("active", payload.get("state", False)))
        elif event_type == "pump":
            state = bool(payload.get("active", payload.get("state", False)))
            self.pump_active = state
            self.pump_events.append((now, state))
        elif event_type == "white_noise":
            state = bool(payload.get("active", False))
            self.noise_active = state
            self.noise_events.append((now, state))
        elif event_type == "camera_ttl":
            self.camera_ttl_count += 1
        elif event_type == "camera_frame":
            self.latest_camera_frame = payload
            self.camera_frame_count += 1
            self.last_camera_frame_mono_s = now
        elif event_type == "task_frame":
            self.latest_task_frame = payload
            self.task_frame_count += 1
            self.last_task_frame_mono_s = now

        self.prune(now)

    def _update_lick_input_state(self, payload: dict[str, Any], *, now: float, source: str) -> None:
        channel = _optional_int(payload.get("channel"))
        if channel is None:
            return
        state = self.lick_input_states.setdefault(channel, {"channel": channel})
        if payload.get("pin") is not None:
            state["pin"] = payload.get("pin")
        state["value"] = payload.get("value")
        state["source"] = source
        state["last_mono_s"] = now
        state["relative_time_s"] = payload.get("relative_time_s")
        if source == "digital":
            state["last_edge_mono_s"] = now
        else:
            state["last_sample_mono_s"] = now
        for key in (
            "baseline",
            "polarity",
            "rising_high_threshold",
            "rising_low_threshold",
            "falling_high_threshold",
            "falling_low_threshold",
            "error",
        ):
            if payload.get(key) is not None:
                state[key] = payload.get(key)
        if source == "digital":
            state["active"] = bool(payload.get("value"))
        elif payload.get("active") is not None:
            state["active"] = bool(payload.get("active"))

    def prune(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else float(now)
        cutoff = now - self.history_s
        for items in (self.force_samples, self.wheel_samples, self.pump_events, self.noise_events):
            while items and items[0][0] < cutoff:
                items.popleft()
        while self.lick_events and self.lick_events[0] < cutoff:
            self.lick_events.popleft()
        while self.lick_edge_events and self.lick_edge_events[0] < cutoff:
            self.lick_edge_events.popleft()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _format_duration_s(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "logic"
    if number < 0.01:
        return f"{number:.3f}s"
    if number < 1.0:
        return f"{number:.2f}s"
    return f"{number:.1f}s"


def _short_text(text: Any, max_chars: int) -> str:
    value = str(text)
    max_chars = max(4, int(max_chars))
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _touch_sample_source_label(source: Any) -> str:
    if source in (None, ""):
        return "analog"
    value = str(source)
    if value == "firmware_digitizer_status":
        return "status"
    return value


def _frame_age_text(last_mono_s: float | None, now: float | None = None) -> str:
    if last_mono_s is None:
        return "age -"
    now = time.perf_counter() if now is None else float(now)
    age_s = max(0.0, now - float(last_mono_s))
    if age_s < 1.0:
        return f"age {age_s * 1000.0:.0f} ms"
    return f"age {age_s:.1f} s"


def _frame_to_netpbm(frame: dict[str, Any] | None) -> bytes | None:
    if not frame:
        return None
    try:
        width = int(frame["width"])
        height = int(frame["height"])
        channels = int(frame.get("channels", 1))
        data = frame["data"]
    except Exception:
        return None
    if not isinstance(data, (bytes, bytearray)) or width <= 0 or height <= 0:
        return None
    raw = bytes(data)
    pixels = width * height
    if channels == 1:
        if len(raw) < pixels:
            return None
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        return header + raw[:pixels]
    if channels >= 3:
        needed = pixels * channels
        if len(raw) < needed:
            return None
        if channels == 3:
            rgb = raw[: pixels * 3]
        else:
            rgb = bytearray(pixels * 3)
            out = 0
            for idx in range(0, needed, channels):
                rgb[out : out + 3] = raw[idx : idx + 3]
                out += 3
            rgb = bytes(rgb)
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        return header + rgb
    return None


def _mono_frame_to_rgb_netpbm(frame: dict[str, Any] | None) -> bytes | None:
    if not frame:
        return None
    try:
        width = int(frame["width"])
        height = int(frame["height"])
        channels = int(frame.get("channels", 1))
        data = frame["data"]
    except Exception:
        return None
    if channels != 1 or not isinstance(data, (bytes, bytearray)) or width <= 0 or height <= 0:
        return None
    pixels = width * height
    raw = bytes(data)
    if len(raw) < pixels:
        return None
    rgb = bytearray(pixels * 3)
    out = 0
    for value in raw[:pixels]:
        rgb[out] = value
        rgb[out + 1] = value
        rgb[out + 2] = value
        out += 3
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + bytes(rgb)


def _frame_to_pillow_image(
    frame: dict[str, Any] | None,
    *,
    max_width: int,
    max_height: int,
) -> Any | None:
    if not frame:
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        width = int(frame["width"])
        height = int(frame["height"])
        channels = int(frame.get("channels", 1))
        data = frame["data"]
    except Exception:
        return None
    if not isinstance(data, (bytes, bytearray)) or width <= 0 or height <= 0:
        return None
    raw = bytes(data)
    pixels = width * height
    try:
        if channels == 1:
            if len(raw) < pixels:
                return None
            image = Image.frombytes("L", (width, height), raw[:pixels])
        elif channels >= 3:
            needed = pixels * channels
            if len(raw) < needed:
                return None
            if channels == 3:
                rgb = raw[: pixels * 3]
            else:
                rgb_buffer = bytearray(pixels * 3)
                out = 0
                for idx in range(0, needed, channels):
                    rgb_buffer[out : out + 3] = raw[idx : idx + 3]
                    out += 3
                rgb = bytes(rgb_buffer)
            image = Image.frombytes("RGB", (width, height), rgb)
        else:
            return None
        if image.width > max_width or image.height > max_height:
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            image = image.copy()
            image.thumbnail((max(1, int(max_width)), max(1, int(max_height))), resampling)
        return image
    except Exception:
        return None


def _run_monitor_process(
    config: MonitorConfig,
    event_q: Any,
    camera_frame_q: Any,
    task_frame_q: Any,
    status_q: Any,
) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    try:
        import tkinter as tk
    except Exception:
        return

    root = tk.Tk()
    try:
        window = _MonitorWindow(root, config, event_q, camera_frame_q, task_frame_q, status_q, tk)
        window.start()
        root.mainloop()
    except Exception:
        logger.exception("Operator monitor crashed")
    finally:
        try:
            root.destroy()
        except Exception:
            pass


class _MonitorWindow:
    def __init__(
        self,
        root: Any,
        config: MonitorConfig,
        event_q: Any,
        camera_frame_q: Any,
        task_frame_q: Any,
        status_q: Any,
        tk: Any,
    ):
        self.root = root
        self.config = config
        self.event_queue = event_q
        self.camera_frame_queue = camera_frame_q
        self.task_frame_queue = task_frame_q
        self.status_queue = status_q
        self.tk = tk
        self.state = OperatorMonitorState(
            history_s=config.history_s,
            trial_history=config.trial_history,
        )
        self.width = max(980, int(config.width))
        self.height = max(720, int(config.height))
        self._images: list[Any] = []
        self._photo_cache: dict[str, tuple[tuple[Any, ...], int, int, Any]] = {}
        self._render_count = 0
        self._last_status_mono_s = 0.0
        self._last_rendered_camera_token: tuple[Any, ...] | None = None
        self._last_rendered_task_token: tuple[Any, ...] | None = None
        self._last_rendered_camera_frame_index: Any = None
        self._last_rendered_task_frame_index: Any = None
        self._unique_camera_frames_rendered = 0
        self._unique_task_frames_rendered = 0
        self._camera_photo_failures = 0
        self._task_photo_failures = 0

        self.root.title("HF Decision Monitor")
        self.root.configure(bg="#edf0f2")
        bounds = _pick_display_bounds(config.display_index)
        x = config.window_x
        y = config.window_y
        if bounds is not None:
            x = bounds.x + 24 if x is None else x
            y = bounds.y + 24 if y is None else y
        if x is None or y is None:
            self.root.geometry(f"{self.width}x{self.height}")
        else:
            self.root.geometry(f"{self.width}x{self.height}+{int(x)}+{int(y)}")
        self.root.minsize(980, 720)

        self.canvas = tk.Canvas(
            self.root,
            bg="#edf0f2",
            highlightthickness=0,
            width=self.width,
            height=self.height,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._request_close)

    def start(self) -> None:
        self._tick()

    def _request_close(self) -> None:
        self.root.quit()

    def _on_resize(self, event: Any) -> None:
        self.width = max(980, int(event.width))
        self.height = max(720, int(event.height))

    def _tick(self) -> None:
        try:
            shutdown = self._drain_queue()
            self._render()
            self._emit_status(force=shutdown)
        except Exception as exc:
            self._emit_status(
                force=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        if shutdown:
            self.root.quit()
            return
        delay_ms = max(20, int(round(1000.0 / max(1, int(self.config.fps)))))
        self.root.after(delay_ms, self._tick)

    def _drain_queue(self) -> bool:
        shutdown = False
        drained = 0
        while drained < 1500:
            try:
                message = self.event_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            drained += 1
            if message.get("type") == "shutdown":
                shutdown = True
                continue
            self.state.apply(message)

        for message in (
            self._latest_frame_message(self.camera_frame_queue, "camera_frame"),
            self._latest_frame_message(self.task_frame_queue, "task_frame"),
        ):
            if message is None:
                continue
            self.state.apply(message)
        return shutdown

    def _latest_frame_message(self, frame_queue: Any, event_type: str) -> dict[str, Any] | None:
        latest: dict[str, Any] | None = None
        drained = 0
        while drained < 64:
            try:
                message = frame_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            drained += 1
            if str(message.get("type", "")) == event_type:
                latest = message
        return latest

    def _render(self) -> None:
        self.state.prune()
        self._images.clear()
        self._render_count += 1
        c = self.canvas
        c.delete("all")
        c.create_rectangle(0, 0, self.width, self.height, fill="#edf0f2", outline="")
        margin = 16
        gap = 14
        top_h = max(330, int(self.height * 0.46))
        video_w = max(360, int(self.width * 0.36))
        task_w = max(260, int(self.width * 0.22))
        status_w = self.width - margin * 2 - gap * 2 - video_w - task_w
        status_w = max(320, status_w)
        video = (margin, margin, video_w, top_h)
        task = (margin + video_w + gap, margin, task_w, top_h)
        right_x = margin + video_w + gap + task_w + gap
        choices_h = max(120, int(top_h * 0.38))
        status_h = top_h - gap - choices_h
        if status_h < 190:
            status_h = 190
            choices_h = max(100, top_h - gap - status_h)
        status = (right_x, margin, status_w, status_h)
        choices = (right_x, margin + status_h + gap, status_w, choices_h)
        bottom_y = margin + top_h + gap
        bottom_h = self.height - bottom_y - margin
        usable_w = self.width - margin * 2 - gap * 2
        events_w = max(300, int(usable_w * 0.33))
        fsm_w = max(320, int(usable_w * 0.31))
        effort_w = usable_w - events_w - fsm_w
        if effort_w < 300:
            events_w = max(280, int(usable_w * 0.31))
            fsm_w = max(300, int(usable_w * 0.31))
            effort_w = usable_w - events_w - fsm_w
        events = (margin, bottom_y, events_w, bottom_h)
        fsm = (margin + events_w + gap, bottom_y, fsm_w, bottom_h)
        effort = (
            margin + events_w + gap + fsm_w + gap,
            bottom_y,
            effort_w,
            bottom_h,
        )

        self._draw_video_panel(*video)
        self._draw_task_panel(*task)
        self._draw_status_panel(*status)
        self._draw_choices_panel(*choices)
        self._draw_events_panel(*events)
        self._draw_fsm_panel(*fsm)
        self._draw_effort_panel(*effort)

    def _panel(self, x: int, y: int, w: int, h: int, title: str) -> tuple[int, int, int, int]:
        c = self.canvas
        c.create_rectangle(x, y, x + w, y + h, fill="#ffffff", outline="#ccd3d8", width=1)
        c.create_rectangle(x, y, x + w, y + 34, fill="#f8fafb", outline="#ccd3d8", width=1)
        c.create_text(
            x + 14,
            y + 18,
            anchor="w",
            text=title,
            fill="#20262b",
            font=("Segoe UI", 11, "bold"),
        )
        return x + 12, y + 46, w - 24, h - 58

    def _draw_video_panel(self, x: int, y: int, w: int, h: int) -> None:
        ix, iy, iw, ih = self._panel(x, y, w, h, "Video")
        frame = self.state.latest_camera_frame
        drew_frame = self._draw_image_or_placeholder(
            ix,
            iy,
            iw,
            ih,
            frame,
            "waiting for camera preview",
            cache_key="camera",
        )
        now = time.perf_counter()
        if frame and drew_frame:
            token = self._frame_cache_token(frame)
            if token != self._last_rendered_camera_token:
                self._last_rendered_camera_token = token
                self._unique_camera_frames_rendered += 1
                self._last_rendered_camera_frame_index = frame.get("frame_index")
        elif frame:
            self._camera_photo_failures += 1
        if frame:
            self._draw_frame_overlay(ix, iy, frame, drew_frame, self.state.last_camera_frame_mono_s, now)
        meta = [f"frames {self.state.camera_frame_count}"]
        if frame:
            meta.append(f"frame {frame.get('frame_index', 0)}")
        meta.append(_frame_age_text(self.state.last_camera_frame_mono_s, now))
        meta.append(f"TTL {self.state.camera_ttl_count}")
        self.canvas.create_text(
            ix + 8,
            y + h - 16,
            anchor="w",
            text="  ".join(meta),
            fill="#51606b",
            font=("Segoe UI", 10),
        )

    def _draw_task_panel(self, x: int, y: int, w: int, h: int) -> None:
        ix, iy, iw, ih = self._panel(x, y, w, h, "Task Screen")
        frame = self.state.latest_task_frame
        drew_frame = self._draw_image_or_placeholder(
            ix,
            iy,
            iw,
            ih,
            frame,
            "waiting for task frame",
            cache_key="task",
        )
        now = time.perf_counter()
        if frame and drew_frame:
            token = self._frame_cache_token(frame)
            if token != self._last_rendered_task_token:
                self._last_rendered_task_token = token
                self._unique_task_frames_rendered += 1
                self._last_rendered_task_frame_index = frame.get("frame_index")
        elif frame:
            self._task_photo_failures += 1
        meta = [f"frames {self.state.task_frame_count}"]
        if frame and frame.get("frame_index") is not None:
            meta.append(f"frame {frame.get('frame_index')}")
        meta.append(_frame_age_text(self.state.last_task_frame_mono_s, now))
        self.canvas.create_text(
            ix + 8,
            y + h - 16,
            anchor="w",
            text="  ".join(meta),
            fill="#51606b",
            font=("Segoe UI", 10),
        )

    def _draw_status_panel(self, x: int, y: int, w: int, h: int) -> None:
        ix, iy, iw, ih = self._panel(x, y, w, h, "Session")
        subject = self.state.session_info.get("subject_id", "")
        phase = self.state.session_info.get("phase", self.state.current_phase)
        trial = "-" if self.state.current_trial is None else str(self.state.current_trial)
        planned = self.state.session_info.get("n_trials", "")
        status_lines = [
            ("Subject", subject),
            ("Phase", phase),
            ("Trial", f"{trial}/{planned}" if planned else trial),
            ("State", self.state.current_phase),
            ("FSM", self.state.active_fsm_state or "-"),
            ("Force cue", self.state.current_force_direction or "-"),
            ("Star side", self.state.current_star_side or "-"),
            ("Circle side", self.state.current_circle_side or "-"),
            ("Licks", str(self.state.lick_count)),
            (
                "Touch input",
                f"edges {self.state.lick_edge_count}  last {self.state.last_lick_value} ({self.state.last_lick_source})"
                if self.state.last_lick_value is not None
                else f"edges {self.state.lick_edge_count}",
            ),
            ("Pump", "ON" if self.state.pump_active else "off"),
            ("Noise", "ON" if self.state.noise_active else "off"),
        ]
        yy = iy
        line_step = max(16, min(22, ih // max(1, len(status_lines))))
        for label, value in status_lines:
            self.canvas.create_text(ix, yy, anchor="w", text=label, fill="#697782", font=("Segoe UI", 9))
            self.canvas.create_text(
                ix + 112,
                yy,
                anchor="w",
                text=str(value),
                fill="#1f2933",
                font=("Segoe UI", 10, "bold"),
            )
            yy += line_step

        self._draw_speaker_icon(ix + iw - 58, iy + 22, active=self.state.noise_active)

    def _draw_choices_panel(self, x: int, y: int, w: int, h: int) -> None:
        ix, iy, iw, ih = self._panel(x, y, w, h, "Latest Choices")
        self._draw_outcomes(ix, iy, iw, ih)

    def _draw_outcomes(self, x: int, y: int, w: int, h: int) -> None:
        row_h = max(38, min(54, h))
        col_w = max(54, min(72, w // max(1, min(self.state.trial_history, 6))))
        symbol_r = max(10, min(14, row_h // 3))
        max_cols = max(1, w // col_w)
        for idx, outcome in enumerate(list(self.state.outcomes)[-self.state.trial_history :]):
            col = idx % max_cols
            row = idx // max_cols
            yy = y + row * row_h
            if yy + row_h > y + h:
                break
            xx = x + col * col_w
            chosen_shape = outcome.get("chosen_shape")
            responded = bool(outcome.get("responded", chosen_shape in ("star", "circle")))
            matched = bool(outcome.get("matched", False))
            fill = "#49b979" if matched else "#5b9ee8" if responded else "#e06060"
            box_w = min(col_w - 8, 58)
            box_h = min(row_h - 8, 44)
            self.canvas.create_rectangle(xx, yy, xx + box_w, yy + box_h, fill="#f8fafb", outline="#d7dde2")
            self.canvas.create_text(
                xx + 6,
                yy + 8,
                anchor="w",
                text=str(outcome.get("trial_index", "")),
                fill="#687780",
                font=("Segoe UI", 8, "bold"),
            )
            cx = xx + box_w // 2 + 8
            cy = yy + box_h // 2 + 4
            if chosen_shape == "star":
                self._draw_star(cx, cy, symbol_r, fill)
            elif chosen_shape == "circle":
                self.canvas.create_oval(
                    cx - symbol_r,
                    cy - symbol_r,
                    cx + symbol_r,
                    cy + symbol_r,
                    fill=fill,
                    outline="",
                )
            else:
                self.canvas.create_line(
                    cx - symbol_r,
                    cy - symbol_r,
                    cx + symbol_r,
                    cy + symbol_r,
                    fill=fill,
                    width=3,
                )
                self.canvas.create_line(
                    cx + symbol_r,
                    cy - symbol_r,
                    cx - symbol_r,
                    cy + symbol_r,
                    fill=fill,
                    width=3,
                )

    def _draw_events_panel(self, x: int, y: int, w: int, h: int) -> None:
        ix, iy, iw, ih = self._panel(x, y, w, h, "Lick / Pump / Noise")
        now = time.perf_counter()
        pin_h = 66
        self._draw_touch_pin_status(ix, iy, iw, pin_h, now)
        rails_y = iy + pin_h + 8
        rail_h = max(34, (ih - pin_h - 8) // 5)
        labels = [("Lick", "#2f7dd3"), ("Pump", "#37a56b"), ("Noise", "#d9a516")]
        for idx, (label, color) in enumerate(labels):
            yy = rails_y + idx * rail_h
            self.canvas.create_text(ix, yy + 14, anchor="w", text=label, fill="#40515f", font=("Segoe UI", 9, "bold"))
            self.canvas.create_line(ix + 64, yy + 18, ix + iw - 10, yy + 18, fill="#d8dee3")
            if label == "Lick":
                events = self.state.lick_edge_events if self.state.lick_edge_events else self.state.lick_events
                self._draw_ticks(ix + 64, yy + 4, iw - 74, 30, now, events, color)
                if (
                    self.state.last_lick_edge_mono_s is not None
                    and now - self.state.last_lick_edge_mono_s < 0.35
                ):
                    self.canvas.create_oval(
                        ix + 39,
                        yy + 8,
                        ix + 55,
                        yy + 24,
                        fill=color,
                        outline="",
                    )
            elif label == "Pump":
                self._draw_state_intervals(ix + 64, yy + 4, iw - 74, 30, now, self.state.pump_events, color, self.state.pump_active)
            else:
                self._draw_state_intervals(ix + 64, yy + 4, iw - 74, 30, now, self.state.noise_events, color, self.state.noise_active)
                self._draw_speaker_icon(ix + iw - 38, yy + 18, active=self.state.noise_active, small=True)

        wheel_y = rails_y + rail_h * 3 + 10
        self.canvas.create_text(ix, wheel_y, anchor="w", text="Wheel (trial)", fill="#20262b", font=("Segoe UI", 10, "bold"))
        plot_y = wheel_y + 18
        self._draw_wheel_plot(ix, plot_y, iw, ih - (plot_y - iy))

    def _draw_touch_pin_status(self, x: int, y: int, w: int, h: int, now: float) -> None:
        power_channel = self.state.cap_touch_power.get("channel")
        power_pin = self.state.cap_touch_power.get("pin")
        power_text = (
            f"power ch{power_channel}"
            + (f"/pin{power_pin}" if power_pin is not None else "")
            + (" ON" if self.state.cap_touch_power_active else " off")
        )
        power_fill = "#daf3e5" if self.state.cap_touch_power_active else "#f4f6f7"
        self.canvas.create_rectangle(x, y, x + w, y + h, fill="#fbfcfd", outline="#e1e6ea")
        self.canvas.create_rectangle(x + 8, y + 8, x + min(w - 8, 170), y + 30, fill=power_fill, outline="#d7dde2")
        self.canvas.create_text(
            x + 16,
            y + 19,
            anchor="w",
            text=power_text,
            fill="#1f2933",
            font=("Segoe UI", 8, "bold"),
        )
        inputs = self.state.lick_inputs or [
            {"channel": channel, **state}
            for channel, state in sorted(self.state.lick_input_states.items())
        ]
        if not inputs:
            self.canvas.create_text(
                x + 16,
                y + 46,
                anchor="w",
                text="touch input: waiting for channel config",
                fill="#687780",
                font=("Segoe UI", 9),
            )
            return
        chip_x = x + 8
        chip_y = y + 36
        for item in inputs[:3]:
            channel = _optional_int(item.get("channel"))
            if channel is None:
                continue
            state = self.state.lick_input_states.get(channel, {})
            pin = state.get("pin", item.get("pin"))
            value = state.get("value", "-")
            source = state.get("source", "digital")
            last_sample_mono = _optional_float(state.get("last_sample_mono_s"))
            last_edge_mono = _optional_float(state.get("last_edge_mono_s"))
            sample_recent = last_sample_mono is not None and now - last_sample_mono < 1.2
            edge_recent = last_edge_mono is not None and now - last_edge_mono < 0.6
            active = bool(state.get("active", False))
            fill = "#dbeafe" if active or edge_recent else "#f9fbfc" if sample_recent else "#f4f6f7"
            outline = "#2f7dd3" if active else "#d7dde2"
            chip_w = min(max(132, w // 3 - 8), w - (chip_x - x) - 8)
            if chip_w < 100:
                break
            text = f"in ch{channel}" + (f"/pin{pin}" if pin is not None else "")
            text += f"  {source}:{value}"
            if state.get("baseline") is not None:
                text += f" b{state.get('baseline')}"
            if state.get("rising_high_threshold") is not None:
                text += f" hi>{state.get('rising_high_threshold')}"
            if state.get("active") is not None:
                text += " ON" if active else " off"
            self.canvas.create_rectangle(chip_x, chip_y, chip_x + chip_w, chip_y + 24, fill=fill, outline=outline)
            self.canvas.create_text(
                chip_x + 8,
                chip_y + 12,
                anchor="w",
                text=_short_text(text, max(12, chip_w // 7)),
                fill="#1f2933",
                font=("Segoe UI", 8, "bold"),
            )
            chip_x += chip_w + 8

    def _draw_fsm_panel(self, x: int, y: int, w: int, h: int) -> None:
        ix, iy, iw, ih = self._panel(x, y, w, h, "Bpod FSM")
        states = self.state.fsm_states
        if not states:
            self.canvas.create_text(
                ix + iw / 2,
                iy + ih / 2,
                text="waiting for FSM definition",
                fill="#8b99a4",
                font=("Segoe UI", 10),
            )
            return
        active = self.state.active_fsm_state
        row_gap = 4
        row_h = max(22, min(38, (ih - row_gap * (len(states) - 1)) // max(1, len(states))))
        compact = row_h < 30
        yy = iy
        for state in states:
            name = str(state.get("name", ""))
            duration = _format_duration_s(state.get("duration_s"))
            logic = str(state.get("logic", ""))
            is_active = name == active
            fill = "#daf3e5" if is_active else "#f9fbfc"
            outline = "#37a56b" if is_active else "#d7dde2"
            text_color = "#143b27" if is_active else "#20262b"
            self.canvas.create_rectangle(ix, yy, ix + iw, yy + row_h, fill=fill, outline=outline, width=1)
            if is_active:
                self.canvas.create_rectangle(ix, yy, ix + 6, yy + row_h, fill="#37a56b", outline="")
            self.canvas.create_text(
                ix + 12,
                yy + (row_h / 2 if compact else 11),
                anchor="w",
                text=_short_text(name, max(10, iw // 12)),
                fill=text_color,
                font=("Segoe UI", 9, "bold"),
            )
            self.canvas.create_text(
                ix + iw - 8,
                yy + (row_h / 2 if compact else 11),
                anchor="e",
                text=duration,
                fill="#40515f",
                font=("Segoe UI", 8, "bold"),
            )
            if not compact:
                self.canvas.create_text(
                    ix + 12,
                    yy + 28,
                    anchor="w",
                    text=_short_text(logic, max(16, iw // 7)),
                    fill="#687780",
                    font=("Segoe UI", 8),
                )
            yy += row_h + row_gap
            if yy + row_h > iy + ih + 2:
                break

    def _draw_effort_panel(self, x: int, y: int, w: int, h: int) -> None:
        ix, iy, iw, ih = self._panel(x, y, w, h, "Effort ADC")
        samples = list(self.state.force_samples)
        now = time.perf_counter()
        self._draw_time_series(
            ix,
            iy,
            iw,
            ih - 36,
            now,
            samples,
            line="#c44d4d",
            fill="#fff9f8",
            zero_line=False,
            y_range=(0.0, self.state.force_session_max),
        )
        if samples:
            values = [value for _, value in samples]
            summary = (
                f"last {values[-1]:.0f}   "
                f"min {min(values):.0f}   mean {sum(values) / len(values):.1f}   "
                f"max {max(values):.0f}   session max {self.state.force_session_max:.0f}"
            )
        else:
            summary = "no force samples"
        self.canvas.create_text(ix, iy + ih - 16, anchor="w", text=summary, fill="#51606b", font=("Segoe UI", 10))

    def _draw_wheel_plot(self, x: int, y: int, w: int, h: int) -> None:
        samples = list(self.state.wheel_samples)
        now = time.perf_counter()
        self._draw_time_series(
            x,
            y,
            w,
            max(80, h - 28),
            now,
            [(t, pos) for t, pos, _net, _cw, _ccw in samples],
            line="#4d6fc4",
            fill="#f8faff",
            zero_line=True,
        )
        if samples:
            _t, pos, net, cw, ccw = samples[-1]
            direction = "CW" if pos > 0 else "CCW" if pos < 0 else "still"
            text = f"{direction}   pos {pos:.1f} deg   net {net}   CW {cw}   CCW {ccw}"
        else:
            text = "waiting for wheel movement"
        self.canvas.create_text(x, y + max(80, h - 28) + 18, anchor="w", text=text, fill="#51606b", font=("Segoe UI", 10))

    def _draw_time_series(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        now: float,
        samples: list[tuple[float, float]],
        *,
        line: str,
        fill: str,
        zero_line: bool,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        c = self.canvas
        c.create_rectangle(x, y, x + w, y + h, fill=fill, outline="#d7dde2")
        for frac in (0.25, 0.5, 0.75):
            yy = y + h * frac
            c.create_line(x, yy, x + w, yy, fill="#e3e8ec")
        for frac in (0.25, 0.5, 0.75):
            xx = x + w * frac
            c.create_line(xx, y, xx, y + h, fill="#e3e8ec")
        if not samples:
            c.create_text(x + w / 2, y + h / 2, text="no samples", fill="#8b99a4", font=("Segoe UI", 10))
            return
        values = [value for _t, value in samples]
        if y_range is not None:
            vmin, vmax = float(y_range[0]), float(y_range[1])
            if vmax <= vmin:
                vmax = vmin + 1.0
        elif zero_line:
            vmin = min(values)
            vmax = max(values)
            span = max(abs(vmin), abs(vmax), 1.0)
            vmin, vmax = -span, span
        else:
            vmin = min(values)
            vmax = max(values)
            if vmax <= vmin:
                vmax = vmin + 1.0
        if zero_line:
            zero_y = y + h - ((0.0 - vmin) / (vmax - vmin)) * h
            c.create_line(x, zero_y, x + w, zero_y, fill="#aab5bd", width=1)
        points: list[float] = []
        cutoff = now - self.state.history_s
        for t, value in samples:
            tx = x + ((t - cutoff) / self.state.history_s) * w
            ty = y + h - ((value - vmin) / (vmax - vmin)) * h
            if x - 2 <= tx <= x + w + 2:
                points.extend([tx, ty])
        if len(points) >= 4:
            c.create_line(*points, fill=line, width=2)
        c.create_text(x + 6, y + 8, anchor="w", text=f"{vmax:.0f}", fill="#7b8993", font=("Segoe UI", 8))
        c.create_text(x + 6, y + h - 10, anchor="w", text=f"{vmin:.0f}", fill="#7b8993", font=("Segoe UI", 8))

    def _draw_ticks(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        now: float,
        events: deque[float],
        color: str,
    ) -> None:
        cutoff = now - self.state.history_s
        for event_t in events:
            xx = x + ((event_t - cutoff) / self.state.history_s) * w
            if x <= xx <= x + w:
                self.canvas.create_line(xx, y, xx, y + h, fill=color, width=2)

    def _draw_state_intervals(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        now: float,
        events: deque[tuple[float, bool]],
        color: str,
        active: bool,
    ) -> None:
        cutoff = now - self.state.history_s
        start: float | None = None
        for event_t, state in events:
            if state:
                start = max(event_t, cutoff)
            elif start is not None:
                self._state_rect(x, y, w, h, cutoff, start, event_t, color)
                start = None
        if active and start is None:
            start = events[-1][0] if events else now
        if start is not None:
            self._state_rect(x, y, w, h, cutoff, max(start, cutoff), now, color)

    def _state_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        cutoff: float,
        start: float,
        end: float,
        color: str,
    ) -> None:
        x1 = x + ((start - cutoff) / self.state.history_s) * w
        x2 = x + ((end - cutoff) / self.state.history_s) * w
        x1 = max(x, min(x + w, x1))
        x2 = max(x, min(x + w, x2))
        if x2 > x1:
            self.canvas.create_rectangle(x1, y + 5, x2, y + h - 5, fill=color, outline="")

    def _draw_image_or_placeholder(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame: dict[str, Any] | None,
        placeholder: str,
        *,
        cache_key: str,
    ) -> bool:
        self.canvas.create_rectangle(x, y, x + w, y + h, fill="#11161b", outline="#11161b")
        photo = self._photo_from_frame(
            frame,
            cache_key=cache_key,
            max_width=w,
            max_height=h,
        )
        if photo is None:
            self.canvas.create_text(
                x + w / 2,
                y + h / 2,
                text=placeholder,
                fill="#98a5ad",
                font=("Segoe UI", 10),
            )
            return False
        self._images.append(photo)
        px = x + max(0, (w - photo.width()) // 2)
        py = y + max(0, (h - photo.height()) // 2)
        self.canvas.create_image(px, py, anchor="nw", image=photo)
        return True

    def _draw_frame_overlay(
        self,
        x: int,
        y: int,
        frame: dict[str, Any],
        drew_frame: bool,
        last_frame_mono_s: float | None,
        now: float,
    ) -> None:
        age_s = None if last_frame_mono_s is None else max(0.0, now - float(last_frame_mono_s))
        fill = "#2bc46f" if drew_frame and (age_s is None or age_s < 0.75) else "#d8a91e"
        if not drew_frame:
            fill = "#e06060"
        frame_index = frame.get("frame_index", "-")
        text = f"#{frame_index}"
        self.canvas.create_rectangle(
            x + 8,
            y + 8,
            x + 96,
            y + 32,
            fill="#10161b",
            outline="#26323a",
        )
        self.canvas.create_oval(x + 15, y + 15, x + 23, y + 23, fill=fill, outline="")
        self.canvas.create_text(
            x + 30,
            y + 20,
            anchor="w",
            text=text,
            fill="#edf0f2",
            font=("Segoe UI", 9, "bold"),
        )

    def _emit_status(self, *, force: bool = False, error: str | None = None) -> None:
        if self.status_queue is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_status_mono_s < 0.5:
            return
        status = {
            "wall_time_s": time.time(),
            "monotonic_s": now,
            "render_count": int(self._render_count),
            "camera_frames_consumed": int(self.state.camera_frame_count),
            "task_frames_consumed": int(self.state.task_frame_count),
            "camera_frames_rendered": int(self._unique_camera_frames_rendered),
            "task_frames_rendered": int(self._unique_task_frames_rendered),
            "last_camera_frame_index": self.state.latest_camera_frame.get("frame_index")
            if self.state.latest_camera_frame
            else None,
            "last_rendered_camera_frame_index": self._last_rendered_camera_frame_index,
            "last_task_frame_index": self.state.latest_task_frame.get("frame_index")
            if self.state.latest_task_frame
            else None,
            "last_rendered_task_frame_index": self._last_rendered_task_frame_index,
            "last_camera_frame_age_s": (
                None
                if self.state.last_camera_frame_mono_s is None
                else max(0.0, now - float(self.state.last_camera_frame_mono_s))
            ),
            "camera_photo_failures": int(self._camera_photo_failures),
            "task_photo_failures": int(self._task_photo_failures),
        }
        if error is not None:
            status["error"] = error
        while True:
            try:
                self.status_queue.put_nowait(status)
                self._last_status_mono_s = now
                return
            except queue.Full:
                try:
                    self.status_queue.get_nowait()
                except Exception:
                    return
            except Exception:
                return

    def _photo_from_frame(
        self,
        frame: dict[str, Any] | None,
        *,
        cache_key: str,
        max_width: int,
        max_height: int,
    ) -> Any | None:
        if frame is None:
            self._photo_cache.pop(cache_key, None)
            return None
        token = self._frame_cache_token(frame)
        cached = self._photo_cache.get(cache_key)
        if cached is not None and cached[:3] == (token, int(max_width), int(max_height)):
            return cached[3]
        photo = self._pillow_photo_from_frame(
            frame,
            max_width=max_width,
            max_height=max_height,
        )
        if photo is not None:
            self._photo_cache[cache_key] = (
                token,
                int(max_width),
                int(max_height),
                photo,
            )
            return photo
        data = _frame_to_netpbm(frame)
        if data is None:
            self._photo_cache.pop(cache_key, None)
            return None
        for candidate, image_format in (
            (data, "PGM" if int(frame.get("channels", 1)) == 1 else "PPM"),
            (_mono_frame_to_rgb_netpbm(frame), "PPM"),
        ):
            if candidate is None:
                continue
            try:
                photo = self.tk.PhotoImage(data=candidate, format=image_format)
                if photo.width() > max_width or photo.height() > max_height:
                    factor = max(
                        1,
                        int(math.ceil(photo.width() / max(1, max_width))),
                        int(math.ceil(photo.height() / max(1, max_height))),
                    )
                    photo = photo.subsample(factor, factor)
                self._photo_cache[cache_key] = (
                    token,
                    int(max_width),
                    int(max_height),
                    photo,
                )
                return photo
            except Exception:
                continue
        self._photo_cache.pop(cache_key, None)
        return None

    def _pillow_photo_from_frame(
        self,
        frame: dict[str, Any],
        *,
        max_width: int,
        max_height: int,
    ) -> Any | None:
        image = _frame_to_pillow_image(
            frame,
            max_width=max_width,
            max_height=max_height,
        )
        if image is None:
            return None
        try:
            from PIL import ImageTk

            return ImageTk.PhotoImage(image, master=self.root)
        except Exception:
            return None

    def _frame_cache_token(self, frame: dict[str, Any]) -> tuple[Any, ...]:
        data = frame.get("data")
        return (
            frame.get("frame_index"),
            frame.get("camera_timestamp_ns"),
            frame.get("monotonic_s"),
            frame.get("width"),
            frame.get("height"),
            frame.get("channels"),
            len(data) if isinstance(data, (bytes, bytearray)) else None,
        )

    def _draw_speaker_icon(self, cx: int, cy: int, *, active: bool, small: bool = False) -> None:
        scale = 0.65 if small else 1.0
        color = "#d9a516" if active else "#93a0a8"
        r = int(14 * scale)
        c = self.canvas
        c.create_polygon(
            cx - r,
            cy - int(6 * scale),
            cx - int(5 * scale),
            cy - int(6 * scale),
            cx + int(4 * scale),
            cy - int(13 * scale),
            cx + int(4 * scale),
            cy + int(13 * scale),
            cx - int(5 * scale),
            cy + int(6 * scale),
            cx - r,
            cy + int(6 * scale),
            fill=color,
            outline="",
        )
        if active:
            for offset in (8, 14):
                c.create_arc(
                    cx - offset,
                    cy - offset,
                    cx + offset,
                    cy + offset,
                    start=-38,
                    extent=76,
                    outline=color,
                    width=2,
                    style="arc",
                )

    def _draw_star(self, cx: int, cy: int, radius: int, fill: str) -> None:
        points: list[float] = []
        for idx in range(10):
            rr = radius if idx % 2 == 0 else radius * 0.45
            angle = -math.pi / 2.0 + idx * math.pi / 5.0
            points.extend([cx + math.cos(angle) * rr, cy + math.sin(angle) * rr])
        self.canvas.create_polygon(*points, fill=fill, outline="")
