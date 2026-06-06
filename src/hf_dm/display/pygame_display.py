"""Minimal pygame display (compatible with screen_shape_force_discount geometry)."""

from __future__ import annotations

import ctypes
import math
import os
import time
from dataclasses import dataclass
from typing import Literal

import pygame

from hf_dm.config import DisplayConfig
from hf_dm.fsm.runtime import Side

Direction = Literal["cw", "ccw"]


@dataclass
class DisplayBounds:
    x: int
    y: int
    width: int
    height: int
    is_primary: bool
    name: str = ""


def list_windows_displays() -> list[DisplayBounds]:
    if os.name != "nt":
        return []

    user32 = ctypes.windll.user32

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

    monitors: list[DisplayBounds] = []
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
            monitors.append(
                DisplayBounds(
                    x=int(info.rcMonitor.left),
                    y=int(info.rcMonitor.top),
                    width=int(info.rcMonitor.right - info.rcMonitor.left),
                    height=int(info.rcMonitor.bottom - info.rcMonitor.top),
                    is_primary=bool(info.dwFlags & 1),
                    name=str(info.szDevice),
                )
            )
        return 1

    user32.EnumDisplayMonitors(0, 0, monitor_enum_proc(_callback), 0)
    monitors.sort(key=lambda mon: (mon.x, mon.y, mon.name))
    return monitors


def pick_display_bounds(display_index: int | None) -> DisplayBounds | None:
    displays = list_windows_displays()
    if not displays:
        return None
    if display_index is not None and 0 <= display_index < len(displays):
        return displays[display_index]
    for display in displays:
        if not display.is_primary:
            return display
    return displays[0]


class TaskDisplay:
    def __init__(self, config: DisplayConfig):
        self.config = config
        display_bounds = pick_display_bounds(config.display_index)
        width = config.width
        height = config.height
        window_x = config.window_x
        window_y = config.window_y

        if display_bounds is not None:
            if window_x is None:
                window_x = display_bounds.x
            if window_y is None:
                window_y = display_bounds.y
            if config.auto_size_to_display:
                width = display_bounds.width
                height = display_bounds.height

        self.width = int(width or 480)
        self.height = int(height or 1920)
        scale_x = self.width / 480.0
        scale_y = self.height / 1920.0
        scale = max(0.5, min(scale_x, scale_y))
        self.shape_radius_px = max(36, int(round(config.shape_radius_px * scale)))
        self.target_radius_px = max(48, int(round(config.target_radius_px * scale)))

        if window_x is not None and window_y is not None:
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{window_x},{window_y}"

        pygame.init()
        pygame.font.init()
        flags = 0
        if config.fullscreen:
            flags |= pygame.FULLSCREEN
        if config.noframe:
            flags |= pygame.NOFRAME
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("HF Decision-Making Task")
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self._decision_shapes, self.shape_bright_pixel_counts = self._make_decision_shapes()

    def fill_black(self) -> float:
        self.screen.fill((0, 0, 0))
        pygame.display.flip()
        return time.perf_counter()

    def preview_frame(
        self,
        max_width: int,
        max_height: int,
        *,
        rotate_left: bool = False,
    ) -> dict | None:
        """Return a small RGB copy of the current task window for the operator GUI."""
        max_width = int(max_width)
        max_height = int(max_height)
        if max_width <= 0 or max_height <= 0:
            return None
        surface = self.screen.copy()
        if rotate_left:
            surface = pygame.transform.rotate(surface, 90)
        src_width, src_height = surface.get_size()
        if src_width <= 0 or src_height <= 0:
            return None
        scale = min(max_width / float(src_width), max_height / float(src_height), 1.0)
        dst_width = max(1, int(round(src_width * scale)))
        dst_height = max(1, int(round(src_height * scale)))
        if dst_width != src_width or dst_height != src_height:
            surface = pygame.transform.smoothscale(surface, (dst_width, dst_height))
        return {
            "width": dst_width,
            "height": dst_height,
            "channels": 3,
            "format": "RGB",
            "data": pygame.image.tostring(surface, "RGB"),
            "monotonic_s": time.perf_counter(),
            "rotation_deg": 90 if rotate_left else 0,
        }

    def pump_events(self) -> None:
        self._pump_quit()

    def show_black(self, duration_s: float) -> None:
        end_t = time.perf_counter() + duration_s
        while time.perf_counter() < end_t:
            self._pump_quit()
            self.fill_black()
            self.clock.tick(self.config.fps)

    def _pump_quit(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                raise KeyboardInterrupt("User aborted")

    def _offset_px(self, signed_deg: float) -> int:
        magnitude_deg = abs(float(signed_deg))
        if magnitude_deg <= 0:
            return 0
        radians = math.radians(magnitude_deg)
        px = (self.config.distance_mm / self.config.pixel_size_mm) * (
            math.sin(radians) / math.sin(math.radians(90.0 - magnitude_deg))
        )
        px = int(round(px))
        return px if signed_deg >= 0 else -px

    def draw_decision(
        self,
        *,
        star_side: Side,
        left_position_deg: float,
        right_position_deg: float,
        force_direction: Direction,
        trial_index: int | None = None,
        response_time_s: float | None = None,
    ) -> float:
        """Draw equal-luminance lateral options moving independently toward center."""
        self._pump_quit()
        self.screen.fill(self.config.background_rgb)
        pygame.draw.circle(
            self.screen,
            self.config.target_rgb,
            self._shape_center(0.0),
            self.target_radius_px,
            width=4,
        )
        left_shape = "star" if star_side == "left" else "circle"
        right_shape = "circle" if star_side == "left" else "star"
        self._blit_shape(left_shape, self._shape_center(left_position_deg))
        self._blit_shape(right_shape, self._shape_center(right_position_deg))
        if self.config.show_debug_overlay:
            self._draw_debug_overlay(force_direction, trial_index, response_time_s)
        pygame.display.flip()
        return time.perf_counter()

    def draw_trial(
        self,
        force_direction: Direction,
        signed_position_deg: float,
        outcome: str | None = None,
        trial_index: int | None = None,
        response_time_s: float | None = None,
    ) -> float:
        self._pump_quit()
        self.screen.fill(self.config.background_rgb)
        self._draw_targets()
        center = self._shape_center(signed_position_deg)

        if outcome == "matched":
            color = self.config.matched_rgb
        elif outcome == "opposite":
            color = self.config.unmatched_rgb
        elif outcome == "timeout":
            color = self.config.timeout_rgb
        else:
            color = self.config.cue_rgb

        points = self._triangle(center, force_direction)
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (25, 25, 25), points, width=3)
        if self.config.show_debug_overlay:
            self._draw_debug_overlay(force_direction, trial_index, response_time_s)
        pygame.display.flip()
        return time.perf_counter()

    def hold_trial(
        self,
        duration_s: float,
        force_direction: Direction,
        signed_position_deg: float,
        outcome: str,
        trial_index: int | None = None,
        response_time_s: float | None = None,
    ) -> None:
        end_t = time.perf_counter() + duration_s
        while time.perf_counter() < end_t:
            self.draw_trial(
                force_direction=force_direction,
                signed_position_deg=signed_position_deg,
                outcome=outcome,
                trial_index=trial_index,
                response_time_s=response_time_s,
            )
            self.clock.tick(self.config.fps)

    def draw_centering(
        self,
        movement_direction: Direction,
        signed_position_deg: float,
        outcome: str | None = None,
        trial_index: int | None = None,
        response_time_s: float | None = None,
    ) -> float:
        """Draw the shaping task: move the lateral cue into the center target."""
        self._pump_quit()
        self.screen.fill(self.config.background_rgb)
        pygame.draw.circle(
            self.screen,
            self.config.target_rgb,
            self._shape_center(0.0),
            self.target_radius_px,
            width=4,
        )
        center = self._shape_center(signed_position_deg)
        if outcome == "centered":
            color = self.config.matched_rgb
        elif outcome == "timeout":
            color = self.config.timeout_rgb
        else:
            color = self.config.cue_rgb
        points = self._triangle(center, movement_direction)
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (25, 25, 25), points, width=3)
        if self.config.show_debug_overlay:
            self._draw_debug_overlay(movement_direction, trial_index, response_time_s)
        pygame.display.flip()
        return time.perf_counter()

    def hold_centering(
        self,
        duration_s: float,
        movement_direction: Direction,
        signed_position_deg: float,
        outcome: str,
        trial_index: int | None = None,
        response_time_s: float | None = None,
    ) -> None:
        end_t = time.perf_counter() + duration_s
        while time.perf_counter() < end_t:
            self.draw_centering(
                movement_direction=movement_direction,
                signed_position_deg=signed_position_deg,
                outcome=outcome,
                trial_index=trial_index,
                response_time_s=response_time_s,
            )
            self.clock.tick(self.config.fps)

    def _shape_center(self, signed_position_deg: float) -> tuple[int, int]:
        cx, cy = self.width // 2, self.height // 2
        offset = self._offset_px(signed_position_deg)
        if self.config.axis == "vertical":
            return cx, cy + offset
        return cx + offset, cy

    def _draw_targets(self) -> None:
        for signed_deg in (-self.config.choice_distance_deg, self.config.choice_distance_deg):
            pygame.draw.circle(
                self.screen,
                self.config.target_rgb,
                self._shape_center(signed_deg),
                self.target_radius_px,
                width=4,
            )

    def _make_decision_shapes(self) -> tuple[dict[str, pygame.Surface], dict[str, int]]:
        diameter = self.shape_radius_px * 2 + 5
        center = diameter // 2
        color = (*self.config.cue_rgb, 255)

        star = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        star_points: list[tuple[int, int]] = []
        for index in range(10):
            radius = self.shape_radius_px if index % 2 == 0 else int(self.shape_radius_px * 0.46)
            angle = -math.pi / 2.0 + index * math.pi / 5.0
            star_points.append(
                (
                    int(round(center + math.cos(angle) * radius)),
                    int(round(center + math.sin(angle) * radius)),
                )
            )
        pygame.draw.polygon(star, color, star_points)

        circle = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(circle, color, (center, center), int(self.shape_radius_px * 0.72))

        target_count = min(
            pygame.mask.from_surface(star).count(),
            pygame.mask.from_surface(circle).count(),
        )
        self._trim_surface_to_pixel_count(star, target_count, center)
        self._trim_surface_to_pixel_count(circle, target_count, center)
        surfaces = {"star": star, "circle": circle}
        counts = {
            name: pygame.mask.from_surface(surface).count()
            for name, surface in surfaces.items()
        }
        return surfaces, counts

    @staticmethod
    def _trim_surface_to_pixel_count(
        surface: pygame.Surface,
        target_count: int,
        center: int,
    ) -> None:
        mask = pygame.mask.from_surface(surface)
        pixels = [
            (x, y)
            for y in range(surface.get_height())
            for x in range(surface.get_width())
            if mask.get_at((x, y))
        ]
        excess = len(pixels) - int(target_count)
        if excess <= 0:
            return
        pixels.sort(
            key=lambda point: (point[0] - center) ** 2 + (point[1] - center) ** 2,
            reverse=True,
        )
        for point in pixels[:excess]:
            surface.set_at(point, (0, 0, 0, 0))

    def _blit_shape(self, shape: str, center: tuple[int, int]) -> None:
        surface = self._decision_shapes[shape]
        rect = surface.get_rect(center=center)
        self.screen.blit(surface, rect)

    def _draw_debug_overlay(
        self,
        force_direction: Direction,
        trial_index: int | None,
        response_time_s: float | None,
    ) -> None:
        lines = [
            f"trial {trial_index}" if trial_index is not None else "trial",
            f"force cue: {force_direction.upper()}",
        ]
        if response_time_s is not None:
            lines.append(f"response time: {response_time_s:.3f}s")
        for idx, line in enumerate(lines):
            surf = self.small_font.render(line, True, (210, 210, 210))
            self.screen.blit(surf, (14, 14 + idx * 24))

    def _triangle(self, center: tuple[int, int], direction: Direction) -> list[tuple[int, int]]:
        cx, cy = center
        radius = self.shape_radius_px
        short = int(radius * 0.82)
        if self.config.axis == "vertical":
            if direction == "cw":
                return [(cx, cy + radius), (cx - short, cy - short), (cx + short, cy - short)]
            return [(cx, cy - radius), (cx - short, cy + short), (cx + short, cy + short)]
        if direction == "cw":
            return [(cx + radius, cy), (cx - short, cy - short), (cx - short, cy + short)]
        return [(cx - radius, cy), (cx + short, cy - short), (cx + short, cy + short)]

    def close(self) -> None:
        try:
            pygame.quit()
        except Exception:
            pass
