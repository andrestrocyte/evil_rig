"""Minimal pygame display (compatible with screen_shape_force_discount geometry)."""

from __future__ import annotations

import math
from typing import Literal

import pygame

from hf_dm.config import DisplayConfig

Direction = Literal["cw", "ccw"]


class TaskDisplay:
    def __init__(self, config: DisplayConfig, display_width: int = 480, display_height: int = 1920):
        self.config = config
        self.width = display_width
        self.height = display_height
        pygame.init()
        pygame.font.init()
        flags = pygame.NOFRAME
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

    def fill_black(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.display.flip()

    def show_black(self, duration_s: float) -> None:
        end_t = pygame.time.get_ticks() / 1000.0 + duration_s
        while pygame.time.get_ticks() / 1000.0 < end_t:
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
        px = (100.0 / 0.114) * (math.sin(radians) / math.sin(math.radians(90.0 - magnitude_deg)))
        px = int(round(px))
        return px if signed_deg >= 0 else -px

    def draw_trial(
        self,
        force_direction: Direction,
        signed_position_deg: float,
        outcome: str | None = None,
        trial_index: int | None = None,
    ) -> None:
        self.screen.fill((0, 0, 0))
        cx, cy = self.width // 2, self.height // 2
        offset = self._offset_px(signed_position_deg)
        if self.config.rotated_choice_axis:
            center = (cx, cy + offset)
        else:
            center = (cx + offset, cy)

        if outcome == "matched":
            color = self.config.matched_rgb
        elif outcome == "opposite":
            color = self.config.unmatched_rgb
        elif outcome == "timeout":
            color = self.config.timeout_rgb
        else:
            color = (245, 245, 245)

        radius = 90
        points = self._triangle(center, radius, force_direction)
        pygame.draw.polygon(self.screen, color, points)
        if trial_index is not None:
            surf = self.font.render(f"trial {trial_index}", True, (200, 200, 200))
            self.screen.blit(surf, (14, 14))
        pygame.display.flip()

    def _triangle(self, center: tuple[int, int], radius: int, direction: Direction) -> list[tuple[int, int]]:
        cx, cy = center
        short = int(radius * 0.82)
        if self.config.rotated_choice_axis:
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
