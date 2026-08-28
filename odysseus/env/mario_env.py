from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image
from pyboy import PyBoy

from odysseus.env.actions import frame_skip_for_buttons, normalize_buttons, pyboy_pressable_buttons
from odysseus.env.memory import get_level_progress, is_game_over


@dataclass
class StepInfo:
    progress: int
    delta_progress: int
    died: bool
    frame_skip: int
    buttons: list[str]


class MarioEnv:
    """Turn-based Super Mario Land environment following the Odysseus protocol."""

    def __init__(
        self,
        rom_path: str,
        *,
        world: int = 1,
        stage: int = 1,
        upscale_factor: int = 8,
        headless: bool = True,
    ) -> None:
        self.rom_path = rom_path
        self.world = world
        self.stage = stage
        self.upscale_factor = upscale_factor
        self.headless = headless

        self._pyboy: PyBoy | None = None
        self._start_progress = 0
        self._last_progress = 0

    @property
    def pyboy(self) -> PyBoy:
        if self._pyboy is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self._pyboy

    def reset(self) -> Image.Image:
        if self._pyboy is not None:
            self._pyboy.stop()

        window = "null" if self.headless else "SDL2"
        self._pyboy = PyBoy(self.rom_path, window=window)
        self._pyboy.set_emulation_speed(0)

        wrapper = self._pyboy.game_wrapper
        wrapper.start_game(world_level=(self.world, self.stage))
        wrapper.reset_game()

        for _ in range(30):
            if not self._pyboy.tick():
                break

        self._start_progress = get_level_progress(self._pyboy)
        self._last_progress = self._start_progress
        return self._capture_frame()

    def step(self, buttons: list[str]) -> tuple[Image.Image, StepInfo]:
        buttons = normalize_buttons(buttons)
        skip = frame_skip_for_buttons(buttons)
        pressable = pyboy_pressable_buttons(buttons)

        for button in pressable:
            self.pyboy.button_press(button)

        for _ in range(skip):
            if not self.pyboy.tick():
                break

        for button in pressable:
            self.pyboy.button_release(button)

        if pressable:
            if not self.pyboy.tick():
                pass

        progress = get_level_progress(self.pyboy)
        delta = progress - self._last_progress
        self._last_progress = progress

        died = is_game_over(self.pyboy)
        info = StepInfo(
            progress=progress,
            delta_progress=delta,
            died=died,
            frame_skip=skip,
            buttons=buttons,
        )
        return self._capture_frame(), info

    def get_progress_from_start(self) -> int:
        return get_level_progress(self.pyboy) - self._start_progress

    def _capture_frame(self) -> Image.Image:
        frame = self.pyboy.screen.image.copy()
        if self.upscale_factor != 1:
            width = frame.width * self.upscale_factor
            height = frame.height * self.upscale_factor
            frame = frame.resize((width, height), Image.NEAREST)
        return frame

    def close(self) -> None:
        if self._pyboy is not None:
            self._pyboy.stop()
            self._pyboy = None

    def __enter__(self) -> MarioEnv:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
