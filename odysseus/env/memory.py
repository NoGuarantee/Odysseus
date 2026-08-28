from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyboy import PyBoy


ADDR_MARIO_X = 0xC202
ADDR_LEVEL_BLOCK = 0xC0AB
ADDR_GAME_OVER = 0xC0A4
ADDR_LIVES = 0xC0A3


def get_mario_screen_x(pyboy: PyBoy) -> int:
    return int(pyboy.memory[ADDR_MARIO_X])


def get_level_block(pyboy: PyBoy) -> int:
    return int(pyboy.memory[ADDR_LEVEL_BLOCK])


def get_level_progress(pyboy: PyBoy) -> int:
    """Global horizontal progress, matching PyBoy GameWrapperSuperMarioLand."""
    level_block = get_level_block(pyboy)
    mario_x = get_mario_screen_x(pyboy)
    scx = pyboy.screen.tilemap_position_list[16][0]
    return level_block * 16 + (scx - 7) % 16 + mario_x


def is_game_over(pyboy: PyBoy) -> bool:
    return int(pyboy.memory[ADDR_GAME_OVER]) == 0x39


def get_lives(pyboy: PyBoy) -> int:
    value = int(pyboy.memory[ADDR_LIVES])
    return (value >> 4) * 10 + (value & 0x0F)
