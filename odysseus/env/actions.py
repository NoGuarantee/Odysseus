from __future__ import annotations

ALLOWED_BUTTONS = frozenset({"a", "b", "up", "down", "left", "right", "noop"})
MAX_SIMULTANEOUS_BUTTONS = 2

PYBOY_BUTTON_MAP = {
    "a": "a",
    "b": "b",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


def normalize_buttons(buttons: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for button in buttons:
        key = button.strip().lower()
        if key not in ALLOWED_BUTTONS:
            raise ValueError(f"Invalid button: {button!r}")
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
        if len(normalized) >= MAX_SIMULTANEOUS_BUTTONS:
            break
    if not normalized:
        return ["noop"]
    if normalized == ["noop"]:
        return ["noop"]
    return [b for b in normalized if b != "noop"] or ["noop"]


def frame_skip_for_buttons(buttons: list[str]) -> int:
    return 15 if "a" in buttons else 5


def pyboy_pressable_buttons(buttons: list[str]) -> list[str]:
    if buttons == ["noop"]:
        return []
    return [PYBOY_BUTTON_MAP[b] for b in buttons if b in PYBOY_BUTTON_MAP]
