from __future__ import annotations

import ast
import re

from odysseus.env.actions import ALLOWED_BUTTONS, normalize_buttons

_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
_LIST_RE = re.compile(r"\[[^\]]*\]")


def _extract_answer_payload(text: str) -> str | None:
    match = _ANSWER_TAG_RE.search(text)
    if match:
        return match.group(1).strip()
    list_match = _LIST_RE.search(text)
    if list_match:
        return list_match.group(0)
    return None


def _parse_button_list(payload: str) -> list[str]:
    payload = payload.strip()
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        tokens = re.findall(r"[a-z]+", payload.lower())
        parsed = tokens

    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip().lower().strip("'\"") for item in parsed]
    raise ValueError(f"Unsupported answer format: {payload!r}")


def parse_action(text: str) -> tuple[list[str], bool]:
    """Parse model output into validated buttons.

    Returns:
        (buttons, parse_error) where parse_error indicates fallback was used.
    """
    try:
        payload = _extract_answer_payload(text)
        if payload is None:
            raise ValueError("No <answer> block found")
        buttons = _parse_button_list(payload)
        return normalize_buttons(buttons), False
    except (ValueError, TypeError):
        return ["noop"], True
