from odysseus.agent.parser import parse_action
from odysseus.env.actions import frame_skip_for_buttons, normalize_buttons


def test_parse_action_from_answer_tag():
    text = (
        "<perception>Mario on ground.</perception>"
        "<reasoning>Jump over pipe.</reasoning>"
        "<answer>['right', 'a']</answer>"
    )
    buttons, error = parse_action(text)
    assert buttons == ["right", "a"]
    assert error is False


def test_parse_action_fallback_on_missing_answer():
    buttons, error = parse_action("just move right")
    assert buttons == ["noop"]
    assert error is True


def test_normalize_buttons_limits_to_two():
    assert normalize_buttons(["right", "a", "b"]) == ["right", "a"]


def test_frame_skip_jump():
    assert frame_skip_for_buttons(["right", "a"]) == 15
    assert frame_skip_for_buttons(["right"]) == 5
    assert frame_skip_for_buttons(["noop"]) == 5
