from odysseus.agent.parser import parse_action

__all__ = ["QwenAgent", "parse_action"]


def __getattr__(name: str):
    if name == "QwenAgent":
        from odysseus.agent.qwen_agent import QwenAgent

        return QwenAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
