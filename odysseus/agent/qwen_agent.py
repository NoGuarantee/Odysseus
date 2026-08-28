from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from odysseus.agent.parser import parse_action
from odysseus.agent.prompt import ODYSSEUS_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    buttons: list[str]
    raw_text: str
    parse_error: bool


class QwenAgent:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 1024,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        logger.info("Loading model %s", model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()
        self.last_raw_response = ""

    def act(self, frame: Image.Image) -> AgentResponse:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": ODYSSEUS_PROMPT},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self.last_raw_response = raw_text

        buttons, parse_error = parse_action(raw_text)
        if parse_error:
            logger.warning("Failed to parse model output, using noop. Output: %s", raw_text[:500])

        return AgentResponse(buttons=buttons, raw_text=raw_text, parse_error=parse_error)


class MockAgent:
    """Deterministic agent for pipeline smoke tests without GPU inference."""

    _SCRIPT = [
        ["right"],
        ["right", "a"],
        ["right"],
        ["noop"],
        ["right", "b"],
    ]

    def __init__(self) -> None:
        self.last_raw_response = ""
        self._step = 0

    def act(self, frame: Image.Image) -> AgentResponse:
        del frame
        buttons = self._SCRIPT[self._step % len(self._SCRIPT)]
        self._step += 1
        raw = (
            "<perception>Mario on a platform.</perception>"
            "<reasoning>Move forward.</reasoning>"
            f"<answer>{buttons}</answer>"
        )
        self.last_raw_response = raw
        parsed, parse_error = parse_action(raw)
        return AgentResponse(buttons=parsed, raw_text=raw, parse_error=parse_error)
