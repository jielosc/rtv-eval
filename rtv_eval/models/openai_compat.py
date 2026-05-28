from __future__ import annotations

import logging
import os

import openai

from rtv_eval.config import ModelConfig
from rtv_eval.models.base import ModelCaller

logger = logging.getLogger(__name__)


class OpenAICompatCaller(ModelCaller):
    def __init__(self, config: ModelConfig):
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise ValueError(
                f"API key env var '{config.api_key_env}' is not set or empty."
            )
        self.client = openai.AsyncOpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=config.timeout,
        )
        self.config = config

    async def call(self, question: str, frame_b64_list: list[str]) -> str:
        content: list[dict] = []
        for b64 in frame_b64_list:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": self.config.detail,
                },
            })
        content.append({"type": "text", "text": question})

        messages: list[dict] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": content})

        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content or ""
