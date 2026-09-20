"""Minimal LLM-to-Gym policy adapter."""

from __future__ import annotations

import json
import re

import numpy as np
from openai import OpenAI

from .policy_config import LLMConfig


class OpenAIClient:
    """OpenAI SDK client for OpenAI-compatible chat-completion endpoints."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        temperature: float = 0,
        max_tokens: int = 16_384,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.thinking = thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def __call__(self, prompt) -> str:
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else prompt
        )
        message = self.chat(messages)
        content = message.content
        if not content:
            reasoning = getattr(message, "reasoning_content", None)
            raise RuntimeError(
                "LLM returned empty content: "
                f"reasoning_content={bool(reasoning)}"
            )
        return content

    def chat(self, messages, *, tools=None):
        """Return one assistant message, optionally with function calls."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking is not None:
            mode = "enabled" if self.thinking else "disabled"
            kwargs["extra_body"] = {"thinking": {"type": mode}}
        if tools:
            kwargs["tools"] = tools
        response = self.client.chat.completions.create(**kwargs)
        if response.usage is not None:
            for name in self.token_usage:
                self.token_usage[name] += int(getattr(response.usage, name, 0) or 0)
        return response.choices[0].message


class LLMPolicy:
    """Policy exposing the ``predict`` method expected by ``evaluate_policy``."""

    def __init__(self, config: LLMConfig):
        self.action_space = config.action_space
        self.client = OpenAIClient(
            config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            thinking=config.thinking,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.system_prompt = config.system_prompt or (
            "You control a piano-playing robot. Return only a JSON array of "
            "floating-point actions in the normalized range [-1, 1]."
        )

    def predict(self, observation, deterministic: bool = True):
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        prompt = (
            f"{self.system_prompt}\n"
            f"The observation has {observation.size} values:\n"
            f"{json.dumps(observation.tolist())}\n"
            f"Return exactly {self.action_space.shape[0]} action values."
        )
        response = self.client(prompt)
        print(f"LLM response: {response}")
        action = self._parse_action(response)
        print(f"LLM action: {action}")
        return np.clip(action, self.action_space.low, self.action_space.high), None

    def _parse_action(self, response: str) -> np.ndarray:
        match = re.search(r"\[[^\]]*\]", response, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"LLM response does not contain a JSON action array: {response}")
        action = np.asarray(json.loads(match.group(0)), dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"LLM returned action shape {action.shape}; "
                f"expected {self.action_space.shape}"
            )
        return action
