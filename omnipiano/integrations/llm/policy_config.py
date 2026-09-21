"""Configuration for LLM evaluation policies."""

from __future__ import annotations

from dataclasses import dataclass

from omnipiano.integrations.policy_config import PolicyConfig


@dataclass(kw_only=True)
class LLMConfig(PolicyConfig):
    action_space: object
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None  # None: OpenAI SDK reads OPENAI_API_KEY.
    system_prompt: str | None = None
    thinking: bool | None = None
    temperature: float = 0
    max_tokens: int = 16_384


@dataclass(kw_only=True)
class KeyframeConfig(PolicyConfig):
    env: object
    action_space: object
    path: str


@dataclass(kw_only=True)
class KeyframeOptimizerConfig(LLMConfig):
    env: object
    path: str
    num_optimization_steps: int = 3
    target_f1: float = 0.9
    practice_seed: int = 0
    resume: bool = False
