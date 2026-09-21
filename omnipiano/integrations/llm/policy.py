"""Minimal LLM-to-Gym policy adapter."""

from __future__ import annotations

import copy
import json
import re

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

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


class ClaudeAgent(OpenAIClient):
    """Claude-compatible agent using the OpenAI-compatible router endpoint.

    The router's Claude adapter expects Anthropic-style tool schemas (name and
    input_schema at the top level), while the rest of the optimizer uses the
    OpenAI chat-completions message format.
    """

    @staticmethod
    def _claude_tools(tools):
        converted = []
        for tool in tools or ():
            if tool.get("type") != "function" or "function" not in tool:
                raise ValueError(f"unsupported Claude tool: {tool.get('type')}")
            function = tool["function"]
            converted.append({
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            })
        return converted

    @staticmethod
    def _claude_messages(messages):
        """Convert OpenAI tool-call history to Claude content blocks."""
        system_parts = []
        converted = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                if isinstance(content, str) and content:
                    system_parts.append(content)
                continue

            if role == "tool":
                converted_message = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": content if isinstance(content, str) else json.dumps(content),
                    }],
                }
            elif role == "assistant" and message.get("tool_calls"):
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for call in message["tool_calls"]:
                    function = call["function"]
                    arguments = function.get("arguments", "{}")
                    try:
                        tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
                    except json.JSONDecodeError:
                        tool_input = {"_raw_arguments": arguments}
                    blocks.append({
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": tool_input,
                    })
                converted_message = {"role": "assistant", "content": blocks}
            else:
                converted_message = {"role": role, "content": content or ""}

            # Claude accepts multiple tool results in one user message. Merge
            # adjacent user content-block messages so multi-tool calls remain
            # a single turn and message roles alternate cleanly.
            if (
                converted
                and converted[-1]["role"] == converted_message["role"] == "user"
                and isinstance(converted[-1]["content"], list)
                and isinstance(converted_message["content"], list)
            ):
                converted[-1]["content"].extend(converted_message["content"])
            else:
                converted.append(converted_message)
        return system_parts, converted

    def chat(self, messages, *, tools=None):
        """Call the router with Claude's top-level tool schema."""
        system_parts, chat_messages = self._claude_messages(messages)
        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": self.max_tokens,
        }
        extra_body = {}
        if system_parts:
            extra_body["system"] = "\n\n".join(system_parts)
        if self.thinking is not None:
            mode = "enabled" if self.thinking else "disabled"
            extra_body["thinking"] = {"type": mode}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = self._claude_tools(tools)
        response = self.client.chat.completions.create(**kwargs)
        if response.usage is not None:
            for name in self.token_usage:
                self.token_usage[name] += int(getattr(response.usage, name, 0) or 0)
        return response.choices[0].message


class OpenAIResponseClient(OpenAIClient):
    """Responses API adapter exposing the existing chat-message interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_items = []
        self._expected_messages = []

    @staticmethod
    def _response_tools(tools):
        result = []
        for tool in tools or ():
            if tool.get("type") != "function" or "function" not in tool:
                raise ValueError(f"unsupported Responses tool: {tool.get('type')}")
            result.append({"type": "function", **tool["function"]})
        return result

    @staticmethod
    def _response_input(messages):
        result = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                output = message.get("content", "")
                if not isinstance(output, str):
                    output = json.dumps(output, separators=(",", ":"))
                result.append({
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": output,
                })
                continue

            content = message.get("content")
            if content:
                result.append({"role": role, "content": content})
            for call in message.get("tool_calls", ()):
                function = call["function"]
                result.append({
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": function["name"],
                    "arguments": function["arguments"],
                })
        return result

    @staticmethod
    def _chat_message(response):
        tool_calls = []
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append({
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                })
        return ChatCompletionMessage.model_validate({
            "role": "assistant",
            "content": response.output_text or None,
            "tool_calls": tool_calls or None,
        })

    def chat(self, messages, *, tools=None):
        """Call Responses while returning a Chat Completions-compatible message."""
        prefix_length = len(self._expected_messages)
        if messages[:prefix_length] != self._expected_messages:
            self._input_items = []
            self._expected_messages = []
            prefix_length = 0

        self._input_items.extend(self._response_input(messages[prefix_length:]))
        kwargs = {
            "model": self.model,
            "input": self._input_items,
            "max_output_tokens": self.max_tokens,
        }
        response_tools = self._response_tools(tools)
        if response_tools:
            kwargs["tools"] = response_tools
        if self.thinking is not None:
            kwargs["reasoning"] = {"effort": "high" if self.thinking else "low"}

        response = self.client.responses.create(**kwargs)
        self._input_items.extend(
            item.model_dump(exclude_none=True, mode="json") for item in response.output
        )
        if response.usage is not None:
            usage_names = {
                "prompt_tokens": "input_tokens",
                "completion_tokens": "output_tokens",
                "total_tokens": "total_tokens",
            }
            for name, response_name in usage_names.items():
                self.token_usage[name] += int(getattr(response.usage, response_name, 0) or 0)

        message = self._chat_message(response)
        self._expected_messages = copy.deepcopy(messages)
        self._expected_messages.append(message.model_dump(exclude_none=True, mode="json"))
        return message


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
