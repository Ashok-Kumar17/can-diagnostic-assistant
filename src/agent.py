"""
agent.py

The agent loop: takes a natural-language question, lets the LLM decide which
CANSignalTools function(s) to call (possibly several, in sequence), feeds the
results back, and returns a final grounded answer.

Provider-agnostic by design: Anthropic and OpenAI's function-calling APIs use
slightly different request/response shapes, but the same conceptual loop
(model proposes a tool call -> we execute it -> we feed the result back ->
repeat until the model returns plain text). This file isolates that
difference into two small adapter classes so the orchestration logic
(LLMAgent.ask) doesn't care which provider is behind it.

Set ANTHROPIC_API_KEY or OPENAI_API_KEY as an environment variable depending
on which provider you want to use.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from src.tools import CANSignalTools

SYSTEM_PROMPT = """You are a vehicle diagnostics assistant. You answer questions about \
a CAN bus log from an EV motor controller, BMS, and diagnostics bus by calling the \
provided tools to look up real signal data and fault codes. Never guess a numeric \
value -- always call a tool to get it. When you've gathered enough information, \
give a clear, concise diagnostic explanation, citing the actual timestamps and \
values you retrieved. If asked why a fault occurred, look at the signal trend \
leading up to the fault timestamp, not just the fault event itself.

IMPORTANT: Call exactly ONE tool per response. Never attempt to call multiple \
tools in a single turn. Wait for each tool's result before deciding whether you \
need to call another one."""


class AnthropicAdapter:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def run_turn(self, messages: list[dict], tool_specs: list[dict]):
        """Returns (assistant_message_blocks, stop_reason)."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tool_specs,  # Anthropic accepts {name, description, input_schema} directly
        )
        return response.content, response.stop_reason

    @staticmethod
    def tool_use_blocks(content_blocks):
        return [b for b in content_blocks if b.type == "tool_use"]

    @staticmethod
    def text_blocks(content_blocks):
        return [b.text for b in content_blocks if b.type == "text"]

    @staticmethod
    def build_assistant_message(content_blocks):
        return {"role": "assistant", "content": content_blocks}

    @staticmethod
    def build_tool_result_message(tool_use_id, result: dict):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result),
                }
            ],
        }


class OpenAIAdapter:
    """
    Works with OpenAI directly, or any OpenAI-compatible endpoint (e.g. Groq's
    free tier, which speaks the same chat-completions + tool-calling format).
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        import openai

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or "gpt-4o"

    @staticmethod
    def _to_openai_tools(tool_specs: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
            for spec in tool_specs
        ]

    def run_turn(self, messages: list[dict], tool_specs: list[dict]):
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=self._to_openai_tools(tool_specs),
            parallel_tool_calls=False,  # one tool call at a time -- more reliable on
                                         # smaller/faster models like Groq's Llama, which
                                         # can mis-format multi-call attempts as raw text
                                         # instead of structured tool_calls (the crash you hit)
        )
        choice = response.choices[0]
        stop_reason = "tool_use" if choice.message.tool_calls else "end_turn"
        return choice.message, stop_reason

    @staticmethod
    def tool_use_blocks(message):
        return message.tool_calls or []

    @staticmethod
    def text_blocks(message):
        return [message.content] if message.content else []

    @staticmethod
    def build_assistant_message(message):
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls,
        }

    @staticmethod
    def build_tool_result_message(tool_call_id, result: dict):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result),
        }


def _select_adapter():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicAdapter()
    if os.environ.get("GROQ_API_KEY"):
        # Groq's free tier: OpenAI-compatible endpoint, no credit card needed.
        # console.groq.com -> API Keys -> create one.
        return OpenAIAdapter(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIAdapter()
    raise RuntimeError(
        "No API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY "
        "(Groq is free, no credit card: console.groq.com)."
    )


class LLMAgent:
    def __init__(self, signal_tools: CANSignalTools, adapter: Optional[object] = None):
        self.tools = signal_tools
        self.adapter = adapter or _select_adapter()
        self.tool_specs = signal_tools.as_tool_specs()

    def ask(self, question: str, max_steps: int = 6) -> str:
        """
        Runs the tool-calling loop for a single user question and returns the
        final text answer. max_steps caps how many tool-call round-trips we
        allow before giving up, so a confused model can't loop forever.
        """
        messages = [{"role": "user", "content": question}]

        for _ in range(max_steps):
            try:
                content, stop_reason = self.adapter.run_turn(messages, self.tool_specs)
            except Exception as e:
                # Smaller/faster models occasionally mis-format a tool call (e.g. emit
                # raw "<function=...>" text instead of a structured call). Rather than
                # crashing the whole conversation, nudge it and retry once.
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last attempt to call a tool was malformed "
                        f"({e}). Call exactly one tool using the structured tool-calling "
                        "interface, not raw text."
                    ),
                })
                try:
                    content, stop_reason = self.adapter.run_turn(messages, self.tool_specs)
                except Exception as e2:
                    return f"(The model failed to format a valid tool call twice in a row: {e2})"

            messages.append(self.adapter.build_assistant_message(content))

            tool_calls = self.adapter.tool_use_blocks(content)
            if not tool_calls:
                texts = self.adapter.text_blocks(content)
                return "\n".join(texts) if texts else "(no response text)"

            # Execute every requested tool call and feed results back
            for call in tool_calls:
                name, raw_input, call_id = self._unpack_tool_call(call)
                result = self.tools.call(name, raw_input)
                messages.append(self.adapter.build_tool_result_message(call_id, result))

        return "(stopped after max_steps tool-call round-trips without a final answer)"

    @staticmethod
    def _unpack_tool_call(call):
        """Normalizes Anthropic's tool_use block vs OpenAI's tool_call object."""
        if hasattr(call, "input"):  # Anthropic tool_use block
            return call.name, call.input, call.id
        # OpenAI tool_call object
        return call.function.name, json.loads(call.function.arguments), call.id
