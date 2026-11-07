"""Small, synchronous boundary around the LLM."""

import time
from functools import lru_cache
from typing import Mapping, Sequence, TypeVar

from openai import OpenAI
from pydantic import BaseModel

import telemetry
from config import AppConfig


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)
ConversationHistory = Sequence[Mapping[str, str]]
MAX_HISTORY_MESSAGES = 6  # Three user/assistant turns.


def _chat_messages(
    instructions: str,
    input_text: str,
    conversation_history: ConversationHistory | None,
) -> list[dict[str, str]]:
    """Build a chat request with at most MAX_HISTORY_MESSAGES/2 conversation turns."""
    history: list[dict[str, str]] = []
    for message in conversation_history or ():
        role = message.get("role")
        content = message.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})

    return [
        {"role": "system", "content": instructions},
        *history[-MAX_HISTORY_MESSAGES:],
        {"role": "user", "content": input_text},
    ]


@lru_cache(maxsize=1)
def get_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Create and cache a client for a configured endpoint."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def generate_text(
    *,
    instructions: str,
    input_text: str,
    config: AppConfig,
    temperature: float,
    conversation_history: ConversationHistory | None = None,
    agent: str = "unknown",
) -> str:
    """Generate plain text with the Chat Completions API."""
    messages = _chat_messages(instructions, input_text, conversation_history)
    started = time.monotonic()
    response = output_text = failure = None
    try:
        response = get_openai_client(
            config.llm_base_url, config.llm_api_key
        ).chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=temperature,
        )
        output_text = response.choices[0].message.content if response.choices else None
        if not output_text:
            raise ValueError("LLM returned no text output")
        return output_text
    except Exception as exc:
        failure = exc
        raise
    finally:
        telemetry.llm_call(
            agent, config.llm_model, (time.monotonic() - started) * 1000, response=response,
            output=output_text, input_text=input_text, instructions=instructions,
            temperature=temperature, history_messages=len(messages) - 2, failure=failure,
        )


def generate_structured(
    *,
    instructions: str,
    input_text: str,
    config: AppConfig,
    temperature: float,
    output_type: type[StructuredOutput],
    conversation_history: ConversationHistory | None = None,
    agent: str = "unknown",
) -> StructuredOutput:
    """Generate and validate a Pydantic model with Chat Completions."""
    messages = _chat_messages(instructions, input_text, conversation_history)
    started = time.monotonic()
    response = failure = None
    try:
        response = get_openai_client(
            config.llm_base_url, config.llm_api_key
        ).chat.completions.parse(
            model=config.llm_model,
            messages=messages,
            temperature=temperature,
            response_format=output_type,
        )
        output_parsed = response.choices[0].message.parsed if response.choices else None
        if output_parsed is None:
            raise ValueError("LLM returned no structured output")
        return output_parsed
    except Exception as exc:
        failure = exc
        raise
    finally:
        raw_output = response.choices[0].message.content if response is not None and response.choices else None
        telemetry.llm_call(
            agent, config.llm_model, (time.monotonic() - started) * 1000, response=response,
            output=raw_output, input_text=input_text, instructions=instructions,
            temperature=temperature, history_messages=len(messages) - 2, failure=failure,
        )
