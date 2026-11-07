"""
Description Generator Agent — location narration.

Generates spoken descriptions for tour stops using RAG context
and location-specific prompt instructions.
"""

import logging

from config import AppConfig
from prompts.description import DESCRIPTION_SYSTEM_PROMPT
from llm_agents.openai_client import generate_text

logger = logging.getLogger(__name__)

_LANG_NAMES = {"en": "English", "de": "German", "zh": "Traditional Chinese (Taiwan)"}


def generate_description(
    location_name: str,
    context_chunks: list[str],
    language: str,
    config: AppConfig,
    prompt_suffix: str = "",
) -> str:
    """
    Generate a spoken description for a tour location.

    Args:
        location_name: Display name of the location
        context_chunks: Retrieved document chunks from RAG
        language: Short language code ("en", "de" or "zh")
        prompt_suffix: Extra instructions for this specific location
        config: Application configuration

    Returns:
        Spoken description text (4-6 sentences)
    """
    lang_name = _LANG_NAMES.get(language, "English")
    context = "\n\n".join(context_chunks) if context_chunks else "(No context available)"

    # Keep instructions static for KV caching; all per-stop values belong in the
    # user message. Context precedes the final command because it is the largest
    # reusable portion when a description is retried.
    user_message = (
        f"LANGUAGE:\n{lang_name}\n\n"
        f"LOCATION:\n{location_name}\n\n"
        "LOCATION-SPECIFIC INSTRUCTIONS:\n"
        f"{prompt_suffix or '(No additional instructions)'}\n\n"
        f"KNOWLEDGE-BASE CONTEXT:\n{context}\n\n"
        "Describe this location now."
    )

    description = generate_text(
        instructions=DESCRIPTION_SYSTEM_PROMPT,
        input_text=user_message,
        config=config,
        temperature=0.5,
        agent="description",
    )

    logger.info(f"Description Agent: generated for '{location_name}' ({len(description)} chars)")
    return description
