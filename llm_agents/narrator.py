"""
Narrator Agent — generates context-aware transition sentences for tours.

Produces only the "glue" text around stops (first/next/last/only framing,
Q&A closers). All factual content is handled by the Description Agent.
"""

import logging
from typing import Literal

from config import AppConfig
from prompts.narrator import NARRATOR_SYSTEM_PROMPT
from llm_agents.openai_client import generate_text

logger = logging.getLogger(__name__)

Position = Literal["only", "first", "middle", "last"]
Phase = Literal["before_navigation", "qa_closer"]


def generate_transition(
    location_name: str,
    language: str,
    position: Position,
    total_stops: int,
    phase: Phase,
    config: AppConfig,
) -> str:
    """
    Generate a short transition sentence for the current tour step.

    Args:
        location_name: Display name of the current stop (in target language).
        language: Short language code ("en", "de" or "zh").
        position: Where this stop sits in the tour sequence.
        total_stops: Total number of stops in the current tour.
        phase: "before_navigation" (intro before we head there) or
               "qa_closer" (invitation to ask questions + transition).
        config: Application configuration.

    Returns:
        Spoken transition text. Never empty; falls back to a deterministic
        phrase if the LLM call fails.
    """
    # Put tour-wide values before per-stop values to extend the shared prefix.
    user_msg = (
        f"language: {language}\n"
        f"total_stops: {total_stops}\n"
        f"phase: {phase}\n"
        f"position: {position}\n"
        f"location_name: {location_name}\n"
    )

    if phase == "qa_closer":
        if position in ("first", "middle"):
            user_msg += (
                "CONSTRAINT: This visitor is on a planned multi-stop tour. "
                "The ONLY forward option is continuing to the next stop. "
                "Do NOT mention canceling, ending, or leaving the tour.\n"
            )
        elif position == "last":
            user_msg += (
                "CONSTRAINT: This is the last stop. "
                "Offer to wrap up or finish naturally. "
                "Do NOT use the word cancel/abbrechen.\n"
            )

    user_msg += "Produce the transition sentence now."

    try:
        text = (
            generate_text(
                instructions=NARRATOR_SYSTEM_PROMPT,
                input_text=user_msg,
                config=config,
                temperature=0.4,
                agent="narrator",
            )
            or ""
        ).strip()
        if not text:
            raise ValueError("Narrator returned empty output")
        logger.info(
            f"Narrator: {phase}/{position} '{location_name}' ({language}) -> "
            f"{text[:80]}{'...' if len(text) > 80 else ''}"
        )
        return text
    except Exception as e:
        logger.warning(f"⚠️  Narrator failed ({e}), using fallback phrase")
        return _fallback_phrase(location_name, language, position, phase)


# ---------------------------------------------------------------------------
# Deterministic fallback — used only if the LLM call fails
# ---------------------------------------------------------------------------

_FALLBACKS = {
    ("before_navigation", "first", "en"):  "Our first stop is {name}. Please follow me.",
    ("before_navigation", "first", "de"):  "Unsere erste Station ist {name}. Bitte folgen Sie mir.",
    ("before_navigation", "middle", "en"): "Next, let's head to {name}. Please follow me.",
    ("before_navigation", "middle", "de"): "Als Nächstes gehen wir zu {name}. Bitte folgen Sie mir.",
    ("before_navigation", "last", "en"):   "Our final stop is {name}. Please follow me.",
    ("before_navigation", "last", "de"):   "Unsere letzte Station ist {name}. Bitte folgen Sie mir.",
    ("before_navigation", "only", "en"):   "Let's head to {name}. Please follow me.",
    ("before_navigation", "only", "de"):   "Gehen wir zu {name}. Bitte folgen Sie mir.",
    ("before_navigation", "first", "zh"):  "我們的第一站是{name}，請跟我來。",
    ("before_navigation", "middle", "zh"): "接下來，我們前往{name}，請跟我來。",
    ("before_navigation", "last", "zh"):   "我們的最後一站是{name}，請跟我來。",
    ("before_navigation", "only", "zh"):   "我們一起去{name}吧，請跟我來。",

    ("qa_closer", "first", "en"):  "Do you have any questions about this work, or shall we continue to the next stop?",
    ("qa_closer", "first", "de"):  "Haben Sie Fragen zu diesem Werk, oder sollen wir zur nächsten Station weitergehen?",
    ("qa_closer", "middle", "en"): "Do you have any questions about this work, or shall we continue to the next stop?",
    ("qa_closer", "middle", "de"): "Haben Sie Fragen zu diesem Werk, oder sollen wir zur nächsten Station weitergehen?",
    ("qa_closer", "last", "en"):   "Do you have any questions about this work, or shall we wrap up the tour?",
    ("qa_closer", "last", "de"):   "Haben Sie Fragen zu diesem Werk, oder sollen wir die Tour abschließen?",
    ("qa_closer", "only", "en"):   "Do you have any questions about this work, or is there anything else I can help you with?",
    ("qa_closer", "only", "de"):   "Haben Sie Fragen zu diesem Werk, oder kann ich sonst noch etwas für Sie tun?",
    ("qa_closer", "first", "zh"):  "關於這件作品，您有什麼問題嗎？還是我們繼續前往下一站？",
    ("qa_closer", "middle", "zh"): "關於這件作品，您有什麼問題嗎？還是我們繼續前往下一站？",
    ("qa_closer", "last", "zh"):   "關於這件作品，您有什麼問題嗎？還是我們就為今天的導覽畫下句點呢？",
    ("qa_closer", "only", "zh"):   "關於這件作品，您有什麼問題嗎？或是還有什麼我可以幫忙的？",
}


def _fallback_phrase(location_name: str, language: str, position: Position, phase: Phase) -> str:
    lang = language if language in ("en", "de", "zh") else "en"
    template = _FALLBACKS.get((phase, position, lang)) or _FALLBACKS[(phase, "only", lang)]
    return template.format(name=location_name)


def compute_position(index: int, total_stops: int) -> Position:
    """Map (index, total_stops) to a position label."""
    if total_stops <= 1:
        return "only"
    if index == 0:
        return "first"
    if index == total_stops - 1:
        return "last"
    return "middle"
