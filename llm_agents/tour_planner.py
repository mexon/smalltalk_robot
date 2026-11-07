"""
Tour Planner Agent — themed tour builder.

Selects and orders locations for themed tours using structured output.
"""

import logging
from typing import Mapping, Sequence

from config import AppConfig
from models.tour import TourPlan
from prompts.tour_planner import TOUR_PLANNER_SYSTEM_PROMPT
from llm_agents.openai_client import generate_structured

logger = logging.getLogger(__name__)


def plan_tour(
    available_locations: list[dict],
    language: str,
    config: AppConfig,
    theme: str | None = None,
    conversation_history: Sequence[Mapping[str, str]] | None = None,
) -> TourPlan:
    """
    Plan a themed museum tour by selecting and ordering locations.

    Args:
        available_locations: List of dicts with 'location_id', 'name', 'themes',
                            and optionally 'snippet' (RAG context)
        language: Short language code ("en", "de" or "zh")
        theme: Visitor's requested theme, or None for a general tour
        config: Application configuration

    Returns:
        TourPlan with ordered location selections and justifications
    """
    # Format locations with snippets for the prompt
    location_lines = []
    for loc in available_locations:
        name = loc.get("name", {}).get(language) or loc.get("name", {}).get("en", loc["location_id"])
        themes = ", ".join(loc.get("themes", []))
        snippet = loc.get("snippet", "")
        line = f"- {loc['location_id']}: {name}"
        if themes:
            line += f" (themes: {themes})"
        if snippet:
            line += f"\n  Context: {snippet[:200]}"
        location_lines.append(line)

    locations_text = "\n".join(location_lines) if location_lines else "(No locations available)"
    effective_theme = theme or "general museum tour — show the best highlights"

    prompt = (
        "AVAILABLE LOCATIONS (with snippets from the knowledge base):\n"
        f"{locations_text}\n\n"
        "VISITOR'S REQUESTED THEME:\n"
        f"{effective_theme}\n\n"
        "Plan the tour now."
    )
    plan = generate_structured(
        instructions=TOUR_PLANNER_SYSTEM_PROMPT,
        input_text=prompt,
        config=config,
        temperature=0.2,
        output_type=TourPlan,
        conversation_history=conversation_history,
        agent="tour_planner",
    )

    logger.info(
        f"Tour Planner: theme='{plan.theme}' locations={[l.location_id for l in plan.locations]}"
    )
    return plan
