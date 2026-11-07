"""
Q&A Agent — RAG-grounded question answering.

Answers visitor questions using retrieved document chunks.
Returns plain spoken text (no structured output).
"""

import logging
from typing import Mapping, Sequence

from config import AppConfig
from prompts.qa import QA_SYSTEM_PROMPT
from llm_agents.openai_client import generate_text

logger = logging.getLogger(__name__)

_LANG_NAMES = {"en": "English", "de": "German", "zh": "Traditional Chinese (Taiwan)"}

_ASSISTANT_SNIPPET_CHARS = 220


def expand_retrieval_query(
    current: str,
    conversation_history: Sequence[Mapping[str, str]] | None = None,
    is_followup: bool = False,
) -> str:
    """Build a RAG query that still finds the prior topic on follow-ups."""
    current = (current or "").strip()
    if not current or not is_followup:
        return current

    last_user = ""
    last_assistant = ""
    for message in reversed(tuple(conversation_history or ())):
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            if not last_user:
                last_user = content
        elif role == "assistant" and not last_assistant:
            last_assistant = content
        if last_user and last_assistant:
            break

    parts: list[str] = []
    if last_assistant:
        snippet = " ".join(last_assistant.split())
        if len(snippet) > _ASSISTANT_SNIPPET_CHARS:
            snippet = snippet[:_ASSISTANT_SNIPPET_CHARS]
        parts.append(snippet)
    if last_user:
        parts.append(last_user)
    parts.append(current)
    return " ".join(parts)


def collect_qa_documents(
    retriever,
    question: str,
    language: str,
    conversation_history: Sequence[Mapping[str, str]] | None = None,
    location_rag_query: str = "",
    is_followup: bool = False,
) -> list[dict]:
    """Retrieve and dedupe documents for a question, expanding follow-up queries."""
    if retriever is None:
        return []

    query = expand_retrieval_query(
        question, conversation_history, is_followup=is_followup
    )
    documents: list[dict] = []
    seen: set[str] = set()
    for purpose, rag_query in (("qa", query), ("qa_stop_context", location_rag_query)):
        if not rag_query:
            continue
        for doc in retriever.retrieve(rag_query, language=language, purpose=purpose):
            content = (doc.get("content") or "").strip()
            if content and content not in seen:
                seen.add(content)
                documents.append(doc)
    return documents


def collect_qa_chunks(
    retriever,
    question: str,
    language: str,
    conversation_history: Sequence[Mapping[str, str]] | None = None,
    location_rag_query: str = "",
    is_followup: bool = False,
) -> list[str]:
    """Retrieve and dedupe chunks for a question, expanding follow-up queries."""
    return [
        doc["content"]
        for doc in collect_qa_documents(
            retriever,
            question,
            language,
            conversation_history=conversation_history,
            location_rag_query=location_rag_query,
            is_followup=is_followup,
        )
    ]


def answer_question(
    question: str,
    context_chunks: list[str],
    language: str,
    config: AppConfig,
    position: str = "only",
    location_name: str = "",
    conversation_history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """Answer a visitor's question using RAG context."""
    lang_name = _LANG_NAMES.get(language, "English")
    context = "\n\n".join(context_chunks) if context_chunks else "(No context available)"

    # Keep instructions static for KV caching. Put the question last so repeat
    # questions at the same stop share the location, language, position, and RAG
    # context prefix.
    user_message = (
        f"LANGUAGE:\n{lang_name}\n\n"
        f"TOUR POSITION:\n{position}\n\n"
        f"CURRENT LOCATION:\n{location_name or '(No current location)'}\n\n"
        f"KNOWLEDGE-BASE CONTEXT:\n{context}\n\n"
        f"VISITOR'S QUESTION:\n{question}"
    )

    answer = generate_text(
        instructions=QA_SYSTEM_PROMPT,
        input_text=user_message,
        config=config,
        temperature=0.3,
        conversation_history=conversation_history,
        agent="qa",
    )
    logger.info(f"QA Agent: answered ({len(answer)} chars)")
    return answer
