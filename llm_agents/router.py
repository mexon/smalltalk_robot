"""
Router Agent — intent classification via structured output & local fast-path.

Classifies user utterances into one of 9 intents using a hybrid architecture:
1. Local Fast-Path (<0.001s): Instant rule-based classification for clear navigation,
   exit, continue, repeat, and full-tour commands.
2. OpenAI Cloud Fallback: Comprehensive LLM classification with structured Pydantic
   output for complex queries, open-domain questions, theme requests, and typos.
"""

import logging
import re
import time
from typing import List, Optional

import telemetry
from config import AppConfig
from models.intents import IntentResult
from prompts.router import ROUTER_SYSTEM_PROMPT
from llm_agents.openai_client import generate_structured

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exhibit Aliases & Fast-Path Dictionaries
# ---------------------------------------------------------------------------

LOCATION_ALIASES = {
    "python": [
        "giant iron python", "articulated iron python", "articulated python",
        "giant python", "iron python", "iron snake", "python", "snake",
        "鐵自在蟒蛇置物", "鐵自在蟒蛇", "大鐵蛇", "大蟒蛇", "蟒蛇", "鐵蛇", "铁蛇", "大蛇", "巨蟒",
        "riesige eisenschlange", "eisenschlange", "riesenschlange", "schlange",
    ],
    "rhinoceros": [
        "iron rhinoceros beetle incense container", "rhinoceros beetle", "iron rhinoceros beetle",
        "iron beetle", "rhino beetle", "rhinoceros", "beetle", "incense container", "incense",
        "鐵自在獨角仙香合", "鐵自在獨角仙", "獨角仙香合", "独角仙香合", "獨角仙", "独角仙",
        "鐵甲蟲", "铁甲虫", "甲蟲", "甲虫", "鐵犀牛", "铁犀牛", "犀牛",
        "nashornkäfer räuchergefäß", "nashornkäfer", "nashornkaefer", "räuchergefäß", "käfer", "kaefer",
    ],
    "lobster": [
        "spiny lobster", "articulated lobster", "iron lobster", "lobster",
        "伊勢海老置物", "伊勢海老", "伊势海老", "大龍蝦", "大龙虾", "龍蝦", "龙虾",
        "japanische languste", "languste", "hummer",
    ],
    "crab": [
        "crab carved from buffalo horn", "buffalo horn crab", "horn crab", "iron crab", "crab",
        "角彫自在蟹", "水牛角蟹", "牛角蟹", "鐵蟹", "铁蟹", "螃蟹",
        "büffelhorn krabbe", "bueffelhorn krabbe", "büffelhornkrabbe", "krabbe",
    ],
    "prawn": [
        "long-armed prawn", "articulated prawn", "iron prawn", "iron shrimp", "prawn", "shrimp",
        "明珍鐵自在長臂蝦置物", "長臂蝦置物", "长臂虾置物", "長臂蝦", "长臂虾", "鐵蝦", "铁虾", "明珍", "蝦", "虾",
        "japanische garnele", "riesengarnele", "garnele", "shrimps",
    ],
}

# Sorted by alias length descending so specific multi-word aliases match before short substrings
_SORTED_ALIASES = sorted(
    [(loc_id, alias.lower()) for loc_id, aliases in LOCATION_ALIASES.items() for alias in aliases],
    key=lambda item: len(item[1]),
    reverse=True,
)

REPEAT_EXACT = {
    "what?", "what", "wie bitte?", "wie bitte", "nochmal", "say that again",
    "repeat please", "repeat", "pardon", "i didn't catch that", "i didnt catch that",
    "什麼？", "什麼", "什么？", "什么", "請再說一次", "请再说一次", "再說一次", "再说一次",
    "重複一次", "重复一次", "再說一遍", "再说一遍", "你說什麼", "你说什么",
}

CANCEL_EXIT_EXACT = {
    "bye", "goodbye", "good bye", "bye bye", "byebye", "exit", "quit", "cancel", "stop",
    "tschüss", "tschuess", "auf wiedersehen", "auf wiedersehn", "ende", "abbrechen", "aufhören",
    "再見", "再见", "掰掰", "拜拜", "結束", "结束", "退出", "算了",
}

CANCEL_EXIT_SUBSTRINGS = [
    "i have to go", "i must go", "we have to go", "we must go", "i need to leave",
    "stop tour", "stop the tour", "end tour", "end the tour", "leave now",
    "ich muss gehen", "ich muss los", "wir müssen los", "wir muessen los",
    "我要走了", "我們該走了", "我们该走了", "離開", "离开", "停止導覽", "停止导览", "不參觀了", "不参观了",
]

REQUEST_TOUR_SUBSTRINGS = [
    # English
    "full tour", "whole tour", "complete tour", "guided tour", "start tour",
    "start the tour", "give me the tour", "give me a tour", "give us a tour",
    "show me everything", "show us everything",
    "show me the exhibition", "show us the exhibition", "show me this exhibition",
    "show me around", "show us around", "take me around", "take us around",
    "show me what's here", "show us what's here", "show what's here",
    "tour the exhibition", "tour the museum",
    # German
    "ganze führung", "ganze fuehrung", "ganze tour", "komplette tour", "geführte tour", "gefuehrte tour",
    "tour starten", "führung starten", "fuehrung starten", "starte die tour",
    "zeig mir alles", "zeig uns alles",
    "zeig mir die ausstellung", "zeig uns die ausstellung",
    "führen sie mich herum", "fuehren sie mich herum", "führe mich herum",
    "ich möchte eine führung", "wir möchten eine führung",
    # Chinese
    "全部導覽", "全部导览", "完整導覽", "完整导览", "開始導覽", "开始导览",
    "參加導覽", "参加导览", "我想參加導覽", "我想参加导览",
    "帶我參觀展覽", "带我参观展览", "帶我們參觀展覽", "带我们参观展览",
    "帶我參觀", "带我参观", "帶我們參觀", "带我们参观",
    "參觀展覽", "参观展览", "參觀整個展覽", "参观整个展览",
    "看全部", "看展覽", "看展览",
]

CONTINUE_TOUR_EXACT = {
    "continue", "next", "skip", "weiter", "weitermachen", "nächste", "naechste",
    "繼續", "继续", "下一站", "下一個", "下一个", "下一件", "下一件作品", "跳過", "跳过", "走吧",
}

CONTINUE_TOUR_SUBSTRINGS = [
    "next stop", "next exhibit", "next artwork", "next station", "next one",
    "move on", "let's continue", "lets continue", "continue tour", "continue the tour",
    "skip this", "go to next", "go to the next", "weitergehen", "nächste station",
    "naechste station", "nächstes kunstwerk", "naechstes kunstwerk",
    "überspringen", "ueberspringen", "往下一站", "走吧", "繼續導覽", "继续导览",
    "it's okay", "its okay", "no it's okay", "no its okay", "it's fine", "its fine",
    "no it's fine", "no its fine", "that's okay", "thats okay", "that's fine", "thats fine",
    "never mind", "nevermind", "no worries", "no more questions", "no problem",
    "kein problem", "macht nichts", "schon gut", "passt schon",
    "沒關係", "没关系", "不用了沒事", "不用了没事", "沒事", "没事",
]

NAV_ACTION_PATTERNS = [
    # English
    "show me", "take me to", "take me", "go to", "bring me to", "bring me",
    "lead me to", "lead me", "walk to", "visit", "head to",
    "can we see", "can we go to", "can we go", "can you show me", "can you take me to",
    "can you take me", "could you show me", "could you take me to",
    "want to see", "want to go", "wants to see", "wants to go",
    "like to see", "like to go", "would like to see", "would like to go",
    "let's go to", "lets go to", "let's see", "lets see",
    "guide me to", "move to", "navigate to",
    # German
    "zeig mir", "zeige mir", "kannst du mir zeigen", "können sie mir zeigen",
    "bring mich zu", "bringen sie mich zu", "bring uns zu", "bringen sie uns zu",
    "gehe zu", "gehen wir zu", "lass uns zu",
    "möchte zu", "will zu", "möchte sehen", "will sehen", "sehen wollen",
    "können wir zu", "können wir sehen", "führen sie mich zu", "führe mich zu",
    # Chinese
    "帶我去", "带我去", "帶我看", "带我看", "想看", "想要看", "帶我們去", "带我们去",
    "帶我們看", "带我们看", "去看", "去看看", "前往", "參觀", "参观",
    "走去", "走到", "可以帶我去", "可以带我去", "可以帶我看", "可以带我看",
    "帶路去", "带路去",
]

QUESTION_INDICATORS = [
    # English
    "what", "who", "whom", "whose", "when", "where", "why", "how",
    "tell me", "tell us", "explain", "describe", "can you tell", "could you tell",
    "is it", "is this", "are these", "are they", "was it", "was this",
    "does it", "did it", "can it", "could it",
    "information about", "info about", "meaning of", "history of",
    # German
    "was", "wer", "wem", "wen", "wessen", "wann", "wo", "woher", "wohin",
    "warum", "wieso", "weshalb", "wie",
    "erzähl", "erzaehl", "erklär", "erklaer", "beschreib",
    "ist es", "ist das", "sind das", "kann es",
    # Chinese
    "什麼", "什么", "誰", "谁", "何時", "何时", "哪裡", "哪里", "哪兒", "哪儿",
    "為何", "为何", "為什麼", "为什么", "怎麼", "怎么", "如何",
    "介紹", "介绍", "告訴我", "告诉我", "說說", "说说", "請問", "请问",
    "是不是", "可不可以介紹", "能不能介紹", "能不能", "可否", "有什麼", "有什么",
    "多說一點", "多说一点",
]

_FILLERS = {
    'the', 'a', 'an', 'please', 'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine',
    'einen', 'einem', 'bitte', '請', '请', '謝謝', '谢谢', 'to', 'zu', 'zum', 'zur'
}


# ---------------------------------------------------------------------------
# Fast-Path Helpers
# ---------------------------------------------------------------------------

def _is_cjk(s: str) -> bool:
    return any('\u4e00' <= char <= '\u9fff' for char in s)


def _contains_alias(text: str, alias: str) -> bool:
    if _is_cjk(alias):
        return alias in text
    return bool(re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE))


def _match_exhibit(text: str) -> Optional[tuple[str, str]]:
    """Match the longest matching exhibit alias from user text."""
    for loc_id, alias in _SORTED_ALIASES:
        if _contains_alias(text, alias):
            return loc_id, alias
    return None


def _is_just_exhibit(text: str, alias: str) -> bool:
    """Check if the utterance consists solely of the exhibit name with optional fillers."""
    remainder = re.sub(re.escape(alias), '', text, flags=re.IGNORECASE)
    remainder = re.sub(r'[,.!?，。！？\s]+', ' ', remainder).strip()
    if not remainder:
        return True
    words = remainder.split()
    return all(w.lower() in _FILLERS for w in words)


def fast_classify_intent(
    user_text: str,
    available_locations: Optional[List[str]] = None,
) -> Optional[IntentResult]:
    """
    Ultra-fast rule-based intent classifier (<0.001s).

    Returns IntentResult if an unambiguous pattern matches; None to fallback to LLM.
    """
    if not user_text:
        return None

    cleaned = user_text.strip().lower()
    cleaned_no_punct = re.sub(r'^[.,!?;:，。！？\s]+|[.,!?;:，。！？\s]+$', '', cleaned)

    # 1. Repeat
    if cleaned in REPEAT_EXACT or cleaned_no_punct in REPEAT_EXACT:
        return IntentResult(intent="repeat", confidence=1.0)

    # 2. Cancel / Exit
    if cleaned in CANCEL_EXIT_EXACT or cleaned_no_punct in CANCEL_EXIT_EXACT:
        return IntentResult(intent="cancel_exit", confidence=1.0)
    if any(phrase in cleaned for phrase in CANCEL_EXIT_SUBSTRINGS):
        return IntentResult(intent="cancel_exit", confidence=1.0)

    # 3. Question check: if question mark or question indicator is present, fall back to LLM
    if "?" in user_text or "？" in user_text:
        return None
    for q in QUESTION_INDICATORS:
        if _is_cjk(q):
            if q in cleaned:
                return None
        else:
            if re.search(r'\b' + re.escape(q) + r'\b', cleaned):
                return None

    # 4. Continue tour (explicit keywords only; bare yes/ok handled by LLM)
    if cleaned in CONTINUE_TOUR_EXACT or cleaned_no_punct in CONTINUE_TOUR_EXACT:
        return IntentResult(intent="continue_tour", confidence=1.0)
    if any(phrase in cleaned for phrase in CONTINUE_TOUR_SUBSTRINGS):
        return IntentResult(intent="continue_tour", confidence=1.0)

    # 5. Full Tour Request
    if any(phrase in cleaned for phrase in REQUEST_TOUR_SUBSTRINGS):
        return IntentResult(intent="request_tour", confidence=1.0)

    # 6. Specific Exhibit Navigation
    matched_exhibit = _match_exhibit(cleaned)
    if matched_exhibit:
        loc_id, alias = matched_exhibit
        has_nav_action = any(action in cleaned for action in NAV_ACTION_PATTERNS)
        is_just_name = _is_just_exhibit(cleaned, alias)

        if has_nav_action or is_just_name:
            target = loc_id
            if available_locations:
                for loc in available_locations:
                    if loc.lower() == loc_id.lower():
                        target = loc
                        break
            return IntentResult(
                intent="goto_specific_place",
                target_location=target,
                confidence=1.0,
            )

    return None


# ---------------------------------------------------------------------------
# Public Router Interface
# ---------------------------------------------------------------------------

def classify_intent(
    user_text: str,
    current_state: str,
    last_robot_utterance: str,
    language: str,
    available_locations: list[str],
    config: AppConfig,
) -> IntentResult:
    """
    Classify a user utterance into an intent via Hybrid Architecture:
    1. Check ultra-fast local rules (<0.001s).
    2. Fallback to OpenAI structured LLM call if not unambiguously matched.

    Args:
        user_text: Raw transcript from listen()
        current_state: Current AppState value name (e.g. "INTENT_ROUTING")
        last_robot_utterance: What the robot last said
        language: Short language code ("en", "de" or "zh")
        available_locations: List of Temi waypoint names
        config: Application configuration

    Returns:
        IntentResult with intent category, optional theme/target, confidence
    """
    started = time.monotonic()

    # 1. Fast-Path Local Matching (<0.001s)
    fast_result = fast_classify_intent(user_text, available_locations)
    if fast_result is not None:
        elapsed_ms = (time.monotonic() - started) * 1000
        telemetry.intent(fast_result, user_text, current_state, elapsed_ms)
        logger.info(
            f"[FAST-PATH] Instant match ({elapsed_ms:.2f}ms): intent={fast_result.intent} "
            f"confidence={fast_result.confidence:.2f}"
            f"{f' target={fast_result.target_location}' if fast_result.target_location else ''}"
        )
        return fast_result

    # 2. OpenAI Fallback
    # Order runtime fields from most stable to most variable. Together with the
    # static system prompt this maximizes the exact prefix eligible for KV reuse.
    context_message = (
        f"Available locations: {', '.join(available_locations)}\n"
        f"Language: {language}\n"
        f"Current state: {current_state}\n"
        f"Last robot utterance: {last_robot_utterance}\n"
        f"User said: {user_text}"
    )

    intent = generate_structured(
        instructions=ROUTER_SYSTEM_PROMPT,
        input_text=context_message,
        config=config,
        temperature=0.0,
        output_type=IntentResult,
        agent="router",
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    telemetry.intent(intent, user_text, current_state, elapsed_ms)

    logger.info(
        f"Router (LLM {elapsed_ms:.0f}ms): intent={intent.intent} confidence={intent.confidence:.2f}"
        f"{f' theme={intent.theme}' if intent.theme else ''}"
        f"{f' target={intent.target_location}' if intent.target_location else ''}"
    )
    return intent
