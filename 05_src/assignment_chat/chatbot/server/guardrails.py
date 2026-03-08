"""
Guardrails: block prompt extraction/modification and restricted topics.
Restricted: cats or dogs, horoscopes/zodiac, Taylor Swift.
"""

import re

# Phrases that suggest the user is trying to see or change the system prompt
PROMPT_LEAK_PATTERNS = [
    r"system\s*prompt",
    r"your\s*instructions",
    r"your\s*prompt",
    r"show\s*(me\s*)?(your|the)\s*(full\s*)?(system\s*)?(prompt|instructions)",
    r"what\s*(are|is)\s*your\s*(instructions|prompt|system\s*prompt)",
    r"ignore\s*(all\s*)?(previous\s*)?(instructions|prompt)",
    r"disregard\s*(all\s*)?(previous\s*)?(instructions|prompt)",
    r"forget\s*(all\s*)?(previous\s*)?(instructions|prompt)",
    r"new\s*instructions",
    r"from\s*now\s*on\s*you\s*(must|will|should)",
    r"reveal\s*(your|the)\s*(system\s*)?prompt",
    r"repeat\s*(your|the)\s*(instructions|prompt)",
    r"print\s*(your|the)\s*(system\s*)?prompt",
    r"output\s*(your|the)\s*(system\s*)?prompt",
]

# Restricted topics (case-insensitive)
RESTRICTED_TOPICS = [
    "cat",
    "cats",
    "dog",
    "dogs",
    "horoscope",
    "horoscopes",
    "zodiac",
    "taylor swift",
]


def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def is_prompt_leak_attempt(message: str) -> bool:
    """True if the message appears to ask for or modify the system prompt."""
    if not message or not message.strip():
        return False
    normalized = _normalize(message)
    for pattern in PROMPT_LEAK_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True
    return False


def is_restricted_topic(message: str) -> bool:
    """True if the message is about a restricted topic (cats/dogs, horoscopes, Taylor Swift)."""
    if not message or not message.strip():
        return False
    normalized = _normalize(message)
    for topic in RESTRICTED_TOPICS:
        if topic in normalized:
            return True
    return False


def check_guardrails(message: str) -> tuple[bool, str | None]:
    """
    Returns (blocked, reason).
    If blocked is True, reason is a short user-facing message; otherwise reason is None.
    """
    if is_prompt_leak_attempt(message):
        return True, "I can't share or change my internal instructions. How can I help you with makeup or general questions?"
    if is_restricted_topic(message):
        return True, "I'm not allowed to discuss that topic. I'm here to help with makeup, product search, and other permitted questions."
    return False, None
