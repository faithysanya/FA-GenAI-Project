"""Guardrails: response grounding checks and safety filters."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

UNCERTAINTY_PHRASES = [
    "I don't know",
    "I cannot find",
    "not mentioned",
    "not in the context",
    "no information",
]

UNSAFE_KEYWORDS = [
    "ignore previous instructions",
    "disregard your instructions",
    "you are now",
    "act as if",
]


def check_grounding(response: str, context: str) -> dict:
    """
    Lightweight heuristic grounding check.
    Returns {is_grounded, warning_message}
    """
    if not context.strip():
        return {"is_grounded": False, "warning": "No context was retrieved"}

    # Check if response acknowledges uncertainty appropriately
    response_lower = response.lower()
    context_lower = context.lower()

    # Extract a few key words from context and check overlap
    context_words = set(context_lower.split()) - {"the", "a", "an", "is", "in", "of", "and", "to"}
    response_words = set(response_lower.split())
    overlap = context_words & response_words
    overlap_ratio = len(overlap) / max(len(context_words), 1)

    is_grounded = overlap_ratio > 0.05  # At least 5% word overlap
    warning = None if is_grounded else "Response may not be grounded in source documents"

    return {"is_grounded": is_grounded, "warning": warning}


def check_safety(text: str) -> dict:
    """
    Check text for prompt injection or unsafe content.
    Returns {is_safe, flag}
    """
    text_lower = text.lower()
    for kw in UNSAFE_KEYWORDS:
        if kw in text_lower:
            logger.warning(f"Unsafe input detected: '{kw}'")
            return {"is_safe": False, "flag": f"Blocked pattern: '{kw}'"}
    return {"is_safe": True, "flag": None}
