"""Planner Agent – decomposes user queries into sub-tasks."""
import json
import logging
from app.llm.claude_client import claude_client
from app.llm.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE

logger = logging.getLogger(__name__)


def plan_query(query: str) -> dict:
    """
    Decompose a user query into a retrieval plan.
    Returns: {sub_queries, strategy, reasoning}
    """
    messages = [
        {"role": "user", "content": PLANNER_USER_TEMPLATE.format(query=query)}
    ]

    raw = claude_client.generate_response(
        messages=messages,
        system_prompt=PLANNER_SYSTEM_PROMPT,
    )

    try:
        # Try to parse JSON from the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            plan = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON found")
    except Exception:
        logger.warning("Planner could not parse JSON – using fallback plan")
        plan = {
            "sub_queries": [query],
            "strategy": "direct",
            "reasoning": "Fallback: single direct query used",
        }

    logger.info(f"Query plan created: strategy={plan.get('strategy')}")
    return plan
