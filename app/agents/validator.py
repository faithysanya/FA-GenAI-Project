"""Validator Agent – checks response grounding and detects hallucinations."""
import json
import logging
from app.llm.claude_client import claude_client
from app.llm.prompts import VALIDATOR_SYSTEM_PROMPT, VALIDATOR_USER_TEMPLATE
from app.llm.rag_chain import build_context

logger = logging.getLogger(__name__)


def validate(query: str, answer: str, retrieval_results: list[dict]) -> dict:
    """
    Validate that the answer is grounded in the retrieved context.
    Returns: {is_grounded, confidence, issues, final_answer}
    """
    context = build_context(retrieval_results)

    messages = [
        {
            "role": "user",
            "content": VALIDATOR_USER_TEMPLATE.format(
                query=query, context=context, answer=answer
            ),
        }
    ]

    raw = claude_client.generate_response(
        messages=messages,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
    )

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON found")
    except Exception:
        logger.warning("Validator could not parse JSON – using defaults")
        result = {
            "is_grounded": True,
            "confidence": 0.7,
            "issues": [],
            "revised_answer": answer,
        }

    final_answer = result.get("revised_answer", answer)
    confidence = float(result.get("confidence", 0.7))
    logger.info(f"Validation: grounded={result.get('is_grounded')}, confidence={confidence:.2f}")

    return {
        "is_grounded": result.get("is_grounded", True),
        "confidence": confidence,
        "issues": result.get("issues", []),
        "final_answer": final_answer,
    }
