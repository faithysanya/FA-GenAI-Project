"""Reasoner Agent – generates a response via multi-step reasoning over context."""
import logging
from app.llm.claude_client import claude_client
from app.llm.prompts import REASONER_SYSTEM_PROMPT, REASONER_USER_TEMPLATE
from app.llm.rag_chain import build_context

logger = logging.getLogger(__name__)


def reason(query: str, retrieval_results: list[dict]) -> dict:
    """
    Perform step-by-step reasoning over retrieved context.
    Returns: {response, reasoning_steps, sources}
    """
    context = build_context(retrieval_results)
    sources = list({
        r.get("metadata", {}).get("filename", f"doc_{i}")
        for i, r in enumerate(retrieval_results)
    })

    messages = [
        {
            "role": "user",
            "content": REASONER_USER_TEMPLATE.format(query=query, context=context),
        }
    ]

    raw = claude_client.generate_response(
        messages=messages,
        system_prompt=REASONER_SYSTEM_PROMPT,
    )

    # Split raw output into reasoning steps and final answer
    lines = raw.strip().split("\n")
    reasoning_steps = [line.strip() for line in lines if line.strip() and line.strip().startswith(("Step", "-", "•", "1.", "2.", "3."))]
    final_answer = raw  # Use full response as answer

    logger.info(f"Reasoner produced {len(reasoning_steps)} reasoning steps")
    return {
        "response": final_answer,
        "reasoning_steps": reasoning_steps or ["Direct response generated"],
        "sources": sources,
    }
