"""Agent Orchestrator – wires Planner → Retriever → Reasoner → Validator."""
import logging
from typing import Optional
from app.agents.planner import plan_query
from app.agents.retriever import agent_retrieve
from app.agents.reasoner import reason
from app.agents.validator import validate

logger = logging.getLogger(__name__)


def run_agent_pipeline(
    query: str,
    document_ids: Optional[list[str]] = None,
    top_k: int = 5,
) -> dict:
    """
    Full agentic pipeline:
      1. Planner  – decompose the query
      2. Retriever – fetch relevant context
      3. Reasoner  – generate a response
      4. Validator – verify grounding

    Returns a dict with the final answer and full agent traces.
    """
    trace = {}

    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    logger.info("Agent pipeline: PLANNING")
    plan = plan_query(query)
    trace["plan"] = plan

    # ── Step 2: Retrieve ──────────────────────────────────────────────────────
    logger.info("Agent pipeline: RETRIEVING")
    retrieval_results = agent_retrieve(
        sub_queries=plan.get("sub_queries", [query]),
        top_k=top_k,
        document_ids=document_ids,
    )
    trace["retrieval_count"] = len(retrieval_results)

    if not retrieval_results:
        return {
            "query": query,
            "response": "No relevant documents found for your query.",
            "sources": [],
            "confidence": 0.0,
            "reasoning_steps": [],
            "validation": {"is_grounded": False, "issues": ["No context retrieved"]},
            "trace": trace,
        }

    # ── Step 3: Reason ────────────────────────────────────────────────────────
    logger.info("Agent pipeline: REASONING")
    reasoning_result = reason(query, retrieval_results)
    trace["reasoning_steps"] = reasoning_result["reasoning_steps"]

    # ── Step 4: Validate ──────────────────────────────────────────────────────
    logger.info("Agent pipeline: VALIDATING")
    validation = validate(query, reasoning_result["response"], retrieval_results)
    trace["validation"] = {
        "is_grounded": validation["is_grounded"],
        "issues": validation["issues"],
    }

    logger.info("Agent pipeline: COMPLETE")
    return {
        "query": query,
        "response": validation["final_answer"],
        "sources": reasoning_result["sources"],
        "confidence": validation["confidence"],
        "reasoning_steps": reasoning_result["reasoning_steps"],
        "validation": validation,
        "trace": trace,
    }
