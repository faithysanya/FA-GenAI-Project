"""RAG (Retrieval-Augmented Generation) pipeline."""
import logging
from typing import Optional
from app.llm.claude_client import claude_client
from app.llm.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

logger = logging.getLogger(__name__)

# Approx max context chars before truncation (~100k chars ≈ ~25k tokens)
MAX_CONTEXT_CHARS = 80_000


def build_context(retrieval_results: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    parts = []
    total = 0
    for i, r in enumerate(retrieval_results):
        doc = r.get("document", r.get("text", ""))
        meta = r.get("metadata", {})
        source = meta.get("filename", meta.get("source", f"Document {i+1}"))
        snippet = f"[Source: {source}]\n{doc}"
        if total + len(snippet) > MAX_CONTEXT_CHARS:
            logger.warning("Context truncated to fit token limit")
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


def rag_generate(query: str, retrieval_results: list[dict]) -> dict:
    """
    Core RAG call: combine retrieved context with the LLM to produce an answer.
    Returns dict with 'response', 'context_used', 'sources'.
    """
    context = build_context(retrieval_results)
    sources = [
        r.get("metadata", {}).get("filename", f"doc_{i}")
        for i, r in enumerate(retrieval_results)
    ]

    messages = [
        {
            "role": "user",
            "content": RAG_USER_TEMPLATE.format(context=context, query=query),
        }
    ]

    response = claude_client.generate_response(
        messages=messages,
        system_prompt=RAG_SYSTEM_PROMPT,
    )

    logger.info(f"RAG response generated (mock={claude_client.is_mock})")
    return {
        "response": response,
        "context_used": context[:500] + "..." if len(context) > 500 else context,
        "sources": list(dict.fromkeys(sources)),  # deduplicated
    }
