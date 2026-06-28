"""Retriever Agent – fetches relevant context using the vector store."""
import logging
from typing import Optional
from app.vector_db.retriever import retrieve

logger = logging.getLogger(__name__)


def agent_retrieve(
    sub_queries: list[str],
    top_k: int = 5,
    document_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Execute retrieval for each sub-query and merge results (deduped).
    Returns list of unique retrieval result dicts.
    """
    seen_ids = set()
    merged: list[dict] = []

    for sq in sub_queries:
        try:
            results = retrieve(sq, top_k=top_k, filters={"document_ids": document_ids} if document_ids else None)
            for r in results:
                uid = r.get("id") or r.get("chunk_id") or r.get("document", "")[:50]
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    merged.append(r)
        except Exception as e:
            logger.error(f"Retrieval failed for sub-query '{sq}': {e}")

    logger.info(f"Retriever agent returned {len(merged)} unique results")
    return merged
