"""
Metadata filter engine — supports exact match, range queries, and list membership.
Plugs into both BM25 and vector retrieval paths.
"""
from __future__ import annotations

from typing import Any

from app.models.schemas import RetrievedChunk
from app.observability import get_logger

log = get_logger(__name__)


class MetadataFilter:
    """
    Applies structured filter rules to a list of chunks.

    Filter spec format:
        {
            "author": "John Smith",              # exact match
            "year": {"gte": 2020, "lte": 2024}, # range
            "tags": {"in": ["AI", "ML"]},        # list membership
            "category": {"nin": ["draft"]},      # not-in
        }
    """

    @staticmethod
    def apply(
        chunks: list[RetrievedChunk],
        filters: dict[str, Any],
    ) -> list[RetrievedChunk]:
        if not filters:
            return chunks

        result: list[RetrievedChunk] = []
        for chunk in chunks:
            if MetadataFilter._matches(chunk.metadata, filters):
                result.append(chunk)

        log.debug(
            "Metadata filter applied",
            extra={"before": len(chunks), "after": len(result), "filters": filters},
        )
        return result

    @staticmethod
    def _matches(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, rule in filters.items():
            value = meta.get(key)
            if isinstance(rule, dict):
                if "gte" in rule and not (value is not None and value >= rule["gte"]):
                    return False
                if "lte" in rule and not (value is not None and value <= rule["lte"]):
                    return False
                if "gt" in rule and not (value is not None and value > rule["gt"]):
                    return False
                if "lt" in rule and not (value is not None and value < rule["lt"]):
                    return False
                if "in" in rule and value not in rule["in"]:
                    return False
                if "nin" in rule and value in rule["nin"]:
                    return False
                if "eq" in rule and value != rule["eq"]:
                    return False
                if "neq" in rule and value == rule["neq"]:
                    return False
            else:
                # Exact match
                if value != rule:
                    return False
        return True


def build_chroma_filter(filters: dict[str, Any]) -> dict[str, Any] | None:
    """
    Translate the unified filter spec into ChromaDB's `where` clause format.
    Only handles simple equality and $in for ChromaDB compatibility.
    """
    if not filters:
        return None

    where_clauses: list[dict[str, Any]] = []
    for key, rule in filters.items():
        if isinstance(rule, dict):
            if "in" in rule:
                where_clauses.append({key: {"$in": rule["in"]}})
            elif "eq" in rule:
                where_clauses.append({key: {"$eq": rule["eq"]}})
            elif "gte" in rule:
                where_clauses.append({key: {"$gte": rule["gte"]}})
            elif "lte" in rule:
                where_clauses.append({key: {"$lte": rule["lte"]}})
        else:
            where_clauses.append({key: {"$eq": rule}})

    if len(where_clauses) == 1:
        return where_clauses[0]
    if len(where_clauses) > 1:
        return {"$and": where_clauses}
    return None


# Alias — build_qdrant_filter is identical to build_chroma_filter
# because QdrantVectorStore accepts the same ChromaDB-style where dict
# and translates it internally.
build_qdrant_filter = build_chroma_filter
