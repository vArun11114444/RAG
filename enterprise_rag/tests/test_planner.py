"""Tests for Phase 4 — Query Planner rule-based classifier."""
import pytest
from app.planner.planner import _rule_classify
from app.models.schemas import QueryType


@pytest.mark.parametrize("query,expected", [
    ("What are the GDPR requirements?", QueryType.COMPLIANCE),
    ("How does inflation affect housing prices?", QueryType.MULTI_HOP),
    ("Who is connected to the acquired subsidiary?", QueryType.KNOWLEDGE_GRAPH),
    ("Provide a comprehensive overview of quantum computing research", QueryType.RESEARCH),
    ("What is AI?", QueryType.SIMPLE),
])
def test_rule_classify(query, expected):
    result = _rule_classify(query)
    assert result == expected
