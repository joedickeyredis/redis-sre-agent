"""Deterministic coverage for the knowledge-source scoring functions.

Proves, without a live model or MCP creds, that the two coverage assertions
(multi-source + citations-resolve) behave correctly on the same envelope shape
the agent produces. This is the deterministic guard for the live
``scripts/knowledge_source_eval.py`` scorecard.
"""

from __future__ import annotations

from typing import Any, Dict

from redis_sre_agent.evaluation.knowledge_source_coverage import (
    Scorecard,
    distinct_knowledge_sources,
    knowledge_source_prefix,
    score_envelopes,
)

# Real page identity captured from a live Atlassian search (mirrors
# test_confluence_citation_eval.py).
_CONFLUENCE_PAGE_ARI = "ari:cloud:confluence:06f73ca7-8f2c-4392-b40a-08288e9d0ba3:page/6416793646"
_CONFLUENCE_PAGE_URL = (
    "https://redislabs.atlassian.net/wiki/spaces/DX/pages/6416793646/Redis+Shaka"
)


def _builtin_kb_envelope() -> Dict[str, Any]:
    """Built-in RAG index search envelope (tool_key contains 'knowledge')."""
    return {
        "tool_key": "knowledge_ffff6f_search",
        "name": "knowledge_ffff6f_search",
        "capability": "knowledge",
        "status": "success",
        "data": {
            "results": [
                {
                    "id": "sre_knowledge:8d4d7e6bac618a84:chunk:9",
                    "title": "Redis client handling",
                    "source": "https://github.com/redis/docs/blob/main/content/develop/reference/clients.md",
                }
            ]
        },
    }


def _confluence_envelope() -> Dict[str, Any]:
    """MCP-backed knowledge envelope whose tool_key lacks the word 'knowledge'."""
    return {
        "tool_key": "mcp_atlassian_rovo_mcp_ffff78_searchConfluenceUsingCql",
        "name": "searchConfluenceUsingCql",
        "capability": "knowledge",
        "status": "success",
        "data": {
            "results": [
                {
                    "id": _CONFLUENCE_PAGE_ARI,
                    "title": "Redis Shaka: Your One-Stop Shop",
                    "url": _CONFLUENCE_PAGE_URL,
                    "type": "page",
                }
            ]
        },
    }


def test_prefix_groups_by_provider_instance():
    assert knowledge_source_prefix("knowledge_ffff6f_search") == "knowledge_ffff6f"
    assert (
        knowledge_source_prefix("mcp_atlassian_rovo_mcp_ffff78_searchConfluenceUsingCql")
        == "mcp_atlassian_rovo_mcp_ffff78"
    )


def test_two_distinct_sources_detected():
    sources = distinct_knowledge_sources([_builtin_kb_envelope(), _confluence_envelope()])
    assert sources == ["knowledge_ffff6f", "mcp_atlassian_rovo_mcp_ffff78"]


def test_both_assertions_pass_with_two_sources():
    score = score_envelopes("q", [_builtin_kb_envelope(), _confluence_envelope()])
    assert score.multi_source_pass is True
    assert score.citations_pass is True
    assert score.passed is True
    assert score.citation_count == 2
    assert score.unresolved_citation_count == 0


def test_single_source_fails_multi_source():
    score = score_envelopes("q", [_builtin_kb_envelope()])
    assert score.multi_source_pass is False
    # Citations still resolve for the one source that was hit.
    assert score.citations_pass is True
    assert score.passed is False


def test_failed_envelope_does_not_count_as_a_source():
    errored = _confluence_envelope()
    errored["status"] = "error"
    sources = distinct_knowledge_sources([_builtin_kb_envelope(), errored])
    assert sources == ["knowledge_ffff6f"]


def test_unresolved_citation_fails_citations_assertion():
    envelope = _confluence_envelope()
    # Strip both resolvable fields from the single result.
    envelope["data"]["results"][0].pop("id")
    envelope["data"]["results"][0].pop("url")
    score = score_envelopes("q", [_builtin_kb_envelope(), envelope])
    assert score.unresolved_citation_count == 1
    assert score.citations_pass is False
    assert score.passed is False


def test_non_knowledge_envelope_ignored():
    diagnostic = {
        "tool_key": "redis_diagnostics_ffff10_run",
        "name": "run",
        "capability": "diagnostics",
        "status": "success",
        "data": {"results": [{"id": "x"}]},
    }
    assert distinct_knowledge_sources([diagnostic]) == []


def test_trace_file_omitted_when_unset_and_included_when_set():
    score = score_envelopes("q", [_builtin_kb_envelope(), _confluence_envelope()])
    assert "trace_file" not in score.to_dict()

    score.trace_file = "q1-what-is-redis-shaka.trace.json"
    assert score.to_dict()["trace_file"] == "q1-what-is-redis-shaka.trace.json"


def test_scorecard_threshold_and_citation_gate():
    passing = score_envelopes("a", [_builtin_kb_envelope(), _confluence_envelope()])
    single = score_envelopes("b", [_builtin_kb_envelope()])

    # 1/2 both-source = 50%; below the 80% default -> fail.
    card = Scorecard(results=[passing, single], min_both_source_rate=0.8)
    assert card.both_source_hits == 1
    assert card.both_source_rate == 0.5
    assert card.passed is False

    # Lower the threshold to 50%; citations still resolve for both -> pass.
    card_low = Scorecard(results=[passing, single], min_both_source_rate=0.5)
    assert card_low.citation_hits == 2
    assert card_low.passed is True
