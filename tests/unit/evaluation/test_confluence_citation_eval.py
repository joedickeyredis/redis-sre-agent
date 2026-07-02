"""Deterministic eval-path coverage for Confluence MCP citations.

Proves, without a live model or API key, that a knowledge-capability MCP tool
result surfaces as a machine-checkable eval source end to end:

    knowledge-capability MCP envelope
        -> extract_citations()            (matches via capability, not tool_key)
        -> _normalize_retrieved_sources() (Confluence page ARI becomes source_id)
        -> score_structured_assertions()  (required_sources assertion passes)

This is the deterministic counterpart to a live ``agent_only`` Confluence
scenario, and guards the capability-based citation gate against regressions.
"""

from __future__ import annotations

from typing import Any

from redis_sre_agent.agent.helpers import extract_citations
from redis_sre_agent.evaluation.assertions import score_structured_assertions
from redis_sre_agent.evaluation.live_suite import _normalize_retrieved_sources
from redis_sre_agent.evaluation.scenarios import EvalScenario

# Real page identity captured from a live Atlassian search.
_CONFLUENCE_PAGE_ARI = "ari:cloud:confluence:06f73ca7-8f2c-4392-b40a-08288e9d0ba3:page/6416793646"
_CONFLUENCE_PAGE_URL = (
    "https://redislabs.atlassian.net/wiki/spaces/DX/pages/6416793646/Redis+Shaka+Your+One-Stop+Shop"
)


def _confluence_search_envelope() -> dict[str, Any]:
    """A knowledge-capability MCP search envelope shaped like the real Atlassian result.

    Note the mangled ``tool_key`` does not contain the word "knowledge"; only the
    ``capability`` field marks it as a knowledge source, so this also exercises the
    capability branch of the citation gate.
    """
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
                    "text": "What is Shaka. Shaka is a Redis-compatible platform...",
                    "url": _CONFLUENCE_PAGE_URL,
                    "type": "page",
                }
            ]
        },
    }


def _scenario(required_sources: list[str]) -> EvalScenario:
    return EvalScenario.model_validate(
        {
            "id": "knowledge/confluence-citation-source",
            "name": "Confluence citation surfaces as a checkable source",
            "provenance": {
                "source_kind": "synthetic",
                "source_pack": "fixture-pack",
                "source_pack_version": "2026-04-14",
                "golden": {"expectation_basis": "human_authored"},
            },
            "execution": {
                "lane": "agent_only",
                "agent": "chat",
                "query": "Search Confluence and tell me about Redis Shaka?",
            },
            "expectations": {"required_sources": required_sources},
        }
    )


def _score_required_sources(required_sources: list[str]):
    citations = extract_citations([_confluence_search_envelope()])
    retrieved_sources = _normalize_retrieved_sources(citations)
    results = score_structured_assertions(
        _scenario(required_sources),
        retrieved_sources=retrieved_sources,
    )
    return citations, retrieved_sources, results


def test_confluence_citation_extracted_and_normalized():
    citations, retrieved_sources, _ = _score_required_sources([_CONFLUENCE_PAGE_ARI])

    # Capability (not tool_key) is what surfaces this citation, and all fields survive.
    assert len(citations) == 1
    assert citations[0]["url"] == _CONFLUENCE_PAGE_URL
    # With no document_hash/ticket_id, the page ARI becomes the normalized source_id.
    assert len(retrieved_sources) == 1
    assert retrieved_sources[0]["source_id"] == _CONFLUENCE_PAGE_ARI


def test_required_sources_matches_confluence_page_ari():
    _, _, results = _score_required_sources([_CONFLUENCE_PAGE_ARI])

    assert results.required_sources[0].status.value == "passed"


def test_required_sources_matches_confluence_page_id_substring():
    # The bare numeric page id matches by substring - immutable and rename-proof
    # (unlike the URL, whose path embeds the mutable page-title slug).
    _, _, results = _score_required_sources(["6416793646"])

    assert results.required_sources[0].status.value == "passed"


def test_required_sources_missing_page_fails():
    _, _, results = _score_required_sources(
        ["ari:cloud:confluence:06f73ca7-8f2c-4392-b40a-08288e9d0ba3:page/0000000000"]
    )

    assert results.required_sources[0].status.value == "failed"
