"""Scoring for multi knowledge-source coverage over agent tool envelopes.

Turns eyeballed federation traces into machine-checkable claims:

- coverage: did the turn draw on at least ``min_sources`` distinct knowledge
  sources (e.g. the built-in RAG index *and* an MCP-backed source)?
- citations: did every extracted knowledge citation resolve to a stable
  reference (an ``id`` or ``url``)?

The functions are pure and operate on ``tool_envelope`` dicts - the same shape
stored in decision traces and returned by ``AgentResponse.tool_envelopes`` - so
they score live runs and captured ``*.trace.json`` files identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from redis_sre_agent.agent.helpers import extract_citations
from redis_sre_agent.tools.models import ToolCapability

# Anchors on the ``{provider_name}_{instance_hash}`` prefix (6-hex instance
# hash) that precedes every operation suffix. Mirrors ``_SOURCE_PREFIX_RE`` in
# redis_sre_agent/agent/knowledge_context.py so grouping here matches the prompt
# guidance the agent was given.
_SOURCE_PREFIX_RE = re.compile(r"^(.+_[0-9a-f]{6})_")

_KNOWLEDGE_CAPABILITY = ToolCapability.KNOWLEDGE.value

# Fields on a citation that count as a resolvable reference, in priority order.
_RESOLVABLE_CITATION_FIELDS = ("url", "id")


def knowledge_source_prefix(name: str) -> str:
    """Return the provider-prefix group for a tool name (or the name itself)."""
    match = _SOURCE_PREFIX_RE.match(name or "")
    return match.group(1) if match else (name or "")


def _is_knowledge_envelope(envelope: Dict[str, Any]) -> bool:
    """Match knowledge tools by tool_key substring OR explicit KNOWLEDGE capability.

    The capability branch covers MCP-backed sources (e.g. Confluence) whose
    generated ``tool_key`` does not contain the word "knowledge" but which are
    declared as knowledge sources in config - the same gate ``extract_citations``
    uses.
    """
    tool_key = str(envelope.get("tool_key", ""))
    capability = str(envelope.get("capability", "")).strip().lower()
    return "knowledge" in tool_key.lower() or capability == _KNOWLEDGE_CAPABILITY


def _is_successful(envelope: Dict[str, Any]) -> bool:
    status = str(envelope.get("status", "success")).strip().lower()
    return status in ("", "success")


def distinct_knowledge_sources(envelopes: Sequence[Dict[str, Any]]) -> List[str]:
    """Distinct provider prefixes among successful knowledge tool envelopes.

    Order of first appearance is preserved for stable, readable reporting.
    """
    sources: List[str] = []
    for envelope in envelopes or []:
        if not isinstance(envelope, dict):
            continue
        if not _is_knowledge_envelope(envelope) or not _is_successful(envelope):
            continue
        name = str(envelope.get("tool_key") or envelope.get("name") or "")
        prefix = knowledge_source_prefix(name)
        if prefix and prefix not in sources:
            sources.append(prefix)
    return sources


def _citation_resolves(citation: Dict[str, Any]) -> bool:
    for key in _RESOLVABLE_CITATION_FIELDS:
        value = citation.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


@dataclass
class QueryScore:
    """Per-query outcome for the two coverage assertions."""

    query: str
    sources: List[str]
    citation_count: int
    unresolved_citation_count: int
    multi_source_pass: bool
    citations_pass: bool
    # Optional pointer to the captured trace behind this row (set by the runner
    # when live traces are saved), so the scorecard cross-references its evidence.
    trace_file: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.multi_source_pass and self.citations_pass

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": self.query,
            "sources": list(self.sources),
            "source_count": len(self.sources),
            "citation_count": self.citation_count,
            "unresolved_citation_count": self.unresolved_citation_count,
            "multi_source_pass": self.multi_source_pass,
            "citations_pass": self.citations_pass,
            "passed": self.passed,
        }
        if self.trace_file is not None:
            payload["trace_file"] = self.trace_file
        return payload


def score_envelopes(
    query: str,
    envelopes: Sequence[Dict[str, Any]],
    *,
    min_sources: int = 2,
) -> QueryScore:
    """Score one turn's envelopes for multi-source coverage and citation resolution."""
    sources = distinct_knowledge_sources(envelopes)
    citations = extract_citations(list(envelopes or []))
    unresolved = [c for c in citations if not _citation_resolves(c)]
    return QueryScore(
        query=query,
        sources=sources,
        citation_count=len(citations),
        unresolved_citation_count=len(unresolved),
        multi_source_pass=len(sources) >= min_sources,
        citations_pass=bool(citations) and not unresolved,
    )


@dataclass
class Scorecard:
    """Aggregate over a batch of scored queries with a pass threshold."""

    results: List[QueryScore] = field(default_factory=list)
    min_both_source_rate: float = 0.8

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def both_source_hits(self) -> int:
        return sum(1 for r in self.results if r.multi_source_pass)

    @property
    def citation_hits(self) -> int:
        return sum(1 for r in self.results if r.citations_pass)

    @property
    def both_source_rate(self) -> float:
        return self.both_source_hits / self.total if self.total else 0.0

    @property
    def citation_rate(self) -> float:
        return self.citation_hits / self.total if self.total else 0.0

    @property
    def passed(self) -> bool:
        """Threshold gate. Multi-source routing is probabilistic (soft nudge), so
        we require a *rate*, not a perfect run. Citations, by contrast, must
        resolve whenever any source is cited."""
        return (
            self.total > 0
            and self.both_source_rate >= self.min_both_source_rate
            and self.citation_hits == self.total
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "both_source_hits": self.both_source_hits,
            "both_source_rate": round(self.both_source_rate, 4),
            "citation_hits": self.citation_hits,
            "citation_rate": round(self.citation_rate, 4),
            "min_both_source_rate": self.min_both_source_rate,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }
