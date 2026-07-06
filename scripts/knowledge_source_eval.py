#!/usr/bin/env python
"""Knowledge-source coverage eval.

Runs a set of source-neutral queries through the real chat agent and scores two
assertions per turn:

1. multi-source coverage - the turn drew on >=2 distinct knowledge sources
   (built-in RAG index + at least one MCP-backed source).
2. citations resolve - every extracted knowledge citation has a stable id/url.

This turns the eyeballed federation traces (see
docs/how-to/knowledge-source-eval.md, artifacts/knowledge-source-eval/*.trace.json)
into a measured scorecard. The queries deliberately omit source-name hints
("Confluence"/"CQL"/"RAG") so we test routing, not keyword steering.

Modes:
  live (default)   run each query through the agent, then score.
  --from-traces    score already-captured *.trace.json files offline
                   (no live model or MCP creds needed).

Examples:
  # Live (inside the stack; see docs/how-to/knowledge-source-eval.md):
  docker compose exec -T sre-agent uv run python scripts/knowledge_source_eval.py

  # Offline re-score of existing captures:
  uv run python scripts/knowledge_source_eval.py \
    --from-traces "artifacts/knowledge-source-eval/*.trace.json"
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from redis_sre_agent.evaluation.knowledge_source_coverage import (
    Scorecard,
    score_envelopes,
)

logger = logging.getLogger("knowledge_source_eval")

# Source-neutral defaults (no "Confluence"/"CQL"/"RAG"): a mix of internal-only,
# plausibly-cross-source, and Redis-core topics so the scorecard is realistic.
DEFAULT_QUERIES: List[str] = [
    "What is Redis Shaka?",
    "Explain to me the purpose of different buffers in Redis Enterprise including those used with CRDB",
    "Walk me through K8s deployment best practices for Redis Enterprise",
    "What are the steps for upgrading the OS in a Redis Enterprise cluster?",
    "What are common causes for CRDB sync issues?",
    "How to troubleshoot latency in Redis?",
]

DEFAULT_OUTPUT = Path("artifacts/knowledge-source-eval/scorecard.json")


def _load_queries(path: Optional[str]) -> List[str]:
    if not path:
        return list(DEFAULT_QUERIES)
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("queries", [])
    if not isinstance(raw, list) or not all(isinstance(q, str) for q in raw):
        raise ValueError(f"{path} must be a list of query strings or {{queries: [...]}}")
    return [q for q in raw if q.strip()]


async def _run_live(
    queries: Sequence[str],
    *,
    max_iterations: int,
) -> List[Dict[str, Any]]:
    """Run each query through the chat agent and collect its turn (query,
    response, tool_envelopes)."""
    from ulid import ULID

    from redis_sre_agent.agent.chat_agent import get_chat_agent
    from redis_sre_agent.tools.mcp.pool import MCPConnectionPool

    mcp_pool = MCPConnectionPool.get_instance()
    await mcp_pool.start()
    turns: List[Dict[str, Any]] = []
    try:
        agent = get_chat_agent()
        for query in queries:
            logger.info("Running query: %s", query)
            try:
                response = await agent.process_query(
                    query,
                    session_id=f"kse:{ULID()}",
                    user_id=None,
                    max_iterations=max_iterations,
                )
                turns.append(
                    {
                        "query": query,
                        "response": response.response,
                        "tool_envelopes": list(response.tool_envelopes or []),
                    }
                )
            except Exception:  # noqa: BLE001 - one bad turn shouldn't abort the batch
                logger.exception("Query failed, recording empty envelopes: %s", query)
                turns.append({"query": query, "response": None, "tool_envelopes": []})
    finally:
        await mcp_pool.shutdown(force=True)
    return turns


def _run_live_sync(
    queries: Sequence[str],
    *,
    max_iterations: int,
) -> List[Dict[str, Any]]:
    """Drive the async runner with the same MCP-safe loop handling as the CLI."""

    def _suppress_shutdown_errors(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exception = context.get("exception")
        if isinstance(exception, RuntimeError):
            msg = str(exception)
            if "different task" in msg or "cancel scope" in msg:
                logger.debug("Suppressed expected shutdown error: %s", msg)
                return
        if isinstance(exception, asyncio.CancelledError):
            logger.debug("Suppressed CancelledError during shutdown")
            return
        loop.default_exception_handler(context)

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(_suppress_shutdown_errors)
    try:
        return loop.run_until_complete(_run_live(queries, max_iterations=max_iterations))
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def _load_from_traces(patterns: Sequence[str]) -> List[Dict[str, Any]]:
    """Read turns from captured trace JSON files (offline scoring)."""
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"No trace files matched: {list(patterns)}")

    turns: List[Dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        envelopes = data.get("tool_envelopes", []) if isinstance(data, dict) else []
        # Newer trace files carry the original query; older ones (and manual
        # captures) do not, so fall back to the filename stem for a label.
        query = data.get("query") if isinstance(data, dict) else None
        turns.append(
            {
                "query": query or Path(path).stem,
                "response": data.get("response") if isinstance(data, dict) else None,
                "tool_envelopes": list(envelopes),
            }
        )
    return turns


def _slugify(text: str, *, max_len: int = 40) -> str:
    """Filesystem-safe, length-capped slug for trace filenames."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "query"


def _write_traces(turns: List[Dict[str, Any]], out_dir: Path) -> None:
    """Persist each turn's evidence next to the scorecard and record its filename.

    Writes two files per turn, sharing the stem ``q{n}-{slug}`` (the index
    preserves order and keeps them unique even if two slugs collide after
    truncation):

    - ``<stem>.trace.json`` - machine-readable evidence (query, response,
      tool_envelopes) used by ``--from-traces``.
    - ``<stem>.answer.md`` - the answer text with real newlines, so it wraps and
      renders naturally instead of a single escaped JSON string.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, turn in enumerate(turns, 1):
        stem = f"q{i}-{_slugify(str(turn['query']))}"
        response = turn.get("response")
        answer_name: Optional[str] = None
        if response:
            answer_path = out_dir / f"{stem}.answer.md"
            answer_path.write_text(response)
            answer_name = answer_path.name
        trace_path = out_dir / f"{stem}.trace.json"
        trace_path.write_text(
            json.dumps(
                {
                    "query": turn["query"],
                    "response": response,
                    "answer_file": answer_name,
                    "tool_envelopes": turn.get("tool_envelopes", []),
                },
                indent=2,
            )
        )
        turn["trace_file"] = trace_path.name


def _print_report(scorecard: Scorecard) -> None:
    print("\nKnowledge-source coverage")
    print("=" * 72)
    for result in scorecard.results:
        status = "PASS" if result.passed else "FAIL"
        sources = ", ".join(result.sources) if result.sources else "(none)"
        cites = f"{result.citation_count - result.unresolved_citation_count}/{result.citation_count}"
        print(f"[{status}] {result.query}")
        print(f"        sources={len(result.sources)} [{sources}]  citations_resolved={cites}")
        if result.trace_file:
            print(f"        trace={result.trace_file}")
    print("-" * 72)
    print(
        f"both-source: {scorecard.both_source_hits}/{scorecard.total} "
        f"(rate {scorecard.both_source_rate:.0%}, threshold {scorecard.min_both_source_rate:.0%})"
    )
    print(f"citations resolve: {scorecard.citation_hits}/{scorecard.total}")
    print(f"RESULT: {'PASS' if scorecard.passed else 'FAIL'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", help="YAML file: a list of strings or {queries: [...]}")
    parser.add_argument(
        "--from-traces",
        nargs="+",
        metavar="GLOB",
        help="Score existing trace JSON files instead of running the agent (offline)",
    )
    parser.add_argument("--min-both-source-rate", type=float, default=0.8)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the JSON scorecard (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--json", action="store_true", help="Print the JSON scorecard to stdout")
    parser.add_argument(
        "--no-save-traces",
        dest="save_traces",
        action="store_false",
        help="Live mode only: skip writing each turn's trace JSON next to the scorecard",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=None,
        help="Directory for live-run trace files (default: alongside --output)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.from_traces:
        turns = _load_from_traces(args.from_traces)
    else:
        turns = _run_live_sync(_load_queries(args.queries), max_iterations=args.max_iterations)
        if args.save_traces:
            traces_dir = args.traces_dir or (args.output.parent if args.output else Path("."))
            _write_traces(turns, traces_dir)

    results = []
    for turn in turns:
        score = score_envelopes(
            turn["query"], turn["tool_envelopes"], min_sources=args.min_sources
        )
        score.trace_file = turn.get("trace_file")
        results.append(score)
    scorecard = Scorecard(results=results, min_both_source_rate=args.min_both_source_rate)

    _print_report(scorecard)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(scorecard.to_dict(), indent=2))
        print(f"\nWrote scorecard: {args.output}")

    if args.json:
        print(json.dumps(scorecard.to_dict(), indent=2))

    return 0 if scorecard.passed else 1


if __name__ == "__main__":
    sys.exit(main())
