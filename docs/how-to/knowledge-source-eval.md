# Knowledge Source Eval: Tracing and Coverage

The agent answers from more than one knowledge source: the built-in RAG index
plus any MCP-backed sources wired in (e.g. Atlassian/Confluence). This document
covers two phases for inspecting and measuring that retrieval:

- **Phase 1 - Tracing:** run a canonical question set and capture each turn's
  answer plus its provenance trace (citations + tool envelopes) for eyeballing.
- **Phase 2 - Coverage scorecard:** score those same questions into a machine
  checkable pass/fail (did the turn draw on multiple sources, and do its
  citations resolve?).

Both phases share one canonical question set so the eyeballed traces and the
scored results line up.

Generated artifacts land in `artifacts/knowledge-source-eval/`, which is
gitignored - they are regenerable, and traces contain internal Confluence
content once the Atlassian MCP is wired in.

## Prerequisites

Stack up and the RAG index populated:

```bash
make quick-demo
docker compose exec -T sre-agent uv run ./scripts/setup_redis_docs_local.sh
docker compose exec -T sre-agent uv run redis-sre-agent pipeline ingest
```

For multi-source retrieval, also wire the Atlassian MCP server into `config.yaml`
(see [Confluence as a live knowledge source](confluence-knowledge-source.md)).
With no MCP source configured, the agent answers from the RAG index alone -
useful as a single-source point of comparison, but Phase 2's multi-source
assertion will not pass.

## Canonical question set

These are the source-neutral questions used by both phases (no
"Confluence"/"CQL"/"RAG" hints, so we exercise routing rather than keyword
steering). They are the defaults baked into `scripts/knowledge_source_eval.py`:

1. What is Redis Shaka?
2. Explain to me the purpose of different buffers in Redis Enterprise including those used with CRDB
3. Walk me through K8s deployment best practices for Redis Enterprise
4. What are the steps for upgrading the OS in a Redis Enterprise cluster?
5. What are common causes for CRDB sync issues?
6. How to troubleshoot latency in Redis?

## Phase 1 - Trace capture

Capture each question's answer and its stored decision trace for manual review.

Notes:
- `--agent chat` keeps it on the demo (Chat agent) path.
- The CLI `query` command prints the answer plus a "Decision trace" id; it does
  not emit `citation_groups`. `thread trace <message_id> --json` reconstructs the
  citations from the stored trace (the same `extract_citation_groups` the UI uses).

The helper below runs a query (teeing the human-readable answer) and then dumps
that turn's citations + tool envelopes as JSON. The 26-char ULID on the
"Decision trace" line is parsed automatically.

```bash
outdir=artifacts/knowledge-source-eval
mkdir -p "$outdir"
capture () {
  local slug="$1"; shift
  local q="$*"
  docker compose exec -T sre-agent uv run redis-sre-agent query "$q" --agent chat \
    | tee "$outdir/$slug.answer.txt"
  local msg
  msg=$(grep 'Decision trace:' "$outdir/$slug.answer.txt" | grep -oE '[0-9A-Z]{26}' | head -1)
  [ -z "$msg" ] && { echo "!! no message id parsed for $slug"; return 1; }
  docker compose exec -T sre-agent uv run redis-sre-agent thread trace "$msg" --json \
    > "$outdir/$slug.trace.json"
  echo "OK $slug: answer + trace ($msg)"
}
```

Run the canonical set:

```bash
capture q1 "What is Redis Shaka?"
capture q2 "Explain to me the purpose of different buffers in Redis Enterprise including those used with CRDB"
capture q3 "Walk me through K8s deployment best practices for Redis Enterprise"
capture q4 "What are the steps for upgrading the OS in a Redis Enterprise cluster?"
capture q5 "What are common causes for CRDB sync issues?"
capture q6 "How to troubleshoot latency in Redis?"
```

Output:

- `artifacts/knowledge-source-eval/<slug>.answer.txt` - the grounded answer (rich-rendered markdown).
- `artifacts/knowledge-source-eval/<slug>.trace.json` - citation groups + tool envelopes (the provenance).

To also capture the thread id (for follow-ups), parse the "Created thread:" line:

```bash
grep 'Created thread:' artifacts/knowledge-source-eval/q1.answer.txt | grep -oE '[0-9A-Z]{26}'
```

## Phase 2 - Coverage scorecard

`scripts/knowledge_source_eval.py` turns the eyeballed traces from Phase 1 into a
scored check. For each query it asserts two things, entirely from the tool
envelopes (no answer text parsing):

1. multi-source coverage - the turn drew on >=2 distinct knowledge sources (the
   built-in RAG index plus at least one MCP-backed source).
2. citations resolve - every extracted knowledge citation has a stable `id`/`url`.

Scoring lives in `redis_sre_agent/evaluation/knowledge_source_coverage.py` (pure
functions, unit-tested in `tests/unit/evaluation/test_knowledge_source_coverage.py`).

### Run it

Live (inside the stack, with the Atlassian MCP wired into `config.yaml`), using
the canonical set above as defaults:

```bash
make eval-knowledge-sources
# or directly:
docker compose exec -T sre-agent uv run python scripts/knowledge_source_eval.py
```

Offline - re-score already-captured Phase 1 traces, no model or MCP creds needed:

```bash
uv run python scripts/knowledge_source_eval.py \
  --from-traces "artifacts/knowledge-source-eval/*.trace.json"
```

### Output

Writes a JSON scorecard to `artifacts/knowledge-source-eval/scorecard.json`
by default, overwritten each run. Keep a specific run by pointing `--output` at a
distinct path:

```bash
uv run python scripts/knowledge_source_eval.py \
  --output "artifacts/knowledge-source-eval/kse-$(date +%Y%m%d-%H%M%S).json"
```

A live run also captures the evidence behind each row. Per query it writes two
files (sharing the stem `q{n}-{slug}`) into the same directory:

- `q{n}-{slug}.trace.json` - machine-readable evidence (`query`, `response`,
  `tool_envelopes`), used by `--from-traces`; each scorecard row gets a
  `trace_file` pointer to it.
- `q{n}-{slug}.answer.md` - the answer text with real newlines, so it wraps and
  renders naturally instead of a single escaped JSON string.

This means the measurement and its traces come from the *same* turns in one
command - so you can drill from a scored row into the exact trace, and re-score
later with `--from-traces`. It also means a live run subsumes Phase 1's manual
capture for the canonical set. Pass `--no-save-traces` to skip, or `--traces-dir`
to redirect them.

Exit code is non-zero when the both-source rate is below `--min-both-source-rate`
(default `0.8`) or any cited turn has an unresolved citation - so it can gate CI
if desired. Multi-source routing is probabilistic (a soft prompt nudge), so the
threshold is a rate, not a demand for a perfect run.

### Custom query set

Override the defaults with `--queries FILE`, where `FILE` is YAML - either a bare
list or a mapping under a `queries:` key:

```yaml
queries:
  - "What is Redis Shaka?"
  - "Explain to me the purpose of different buffers in Redis Enterprise including those used with CRDB"
  - "How to troubleshoot latency in Redis?"
```

```bash
uv run python scripts/knowledge_source_eval.py --queries my_queries.yaml
```
