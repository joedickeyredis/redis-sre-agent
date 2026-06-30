# RAG-only Baseline Capture

Reproducible "before" snapshot of the agent answering from the RAG knowledge base
only (no Atlassian/Confluence MCP wired). Captured for the Confluence MCP POC so we
can show a before/after once live Confluence retrieval is added.

The generated artifacts land in `baseline/` (gitignored - regenerable, and the
traces will contain internal Confluence content once the Atlassian MCP is wired in).

## Prerequisites

Stack up and RAG populated:

```bash
make quick-demo
docker compose exec -T sre-agent uv run ./scripts/setup_redis_docs_local.sh
docker compose exec -T sre-agent uv run redis-sre-agent pipeline ingest
```

Notes:
- Run this BEFORE wiring the Atlassian MCP server into `config.yaml` so the capture
  is genuinely RAG-only.
- `--agent chat` keeps it on the demo (Chat agent) path.
- The CLI `query` command only prints the answer + a "Decision trace" id; it does not
  emit `citation_groups`. `thread trace <message_id> --json` reconstructs the
  citations from the stored trace (same `extract_citation_groups` the UI uses).

## Capture helper

Each call runs the query (teeing the human-readable answer) and then dumps that
turn's citations + tool envelopes as JSON. The 26-char ULID on the "Decision trace"
line is parsed automatically.

```bash
mkdir -p baseline
capture () {
  local slug="$1"; shift
  local q="$*"
  docker compose exec -T sre-agent uv run redis-sre-agent query "$q" --agent chat \
    | tee "baseline/$slug.answer.txt"
  local msg
  msg=$(grep 'Decision trace:' "baseline/$slug.answer.txt" | grep -oE '[0-9A-Z]{26}' | head -1)
  [ -z "$msg" ] && { echo "!! no message id parsed for $slug"; return 1; }
  docker compose exec -T sre-agent uv run redis-sre-agent thread trace "$msg" --json \
    > "baseline/$slug.trace.json"
  echo "OK $slug: answer + trace ($msg)"
}
```

## Baseline question set

This is the canonical set actually captured for the RAG-only baseline (the
`baseline/` artifacts are gitignored; re-run these to regenerate them):

```bash
capture q1 "How do I collect troubleshooting logs from a Redis Enterprise K8s deployment?"
capture q2 "What are the steps for upgrading the OS in a Redis Enterprise cluster?"
capture q3 "What are common causes for CRDB sync issues?"
capture q4 "How do I troubleshoot replication lag in Redis?"
capture q5 "What are Redis Search indexing best practices?"
```

## Output

- `baseline/<slug>.answer.txt` - the grounded answer (rich-rendered markdown).
- `baseline/<slug>.trace.json` - citation groups + tool envelopes (the provenance "before").

To also capture the thread id (for follow-ups), parse the "Created thread:" line:

```bash
grep 'Created thread:' baseline/q1.answer.txt | grep -oE '[0-9A-Z]{26}'
```
