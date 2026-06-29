# OpenWeave — Production Deployment

OpenWeave watches **real AI agents and multi-agent systems** in production and
flags failure modes that generic LLM tracing misses: redundant/looping tool
calls, reasoning cycles, prompt-injection, anomalous latency/cost, and
multi-agent coordination breakdowns — then classifies each into an incident and
(optionally) scores it with a live LLM judge.

It runs **on top of any Langfuse-compatible trace store** (Langfuse Cloud or
self-hosted). Your agents keep emitting OpenTelemetry/Langfuse traces; OpenWeave
reads them and adds the analysis layer.

```
  ┌────────────┐   traces    ┌────────────┐   poll/read   ┌─────────────────┐
  │ your agents │ ─────────▶ │  Langfuse   │ ◀──────────── │ OpenWeave        │
  │ (1..N)      │            │  (store)    │               │  observer +      │
  └────────────┘             └────────────┘  ──analysis──▶ │  sidecar         │
                                                           └─────────────────┘
```

Two runtime roles, one image:

| Role | Command | Purpose |
|------|---------|---------|
| **sidecar** | `uvicorn openweave_core.sidecar.app:app` | On-demand `POST /analyze` for one trace (UI / ad-hoc). |
| **observer** | `python -m openweave_core.observer` | **Continuously** discovers new agent traces and analyzes them automatically. |

---

## 1. Prerequisites

- A Langfuse project your agents write to (Cloud or self-hosted). You need its
  **public + secret API keys**.
- Optional: `ANTHROPIC_API_KEY` (deep-eval judge) and `NVIDIA_API_KEY`
  (semantic cycle confirmation). Both degrade gracefully if absent — analysis
  still runs, just without the LLM judge / semantic confirmation.

## 2. Configuration (all via environment variables)

| Variable | Required | Default | Meaning |
|----------|----------|---------|---------|
| `LANGFUSE_BASE_URL` / `LANGFUSE_HOST` | – | `https://cloud.langfuse.com` | Langfuse API base. |
| `LANGFUSE_PUBLIC_KEY` | ✅ | – | Scopes the observer to one project. |
| `LANGFUSE_SECRET_KEY` | ✅ | – | |
| `ANTHROPIC_API_KEY` | – | – | Enables the deep-eval judge. |
| `NVIDIA_API_KEY` | – | – | Enables semantic cycle confirmation. |
| `OPENWEAVE_ENV_FILE` | – | – | Explicit path to a `.env` to load (else `./.env`). Real env vars always win. |
| `OPENWEAVE_OBSERVE_INTERVAL` | – | `15` | Seconds between polls. |
| `OPENWEAVE_OBSERVE_CONCURRENCY` | – | `4` | Traces analyzed in parallel. |
| `OPENWEAVE_OBSERVE_RUN_EVAL` | – | `false` | Run the LLM judge on flagged traces (costs money). |
| `OPENWEAVE_OBSERVE_SETTLE` | – | `30` | Wait this long after a trace's timestamp before analyzing, so all spans have flushed. |
| `OPENWEAVE_OBSERVE_LOOKBACK` | – | `3600` | Cold-start: how far back (s) to begin watching. |
| `OPENWEAVE_OBSERVE_ENVIRONMENT` | – | – | Only watch this Langfuse environment. |
| `OPENWEAVE_OBSERVE_TAGS` | – | – | Comma-separated; only watch traces carrying these tags. |
| `OPENWEAVE_OBSERVE_STATE` | – | `.openweave_observer_state.json` | Durable cursor file. |
| `OPENWEAVE_OBSERVE_OUTPUT_DIR` | – | `openweave_reports` | Where per-trace JSON reports land. |
| `OPENWEAVE_EMBED_TIMEOUT` / `OPENWEAVE_DETECTOR_TIMEOUT` | – | `10` / `20` | Resilience timeouts (see `openweave_core.resilience`). |

Secrets are read from process env in production. Files (`OPENWEAVE_ENV_FILE`,
`./.env`) are a dev convenience and never override an already-set variable.

## 3. Run with Docker Compose (recommended)

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com
export ANTHROPIC_API_KEY=sk-ant-...   # optional

docker compose -f docker-compose.openweave.yml up -d --build
docker compose -f docker-compose.openweave.yml logs -f observer
```

The observer prints one line per analyzed trace and writes a full
`AnalysisPayload` JSON to the `openweave-data` volume (`/data/reports`). The
cursor in `/data/state` makes restarts idempotent — no re-analysis of old
traces.

## 4. Run without Docker

```bash
pip install -r requirements.txt          # or: pip install -e .
# Observe continuously:
LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... python -m openweave_core.observer
# Or serve the on-demand API:
uvicorn openweave_core.sidecar.app:app --host 0.0.0.0 --port 8000
# Analyze a single trace from a script:
python -c "from openweave_core.sidecar.assemble import assemble_analysis_sync as a; import json; print(json.dumps(a('<trace_id>', run_eval=False))[:500])"
```

## 5. Reading the output

Each report is the same `AnalysisPayload` the UI consumes:

- `overallSeverity` — `INFO` / `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`
- `flagged`, `flagCount`, `incidentCount`
- `incidents[]` — classified failures with member flags + (optional) eval scores
- `graph` + `propagationPaths` — abstracted agent/tool call graph and how a
  failure propagated through a multi-agent run
- `detectorResults[]` — per-detector raw output (sentinel, cycle, baseline)

To surface results in the OpenWeave web UI, set `OPENWEAVE_OBSERVE_TRPC_URL` +
`OPENWEAVE_OBSERVE_TRPC_PROJECT_ID` (+ `OPENWEAVE_OBSERVE_TRPC_AUTH`) so the
observer also POSTs each payload to the Node writer.

## 6. Production hardening (already built in)

- **Bounded external calls.** Embedding/judge clients carry timeout + retry; a
  process-wide **circuit breaker** fast-fails a degraded endpoint (`openweave_core.resilience`).
- **Per-detector timeout.** A wedged detector is cut off; the others still report.
- **Async isolation.** One slow trace occupies one worker; the rest keep
  draining (`anomaly_pipeline.processing_queue`, and the observer's bounded
  concurrency).
- **Idempotent restarts.** Atomic cursor file; recently-seen ids de-duped.
- **Graceful shutdown.** SIGINT/SIGTERM drain in-flight analyses.
- **Non-root container**, deferred heavy imports (fast startup, `/health`
  liveness independent of the pipeline).

## 7. Scaling

- Increase `OPENWEAVE_OBSERVE_CONCURRENCY` for higher trace throughput (CPU-bound
  detectors are vectorized; see the scaling notes).
- Run multiple observers against **disjoint** tag/environment filters to shard a
  busy project; each keeps its own cursor.
- Keep `OPENWEAVE_OBSERVE_RUN_EVAL=false` for always-on watching; trigger the
  LLM judge selectively (e.g. a second observer scoped to high-severity tags).
