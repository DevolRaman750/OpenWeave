"""
app.py — FastAPI compute service for the OpenWeave UI.

Endpoints
---------
GET  /health           -> {"status": "ok"}  (liveness; no pipeline import)
POST /analyze          -> AnalysisPayload    (full pipeline for one trace)

The heavy pipeline imports are deferred to request time so the process starts
fast and a missing optional dependency only fails the request, not startup.
Run with:  uvicorn openweave_core.sidecar.app:app --port 8000
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Importing the parser bootstraps local env (LANGFUSE_* / ANTHROPIC_API_KEY)
# from openweave-research/agents/.env — see trace_fetcher._bootstrap_local_environment.
from openweave_core.parser import trace_fetcher  # noqa: F401

log = logging.getLogger("openweave.sidecar")

app = FastAPI(title="OpenWeave Sidecar", version="1.0.0")


class AnalyzeRequest(BaseModel):
    trace_id: str = Field(..., min_length=1)
    project_id: Optional[str] = None
    run_eval: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    # Deferred import: pulls in the full pipeline (numpy/sentinel/etc.).
    from openweave_core.sidecar.assemble import assemble_analysis

    try:
        payload = await assemble_analysis(req.trace_id, run_eval=req.run_eval)
    except (KeyError, ValueError) as exc:
        # Trace not found / unparseable — caller error.
        log.warning("analyze 422 for %s: %s", req.trace_id, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface anything else as 500
        log.exception("analyze failed for %s", req.trace_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if req.project_id:
        payload["projectId"] = req.project_id
    return payload
