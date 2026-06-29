"""
full_pipeline_test.py — REAL end-to-end pipeline exercise (not a smoke test).

Runs the whole OpenWeave chain against a *live* Langfuse trace and answers the
operator's questions explicitly:

  1. Does the silent anomaly-detection engine actually fire on a real trace,
     and which engines contribute? (per-source flag counts + evidence)
  2. Is the output *correct* — i.e. not a constant? We prove this two ways:
       (a) DETERMINISM: re-run the engines N times on the same spans; the flag
           set must be stable (same inputs -> same verdict).
       (b) DISCRIMINATION (true-positive vs true-negative): run the engines on
           the live (dirty) trace AND on a synthetically-cleaned copy of the
           same spans; the dirty trace must flag and the clean one must not.
  3. Does deep evaluation actually run for RISK/CRITICAL traces, and what are
     the outcomes? (live Claude judge via DeepEval)
  4. Emits a JSON report to scripts/out/<trace_id>.report.json for the UI.

Usage
-----
    cd openweave-observabily
    python scripts/full_pipeline_test.py [<trace_id>]   # default: latest trace
    python scripts/full_pipeline_test.py --no-eval       # skip live Claude calls
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from base64 import b64encode
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

# Ensure the package + .env bootstrap (the parser does this on import).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openweave_core.parser.trace_fetcher import parse_langfuse_trace_full  # noqa: E402
from openweave_core.anomaly_pipeline import (  # noqa: E402
    run_anomaly_pipeline_for_spans_full,
    ALL_SOURCES,
)
from openweave_core.incident_classification import classify_trace  # noqa: E402
from openweave_core.sentinel_agent.findings import Severity  # noqa: E402

GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; DIM = "\033[2m"; RST = "\033[0m"
RISK_SEVERITIES = (Severity.RISK, Severity.CRITICAL)


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _discover_latest_trace_id() -> str:
    base = (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
            or "https://cloud.langfuse.com").rstrip("/")
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
    auth = b64encode(f"{pub}:{sec}".encode()).decode()
    req = urllib.request.Request(
        f"{base}/api/public/traces?limit=1",
        headers={"Authorization": f"Basic {auth}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    traces = data.get("data") or []
    if not traces:
        raise RuntimeError("No traces found in Langfuse")
    return traces[0]["id"]


def _flag_signature(report) -> tuple:
    """A stable, order-independent fingerprint of a report's flags."""
    return tuple(sorted(
        (f.source_pipeline, f.category, f.subject_type, f.subject_id, f.severity.value)
        for f in report.flags
    ))


_INJECTION_MARKERS = (
    "ignore previous instructions", "exfiltrate", "system secrets",
    "reveal the system prompt", "system prompt", "ignore previous",
)


def _clean_spans(spans):
    """Return a copy of the spans with the anomalies scrubbed out.

    A genuinely clean trace must defeat BOTH detectors:
      * sentinel    — strip every prompt-injection / jailbreak marker from text.
      * cycle (structural) — break the repeated operation sequence. The
        structural CDCS keys on the *op-name* sequence (tool_name / span_type),
        not on text, so we must make each iteration's op-name unique too.

    This is the true-negative control: same overall shape, no anomalies.
    """
    cleaned = []
    for i, s in enumerate(deepcopy(spans)):
        txt_in = (s.input_text or "")
        txt_out = (s.output_text or "")
        for bad in _INJECTION_MARKERS:
            txt_in = txt_in.replace(bad, "summarise the findings")
            txt_out = txt_out.replace(bad, "summary of findings")
        # Make op-names unique per iteration so no operation sequence repeats.
        tool = s.tool_name
        if tool and tool != "research-agent":
            tool = f"{tool}-{i}"
        s = replace(
            s,
            tool_name=tool,
            input_text=f"{txt_in} [step {i} unique]",
            output_text=f"{txt_out} [result {i} unique]",
        )
        cleaned.append(s)
    return cleaned


def _summarise_report(report, label: str) -> None:
    print(f"  [{label}] flagged={report.flagged}  "
          f"overall_severity={report.overall_severity.value}  "
          f"n_flags={len(report.flags)}")
    print(f"          by_source = {report.by_source}")
    failed = report.failed_detectors()
    if failed:
        print(f"          {YEL}failed_detectors = {failed}{RST}")
    for r in report.detector_results:
        status = f"{GREEN}ok{RST}" if r.ok else f"{RED}FAIL{RST}"
        warn = f"  warn={r.warning}" if r.warning else ""
        print(f"            - {r.detector:<18} {status}  flags={r.flag_count}  "
              f"{r.duration_seconds*1000:6.1f}ms{warn}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_eval = "--no-eval" not in sys.argv

    _hr("STAGE 0 — Parse live Langfuse trace")
    trace_id = args[0] if args else _discover_latest_trace_id()
    print(f"  trace_id = {trace_id}")
    envelope, spans = parse_langfuse_trace_full(trace_id)
    print(f"  parsed {len(spans)} spans; "
          f"envelope.input={(envelope.input_text or '')[:60]!r}")
    print(f"  envelope.tags={envelope.tags}  session={envelope.session_id}")
    if not spans:
        print(f"  {RED}No spans parsed — is the worker ingesting? Aborting.{RST}")
        return 2

    # ------------------------------------------------------------------ 1+2a
    _hr("STAGE 1 — Silent anomaly engines on the DIRTY (live) trace")
    report_dirty, bundle = run_anomaly_pipeline_for_spans_full(
        spans, trace_id=trace_id, envelope=envelope
    )
    _summarise_report(report_dirty, "dirty")

    sources_fired = {f.source_pipeline for f in report_dirty.flags}
    print(f"\n  engines that fired: {sorted(sources_fired)}")
    print(f"  sample evidence (first 6 flags):")
    for f in report_dirty.flags[:6]:
        print(f"    [{f.severity.value:<8}] {f.source_pipeline:<16} {f.category:<26} "
              f"conf={f.confidence:.2f}  subj={f.subject_id[:16]}")
        if f.message:
            print(f"        {DIM}{f.message[:90]}{RST}")

    _hr("STAGE 2a — DETERMINISM: re-run engines 3x on identical spans")
    sig0 = _flag_signature(report_dirty)
    stable = True
    for i in range(1, 4):
        rep_i, _ = run_anomaly_pipeline_for_spans_full(
            spans, trace_id=trace_id, envelope=envelope
        )
        sig_i = _flag_signature(rep_i)
        same = (sig_i == sig0)
        stable = stable and same
        print(f"  run#{i}: n_flags={len(rep_i.flags)}  signature_matches_run0={same}")
    print(f"  -> deterministic: {GREEN+'YES'+RST if stable else RED+'NO'+RST} "
          f"(same spans must give same verdict)")

    _hr("STAGE 2b — DISCRIMINATION: true-positive vs true-negative control")
    clean = _clean_spans(spans)
    report_clean, _ = run_anomaly_pipeline_for_spans_full(
        clean, trace_id=f"{trace_id}-cleaned", envelope=envelope
    )
    _summarise_report(report_clean, "clean")
    dirty_anom = report_dirty.has_anomaly()
    clean_anom = report_clean.has_anomaly()
    tp = dirty_anom            # dirty SHOULD raise a RISK/CRITICAL
    tn = not clean_anom        # clean SHOULD stay below RISK
    print(f"\n  dirty trace RISK/CRITICAL present : {dirty_anom}  "
          f"({GREEN+'true-positive OK'+RST if tp else RED+'MISS (false-negative)'+RST})")
    print(f"  clean trace RISK/CRITICAL present : {clean_anom}  "
          f"({GREEN+'true-negative OK'+RST if tn else RED+'false-positive'+RST})")
    discriminates = (_flag_signature(report_dirty) != _flag_signature(report_clean))
    print(f"  dirty vs clean produce DIFFERENT flags: "
          f"{GREEN+'YES'+RST if discriminates else RED+'NO (constant output!)'+RST}")

    # -------------------------------------------------------------------- 3
    _hr("STAGE 3 — Classification: incidents + categories + evaluation plan")
    incident_report = classify_trace(report_dirty, bundle)
    print(f"  n_incidents = {len(incident_report.incidents)}  "
          f"overall_severity = {incident_report.overall_severity.value}")
    print(f"  category_summary = "
          f"{ {c.value: n for c, n in incident_report.category_summary.items()} }")
    risk_incidents = [i for i in incident_report.incidents
                      if i.severity in RISK_SEVERITIES]
    print(f"  RISK/CRITICAL incidents = {len(risk_incidents)} of "
          f"{len(incident_report.incidents)}")
    for inc in incident_report.incidents:
        print(f"    [{inc.id}] sev={inc.severity.value:<8} "
              f"sources={sorted(inc.source_pipelines)} "
              f"cats={[c.value for c in inc.categories]}")
    print(f"  evaluation_plan: {len(incident_report.evaluation_plan)} tasks "
          f"(one per incident x category)")

    eval_batch = None
    if do_eval:
        _hr("STAGE 4 — Deep evaluation on RISK traces (LIVE Claude judge)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(f"  {YEL}ANTHROPIC_API_KEY not set — skipping live eval{RST}")
        else:
            from openweave_core.deep_evaluation import evaluate_incident_report
            judge_model = os.environ.get("OPENWEAVE_JUDGE_MODEL", "claude-sonnet-4-6")
            print(f"  judge model = {judge_model}")
            print(f"  sampling policy: CRITICAL=1.0 RISK=0.5 WARNING=0.1 INFO=0.0")
            eval_batch = evaluate_incident_report(incident_report)
            print(f"  tasks: total={eval_batch.n_tasks_total} "
                  f"sampled(ran)={eval_batch.n_tasks_sampled} "
                  f"skipped={eval_batch.n_tasks_skipped} "
                  f"errored={eval_batch.n_tasks_errored}")
            print(f"  duration = {eval_batch.duration_ms:.0f}ms")
            print(f"\n  outcomes:")
            for o in eval_batch.outcomes():
                if o.skipped_reason:
                    print(f"    {DIM}- SKIP  {o.category.value:<16} {o.skipped_reason}{RST}")
                    continue
                verdict = (f"{GREEN}PASS{RST}" if o.passed
                           else f"{RED}FAIL{RST}" if o.passed is False
                           else f"{YEL}ERR{RST}")
                score = f"{o.score:.2f}" if o.score is not None else "  - "
                print(f"    - {verdict}  {o.category.value:<16} {o.metric_name:<34} "
                      f"score={score}  inc={o.incident_id}")
                if o.reason:
                    print(f"        {DIM}{o.reason[:110]}{RST}")
                if o.error:
                    print(f"        {RED}error: {o.error[:110]}{RST}")
    else:
        _hr("STAGE 4 — Deep evaluation SKIPPED (--no-eval)")

    # -------------------------------------------------------------------- 5
    _hr("STAGE 5 — Emit JSON report for the UI")
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "trace_id": trace_id,
        "anomaly": report_dirty.as_dict(),
        "incidents": incident_report.as_dict(),
        "evaluation": eval_batch.as_dict() if eval_batch else None,
        "checks": {
            "engines_fired": sorted(sources_fired),
            "deterministic": stable,
            "true_positive": tp,
            "true_negative": tn,
            "discriminates_dirty_vs_clean": discriminates,
        },
    }
    out_path = out_dir / f"{trace_id}.report.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {out_path}")

    # -------------------------------------------------------------------- verdict
    _hr("VERDICT")
    checks = payload["checks"]
    ok = (checks["engines_fired"] and checks["deterministic"]
          and checks["true_positive"] and checks["true_negative"]
          and checks["discriminates_dirty_vs_clean"])
    for k, v in checks.items():
        mark = GREEN + "PASS" + RST if v else RED + "FAIL" + RST
        print(f"  [{mark}] {k}: {v}")
    if do_eval and eval_batch is not None:
        ran = eval_batch.n_tasks_sampled > 0
        print(f"  [{GREEN+'PASS'+RST if ran else YEL+'WARN'+RST}] "
              f"deep_eval_ran_on_risk_traces: {ran} "
              f"({eval_batch.n_tasks_sampled} judged)")
    print("\n  " + (GREEN + "FULL PIPELINE HEALTHY" + RST if ok
                    else YEL + "PIPELINE RAN — see failed checks above" + RST))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
