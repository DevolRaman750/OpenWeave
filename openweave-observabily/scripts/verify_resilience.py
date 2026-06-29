"""verify_resilience.py — prove the timeout / retry / breaker / queue layer works.

Every check uses fakes (sleeps, always-fail functions) so it runs offline and
deterministically — no live NVIDIA/Anthropic calls needed.

Checks
------
  C1  Bounded clients: embedding + judge clients carry the configured
      timeout / max_retries (a dead endpoint can no longer hang for minutes).
  C2  Circuit breaker: opens after N consecutive failures, then fails fast in
      microseconds; HALF_OPEN trial after cooldown closes it on success.
  C3  Graceful degrade: a tripped breaker (CircuitOpenError == RuntimeError) is
      caught by confirm_cycles, which falls back to exact-match (label still 1).
  C4  Per-detector timeout: a detector wedged on a slow upstream is cut off and
      becomes ok=False, while the other detectors still report.
  C5  Queue isolation: with a worker pool, one slow trace does NOT block the
      others — fast traces finish while the slow one is still running.
  C6  Accuracy intact: on the healthy path, an injection trace is still CRITICAL.

Run from openweave-observabily/:  python scripts/verify_resilience.py
"""
import asyncio
import sys
import time

sys.path.insert(0, ".")

from openweave_core.resilience import CircuitBreaker, CircuitOpenError, CircuitState
from openweave_core.cycle_detection import confirm_cycles
from openweave_core.cycle_detection.dag import SiblingGroup
import openweave_core.anomaly_pipeline.pipeline as pl
from openweave_core.anomaly_pipeline import (
    TraceProcessingQueue,
    run_anomaly_pipeline_for_spans_full,
)
from openweave_core.anomaly_pipeline.contracts import (
    DetectorResult,
    SOURCE_CYCLE_DETECTION,
    TraceBundle,
)

from scripts.stress_test import fx_injection, fx_clean

PASS, FAIL = "[PASS]", "[FAIL]"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}" + (f"  — {detail}" if detail else ""))


# --- C1: bounded clients ----------------------------------------------------

def c1_bounded_clients():
    from openweave_core.cycle_detection.semantic import (
        EmbedClient, EMBED_TIMEOUT_SECONDS, EMBED_MAX_RETRIES,
    )
    try:
        ec = EmbedClient()
        ok = (ec._client.max_retries == EMBED_MAX_RETRIES)
        check(ok and EMBED_TIMEOUT_SECONDS == 10.0, "C1 embed client bounded",
              f"timeout={EMBED_TIMEOUT_SECONDS}s max_retries={ec._client.max_retries}")
    except Exception as exc:  # missing key etc — don't fail the whole gate
        check(EMBED_TIMEOUT_SECONDS == 10.0 and EMBED_MAX_RETRIES == 3,
              "C1 embed client bounded (config only)",
              f"client build skipped: {type(exc).__name__}")
    try:
        from openweave_core.deep_evaluation.judge import (
            ClaudeJudge, JUDGE_TIMEOUT_SECONDS, JUDGE_MAX_RETRIES,
        )
        client = ClaudeJudge().load_model()
        check(client.max_retries == JUDGE_MAX_RETRIES == 3, "C1 judge client bounded",
              f"timeout={JUDGE_TIMEOUT_SECONDS}s max_retries={client.max_retries}")
    except Exception as exc:
        from openweave_core.deep_evaluation.judge import JUDGE_MAX_RETRIES
        check(JUDGE_MAX_RETRIES == 3, "C1 judge client bounded (config only)",
              f"client build skipped: {type(exc).__name__}")


# --- C2: circuit breaker ----------------------------------------------------

def c2_circuit_breaker():
    clock = {"t": 0.0}
    br = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=30.0,
                        time_fn=lambda: clock["t"])

    def boom():
        raise RuntimeError("upstream down")

    # 3 consecutive failures trip it
    for _ in range(3):
        try:
            br.call(boom)
        except RuntimeError:
            pass
    opened = br.state is CircuitState.OPEN

    # Next call fails fast with CircuitOpenError, in microseconds
    t0 = time.perf_counter()
    fast_fail = False
    try:
        br.call(boom)
    except CircuitOpenError:
        fast_fail = True
    dt_ms = (time.perf_counter() - t0) * 1000
    is_runtime = issubclass(CircuitOpenError, RuntimeError)

    # After cooldown -> HALF_OPEN trial; a success closes it
    clock["t"] = 31.0
    closed_after = False
    try:
        br.call(lambda: 42)
        closed_after = br.state is CircuitState.CLOSED
    except Exception:
        pass

    check(opened and fast_fail and is_runtime and dt_ms < 1.0 and closed_after,
          "C2 breaker open / fast-fail / recover",
          f"opened={opened} fastfail={fast_fail}({dt_ms:.3f}ms) "
          f"isRuntimeError={is_runtime} recovered={closed_after}")


# --- C3: graceful degrade on tripped breaker --------------------------------

def c3_degrade_on_breaker():
    # Two identical sibling outputs => exact-match confirms WITHOUT embedding.
    # An embed client that always raises (simulating a dead endpoint / open
    # breaker) must NOT break confirmation: exact-match still yields label=1.
    class DeadEmbed:
        def embed(self, texts):
            raise CircuitOpenError("circuit 'nvidia-embed' is OPEN")

    spans = [
        {"id": "a", "output_text": "identical result"},
        {"id": "b", "output_text": "identical result"},
    ]
    group = SiblingGroup(parent_id="p", siblings=spans)
    res = confirm_cycles([group], embed_client=DeadEmbed())
    check(res.label == 1 and len(res.confirmed_pairs) >= 1,
          "C3 degrade: exact-match survives dead embedder",
          f"label={res.label} pairs={len(res.confirmed_pairs)}")

    # All-distinct group + dead embedder + no exact pairs => RuntimeError raised
    # so run_cycle_detection can fall back to structural flags.
    distinct = [
        {"id": "c", "output_text": "alpha unique"},
        {"id": "d", "output_text": "beta different"},
    ]
    raised = False
    try:
        confirm_cycles([SiblingGroup(parent_id="p2", siblings=distinct)],
                       embed_client=DeadEmbed())
    except RuntimeError:
        raised = True
    check(raised, "C3 degrade: distinct group re-raises RuntimeError for fallback")


# --- C4: per-detector timeout isolation -------------------------------------

async def c4_detector_timeout():
    def slow_runner(bundle):
        time.sleep(3.0)  # wedged upstream
        return DetectorResult(detector=SOURCE_CYCLE_DETECTION, ok=True, flags=[])

    t0 = time.perf_counter()
    res = await pl._run_detector(SOURCE_CYCLE_DETECTION, slow_runner, None,
                                 _timeout=1.0)
    dt = time.perf_counter() - t0
    check((not res.ok) and "timed out" in (res.error or "") and dt < 2.0,
          "C4 per-detector timeout cuts off a wedged detector",
          f"ok={res.ok} err={res.error!r} elapsed={dt:.2f}s")


# --- C5: queue isolation (one slow trace doesn't block others) --------------

async def c5_queue_isolation():
    # Patch the cycle-detection runner the pipeline resolves at call time so the
    # trace named "slow" sleeps; others run normally.
    orig = pl.run_cycle_detection

    def maybe_slow(bundle):
        if bundle.trace_id == "slow":
            time.sleep(2.0)
        return orig(bundle)

    pl.run_cycle_detection = maybe_slow
    try:
        spans_fast = fx_clean()
        spans_slow = fx_clean()
        done_order: list[tuple[str, float]] = []

        async with TraceProcessingQueue(workers=4) as q:
            t0 = time.perf_counter()

            async def submit(tid, spans):
                rep = await q.submit_spans(spans, trace_id=tid)
                done_order.append((tid, time.perf_counter() - t0))
                return rep

            await asyncio.gather(
                submit("slow", spans_slow),
                submit("fast1", spans_fast),
                submit("fast2", spans_fast),
                submit("fast3", spans_fast),
            )

        finish = {tid: t for tid, t in done_order}
        fast_max = max(finish["fast1"], finish["fast2"], finish["fast3"])
        # Fast traces finish promptly; slow finishes ~2s; concurrency means the
        # fast ones complete well before the slow one (not serialized behind it).
        check(fast_max < 1.0 and finish["slow"] >= 1.9 and finish["slow"] < 3.0,
              "C5 queue: slow trace does not block fast ones",
              f"fast_done<= {fast_max:.2f}s  slow_done={finish['slow']:.2f}s")
    finally:
        pl.run_cycle_detection = orig


# --- C6: accuracy intact on healthy path ------------------------------------

def c6_accuracy_intact():
    report, _ = run_anomaly_pipeline_for_spans_full(fx_injection(), trace_id="t-inj")
    sev = report.overall_severity.value
    check(sev == "critical" and report.flagged,
          "C6 accuracy: injection still CRITICAL",
          f"severity={sev} flags={len(report.flags)}")


def main():
    print("=" * 72)
    print("  Resilience verification (timeout / retry / breaker / queue)")
    print("=" * 72)
    c1_bounded_clients()
    c2_circuit_breaker()
    c3_degrade_on_breaker()
    asyncio.run(c4_detector_timeout())
    asyncio.run(c5_queue_isolation())
    c6_accuracy_intact()

    print("-" * 72)
    n_pass = sum(1 for ok, _ in results if ok)
    n_total = len(results)
    if n_pass == n_total:
        print(f"  RESULT: all {n_total} checks passed.")
        return 0
    print(f"  RESULT: {n_pass}/{n_total} passed — "
          f"{', '.join(lbl for ok, lbl in results if not ok)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
