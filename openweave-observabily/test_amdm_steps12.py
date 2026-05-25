"""
test_amdm_steps12.py — Integration test for AMDM Step 1 + Step 2.

Builds synthetic Langfuse-shaped ParsedSpan traces (no network/Langfuse
dependency), runs them through:

    MetricExtractor (Step 1)  ->  Normalizer (Step 2)

and asserts that:
  * every tracked metric on every axis emits a numeric z-score for spans
    with present inputs, and ``None`` for spans whose raw input is missing;
  * anomalous spans produce |z| well above the surrounding baseline;
  * goal-drift (sudden new tool usage) lights up Axis 3;
  * a failure span drives the success z-score sharply negative;
  * window size is exactly 80 and z-scores are 0 during cold start.

Run:
    cd openweave-observabily
    python test_amdm_steps12.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from typing import Optional

from openweave_core.adaptive_baseline import (
    AXIS_FIELDS,
    WINDOW_SIZE,
    MetricExtractor,
    NormalizedMetricVector,
    Normalizer,
)
from openweave_core.models.span import ParsedSpan

# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def _ok(name: str) -> None:
    print(f"  {GREEN}[PASS]{RESET}  {name}")


def _fail(name: str, msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET}  {name}: {msg}")


# ---------------------------------------------------------------------------
# Synthetic span factory
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 5, 24, 12, 0, 0)
_counter = {"i": 0}


def make_span(
    *,
    trace_id: str = "trace-test-1",
    tool_name: Optional[str] = "search",
    latency: Optional[float] = 0.5,
    total_tokens: int = 200,
    cost: Optional[float] = 0.0005,
    status_message: Optional[str] = None,
    level: Optional[str] = None,
    output_text: str = "ok",
    metadata: Optional[dict] = None,
    parent_id: Optional[str] = "root",
    span_type: str = "tool",
) -> ParsedSpan:
    """Build a minimally-populated ParsedSpan with a monotonically increasing timestamp."""
    _counter["i"] += 1
    return ParsedSpan(
        id=f"s{_counter['i']:04d}",
        trace_id=trace_id,
        span_type=span_type,
        tool_name=tool_name,
        input_text="...",
        output_text=output_text,
        input_hash="",
        input_tokens=total_tokens // 2,
        output_tokens=total_tokens - (total_tokens // 2),
        total_tokens=total_tokens,
        latency=latency,
        cost=cost,
        timestamp=_T0 + timedelta(seconds=_counter["i"]),
        start_time=_T0 + timedelta(seconds=_counter["i"]),
        end_time=_T0 + timedelta(seconds=_counter["i"] + (latency or 0.5)),
        parent_id=parent_id,
        model="gpt-test",
        metadata=metadata,
        status_message=status_message,
        level=level,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_window_size_constant() -> None:
    assert WINDOW_SIZE == 80, f"WINDOW_SIZE must be 80 per IDEA.md, got {WINDOW_SIZE}"


def test_cold_start_returns_zero() -> None:
    """First span has no prior history → all observed z-scores must be 0.0."""
    ext, norm = MetricExtractor(), Normalizer()
    span = make_span(metadata={"user_rating": 1})
    raw = ext.extract(span)
    z = norm.normalize(raw)

    # All present inputs → z == 0.0 (cold start, n=1).
    for name in ("latency", "total_tokens", "success", "cost",
                 "tool_diversity", "tool_drift_score", "new_tool_introduced",
                 "user_feedback"):
        value = z.flat()[name]
        assert value == 0.0, f"cold-start z for {name} should be 0.0, got {value}"

    # safety_score has no judge wired → must stay None
    assert z.safety.safety_score is None, "safety_score must be None without a judge"


def test_all_axes_populated_per_span() -> None:
    """After warmup, every metric (except safety w/o judge) emits a float."""
    ext, norm = MetricExtractor(), Normalizer()
    # Feed 10 stable spans so windows have >=2 samples for every metric.
    for _ in range(10):
        norm.normalize(ext.extract(make_span(metadata={"rating": 1.0})))

    z = norm.normalize(ext.extract(make_span(metadata={"rating": 1.0})))
    flat = z.flat()
    for axis, fields in AXIS_FIELDS.items():
        for f in fields:
            v = flat[f]
            if f == "safety_score":
                assert v is None, "safety_score must be None without a judge"
                continue
            assert isinstance(v, float) and not math.isnan(v), (
                f"{axis}.{f} produced non-float z-score: {v!r}"
            )


def test_anomaly_latency_and_tokens() -> None:
    """80 stable spans, then a spike → z >> 2 on latency and total_tokens."""
    ext, norm = MetricExtractor(), Normalizer()

    # Warmup window (slightly noisy baseline)
    for i in range(WINDOW_SIZE):
        norm.normalize(ext.extract(make_span(
            latency=0.5 + 0.02 * (i % 5),     # 0.50 .. 0.58
            total_tokens=200 + (i % 7),       # 200 .. 206
        )))

    # Anomalous span
    spike = norm.normalize(ext.extract(make_span(
        latency=50.0,
        total_tokens=20000,
    )))

    assert spike.capability.latency is not None
    assert spike.capability.total_tokens is not None
    assert spike.capability.latency > 5.0, (
        f"latency spike z should be huge, got {spike.capability.latency:.2f}"
    )
    assert spike.capability.total_tokens > 5.0, (
        f"total_tokens spike z should be huge, got {spike.capability.total_tokens:.2f}"
    )


def test_failure_signal_drives_success_negative() -> None:
    """Mostly success=1, then one ERROR span → success z should be deeply negative."""
    ext, norm = MetricExtractor(), Normalizer()

    # Mostly successes with one early failure so stdev > 0
    norm.normalize(ext.extract(make_span(level="ERROR")))
    for _ in range(WINDOW_SIZE - 1):
        norm.normalize(ext.extract(make_span()))

    failure = norm.normalize(ext.extract(make_span(
        level="ERROR", status_message="tool call failed"
    )))

    assert failure.capability.success is not None
    assert failure.capability.success < -2.0, (
        f"success z on failure should be < -2, got {failure.capability.success:.2f}"
    )


def test_goal_drift_lights_axis3() -> None:
    """Long stable run on one tool, then a new tool → new_tool_introduced z spikes."""
    ext, norm = MetricExtractor(), Normalizer()

    # Same-tool warmup (no introductions after the first span)
    for _ in range(WINDOW_SIZE):
        norm.normalize(ext.extract(make_span(tool_name="search", trace_id="drift-trace")))

    # Sudden new tool
    drift = norm.normalize(ext.extract(make_span(
        tool_name="execute_shell", trace_id="drift-trace",
    )))

    assert drift.robustness.new_tool_introduced is not None
    assert drift.robustness.new_tool_introduced > 2.0, (
        f"new-tool z should spike, got {drift.robustness.new_tool_introduced:.2f}"
    )
    assert drift.robustness.tool_diversity is not None
    assert drift.robustness.tool_diversity > 2.0, (
        f"tool_diversity z should spike, got {drift.robustness.tool_diversity:.2f}"
    )


def test_none_input_yields_none_zscore() -> None:
    """A span with cost=None / no feedback must emit None for those z-scores."""
    ext, norm = MetricExtractor(), Normalizer()

    # Warm up cost so the window has data
    for _ in range(10):
        norm.normalize(ext.extract(make_span(cost=0.001)))

    z = norm.normalize(ext.extract(make_span(
        cost=None,
        metadata=None,      # no user feedback at all
    )))

    assert z.economic.cost is None, "cost z should be None when input cost is None"
    assert z.human.user_feedback is None, (
        "user_feedback z should be None when metadata has no feedback key"
    )
    # Sanity: a present field still produces a number.
    assert isinstance(z.capability.latency, float)


def test_window_cap_is_80() -> None:
    """Feed 200 fully-populated spans → every metric's window must cap at 80.

    Fields that receive ``None`` inputs (e.g. safety_score without a judge,
    user_feedback with no metadata) are correctly skipped — their windows
    stay empty. To make the cap assertion meaningful we feed *every*
    observable field on every span.
    """
    ext, norm = MetricExtractor(), Normalizer()
    for _ in range(200):
        norm.normalize(ext.extract(make_span(metadata={"rating": 1.0})))

    sizes = norm.window_sizes()
    for name, size in sizes.items():
        if name == "safety_score":
            # No judge wired in → window correctly stays empty.
            assert size == 0, f"safety window should be empty without a judge, got {size}"
            continue
        assert size == 80, f"window for {name} should cap at 80, got {size}"


def test_metadata_thumbs_string_parsing() -> None:
    """thumbs_up / thumbs_down strings must become 1.0 / 0.0 before normalisation."""
    ext, norm = MetricExtractor(), Normalizer()
    # Warmup with thumbs_up
    for _ in range(WINDOW_SIZE):
        norm.normalize(ext.extract(make_span(metadata={"user_feedback": "thumbs_up"})))
    # Anomalous thumbs_down
    z = norm.normalize(ext.extract(make_span(metadata={"user_feedback": "thumbs_down"})))
    assert z.human.user_feedback is not None
    assert z.human.user_feedback < -2.0, (
        f"thumbs_down after run of thumbs_up should be z << -2, got {z.human.user_feedback:.2f}"
    )


# ---------------------------------------------------------------------------
# Visual end-to-end demo (also acts as smoke test)
# ---------------------------------------------------------------------------

def end_to_end_demo() -> NormalizedMetricVector:
    """Show what a real Step1->Step2 pipeline emits for one anomalous span."""
    print(f"\n{DIM}--- end-to-end demo ---{RESET}")

    ext, norm = MetricExtractor(), Normalizer()

    # 80 baseline spans
    for i in range(WINDOW_SIZE):
        norm.normalize(ext.extract(make_span(
            tool_name="search",
            latency=0.5 + 0.05 * (i % 4),
            total_tokens=200 + (i % 5),
            cost=0.0005,
            metadata={"rating": 1.0},
            trace_id="demo-trace",
        )))

    # 1 anomalous span: huge latency + new tool + failure + thumbs-down
    anomaly_raw = ext.extract(make_span(
        tool_name="execute_arbitrary_code",
        latency=42.0,
        total_tokens=15000,
        cost=0.05,
        level="ERROR",
        status_message="hard timeout",
        metadata={"rating": 0.0},
        trace_id="demo-trace",
    ))
    anomaly_z = norm.normalize(anomaly_raw)

    print(f"\n  Anomalous span z-scores:")
    flat = anomaly_z.flat()
    for axis, fields in AXIS_FIELDS.items():
        line = f"    {axis:<11} "
        parts = []
        for f in fields:
            v = flat[f]
            parts.append(f"{f}={v:+.2f}" if isinstance(v, float) else f"{f}=None")
        print(line + "  ".join(parts))

    return anomaly_z


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("window-size-constant",            test_window_size_constant),
    ("cold-start-returns-zero",         test_cold_start_returns_zero),
    ("all-axes-populated-per-span",     test_all_axes_populated_per_span),
    ("anomaly-latency-and-tokens",      test_anomaly_latency_and_tokens),
    ("failure-signal-drives-success",   test_failure_signal_drives_success_negative),
    ("goal-drift-lights-axis-3",        test_goal_drift_lights_axis3),
    ("none-input-yields-none-zscore",   test_none_input_yields_none_zscore),
    ("window-cap-is-80",                test_window_cap_is_80),
    ("metadata-thumbs-string-parsing",  test_metadata_thumbs_string_parsing),
]


def main() -> int:
    print("=" * 64)
    print("  AMDM Step 1 + Step 2 — Integration Test")
    print("=" * 64)

    failures: list[tuple[str, str]] = []
    for name, fn in TESTS:
        # Reset the global id counter so tests are independent.
        _counter["i"] = 0
        try:
            fn()
        except AssertionError as exc:
            failures.append((name, str(exc)))
            _fail(name, str(exc))
        except Exception as exc:  # unexpected hard error
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            _fail(name, f"{type(exc).__name__}: {exc}")
        else:
            _ok(name)

    end_to_end_demo()

    print("\n" + "=" * 64)
    if failures:
        print(f"  RESULT: {len(failures)} of {len(TESTS)} tests FAILED")
        print("=" * 64)
        return 1
    print(f"  RESULT: all {len(TESTS)} tests passed")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
