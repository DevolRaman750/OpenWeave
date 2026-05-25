"""
adaptive_baseline — Adaptive Multi-Dimensional Monitoring (AMDM) pipeline.

Implements the three-step AMDM algorithm described in IDEA.md:

    Step 1 : MetricExtractor      → raw 5-axis MetricVector per span
    Step 2 : Normalizer           → rolling z-score per metric (w = 80)
    Step 3 : AxisAggregator       → per-axis aggregation + EWMA + axis anomaly
    Step 4 : MahalanobisDetector  → 5-D joint anomaly via χ²₅(0.99) + shrinkage

Public entry points
-------------------
    from openweave_core.adaptive_baseline import MetricExtractor, Normalizer

    extractor  = MetricExtractor(safety_evaluator=my_judge)
    normaliser = Normalizer()                         # window_size=80

    for span in parse_langfuse_trace(trace_id):
        raw  = extractor.extract(span)
        zvec = normaliser.normalize(raw)
        # zvec.capability.latency, zvec.economic.cost, ...

    extractor.finalize_trace(trace_id)
"""

from openweave_core.adaptive_baseline.metrics import (
    BASELINE_HISTORY_SIZE,
    BASELINE_MIN_SAMPLES,
    MetricExtractor,
    MetricVector,
    NullSafetyEvaluator,
    SafetyEvaluator,
    ToolVariationBaseline,
)
from openweave_core.adaptive_baseline.normalization import (
    AXIS_FIELDS,
    WINDOW_SIZE,
    CapabilityZ,
    EconomicZ,
    HumanZ,
    NormalizedMetricVector,
    Normalizer,
    RobustnessZ,
    RollingZScorer,
    SafetyZ,
)
from openweave_core.adaptive_baseline.aggregation import (
    AXIS_ORDER,
    K_DEFAULT,
    LAMBDA_DEFAULT,
    WARMUP_STEPS_DEFAULT,
    AxisAggregator,
    AxisEvaluation,
    AxisScore,
    EWMABaseline,
)
from openweave_core.adaptive_baseline.joint_anomaly import (
    CHI2_5_0_99,
    DEFAULT_ALPHA,
    DEFAULT_SHRINKAGE_FLOOR,
    DEFAULT_WARMUP_STEPS,
    JOINT_DIM,
    JointAnomalyResult,
    MahalanobisDetector,
    WelfordCovariance,
)

__all__ = [
    # Step 1
    "MetricExtractor",
    "MetricVector",
    "SafetyEvaluator",
    "NullSafetyEvaluator",
    "ToolVariationBaseline",
    "BASELINE_HISTORY_SIZE",
    "BASELINE_MIN_SAMPLES",
    # Step 2
    "Normalizer",
    "NormalizedMetricVector",
    "RollingZScorer",
    "CapabilityZ",
    "EconomicZ",
    "RobustnessZ",
    "SafetyZ",
    "HumanZ",
    "AXIS_FIELDS",
    "WINDOW_SIZE",
    # Step 3
    "AxisAggregator",
    "AxisEvaluation",
    "AxisScore",
    "EWMABaseline",
    "AXIS_ORDER",
    "K_DEFAULT",
    "LAMBDA_DEFAULT",
    "WARMUP_STEPS_DEFAULT",
    # Step 4
    "MahalanobisDetector",
    "JointAnomalyResult",
    "WelfordCovariance",
    "JOINT_DIM",
    "CHI2_5_0_99",
    "DEFAULT_ALPHA",
    "DEFAULT_SHRINKAGE_FLOOR",
    "DEFAULT_WARMUP_STEPS",
]
