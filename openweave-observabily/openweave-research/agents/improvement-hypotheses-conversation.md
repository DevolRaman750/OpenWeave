# Improvement Hypotheses — Full Conversation Reference

> Written 2026-04-20. Reference this document in new Claude Code chats by saying:
> "Read improvement-hypotheses-conversation.md in the repo root and use it as context."

---

## 1. What We Are Building

A LangWatch feature called **Improvement Hypotheses** that goes beyond detecting agent failures.
It helps teams understand *why* an agent failed or is inefficient, maps failures to a curated pattern library, and generates ranked, evidence-backed improvement suggestions — with humans reviewing before anything is applied.

**The core promise:**
> "When an agent repeatedly underperforms, LangWatch should suggest grounded, testable improvement hypotheses drawn from trace evidence, curated implementation patterns, and selected external knowledge, with human review before action."

This is NOT:
- An autonomous code rewriter
- A prompt auto-tuner
- A paper-search engine
- A system that applies anything without human approval

---

## 2. The PRD (condensed)

Full PRD is in `improvement_hypotheses_prd.txt` in the repo root.

Key sections to reference:
- §5 Non-Goals — no auto-apply, no uncurated internet ingestion
- §12 Functional Requirements — ingestion, clustering, taxonomy, KB, ranking, human review
- §13 Knowledge Base Design — pattern units, not raw docs
- §14 Why Raw Repo Search Is Not Enough — explains the anti-pattern
- §18 Human-in-the-Loop Rules — mandatory, no exceptions in V1
- §22 V1 Release Plan — Phase 1 (taxonomy + clustering), Phase 2 (recommendation engine), Phase 3 (UI + workflow)

Architecture doc is in `langwatch_architecture.md` in the repo root (MVP + complete product diagrams in Mermaid).

---

## 3. What LangWatch Already Has (Relevant to This Feature)

### Already mature and usable:
| Capability | Key files |
|---|---|
| Trace ingestion (OTel spans, LLM calls, tool calls) | `langwatch/prisma/schema.prisma:255-323` |
| Evaluations (42 evaluators, built-in + custom) | `langwatch/src/server/evaluations/evaluators.generated.ts` |
| Human feedback/annotations (thumbs, scores, queues) | `schema.prisma:623-723`, `annotation.service.ts` |
| Topic clustering (centroid-based, hierarchical) | `langwatch/src/server/topicClustering/` |
| Datasets + experiments (batch eval, DSPY, V3) | `schema.prisma:441-559` |
| Monitors + triggers (email, Slack, add-to-dataset) | `schema.prisma:359-594` |
| Prompt versioning (full history, tags, org/project scope) | `schema.prisma:925-1019` |

### Already scaffolded (the new feature, work-in-progress on `main`):
| File | What it does |
|---|---|
| `langwatch/src/server/improvement-hypotheses/types.ts` | All type definitions |
| `langwatch/src/server/improvement-hypotheses/taxonomy.ts` | 9 failure/inefficiency labels |
| `langwatch/src/server/improvement-hypotheses/pattern-library.ts` | 29 hardcoded patterns |
| `langwatch/src/server/improvement-hypotheses/signal-extraction.ts` | Reads a trace → emits labels + signal tags |
| `langwatch/src/server/improvement-hypotheses/improvement-hypotheses.service.ts` | Clusters diagnostics, ranks patterns, builds hypotheses |
| `langwatch/src/server/improvement-hypotheses/index.ts` | Public exports |
| `langwatch/src/components/traces/ImprovementHypotheses.tsx` | UI panel (cluster + hypothesis display) |
| `langwatch/src/app/api/traces/[[...route]]/app.v1.ts` | Hono v1 API (alongside tRPC) |

**Key gaps in what's scaffolded:**
- No persistence — clusters are stateless, regenerated per query
- No accept/dismiss logging — human decisions are not stored
- No feedback loop — nothing learns from which hypotheses were useful
- Pattern library is a hardcoded TS file, not a DB table
- No external source ingestion pipeline

---

## 4. The 42 Evaluators — What Exists vs What's Missing

### Exists (relevant subset):
- RAG: faithfulness, context precision/recall, factual correctness, summarization
- Safety: PII, moderation, jailbreak, prompt injection
- Custom: llm_boolean, llm_score, llm_category, valid_format
- Agentic: `query_resolution` (pass/fail only, conversation-level)

### Missing (agentic metrics that matter for this feature):
| Missing metric | Maps to taxonomy label | Why it matters |
|---|---|---|
| Tool Correctness (tool name + args scored) | `wrong_tool_selection` | Turns heuristic label into a number |
| Step Efficiency (steps taken vs minimum) | `excessive_reasoning_or_latency` | Detects inefficiency without failure |
| Task Completion (scored, not just pass/fail) | `shallow_final_answer` | Numeric input to ProblemSpecification |
| Goal Accuracy | `shallow_final_answer` | End-to-end quality measure |

**Decision:** Do not add deepeval as a dependency. Add the 3 most critical agentic metrics to langevals directly, so they produce numeric `DeficiencyMetric` values for the problem spec.

---

## 5. The Inefficiency Extension (New Idea from This Conversation)

### The instinct
Most observability platforms detect *failures*. We want to also detect *working-but-suboptimal* subsystems — e.g., a semantic classifier that works at 82% accuracy when a better approach exists that achieves 94%.

### Why the original framing was problematic
- "Detect inefficiency from traces alone" — traces show behavior, not optimality. You need eval ground truth.
- "Auto-apply from a 3-day-old paper" — this directly contradicts PRD §5 and §18. New sources must go through curated staging.
- "The engine must know what prompt to write" — vague. It must produce a **structured problem specification**, not prose.

### How it fits the PRD without breaking principles
- Scope to known-better patterns in the curated KB (same as V1)
- Subsystem efficiency detected via **labeled eval slices**, not trace vibes alone
- Fresh sources (papers, repos) enter a **candidate_patterns staging table**, human promotes to live KB
- Output is still a ranked hypothesis + suggested eval — same workflow as failure detection

### Where inefficiency detection lives in the code today
In `signal-extraction.ts`, line ~301:
```ts
if (runtimeMs > 45_000 || trace.spans.length >= 8) {
  labels.add("excessive_reasoning_or_latency");
```
This is the ONLY current inefficiency trigger. It fires on time/step count alone.

**The extension needed:** a third trigger — subsystem tag + eval score below an *excellence* threshold (e.g., 0.75), not the *failure* threshold (0.6). That single addition makes the system surface "working but could be better."

---

## 6. The Problem Specification Schema (Your Task)

You and your friend split the work. Your part is building the **Problem Specification Engine** — takes a failure/inefficiency report + codebase context → outputs a structured spec the repo master agent consumes.

### Why it must be a schema, not a prompt
- A schema forces measurability — no "make it better" fields
- Debug path: if repo master returns garbage, you can point to which spec field misled it
- Interface contract: your friend's repo master component depends only on the schema, not on your implementation

### The full schema (agreed in this conversation)

```ts
// Location: langwatch/src/server/improvement-hypotheses/problem-specification/types.ts

type ProblemSpecification = {
  specId: string;
  generatedAt: number;
  sourceClusterLabel: FailureTaxonomyLabel;  // from existing types.ts
  projectId: string;

  problem: {
    subsystemType: SubsystemType;
    subsystemName: string;
    observedDeficiency: string;          // one sentence, measurable
    deficiencyMetrics: DeficiencyMetric[];
    failureSignals: string[];            // from TraceDiagnostic.signalTags
    affectedTraceCount: number;
    severity: FailureSeverity;           // from existing types.ts
  };

  currentApproach: {
    technique: string;
    entryPoint: CodeLocation;
    dependencies: CodeLocation[];
    callers: CodeLocation[];
    config: Record<string, string>;
    testFiles: CodeLocation[];
  };

  interfaceContract: {
    signature: string;
    invariants: string[];
    sideEffects: string[];
  };

  constraints: {
    mustHave: string[];
    niceToHave: string[];
    exclude: string[];
    performance: {
      maxLatencyMs?: number;
      maxCostPerCallUsd?: number;
    };
    environment: {
      language: string;
      framework?: string;
      runtime?: string;
      availableDependencies: string[];
    };
  };

  evidenceRequirements: {
    minBenchmarkOverlap: string;
    recencyPreference: "strict" | "preferred" | "none";
    reproducibility: "code_required" | "method_sufficient";
    minEvidenceStrength: number;  // 0..1
  };

  validationPlan: {
    suggestedEval: ImprovementHypothesisPattern["suggestedEval"];  // reuses existing type
    successCriteria: string;
    regressionGuards: string[];
  };

  nonGoals: string[];  // load-bearing — prevents repo master scope creep
};

type SubsystemType =
  | "classifier" | "retriever" | "planner" | "tool_router"
  | "reranker" | "memory" | "guardrail" | "other";

type DeficiencyMetric = {
  name: string;
  observed: number;
  baseline?: number;
  slice?: string;
};

type CodeLocation = {
  path: string;
  symbol?: string;
  startLine?: number;
  endLine?: number;
  justification: string;  // one line: why the slicer included this
};
```

### The three-stage pipeline for your component

```
LangWatch signals (TraceDiagnostic, FailureCluster — already computed)
        ↓
(1) Codebase Slicer        — static analysis, NO LLM
    output: { entryPoint, deps, callers, config, tests }
        ↓
(2) Problem Characterizer  — single LLM pass + schema validation
    output: draft ProblemSpecification
        ↓
(3) Schema Validator       — Zod schema, reject vague fields
    output: final ProblemSpecification → repo master
```

### Build order recommendation
Build stage (3) first — write the Zod schema and a validator that rejects bad specs. Then hand-write 3 reference specs for real failure types from `pattern-library.ts`. Only then build stages (1) and (2). Working backward from the contract is what keeps this shippable.

### Three questions to answer before writing code
1. **What schema does the repo master accept?** Agree in writing with your friend today.
2. **What's the smallest failure type to target end-to-end first?** One concrete example, hand-built ideal spec, then reverse-engineer the engine.
3. **How will you evaluate your prompt was good before repo master runs?** Need (failure, codebase) → ideal spec pairs offline.

---

## 7. How the Existing Tests Work (Run These Yourself)

### Run command
```bash
cd langwatch
pnpm test:unit src/server/improvement-hypotheses/__tests__/improvement-hypotheses.service.unit.test.ts
```

### Test 1: failure detection
Trace has: ambiguous input + tool span + no question + thumbs-down + score 0.2
Signal extraction produces: `too_few_clarifying_questions`, `premature_tool_execution`, `wrong_tool_selection`, `shallow_final_answer`

### Test 2: retrieval grounding
Trace has: rag span with `contexts: []` + groundedness score 0.3
Trigger: `ragSpans.length > 0` AND `contextCount === 0` simultaneously → `weak_retrieval_grounding`

### Test 3: inefficiency detection (add this yourself)
```ts
it("flags excessive reasoning even without negative feedback or low scores", () => {
  const service = new ImprovementHypothesesService();
  const spans = Array.from({ length: 9 }, (_, i) => ({
    span_id: `span-${i}`, trace_id: "trace-slow", type: "llm" as const,
    timestamps: { started_at: i * 5_000, finished_at: (i + 1) * 5_000 },
  }));
  const result = service.generate([
    makeTrace({
      trace_id: "trace-slow",
      spans,
      timestamps: { started_at: 0, inserted_at: 0, updated_at: 60_000 },
      events: [],       // no negative feedback
      evaluations: [{   // passing — agent IS completing the task
        evaluation_id: "eval-ok", evaluator_id: "evaluator-ok",
        name: "task_completion", status: "processed",
        score: 0.85, passed: true,
        timestamps: { started_at: 55_000, finished_at: 55_100 },
      }],
    }),
  ]);
  expect(result.clusters.map((c) => c.label)).toContain("excessive_reasoning_or_latency");
  const cluster = result.clusters.find((c) => c.label === "excessive_reasoning_or_latency");
  expect(cluster?.hypotheses[0]?.supportingPattern.id).toBe("latency-aware-reflection");
  expect(cluster?.hypotheses[0]?.suggestedEval.metric).toBe("quality_per_second");
});
```

---

## 8. Key Design Decisions Made in This Conversation

| Decision | Reasoning |
|---|---|
| No auto-apply ever | PRD §18 — trust model, liability, customers won't allow it |
| No raw internet-scale repo search | PRD §14 — semantic similarity ≠ production fitness |
| Fresh sources enter candidate staging, human promotes | Preserves curation without blocking fresh discovery |
| Problem spec is a schema, not prose | Forces measurability, gives debug path, creates clean interface |
| Build Zod validator (stage 3) before the engine (stages 1-2) | Working backward from the contract keeps it shippable |
| Add 3 agentic metrics to langevals, not deepeval | Avoid new dependency; produce numbers for DeficiencyMetric |
| Inefficiency trigger needs eval slice, not trace heuristics alone | Traces show behavior, not optimality — need ground truth |

---

## 9. Ranking Formula (Current Implementation)

In `improvement-hypotheses.service.ts`:
```ts
score = pattern.evidenceStrength * 3 + signalOverlap
```
Where `signalOverlap` = count of `pattern.requiredSignals` that appear in the cluster's `recurringSignals`.

Confidence thresholds:
- `score >= 4.4` → "high"
- `score >= 3.2` → "medium"
- below → "low"

Severity thresholds (cluster level):
- `traceCount * 2 + negativeFeedbackCount + (avgScore < 0.4 ? 2 : 0)`
- `>= 5` → "high", `>= 3` → "medium", else "low"

---

## 10. Next Steps (Where We Left Off)

1. **Your task:** Build the Problem Specification Engine (3-stage pipeline above)
   - Start: write the Zod schema for `ProblemSpecification`
   - Then: hand-write 3 reference specs from `pattern-library.ts` patterns
   - Then: build the codebase slicer (stage 1, static analysis)
   - Then: build the problem characterizer (stage 2, LLM pass)

2. **Shared boundary with friend:** Agree on the `ProblemSpecification` schema as the interface contract before either of you writes engine code.

3. **Inefficiency extension:** Add a third trigger to `signal-extraction.ts` — subsystem tag + eval score in "suboptimal" range (0.6–0.75) — to surface "working but could be better" without requiring failures.

4. **Missing agentic metrics:** Add Tool Correctness, Step Efficiency, Task Completion (scored) to langevals. These feed `deficiencyMetrics[]` in the spec.

5. **Pattern library persistence:** Move `pattern-library.ts` from hardcoded TS to a DB table. Add candidate_patterns staging table for human curation of fresh sources.

---

## 11. Files to Read in a New Chat for Full Context

```
improvement-hypotheses-conversation.md          ← this file
improvement_hypotheses_prd.txt                  ← full PRD
langwatch_architecture.md                       ← MVP + complete architecture diagrams

langwatch/src/server/improvement-hypotheses/types.ts
langwatch/src/server/improvement-hypotheses/taxonomy.ts
langwatch/src/server/improvement-hypotheses/pattern-library.ts
langwatch/src/server/improvement-hypotheses/signal-extraction.ts
langwatch/src/server/improvement-hypotheses/improvement-hypotheses.service.ts
langwatch/src/server/improvement-hypotheses/__tests__/improvement-hypotheses.service.unit.test.ts
```
