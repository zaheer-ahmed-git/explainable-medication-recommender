# Plan: explain the frozen Transformer–GNN ranking

## Goal and status

Implement the [proposed idea](ProposedExplainabilityIdea.md) in bounded stages
under [Roadmap Milestone 9](../DataFoundationRoadmap.md#milestone-9-grounded-explanation).
The first deliverable is a numerical core and a shared-intervention attribution
runner tested on synthetic slates. Full clinical explanation and novelty remain
unvalidated. This plan does not reinstate the removed evidence-ledger schema.

## Context and assumptions

- The authoritative fusion is `pipeline/gnn_training/fusion.py`: population
  standard deviation, `max(std, 1e-6)`, and the joint finite candidate mask.
- The selected blend is additive in normalized scores. There is no learned
  cross-branch interaction term to attribute. Shared inputs and nonlinear
  within-branch processing can still create input interactions.
- The target is a final uncalibrated score margin; probability calibration is
  separate. A zero margin is a tie, not evidence for either candidate.
- Fixed-slate intervention states must have identical candidate identities,
  order, masks, epsilon, and fusion weight. Membership edits require a separate
  future estimand. Never silently discard a candidate that becomes non-finite.
- Source edits rebuild every dependent representation. Message/edge masks test
  computational routes and use a separate experiment; their effects must not be
  added to source-level effects as independent evidence.

## Milestones and acceptance gates

| Phase | Deliverable | Acceptance gate | Status |
| --- | --- | --- | --- |
| 1 | Pure CPU full-slate normalization, branch margins, finite-change direct/normalization routes, candidate allocation | Synthetic agreement with production float32/float64 fusion; third-candidate reversal; masks, ties, floor crossings; antisymmetry and comparison consistency; explicit numerical residuals | Implemented; synthetic verification |
| 2 | Shared-coalition grouped attribution over supplied full-slate scorer | Small exact-game oracle; seeded permutation reproducibility; shared calls across comparisons; bounded query count and cache; reject mismatched slates; report baseline and sampling error separately | Implemented; synthetic verification |
| 3 | Immutable frozen-model adapter and source intervention builder | Checkpoint/config/preprocessing/candidate provenance; reproduce authoritative scores and catalog tie rule at recorded precision; update all descendants; freeze global train-fit graph; no protected rows in output/logs | Planned |
| 4 | Whole-ranking replay and bounded rank-reversal search | Replay retained/deleted evidence on original scorer; preserve all adjacent margins for full strict order and min-inside/max-outside for top-k membership; verify every reversal; certify minimum cost only for exhaustive small spaces | Planned |
| 5 | Explanation benchmark and novelty assessment | Matched references, interventions, budgets and sparsity; strong grouped Shapley, ranking and compositional baselines; patient-separated selection/evaluation; aggregate uncertainty and failure coverage | Planned |
| 6 | Minimal result representation and presentation | Design only after computation/replay is stable; separate model dependence, actual rules and external support; deterministic text first; no unsupported clinical claims | Planned |

The current task completes phases 1–2 on synthetic examples. Subsequent phases
build on their tested interfaces; these phases do not authorize extraction,
training, GPU jobs, new clinical rules, or another one-shot ranking gate.

## Numerical and algorithmic design

Use existing NumPy and the Python standard library; add no dependencies. The
core is torch-independent and does no file, network, model, or dataset access.
Its float64 results are a numerical reference, not bitwise reproduction of
float32/BF16 inference. The clinical adapter must check original-precision
parity before explaining a stored ranking, especially near ties.

For branch weight `w`, raw gap `d`, and reciprocal scale `r`, allocate a finite
change using the symmetric product identity:

\[
\Delta(wdr)=w\bar r\,\Delta d+w\bar d\,\Delta r.
\]

Allocate the variance change by candidate as in the proposal. Transport it
through the reciprocal-scale divided difference, using a stable expression
above the variance floor and a documented zero slope at the floor when both
endpoints are floored. Retain the residual; never force completeness by
rescaling contributions. Large opposing route terms near a floor can magnify
roundoff and must be visible in the numerical residual.

The attribution runner takes a deterministic callback from a retained group
set to a complete score slate. It does not invent missing clinical features or
build a graph. Empty/full coalitions define the one-reference baseline and
observed state. Uniform seeded permutations estimate ordinary grouped Shapley
totals; exact enumeration is limited to small games. Every comparison uses the
same coalitions. Cache entries, evaluation count, and exact enumeration are
bounded. Route averages are computational allocations, not independently
identified causal effects. Sampling standard errors measure permutation
variation only; they do not describe clinical validity or baseline uncertainty.

## Data safety and scope

Phase 1–2 fixtures are invented numerical scores. No protected artifacts are
loaded. Runtime paths for later phases must use `pipeline/config.py` and
`PROJECT_HOME`, `DATASET_ROOT`, `REPORTS_ROOT`, `WORK_SCRATCH`. Patient-level
scores and explanations remain restricted derived data even without names.
Avoid score/identifier values in validation errors and automatic object reprs.
Callbacks must also avoid sensitive output; the runner cannot police their logs.

No cohort, label, split, medication mapping, scorer, model weight, or calibrated
probability changes belong to this milestone. Preserve the prior ledger and
contract-test deletion and unrelated worktree edits.

## Impacted files and verification

- New numerical and attribution modules under `pipeline/explainability/`.
- Focused synthetic tests under `tests/test_explainability_*.py`.
- Proposal/status, README, architecture, roadmap, testing/workflow and changelog
  updates distinguishing numerical implementation from clinical validation.

Run focused pytest, repository Ruff lint, formatting checks for added Python,
TOML parsing, and diff review on the Calculco login node with `uv`. Optional
torch parity tests skip explicitly when the neural dependency is unavailable.
Do not install a GPU framework solely for these tests.

### Minimal synthetic use

This example contains invented scores. It does not load or explain a patient.
Run it with `uv run --no-sync python` from the project root.

```python
from pipeline.explainability.margins import ScoreSlate, decompose_transition
from pipeline.explainability.attribution import attribute_groups

def synthetic_scorer(retained: frozenset[str]) -> ScoreSlate:
    third_score = 100.0 if "third_score_edit" in retained else 0.0
    return ScoreSlate(
        candidate_ids=("A", "B", "C"),
        transformer=(2.0, 1.0, third_score),
        gnn=(0.0, 2.0, 1.0),
        gnn_weight=0.25,
    )

before = synthetic_scorer(frozenset())
after = synthetic_scorer(frozenset({"third_score_edit"}))
routes = decompose_transition(before, after, "A", "B")
result = attribute_groups(
    synthetic_scorer,
    ("third_score_edit",),
    (("A", "B"), ("A", "C")),
    n_permutations=None,
    max_evaluations=2,
)
assert before.margin("A", "B").total > 0
assert after.margin("A", "B").total < 0
assert routes.transformer.direct == 0
assert result.evaluations == 2
```

This is a score-level computational probe, not a clinical source intervention.
The complete clinical adapter must supply a provenance-checked scorer and
reference policy; arbitrary raw-score edits do not establish clinical validity.

## Risks and rollback

Main risks: inconsistent source interventions, mismatched candidate sets,
near-tie precision, cancellation near constant slates, reference sensitivity,
unbounded callback costs, and claims exceeding synthetic evidence. Query limits
bound calls rather than their individual runtime. Later adapters need their
own graph/batch limits and provenance. Remove only this milestone's new modules,
tests and documentation edits to roll it back; it creates no dataset artifacts
and does not alter trained models.

## Results

- Phases 1–2 implemented in `margins.py` and `attribution.py`; no dependencies
  added and no protected inputs or model artifacts accessed.
- `uv run --offline --no-sync pytest -q tests/test_explainability_margins.py
  tests/test_explainability_attribution.py`: **68 passed**, including the
  independent subset-Shapley oracle and actual production fusion in float32
  and float64. Synthetic verification does not constitute the proposed clinical
  or comparative experiment.
- Repository Ruff lint and TOML parsing pass. Added Python files pass formatting.
  Whole-repository format checking identifies one pre-existing issue in
  `visualization/hybrid_architecture_diagrams.py`, which this task does not edit.
- Data safety/reproducibility review: in-memory invented scores only, no
  persisted patient-level output, no retraining or gate consumption, fixed
  references and seeded permutations, hard query/cache limits, and residuals
  retained. Clinical-adapter validity remains the phase-3 gate.
- Next milestone: phase 3, beginning with checkpoint/scorer provenance and exact
  preprocessing/graph dependency mapping before any protected evaluation.
