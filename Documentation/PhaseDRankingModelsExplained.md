# Phase D Ranking Models Explained

This note explains the three protected ranking branches used in Phase D, why
**hybrid late fusion was not promoted**, what **standalone GNN qualification**
means, and how **late fusion** works.

It is a research/engineering description only. Observed prescriptions are
historical positives, not proof of optimal care. Aggregate metrics below come
from development (MIMIC validation) reports under `reports/`.

---

## 1. Shared task (all three models)

All three models solve the same ranking problem:

> For one **ranking group** = one patient stay + one index condition, rank a
> fixed slate of **candidate medications**.

| Concept | Meaning |
|---------|---------|
| Ranking group | One `(source, split, ranking_group_id)` row family |
| Index condition | Normalized condition token (e.g. CCS-style) that scopes the slate |
| Candidates | Medications from the locked catalog / subgraph candidate table |
| Label | `label_prescribed`: historically observed in that stay (weak observational positive) |
| Primary metric | NDCG@10 on MIMIC validation (patient-grouped); MRR@10 and Hit@10 as guards |

**Common outputs (scoring):** per-candidate logits or scores → ranked list →
aggregate ranking metrics. Restricted score tables stay on protected storage;
reports keep aggregates only.

---

## 2. The three model types

### 2.1 Frozen Transformer (`transformer_patient_context`)

**Role:** Stage 2 patient/context branch. Scores candidates from long-range
clinical context (static stay features + event sequence), without running the
relation GNN at inference time for this baseline.

**What it takes (inputs):**

- Per ranking group:
  - Stay / patient **static context** (train-fit vocabularies and numeric
    features from Phase 8 caches).
  - **Event sequence** before the decision cutoff (labs, vitals, meds, etc.,
    temporally bounded).
  - **Candidate set** for the index condition (ids, catalog rank, train-fit
    priors / candidate-side features).
- Frozen checkpoint + temperature calibration from neural training
  (`$DATASET_ROOT/processed/phase8_p0/neural/checkpoints/`).

**What it does:**

1. Encodes the patient/stay context with a Transformer.
2. Scores each candidate with a dual-path scorer (MLP path + context–candidate
   interaction), using candidate-side features.
3. Applies a fitted temperature for calibration-sensitive uses.

**What it outputs:**

- Per-candidate **logits / scores** for the group.
- Development validation ranking metrics (NDCG@10, MRR@10, Hit@10, …).
- In Phase D fusion, those logits are cached as a **frozen covariate** under
  the GNN frozen-Transformer cache (`candidate_logits` + `contexts`), not
  re-trained during GNN/fusion.

**Protected development reference (validation @10):**  
nDCG ≈ **0.4129**, MRR ≈ **0.5261**, Hit ≈ **0.8794**  
(from fusion score report comparison rows).

**Important:** This Transformer was trained earlier and is treated as
**immutable** in Phase D. Fusion must not backprop into it.

---

### 2.2 Standalone relation GNN (`gnn_relation_rgcn`)

**Role:** Phase D graph branch. Ranks candidates using a patient subgraph
(heterogeneous relations) plus a stay-query / candidate scorer. Selected
ablation on protected data: **`no_dense_lab_vital`**.

**What it takes (inputs):**

- **Patient subgraph** for the ranking group:
  - Nodes (concepts, stay-query, candidates, …) with typed roles/features.
  - Typed edges (conditions, meds, labs/vitals/interventions as configured by
    the ablation; dense lab–vital edges removed for `no_dense_lab_vital`).
  - Candidate rows with ranks and observed labels for training/eval.
- Fold-safe caches for selection (cross-fit graphs that exclude the held-out
  fold); full-train graph cache for final refit.
- Optional frozen Transformer **context vectors** as side features in the
  GNN architecture (context dim), without training the Transformer.

**Training path (protected):**

1. **Cross-fit:** 5 patient folds × ablation variants → fold checkpoints.
2. **`select-gnn`:** pick variant by weighted held-out nDCG@10; write OOF
   logits; fit OOF temperature.
3. **`refit-gnn`:** retrain selected variant on the full-train graph for a
   fixed epoch count (here: 3).
4. **`score-gnn`:** score MIMIC validation; gate vs graph-only XGBoost.

**What it outputs:**

- Per-candidate GNN logits.
- Refit checkpoint: `.../gnn/checkpoints/gnn_relation_branch.pt`
- Temperature: `gnn_temperature_calibration.json`
- Validation metrics and a **selection / qualification** report.

**Protected development (validation @10):**  
nDCG ≈ **0.4015**, MRR ≈ **0.5092**, Hit ≈ **0.8719**.

---

### 2.3 Hybrid late fusion (`hybrid_late_fusion`)

**Role:** Combine **already-trained** Transformer scores and **already-trained**
GNN scores into one ranking score **without** jointly training a new deep
fusion network for the promoted late path (residual fusion was the alternative
candidate; late won the internal choice).

**What it takes (inputs):**

- Frozen Transformer per-candidate logits (from cache).
- GNN per-candidate logits (refit model at validation; OOF GNN logits when
  fitting the fusion weight on train).
- Same candidate mask / ranking groups as the locked tables.

**What it outputs:**

- Fused per-candidate scores (see §4).
- Fusion checkpoint metadata (selected weight, temperature, model choice).
- Validation metrics and a **hybrid promotion gate** vs the Transformer.

**Protected development (validation @10):**  
nDCG ≈ **0.4177**, MRR ≈ **0.5300**, Hit ≈ **0.8842**  
Selected GNN weight α ≈ **0.1** (see §4).

Residual hybrid (not selected): nDCG ≈ **0.4063** — worse than late on the
pre-registered selection order.

---

## 3. What “standalone GNN qualified vs graph XGBoost” means

### 3.1 Who is compared?

| Name | Role in the gate |
|------|------------------|
| **Candidate** | Standalone GNN after refit (`gnn_relation_rgcn`) |
| **Reference** | Locked Milestone 8B **graph-only XGBoost** (`graph_only_xgboost`) |

This gate answers: *Does the learned relation GNN beat the strong graph-tabular
XGBoost baseline enough to be frozen as a standalone relation branch?*

It does **not** compare GNN to the Transformer. That is a different gate
(hybrid vs Transformer).

### 3.2 Gate rule (pre-registered)

From `qualification_decision` in `pipeline/gnn_training/scoring.py`:

1. **NDCG@10 lift:** candidate ≥ reference + **0.005**
2. **MRR@10:** candidate ≥ reference − **0.01** (non-inferiority)
3. **Hit@10:** candidate ≥ reference − **0.01**

### 3.3 What happened on protected validation

| Model | nDCG@10 | MRR@10 | Hit@10 |
|-------|--------:|-------:|-------:|
| GNN | **0.4015** | **0.5092** | **0.8719** |
| Graph-only XGBoost | 0.2985 | 0.3975 | 0.8045 |
| Δ (GNN − XGB) | **+0.103** | **+0.112** | **+0.067** |

Required nDCG was ≈ 0.3035. GNN cleared easily.

**Result:** `decision = qualify_standalone_gnn`,  
`standalone_qualified = true`,  
`final_scoring_authorized = true` for **standalone GNN**  
(`reports/phase8_p0_gnn_training_selection.json`).

**Meaning in plain language:** The relation GNN is allowed to be treated as a
**frozen standalone scorer** and may run the optional one-shot MIMIC **test**
scoring command. It is a justified improvement over graph-only XGBoost under
the project’s own gate.

---

## 4. Late fusion: definition and mechanics

### 4.1 Idea

**Late fusion** = fuse **final scores** from two frozen models, not their
hidden layers.

Both models already produced a score for every candidate in a group. Fusion
only mixes those scores with one scalar weight.

### 4.2 Formula (implemented)

In `pipeline/gnn_training/fusion.py` → `late_fusion_logits`:

1. Within each ranking group, z-score Transformer logits (masked, finite only).
2. Within each ranking group, z-score GNN logits the same way.
3. Combine:

\[
s_{\text{fused}} = (1 - \alpha)\, z(s_{\text{Transformer}}) + \alpha\, z(s_{\text{GNN}})
\]

where \(\alpha =\) `gnn_weight` ∈ `[0, 1]`.

Invalid / masked candidates get \(-\infty\) so they do not rank.

Z-scoring matters because raw Transformer and GNN logits live on different
scales; mixing without normalization would let one model dominate by accident.

### 4.3 How \(\alpha\) was chosen (protected run)

- Grid: \(\alpha \in \{0.00, 0.05, \ldots, 1.00\}\) (`DEFAULT_LATE_FUSION_WEIGHTS`).
- Fit on **train** using:
  - GNN **OOF** logits (fold-safe), and
  - frozen Transformer train logits from the full-train cache.
- Intersection policy: full-train / frozen caches exclude **zero-positive**
  train groups; OOF keeps the full PCM train set. Meta-fit uses the
  intersection (exact coverage against the frozen subset).
- Selection of \(\alpha\): best train nDCG@10 (then MRR, Hit; prefer smaller
  \(\alpha\) on ties).

**Selected:** \(\alpha \approx 0.1\) → fused score is **90% Transformer
z-score + 10% GNN z-score**.

That small weight matches the outcome: hybrid barely beats the Transformer,
and does not clear the +0.005 promotion bar.

### 4.4 Late vs residual (internal choice, before the Transformer gate)

| Candidate | Mechanism |
|-----------|-----------|
| **Late** | Weighted mix of two frozen logits (above). |
| **Residual** | Train a small head + GNN copy that **adds** a residual to frozen Transformer logits (zero-init so training starts equal to Transformer). |

On validation, late beat residual (0.4177 vs 0.4063 nDCG@10), so the fusion
checkpoint records `selected_model = late`.

---

## 5. Why hybrid promotion failed

### 5.1 Different gate, different reference

| Gate | Candidate | Reference | Purpose |
|------|-----------|-----------|---------|
| GNN standalone | GNN | Graph XGBoost | Is the GNN a useful **relation** branch? |
| Hybrid promotion | Late fusion | **Frozen Transformer** | Does fusion beat the **context** model enough to replace it? |

Hybrid is **not** judged against XGBoost. It must beat the Transformer.

### 5.2 Numbers

| Model | nDCG@10 | Required for hybrid (Transformer + 0.005) |
|-------|--------:|------------------------------------------:|
| Transformer | 0.4129 | — |
| Late fusion | **0.4177** | **≈ 0.4179** |
| Δ | **+0.00484** | need **≥ +0.005** |

MRR/Hit deltas were positive (fine). The **only** failure was nDCG lift
**0.00016 short** of the pre-registered +0.005 bar.

**Decision:** `retain_frozen_transformer`  
`hybrid_qualified = false`  
`final_scoring_authorized = false` for hybrid  
(`reports/phase8_p0_fusion_training_selection.json`).

### 5.3 What that does *not* mean

- It does **not** mean fusion is useless or that GNN adds nothing.
- Hybrid still improves Transformer slightly (+0.0048 nDCG@10).
- It **does** mean the project’s fail-closed rule refuses to **promote** hybrid
  as the primary gated scorer / authorize one-shot **final hybrid** test
  scoring.

Gates exist so tiny noisy lifts do not rewrite the frozen stack.

---

## 6. Why earlier advice said “Transformer or qualified GNN” (not promoted hybrid)

That advice was about **promotion and final-scoring policy**, not about
deleting hybrid from the research story.

| Artifact | Use it for |
|----------|------------|
| **Qualified standalone GNN** | Relation-branch claims; optional `final score-gnn`; graph attributions in explainability |
| **Frozen Transformer** | Context-branch claims; hybrid gate reference; token/context attributions |
| **Late fusion (not promoted)** | Ablation / analysis: “10% GNN z-score helps a little but misses +0.005”; evidence that fusion is near the bar |

You **can** still study hybrid in papers and explainability prototypes. You
should **not** call it the gated promoted hybrid or run
`final score-fusion` without a reviewed policy change, because
`final_scoring_authorized` is false.

For **Milestone 9 grounded explanation**, the natural primary targets are:

1. Explain **Transformer** rankings (context / feature evidence), and/or  
2. Explain **GNN** rankings (subgraph / relation evidence),  

and optionally show how late fusion would re-rank when \(\alpha > 0\), without
pretending hybrid cleared promotion.

---

## 7. End-to-end data flow (compact)

```text
Locked ranking table (PCM) + subgraphs + features
        │
        ├─► Transformer train/score ──► frozen logits/contexts
        │
        ├─► GNN cross-fit → select → refit → score vs graph XGB
        │         │                    │
        │         └─ OOF logits        └─ gnn_relation_branch.pt
        │                   │
        └───────────────────┴─► Late fusion (α on z-scores)
                                      │
                                      ├─ vs Transformer gate → not promoted
                                      └─ checkpoint kept for analysis
```

---

## 8. Inputs / outputs cheat sheet

| Model | Main inputs | Main outputs |
|-------|-------------|--------------|
| Transformer | Stay context, event sequence, candidate features | Per-candidate logits; validation/test score tables |
| Standalone GNN | Patient subgraph (nodes/edges/candidates), optional frozen context features | Per-candidate logits; fold/refit checkpoints; OOF parquet |
| Late fusion | Transformer logits + GNN logits (+ mask) | Fused scores; selected \(\alpha\); hybrid score table |

---

## 9. Report pointers (aggregates only)

| Report | What it records |
|--------|-----------------|
| `reports/phase8_p0_gnn_score_evaluation.json` | GNN vs graph XGB metrics |
| `reports/phase8_p0_gnn_training_selection.json` | GNN qualification / final auth |
| `reports/phase8_p0_fusion_training_evaluation.json` | Late vs residual; α; train meta-fit |
| `reports/phase8_p0_fusion_score_evaluation.json` | Hybrid vs Transformer metrics |
| `reports/phase8_p0_fusion_training_selection.json` | Hybrid gate decision |

Code entry points:

- Transformer: `pipeline/neural_training/`
- GNN: `pipeline/gnn_training/` (`train-gnn-fold`, `select-gnn`, `refit-gnn`, `score-gnn`)
- Fusion: `pipeline/gnn_training/fusion.py`, `train_fusion.py`, `score_fusion.py`

---

## 10. Clinical / research caveats

- Labels are **observed prescribing**, not gold-standard optimal therapy.
- Development metrics are **validation**, not the one-shot MIMIC test.
- eICU remains coverage-oriented until a separately reviewed mapping rebuild.
- This document does not authorize clinical use or prescribe treatments.
