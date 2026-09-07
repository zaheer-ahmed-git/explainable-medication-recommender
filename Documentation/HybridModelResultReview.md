Here is a consolidated picture of **all model outputs** and **how much late fusion improved**, from the aggregate reports only.

---

## The three model families

All solve the same task: rank candidate medications per patient stay + index condition. Primary metric is **NDCG@10**; **MRR@10** and **Hit@10** are guard metrics.

| Model | What it is | Main output |
|-------|------------|-------------|
| **Frozen Transformer** | Patient/context branch (event sequence + stay features) | Per-candidate logits |
| **Standalone GNN** | Graph branch (patient subgraph, 6 ablation variants) | Per-candidate logits |
| **Paired late fusion** | `(1−α)·z(Transformer) + α·z(GNN)` on z-scored logits | Blended ranking scores |

**Selected paired fusion:** GNN variant **`full`**, **α = 0.25** (25% GNN, 75% Transformer in z-score space).

---

## 1. Train OOF (paired selection — leakage-safe)

This is where variant and α were chosen. All metrics on **MIMIC train**, out-of-fold.

### Standalone branches

| Model | NDCG@10 | MRR@10 | Hit@10 |
|-------|--------:|-------:|-------:|
| **Transformer OOF** | 0.4137 | 0.5237 | 0.8846 |
| **Best standalone GNN OOF** (`no_dense_lab_vital`) | 0.4068 | 0.5132 | 0.8775 |
| **GNN `full` OOF alone** | 0.4059 | 0.5130 | 0.8765 |

On train OOF, **Transformer beats every standalone GNN**.

### All six GNN variants at their best α (OOF)

| GNN variant | Best α | NDCG@10 |
|-------------|-------:|--------:|
| **`full`** (selected) | 0.25 | **0.4229** |
| `no_dense_lab_vital` | 0.25 | 0.4221 |
| `no_condition_medication` | 0.25 | 0.4218 |
| `no_lab_vital_intervention` | 0.25 | 0.4219 |
| `no_message_passing` | 0.25 | 0.4206 |
| `rank_only` | 0.00 | 0.4137 (= Transformer only) |

### Paired late fusion lift on train OOF

| Comparison | NDCG@10 | Δ NDCG@10 |
|------------|--------:|----------:|
| Best single branch (Transformer) | 0.4137 | — |
| **Selected fusion (`full`, α=0.25)** | **0.4229** | **+0.0092** |
| | MRR@10 | Δ MRR@10 |
| Transformer | 0.5237 | — |
| **Fusion** | **0.5317** | **+0.0080** |
| | Hit@10 | Δ Hit@10 |
| Transformer | 0.8846 | — |
| **Fusion** | **0.8918** | **+0.0072** |

**Train OOF lift: +0.0092 NDCG@10** (~**+2.2% relative**). That is almost **2×** the +0.005 material threshold.

---

## 2. MIMIC test gate (paired protocol v2 — one-shot score)

This is the **new** frozen test gate (`mimiciv-test-paired-oof-v1`). Same candidates for hybrid and Transformer.

| Model | NDCG@10 | MRR@10 | Hit@10 |
|-------|--------:|-------:|-------:|
| **Frozen Transformer** | 0.4119 | 0.5157 | 0.8766 |
| **Paired late fusion** | **0.4250** | **0.5279** | **0.8882** |
| **Lift (fusion − Transformer)** | **+0.0131** | **+0.0122** | **+0.0116** |

**Promotion gate: PASSED** (`qualified: true`)

- Required NDCG lift: ≥ +0.005 → got **+0.0131**
- MRR / Hit guards: drop ≤ −0.01 → both **improved**

So on test, paired late fusion is **clearly better** than the frozen Transformer on all three ranking metrics.

### Probability calibration on the same test rows

Ranking quality and probability calibration are different properties. The gate
report also records row-level aggregates on the same MIMIC-test candidates:

| Model | Brier score (↓ better) | 10-bin ECE (↓ better) | ROC-AUC | Average precision |
|-------|-----------------------:|----------------------:|--------:|------------------:|
| Frozen Transformer | **0.210** | **0.337** | 0.743 | **0.215** |
| Paired late fusion | 0.262 | 0.420 | **0.744** | 0.210 |

“Worse probability calibration” means: if you treat the model’s score as an
estimated probability that a candidate was prescribed, the hybrid’s numbers are
**farther from the observed 0/1 labels** than the Transformer’s (higher Brier
and ECE). That can happen even when **rank order** improves, because late
fusion blends z-scored logits for ranking and does not re-calibrate the fused
scores to probabilities.

**Disclosure:** Transformer was scored on MIMIC test before in the broader program; this gate is not wholly unseen for the Transformer, but the **paired fusion** score is new and passed the protocol gate.

---

## 3. Older development scores (validation — asymmetric protocol)

For context vs the **previous** late-fusion attempt (Transformer not OOF; α ≈ 0.1):

| Model | Split | NDCG@10 | MRR@10 | Hit@10 |
|-------|-------|--------:|-------:|-------:|
| Transformer | validation | 0.4129 | 0.5261 | 0.8794 |
| Standalone GNN (`no_dense_lab_vital`) | validation | 0.4015 | 0.5092 | 0.8719 |
| **Old asymmetric hybrid** | validation | 0.4177 | 0.5300 | 0.8842 |
| Old hybrid lift vs Transformer | validation | **+0.0048** | +0.0039 | +0.0048 |

The **old hybrid failed promotion** (+0.0048 &lt; +0.005 threshold) and was **not** authorized for final scoring.

---

## How much did late fusion increase? (summary)

| Setting | vs reference | Δ NDCG@10 | Δ MRR@10 | Δ Hit@10 | Gate |
|---------|--------------|----------:|---------:|---------:|------|
| **Paired OOF (train)** | vs best base (Transformer) | **+0.0092** | +0.0080 | +0.0072 | selection only |
| **Paired gate (test)** | vs frozen Transformer | **+0.0131** | +0.0122 | +0.0116 | **passed** |
| **Old hybrid (validation)** | vs Transformer | +0.0048 | +0.0039 | +0.0048 | **failed** |

In plain terms:

- **Paired late fusion adds about +0.9 to +1.3 percentage points of NDCG@10** over Transformer alone, depending on split.
- On **test**, that is roughly **+3.2% relative** NDCG improvement (0.4119 → 0.4250).
- The **symmetric OOF protocol** both **selects a stronger blend** (α=0.25 vs old α≈0.1) and **generalizes better** to the held-out gate than the old asymmetric fusion.

---

## What each model is “good for” now

| Artifact | Role |
|----------|------|
| **Frozen Transformer** | Context-branch baseline; gate reference |
| **Standalone GNN (`full` refit)** | Graph-branch evidence; explainability target |
| **Paired late fusion** | **Promoted** hybrid for protocol v2; test gate passed |

**Bottom line:** GNN alone does not beat Transformer on OOF or validation, but **blending 25% GNN + 75% Transformer** gives a consistent lift. The paired protocol turned a marginal old fusion (+0.0048) into a **material, gate-passing** improvement (+0.0131 on test).