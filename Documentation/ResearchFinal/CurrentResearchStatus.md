# Research paper status (ranking evidence and next XAI work)

Ranking review: 2026-09-07. Explainability implementation status updated after
the initial numerical-core milestone; see the implementation plan below.

This note records what can be claimed from the current tree, after the
paired-OOF late-fusion protocol and its post-hoc calibration stage
completed. It is an internal planning document, not a manuscript.

Canonical ranking sources:

- Protocol: [PairedOOFLateFusionProtocol.md](PairedOOFLateFusionProtocol.md)
  (`phase8-p0-paired-oof-late-fusion-v2`)
- Aggregate metrics: [HybridModelResultReview.md](HybridModelResultReview.md)
- Gate report: `reports/phase8_p0_paired_oof_late_fusion_gate_score.json`
- Calibration report: `reports/phase8_p0_paired_oof_calibration.json`
  (`phase8-p0-paired-oof-calibration-v1`, OAR `22662`)
- XAI design and bounded implementation:
  [ProposedExplainabilityIdea.md](ProposedExplainabilityIdea.md) and
  [ExplainabilityImplementationPlan.md](ExplainabilityImplementationPlan.md)
- Public status: [README.md](../README.md), [ARCHITECTURE.md](../ARCHITECTURE.md)

Observed prescriptions remain historical labels, not optimal treatment. Ranking
promotion is an offline research decision. It does not authorize clinical use.

## Locked ranking result

Protocol `phase8-p0-paired-oof-late-fusion-v2` finished protected OOF,
selection, refit, and the one-shot MIMIC-test gate
`mimiciv-test-paired-oof-v1`. Selected pair: GNN variant `full`,
\(\alpha = 0.25\). Gate decision: `qualified=true`,
`final_scoring_authorized=true`.

| Setting | Reference | \(\Delta\) NDCG@10 | \(\Delta\) MRR@10 | \(\Delta\) Hit@10 | Role |
|---------|-----------|-------------------:|------------------:|------------------:|------|
| Asymmetric hybrid (validation) | Frozen Transformer | +0.00484 | +0.0039 | +0.0048 | Historical negative control; failed +0.005 |
| Paired OOF (train selection) | Best single branch (Transformer) | +0.0092 | +0.0080 | +0.0072 | Leakage-safe selection only |
| Paired gate (MIMIC test) | Frozen Transformer | +0.0131 | +0.0122 | +0.0116 | Passed; hybrid promoted for ranking |

Selected blend on test: Transformer NDCG@10 0.4119 → hybrid 0.4250. GNN
alone does not beat the Transformer on train OOF (best GNN 0.4068 vs
Transformer 0.4137). The `rank_only` partner selected \(\alpha = 0.00\);
`full` selected \(\alpha = 0.25\).

Ranking promotion is not probability calibration. The ranking-gate Brier
(hybrid 0.262 vs Transformer 0.210) and 10-bin ECE (0.420 vs 0.337) are
**uncalibrated** score disclosures on the same test rows: they describe raw
rank scores treated as probabilities, not the frozen mapping. Post-hoc
calibration (`phase8-p0-paired-oof-calibration-v1`) completed on OAR `22662`.
Both branches froze affine sigmoid on paired train OOF. Cross-fitted Brier
≈ 0.069 and 10-bin ECE ≈ 0.001 (hybrid 0.0688 / 0.0014; Transformer 0.0687 /
0.0013). Ranking `rank_score` stays frozen. Applying that mapping to the
consumed MIMIC-test gate would be retrospective only.

Disclosure: the frozen Transformer had already been scored on MIMIC test in
the broader program. The paired-fusion score is new. A new patient or
temporal holdout would isolate the gate more strongly.

## What is actually in the repository

| Layer | Status | Paper use |
|---|---|---|
| MIMIC-IV / eICU data foundation, 24h feature cutoff, (24h, 48h] labels, patient splits | Implemented | Case-study protocol |
| Popularity, linear, XGBoost baselines | Implemented; Stage 1 winner is `xgboost_rank_ndcg_oof_late_fusion` (val NDCG@10 ≈ 0.395) | Required controls |
| Transformer patient/context ranker | Implemented and trained | Sequential view |
| Patient-subgraph R-GCN, 6 pre-registered ablations | Implemented; standalone GNN beat graph-only XGBoost (0.4015 vs 0.2985 NDCG@10) | Relational view |
| Late fusion `(1-α) z(T) + α z(G)` | Implemented | Fusion operator |
| Asymmetric fusion (GNN OOF + full-train Transformer) | Completed; +0.00484 NDCG@10, failed the +0.005 gate | Negative control |
| Paired-OOF fusion protocol | Completed; `full`, \(\alpha=0.25\); test gate passed (+0.0131 NDCG@10) | Measured ranking identification result |
| Paired-OOF probability calibration | Completed (OAR `22662`); both branches affine sigmoid; train-OOF Brier ≈ 0.069, ECE ≈ 0.001 | Separate from ranking; test Brier/ECE remain uncalibrated disclosures |
| Fused-rank explanation method | Numerical margins/routes and bounded grouped attribution implemented with synthetic tests; no former ledger schema | Clinical integration, replay, reversal search and comparative evaluation remain planned |
| Clinical source attribution, GNN paths, rules, LLM verbalization | Not implemented; score-level numerical attribution is separate | Cannot claim clinical explanation capability |
| Conversational understanding | Not implemented | Poster, not a result |
| eICU ranking performance | Coverage-only | No external-validity claim |

The Transformer already includes Stage-1 graph **tabular** summaries. Fusion
tests whether **message passing** adds ranking signal on top of those
summaries.

## Professor feedback and paper object

The clinical product framing in [ResearchDetail.md](ResearchDetail.md)
(“explainable conversational medication recommender”) is still an engineering
system description. A research paper needs a domain-independent question.
Medicine is the experimental domain.

Two defensible objects now sit in the repo:

1. **Ranking identification (measured).** Hybrid sequential–relational lift is
   not identified unless graph construction, preprocessing, and the fusion
   weight are estimated out of fold. The asymmetric protocol failed the
   pre-registered bar; the paired protocol passed it on MIMIC test.
2. **Faithful explanation of that fused ranker (initial numerics implemented).** The proposed
   method is a local, post-hoc, model-specific, contrastive explanation of the
   final fused ranking, separating branch raw-gap effects from slate-normalisation
   effects and verifying rank-reversal counterfactuals. The numerical core and
   shared-intervention attribution runner have synthetic tests. Source/clinical
   adapters, rank-reversal search and explanatory validation remain planned.

Use the locked hybrid as the **model under explanation**, not as a claim that
the conversational CDS system exists.

## Why “medical recommender” is engineering here

Medication combination on MIMIC is a crowded line: GAMENet (AAAI 2019),
SafeDrug (IJCAI 2021), MICRON, COGNet, MoleRec, 4SDrug, MGRN (2024), EDRMM
(2025), DrugDoctor, GraphCare (ICLR 2024), KEHGCN (AAAI 2026), ExpDrug
(2025), GraphDiffMed (2026), IngMamba (2026), SafeRx-Agent (2026). Almost all
of them predict a **visit-level drug set** and report Jaccard / F1 / PRAUC /
DDI rate. They do not solve the temporally cut candidate-ranking task used
here, but they occupy the “we recommend medications with graphs and
sequences” niche.

Sequence+graph hybrids are also known on the RecSys side: MCLSR, MRGSRec,
GSAU, LOOM, and RG-Hybrid (2026), which trains a language encoder and a graph
encoder independently and averages logits. Graph leakage during message
passing is a known RecSys issue (STAR-GCN, EFLEC). OOF stacking to stop
label leakage in graph models is also known (bagged stacking on graphs,
2022).

A new hybrid **architecture** is not a contribution. Two things can be:

- the **identification procedure** (now measured on MIMIC, still needs a
  second domain and statistical tests);
- the **fused-rank explanation contract** (specified, not evaluated).

The GNN is useful against graph-only XGBoost and is not stronger than the
Transformer alone. The event that matters for identification is that
**asymmetric stacking almost invented a hybrid, then the promotion rule
rejected it**, and **paired OOF selected a heavier GNN weight (\(\alpha=0.25\)
vs \(\approx 0.1\)) and passed the test gate**.

## Generic problems sitting inside the current code

These are domain-independent. Medicine is where the project has scale, weak
labels, and a typed graph.

1. **Complementary-view ranking under shared entities.** One view is a
   sequence/context encoder. The other is a heterogeneous graph over the same
   entities. The question is when fusion is complementarity rather than
   double-counting.

2. **Transductive graph leakage in ranking.** If coprescription, support
   counts, vocabularies, and edge weights are fit on all training patients, a
   held-out patient’s neighborhood still contains training labels of related
   entities. Fold-excluded graph caches are the operational answer.

3. **Asymmetric stacking leakage.** Fitting \(\alpha\) on GNN OOF plus a
   full-train Transformer covariate is not hybrid OOF. Paired OOF is the
   correction, and the two protocols disagreed on promotion.

4. **Weak observational ranking.** Observed actions are positives; unobserved
   candidates are not true negatives; popularity and future events leak. The
   24h/48h contract is a careful instance of ranking-under-logging.

5. **Faithful explanation of a fused ranker with hard constraints.**
   RankingSHAP / RankSHAP / ShaRP explain listwise tabular rankers.
   GNNExplainer-style methods explain graphs. Almost nobody explains the
   **final fused pairwise margin**, with hard exclusions kept out of the
   additive decomposition. That design is in
   [ProposedExplainabilityIdea.md](ProposedExplainabilityIdea.md). Initial
   numerical attribution exists; clinical explanation remains unvalidated.

## Engineering vs scientific contribution

**Engineering (appendix / reproducibility):** DuckDB pipeline, harmonization,
Calculco jobs, candidate catalogs, temperature calibration, checkpoint
hashing, fail-closed CLI gates, the clinical UI story.

**Scientific (ranking, now partly measured):** the estimand for hybrid lift
under entity-excluded graphs and paired OOF; the finding that a 0.005-scale
fusion decision flipped with that protocol; the six GNN relation ablations;
the ranking-with-temporal-cutoff task as a case study.

**Scientific (XAI, not yet measured):** the proposed fused-rank explanation
method, which separates branch raw-gap and slate-normalisation effects and
verifies rank-reversal interventions on the promoted hybrid.

**Not a contribution unless implemented and evaluated:** LIME+KG+rules+LLM
assembly, conversational CRS, clinician trust. The XAI note already treated
that assembly as a straightforward combination.

## Research questions and hypotheses

**RQ1 (ranking identification; MIMIC evidence exists).** Does paired
out-of-fold fusion of a sequential ranker and an entity-excluded relational
ranker identify a hybrid ranking gain that asymmetric stacking misestimates?

- **H1a (supported on MIMIC, pending stats and a second domain).** Under
  paired OOF, fused NDCG@10 exceeded the better single view by +0.0092 on
  train OOF and +0.0131 on the MIMIC-test gate, with MRR@10 and Hit@10 also
  improved.
- **H1b (supported as a protocol contrast, not a causal proof).** Asymmetric
  stacking selected \(\alpha \approx 0.1\) and failed validation promotion
  (+0.00484). Paired OOF selected \(\alpha = 0.25\) and passed the test gate.
  The Transformer was not wholly unseen on that test split.

**RQ2 (complementarity; supported as an ablation, not as architecture
novelty).** Is the graph view complementary, or a second copy of catalog
rank / coprescription frequency?

- **H2 (directionally supported).** `rank_only` selected \(\alpha = 0.00\)
  (Transformer only). `full` was selected at \(\alpha = 0.25\) and beat
  `no_message_passing` as a fusion partner on train OOF (0.4229 vs 0.4206
  NDCG@10). GNN alone still trails the Transformer.

**RQ3 (still required for a domain-independent ranking paper).** Does the
same protocol transfer to a non-clinical sequential+graph ranking dataset?

- **H3.** On that dataset, paired OOF fusion again beats the best single
  view, and transductive / in-sample stacking overstates or destabilizes
  \(\alpha\). **Not tested.**

**RQ4 (intended XAI contribution; not evaluated).** Can a
conservation-constrained pairwise evidence ledger reconstruct the fused
margin more faithfully than independent pointwise attributions?

- **H4.** At matched sparsity, ledger explanations have lower pairwise-margin
  reconstruction error and higher rank preservation/deletion fidelity than
  RankingSHAP/RankSHAP on the sequential branch plus GNNExplainer/PGExplainer
  on the graph, concatenated after the fact.

**RQ5 (later than RQ4).** Does clause-level checking of generated text
against the ledger cut unsupported claims relative to unconstrained LLM
rationales? Do not put this in a ranking-only paper, and do not claim it
before RQ4 exists.

## Mathematical formulation (ranking identification)

**Instances.** Ranking group \(g\) (stay + index condition, or user + query
in a public dataset). Candidate slate \(C_g\). Weak labels
\(y_{g,m}\in\{0,1\}\).

**Sequential view.** Events \(E_g\) before cutoff \(t_{\text{pred}}\), static
context \(x_g\):

\[
\mathbf{h}_g = f_\theta(x_g, E_g),\qquad
t_{g,m} = s_\theta(\mathbf{h}_g, c_m).
\]

Loss: multi-positive listwise softmax plus auxiliary BCE
(`pipeline.neural_training.losses`).

**Relational view.** Heterogeneous graph \(\mathcal{G}^{(-k)}\) fit
**without** fold-\(k\) entities. Patient (or user) subgraph
\(G_g\subset\mathcal{G}^{(-k)}\). R-GCN-style update (implemented grouped
operator):

\[
\mathbf{h}_v^{(l+1)}
=
\mathrm{LN}\Bigl(
\mathbf{h}_v^{(l)}
+
\sigma\Bigl(
\sum_{r\in\mathcal{R}}
\gamma_r
W_r
\sum_{u\in\mathcal{N}_r(v)}
w_{uv}\,
\mathbf{h}_u^{(l)}
\Bigr)
\Bigr),
\]

with \(\gamma_r=\sigma(\beta_r)\). Candidate score
\(g_{g,m}=q_\phi(\mathbf{h}_{\text{query}},\mathbf{h}_m,\mathbf{h}_{\text{ctx}})\).

**Fusion.** Within-group z-scores over finite, unmasked candidates:

\[
z(s)_{g,m}
=
\frac{s_{g,m}-\mu_g(s)}{\sigma_g(s)+\varepsilon},
\qquad
u_{g,m}
=
(1-\alpha)\,z(t)_{g,m}
+
\alpha\,z(g)_{g,m}.
\]

Locked selection: \(\alpha=0.25\), variant `full`.

**Estimand.** Let \(\mathcal{M}^*=\arg\max\{\mathrm{NDCG}@10(t),\,\mathrm{NDCG}@10(g)\}\).
Hybrid contribution is

\[
\Delta
=
\mathrm{NDCG}@10(u_{\hat\alpha})
-
\mathrm{NDCG}@10(\mathcal{M}^*),
\]

where \((\text{variant},\hat\alpha)\) is chosen on **paired train OOF only**,
then evaluated once on a frozen gate.

**Identification conditions:** patient/user-level splits; temporal cutoff;
train-only candidate catalog; fold-excluded graph statistics; no test edges
in message passing; \(\alpha\) not tuned on the gate split.

If hard safety rules are added, keep them **outside** \(u\) as a certificate
\(h_m\in\{0,1,\text{unknown}\}\). That belongs with RQ4.

## Experiments

**Case study A (done for ranking metrics, not for stats).** MIMIC-IV ICU
stay, condition-scoped candidate ranking, NDCG@10 primary, MRR@10 and Hit@10
guards. Labels = observed starts in (24h, 48h]. Imitation of historical
prescribing, not optimal therapy.

**Case study B (still missing for a generic ranking paper).** One public
sequential+graph ranking dataset (for example Amazon-Books or LastFM-17K
with item–item / KG edges). Same protocol: sequence encoder, relational
encoder, paired OOF fusion, transductive vs entity-excluded graphs.

**Baselines**

- Sequential only (Transformer; SASRec/BERT4Rec on dataset B)
- Graph only (GNN; LightGCN or R-GCN on dataset B)
- XGBoost / LambdaMART on tabular features
- Equal-weight late fusion (RG-Hybrid style, \(\alpha = 0.5\))
- In-sample stacking (full-train sequential + GNN OOF) — the failed
  asymmetric protocol, already run on MIMIC validation
- Transductive graph vs fold-excluded
- GAMENet/SafeDrug only if their visit-level set task is reimplemented.
  Do not compare Jaccard on their task to NDCG on this one.

**Ablations (already scored on train OOF):** `full`, `rank_only`,
`no_message_passing`, `no_condition_medication`, `no_dense_lab_vital`,
`no_lab_vital_intervention`. Late vs residual fusion was compared on the
older asymmetric run (late won).

**Metrics:** NDCG@10, MRR@10, Hit@10; train-OOF calibrated Brier/ECE as a
separate probability axis (ranking-gate Brier/ECE disclose uncalibrated
scores); cold-start slice; \(\alpha\) selected vs \(\alpha = 0/0.5/1\).

**Statistics still missing:** patient-clustered bootstrap CIs on \(\Delta\);
paired test on per-group NDCG. The +0.005 rule is a pre-registered **decision
policy**, not a theorem. Report the interval as well. Extra seeds are
stability analyses, not fishing.

**What not to claim:** clinical usefulness, eICU SOTA, conversational
quality, explanation faithfulness, or that the ranking-gate Brier/ECE are
calibrated probabilities (those figures disclose uncalibrated scores; the
train-OOF affine-sigmoid fit is a separate probability layer).

## How medicine stays in the paper without becoming the paper

Write the method with users, items, entities, views, graphs, and OOF.

Section 5.1: MIMIC-IV is experimental domain 1, weakly labeled, temporally
cut ranking over a heterogeneous clinical graph.

Section 5.2 (if RQ3 is in scope): public RecSys dataset, same method.

If the manuscript is the XAI paper, the hybrid is a frozen system under
explanation. MIMIC ranking numbers support that the object being explained is
the promoted fused ranker, not a Transformer-only model and not the failed
asymmetric blend.

## What is still missing

**Done (do not list as pending):** paired-OOF recovery, freeze of
(`full`, 0.25), one-shot MIMIC-test ranking gate, protected paired-OOF
calibration fit (OAR `22662`; both branches affine sigmoid).

**Still needed if the paper is ranking identification:**

1. Patient-clustered bootstrap CIs and a paired test on NDCG@10 deltas.
2. Explicit comparison table that also includes \(\alpha = 0.5\) and a
   transductive-graph control (asymmetric vs paired is already available).
3. One non-clinical sequential+graph dataset (RQ3).
4. Rewrite [ResearchDetail.md](ResearchDetail.md) so the repo question
   matches the manuscript.

**If the paper reports probabilities:** use the frozen affine-sigmoid
mapping (`probability_observed_prescription`) and keep ranking claims on
`rank_score`. Do not cite the ranking-gate Brier/ECE as calibrated. A
retrospective test application of that mapping is optional and is not a
new confirmatory split.

**Should have:**

5. Cold-start and low-support slices.
6. eICU performance only if mapping produces in-catalog positives.
7. Runtime/cost of fold-excluded graphs, as a limitation.
8. A stronger unseen gate (new patients or a later time cut) if reviewers
   reject reuse of MIMIC test for the Transformer.

**Needed before an XAI paper (current design/implementation target):**

9. Grouped IG/occlusion on the Transformer branch.
10. Contrastive subgraph masks with necessity/sufficiency on the GNN.
11. Exact fusion decomposition on a fixed slate; conservation residual.
12. RankingSHAP/RankSHAP and GNNExplainer/PGExplainer baselines.
13. Optional small expert study if the venue is clinical.

**Not needed for either paper yet:** LLM conversation, production LIME,
clinician deployment, DDI-aware decoding, notes.

## Papers to compare against, and venues

**Protocol / fusion / leakage:** RG-Hybrid (2026); LOOM / MCLSR
sequence–graph hybrids; STAR-GCN; EFLEC (SIGIR 2022); bagged graph stacking
(2022); LightGCN.

**EHR graphs / patient subgraphs:** GraphCare (ICLR 2024); R-GCN
(Schlichtkrull et al.).

**Medication rec, related work with a different estimand:** GAMENet,
SafeDrug, MICRON, MoleRec, COGNet, MGRN, KEHGCN, DrugDoctor.

**XAI:** RankingSHAP (SIGIR 2025), RankSHAP, ShaRP, GNNExplainer,
PGExplainer, KAPER.

**Venues for a ranking-identification paper:** ACM RecSys, CIKM, WSDM, The
Web Conference (short), ACM TOIS. KDD only if dataset B and the statistical
tests are clean.

**Venues for the XAI paper:** SIGIR, FAccT, TOIS, or JAMIA after explainers
and a small expert study exist.

## Paper structure (professor outline)

1. **Title / abstract.** Name the scientific object. Ranking paper example:
   *Leakage-safe fusion of sequential and relational rankers.* State H1, the
   asymmetric miss (+0.00484), and the paired-OOF test pass (+0.0131). XAI
   paper example: *Contrastive explanations for fused sequential–relational
   rankers.*
2. **Introduction.** Complementary views; shared-entity graph leakage;
   stacking leakage; EHR ranking as the stress test. For XAI, add that
   component-wise attributions do not explain the final fused margin.
3. **Related work.** Sequence–graph hybrids; GNN leakage; stacking;
   medication rec as a crowded application; ranking XAI and GNN explainers.
4. **Method.** Views, fold-excluded graphs, z-score fusion, selection rule.
   For XAI, the proposed route decomposition, intervention operators, and
   hard/soft certificate boundary.
5. **Experiments.** MIMIC ranking table (locked); optional public dataset;
   ablations; statistical tests. For XAI, fidelity/deletion/conservation and
   baselines on the frozen hybrid.
6. **Conclusion.** What was identified; labels are logged behavior; pointer
   to explanation work or remaining faithfulness limits.
7. **References.**

Put Calculco, DuckDB, and clinician-facing diagrams in a reproducibility
appendix or a separate technical report.

## Practical sequence

1. Treat paired-OOF ranking promotion as locked. Do not rerun the one-shot
   gate.
2. Agree with the supervisor whether the next manuscript is ranking
   identification (needs RQ3 + CIs) or the proposed explanation method (needs
   an implementation and evaluation on the frozen hybrid).
3. If ranking: add dataset B and bootstrap CIs before drafting related work.
4. If XAI: implement Stage 0–2 of
   [XAIInterpretabilityResearch.md](XAIInterpretabilityResearch.md) against
   the promoted paired fusion (`full`, \(\alpha=0.25\)), not the failed
   asymmetric blend.
5. Rewrite [ResearchDetail.md](ResearchDetail.md) after that choice so the
   public research question matches the paper.
6. If the paper reports probabilities, use the frozen train-OOF affine-sigmoid
   mapping and keep it out of the ranking claim. The ranking-gate Brier/ECE
   stay uncalibrated disclosures.

**Legitimate ranking claim (MIMIC, with the test-split disclosure):** hybrid
sequential–relational ranking gains on this task were not identified by
asymmetric stacking and were identified by paired, entity-excluded OOF
fusion (`full`, \(\alpha=0.25\), +0.0131 NDCG@10 on the protocol gate).

**Legitimate XAI quality/novelty claim (not yet available):** none. Initial
numerical implementation and synthetic verification do not establish clinical
faithfulness or superiority. See the implementation plan for remaining gates.
