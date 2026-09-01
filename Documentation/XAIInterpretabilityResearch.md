# Explainability and Interpretability for the Medication Recommender

**Architecture Design**

## My Decision

The strongest practical design is a layered explanation system in which each claim has a distinct evidential role:

1. **Transformer evidence:** clinically grouped, candidate-contrastive Integrated Gradients (IG), checked by event/time-window occlusion.
2. **GNN evidence:** a candidate-conditioned sparse subgraph selected for necessity and sufficiency, then rendered as typed, temporally valid paths.
3. **Safety evidence:** exact, versioned rule traces for allergies, contraindications, interactions, dose or organ-function constraints, including reasons that candidates were rejected.
4. **Knowledge evidence:** retrieved guideline or biomedical knowledge with source, version, date, and scope; this supports clinical interpretation but is not automatically evidence of what the learned model used.
5. **Uncertainty:** calibrated recommendation uncertainty and explanation stability/intervals, with an abstention or “insufficiently stable to explain” state.
6. **Language:** an LLM may verbalize a structured evidence object, Every sentence should be mechanically traceable to an evidence atom and checked after generation.

Close prior work already contains most of those ingredients in isolation or loose combination. *The defensible opportunity is an **end-to-end, constraint-aware rank-evidence ledger*** that explains the final pairwise medication ranking across temporal and graph branches, separates hard feasibility certificates from learned preference evidence, conserves the explained rank margin, quantifies explanation uncertainty, and constrains every natural-language claim to verified evidence.

This is a **potentially meaningful methodological contribution**, not yet a proven novelty claim. Pairwise/listwise Shapley ranking explanations, component attribution, graph-path explanations, safety-aware recommenders, and grounded LLM recommenders already exist. A paper would need to show that the proposed common decision target, intervention semantics, hard/soft constraint separation, conservation diagnostics, and end-to-end evaluation produce better faithfulness and clinical error detection than those components used separately.

## Scope, evidence standard, and terminology

Claims are separated as follows:

- **Established method:** mechanism or result supported by cited literature.
- **Literature finding:** empirical or review evidence reported by the cited authors.
- **Project inference:** a conclusion derived from the literature and the repository's architecture.
- **Proposed research:** a design that has not yet been implemented or validated here.

The active tree contains an experimental [Transformer event-sequence ranker](../pipeline/neural_training/model.py), [candidate-conditioned relational GNN](../pipeline/gnn_training/model.py), and [fusion implementations](../pipeline/gnn_training/fusion.py). A first, framework-independent [explanation evidence contract](../pipeline/explainability/contract.py) now implements part of Stage 0 below: it records a versioned pairwise decision, hierarchical signed evidence, conservation residual, hard safety certificates, external-knowledge support status, constrained model counterfactuals, and protected-reference declarations. It validates and serializes those records but does **not** yet compute Transformer/GNN explanations, log protected runs, generate clinical language, or establish clinical validity. This report therefore maps XAI to the code that exists without claiming a production recommender or validated clinical benefit.

Several terms are often conflated:

- **Interpretability** is the degree to which a person can understand a model or decision. An intrinsically interpretable model exposes meaningful internal variables or rules; a post-hoc explainer approximates or probes an already trained model.
- **Transparency** describes inspectability of data, transformations, model versions, rules, and provenance. It is necessary for auditability but does not itself establish faithfulness.
- **Faithfulness** asks whether the explanation tracks the computational factors that changed the model decision.
- **Fidelity** often means agreement between an explanatory surrogate and the black box. Because the literature uses fidelity and faithfulness inconsistently, this report defines every experimental metric operationally.
- **Plausibility/coherence** asks whether an explanation looks sensible to a person. A plausible explanation can be unfaithful; a faithful one can be clinically meaningless.
- **Clinical evidence** can justify why a relationship is medically credible without proving that the learned model used it. Conversely, a faithful attribution can reveal a model dependency that is clinically wrong.
- **Causal explanation** requires causal assumptions or interventions. Attribution and ordinary counterfactual model queries are not causal treatment effects.

The distinction between correctness and human-facing coherence is central to the Co-12 framework, which organizes explanation quality into twelve properties rather than treating “interpretability” as one number ([Nauta et al., 2023, DOI 10.1145/3583558](https://doi.org/10.1145/3583558)). In high-stakes domains, Rudin argues that interpretable models should be preferred when competitive, rather than automatically adding post-hoc explanations to black boxes ([Rudin, 2019, DOI 10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x)). For this project, however, the heterogeneous temporal, graph, and rule pipeline makes a hybrid of intrinsic evidence objects and validated post-hoc probes more realistic than a single globally transparent model.

## 1. XAI method families: mechanism, trade-offs, and project fit



### 1.1 Comparative summary


| Method family                             | Mechanism and model compatibility                                                                                                | Strengths                                                                                                       | Principal limitations                                                                                                                                                           | Typical explanation cost                                                             | Project verdict                                                                                                 |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| SHAP / KernelSHAP                         | Allocates the difference from a baseline expectation using Shapley values; model-agnostic sampling variants work with any scorer | Signed local contributions, additive completeness under the chosen game, global aggregation                     | Result depends on background distribution, masking semantics, and feature dependence; exact enumeration is exponential; a point-score explanation can miss ranking interactions | Exact general SHAP is exponential in feature groups; sampling needs many model calls | Useful as a baseline and for grouped inputs, but not on embedding dimensions and not as the sole rank explainer |
| TreeSHAP                                  | Computes Shapley-style attributions efficiently for tree ensembles                                                               | Fast and exact for the specified tree game; excellent reference-model explainer                                 | Explains the tree model, not a Transformer/GNN; conditional/interventional interpretation still matters                                                                         | Polynomial in trees/leaves/depth rather than feature coalitions                      | Use for tree baselines and audit comparisons                                                                    |
| LIME                                      | Samples a local neighbourhood, queries the black box, and fits a sparse interpretable surrogate                                  | Simple, model-agnostic, can expose local nonlinear behaviour approximately                                      | Neighbourhood and kernel are arbitrary; perturbations may be implausible; repeated runs can vary; surrogate fidelity is only local                                              | Approximately the sampled model calls plus a sparse regression                       | Retain only as a conventional baseline, not the production explanation                                          |
| Integrated Gradients                      | Integrates input gradients from a reference point to the input; applies to differentiable models                                 | Satisfies sensitivity and implementation invariance; contributions sum to output difference for the chosen path | Baseline and path can be clinically implausible; saturation/noise; token embeddings require grouping; correlation is not causation                                              | Roughly `m` forward/backward passes for `m` integration steps                        | Primary Transformer attribution, with a clinically valid baseline and occlusion tests                           |
| Raw gradients / saliency / DeepLIFT / LRP | Propagates local sensitivity or relevance through the network                                                                    | Cheap; fine-grained; useful for developer diagnostics                                                           | Noisy, scale-sensitive, may fail sanity/input-invariance tests; relevance is easy to overinterpret                                                                              | One or a few backward passes                                                         | Developer diagnostic only unless it passes model/data randomization and intervention checks                     |
| Attention analysis                        | Displays learned attention weights or aggregates them across heads/layers                                                        | Native to attention models; low incremental cost; can help inspect information routing                          | Attention need not correlate with output importance; alternative attention distributions may preserve predictions; layer mixing complicates interpretation                      | Usually one forward pass; rollout/flow adds modest graph operations                  | Never call raw attention a faithful explanation; use as a secondary diagnostic                                  |
| Permutation importance                    | Permutes a feature/group and measures performance loss                                                                           | Simple global model reliance; model-agnostic                                                                    | Breaks dependencies, creates out-of-distribution records, shares or hides importance among correlated features; not patient-local                                               | About one or more evaluation passes per feature/group                                | Use only at cohort level with grouped/conditional perturbations                                                 |
| Counterfactual explanations               | Searches for a small change that flips a prediction or ranking                                                                   | Contrastive and actionable in form; directly answers “what would change the result?”                            | Plausibility, immutability, temporal feasibility, and causal validity are hard; multiple counterfactuals exist; optimisation can be expensive                                   | Repeated inference/gradient or combinatorial search                                  | High-value later layer, but only with clinically constrained interventions and “model counterfactual” labelling |
| Prototypes / examples / case retrieval    | Returns representative training examples, learned prototypes, or similar prior cases                                             | Natural comparison; can expose coverage and unusual cases                                                       | Similarity may not be clinically meaningful; privacy and memorisation risks; historical treatment is observed, not necessarily optimal                                          | Index lookup is cheap; prototype learning and influence estimates can be costly      | Use only with validated similarity, access control, provenance, and explicit non-optimality wording             |
| Rules / Anchors / global surrogates       | Produces if–then rules or fits a simple model to approximate black-box decisions                                                 | Inspectable and auditable; exact safety rules can be intrinsically faithful to the rule engine                  | A surrogate can conceal errors outside its coverage; rule sets become large; learned anchors describe local precision, not causality                                            | Rule execution is cheap; discovery/surrogate training can be expensive               | Exact rule traces are mandatory for safety; learned rules are optional audit aids                               |
| Concept explanations / CBMs / TCAV        | Maps hidden representations to human concepts; a CBM predicts concepts before the outcome and can permit interventions           | Matches clinical vocabulary; supports concept-level auditing and correction                                     | Concept annotations are costly; bottleneck leakage and concept misalignment; interventions may not behave causally                                                              | Added concept model/training; inference usually modest                               | Promising intrinsic extension after concepts and labels are validated, not an immediate post-hoc fix            |
| Knowledge-graph paths                     | Retrieves typed paths between patient factors, conditions, drugs, and evidence                                                   | Human-readable relations and provenance; supports evidence retrieval                                            | A true path is not necessarily used by the GNN, clinically sufficient, or causal; path ranking can favour popular hubs                                                          | Graph traversal is manageable but candidate-path explosion can be large              | Use as clinical/provenance evidence only after linking paths to score interventions                             |
| GNN explainers                            | Optimises edge/node/feature masks, learns an amortised mask, or searches subgraphs                                               | Can expose relational structure that tabular attribution loses                                                  | Masking can create invalid graphs; different explainers disagree; per-instance search may be slow; subgraphs are not automatically paths or clinical rationales                 | From iterative mask optimisation to expensive Monte Carlo tree search                | Use candidate-conditioned mask methods, then validate necessity/sufficiency and render typed paths              |
| Causal explanations                       | Uses a structural causal model or interventional value function                                                                  | Separates intervention from observation; supports causal counterfactuals in principle                           | Requires defensible graph, identifiability, and assumptions rarely available in retrospective EHR data                                                                          | Model-specific; often repeated causal inference/simulation                           | Long-term research only; do not use causal language for ordinary feature attribution                            |
| Uncertainty-aware explanation             | Explains predictive entropy or adds intervals/stability estimates to explanations                                                | Shows when both decision and explanation are unreliable                                                         | Predictive uncertainty and explanation uncertainty are different; intervals inherit model/perturbation assumptions                                                              | Ensembles/bootstrap multiply inference; some wrappers are expensive                  | Required for deployment-quality explanations; begin with bootstrap/seed stability and calibration               |
| LLM-generated explanation                 | Converts evidence into fluent text, sometimes with retrieval or multi-agent critique                                             | Flexible, conversational, and adaptable to clinician questions                                                  | Fluency amplifies plausibility; self-explanations and chain-of-thought can rationalise outputs; unsupported facts and citations remain possible                                 | High and variable latency/cost                                                       | Restrict to a schema-constrained verbalizer with deterministic validation and fallback                          |
| Hybrid multi-method XAI                   | Combines complementary explanation channels                                                                                      | Can answer attribution, relational, safety, contrastive, and uncertainty questions together                     | Concatenation can produce contradictions and does not create end-to-end faithfulness                                                                                            | Sum of component costs plus orchestration                                            | Recommended only with a common decision target, provenance, conflict handling, and fidelity tests               |




### 1.2 Feature attribution and local surrogates

SHAP unifies additive feature-attribution methods and identifies a unique solution under local accuracy, missingness, and consistency for its chosen value function ([Lundberg and Lee, 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)). TreeSHAP makes this practical for tree ensembles ([Lundberg et al., 2020, DOI 10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)). Those axioms do not select a clinically correct background distribution or intervention. Multiple Shapley operationalisations can yield materially different answers ([Sundararajan and Najmi, 2020](https://proceedings.mlr.press/v119/sundararajan20b.html)). For correlated EHR variables, replacing one measurement independently of related diagnoses, treatments, or time can query implausible states.

LIME learns a sparse local surrogate around the patient being explained ([Ribeiro et al., 2016, DOI 10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)). Its value is diagnostic simplicity, not a guarantee of truth. BayLIME's motivation and experiments document inconsistency across repeated LIME explanations and sensitivity to kernel settings ([Zhao et al., 2021](https://proceedings.mlr.press/v161/zhao21a.html)). In this project, a local surrogate built from randomly toggled EHR events could be faithful to an artificial neighbourhood while misleading about the observed clinical manifold.

Integrated Gradients (IG) accumulates gradients along a path from baseline `x'` to input `x`:


\operatorname{IG}_i(x)=(x_i-x'_i)\int_0^1
\frac{\partial f(x'+\alpha(x-x'))}{\partial x_i}d\alpha .


It was designed to satisfy sensitivity and implementation invariance, with a completeness relation to the baseline output ([Sundararajan et al., 2017](https://proceedings.mlr.press/v70/sundararajan17a.html)). The baseline is therefore part of the scientific question, not a plotting option. Very recent EHR work proposes a validation-set mean embedding as a manifold-aware baseline and group-sparse IG to reduce dense token explanations; it reports improved comprehensiveness/sufficiency and 9–18% lower token-level group density on two EHR tasks ([Amirahmadi et al., 2026](https://proceedings.mlr.press/v297/amirahmadi26a.html)). This is strong prior art against claiming novelty for sparse or EHR-specific IG alone, but it is directly useful as a baseline and design reference.

Permutation importance is best interpreted as model reliance under a specified perturbation. Model reliance formalises performance changes across models and perturbations ([Fisher et al., 2019](https://www.jmlr.org/papers/v20/18-760.html)). Unrestricted permutation can force extrapolation into low-density regions and inflate importance for correlated variables ([Hooker and Mentch, 2021, DOI 10.1007/s11222-021-10057-z](https://doi.org/10.1007/s11222-021-10057-z)). Conditional or group permutation over clinical concepts and time windows is safer than independent column shuffling.

### 1.3 Attention and Transformer explanations

Attention is useful internal instrumentation but weak standalone evidence. Jain and Wallace found that attention weights were often weakly correlated with gradient-based importance and that very different attention distributions could produce equivalent predictions ([Jain and Wallace, 2019, DOI 10.18653/v1/N19-1357](https://aclanthology.org/N19-1357/)). Across Transformer layers, representations mix token information; attention rollout and attention flow improve correlation with ablation and gradient signals compared with raw attention but remain approximations ([Abnar and Zuidema, 2020, DOI 10.18653/v1/2020.acl-main.385](https://aclanthology.org/2020.acl-main.385/)). Relevance-propagation methods explicitly handle residual connections and attention layers and can conserve relevance across layers ([Chefer et al., 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)).

RETAIN is a seminal intrinsically interpretable EHR architecture using reverse-time visit attention and variable-level attention ([Choi et al., 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html)). Its clinical vocabulary and temporal structure remain valuable, but the general attention-faithfulness caveat still applies unless interventions confirm that attended events affected the output.

For the active Transformer branch, the correct target is not “importance to the candidate's probability” but a contrast such as the pre-safety score margin between candidate `a` and candidate `b`. Attributions should be aggregated from embedding coordinates into concepts a clinician can inspect: diagnosis/event, measured value, missingness indicator, visit, and time window. Raw embedding-dimension saliency is not human-interpretable.

### 1.4 Counterfactual, example, rule, concept, and causal explanations

Counterfactual explanations search for a nearby input that changes the decision. DiCE explicitly optimises both proximity and diversity so users can see several alternative routes ([Mothilal et al., 2020, DOI 10.1145/3351095.3372850](https://doi.org/10.1145/3351095.3372850)); a broad survey shows that feasibility, actionability, sparsity, plausibility, and causal validity remain distinct design choices ([Guidotti, 2022, DOI 10.1007/s10618-022-00831-6](https://doi.org/10.1007/s10618-022-00831-6)). Medication counterfactuals must hold immutable facts fixed, respect temporal order and coupled variables, distinguish recorded changes from interventions, and never be phrased as advice to manipulate a lab or diagnosis.

Prototype and case-based explanations answer “what known pattern is this like?” ProtoPNet learns class prototypes and compares parts of a new input with them ([Chen et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html)). Clinical case retrieval can be intuitive, but its value depends on a validated distance and on showing trajectories and outcomes relevant to the decision. Retrieved historical prescriptions are observations, not labels of optimal care. Protected patient-level examples also require stricter access and display controls than aggregate model evidence.

Anchors return high-precision if–then conditions under a perturbation distribution ([Ribeiro et al., 2018, DOI 10.1609/aaai.v32i1.11491](https://ojs.aaai.org/index.php/AAAI/article/view/11491)). They can be useful local summaries, but an anchor's empirical coverage and precision must be displayed. Exact safety rules are different: if the safety engine itself applied a versioned contraindication rule, replaying that rule is an intrinsically faithful explanation of that engine.

Concept Bottleneck Models (CBMs) predict named concepts and then use those concepts for the final task, permitting human interventions at the bottleneck ([Koh et al., 2020](https://proceedings.mlr.press/v119/koh20a.html)). TCAV instead tests sensitivity along learned concept directions in hidden space ([Kim et al., 2018](https://proceedings.mlr.press/v80/kim18d.html)). Newer CBM research examines when interventions help and how concept interactions can be represented ([Steinmann et al., 2024](https://proceedings.mlr.press/v235/steinmann24a.html); [Zhu et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1c364d98a5cdc426fd8c76fbb2c10e34-Abstract-Conference.html)). For this project, concepts such as indication class, renal-risk context, infection severity, or prior intolerance would require clinician-defined semantics, reliable labels, and checks for information leaking around the bottleneck.

Feature attribution is associational unless the value function and data-generating assumptions are causal. Janzing et al. show how causal structure changes feature relevance ([Janzing et al., 2020](https://proceedings.mlr.press/v108/janzing20a.html)), while causal Shapley values incorporate a causal graph into the coalitional expectation ([Heskes et al., 2020](https://papers.nips.cc/paper/2020/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf)). A retrospective medication dataset cannot by itself identify how changing a treatment would change a patient's outcome; observed prescribing also contains confounding by indication. The near-term explanation layer should therefore say “the model score would change under this valid input edit,” not “the patient's outcome would improve.”

### 1.5 Graph-specific explanations and path evidence

GNNExplainer optimises a compact subgraph and feature mask to maximise information about a particular prediction ([Ying et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)). PGExplainer amortises edge-mask generation across instances ([Luo et al., 2020](https://proceedings.neurips.cc/paper/2020/file/e37b08dd3015330dcbb5d6663667b8b8-Paper.pdf)); GraphMask learns differentiable edge gates and can investigate message-passing paths ([Schlichtkrull et al., 2021](https://openreview.net/forum?id=WznmQa42ZAx)); SubgraphX combines Monte Carlo tree search with Shapley values to capture subgraph interactions at substantially greater cost ([Yuan et al., 2021](https://proceedings.mlr.press/v139/yuan21c.html)).

GraphFramEx shows why a selected subgraph should be tested for both necessity and sufficiency rather than judged visually. Its `Fid+` family asks how much the prediction changes when the explanation is removed; `Fid−` asks how well the explanatory subgraph alone preserves the prediction ([Amara et al., 2022](https://proceedings.mlr.press/v198/amara22a.html)). A 2025 ACM survey still characterises GNN explanation as an immature area with unresolved correctness, robustness, usability, understandability, and cost questions ([Li and Wang, 2025, DOI 10.1145/3711122](https://doi.org/10.1145/3711122)).

Knowledge-graph recommendation has long used paths as rationales. PGPR performs policy-guided path reasoning ([Xian et al., 2019](https://arxiv.org/abs/1906.05237)); KPRN scores user–item paths ([Wang et al., 2019](https://arxiv.org/abs/1811.04540)); PEARLM constrains generated paths to valid KG entities and relations ([Balloccu et al., 2023](https://arxiv.org/abs/2310.16452)). A reusable model-agnostic framework trains a white-box KG reasoner and assesses agreement with a black-box recommender ([Zhang et al., 2023, DOI 10.1145/3605357](https://doi.org/10.1145/3605357)). KAPER now combines KG path evidence with LLM reranking and explanation generation ([Yang et al., 2026, DOI 10.1016/j.knosys.2026.116236](https://doi.org/10.1016/j.knosys.2026.116236)). These works make “KG paths plus natural language” established prior art.

A graph path has three independent validity questions:

1. **Graph validity:** do all typed edges exist in the versioned graph?
2. **Clinical/evidential validity:** do the sources and relation semantics support the statement for this patient context?
3. **Model faithfulness:** did removing or retaining that path materially affect the candidate-relative GNN or final rank score?

The active GNN's learned relation gates and candidate-conditioned attention can help developers locate candidate evidence, but they should not be displayed as the final explanation unless path/subgraph interventions verify the third property.

### 1.6 Uncertainty-aware and LLM-generated explanations

Prediction confidence, model calibration, epistemic uncertainty, and explanation instability are not interchangeable. Temperature scaling is a useful calibration baseline ([Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)), but it does not expose epistemic uncertainty or guarantee that an attribution is stable. InfoSHAP attributes conditional predictive entropy rather than only the point prediction ([Watson et al., 2023, DOI 10.52202/075280-0320](https://proceedings.neurips.cc/paper_files/paper/2023/hash/16e4be78e61a3897665fa01504e9f452-Abstract-Conference.html)). GPEC estimates explanation uncertainty from decision-boundary complexity and approximation uncertainty ([Hill et al., 2024](https://proceedings.mlr.press/v238/hill24a.html)); other wrappers provide conformal or Bayesian intervals around existing attribution methods ([Marx et al., 2023](https://proceedings.mlr.press/v206/marx23a.html)). Calibrated Explanations combine rule-like feature effects, uncertainty, and counterfactuals ([Löfström et al., 2024, DOI 10.1016/j.eswa.2024.123154](https://doi.org/10.1016/j.eswa.2024.123154)).

An LLM's explanation of its own answer is not reliable process evidence. Chain-of-thought can rationalise biased decisions without mentioning the bias ([Turpin et al., 2023, DOI 10.52202/075280-3275](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html)), and self-explanation faithfulness varies by model and task ([Madsen et al., 2024](https://aclanthology.org/2024.findings-acl.19/)). A 2024 survey emphasises that NLP faithfulness needs behavioural tests, not merely resemblance to human rationales ([Lyu et al., 2024, DOI 10.1162/coli_a_00511](https://direct.mit.edu/coli/article/50/2/657/119158/Towards-Faithful-Model-Explanation-in-NLP-A-Survey)). Verbalised uncertainty is also unreliable without calibration ([Tanneru et al., 2024](https://proceedings.mlr.press/v238/harsha-tanneru24a.html)).

The safe role for the project's LLM is consequently narrow but useful: extract a structured profile with span provenance, and later verbalise a closed, validated evidence packet. The packet—not the prose—is the authoritative explanation.

## 2. Healthcare evidence and clinician explanation needs



### 2.1 What the literature supports

Clinicians do not ask for a single universal explanation. Interview work in critical-care ML found demand for patient-specific context, uncertainty, actionable factors, and examples that make sense for the clinical trajectory ([Tonekaboni et al., 2019](https://proceedings.mlr.press/v106/tonekaboni19a.html)). A recent comparison of explanation types with 39 hospital and critical-care clinicians found the most positive overall response to feature attribution, while almost half preferred access to multiple explanation forms; the limited two-centre sample prevents broad generalisation ([clinician user study, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13370522/)). Reviews likewise emphasise clear, actionable, patient-relevant explanations that support validation and decision-making rather than generic transparency ([healthcare-professional scoping review, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13175488/)).

Explanation does not monotonically increase appropriate trust. A 2024 systematic review found that explanations can increase or decrease clinician trust and can also produce blind reliance ([Rosenbacke et al., 2024, DOI 10.2196/53207](https://doi.org/10.2196/53207)). Reader studies show strong human variability and cases where added explanations confuse rather than help ([Nicolson et al., 2025, DOI 10.1038/s41746-025-02023-0](https://doi.org/10.1038/s41746-025-02023-0)). Recent experiments with more than 300 medical professionals favour a human-first, AI-second workflow for appropriate reliance rather than immediate exposure to AI advice ([Wang et al., 2026](https://ojs.aaai.org/index.php/AAAI/article/view/41457)). The goal should therefore be **trust calibration and contestability**, not maximal trust.

The evaluation evidence remains weak relative to the volume of healthcare XAI publications. A 2025 meta-analysis of 62 XAI-CDSS studies reports sparse real-world evaluation of fidelity, trust, and usability ([Albahri et al., 2025, DOI 10.3390/healthcare13172154](https://doi.org/10.3390/healthcare13172154)). A 2026 lifecycle review of 136 health recommender papers reports that 59.6% provided no explanation, with 9.6% post-hoc, 12.5% intrinsic, and 18.4% evidence-linked explanations; explanation and assurance were uneven and often optional ([Liu et al., 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13273477/)). This supports a gap in evaluated, lifecycle-wide explanation rather than a shortage of explanation visualisations.

### 2.2 Clinical questions and the evidence needed to answer them


| Clinician question                                                   | Required evidence object                                                         | Suitable method                                                                                 | Unsafe shortcut                                            |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Which patient factors influenced this recommendation?                | Signed contributions grouped by clinical concept and time                        | Contrastive IG plus in-distribution occlusion; TreeSHAP for tree baseline                       | Raw gradient or embedding dimension                        |
| Which diagnosis, symptom, laboratory result, or vital sign mattered? | Input provenance, value/time, direction, magnitude, stability                    | Grouped temporal attribution with missingness represented separately                            | Treating missing values as normal values or causal factors |
| Why is medication `a` above `b`?                                     | Pairwise rank-margin decomposition on a fixed candidate slate                    | ShaRP/RankingSHAP-style rank value function, branch decomposition, grouped IG/GNN interventions | Two unrelated pointwise explanations                       |
| Which treatments were considered?                                    | Candidate slate, stage-by-stage score/rank, filtering reason                     | Pipeline trace                                                                                  | Display only the winner                                    |
| What supported the recommendation relationally?                      | Necessary/sufficient typed subgraph and path provenance                          | GraphMask/PGExplainer/GNNExplainer plus path intervention                                       | Highest attention weight or any path that exists           |
| Which safety constraints changed the result?                         | Exact triggered rule, inputs, severity, action, source and rule version          | Intrinsic rule trace/replay                                                                     | LLM paraphrase without rule identifier                     |
| Why was a candidate rejected?                                        | Hard-veto certificate or quantified soft penalty                                 | Constraint trace plus counterfactual replay                                                     | A generic “unsafe” label                                   |
| What medical evidence supports the statement?                        | Versioned retrieved source and passage/structured relation                       | Evidence retrieval with citation validation                                                     | Treating model attribution as medical evidence             |
| How certain is the system?                                           | Calibrated score/rank set, distribution-shift warning, explanation stability     | Calibration, ensemble/bootstrap, conformal set where valid                                      | The LLM saying “high confidence”                           |
| What would need to change for the rank to change?                    | Minimal feasible model counterfactual with immutable/causal restrictions         | Clinically constrained optimisation                                                             | Unconstrained feature edits or treatment advice            |
| Can I challenge or inspect this?                                     | Expandable evidence packet, alternatives, unknowns, provenance, feedback channel | Interactive evidence view and audit log                                                         | Static persuasive paragraph                                |


The interface should provide progressive disclosure: a concise rank contrast and safety status first, then temporal details, graph paths, counterfactuals, sources, and audit metadata on demand. This answers the finding that clinicians work under time pressure while preserving depth for difficult or contested cases.

### 2.3 Medication-recommendation prior art

Medication recommendation research commonly treats safety as a training objective rather than an explanation. GAMENet combines longitudinal EHR representations with drug-memory graphs and a DDI loss ([Shang et al., 2019, DOI 10.1609/aaai.v33i01.33011126](https://ojs.aaai.org/index.php/AAAI/article/view/3905)). SafeDrug uses molecular substructure encoders and a controllable DDI objective ([Yang et al., 2021, DOI 10.24963/ijcai.2021/514](https://www.ijcai.org/proceedings/2021/514)). These are important safety-aware baselines, but a lower aggregate DDI rate does not tell a clinician which patient-specific rule changed a rank.

Recent work moves closer to explanation:

- **ExpDrug** maps latent patient representations into an aspect space and controls DDI with a threshold strategy ([Lu et al., 2025, DOI 10.1016/j.neucom.2024.129021](https://doi.org/10.1016/j.neucom.2024.129021)). Its aspect associations improve inspectability, but attention/mapping alignment is not by itself intervention-based faithfulness.
- **KGDNet** combines longitudinal EHR, knowledge graphs, GNNs, and Transformer attention, using case and ablation analyses rather than a comprehensive explanation-fidelity protocol ([Guo et al., 2024, DOI 10.1038/s41598-024-75784-5](https://doi.org/10.1038/s41598-024-75784-5)).
- **DMGExNet** combines a dual-stream Transformer, multiple drug graphs, interpretable aspect mapping, and DDI control. Its published limitation explicitly notes the absence of formal human evaluation of interpretability ([Pan et al., 2026, DOI 10.1002/eng2.70899](https://doi.org/10.1002/eng2.70899)).
- **KEHGCN** incorporates external knowledge, higher-order hypergraphs, explicit contraindication relations, and top-weighted generalized metapaths ([Zhang et al., AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38681)). The paper presents explanation case studies but does not report formal fidelity or human evaluation, so metapath weights should not be read as a validated account of the final decision.
- **SafeRx-Agent** uses patient context, indication knowledge, DDI/contraindication resources, multiple LLM agents, critique, and safety verification ([Wang et al., 2026, arXiv preprint](https://arxiv.org/abs/2605.29146)). It is very close prior art for “patient context + knowledge + safety + traceable report,” but it does not solve attribution to a learned Transformer–GNN ranker, explanation uncertainty, or intervention-based end-to-end faithfulness. Its preprint status also warrants caution.
- A very recent **multi-agent GraphRAG pharmacotherapy verifier** encodes indications, contraindications, interactions, laboratory thresholds, dose limits, graph traversal, and safety-directed reranking ([Sovrano et al., 2026, DOI 10.3389/fmed.2026.1898857](https://doi.org/10.3389/fmed.2026.1898857)). It strengthens the case for explicit safety verification and provenance, while remaining a different task from explaining learned candidate-relative rank contributions.

The literature boundary is consequently sharp: sequential features, graph knowledge, aspect mappings, metapaths, DDI losses, safety verification, and LLM reports are already established. What is missing is a common, testable account of how those heterogeneous stages jointly produced the *final constrained ordering*.

## 3. Mapping XAI to the project architecture



### 3.1 End-to-end explanation flow

```text
Conversational input
  -> structured PatientProfile with source spans, missingness, contradictions
  -> candidate slate and Transformer temporal scores
  -> GNN relational scores and candidate-conditioned evidence subgraphs
  -> fixed-slate fusion / rank-margin computation
  -> hard safety feasibility + soft safety/evidence adjustments
  -> calibrated rank uncertainty and explanation stability
  -> versioned evidence packet with completeness/conflict diagnostics
  -> constrained LLM verbalisation and clause verifier
  -> clinician review, expansion, correction, and contestation
```



### 3.2 Conversational input and PatientProfile extraction

The extraction model should expose a structured audit trail before recommendation:

- normalized concept and value;
- source span or structured-field identifier;
- timestamp or uncertainty about time;
- negation, experiencer, and status;
- extraction confidence calibrated on held-out data;
- conflicts between conversation and record;
- required fields that are unknown rather than inferred.

For each extracted field, the useful explanation is provenance and extraction uncertainty, not hidden-token saliency. A clinician must be able to correct the profile, and the system should rerun ranking and show what changed. Corrections create valuable audit data but must not be treated as ground-truth treatment labels without review.

### 3.3 Transformer branch

Let `t_m(x)` be the Transformer score for medication candidate `m`. The primary question is contrastive:


\Delta^{T}_{a,b}(x)=t_a(x)-t_b(x),


not merely `t_a(x)`. Apply IG to `Delta^T` from a carefully defined background, then aggregate along the model's input structure:

```text
embedding dimension -> event/value/missingness -> clinical concept
                    -> visit/time window -> positive or negative evidence
```

Recommended validation:

1. compare zero, masked-token, cohort/reference, and manifold-aware baselines;
2. delete top attributed event groups and measure the decline in `Delta^T`;
3. retain only the explanation and measure sufficiency;
4. compare with time-window and concept-group occlusion;
5. repeat across seeds, bootstrap models, and clinically irrelevant perturbations;
6. run parameter- and label-randomization sanity checks.

Attention rollout, attention flow, and relevance propagation can be retained as secondary diagnostics. Agreement across methods is not proof; disagreement is a useful warning. SHAP at clinically grouped input units can be a model-agnostic audit baseline on a small sample, but KernelSHAP over all tokens is too expensive and its independent masking semantics are risky.

### 3.4 GNN branch

Let `g_m(x,G)` be the relational score from the patient/candidate graph. The explainer should optimise a sparse, candidate-contrastive mask for


\Delta^{G}_{a,b}(x,G)=g_a(x,G)-g_b(x,G).


Use GNNExplainer as an initial per-instance baseline; evaluate an amortised PGExplainer or GraphMask-style explainer if explanation latency matters. SubgraphX is a useful interaction-aware audit on small graphs, not the likely serving method. Every extracted subgraph should be converted to one or more typed paths only after it passes:

- **necessity:** removing the subgraph/path materially reduces the original margin;
- **sufficiency:** retaining it approximately preserves the margin;
- **validity:** all nodes/edges are present and types/directions are meaningful;
- **temporal validity:** no future or post-treatment event appears in the explanation window;
- **stability:** similar models/inputs preserve the essential relation pattern;
- **clinical review:** the path is relevant and not a spurious coding/co-occurrence relation.

For a typed patient graph, a displayable path might have the abstract form `observed diagnosis -> relation -> clinical concept -> relation -> candidate`, with signs and source provenance. It must never be described as causal unless the graph edge has a causal interpretation supported by an explicit causal model.

### 3.5 Fusion and ranking

The active late-fusion design combines standardized Transformer and GNN scores on a ranking group. For a linear fusion such as


r_m=(1-\alpha)z(t_m)+\alpha z(g_m),


the branch contributions to a fixed-slate pairwise margin are exactly inspectable:


\Delta^{rank}_{a,b}
=(1-\alpha)[z(t_a)-z(t_b)]
+\alpha[z(g_a)-z(g_b)].


This exact branch decomposition should be shown before using a model-agnostic explainer. Interventions must hold the candidate slate fixed and recompute the same standardization rule; otherwise removing one candidate can change all standardized scores and confound the explanation. Residual/nonlinear fusion would require a separate component-attribution or Shapley interaction analysis.

Recent rank-specific XAI is directly relevant. RankingSHAP defines listwise Shapley value functions over ranking properties and evaluates preservation/deletion of ranked outcomes ([Heuss et al., SIGIR 2025, DOI 10.1145/3726302.3729971](https://doi.org/10.1145/3726302.3729971)). ShaRP supports rank, top-k, and pairwise preference value functions ([Pliatsika et al., 2025, DOI 10.14778/3749646.3749682](https://doi.org/10.14778/3749646.3749682)). These methods rule out novelty claims based only on pairwise Shapley attribution; they should be baselines for the final-rank target.

### 3.6 Safety and evidence layer

Hard and soft constraints require separate semantics:

- **Hard constraints** determine feasibility. Their explanation is a proof/certificate: rule identifier and version, patient facts used, candidate relation, severity, action, source, and timestamp. A hard veto should not be forced into an additive “importance score.”
- **Soft safety adjustments** affect ranking and can be included quantitatively in the pairwise margin, with the exact penalty/bonus and triggered rules.
- **Unknown safety state** is not a pass. Missing renal/hepatic information, unresolved allergy, unavailable dose, or stale knowledge should produce a warning or abstention according to policy.

The rule trace should answer both “why was this selected?” and “why was that rejected?” It should also expose the counterfactual replay: if the rule were not triggered, where would the candidate rank? This is a model/pipeline audit, not a recommendation to ignore the rule.

Aggregate DDI rate remains a model safety metric, but patient-level explanation needs the actual interaction pair, evidence source, severity, and consequence for the candidate. DDI or contraindication knowledge must be versioned because rules and evidence can change.

### 3.7 Knowledge and retrieved evidence

Knowledge retrieval should return typed evidence atoms rather than a paragraph:

```json
{
  "claim_id": "K17",
  "subject": "candidate_a",
  "relation": "has_versioned_relation",
  "object": "clinical_context_x",
  "source_id": "source-version-passage",
  "valid_from": "YYYY-MM-DD",
  "scope": "population/context",
  "retrieval_score": 0.0,
  "support_status": "supports|contradicts|insufficient"
}
```

The system should distinguish model evidence from external clinical evidence in the interface. A retrieved source can make a recommendation rationale better grounded clinically while having no role in the original model computation. If retrieval also reranks candidates, its score change must be logged as a separate computational contribution.

### 3.8 LLM verbalisation

The LLM receives only the validated evidence packet, an output schema, and controlled terminology. It may compress, order, and phrase evidence; it may not introduce a new diagnosis, treatment relation, safety assertion, causal claim, certainty statement, or citation. A post-generation verifier should parse the prose into clauses and require each material clause to map to one or more packet IDs. Unsupported clauses are removed or the system falls back to deterministic templates.

The final display should retain machine-readable anchors, for example:

```text
Candidate A ranks above Candidate B mainly because [T3, T8] and [G2].
Candidate B was excluded by hard rule [R14].
This ordering was stable in 84% of bootstrap models [U4].
External evidence [K17] supports the stated relation but did not affect the model score.
```

This is an abstract format, not a clinical recommendation. Chain-of-thought should neither be requested nor shown. The evidence packet is auditable; free-form hidden reasoning is not.

## 4. Faithfulness versus plausibility



### 4.1 Operational definitions for this project


| Property                      | Operational question                                                                               | Proposed measurement                                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Local accuracy / conservation | Do signed contributions reconstruct the defined score or rank margin?                              | Absolute and relative conservation residual                                                            |
| Fidelity                      | Does a surrogate reproduce the ranker in the relevant neighbourhood?                               | Pairwise agreement, top-k overlap, Kendall's tau, score/margin error on valid perturbations            |
| Faithfulness                  | Does changing the claimed evidence change the actual decision as claimed?                          | Necessity/sufficiency interventions, deletion/insertion curves, rank flips, retraining checks          |
| Completeness                  | How much of the decision is covered by displayed evidence?                                         | Conserved margin fraction plus unallocated interaction/residual; coverage of triggered pipeline stages |
| Stability                     | Does essentially the same case/model produce essentially the same explanation?                     | Rank correlation, top-k Jaccard, sign agreement across seeds/bootstrap and small invariant changes     |
| Robustness                    | Can irrelevant or adversarially small changes alter the explanation while preserving the decision? | Worst-case explanation distance under prediction-preserving valid perturbations                        |
| Sparsity                      | Is the explanation concise?                                                                        | Number of event groups, nodes, edges, paths, and clauses at a fixed fidelity threshold                 |
| Consistency                   | Are equivalent implementations/candidates handled coherently?                                      | Implementation-invariance tests, symmetry tests, subgroup consistency, cross-method conflict rate      |
| Clinical validity             | Are concepts, paths, rules, and sources medically correct in context?                              | Blinded clinician ratings, rule/source audit, temporal-validity error rate                             |
| Usefulness                    | Does the explanation improve the clinician's task?                                                 | Decision accuracy, error detection, time, contestation quality, appropriate reliance                   |




### 4.2 Why visual agreement is insufficient

Feature-attribution methods can produce visually plausible outputs even after model parameters or labels are randomized. Sanity checks therefore compare explanations before and after these randomizations ([Adebayo et al., 2018](https://research.google/pubs/sanity-checks-for-saliency-maps/)). Saliency methods can also violate input invariance and assign different explanations to functionally equivalent inputs ([Kindermans et al., 2019](https://research.google/pubs/the-unreliability-of-saliency-methods/)).

Deletion tests can create off-manifold inputs and reward an explainer for exploiting the model's behaviour on nonsense records. ROAR removes features and retrains to reduce this mismatch ([Hooker et al., 2019](https://research.google/pubs/evaluating-feature-importance-estimates/)), though retraining is expensive and asks a population-level question. ERASER's comprehensiveness and sufficiency metrics are useful templates for rationale evaluation ([DeYoung et al., 2020, DOI 10.18653/v1/2020.acl-main.408](https://aclanthology.org/2020.acl-main.408/)). Infidelity and sensitivity provide model-agnostic mathematical measures of explanation behaviour under perturbations ([Yeh et al., 2019](https://papers.nips.cc/paper_files/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html)). Saliency metric rankings themselves can be unstable, so no single benchmark score should define success ([Tomsett et al., 2020, DOI 10.1609/aaai.v34i04.6064](https://doi.org/10.1609/aaai.v34i04.6064)). Quantus offers an implementation framework spanning multiple quantitative properties ([Hedström et al., 2023](https://www.jmlr.org/papers/v24/22-0142.html)).

### 4.3 Experimental verification ladder

An explanation should pass increasingly realistic tests:

1. **Axiomatic/unit tests:** dummy features receive zero, symmetric features behave symmetrically, contributions reconstruct the selected output, identical functions produce equivalent IG results.
2. **Known-ground-truth synthetic tasks:** generate temporal and graph cases where only specified events/paths determine a pairwise rank.
3. **Model randomization:** randomise weights or labels; explanations should lose task-specific structure.
4. **Clinically valid perturbation tests:** remove/retain grouped events, values, paths, or rules using data-consistent replacements; measure the actual rank margin.
5. **Rank-specific preservation/deletion:** retain only top evidence or delete it and recompute the entire fixed-slate ranking; report Kendall's tau, NDCG/top-k changes, and target pair flips.
6. **Graph necessity/sufficiency:** use GraphFramEx-style metrics for the candidate-relative margin and check path validity.
7. **Pipeline replay:** rerun fusion, safety, retrieval, and reranking from the same versioned inputs; verify the final evidence ledger.
8. **Counterfactual validity:** require the claimed minimal edit to flip the rank when replayed, and test its plausibility/immutability constraints independently.
9. **Language grounding:** split generated text into atomic claims; calculate supported-claim precision/recall, citation correctness, contradiction rate, omitted-critical-rule rate, and numerical consistency.
10. **Clinical application-grounded evaluation:** determine whether clinicians find incorrect recommendations and calibrate reliance better, not merely whether they prefer the prose.

Doshi-Velez and Kim distinguish functionally grounded, human-grounded, and application-grounded evaluation ([Doshi-Velez and Kim, 2017](https://arxiv.org/abs/1702.08608)). The project's technical metrics cover the first; clinician studies must cover the latter two. The System Causability Scale can complement, but not replace, performance-based human evaluation ([Holzinger et al., 2020, DOI 10.1007/s13218-020-00636-z](https://doi.org/10.1007/s13218-020-00636-z)). A 2024 human-centred review identifies explanation quality, interaction, and human–AI task performance as separate evaluation groups ([Rong et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525002/)).

## 5. Research gap and novelty assessment



### 5.1 What is established

The following are already established or close to established:

- local feature attribution with SHAP, TreeSHAP, IG, gradients, and local surrogates;
- attention/rollout and relevance propagation for Transformers;
- sparse subgraph and edge-mask explanations for GNNs;
- knowledge-graph reasoning paths and path-to-text explanations;
- rank-, top-k-, pairwise-, and listwise-specific Shapley attribution;
- concept bottlenecks, prototypes, examples, counterfactuals, and rule explanations;
- DDI-aware and contraindication-aware medication recommendation;
- aspect-space explanations for medication models;
- uncertainty attribution and intervals around explanations;
- evidence-grounded or critiqued LLM explanations;
- multi-agent medication generation with safety verification.

It would therefore be inaccurate to claim novelty for any single item, or for simply placing their outputs next to one another.

### 5.2 What remains unresolved


| Gap                                | What related work already covers                                         | What is still missing for this project                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Ranking rather than classification | RankingSHAP and ShaRP define listwise/pairwise value functions           | Clinical event/graph/rule interventions on the same final medication rank target                                             |
| Temporal EHR attribution           | IG, sparse/manifold-aware IG, RETAIN                                     | Candidate-contrastive ranking, valid temporal baselines, missingness, and downstream safety/fusion effects                   |
| GNN subgraph/path explanation      | GNNExplainer, PGExplainer, GraphMask, SubgraphX, KG path recommenders    | Typed patient-specific paths proven necessary/sufficient for the candidate-relative score and final rank                     |
| Medication safety                  | GAMENet, SafeDrug, KEHGCN, SafeRx-Agent, GraphRAG safety verification    | Exact separation of hard feasibility from soft preference evidence in the final explanation object                           |
| Clinical grounding                 | KG retrieval, RAG, guideline citations, path-to-text                     | Clear distinction between “the model used this,” “a rule fired,” and “external evidence supports this”                       |
| Uncertainty                        | Calibration, InfoSHAP, GPEC, conformal explanation work                  | Joint presentation of rank uncertainty, attribution stability, path stability, and knowledge/rule unknowns                   |
| Natural-language explanation       | LLM-generated recommender rationales and multi-agent critique            | Clause-level machine verification against an immutable, versioned evidence packet                                            |
| Whole pipeline                     | Work on model stacks, component attribution, and pipeline Shapley values | A clinical ranking protocol that replays extraction, branches, fusion, constraints, retrieval, and language generation       |
| Clinical evaluation                | Preference/trust/usability studies                                       | Appropriate reliance, error detection, contestability, and workflow value under correct and deliberately incorrect AI advice |


This review did **not** find a peer-reviewed method that jointly provides all of the following for medication recommendation:

```text
candidate-relative temporal attribution
+ intervention-validated relational paths
+ exact hard/soft safety provenance
+ final-rank conservation across components
+ predictive and explanation uncertainty
+ evidence-atom-constrained clinical language
+ technical and clinician evaluation
```

This is a targeted-review finding, not proof of absence. A formal novelty claim would require a registered search across PubMed/MEDLINE, Scopus, Web of Science, IEEE Xplore, ACM Digital Library, ACL Anthology, and major ML proceedings, with backward/forward citation screening and independent review.

### 5.3 Candidate ideas and honest novelty classification


| Candidate idea                                                                      | Novelty assessment                                                                                                                                                                                                          | Recommendation                                                                        |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Concatenate SHAP/IG, GNN paths, safety rules, retrieval, uncertainty, and LLM prose | **Straightforward combination.** Each ingredient and several close hybrids already exist.                                                                                                                                   | Useful engineering baseline, not the paper's methodological claim                     |
| Apply pairwise Shapley values to medication candidates                              | **Already established in general ranking XAI.** ShaRP explicitly covers pairwise preferences.                                                                                                                               | Baseline or adaptation study only                                                     |
| Sparse/manifold-aware IG for EHR Transformers                                       | **Already established by 2026 EHR work.**                                                                                                                                                                                   | Reuse and validate; do not claim as new                                               |
| Use weighted metapaths and contraindications                                        | **Already close to KEHGCN.**                                                                                                                                                                                                | Baseline; add intervention validation if used                                         |
| Use KG paths to prompt an LLM                                                       | **Already established in explainable recommendation and KAPER.**                                                                                                                                                            | Implementation pattern only                                                           |
| Ground an LLM in patient context and safety resources                               | **Close to SafeRx-Agent and pharmacotherapy GraphRAG.**                                                                                                                                                                     | Not standalone novelty                                                                |
| Explain the final constrained pairwise rank with a conserved, typed evidence ledger | **Potentially meaningful contribution.** Ingredients exist, but the shared decision target, hard/soft separation, cross-channel intervention contract, residual, and verification could be novel as a method and benchmark. | Primary research direction, conditional on formal prior-art review and empirical gain |
| Intrinsic concept–path medication ranker                                            | **Incremental unless it solves concept/path faithfulness in a new way.** CBMs, aspect mapping, and metapath models are close.                                                                                               | Longer-term secondary direction                                                       |
| Causal medication counterfactuals                                                   | **Scientifically valuable but presently underidentified.** Novelty depends on a defensible causal model, not terminology.                                                                                                   | Long-term work after causal assumptions/data are established                          |




## 6. Proposed method: an end-to-end constraint-aware rank-evidence ledger



### 6.1 Core idea

**Proposed research.** Treat explanation as a typed, verifiable decomposition of a particular final ranking contrast, not as prose and not as a bag of independent importance scores. The ledger records:

- what patient evidence changed the Transformer margin;
- what graph evidence changed the GNN margin;
- how branch fusion changed the final learned margin;
- which soft rules adjusted it;
- which hard rules made candidates infeasible;
- what external knowledge supports or contradicts the interpretation;
- how stable each item is;
- what remains unexplained;
- which evidence IDs license every generated clause.

A provisional descriptive name is **constraint-aware rank-evidence ledger**. A branded name should wait for a formal prior-art/name search.

### 6.2 Formal decision target

Let:

- `x` be the versioned patient profile and temporal events;
- `G` be the patient/clinical graph;
- `M` be the fixed candidate slate;
- `t_m`, `g_m` be Transformer and GNN scores;
- `F(t_m,g_m)` be the learned fusion score;
- `q_m(x,G)` be an exact sum of soft rule/evidence adjustments;
- `h_m(x,G) in {0,1,unknown}` be hard feasibility.

Define the feasible utility


u_m=
\begin{cases}
F(t_m,g_m)+q_m, & h_m=1,
-\infty, & h_m=0,
\operatorname{abstain}, & h_m=\text{unknown and policy requires resolution}.
\end{cases}


For two feasible candidates `a` and `b`, the explanatory target is


\Delta_{a,b}=u_a-u_b.


Hard exclusions are represented by a separate certificate `C_m`, because an infinite/Boolean feasibility decision is not meaningfully decomposed into additive feature weights. If `b` is hard-excluded, the explanation says that `C_b` caused exclusion and optionally reports the learned preference margin that would have existed before the safety gate. It must not imply that a learned feature “outweighed” a contraindication.

### 6.3 Hierarchical evidence game

Partition evidence units into clinically meaningful groups:


\mathcal E = \mathcal E_T \cup \mathcal E_G \cup \mathcal E_Q,


where `E_T` contains temporal event/value/missingness groups, `E_G` contains typed graph substructures or paths, and `E_Q` contains soft rule adjustments. Branch identity is a higher-level partition.

For coalition `S`, define a **clinically valid intervention operator** `I_S` that retains evidence in `S` and replaces other evidence using a conditional/manifold-aware reference while preserving immutable variables, temporal order, graph typing, and the fixed candidate slate. The value function is


v_{a,b}(S)=
\mathbb E\left[\Delta_{a,b}\bigl(I_S(x,G)\bigr)\right]
-\mathbb E\left[\Delta_{a,b}\bigl(I_\varnothing(x,G)\bigr)\right].


Use hierarchical/Owen-style or partitioned Shapley attribution to allocate the margin first across branches and then across evidence units, preserving cross-unit interactions as defined by the chosen game. This is not claimed as a new game-theoretic solution; the research question is whether the clinically constrained value function and cross-pipeline evidence contract improve faithfulness over pointwise or independently computed explanations.

For each unit `j`, estimate signed contribution `phi_j` and a conservation residual


\epsilon_{a,b}=\Delta_{a,b}
-\left(\phi_0+\sum_{j\in\mathcal E}\phi_j\right).


The interface reports the unexplained fraction


\rho_{a,b}=\frac{|\epsilon_{a,b}|}{|\Delta_{a,b}|+\eta}.


An explanation should be suppressed or labelled incomplete when `rho` exceeds a preregistered threshold. If the architecture permits an exact branch decomposition, use it rather than approximate Shapley attribution at that level.

### 6.4 Relational evidence extraction

For each candidate contrast, learn or optimise an edge mask `M_G` on the GNN computation graph:


\max_{M_G} \operatorname{Fid}_{a,b}(M_G)
-\lambda_1\lVert M_G\rVert_0
-\lambda_2\operatorname{Invalid}(M_G)
-\lambda_3\operatorname{Unstable}(M_G),


where fidelity measures preservation of the pairwise graph margin, `Invalid` penalises type/temporal/source violations, and `Unstable` penalises explanations that change across model/bootstrap replicas. Extract short typed paths from the selected subgraph, but retain the subgraph as the primary computational explanation because multiple interacting paths can carry the effect.

For every displayed path `p`, record:


N_p=\Delta^G_{a,b}(G)-\Delta^G_{a,b}(G\setminus p)


as a necessity effect and


S_p=\left|\Delta^G_{a,b}(G)-\Delta^G_{a,b}(p)\right|


as a sufficiency error, subject to graph-valid intervention semantics. A path may be clinically supportive yet fail model-faithfulness tests; the ledger must preserve that distinction.

### 6.5 Clinically constrained counterfactuals

For a feasible competitor `b`, search over allowed event/value edits `delta`, graph edits `gamma`, and resolvable soft constraints:


\begin{aligned}
\min_{\delta,\gamma}\quad & c(\delta,\gamma)
\text{s.t.}\quad & u_b(I_{\delta,\gamma}(x,G))
\ge u_a(I_{\delta,\gamma}(x,G))+\kappa,
& h_a,h_b=1,
& (\delta,\gamma)\in\mathcal A_{clinical},
& \text{immutable and temporal constraints hold}.
\end{aligned}


The cost should be group-sparse and scaled by clinical plausibility, not Euclidean distance in an embedding. The result is labelled a **model-rank counterfactual**. Unless supported by a structural causal model, it does not claim that inducing the change is possible, safe, or outcome-improving.

### 6.6 Explanation uncertainty

Fit or retain bootstrap/ensemble model replicas. For each evidence atom `j`, report:

- contribution interval `[L_j,U_j]`;
- sign stability `P(phi_j > 0)` or `P(phi_j < 0)`;
- top-k inclusion frequency;
- path-edge inclusion frequency;
- rank-order stability `P(u_a > u_b)`;
- whether the case is out of the calibration/reference distribution.

Calibrated probabilities and conformal candidate sets can communicate predictive uncertainty under their coverage assumptions; they do not certify an explanation. Explanation uncertainty must be evaluated separately. Frequency-style communication may be easier to interpret than an unsupported confidence percentage, consistent with a 2024 uncertainty-display study ([Zukerman and Maruf, 2024, DOI 10.18653/v1/2024.inlg-main.4](https://aclanthology.org/2024.inlg-main.4/)).

### 6.7 Evidence packet and language contract

The ledger is a versioned object, schematically:

```json
{
  "decision": {
    "patient_profile_version": "...",
    "model_versions": {"transformer": "...", "gnn": "...", "fusion": "..."},
    "candidate_slate_hash": "...",
    "candidate_a": "...",
    "candidate_b": "...",
    "pairwise_margin": 0.0,
    "rank_stability": 0.0
  },
  "evidence": [
    {
      "id": "T3|G2|Q4",
      "channel": "temporal|graph|soft_rule",
      "direction": "supports_a|supports_b",
      "contribution": 0.0,
      "interval": [0.0, 0.0],
      "stability": 0.0,
      "provenance": "...",
      "necessity": 0.0,
      "sufficiency_error": 0.0
    }
  ],
  "hard_constraint_certificates": [
    {"id": "R14", "rule_version": "...", "status": "pass|fail|unknown", "source": "..."}
  ],
  "external_knowledge": [
    {"id": "K17", "source_version": "...", "support_status": "supports|contradicts|insufficient"}
  ],
  "counterfactuals": [],
  "conservation_residual": 0.0,
  "warnings": []
}
```

Every material natural-language clause must cite packet IDs. The validator checks entity/value consistency, polarity, numeric ranges, rule status, source mapping, unsupported medical relations, and omitted critical warnings. A second unconstrained LLM is not a sufficient verifier; deterministic schema checks and source entailment tests are required, with clinician review for high-risk semantics.

### 6.8 Algorithm sketch

```text
INPUT: versioned patient profile x, graph G, fixed slate M,
       trained Transformer/GNN/fusion, rule engine, knowledge index

1. Validate extraction provenance, required fields, time windows, and missingness.
2. Score every candidate with Transformer and GNN branches on the fixed slate.
3. Recompute the deployed fusion exactly; store branch-level pairwise margins.
4. Run hard safety rules. Emit pass/fail/unknown certificates and remove/abstain by policy.
5. Apply and log soft rule/evidence adjustments; select each displayed rank contrast.
6. Attribute the Transformer contrast to clinical event/time groups; validate by occlusion.
7. Extract the GNN contrastive subgraph; test necessity, sufficiency, validity, and stability;
   derive displayable typed paths only from the validated subgraph.
8. Allocate cross-channel evidence for the final feasible margin; calculate conservation residual.
9. Estimate rank and explanation uncertainty across replicas/bootstraps; set warning/abstention flags.
10. Generate clinically constrained model counterfactuals when requested and validate by replay.
11. Retrieve source-versioned clinical evidence; mark it as computational or contextual evidence.
12. Serialize the ledger. Generate text from allowed atoms, verify every clause, or use a template.
13. Log clinician expansion, correction, rejection, and contestation without exposing raw records.

OUTPUT: machine-readable ledger + verified clinician-facing explanation
```



### 6.9 Expected advantages, weaknesses, and complexity

Expected advantages:

- explains the actual ranking contrast rather than unrelated candidate probabilities;
- separates model dependence, rule execution, and external clinical support;
- makes interactions and unexplained residual visible;
- exposes both positive and negative/rejection evidence;
- supports contestability and replay;
- turns LLM prose into a view over evidence rather than the source of evidence;
- permits technical and human evaluation at matching levels.

Likely weaknesses:

- Shapley-style allocation remains sensitive to baseline, grouping, and intervention semantics;
- valid EHR/graph interventions are difficult and may require generative conditional models;
- exact cross-channel attribution is exponential; approximations add variance;
- graph masks can be non-identifiable when redundant paths exist;
- contribution intervals may be wide, yielding frequent abstention;
- rules and knowledge require continual versioning and governance;
- the interface may overload clinicians unless progressive disclosure is carefully designed;
- retrospective clinical evaluation cannot establish patient benefit.

For `d` evidence groups and `K` sampled coalitions, a model-agnostic hierarchical attribution is approximately `O(K * C_pipeline)`, with exact Shapley exponential in `d`. IG is approximately `O(m * C_transformer)` for `m` steps. Iterative graph masking is approximately `O(I * C_GNN)` for `I` optimisation steps; amortised PGExplainer/GraphMask shifts cost to training. An ensemble of `B` replicas roughly multiplies the inference/explanation work by `B`, though branch/coalition calls can be batched and cached. LLM generation is not the dominant scientific cost but adds latency and validation work.

### 6.10 What would be genuinely novel

The proposed paper should **not** claim the individual attribution algorithms as new. A credible contribution would need all of the following:

1. a formally defined final-rank value function for a heterogeneous clinical pipeline;
2. intervention operators that preserve clinical grouping, temporal validity, graph typing, and fixed-slate semantics;
3. explicit mathematical separation of hard feasibility certificates from additive preference evidence;
4. a cross-channel evidence ledger with measurable margin conservation, interactions/residual, and uncertainty;
5. clause-level grounding between ledger atoms and clinician-facing language;
6. a medication-ranking benchmark/protocol that jointly evaluates temporal, graph, rule, uncertainty, and language faithfulness;
7. evidence that the unified method improves both technical faithfulness and clinician error detection/appropriate reliance over independent explainers.

If the work merely connects existing modules and generates a nicer report, its honest contribution would be an **auditable clinical-AI systems architecture** rather than a new XAI technique. That can still be publishable in biomedical informatics if rigorously evaluated.

## 7. Evaluation framework



### 7.1 Prerequisites and split discipline

Explanation evaluation cannot rescue an invalid recommender. All experiments must use patient-level splits and temporal cutoffs; fit backgrounds, normalisers, concept maps, explainers, calibration, retrieval indices, and thresholds on training/development data only. Record cohort definition, source dataset/version, feature and label windows, seed, model version, rule/knowledge version, candidate slate, and explainer configuration. Historical medication and post-treatment events are leakage risks. Clinical records must remain protected; published examples should be synthetic or safely aggregate.

Use three evaluation strata:

1. **Synthetic/semisynthetic ground truth:** known temporal rules, paths, and constraint triggers.
2. **Retrospective held-out EHR:** model-behaviour and clinical plausibility, with no claim of treatment optimality.
3. **Prospective or silent-mode clinician study:** workflow value and error detection, only after safety/governance review.



### 7.2 Technical metrics


| Dimension                 | Metric                                                                              | Concrete measurement for this system                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Conservation/completeness | Relative residual `rho`, covered-stage rate                                         | Difference between final pairwise margin and sum of baseline/evidence allocations; fraction of active stages represented |
| Rank fidelity             | Pairwise agreement, Kendall's tau, NDCG/top-k preservation                          | Rerank valid perturbations with surrogate/retained evidence and compare with full pipeline                               |
| Necessity                 | Deletion AOPC, target-pair flip rate, `Fid+`                                        | Remove top temporal groups, graph subgraphs/paths, or soft rules and recompute final rank                                |
| Sufficiency               | Retention error, `Fid−`, margin reconstruction                                      | Retain only explanatory evidence with a valid reference and compare target margin/rank                                   |
| Stability                 | Top-k Jaccard, Spearman correlation, sign consistency                               | Across model seeds, bootstrap resamples, IG steps/baselines, explanation samples, and small valid input changes          |
| Robustness                | Worst-case explanation distance under invariant decision                            | Semantics/prediction-preserving perturbations and adversarial explanation attacks                                        |
| Sparsity                  | Evidence count at fixed fidelity                                                    | Events, time windows, nodes, edges, paths, rules, sources, and clauses needed to meet a threshold                        |
| Graph validity            | Typed-edge/path validity, temporal leakage rate                                     | Automatic ontology/type/direction/source/time checks                                                                     |
| Counterfactual validity   | Replay success, feasibility, diversity, cost                                        | Fraction that really flip the target rank and satisfy clinical/immutable constraints                                     |
| Rule faithfulness         | Trigger precision/recall and replay agreement                                       | Exact comparison with deployed rule engine; critical-rule omission target should be zero                                 |
| Predictive uncertainty    | ECE, Brier score, log loss, coverage/set size                                       | Candidate probability/rank calibration and conformal coverage where assumptions are justified                            |
| Explanation uncertainty   | Interval coverage on synthetic truth, sign/top-k stability                          | Bootstrap/conformal intervals and empirical inclusion frequencies                                                        |
| Language grounding        | Claim precision/recall, unsupported-claim rate, contradiction, citation correctness | Atomic clause-to-ledger/source alignment; critical omission rate                                                         |
| Cost                      | p50/p95 latency, model calls, GPU/CPU memory, cache hit rate                        | Per candidate slate and per requested detail level                                                                       |


Report fidelity against random, frequency/popularity, and shuffled-explanation baselines. Measure confidence intervals and paired significance/effect sizes. Avoid selecting a method because it wins one deletion metric; triangulate with ground-truth, conservation, stability, and clinical validity.

### 7.3 Baselines

Recommended explanation baselines:

- random and frequency/popularity evidence;
- raw attention and attention rollout/flow;
- raw gradients, IG, and the 2026 manifold-aware/group-sparse IG approach;
- LIME and KernelSHAP on clinical feature groups;
- TreeSHAP on the strongest tree ranker/reference model;
- pointwise SHAP/IG versus RankingSHAP and ShaRP-style pairwise/listwise objectives;
- GNNExplainer, PGExplainer, GraphMask, and SubgraphX on a tractable subset;
- top attention/gate-weight graph paths without intervention validation;
- exact rule trace versus a learned rule/surrogate summary;
- unconstrained counterfactual versus clinically constrained counterfactual;
- free-form LLM rationale, template-only rationale, evidence-packet LLM, and packet+validator;
- full unified ledger versus independent explanations displayed side by side.

For recommender/model comparisons, include current tree, Transformer, GNN, and fusion baselines and representative safety/explanation systems where reproducible: GAMENet, SafeDrug, ExpDrug, KEHGCN, and DMGExNet. SafeRx-Agent is a useful contemporary conceptual baseline but, as a preprint and LLM-agent system, may not be directly comparable to a supervised ranker.

### 7.4 Required ablations

1. pointwise candidate explanation versus pairwise/listwise rank target;
2. naive zero/mask baseline versus cohort/manifold-aware baseline;
3. independent event perturbation versus clinically grouped/conditional intervention;
4. raw attention versus IG versus IG+occlusion validation;
5. raw GNN attention/relation gates versus selected subgraph versus intervention-validated paths;
6. fixed candidate slate and normalisation versus incorrectly changing the slate;
7. no cross-channel interaction allocation versus hierarchical allocation;
8. no conservation residual/warning versus residual-gated display;
9. safety embedded as a score versus separate hard certificate plus soft penalties;
10. no uncertainty versus predictive uncertainty only versus predictive+explanation uncertainty;
11. no counterfactual versus unconstrained versus clinically constrained counterfactual;
12. no external evidence versus retrieved evidence with and without version/provenance;
13. deterministic template versus LLM verbalizer versus LLM+clause validator;
14. removal of each evidence channel: Transformer, graph, safety, retrieved knowledge, uncertainty;
15. full method versus equal-cost/latency alternatives.



### 7.5 Clinical and human evaluation

Use a preregistered, blinded, within-subject/crossover study with representative clinicians. Include cases stratified by specialty, complexity, missingness, distribution shift, rule trigger, close versus wide rank margin, and explanation stability. Crucially, include both correct and deliberately incorrect AI recommendations and explanations; otherwise the study mainly measures persuasion.

Suggested conditions:

1. patient information only;
2. recommendation/ranking only;
3. ranking plus feature attribution;
4. ranking plus attribution and graph/safety evidence;
5. full ledger-based explanation with uncertainty and counterfactual;
6. full evidence but template language versus constrained LLM language.

Primary human outcomes:

- correctness of the clinician's final judgement;
- detection of incorrect or unsafe AI advice;
- **appropriate reliance:** acceptance when AI is correct and rejection when wrong;
- ability to identify the relevant safety rule or missing information;
- quality of challenge/contest response;
- time to decision and time to locate evidence.

Secondary outcomes:

- clinical relevance and completeness;
- understandability and cognitive load;
- usefulness/actionability;
- trust calibration rather than raw trust;
- confidence calibration;
- expert agreement on concepts, paths, and counterfactual plausibility;
- System Causability Scale and qualitative thematic analysis;
- alert fatigue and information overload.

Present clinicians' initial judgement before AI output, then collect post-AI judgement and reasons. This makes anchoring and overreliance measurable. Analyse expertise and specialty effects and report disagreement, not only mean ratings. Do not use the retrospective observed prescription as unquestioned clinical ground truth; adjudicate with guidelines and multiple experts where feasible.

## 8. Recommended practical architecture



### 8.1 Existing techniques to use now

- **TreeSHAP** for tree-reference models.
- **Candidate-contrastive IG**, with clinical grouping and multiple baseline checks, for the Transformer.
- **Event/time-window occlusion** as a direct validation of Transformer attributions.
- **GNNExplainer initially**, then PGExplainer or GraphMask if amortised latency is needed.
- **GraphFramEx-style necessity/sufficiency** and typed/temporal path validation.
- **Exact rule replay and versioned safety certificates**.
- **Calibration, bootstrap/ensemble stability, and explicit abstention**.
- **Structured evidence retrieval with provenance**.
- **Deterministic templates first**, followed by an evidence-constrained LLM verbalizer only after clause validation exists.



### 8.2 Techniques to combine

Combine techniques only around a shared final decision object:

```text
Patient profile provenance
  + Transformer pairwise temporal evidence
  + GNN pairwise subgraph/path evidence
  + exact branch/fusion contribution
  + hard safety certificate and soft rule deltas
  + retrieved clinical support/contradiction
  + rank and explanation uncertainty
  + clinically valid model counterfactual
  -> versioned evidence ledger
  -> verified concise explanation
  -> expandable clinician audit/contest view
```

The user interface should label channels explicitly: **model evidence**, **safety rule**, **external evidence**, **uncertainty**, and **model counterfactual**. It should show supporting and opposing factors and preserve unknowns.

### 8.3 Techniques to avoid or sharply limit

- “We used LIME/SHAP” as the complete explanation strategy.
- SHAP values over embedding coordinates.
- raw attention, candidate-conditioned pooling weights, or relation gates presented as faithful reasoning.
- arbitrary KG paths selected only because they are semantically plausible.
- unconstrained independent perturbations of correlated EHR variables.
- global permutation importance presented as an individual-patient rationale.
- unconstrained counterfactuals or causal wording from observational data.
- prototypes/cases with unvalidated similarity or protected patient detail.
- DDI rate or safety loss presented as a patient-specific safety explanation.
- LLM chain-of-thought, free-form rationales, invented citations, or LLM-reported confidence.
- explanation agreement or clinician preference treated as proof of faithfulness.
- a single faithfulness metric or visually selected examples.



### 8.4 Staged implementation and research plan

**Stage 0 — Explanation contract and logging (partially implemented).** The versioned, torch-free evidence schema and fail-closed synthetic contract tests are implemented in `[pipeline.explainability](../pipeline/explainability/contract.py)` and `[tests/test_explanation_contract.py](../tests/test_explanation_contract.py)`. The current schema enforces evidence hierarchy and margin conservation, distinguishes hard safety certificates from learned evidence, links retrieved knowledge to reranking evidence explicitly, requires replayed clinically permitted model counterfactuals, and stores references instead of clinical rows or source text. Candidate-slate capture beyond the explained pair, protected logging/storage policy, component adapters, and runtime integration remain planned. This is necessary engineering, not evidence of novelty or clinical validation.

**Stage 1 — Validated component explanations.** Implement grouped contrastive IG plus occlusion, a GNNExplainer baseline plus necessity/sufficiency, exact fusion decomposition, safety certificates, and deterministic templates. Benchmark cost and stability on synthetic fixtures and held-out development data.

**Stage 2 — Unified rank ledger.** Implement hierarchical cross-channel allocation, conservation residual, fixed-slate interventions, conflict handling, and machine replay. Compare with pointwise, pairwise ranking, and independent multi-method baselines.

**Stage 3 — Uncertainty and counterfactuals.** Add bootstrap/ensemble explanation stability, calibrated rank sets, clinically constrained counterfactuals, and abstention criteria.

**Stage 4 — Grounded language and clinician evaluation.** Add clause-level grounding/validation, then conduct a human-first clinician study with correct and incorrect AI advice. Do not make deployment claims from retrospective evaluation.

## 9. Final research recommendation



### 9.1 Recommended methods and combination

The minimum defensible explanation stack is:

1. grouped pairwise IG plus occlusion for temporal Transformer evidence;
2. candidate-contrastive masked subgraphs plus necessity/sufficiency and typed paths for GNN evidence;
3. exact fusion and soft-rule score decomposition on a fixed candidate slate;
4. separate hard safety rule certificates and rejection reasons;
5. versioned external knowledge with support/contradiction status;
6. rank calibration plus explanation stability/intervals and abstention;
7. optional clinically constrained model counterfactuals;
8. a structured evidence ledger verbalised by templates or a clause-verified LLM.



### 9.2 Strongest research gap

Current systems usually explain an isolated model component, expose semantically plausible attention/aspect/path weights, optimise aggregate DDI safety, or generate a grounded report. They rarely demonstrate that the temporal factors, graph paths, safety rules, uncertainty, and cited evidence jointly correspond to the final constrained medication ordering. The strongest gap is therefore **faithful explanation of the complete ranking process, including rejection and uncertainty, at a clinically inspectable level**.

### 9.3 Is a new technique justified?

**Yes, conditionally.** A new *framework/protocol* is justified because the cross-component faithfulness problem is real and clinically important. A new algorithm is justified only if it introduces and validates decision-aligned intervention semantics or allocation that existing rank and component explainers do not provide. Simply integrating existing outputs is not a new XAI technique.

The most realistic contribution is a combination of:

- a formal final-rank explanation target;
- constraint-aware, clinically valid perturbations;
- exact hard/soft safety separation;
- a conserved and uncertainty-aware evidence ledger;
- clause-verifiable language;
- a multi-level evaluation benchmark.



### 9.4 Proposed research question and hypotheses

**Primary research question**

> Can a constraint-aware, evidence-conserving explanation of the final medication rank—integrating temporal attribution, intervention-validated graph evidence, safety certificates, and explanation uncertainty—improve faithfulness and clinicians' ability to detect and contest unsafe or incorrect recommendations compared with pointwise and independently combined XAI methods?

Supporting questions:

- **RQ1:** Does a final-rank value function produce higher preservation/deletion fidelity than pointwise explanations?
- **RQ2:** Do clinically valid temporal/graph interventions improve stability and human validity over naive masking?
- **RQ3:** Does separating hard safety certificates from preference attribution reduce explanation errors and improve rejection understanding?
- **RQ4:** Do conservation residuals and explanation intervals improve appropriate reliance under unstable or shifted cases?
- **RQ5:** Does a clause-verified LLM retain the usability of natural language without increasing unsupported-claim rate relative to templates?

Preregistered hypotheses could be:

- `H1`: the unified ledger reduces pairwise margin reconstruction error and improves rank preservation/deletion over pointwise SHAP/IG, RankingSHAP/ShaRP adaptations, and independent explainers at matched sparsity;
- `H2`: intervention-validated graph paths have higher necessity/sufficiency and clinician relevance than attention/metapath-weight paths;
- `H3`: displaying rule certificates and uncertainty increases error detection and appropriate reliance without materially increasing decision time;
- `H4`: clause validation reduces unsupported clinical claims to a preregistered near-zero threshold compared with free-form LLM explanation.



### 9.5 Paper-level contribution

A strong journal or conference paper would contribute:

1. the formal constraint-aware final-rank explanation problem;
2. the evidence-ledger algorithm and clinically valid intervention operators;
3. a reproducible synthetic/retrospective benchmark for temporal, graph, rule, and language faithfulness;
4. comparison with rank-aware, attribution, GNN, path, counterfactual, and LLM baselines;
5. complete ablations and computational analysis;
6. a clinician study measuring error detection, appropriate reliance, and contestability;
7. an honest limitations section separating model behaviour, medical evidence, and causal claims.

If the algorithmic novelty proves incremental after a formal review, the work can still make a valuable biomedical-informatics contribution as an audited, safety-aware explanation architecture with unusually rigorous end-to-end and clinician evaluation.

## 10. Selected bibliography



### General XAI and evaluation

- Ribeiro, Singh, and Guestrin. “Why Should I Trust You?” Explaining the Predictions of Any Classifier. KDD 2016. [DOI 10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778).
- Lundberg and Lee. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017. [Paper](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).
- Sundararajan, Taly, and Yan. Axiomatic Attribution for Deep Networks. ICML 2017. [Paper](https://proceedings.mlr.press/v70/sundararajan17a.html).
- Lundberg et al. From Local Explanations to Global Understanding with Explainable AI for Trees. Nature Machine Intelligence 2020. [DOI 10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9).
- Mothilal, Sharma, and Tan. Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations. FAT* 2020. [DOI 10.1145/3351095.3372850](https://doi.org/10.1145/3351095.3372850).
- Koh et al. Concept Bottleneck Models. ICML 2020. [Paper](https://proceedings.mlr.press/v119/koh20a.html).
- Rudin. Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead. Nature Machine Intelligence 2019. [DOI 10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x).
- Nauta et al. From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI. ACM Computing Surveys 2023. [DOI 10.1145/3583558](https://doi.org/10.1145/3583558).
- Hedström et al. Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond. JMLR 2023. [Paper](https://www.jmlr.org/papers/v24/22-0142.html).
- Watson et al. Explaining Predictive Uncertainty with Information Theoretic Shapley Values. NeurIPS 2023. [DOI 10.52202/075280-0320](https://doi.org/10.52202/075280-0320).
- Hill et al. Boundary-Aware Uncertainty for Feature Attribution Explainers. AISTATS 2024. [Paper](https://proceedings.mlr.press/v238/hill24a.html).



### Transformer, GNN, ranking, and recommender explanation

- Jain and Wallace. Attention Is Not Explanation. NAACL 2019. [DOI 10.18653/v1/N19-1357](https://doi.org/10.18653/v1/N19-1357).
- Abnar and Zuidema. Quantifying Attention Flow in Transformers. ACL 2020. [DOI 10.18653/v1/2020.acl-main.385](https://doi.org/10.18653/v1/2020.acl-main.385).
- Chefer, Gur, and Wolf. Transformer Interpretability Beyond Attention Visualization. CVPR 2021. [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html).
- Amirahmadi, Etminani, and Ohlsson. Group-Sparse Manifold-Aware Integrated Gradients for Multimodal Transformers on EHR Trajectories. PMLR 2026. [Paper](https://proceedings.mlr.press/v297/amirahmadi26a.html).
- Ying et al. GNNExplainer. NeurIPS 2019. [Paper](https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html).
- Luo et al. Parameterized Explainer for Graph Neural Network. NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html).
- Schlichtkrull et al. Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking. ICLR 2021. [Paper](https://openreview.net/forum?id=WznmQa42ZAx).
- Yuan et al. On Explainability of Graph Neural Networks via Subgraph Explorations. ICML 2021. [Paper](https://proceedings.mlr.press/v139/yuan21c.html).
- Amara et al. GraphFramEx. LoG/PMLR 2022. [Paper](https://proceedings.mlr.press/v198/amara22a.html).
- Li and Wang. Can Graph Neural Networks be Adequately Explained? ACM Computing Surveys 2025. [DOI 10.1145/3711122](https://doi.org/10.1145/3711122).
- Heuss, de Rijke, and Anand. RankingSHAP. SIGIR 2025. [DOI 10.1145/3726302.3729971](https://doi.org/10.1145/3726302.3729971).
- Pliatsika et al. ShaRP: Explaining Rankings and Preferences with Shapley Values. PVLDB 2025. [DOI 10.14778/3749646.3749682](https://doi.org/10.14778/3749646.3749682).
- Zhang et al. A Reusable Model-agnostic Framework for Faithfully Explainable Recommendation and System Scrutability. ACM TOIS 2023. [DOI 10.1145/3605357](https://doi.org/10.1145/3605357).
- Yang et al. Leveraging Graph Path Evidence for Explainable Recommender Systems (KAPER). Knowledge-Based Systems 2026. [DOI 10.1016/j.knosys.2026.116236](https://doi.org/10.1016/j.knosys.2026.116236).



### Healthcare and medication recommendation

- Choi et al. RETAIN: An Explainable Predictive Model for Healthcare Using Reverse Time Attention Mechanism. NeurIPS 2016. [Paper](https://proceedings.neurips.cc/paper_files/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html).
- Tonekaboni et al. What Clinicians Want: Contextualizing Explainable Machine Learning for Clinical End Use. MLHC/PMLR 2019. [Paper](https://proceedings.mlr.press/v106/tonekaboni19a.html).
- Ghassemi, Oakden-Rayner, and Beam. The False Hope of Current Approaches to Explainable Artificial Intelligence in Health Care. Lancet Digital Health 2021. [DOI 10.1016/S2589-7500(21)00208-9](https://doi.org/10.1016/S2589-7500(21)00208-9).
- Holzinger et al. Measuring the Quality of Explanations: The System Causability Scale. KI 2020. [DOI 10.1007/s13218-020-00636-z](https://doi.org/10.1007/s13218-020-00636-z).
- Ali et al. Deep Learning for Medication Recommendation: A Systematic Survey. Data Intelligence 2023. [DOI 10.1162/dint_a_00197](https://doi.org/10.1162/dint_a_00197).
- Shang et al. GAMENet. AAAI 2019. [DOI 10.1609/aaai.v33i01.33011126](https://doi.org/10.1609/aaai.v33i01.33011126).
- Yang et al. SafeDrug. IJCAI 2021. [DOI 10.24963/ijcai.2021/514](https://doi.org/10.24963/ijcai.2021/514).
- Guo et al. KGDNet. Scientific Reports 2024. [DOI 10.1038/s41598-024-75784-5](https://doi.org/10.1038/s41598-024-75784-5).
- Lu et al. ExpDrug. Neurocomputing 2025. [DOI 10.1016/j.neucom.2024.129021](https://doi.org/10.1016/j.neucom.2024.129021).
- Pan et al. DMGExNet. Engineering Reports 2026. [DOI 10.1002/eng2.70899](https://doi.org/10.1002/eng2.70899).
- Zhang et al. Knowledge-Enhanced Explainable Hypergraph Convolution Network for Medication Recommendation. AAAI 2026. [DOI 10.1609/aaai.v40i19.38681](https://doi.org/10.1609/aaai.v40i19.38681).
- Wang et al. SafeRx-Agent. 2026 preprint. [arXiv:2605.29146](https://arxiv.org/abs/2605.29146).
- Sovrano et al. A Multi-agent GraphRAG Framework for Pharmacotherapy Safety Verification in Clinical Decision Support Systems. Frontiers in Medicine 2026. [DOI 10.3389/fmed.2026.1898857](https://doi.org/10.3389/fmed.2026.1898857).
- Albahri et al. Explainable Artificial Intelligence in Clinical Decision Support Systems: Meta-analysis. Healthcare 2025. [DOI 10.3390/healthcare13172154](https://doi.org/10.3390/healthcare13172154).
- Liu et al. Explainability, Safety, Privacy, and Fairness Across the Health Recommender Lifecycle. 2026. [Open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13273477/).



### LLM explanation faithfulness

- Turpin et al. Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. NeurIPS 2023. [DOI 10.52202/075280-3275](https://doi.org/10.52202/075280-3275).
- Madsen et al. Are Self-Explanations from Large Language Models Faithful? Findings of ACL 2024. [Paper](https://aclanthology.org/2024.findings-acl.19/).
- Lyu et al. Towards Faithful Model Explanation in NLP: A Survey. Computational Linguistics 2024. [DOI 10.1162/coli_a_00511](https://doi.org/10.1162/coli_a_00511).
- Asgari et al. A Framework to Assess Clinical Safety and Hallucination Rates of LLMs for Medical Text Summarisation. npj Digital Medicine 2025. [DOI 10.1038/s41746-025-01670-7](https://doi.org/10.1038/s41746-025-01670-7).



## Bottom line

The project should move beyond “SHAP/LIME explainability,” but it should also resist novelty by assembly. The practical system should be built from validated existing techniques now. The research contribution should target the unresolved end-to-end contract: a final-rank explanation that is computationally faithful, safety-explicit, evidence-aware, uncertainty-aware, language-verifiable, and useful for clinician challenge rather than merely persuasive.