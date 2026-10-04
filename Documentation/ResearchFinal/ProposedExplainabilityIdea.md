# Proposed explainability idea for the fused ranker

Status: the complete research method remains proposed and experimentally
unvalidated. The numerical margin/route core and bounded grouped-attribution
runner are implemented with synthetic unit tests. Clinical model integration,
source interventions, graph explanation, replay, reversal search, and comparative
evaluation remain planned. See the
[implementation plan](ExplainabilityImplementationPlan.md) for current gates.

## Research object

The model under explanation is the frozen paired-OOF hybrid ranker: a
Transformer event-sequence branch and a candidate-conditioned relational GNN
branch. For a fixed candidate slate \(C\), the deployed fusion is

\[
u_c=(1-\alpha)z_C(t_c)+\alpha z_C(g_c),
\]

where \(t_c\) and \(g_c\) are the branch scores and \(z_C\) is the finite,
within-slate population z-normalisation used by the scorer. The currently
promoted ranking configuration has \(\alpha=0.25\). The explanation target is
the final ordering, especially the pairwise margin

\[
M_{a,b}=u_a-u_b.
\]

The method must answer why candidate \(a\) is above candidate \(b\), how each
branch and evidence group changes that ordering, how other candidates affect
the comparison through slate normalisation, and which admissible changes would
reverse the order.

## Classification

Using the taxonomy in [Explanation.pdf](../Explanation.pdf), this is primarily
a **local, post-hoc, model-specific, contrastive ranking explanation**. It has
the following supporting forms:

- feature/evidence attribution for event and context groups;
- graph/structural explanation for candidate-conditioned relations;
- computational decomposition of branch disagreement and slate effects;
- temporal/sequential explanation for event-window contributions; and
- model-rank counterfactual explanation for verified rank reversal.

The primary question is why the model produced this ordering. The numerical
decomposition and future replay procedure provide evidence for that answer.
Describing how an explainer is implemented is not itself an explanation of a
model decision. The project's “why/how” distinction is an operational convention;
interpretability and explainability are not universally disjoint literature terms.

The selected fusion is additive in normalized branch scores: it has no learned
cross-branch interaction term. Source information can enter both branches, and
each branch may be nonlinear. Do not invent an additional fusion/synergy term
or add it to contributions already assigned to the two branches.

It is not currently an intrinsic explanation, a global surrogate, an
example-based explanation, a causal treatment explanation, or a natural-
language rationale method. Natural language, clinical rules, external
knowledge, and uncertainty may later present or qualify the result, but they
must not replace the model-behaviour explanation.

## Proposed explanation operator

For a branch with score gap \(d=s_a-s_b\) and reciprocal scale \(r=1/h\),
where \(h=\max(\sigma_C,\epsilon)\), an intervention changes the branch from
\((d,r)\) to \((d',r')\). Its exact margin
change is

\[
wr'd'-wrd
=\underbrace{w\frac{r+r'}{2}(d'-d)}_{\text{raw-gap route}}
+\underbrace{w\frac{d+d'}{2}(r'-r)}_{\text{normalisation route}},
\]

where \(w\) is \(1-\alpha\) for the Transformer or \(\alpha\) for the GNN.
The first route says that the evidence changed the relative scores of \(a\)
and \(b\). The second says that it changed the candidate-set scale and hence
changed how strongly that branch's preference counted. The two terms sum to the
exact finite change; the symmetric split is a declared allocation convention,
not a causal claim.

The normalisation route can be decomposed by candidate. For population variance
\(v\),

\[
v'-v=\sum_{c\in C}
\frac{(s'_c+s_c-\mu'-\mu)(s'_c-s_c)}{|C|}.
\]

This records when a third candidate changes the A-versus-B ordering even though
A and B's raw scores do not change. Every intervention recomputes both branch
scores and both slate normalisations for the complete fixed slate.

Writing each candidate's variance allocation as \(\ell_c\), transport it
through the reciprocal-scale divided difference:

\[
K=\frac{r(v')-r(v)}{v'-v},\qquad
N_c=w\frac{d+d'}{2}K\ell_c.
\]

Above the floor, use \(K=-1/(hh'(h+h'))\), including equal variances.
When both variances are at or below the floor, choose \(K=0\); for a crossing,
use the finite divided difference. These conventions allocate opposing effects
even when their sum is zero. They do not uniquely identify causal routes.
The implementation reports floating-point residuals rather than adjusting
contributions to make a sum appear exact.

With common interventions, branch and margin completeness, comparator
antisymmetry, and \(\Phi_{j,AC}=\Phi_{j,AB}+\Phi_{j,BC}\) are testable
properties. They are requirements, not unique novelty claims. Preserve all
adjacent margins simultaneously to validate a complete strict order; top-k
membership requires min-inside greater than max-outside after replay.

## Attribution and replay algorithm

1. Freeze model versions, preprocessing, graph statistics, candidate
   membership, masks, tie handling, and the fusion weight.
2. Partition underlying source information into disjoint meaningful groups:
   temporal observations and values, static context, and their derived
   representations. Conduct graph-message/edge masking as a separate
   computational intervention experiment. Do not add a source effect to a
   downstream route-mask effect as independent evidence. Actual soft adjustments
   and hard eligibility policies require separate future adapters; neither is
   part of the current numerical blend.
3. Define source-consistent interventions. Removing an event must update every
   dependent Transformer and graph representation. Unsupported or invalid
   interventions are recorded as unavailable, not silently discarded.
4. For each sampled evidence-addition order, recompute the full candidate slate
   after every intervention. Record branch gaps, scales, final margins, and
   raw-gap/normalisation route values.
5. Average transition effects over declared references and permutations. This
   produces grouped attribution for the final pairwise margin while retaining
   the additional computational route labels.
6. Validate a compact explanation by replay: retain the selected evidence,
   remove it, and test necessity and sufficiency on the final pairwise margin
   and the declared whole-ranking constraints.
7. Search admissible input or relation edits for a strict rank reversal,
   \(M_{a,b}\le -\kappa\), and verify every proposed counterfactual by running
   the original full-slate scorer. Report the best verified edit found within
   budget; do not call it globally minimal without a certificate.
8. Store the decision snapshot, route contributions, intervention metadata,
   replay results, residual/error estimates, and unsupported capabilities. A
   future evidence bundle or language layer may consume this result.

## Claims and limits

This is not a claim that the individual ingredients are new. RankingSHAP,
RankSHAP, ShaRP, preference Shapley methods, graph explainers, and attribution
flow methods are relevant prior art. The candidate contribution is a
mechanism-aware explanation of a candidate-normalised Transformer--GNN fusion:
it preserves the final ranking target, separates raw preference from slate
normalisation, and uses the same replay semantics to guide rank-reversal search.

The route totals reduce to an ordinary grouped Shapley-style allocation when
averaged over evidence permutations. Therefore novelty must be established by
the route information, consistent whole-slate replay, and measurable benefit
over strong rank-aware and independently-composed baselines—not by claiming a
new Shapley theorem or by conservation alone.

The method explains model dependence. It does not establish that a treatment is
clinically optimal, that an observed prescription causes a good outcome, or
that a proposed input change is clinically safe. Hard rules, external medical
evidence, and uncertainty must remain separately typed and versioned.

## Evaluation required before any novelty claim

Use synthetic rankers with known raw-gap and normalisation routes, then the
frozen promoted hybrid. Compare against grouped Shapley/RankingSHAP/ShaRP with
the same interventions, branchwise attribution without route accounting, and
graph explanation baselines. Measure margin reconstruction, pairwise and
top-k rank preservation, deletion/sufficiency fidelity, reversal success and
edit cost, route recovery on synthetic data, stability, coverage, and runtime.

Until those experiments are complete, the repository should distinguish the
implemented numerical prototype from an evaluated explanation system. Synthetic
unit tests establish arithmetic and algorithmic behavior, not clinical
faithfulness, novelty, or superiority over existing explanation methods.
