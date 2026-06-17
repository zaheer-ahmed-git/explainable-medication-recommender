# Poster Presentation Guide

**Poster:** *Towards an Explainable Conversational Medication Recommender System for Clinical Decision Support*
**Authors:** AHMAD Adeel¹, AHMED Zaheer²
**Lab:** SysReIC, Laboratoire d'Informatique Signal et Image de la Côte d'Opale (LISIC), Université du Littoral Côte d'Opale (ULCO), Calais, France


## 1. Opening scripts (memorize these)

### 1a. The 30-second elevator pitch
> "Many AI systems in healthcare can suggest a treatment, but they can't faithfully explain *why*. Our system is an explainable, conversational medication recommender for clinical decision support. A clinician describes a patient in natural language; a language model turns that into a structured patient profile; a hybrid Transformer + Graph Neural Network model ranks the medication options; and a grounded explanation layer justifies every ranking using the patient's own features, medical-knowledge-graph paths, and clinical rules. Crucially, we *separate* recommendation from explanation, so the justification is real evidence, not a fluent story. It supports the doctor's judgment — it never replaces it."

### 1b. The 2-minute walkthrough
> "The problem: clinical decision-making is hard because patients are *multifactorial* — many conditions, labs, and medications at once — and the treatment option space is large. Existing AI often gives *opaque* recommendations with no faithful reason. (Point to **Problematic & Motivation**.)
>
> Our objective (point to **Objective**) is a system that does three things: understands clinical dialogue and structured records, ranks medication candidates with a hybrid model, and explains each ranking with grounded evidence.
>
> The architecture has layers (point to **System Architecture**). Layer 1 is a language-model conversational layer that reads the clinician's text and EHR and builds a structured patient profile, asking for missing data when needed. Layer 2 is the hybrid recommender: a Transformer branch models interactions across patient features, and a GNN branch models medical relationships like symptom–diagnosis–treatment; the two are fused to produce a ranked medication list plus evidence artifacts. Layer 3 is grounded explainability: it combines LIME feature attribution, knowledge-graph paths, rule checks, and model scores, and the LLM only *verbalizes* this evidence. Layer 4 is the medical knowledge and rule layer that feeds and checks everything — for example, contraindications can override a high model score.
>
> The output (point to the **Top Recommended Options** table and **Functional Scenario**) is a clinician-reviewable ranked list, each row with a confidence/score, a rule-check result, an uncertainty indicator, and provenance — where the evidence came from.
>
> We train and evaluate on **real de-identified ICU and hospital data** from **MIMIC-IV** (single-center, Beth Israel Deaconess, 2008–2022) and **eICU-CRD** (multi-center US ICUs, 208 hospitals). MIMIC gives depth — structured prescriptions, labs, diagnoses, and clinical notes; eICU tests whether the approach generalizes across hospitals. The next steps are harmonizing both schemas into one patient profile, building the knowledge graph, and connecting the conversational front end."

### 1c. The 5-minute version
Use 1b, then add: **why MIMIC-IV + eICU** (Section 4.1), scale numbers (364k patients / 546k admissions in MIMIC; 200k ICU stays / 139k patients in eICU), how tables map to each architecture layer (Section 4.5), PhysioNet access ethics (Section 4.6), pipeline/baseline metrics if asked (Section 4.8), leakage controls, and the roadmap. Finish on the conclusion sentence: *"auditable, evidence-grounded prioritization while preserving clinician authority."*

---

## 2. Section-by-section explanation (every block + diagram)

The poster reads in three columns. Left column: Objective → Problematic & Motivation → Solution Strategy → Explainable AI. Middle column: System Architecture → Model Training & Evaluation Pipeline. Right column: Functional Scenario → chat + results table → Integrated Output → Conclusion.

---

### 2.1 Title bar
**Text:** *Towards an Explainable Conversational Medication Recommender System For Clinical Decision Support.*

How to explain each word of the title (judges love this):
- **"Towards"** — it's an in-progress research framework/direction, not a finished commercial product. This is honest and standard in research.
- **"Explainable"** — every recommendation comes with a faithful reason, not a black box.
- **"Conversational"** — the clinician interacts in natural language (dialogue), not by filling forms.
- **"Medication Recommender"** — the task is ranking *drugs/medications* for a patient-condition, not diagnosing disease.
- **"For Clinical Decision Support"** — it *supports* a professional's decision; it is not autonomous and not patient-facing.

---

### 2.2 Objective (top-left)
**Poster text (paraphrased):** Develop an explainable conversational medication recommender for clinical decision support that helps professionals evaluate and prioritize medication options by (1) understanding clinical dialogue and structured records, (2) ranking candidates with a hybrid recommendation module that models both contextual dependencies and medical relationships, and (3) explaining each ranking with grounded evidence from patient features, graph paths, and clinical rules. By **separating recommendation from explanation**, the research addresses the lack of faithful, clinically understandable justification in many healthcare AI systems.

How to present it:
- Read the three numbered goals as "understand → rank → explain." That triad is the spine of the whole poster.
- Emphasize the last sentence: the **separation of recommendation and explanation** is your core novelty. Say *why* it matters: if you ask one model to both decide and explain, it can invent a convincing but false reason. Splitting them forces the explanation to be built from actual evidence.

Key terms here: *contextual dependencies* = how patient features interact together; *medical relationships* = explicit links between medical entities (symptom→diagnosis→treatment); *grounded evidence* = justification backed by concrete sources (features, graph paths, rules) rather than free-form text.

---

### 2.3 Problematic & Motivation (left, hub-and-spoke diagram)
**Diagram:** A central dark circle labeled **"Clinical Decision Support"** with arrows to/from five surrounding boxes:
- **Challenges** (magnifier icon) — the framing label.
- **Multifactorial patient profiles** — real patients have many conditions, labs, and meds simultaneously.
- **High treatment option space** — many possible medications per condition; hard to prioritize.
- **Opaque AI recommendations** — current AI often can't show a faithful reason (⚠️ warning icon).
- **Need for patient-specific ranking** — generic guidance isn't enough; ranking must fit *this* patient.

Below it, an **"Objectives"** strip with four green-icon cards (this previews the solution):
- **Understand clinical context** — extract structured factors from dialogue and EHR.
- **Rank relevant medications** — score and prioritize condition-specific candidates using context and relationships.
- **Explain with grounded evidence** — justify rankings via feature attribution, knowledge graphs, and rules.
- **Support clinician review** — present ranked options with uncertainty and warnings.

How to present the diagram:
> "This hub-and-spoke shows the four pain points that make clinical decision support hard. They all point at the center because each one makes the decision harder. Our four objectives, in the green strip below, map one-to-one onto solving those pain points."

Term to define if asked: **multifactorial** = caused by/involving many factors at once (multiple conditions, labs, lifestyle, meds).

---

### 2.4 Solution Strategy (left, 5 stacked process diagrams)
This is the "how" — five numbered stages, each a mini-flow:

**① Conversational Clinical Understanding (LLM)** — *Understand · Clarify · Structure.*
Icons flow: Case Description → Follow-up Questions → (brain) → EHR Fields (Structured) → Missing Info Detection → **Structured Patient Representation**.
Say: "Stage 1 reads the case, asks follow-ups for missing data, pulls in structured EHR fields, and outputs a clean structured patient representation — not a treatment decision yet."

**② Hybrid Medication Recommendation (Transformer + GNN)** — *Model · Relate · Rank.*
Shows a **Transformer** box (long-range dependencies: symptoms, labs, history, constraints) and a **GNN (Medical Graph)** box (relational evidence: symptom–diagnosis–treatment, similarity, constraints) feeding a **Fusion** step → **Score & Rank Candidate Treatments**.
Say: "Stage 2 is the engine. The Transformer understands how the patient's features interact; the GNN understands medical relationships; we fuse both and rank candidate medications."

**③ Grounded Explainability** — *Explain · Verify · Quantify.*
Four icons: **Local Attribution (LIME)**, **Knowledge Graph Paths**, **Rule-based Checks**, **Uncertainty & Contradictions** → "**LLM Verbalizes Grounded Evidence (No Invention)**".
Say: "Stage 3 builds the explanation from four evidence sources, then the LLM only puts it into readable language — it does not invent reasons."

**④ Medical Knowledge & Rule Integration** — *Constrain · Validate · Audit.*
Icons: **Indications**, **Contraindications**, **Constraints & Rules**, **Provenance & Audit Log**; with a legend of Inputs, Model Version, Evidence Used.
Say: "Stage 4 is the medical knowledge and rules that constrain and validate everything, and it logs provenance for auditing."

**⑤ Clinician-Centered Decision Support** — *Support · Inform · Empower.*
Shows **Ranked Treatment Options (with Scores)** plus bullet outputs: Explanations (grounded evidence), Warnings & Contraindications, Missing Data Prompts, Prioritize & Review — and a clinician icon labeled **"Supports Clinical Judgment, Does Not Prescribe."**
Say: "Stage 5 is what the clinician actually sees: a ranked, scored, explained, warning-annotated list they review. The system supports judgment; it does not prescribe."

How to present the whole strip: "Solution Strategy is the same pipeline as the architecture, drawn as a process: understand → recommend → explain → constrain/audit → present to clinician."

---

### 2.5 Explainable AI for Medical Recommendation (bottom-left, 5 icon cards)
Five cards that detail the explanation layer's outputs, all feeding one box: **Multi-source grounded explanation.**
1. **Feature attribution** — highlights influential variables (e.g., HbA1c, blood pressure, age, renal function).
2. **Knowledge evidence** — links diagnoses and treatments through explicit medical relationships.
3. **Rule checks** — flags contraindications, conflicts, and missing clinical information.
4. **Provenance logging** — records model version, input snapshot, evidence sources, and explanation lineage.
5. **Uncertainty display** — surfaces low-confidence or conflicting evidence instead of definitive claims.

How to present: "These five are the concrete ingredients of a *faithful* explanation. Together they make a multi-source grounded explanation — the opposite of a black box."

Define if asked: **provenance** = the recorded trail of where each piece of evidence and each output came from (which model version, which inputs, which sources), so a result can be audited and reproduced. **Feature attribution** = quantifying how much each input feature contributed to a specific prediction.

---

### 2.6 System Architecture (center, layered block diagram) — the most important panel
Top **INPUTS** row: *Clinical dialogue (chat/questions)* · *Case notes (free text)* · *Structured EHR / patient tables.*

**① Layer 1 — Conversational clinical understanding (LLM):** dialogue & case interpretation; missing-information prompts (e.g., renal function, allergies, meds); extraction → structured patient profile (no treatment decision yet).

**② Structured patient profile:** symptoms, diagnosis, labs, history, constraints, prior interventions — the normalized single representation.

**③ Layer 2 — Hybrid Transformer–GNN recommendation:**
- **3a Transformer branch:** patient-context & feature interactions (long-range clinical dependencies).
- **3b GNN branch:** relational & graph-aware embeddings (symptom–diagnosis–treatment similarity).
- **HYBRID FUSION** merges them → **Ranked medication candidates** + intermediate evidence for explanation.

**⑤ Layer 3 — Grounded explainability (multi-source evidence):** **LLM** (local features) + **KG paths** (graph evidence) + **Rule checks** (contraindications, consistency) + **Model scores** (rank & confidence); faithfulness/contradiction handling → **provenance log** (inputs, model version, evidence trail); the LLM **verbalizes evidence only**.

**⑥ Layer 4 — Clinician review interface:** rankings · grounded rationale · warnings · uncertainty · audit trail.

**Right side panel — ④ MEDICAL KNOWLEDGE & RULE LAYER (continuous support):** a vertical bar listing symptom–diagnosis–treatment relations, constraints & contraindications, consistency checks & rules, evidence paths & similarity, curated medical knowledge sources. A dashed box says it **"feeds candidate generation, GNN, rules, and explanations."**

How to present the diagram (top-to-bottom, then the side panel):
> "Read it top to bottom. Inputs come in as three kinds of data — and in our implementation those map to **MIMIC-IV and eICU**: structured tables (diagnoses, labs, prescriptions), **MIMIC-IV-Note / eICU notes** for case narrative, and a conversational interface on top. Layer 1 turns them into a structured patient profile. Layer 2 is the hybrid recommender — Transformer branch plus GNN branch, fused into a ranked candidate list and evidence. Layer 3 turns that evidence into a grounded, faithful explanation and writes a provenance log. Layer 4 is what the clinician reviews. And running down the right side, the Medical Knowledge & Rule Layer supports *every* stage continuously — it feeds the GNN, supplies rules, and grounds explanations. The arrows show data and evidence flowing down, while knowledge and rules feed in from the side."

The numbered list **1–5 below the diagram** (on the poster) restates each layer in sentences — read those if a visitor wants more depth. Note Layer 4's key power: *"can override or flag high scores when rules conflict."* Stress this — it's a safety feature.

---

### 2.7 Functional Scenario (top-right) — the story that makes it concrete
**Text:** A clinician submits a complex patient case (diagnosis, symptoms, lab trends, comorbidities, current medications, and constraints) through the conversational interface. The system extracts and validates patient factors, generates condition-appropriate medication candidates, ranks options using a hybrid Transformer-GNN model, and returns clinician-reviewable recommendations with grounded evidence, rule-check outcomes, uncertainty indicators, and provenance.

How to present: "This is one end-to-end example in words. It's the same pipeline, told as a user story — input case → extract/validate → generate candidates → rank → return reviewable, evidence-backed recommendations."

---

### 2.8 Chat mockup + Top Recommended Options table (right-center) — the demo visual
**Chat bubbles:**
- *Clinician:* "Prioritize suitable medications for this patient and explain why."
- *System:* "Top options are ranked using patient context and medical relations. Recommendation 1 is supported by glycemic profile, renal constraints, and evidence-path checks. Allergy and interaction rules passed."

**Table — "Top Recommended Options"** (this is a worked diabetes example): columns are **Rank · Medication · Rationale (Key Factors) · Uncertainty · Rule Check · Evidence & Provenance.**
| Rank | Medication | Rationale | Uncertainty | Rule Check | Evidence & Provenance |
|---|---|---|---|---|---|
| 1 | SGLT2 Inhibitor | improves glycemic control; renal protective benefit; CV risk reduction | Low | Passed (all clear) | KG Path; Guidelines, RCTs |
| 2 | DPP-4 Inhibitor | glycemic control; weight neutral; low hypoglycemia risk | Low–Med | Passed (all clear) | KG Path; Guidelines, Meta-analyses |
| 3 | GLP-1 RA | high efficacy; weight-loss benefit; CV benefit | Med | Passed (all clear) | KG Path; RCTs, Guidelines |

Footer chips: **Allergy Check: Passed · Interaction Check: Passed · Renal Dose Check: Passed.** Provenance: PubMed · Clinical Guidelines · Drug Databases · Knowledge Graph.

How to present: "This is what a clinician sees. Every row is ranked, has a plain-language rationale, an uncertainty level, a rule-check result, and provenance — the source of the evidence. Notice nothing is just 'trust me': each recommendation cites a knowledge-graph path and clinical sources."
- These drug names are illustrative of a type-2-diabetes case (SGLT2 inhibitors, DPP-4 inhibitors, GLP-1 receptor agonists are all real diabetes drug classes). If asked, say it's an illustrative example of the output format.

---

### 2.9 Model Training & Evaluation Pipeline (center-bottom flowchart) — your *implemented* work
**Flow:** *Clinical & Synthetic Patient Data* → *Cleaning, Aggregation & Structured Patient Profiles* → *Patient-Level Train / Validation / Test Split* → (into model) **Transformer Encoder (context & sequence)** and **GNN (e.g., R-GCN / Heterogeneous GNN)** → **Fusion** → **Ranked Medication Candidates** → **Ranking Metrics + Explanation Faithfulness / Groundedness.**

How to present: "This is the concrete ML pipeline behind the architecture. **Clinical data** means our **MIMIC-IV and eICU** extracts; **structured patient profiles** means harmonized tables we build from diagnoses, labs, vitals, and prescriptions. We clean and aggregate per patient/admission, split **by patient** so nobody leaks across train and test, then train the ranker — today a baseline/XGBoost path, tomorrow the full Transformer–GNN fusion. The right end shows we evaluate both *recommendation quality* and *explanation faithfulness/groundedness*. A strong design choice is **MIMIC for development, eICU for external validation** across 208 hospitals."
- **Patient-level split** is worth emphasizing: it prevents data leakage (the same patient never appears in both train and test), which makes the evaluation honest.
- **Temporal cutoffs** (for real EHR): only use clinical data documented *before* the prescription you are predicting — say this if a clinician asks about leakage.

---

### 2.10 Integrated Recommendation, Evidence, and Clinician Output (right)
**Text (paraphrased):** Clinical dialogue and EHR data are transformed into a structured patient representation, processed by the hybrid recommendation module that learns patient-context interactions and medical relationships to rank medications by relevance; for each option the system produces grounded evidence by combining model-attribution signals, knowledge-based relational support, and rule-consistency checks, flags missing or conflicting information, and presents a clinician-reviewable output with rank, confidence/score, explanation status, uncertainty cues, and provenance — so the recommendation stays transparent, auditable, and explicitly supportive of professional judgment rather than a replacement for clinical decision-making.

How to present: "This paragraph is the full pipeline in one sentence, ending on our core promise: transparent, auditable, and supportive — not a replacement." Use it as a summary if a visitor is short on time.

---

### 2.11 Conclusion (bottom)
**Text (paraphrased):** This research delivers an explainable conversational medication recommendation framework for clinical decision support, integrating clinical understanding, structured patient representation, hybrid recommendation modeling, and grounded evidence generation into one clinician-reviewable workflow. It ranks medications using patient context and medical relationships, and justifies each recommendation through multi-source evidence, rule-consistency checks, uncertainty signaling, and provenance tracking to improve transparency, trust, and practical usefulness. It supports professionals with auditable, evidence-grounded prioritization while preserving clinician authority over all final treatment decisions.

How to present: Read the last clause slowly — *"preserving clinician authority over all final treatment decisions."* That is your safety and ethics statement; it preempts the most common objection.

---

### 2.12 Research datasets — what to say when they ask "what data?"
You are **not** using a toy synthetic cohort as your primary evidence base. Point to **Section 4** and say:

> "We use **MIMIC-IV** for deep single-center hospital and ICU records — prescriptions, labs, ICD diagnoses, and **MIMIC-IV-Note** for discharge text. We use **eICU** for **208-hospital** ICU data to test external validity. The poster pipeline box is the same: clean EHR → structured profile → rank medications → explain with evidence. `DemoDataset/` in the repo was only for early pipeline debugging."

Memorize: **MIMIC = depth + notes; eICU = multi-center validation.**

---

### 2.13 Logos / footer
**ULCO** (Université du Littoral Côte d'Opale) and **LISIC** (the lab) logos; address in Calais, France. Just identify them as your university and research lab if asked.

---

## 3. Glossary — every technical term, in plain language

**EHR (Electronic Health Record)** — the digital patient chart: demographics, diagnoses, labs, medications, vitals.

**Clinical Decision Support (CDS)** — software that helps a clinician decide, without making the decision for them.

**Conversational / dialogue interface** — you type or speak the case in natural language instead of filling structured forms.

**LLM (Large Language Model)** — an AI model trained on text (e.g., GPT-style) that understands and generates language. In this system it has a *limited* job: read the case, ask for missing info, and *verbalize* evidence — it does not decide the ranking and does not invent reasons.

**Structured patient representation/profile** — the normalized, machine-readable summary of the patient (symptoms, diagnoses, labs, history, constraints) used by the model and the audit trail.

**Recommendation / recommender system** — a model that scores and ranks candidate items (here: medications) for a given context (here: a patient + condition).

**Ranking** — ordering candidates best-to-worst by a score; the top-k are the recommendations.

**Candidate generation** — first narrowing the full drug list to plausible options for the condition before ranking them (a two-stage retrieve-then-rank design).

**Transformer** — a neural network architecture built on *attention*; it's excellent at modeling how many features/tokens interact across long contexts. Here it answers: "Given all of this patient's features together, what patterns matter?"

**Attention / self-attention** — the mechanism that lets a Transformer weigh how much each input element relates to every other element.

**Encoder** — the part of a Transformer that turns inputs into a rich internal representation (an embedding).

**Embedding** — a numeric vector that represents an entity (a patient, a drug, a feature) so a model can compute with it; similar things get similar vectors.

**GNN (Graph Neural Network)** — a neural network that operates on a *graph* (nodes + edges). It propagates information between connected entities. Here it answers: "What does the medical relationship network suggest for this patient?"

**Graph** — nodes (e.g., symptoms, diagnoses, treatments, patients) connected by edges (relations like "treats", "contraindicates", "similar-to").

**GCN (Graph Convolutional Network)** — a basic GNN that updates each node from its neighbors.

**GAT (Graph Attention Network)** — a GNN that learns *how much* to weight each neighbor (attention on the graph).

**GraphSAGE** — a GNN that builds a node's representation by sampling and aggregating its local neighborhood (scales to big graphs).

**R-GCN (Relational GCN)** — a GNN that handles *multiple edge types* (e.g., symptom–diagnosis vs diagnosis–treatment). Named on your pipeline figure.

**Heterogeneous GNN** — a GNN over a graph with *multiple node types* (patients, symptoms, diagnoses, treatments, rules). Also named on your figure.

**Hybrid model / fusion** — combining the Transformer output and the GNN output into one representation used to rank. "Fusion" is just the step that merges them.

**Knowledge Graph (KG)** — a curated graph of medical facts/relations (symptom→diagnosis, diagnosis→treatment, contraindications). Provides explicit, trustworthy relational evidence.

**KG path / evidence path** — a chain of relations in the knowledge graph that connects the patient's condition to the recommended drug (e.g., diabetes → guideline → SGLT2 inhibitor). It's human-checkable evidence.

**Rule-based reasoning / clinical rules** — explicit if-then medical logic (e.g., "if eGFR < 30, avoid drug X"). Used to *check* and, if needed, *override* model scores.

**Contraindication** — a reason a drug should *not* be given to this patient (e.g., allergy, renal impairment, dangerous interaction).

**Indication** — the condition a drug is appropriate *for*.

**LIME (Local Interpretable Model-agnostic Explanations)** — an XAI method that explains *one* prediction by approximating the model locally and reporting which input features pushed the score up or down. "Local" = for this specific patient; "model-agnostic" = works on any model.

**Feature attribution** — assigning a contribution value to each input feature for a given prediction (LIME is one way to do it).

**XAI (Explainable AI)** — the field of making model decisions understandable to humans.

**Grounded explanation** — an explanation built from concrete evidence (features, KG paths, rules, scores), not free-form generated text.

**Faithfulness** — the explanation truly reflects what the model actually did (vs. a plausible-sounding but false story). A key evaluation target.

**Groundedness** — how well the explanation is supported by real evidence sources.

**Contradiction detection** — flagging when the evidence sources disagree (e.g., model score is high but a rule says contraindicated) and surfacing that uncertainty instead of hiding it.

**Uncertainty / confidence** — how sure the system is; shown so the clinician knows when to be cautious.

**Provenance** — the recorded trail: model version, input snapshot, evidence sources, and rationale lineage — enabling audit and reproducibility.

**Audit trail** — the log a reviewer can inspect to see exactly how a recommendation was produced.

**Verbalizer** — the LLM's role of turning structured evidence into readable text *without* being the source of truth.

**MIMIC-IV (Medical Information Mart for Intensive Care, version IV)** — a de-identified EHR database from Beth Israel Deaconess Medical Center (Boston), hosted on PhysioNet. Your project uses **v3.1** (`Dataset/mimiciv/3.1/`). It is the most widely used public critical-care research database.

**eICU-CRD (eICU Collaborative Research Database)** — a de-identified, **multi-center** US ICU database from the Philips eICU telehealth program. Your project uses **v2.0** (`Dataset/eicu-crd/2.0/`). It complements MIMIC by testing generalization across hospitals.

**MIMIC-IV-Note** — a separate PhysioNet module with free-text clinical notes linked to MIMIC-IV. Your project has **v2.2** (`Dataset/2.2/note/`: discharge and radiology reports).

**PhysioNet** — the platform that hosts MIMIC and eICU. Access requires CITI human-subjects training and signing a Data Use Agreement (DUA); data must not be re-identified or shared.

**De-identified / HIPAA-aligned** — patient identifiers are removed or transformed so researchers can study real clinical patterns without accessing protected health information directly.

**ICD-10** — International Classification of Diseases, 10th revision; diagnosis codes used in MIMIC (`diagnoses_icd`, `d_icd_diagnoses`).

**subject_id / hadm_id / stay_id (MIMIC)** — patient, hospital admission, and ICU stay identifiers that link tables together.

**patientunitstayid / uniquepid (eICU)** — ICU unit-stay and patient identifiers that link eICU tables.

**hosp module vs icu module (MIMIC)** — `hosp/` = hospital-wide EHR (labs, prescriptions, diagnoses); `icu/` = ICU bedside system MetaVision (chartevents, infusions, ICU stays).

**Comorbidity** — having more than one condition at the same time.

**Charlson Comorbidity Index (CCI)** — a validated score summarizing how many/serious a patient's comorbidities are; a mortality-risk proxy.

### Metric terms (for the results)
**Positive / negative label** — positive = the medication was actually prescribed for that patient-condition; negative = a plausible candidate for the condition that was *not* prescribed to this patient.

**precision@k** — of the top-k recommended drugs, what fraction were correct (actually prescribed).

**recall@k** — of all the correct drugs, what fraction appear in the top-k.

**hit_rate@k** — fraction of patient-condition cases where *at least one* correct drug is in the top-k.

**NDCG@k (Normalized Discounted Cumulative Gain)** — a ranking quality score (0–1) that rewards putting correct items *higher* in the list.

**MRR@k (Mean Reciprocal Rank)** — average of 1/(rank of the first correct item); higher means correct items appear earlier.

**ROC-AUC** — probability the model scores a random positive higher than a random negative (0.5 = chance, 1.0 = perfect).

**Average precision (AP)** — area under the precision–recall curve; summarizes ranking quality for imbalanced data.

**Data leakage** — when information from the test set sneaks into training, giving falsely high scores. Prevented here by **patient-level splitting** and by excluding leaky features.

---

## 4. Datasets: MIMIC-IV, eICU-CRD, and what to say at the poster

Your project stores data under `Dataset/`:
- `Dataset/mimiciv/3.1/` — **MIMIC-IV v3.1** (hospital + ICU modules)
- `Dataset/eicu-crd/2.0/` — **eICU Collaborative Research Database v2.0**
- `Dataset/2.2/note/` — **MIMIC-IV-Note v2.2** (discharge + radiology text)

The poster's **Model Training & Evaluation Pipeline** box says *"Clinical & Synthetic Patient Data."* When you present, say you use **real de-identified clinical data (MIMIC-IV + eICU)**; the word "synthetic" on the poster can mean *engineered feature tables* built from real EHR, not fake patients.

---

### 4.1 Why MIMIC-IV and eICU together (the "why" answer)

Use this as your main dataset justification:

> "We use two benchmark ICU/hospital databases because medication decision support must work on **real clinical complexity**, and because **one database alone is not enough to prove generalization**. MIMIC-IV is the standard single-center deep EHR: rich prescriptions, labs, ICD diagnoses, ICU stays, and a notes module for conversational understanding. eICU-CRD is multi-center — **208 US hospitals, 335 ICU units** — so we can test whether ranking and explanations hold across different sites and EHR implementations. Together they give **scale, credibility, and external validation**, which reviewers expect in medical ML and clinical decision support research."

**Six concrete reasons (pick 2–3 at the poster):**

| Reason | What to say |
|--------|-------------|
| **Real clinical signal** | Prescribing, labs, diagnoses, vitals, and allergies come from real care pathways — not simulated demographics. |
| **Community benchmarks** | MIMIC and eICU are the most cited public critical-care datasets; results are comparable to prior work. |
| **Depth (MIMIC)** | One institution, long timeline (2008–2022), hospital-wide + ICU modules, explicit prescription and administration tables. |
| **Breadth (eICU)** | Many hospitals and units; reduces risk that the model only learns one site's documentation habits. |
| **Architecture fit** | Structured tables feed the hybrid ranker; **notes** (MIMIC-IV-Note + eICU `note`) feed the LLM conversational layer; **diagnosis–medication–lab** links feed the GNN/knowledge graph. |
| **Responsible research** | Both require PhysioNet credentialing, CITI training, and a DUA — aligned with ethical use of patient data. |

**One sentence for judges:** "MIMIC teaches the model *what real prescribing looks like in depth*; eICU tests whether it *still works elsewhere*."

---

### 4.2 MIMIC-IV v3.1 — details and tables in your repo

**What it is:** De-identified EHR from **Beth Israel Deaconess Medical Center (BIDMC), Boston**, for patients seen in the emergency department, hospital, or ICU. Released on PhysioNet (October 2024 for v3.1). Modular design: hospital-wide data vs ICU bedside data are separated on purpose (different source systems).

**Official scale (MIMIC-IV v3.1 / v3.0 documentation, PhysioNet):**
- **364,627** unique patients (`subject_id`)
- **546,028** hospital admissions (`hadm_id`)
- **94,458** ICU stays (`stay_id`)
- Coverage roughly **2008–2022** (v3.x added more recent years than earlier releases)

**Your local modules (`Dataset/mimiciv/3.1/`):**

**Hospital module (`hosp/`) — main source for structured patient profile + medication ranking**

| File | Role in your system |
|------|---------------------|
| `patients.csv.gz` | Demographics (age, sex, date of birth anchor, death date if applicable) |
| `admissions.csv.gz` | Admission/discharge times, admission type, location, mortality flags |
| `diagnoses_icd.csv.gz` + `d_icd_diagnoses.csv.gz` | ICD diagnoses per admission → conditions/comorbidities for ranking & KG |
| `labevents.csv.gz` + `d_labitems.csv.gz` | Laboratory results (very large) → lab features, renal/hepatic/metabolic context |
| `prescriptions.csv.gz`, `pharmacy.csv.gz` | **Medication orders** → positive labels & candidate medications for ranking |
| `emar.csv.gz`, `emar_detail.csv.gz` | Medication administration records → adherence/timing context |
| `poe.csv.gz`, `poe_detail.csv.gz` | Provider orders → additional treatment context |
| `microbiologyevents.csv.gz` | Infections/antibiotic context |
| `omr.csv.gz` | Online medical record (e.g., height, weight from outpatient/ED) |
| `transfers.csv.gz`, `services.csv.gz` | Movement between wards/services |
| `procedures_icd.csv.gz`, `drgcodes.csv.gz`, `hcpcsevents.csv.gz` | Procedures, billing, DRG — severity/comorbidity context |

**ICU module (`icu/`) — high-frequency ICU context for critical-care CDS**

| File | Role in your system |
|------|---------------------|
| `icustays.csv.gz` | Links `subject_id` ↔ `hadm_id` ↔ `stay_id`, ICU timing |
| `chartevents.csv.gz` | Bedside vitals and assessments (largest ICU table) |
| `inputevents.csv.gz`, `outputevents.csv.gz` | Fluids, infusions, inputs/outputs |
| `ingredientevents.csv.gz` | Drug ingredients in infusions |
| `datetimeevents.csv.gz`, `procedureevents.csv.gz` | Timed events and procedures in ICU |
| `d_items.csv.gz`, `caregiver.csv.gz` | Item dictionary and staffing |

**How to present MIMIC in one breath:**
> "MIMIC-IV splits hospital EHR and ICU MetaVision data. We use hospital tables for diagnoses, labs, and prescriptions to build the structured patient profile and medication labels, and ICU tables when we need stay-level vitals and infusions for critical-care cases."

---

### 4.3 MIMIC-IV-Note v2.2 — conversational / LLM layer

**What it is:** A PhysioNet add-on with **de-identified free text** linked to MIMIC-IV admissions.

**Your local files (`Dataset/2.2/note/`):**

| File | Content |
|------|---------|
| `discharge.csv.gz` | Discharge summaries (main narrative for case understanding) |
| `discharge_detail.csv` | Metadata/structure for discharge notes |
| `radiology.csv.gz` | Radiology report text |
| `radiology_detail.csv` | Radiology report metadata |

**Why it matters for your poster:** Layer 1 (LLM conversational understanding) needs **case notes and dialogue-like narrative**. Discharge summaries are realistic input for "clinician describes the case" prototypes before a live chat UI exists. The LLM still only **structures** text; it does not decide rankings.

**Say:** "Structured MIMIC tables feed the ranker; MIMIC-IV-Note feeds the conversational understanding layer with real clinical language."

---

### 4.4 eICU-CRD v2.0 — details and tables in your repo

**What it is:** De-identified ICU database from the **Philips eICU telehealth program**, spanning **208 hospitals** and **335 ICU units** in the United States. Described in *Scientific Data* (Pollard et al., 2018). Hosted on PhysioNet.

**Official scale (eICU-CRD v2.0):**
- **200,859** ICU unit stays (patient-unit encounters)
- **139,367** unique patients
- Admissions primarily **2014–2015**
- High granularity: vitals, labs, meds, care plans, APACHE severity, diagnoses, treatments, notes

**Your local tables (`Dataset/eicu-crd/2.0/`) — 31 tables; most important for your project:**

| File | Role in your system |
|------|---------------------|
| `patient.csv.gz` | Core patient/unit stay table — anchor for all joins |
| `diagnosis.csv.gz` | Diagnoses (ICD-style coding in eICU) → conditions for ranking & KG |
| `lab.csv.gz` | Laboratory measurements (very large) |
| `medication.csv.gz` | Medications — **primary source for prescribing labels** in eICU |
| `infusionDrug.csv.gz`, `admissionDrug.csv.gz` | IV and admission drug records |
| `allergy.csv.gz` | Allergies → **rule layer** (contraindication checks) |
| `treatment.csv.gz` | Non-drug treatments |
| `note.csv.gz` | Clinical notes → conversational / LLM layer |
| `pastHistory.csv.gz` | Past medical history |
| `apachePatientResult.csv.gz`, `apacheApsVar.csv.gz`, `apachePredVar.csv.gz` | APACHE severity — illness severity & risk context |
| `vitalPeriodic.csv.gz`, `vitalAperiodic.csv.gz` | Vitals (periodic and aperiodic) |
| `nurseCharting.csv.gz`, `nurseAssessment.csv.gz`, `nurseCare.csv.gz` | Nursing documentation |
| `physicalExam.csv.gz` | Exam findings |
| `intakeOutput.csv.gz` | Fluid balance |
| `respiratoryCharting.csv.gz`, `respiratoryCare.csv.gz` | Respiratory support data |
| `carePlanGeneral.csv.gz`, `carePlanGoal.csv.gz`, … | Care plans (goals, infectious disease, EOL, etc.) |
| `microLab.csv.gz`, `customLab.csv.gz` | Microbiology and custom labs |
| `hospital.csv.gz` | Hospital-level metadata (e.g., region, teaching status) — useful for **multi-center** analysis |

**How to present eICU in one breath:**
> "eICU gives us the same clinical ingredients — diagnoses, labs, medications, allergies, notes, severity scores — but from many hospitals. We use it to validate that our patient profile and medication-ranking pipeline are not overfitted to a single center."

---

### 4.5 How both datasets map to your poster architecture

Use this table when a visitor asks "where does the data go?"

| Poster component | MIMIC-IV (+ Note) | eICU-CRD |
|------------------|-------------------|----------|
| **Inputs:** clinical dialogue, case notes, structured EHR | `discharge.csv`, `radiology.csv` (Note); `admissions`, narrative fields | `note.csv` |
| **Structured patient profile** | `patients`, `admissions`, `diagnoses_icd`, `labevents`, `omr` | `patient`, `diagnosis`, `lab`, `pastHistory` |
| **Medication candidates & labels** | `prescriptions`, `pharmacy`, `emar` | `medication`, `infusionDrug`, `admissionDrug` |
| **Hybrid ranker features** | Labs, vitals (`chartevents`), demographics, diagnosis flags | `lab`, `vitalPeriodic`/`Aperiodic`, APACHE variables |
| **GNN / knowledge graph edges** | ICD diagnosis ↔ drug ↔ lab relations from co-occurrence + ontologies | Same logic on eICU diagnosis–medication–lab relations |
| **Rule layer** (allergies, contraindications) | Derive from notes/orders where available; pair with eICU `allergy` | `allergy.csv` (explicit) |
| **Severity / uncertainty context** | Admission type, ICU stay, mortality flags | APACHE scores, care plans |
| **External validation** | Train or develop on MIMIC | Test on eICU (or vice versa) — **different hospitals** |

**Preprocessing story (aligned with your pipeline diagram):**
1. **Extract** per-admission/stay slices from each database.
2. **Harmonize** to a common schema (patient ID, admission ID, condition, medication, labs, vitals, allergies).
3. **Build** `patient + condition + medication → prescribed?` candidate rows (same idea as your `patient_condition_medication` table).
4. **Split by patient** (never by row) to avoid leakage.
5. **Train** baseline / XGBoost / future Transformer–GNN on harmonized data; report metrics **per database** and **cross-database**.

**Say at the poster:** "We are not mixing rows blindly — we harmonize schemas first, then we can train on MIMIC and report external validation on eICU."

---

### 4.6 Access, ethics, and citations (say this if asked about HIPAA / consent)

- Both databases are **de-identified** and distributed via **PhysioNet** under the **Credentialed Health Data License**.
- Access requires: **CITI "Data or Specimens Only Research"** (or equivalent) + **signed DUA** + no attempt to re-identify patients.
- Data **must not be redistributed**; keep analysis on secured machines.
- **Cite** the official resources in publications/posters:
  - **MIMIC-IV:** Johnson et al., PhysioNet / Scientific Data (check the citation block on [physionet.org/content/mimiciv/3.1](https://physionet.org/content/mimiciv/3.1/)).
  - **eICU-CRD:** Pollard et al., *Scientific Data* 5:180046 (2018); [physionet.org/content/eicu-crd/2.0](https://physionet.org/content/eicu-crd/2.0/).

**Ethics line for the poster:** "Real patient data, de-identified and used under PhysioNet agreements — decision support only, with clinician review."

---

### 4.7 MIMIC vs eICU — quick comparison for Q&A

| | **MIMIC-IV v3.1** | **eICU-CRD v2.0** |
|---|-------------------|-------------------|
| **Setting** | Single center (BIDMC, Boston) | **208 hospitals**, 335 ICU units (US) |
| **Time span** | ~2008–2022 | 2014–2015 |
| **Patients / stays** | 364k patients, 546k hospitalizations, 94k ICU stays | 139k patients, **200,859** ICU unit stays |
| **Strength** | Depth, longitudinal hospital+ICU, prescriptions at scale, notes module | **Multi-center generalization**, explicit allergy table, APACHE built-in |
| **Schema** | `subject_id`, `hadm_id`, `stay_id`; `hosp` + `icu` modules | `patientunitstayid`, `uniquepid`; flat table family |
| **Your role** | Primary development & rich feature engineering | External validation & multi-site robustness |

---


**Leakage controls (keep saying this):** exclude medication-history, outcome, and popularity features by default; **split by patient**; for MIMIC/eICU add **temporal cutoffs** (only use labs/meds documented *before* the prescription time) when building labels — critical for real EHR credibility.

---

## 5. Anticipated questions & strong answers

**Q: Isn't this just ChatGPT for medicine?**
A: No. The LLM only reads the case, requests missing data, and *verbalizes* evidence. It does not decide the ranking and does not invent justifications. Ranking is done by the hybrid Transformer–GNN model, and explanations are built from LIME, knowledge-graph paths, and clinical rules. That separation is the whole point.

**Q: Why both a Transformer *and* a GNN? Isn't one enough?**
A: They capture different things. The Transformer models how a patient's many features *interact* (context). The GNN models explicit *medical relationships* (symptom→diagnosis→treatment, patient similarity, contraindications). Fusing them gives more robust ranking and richer evidence than either alone.

**Q: What makes the explanation "faithful" rather than just plausible?**
A: We build it from the actual signals the model used (feature attributions), real KG paths, and rule checks, and we log provenance. We also detect contradictions between these sources and surface uncertainty rather than forcing a confident story. We evaluate faithfulness explicitly.

**Q: How do you prevent dangerous recommendations?**
A: The medical knowledge & rule layer runs contraindication, interaction, and dose checks, and it can **override or flag** a high model score. Plus, the clinician reviews everything — the system never prescribes.

**Q: Which datasets do you use?**
A: **MIMIC-IV v3.1** (single-center hospital + ICU EHR from BIDMC; ~365k patients, ~546k admissions) and **eICU-CRD v2.0** (multi-center US ICUs; ~139k patients, ~201k ICU stays across 208 hospitals). We also use **MIMIC-IV-Note v2.2** for discharge and radiology text to support the conversational layer. Both require PhysioNet credentialing and a data use agreement.

**Q: Why MIMIC *and* eICU — isn't one enough?**
A: MIMIC gives depth and is the standard benchmark for medication and ICU research. eICU tests **generalization across hospitals** and EHR vendors. A system that only works on one center is less convincing for clinical decision support.

**Q: Is this trained on real patients?**
A: Yes — on **de-identified** real ICU/hospital records, not synthetic patients. Identifiers are removed under PhysioNet's credentialed license. We follow the DUA: no re-identification, no redistribution, research use only. The clinician still makes the final decision.

**Q: How is this evaluated?**
A: Recommendation quality with ranking metrics (precision/recall/hit-rate/NDCG/MRR @k, ROC-AUC, average precision) on **patient-level splits**; ideally **train/develop on MIMIC, validate externally on eICU**. Explanation quality with faithfulness, groundedness, clinical understandability, and expert-oriented usefulness. For real EHR we add **temporal cutoffs** so future labs/meds do not leak into past prescriptions.

**Q: Why are the AUC/precision numbers not near 1.0?**
A: Medication choice is inherently multi-factor and noisy; also we use honest leakage controls. Any demo-cohort baseline in the repo is a **pipeline check**, not the final MIMIC/eICU score. The hybrid model, richer features, and knowledge graph are expected to improve performance; eICU external test is the harder bar.

**Q: What's the difference between this and standard XAI like LIME alone?**
A: LIME alone explains one model's features. Ours is *multi-source* and grounded: feature attribution **plus** knowledge-graph relational evidence **plus** rule validation **plus** provenance and contradiction handling — designed for clinical trust, not just feature importance.

**Q: What's done vs. what's planned?**
A: **Done:** MIMIC-IV and eICU-CRD acquired locally; ranking pipeline (preprocessing, patient-level split, baseline + XGBoost rankers, metrics). **In progress:** harmonized MIMIC→eICU feature extraction, temporal labeling for prescriptions. **Planned:** knowledge-graph integration, full hybrid Transformer–GNN, grounded explainability (LIME + KG + rules), conversational LLM on MIMIC-IV-Note / eICU notes.

**Q: Who is the user — patients or doctors?**
A: Healthcare professionals only. It's decision *support*, not a patient-facing tool, and it preserves clinician authority over every final decision.

**Q: What data does the model actually use to rank?**
A: From MIMIC: demographics (`patients`), admission context (`admissions`), **ICD diagnoses** (`diagnoses_icd`), **labs** (`labevents`), and **prescription labels** (`prescriptions`/`pharmacy`/`emar`), plus ICU vitals/infusions when needed (`chartevents`, `inputevents`). From eICU: `patient`, `diagnosis`, `lab`, `medication`, vitals, APACHE severity, and `allergy` for rule checks. Features are aggregated per patient–condition–medication candidate (labs: means, trends, abnormality flags; comorbidity/diagnosis flags; severity scores).

**Q: Can you share the data on a USB / GitHub?**
A: No. PhysioNet DUAs prohibit redistribution. Only credentialed researchers can download MIMIC and eICU for themselves. We share **code**, not the raw tables.

---

## 6. One-line cheat sheet (rapid recall before you present)

- **Pitch:** "Ranks medications and faithfully explains each one; supports the doctor, never replaces them."
- **3 goals:** Understand → Rank → Explain.
- **4 layers:** (1) LLM understanding → (2) Hybrid Transformer+GNN recommender → (3) Grounded explainability → (4) Knowledge & rules (continuous, can override).
- **Transformer =** feature interactions/context. **GNN =** medical relationships.
- **Explanation =** LIME + KG paths + rule checks + scores; LLM only verbalizes; provenance logged.
- **Output table columns:** Rank · Medication · Rationale · Uncertainty · Rule Check · Evidence/Provenance.
- **Data:** **MIMIC-IV v3.1** (364k patients, 546k admissions, 94k ICU stays, BIDMC) + **eICU v2.0** (139k patients, 201k ICU stays, **208 hospitals**) + **MIMIC-IV-Note** for text.
- **Why both:** MIMIC = depth/benchmark; eICU = multi-center external validation.
- **Key tables:** MIMIC `prescriptions`, `labevents`, `diagnoses_icd`; eICU `medication`, `lab`, `diagnosis`, `allergy`, `note`.
- **Ethics:** PhysioNet credentialed access, de-identified, DUA, no re-ID, clinician review.
- **Metrics:** Report MIMIC/eICU after harmonization; demo `metrics.json` = pipeline prototype only.
- **Core novelty:** *separating recommendation from explanation* for faithful, grounded justification.
- **Ethics line:** "preserves clinician authority over all final treatment decisions."
