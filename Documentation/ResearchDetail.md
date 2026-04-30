# Towards an Explainable Conversational Medical Recommender System Based on LLMs

## Working Topic

**Towards an Explainable Conversational Medical Recommender System Based on LLMs: Integrating Hybrid Recommendation, Grounded Explainability, and Medical Knowledge**

## Core Idea

This project proposes a **medical explainable conversational recommender system** for clinical decision support. The system uses an LLM to understand clinical conversation and extract structured patient preferences and clinical context. A hybrid recommendation module based on **Transformer models** and **Graph Neural Network (GNN) models** generates treatment suggestions. A grounded explainability module then justifies those suggestions using local interpretable explanations, knowledge-graph evidence, and rule-based clinical reasoning.

The system is designed to assist healthcare professionals in evaluating and prioritizing treatment options for a given medical condition. It is not intended to function as a patient-facing recommendation tool or to replace clinical judgment. Instead, it supports domain experts by organizing relevant patient information, identifying clinically meaningful patterns, generating treatment suggestions, and providing transparent explanations for those suggestions.

## One-Paragraph Project Summary

This project aims to develop an explainable conversational medical recommender system for clinical decision support. An LLM-based conversational layer processes clinical dialogue, case descriptions, and structured patient data to extract key clinical factors such as symptoms, diagnoses, medical history, laboratory indicators, prior interventions, and contextual constraints. A hybrid recommendation module then combines Transformer-based modeling of long-range clinical dependencies with GNN-based modeling of semantic, structural, and similarity-based relations among symptoms, diagnoses, treatments, and patient profiles. To ensure transparency and usefulness for domain experts, a grounded explainability layer integrates LIME, medical knowledge graphs, and rule-based clinical reasoning. The overall goal is to support expert judgment with recommendations that are relevant, faithful, interpretable, and clinically understandable.

## Research Problem

Existing conversational and recommendation-based AI systems in healthcare often lack faithful, grounded, and clinically interpretable explanation mechanisms. Many systems can generate fluent recommendations, but they do not always clearly show why a treatment was suggested, which patient factors influenced the recommendation, which medical relations support it, or whether the recommendation is consistent with rule-based clinical reasoning.

This work addresses that gap by proposing a modular medical recommender system that separates **recommendation generation** from **explanation generation**. Recommendations are produced by a hybrid Transformer-GNN recommendation module, while explanations are constructed from explicit evidence derived from patient features, learned medical dependencies, similarity relations, knowledge-graph evidence, and rule-based reasoning.

## Main Research Question

**How can a medical conversational recommender generate treatment recommendations that are not only accurate, but also faithful, grounded, and clinically understandable?**

## System Objective

The objective is to build a modular system that can:

1. Understand clinical dialogue, case notes, and structured patient information.
2. Extract patient-specific symptoms, diagnosis, history, laboratory indicators, prior interventions, and contextual constraints.
3. Generate treatment suggestions using a hybrid Transformer-GNN recommendation module.
4. Explain treatment suggestions using grounded evidence rather than generic LLM-generated justification.
5. Support healthcare professionals in evaluating and prioritizing treatment options.

## Proposed System Architecture

The proposed system contains four main layers:

1. **LLM-based conversational understanding layer**
2. **Hybrid Transformer-GNN recommendation layer**
3. **Grounded explainability layer**
4. **Medical knowledge and rule-based reasoning layer**

These layers work together to convert clinical conversation and patient data into structured inputs, generate treatment recommendations, and provide evidence-based explanations for those recommendations.

## Layer 1: LLM-Based Conversational Understanding

The LLM-based conversational layer interacts with the user and processes clinical dialogue, case descriptions, and structured patient records. Its role is not to directly invent treatment recommendations, but to extract and organize clinically relevant information.

The layer extracts:

- Symptoms
- Diagnosis or suspected diagnosis
- Medical history
- Laboratory indicators
- Prior interventions
- Patient-specific constraints
- Contextual preferences
- Relevant clinical notes

The extracted information becomes the structured patient representation used by the recommendation and explanation modules.

## Layer 2: Hybrid Recommendation Module

The recommendation module is designed as a hybrid model that combines Transformer-based modeling and GNN-based modeling.

The best interpretation of this module is:

- **Transformer models** capture feature interaction and long-range dependency modeling.
- **GNN models** capture relational, semantic, structural, and similarity-based dependency modeling.
- Both representations are combined to rank or predict clinically relevant treatment options.

This hybrid design allows the system to capture both deep contextual interactions and explicit relational dependencies that are important in medical decision-making.

## Transformer Models in the Recommendation Module

Transformer models are used to model long-range dependencies across patient features and clinical history. They are useful when the system needs to understand how symptoms, diagnosis, laboratory indicators, prior interventions, and contextual constraints interact with one another.

Possible Transformer-based components include:

| Transformer Model Type | Role in the System |
| --- | --- |
| **Transformer Encoder** | Encodes structured patient features, clinical notes, and extracted dialogue information into contextual representations. |
| **Clinical Text Transformer** | Processes clinical case notes or dialogue-derived text to capture medical context. |
| **Feature Interaction Transformer** | Learns dependencies among symptoms, diagnosis, history, laboratory indicators, and treatment constraints. |
| **Cross-Attention Transformer** | Aligns patient representations with candidate treatment representations. |
| **Sequential Transformer** | Models treatment history, prior interventions, and temporal clinical patterns when patient history is available. |

The Transformer component can produce a patient-context embedding that represents the overall clinical situation. This embedding can then be combined with graph-based representations from the GNN module.

## GNN Models in the Recommendation Module

Graph Neural Network models are used to represent relations among medical entities such as symptoms, diagnoses, treatments, patient profiles, and clinical similarity links. The GNN module helps the system reason over semantic, structural, and similarity-based medical relationships.

Possible GNN-based components include:

| GNN Model Type | Role in the System |
| --- | --- |
| **Graph Convolutional Network (GCN)** | Learns representations from neighboring medical entities in a medical knowledge graph or patient-treatment graph. |
| **Graph Attention Network (GAT)** | Assigns different importance weights to related symptoms, diagnoses, treatments, or patient profiles. |
| **GraphSAGE** | Learns patient or treatment representations by aggregating information from local graph neighborhoods. |
| **Relational GNN / R-GCN** | Handles multiple relation types, such as symptom-diagnosis, diagnosis-treatment, patient-treatment, and treatment-constraint relations. |
| **Heterogeneous GNN** | Models graphs containing different node types, such as patients, symptoms, diagnoses, treatments, and rules. |

The GNN component can generate graph-aware embeddings for medical entities and patient-treatment relations. These embeddings help the system recommend treatments that are supported by relational medical evidence.

## Hybrid Transformer-GNN Fusion

The hybrid recommendation module combines the outputs of the Transformer and GNN components.

The Transformer captures:

- Long-range dependencies across clinical features
- Contextual interactions among symptoms, history, diagnosis, and constraints
- Representations of clinical dialogue and case notes
- Sequential or historical treatment patterns

The GNN captures:

- Relations among symptoms, diagnoses, treatments, and patients
- Medical knowledge graph paths
- Similarity relations among patient profiles
- Structural dependencies in patient-treatment graphs
- Rule-aware or relation-aware medical patterns

The fused representation is used to:

1. Score candidate treatments.
2. Rank treatment options.
3. Predict clinically relevant treatment suggestions.
4. Provide intermediate evidence for the explanation module.

## Layer 3: Grounded Explainability Module

The explanation module should be treated as a **grounded multi-source explanation layer**, not as a generic XAI block. Its role is to explain why a treatment was recommended using explicit evidence.

The explanation should combine evidence from:

- Conversationally extracted patient information
- Important local predictive features from LIME
- Relational evidence from the knowledge graph
- Rule-based clinical reasoning
- Recommendation-module outputs

This approach is stronger than directly asking the LLM to invent an explanation, because the final explanation is grounded in interpretable evidence.

## Explanation Pipeline

The recommended explanation pipeline is:

1. The recommendation module predicts or ranks treatments.
2. LIME identifies the locally important patient features that influenced the recommendation.
3. The knowledge graph provides relevant medical relations or evidence paths.
4. Rule-based reasoning checks clinical consistency and provides symbolic justification.
5. The LLM converts the grounded evidence into a human-readable explanation.

The final explanation should help the clinician understand:

- Why a treatment has been suggested
- Which patient factors contributed most
- What medical relations support the recommendation
- Whether the recommendation is consistent with encoded clinical rules
- What evidence was used to support the recommendation

## Layer 4: Medical Knowledge Integration

Medical knowledge integration makes the system more grounded, clinically meaningful, and transparent. The knowledge layer can represent relationships among medical entities and can support both recommendation and explanation.

The knowledge layer may include:

- Symptom-diagnosis relations
- Diagnosis-treatment relations
- Treatment-constraint relations
- Patient-treatment similarity relations
- Rule-based clinical logic
- Evidence paths used in explanations

This layer allows the system to connect learned model outputs with explicit medical reasoning structures.

## Proposed Contribution

This research contributes:

1. A **modular explainable conversational medical recommender system for clinical decision support**, designed to assist domain experts in evaluating and prioritizing treatment options.
2. A **hybrid recommendation module** that combines Transformer-based contextual modeling of long-range clinical dependencies with GNN-based relational modeling of medical entities, semantic relations, and patient-treatment similarities.
3. A clearer integration of specific **Transformer model types** and **GNN model types** within the recommendation architecture.
4. A **grounded explainability layer** that integrates local feature attribution, knowledge-graph evidence, and rule-based clinical reasoning to provide transparent and clinically meaningful justification for recommended treatments.
5. An explicit separation between **recommendation generation** and **explanation generation**, reducing the risk of fluent but ungrounded justifications.
6. An expert-oriented evaluation direction focused on recommendation quality, explanation faithfulness, trustworthiness, and usability.

## Evaluation Direction

The framework should be evaluated in terms of:

- **Recommendation quality:** how accurately and relevantly the system ranks or predicts treatment options.
- **Explanation faithfulness:** how well the explanation reflects the actual recommendation process.
- **Groundedness:** how clearly the explanation is supported by extracted patient features, knowledge-graph evidence, and rule-based reasoning.
- **Clinical understandability:** how clearly the explanation can be interpreted by healthcare professionals.
- **Expert-oriented usefulness:** how effectively the system supports professional decision-making.

## Short Version

We propose an explainable conversational medical recommender system for clinical decision support, in which an LLM extracts structured clinical information, a hybrid Transformer-GNN recommendation module identifies relevant treatment options, and a grounded explainability layer justifies recommendations using LIME, knowledge graphs, and rule-based reasoning to assist domain experts.

## Recommended Working Version

A modular explainable conversational medical recommender system for clinical decision support that uses LLM-based clinical information understanding, a hybrid Transformer-GNN recommendation module, and a grounded explanation layer based on LIME, knowledge graphs, and rule-based reasoning to generate transparent and expert-assisting treatment recommendations.
