A medical explainable conversational recommender system in which an LLM extracts structured patient preferences and clinical context, a hybrid recommendation module based on Transformer and GNN modeling generates treatment suggestions, and an explainability module grounds recommendations using local interpretable explanations, knowledge-graph evidence, and rule-based clinical reasoning.

Working Topic

Towards an Explainable Conversational Medical Recommender System Based on LLMs: Integrating Hybrid Recommendation, Grounded Explainability, and Medical Knowledge

This project aims to develop an Explainable Conversational Medical Recommender System for clinical decision support, designed to assist healthcare professionals in evaluating and prioritizing treatment options for a given medical condition. Instead of functioning as a patient-facing recommendation tool, the system is intended to support the domain expert by organizing relevant patient information, identifying clinically meaningful patterns, generating treatment suggestions, and providing transparent, evidence-grounded explanations for those suggestions.

The system should combine:

LLM-based conversation understanding to interact with users and extract patient-specific information
Hybrid recommendation module to generate treatment suggestions
Explainability module to justify why a treatment is recommended
Medical knowledge integration to make explanations more grounded, clinically meaningful, and transparent

We propose an explainable conversational medical recommender system in which treatment recommendation is separated from explanation generation, and explanations are grounded in explicit evidence derived from patient features, learned medical dependencies, similarity relations, and knowledge-based reasoning.

How can a medical conversational recommender generate treatment recommendations that are not only accurate, but also faithful, grounded, and clinically understandable?


The system uses an LLM-based conversational and reasoning layer to process clinical dialogue, case notes, and structured patient data, extracting key factors such as symptoms, diagnosis, medical history, laboratory indicators, prior interventions, and contextual constraints. A hybrid recommendation module then combines Transformer-based modeling of long-range dependencies across clinical features with Graph Neural Network (GNN)-based modeling of semantic, structural, and similarity-based relations among medical entities such as symptoms, diagnoses, treatments, and patient profiles. This hybrid design enables the system to capture both deep contextual interactions and explicit relational dependencies that are highly relevant in medical decision-making.

To ensure that recommendations are trustworthy and clinically useful, the system incorporates a grounded explainability module that does not merely generate fluent justifications, but instead constructs explanations from explicit evidence. This layer integrates LIME for local feature-level interpretability, knowledge graphs for clinically meaningful relational evidence, and rule-based reasoning for domain constraints, expert logic, and decision-support validation. The final explanation is then presented in a form that helps the clinician understand why a treatment has been suggested, which patient factors contributed most, what medical relations support it, and whether the recommendation is consistent with encoded clinical rules.

We propose an explainable conversational medical recommender system for clinical decision support, in which an LLM extracts structured clinical information, a hybrid Transformer-GNN recommendation module identifies relevant treatment options, and a grounded explainability layer justifies recommendations using LIME, knowledge graphs, and rule-based reasoning, with the primary goal of assisting domain experts rather than replacing their judgment.

The recommendation module is designed as a hybrid model combining a Transformer-based encoder and a Graph Neural Network. The Transformer captures long-range contextual dependencies across patient features and clinical history, while the GNN captures semantic, structural, and similarity-based relations among medical entities such as symptoms, diagnoses, and treatments. Their combined representations are used to rank or predict clinically relevant treatment options.
Hybrid recommendation module

The best interpretation is:

Transformer for feature interaction and long-range dependency modeling
GNN for relational and semantic dependency modeling
combine both into a hybrid treatment recommendation module

The explanation module should now be treated as a grounded multi-source explanation layer, not just a generic XAI block.

The explanation should combine evidence from:

conversationally extracted patient information
important local predictive features from LIME
relational evidence from the knowledge graph
rule-based clinical reasoning
recommendation-module outputs

Then the final explanation can be written in natural language.

Recommended explanation pipeline
The recommendation module predicts or ranks treatments.
LIME identifies the locally important patient features.
The knowledge graph provides relevant medical relations or evidence paths.
Rule-based reasoning checks clinical consistency or provides symbolic justification.
The LLM converts this grounded evidence into a human-readable explanation.

This is much better than directly asking the LLM to invent an explanation.

Existing conversational and recommendation-based AI systems in healthcare often lack faithful, grounded, and clinically interpretable explanation mechanisms. This work addresses that gap by proposing a hybrid medical recommender that combines conversational preference and patient-state understanding with Transformer- and GNN-based recommendation, and explains its outputs through LIME, knowledge-graph evidence, and rule-based reasoning.


1. We propose a **modular explainable conversational medical recommender system for clinical decision support**, designed to assist domain experts in evaluating and prioritizing treatment options.

2. We design a **hybrid recommendation module** that combines **Transformer-based contextual modeling** of long-range clinical dependencies with **GNN-based relational modeling** of medical entities, semantic relations, and patient-treatment similarities.

3. We introduce a **grounded explainability layer** that integrates **local feature attribution**, **knowledge-graph evidence**, and **rule-based clinical reasoning** to provide transparent and clinically meaningful justification for recommended treatments.

4. We explicitly separate **recommendation generation** from **explanation generation**, improving interpretability and reducing the risk of fluent but ungrounded justifications.

5. We evaluate the framework in terms of **recommendation quality**, **explanation faithfulness**, and **expert-oriented trustworthiness and usability**, focusing on how effectively the system supports professional decision-making.



> This project aims to develop an explainable conversational medical recommender system for clinical decision support. Rather than serving as a patient-facing recommendation tool, the system is designed to assist healthcare professionals by organizing patient information, identifying clinically relevant patterns, generating treatment suggestions, and providing grounded explanations for those suggestions. An LLM-based conversational layer processes dialogue, case descriptions, and structured patient data to extract key clinical factors such as symptoms, diagnoses, history, and contextual constraints. A hybrid recommendation module then combines Transformer-based modeling of long-range clinical dependencies with Graph Neural Network modeling of semantic, structural, and similarity-based relations among medical entities to identify relevant treatment options. To ensure transparency and usefulness for the domain expert, the recommendation process is complemented by a grounded explainability layer that integrates LIME, medical knowledge graphs, and rule-based clinical reasoning. The overall goal is to support expert judgment with recommendations that are not only relevant, but also faithful, interpretable, and clinically understandable.


> We propose an explainable conversational medical recommender system for clinical decision support, in which an LLM extracts structured clinical information, a hybrid Transformer-GNN recommendation module identifies relevant treatment options, and a grounded explainability layer justifies recommendations using LIME, knowledge graphs, and rule-based reasoning to assist domain experts.


> A modular explainable conversational medical recommender system for clinical decision support that uses LLM-based clinical information understanding, a hybrid Transformer-GNN recommendation module, and a grounded explanation layer based on LIME, knowledge graphs, and rule-based reasoning to generate transparent and expert-assisting treatment recommendations.
