# AthleteGuard AI — Prompt Log

> **Purpose:** This file documents the design decisions, prompting strategy, and key choices made in building this prototype. It is part of the project submission for transparency and reproducibility.

---

## Project Overview

**System Name:** AthleteGuard AI — Player Health & Injury Risk System  
**Type:** Educational Prototype (University Project)  
**Date:** October 2026  
**Status:** Synthetic data only — NOT a clinical tool

---

## Prompt Engineering Philosophy

This system uses **template-based briefing generation** (not a live LLM) by default, ensuring the application runs without any API key or internet connection. If a `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable is present, the system can optionally use an LLM for richer briefing text.

### Core Prompting Principles Applied:

1. **Grounded retrieval before generation:** Protocol sections are retrieved via TF-IDF similarity before any briefing is generated. The briefing explicitly cites section headings.
2. **No hallucination of clinical facts:** The template system fills only from retrieved text and computed values. It does not invent statistics or clinical evidence.
3. **Transparent uncertainty:** When retrieved evidence is sparse or thresholds conflict, the briefing flags this explicitly.
4. **Role framing:** The briefing frames output as "advisory for medical review" not "diagnosis."
5. **False positive / false negative awareness:** The briefing always acknowledges the possibility of model error and recommends human override.

---

## Synthetic Data Generation Strategy

**Goal:** Generate a plausible 500+ row dataset that avoids data leakage and has defensible synthetic labelling.

**Features generated:**
| Feature | Distribution | Rationale |
|---|---|---|
| acute_load | Normal(1500, 400) clipped [200, 4000] | Reflects realistic session RPE × duration |
| chronic_load | Normal(1300, 350) clipped [300, 3000] | 28-day baseline |
| sleep_hours | Normal(7.0, 1.2) clipped [3, 10] | Sleep science norms |
| soreness_score | Uniform-ish Beta(2,3)×10 | Right-skewed — most players have low soreness |
| fatigue_score | Similar Beta distribution | Right-skewed |
| wellness_score | Inverted Beta | Most players report good wellness |
| workload_increase_pct | Normal(5, 15) clipped [-30, 60] | Week-on-week % change |
| previous_injury | Bernoulli(0.25) | ~25% squad injury prevalence (illustrative) |

**Synthetic label logic (documented, not hidden):**
The binary `elevated_risk` label is assigned using a scoring rule that mirrors the fictional protocol thresholds. Risk points are accumulated from:
- ACWR > 1.5: +2 points
- Workload increase > 30%: +2 points  
- Sleep < 5 hours: +2 points
- Soreness ≥ 8: +1 point
- Fatigue ≥ 8: +1 point
- Wellness ≤ 3: +1 point
- Previous injury: +1 point

If score ≥ 3: elevated risk = 1. This is a **rule-based synthetic label**, not a real injury outcome.

**Intentional noise:** 8% random label flip to simulate measurement error and prevent overfitting to a perfect rule.

---

## Model Selection Rationale

**Model:** LogisticRegression (scikit-learn)
- Interpretable coefficients
- Calibrated probabilities suitable for threshold adjustment
- Fast to train on small synthetic datasets
- Appropriate for a prototype

**Pipeline:** StandardScaler → LogisticRegression
- Prevents scale-sensitive bias in logistic regression
- Clean API for deployment

**Baseline:** DummyClassifier (most_frequent strategy)
- Provides a sanity check; a real model should outperform random

**Metric Priority:**
1. ROC-AUC (overall discrimination)
2. Recall (minimizing missed high-risk cases — more important in health context)
3. Precision (minimizing false alarms — operational cost)
4. F1 (balance metric)

**Threshold choice:** Default 0.5, user-configurable. In a real system, threshold would be set by medical team based on acceptable false-positive rate.

---

## RAG / Retrieval Component

**Method:** TF-IDF (sklearn TfidfVectorizer) over protocol section texts  
**Why TF-IDF not embeddings:** No API required, fully offline, interpretable term weights  
**Retrieval:** Cosine similarity of query vector against section vectors; top-3 sections returned  
**Query construction:** Query string built from player's input values (e.g., "acute load 2500 sleep 5 soreness 8")

**Limitations acknowledged in output:**
- TF-IDF is keyword-based; semantic nuance may be missed
- Protocol has limited sections; some queries may not find highly relevant matches
- Retrieved text is not re-ranked by an LLM

---

## Briefing Template Design

The template-based briefing follows this structure:
1. **Risk Summary:** State the computed risk score and classification
2. **Triggered Signals:** List which input values crossed illustrative thresholds
3. **Retrieved Protocol References:** Cite the exact section headings and relevant text
4. **Conflicting Evidence Flags:** Note when signals contradict (e.g., high load but good sleep/wellness)
5. **Questions for Medical Team:** Pre-populated standard questions
6. **Limitations Disclosure:** Always included

---

## Known Limitations Documented

1. **Synthetic data only:** Model has no predictive validity for real players
2. **Rule-based labels:** The "ground truth" was generated by the same rules the model learns — risk of circular validation
3. **No temporal structure:** Observations are i.i.d.; real injury risk is time-series dependent
4. **Single model:** No ensemble or uncertainty quantification
5. **TF-IDF retrieval:** May miss semantically relevant sections with different terminology
6. **Session state storage:** Reviews are not persisted to a database; lost on app restart
7. **No authentication:** Not suitable for production medical use
8. **Threshold calibration:** The 0.5 default threshold is not clinically calibrated

---

## Ethical Considerations

- All data is synthetic; no real player data was used
- Explicit disclaimers throughout UI
- Human oversight required before any decision
- False positive consequence: unnecessary restriction (operational cost)
- False negative consequence: missed risk (health/safety cost) — more serious
- System must never be presented as a replacement for clinical judgment

---

*End of Prompt Log — AthleteGuard AI Educational Prototype*
