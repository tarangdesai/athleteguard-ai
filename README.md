# AthleteGuard AI — Player Health & Injury Risk System

> **⛔ EDUCATIONAL PROTOTYPE — NOT MEDICAL ADVICE**
> Synthetic data only. Injury-risk scores are NOT diagnoses.
> All decisions require qualified medical review.

---

## Overview

**AthleteGuard AI** is a Python Streamlit web application built as a university project to demonstrate how machine learning and document-grounded retrieval (RAG) can be combined to support — not replace — sports medicine decisions.

The system:
- Generates a fully synthetic player-wellness dataset and trains a Logistic Regression model
- Retrieves relevant sections from a fictional club protocol using TF-IDF similarity
- Generates a structured advisory briefing citing retrieved protocol sections
- Provides a medical reviewer checkpoint with explicit confirmation requirements
- Includes a synthetic squad overview, model evaluation, business-case comparison, and limitations analysis

**Everything in this system is fictional or synthetic.** No real player data, real injuries, or validated clinical thresholds are used.

---

## Quick Start

### 1. Install dependencies

```bash
cd /Users/tarang/Desktop/player-health-ai

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

### 3. Run the tests

```bash
python -m pytest tests.py -v
# or
python tests.py
```

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application — all UI, ML, and RAG logic |
| `requirements.txt` | Python dependencies (no paid APIs required) |
| `club_protocol.md` | Fictional club medical & training-load protocol (RAG knowledge base) |
| `prompt_log.md` | Design decisions, prompting strategy, data generation documentation |
| `tests.py` | Unit tests for data generation, model, predictions, retrieval, and signals |
| `README.md` | This file |

---

## Application Features

### 🏥 Player Analysis Tab
- Input via sidebar: acute load, chronic load, sleep, soreness, fatigue, wellness, workload change, previous injury indicator
- Three pre-configured demo cases: **Low Risk**, **High Risk**, **Borderline / Conflicting**
- Risk gauge, radar chart, and key metrics panel
- Risk signals detected against fictional protocol thresholds
- Conflicting evidence detection (e.g. high load but good wellness)
- TF-IDF retrieval of top-3 relevant protocol sections with similarity scores
- Advisory briefing citing retrieved section headings (template-based — not an LLM)

### 👥 Squad Overview Tab
- Synthetic 22-player squad snapshot with status flags (🟢 Normal / 🟡 Monitor / 🔴 Flag)
- Acute vs Chronic Load scatter plot

### 🤖 Model Evaluation Tab
- Precision, Recall, F1, ROC-AUC for LogisticRegression vs DummyClassifier
- Confusion matrices with business interpretation of false positives / negatives
- Feature coefficient chart
- Training data distributions

### 📋 Medical Reviewer Tab
- Reviewer name/role, decision (Pending / Clear / Modify / Hold), clinical notes
- Mandatory confirmation checkbox before saving
- Session-state review history with clear disclosure of non-persistence

### 💼 Business Case Tab
- Illustrative 3-year Build vs Buy vs Rent cost comparison
- All figures explicitly marked as hypothetical assumptions
- Pre-deployment validation checklist

### ⚠️ Limitations Tab
- Synthetic data, circular label logic, temporal structure absence
- Bias, false-alarm fatigue, hallucination risks, privacy, human oversight
- References and acknowledgements

---

## Architecture

```
app.py
├── Synthetic Data Generation     (numpy, rule-based labels + noise)
├── Model Training                (sklearn Pipeline: StandardScaler → LogisticRegression)
├── Baseline Comparison           (DummyClassifier)
├── Protocol Loading & Indexing   (TfidfVectorizer on club_protocol.md sections)
├── Protocol Retrieval            (cosine_similarity, top-k sections)
├── Signal Detection              (threshold-based rules vs fictional protocol)
├── Briefing Generation           (template-based — no LLM required)
└── Streamlit UI                  (6 tabs, sidebar, plotly charts)
```

---

## Optional LLM Integration

If an `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable is set, the system could be extended to use an LLM for richer briefing text. The current version uses a transparent template-based approach that is fully offline and never pretends to be an LLM.

```bash
# Optional — only if extending the app to use an LLM
export OPENAI_API_KEY="sk-..."
```

---

## Ethical Notices

1. **Not for real use.** This system must not be used for actual player health monitoring.
2. **Synthetic data only.** No real player data was used to build or train this system.
3. **Labels are fabricated.** The `elevated_risk` label is assigned by a documented rule, not by real injury outcomes.
4. **Model metrics are illustrative.** Reported precision/recall/AUC are computed on synthetic holdout data.
5. **Protocol thresholds are fictional.** All numbers in `club_protocol.md` are hypothetical assumptions.
6. **Human oversight required.** No automated decision must be acted upon without qualified medical review.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `scikit-learn` | ML model, preprocessing, metrics, TF-IDF |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `matplotlib`, `seaborn` | (available for additional plots) |

No paid API or external service is required.

---

## Test Coverage

| Test Class | What It Tests |
|---|---|
| `TestSyntheticData` | Row count, columns, nulls, binary target, prevalence, ranges, reproducibility |
| `TestModelTraining` | Pipeline steps, train/test split, prediction types, probability validity, AUC > baseline |
| `TestPredictPlayer` | Single-player prediction, probability range, high/low risk inputs, threshold sensitivity |
| `TestDemoCases` | Low-risk demo < 0.5, high-risk ≥ 0.5, borderline between low and high |
| `TestSignalDetection` | ACWR, sleep, fatigue, previous injury, workload spike signal detection |
| `TestProtocolRetrieval` | TF-IDF retrieval quality, result count, score range |
| `TestProtocolFileExists` | Protocol file presence and minimum length |

---

*AthleteGuard AI — University Project 2026 — Educational Prototype*
