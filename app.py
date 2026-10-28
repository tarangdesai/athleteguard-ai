"""
AthleteGuard AI — Player Health & Injury Risk System
======================================================
EDUCATIONAL PROTOTYPE — NOT MEDICAL ADVICE.
Synthetic data only. Injury-risk scores are NOT diagnoses.
All decisions require qualified medical review.
"""

import os
import re
import math
import json
import warnings
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AthleteGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 1. GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1529 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #ccd6f6 !important; }

/* ── Banner ── */
.banner {
    background: linear-gradient(90deg, #0d47a1 0%, #1565c0 40%, #0277bd 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    border: 1px solid #1e88e5;
    box-shadow: 0 4px 24px rgba(13,71,161,0.4);
}
.banner h1 { color: #fff; font-size: 2rem; font-weight: 800; margin: 0; }
.banner p  { color: #90caf9; font-size: 0.9rem; margin: 4px 0 0; }

/* ── Warning banner ── */
.warning-banner {
    background: linear-gradient(90deg, #7f0000 0%, #c62828 100%);
    border-radius: 10px;
    padding: 14px 20px;
    margin: 12px 0;
    border: 1px solid #ef5350;
    color: #ffcdd2;
    font-weight: 600;
    font-size: 0.88rem;
    text-align: center;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #112240 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 16px;
    margin: 6px 0;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #1e88e5; }
.metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; margin: 2px 0; }
.metric-sub   { font-size: 0.78rem; color: #94a3b8; }

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #90caf9;
    border-left: 3px solid #1e88e5;
    padding-left: 10px;
    margin: 20px 0 12px;
}

/* ── Risk badges ── */
.risk-high {
    background: linear-gradient(90deg,#7f0000,#c62828);
    color: #ffcdd2; padding: 6px 16px; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; display: inline-block;
}
.risk-low {
    background: linear-gradient(90deg,#1b5e20,#388e3c);
    color: #c8e6c9; padding: 6px 16px; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; display: inline-block;
}
.risk-borderline {
    background: linear-gradient(90deg,#4a3000,#f57f17);
    color: #fff3e0; padding: 6px 16px; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; display: inline-block;
}

/* ── Protocol excerpt box ── */
.protocol-box {
    background: #0a1929;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #29b6f6;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
    font-size: 0.84rem;
    color: #b0bec5;
}
.protocol-box strong { color: #29b6f6; }

/* ── Briefing box ── */
.briefing-box {
    background: linear-gradient(135deg,#0a1929 0%,#0d2137 100%);
    border: 1px solid #0277bd;
    border-radius: 10px;
    padding: 20px;
    font-size: 0.88rem;
    color: #cfd8dc;
    line-height: 1.7;
}

/* ── Review card ── */
.review-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 6px 0;
    font-size: 0.84rem;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-weight: 600;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1e88e5 !important;
    border-bottom: 2px solid #1e88e5 !important;
}

/* ── Inputs ── */
.stSlider > div > div > div > div { background: #1e88e5 !important; }
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea > div > div > textarea {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(90deg,#0d47a1,#1565c0) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg,#1565c0,#1976d2) !important;
    box-shadow: 0 4px 12px rgba(21,101,192,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 2. CONSTANTS & DEMO CASES
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "acute_load", "chronic_load", "sleep_hours",
    "soreness_score", "fatigue_score", "wellness_score",
    "workload_increase_pct", "previous_injury",
]

DEMO_CASES = {
    "Low Risk — Player A (Fit & Rested)": {
        "player_id": "PLY-007",
        "acute_load": 1200.0,
        "chronic_load": 1400.0,
        "sleep_hours": 8.5,
        "soreness_score": 2,
        "fatigue_score": 2,
        "wellness_score": 8,
        "workload_increase_pct": 5.0,
        "previous_injury": 0,
    },
    "High Risk — Player B (Overloaded)": {
        "player_id": "PLY-023",
        "acute_load": 3200.0,
        "chronic_load": 1300.0,
        "sleep_hours": 4.5,
        "soreness_score": 9,
        "fatigue_score": 9,
        "wellness_score": 2,
        "workload_increase_pct": 45.0,
        "previous_injury": 1,
    },
    "Borderline — Player C (Mixed Signals)": {
        "player_id": "PLY-041",
        "acute_load": 1900.0,
        "chronic_load": 1450.0,
        "sleep_hours": 6.0,
        "soreness_score": 6,
        "fatigue_score": 5,
        "wellness_score": 5,
        "workload_increase_pct": 22.0,
        "previous_injury": 0,
    },
}

# ─────────────────────────────────────────────────────────────
# 3. SYNTHETIC DATA GENERATION  (documented label logic)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generate reproducible synthetic training data.

    SYNTHETIC DATA NOTICE:
    All observations are randomly generated. The binary target
    'elevated_risk' is assigned by a rule-based scoring system
    that mirrors the fictional club protocol thresholds.
    This is NOT derived from real player injuries.

    Label logic (points accumulated):
      ACWR > 1.5            → +2
      workload_increase > 30%→ +2
      sleep < 5 hrs         → +2
      soreness ≥ 8          → +1
      fatigue ≥ 8           → +1
      wellness ≤ 3          → +1
      previous_injury       → +1
      elevated_risk = 1 if total ≥ 3, else 0
    8% random label noise added intentionally.
    """
    rng = np.random.RandomState(seed)

    acute_load = rng.normal(1500, 400, n).clip(200, 4000)
    chronic_load = rng.normal(1300, 350, n).clip(300, 3000)
    sleep_hours = rng.normal(7.0, 1.2, n).clip(3.0, 10.0)
    soreness_score = (rng.beta(2, 3, n) * 10).clip(0, 10).round(1)
    fatigue_score = (rng.beta(2, 3, n) * 10).clip(0, 10).round(1)
    wellness_score = (rng.beta(3, 2, n) * 10).clip(0, 10).round(1)
    workload_increase_pct = rng.normal(5, 15, n).clip(-30, 60)
    previous_injury = (rng.uniform(0, 1, n) < 0.25).astype(int)

    acwr = np.where(chronic_load > 0, acute_load / chronic_load, 1.0)

    # Rule-based synthetic label
    risk_points = (
        (acwr > 1.5).astype(int) * 2
        + (workload_increase_pct > 30).astype(int) * 2
        + (sleep_hours < 5).astype(int) * 2
        + (soreness_score >= 8).astype(int)
        + (fatigue_score >= 8).astype(int)
        + (wellness_score <= 3).astype(int)
        + previous_injury
    )
    elevated_risk = (risk_points >= 3).astype(int)

    # Intentional 8% label noise
    noise_mask = rng.uniform(0, 1, n) < 0.08
    elevated_risk[noise_mask] = 1 - elevated_risk[noise_mask]

    df = pd.DataFrame({
        "player_id": [f"SYN-{i:03d}" for i in range(n)],
        "acute_load": acute_load.round(1),
        "chronic_load": chronic_load.round(1),
        "sleep_hours": sleep_hours.round(2),
        "soreness_score": soreness_score,
        "fatigue_score": fatigue_score,
        "wellness_score": wellness_score,
        "workload_increase_pct": workload_increase_pct.round(2),
        "previous_injury": previous_injury,
        "acwr": acwr.round(3),
        "elevated_risk": elevated_risk,
    })
    return df

# ─────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(seed: int = 42):
    """Train LogisticRegression pipeline and DummyClassifier baseline."""
    df = generate_synthetic_data(seed=seed)
    X = df[FEATURE_COLS]
    y = df["elevated_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Main model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    # Dummy baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
    dummy.fit(X_train, y_train)

    return pipeline, dummy, X_train, X_test, y_train, y_test, df

# ─────────────────────────────────────────────────────────────
# 5. MODEL EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────
def compute_metrics(model, X_test, y_test, threshold: float = 0.5, label: str = "Model"):
    """Return a dict of evaluation metrics."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = float("nan")
    else:
        y_pred = model.predict(X_test)
        auc = float("nan")

    return {
        "label": label,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": auc,
        "cm": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }

def predict_player(pipeline, inputs: dict, threshold: float = 0.5):
    """Return probability and classification for a single player."""
    row = pd.DataFrame([{col: inputs[col] for col in FEATURE_COLS}])
    proba = pipeline.predict_proba(row)[0, 1]
    label = "ELEVATED RISK" if proba >= threshold else "LOW RISK"
    return proba, label

def get_feature_importance(pipeline) -> dict:
    """Extract logistic regression coefficients as importance scores."""
    coef = pipeline.named_steps["clf"].coef_[0]
    return dict(zip(FEATURE_COLS, coef))

# ─────────────────────────────────────────────────────────────
# 6. RAG — PROTOCOL RETRIEVAL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_protocol_sections(filepath: str = "club_protocol.md"):
    """
    Parse club_protocol.md into sections and build a TF-IDF index.
    Returns (sections_dict, vectorizer, matrix).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return {}, None, None

    # Split on level-2 and level-3 headings
    pattern = re.compile(r"(#{2,3} .+)", re.MULTILINE)
    parts = pattern.split(text)

    sections = {}
    current_heading = "Introduction"
    buffer = []
    for part in parts:
        if pattern.match(part):
            if buffer:
                sections[current_heading] = " ".join(buffer).strip()
            current_heading = part.strip("# ").strip()
            buffer = []
        else:
            buffer.append(part.strip())
    if buffer:
        sections[current_heading] = " ".join(buffer).strip()

    # Remove empty sections
    sections = {k: v for k, v in sections.items() if len(v) > 40}

    if not sections:
        return {}, None, None

    headings = list(sections.keys())
    texts = [sections[h] for h in headings]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    return sections, vectorizer, matrix, headings

def retrieve_protocol_sections(query: str, top_k: int = 3):
    """Return top-k protocol sections most relevant to the query."""
    result = load_protocol_sections()
    if len(result) == 4:
        sections, vectorizer, matrix, headings = result
    else:
        return []

    if vectorizer is None or not sections:
        return []

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    retrieved = []
    for idx in top_idx:
        heading = headings[idx]
        score = float(sims[idx])
        excerpt = sections[heading][:600] + ("…" if len(sections[heading]) > 600 else "")
        retrieved.append({"heading": heading, "score": score, "excerpt": excerpt})
    return retrieved

def build_query(inputs: dict) -> str:
    """Construct a protocol search query from player inputs."""
    parts = []
    acwr = inputs["acute_load"] / max(inputs["chronic_load"], 1)
    parts.append(f"acute load {inputs['acute_load']:.0f}")
    parts.append(f"chronic load {inputs['chronic_load']:.0f}")
    parts.append(f"acwr {acwr:.2f}")
    parts.append(f"sleep {inputs['sleep_hours']:.1f} hours")
    parts.append(f"soreness {inputs['soreness_score']}")
    parts.append(f"fatigue {inputs['fatigue_score']}")
    parts.append(f"wellness {inputs['wellness_score']}")
    parts.append(f"workload increase {inputs['workload_increase_pct']:.1f} percent")
    if inputs["previous_injury"]:
        parts.append("previous injury recent")
    return " ".join(parts)

# ─────────────────────────────────────────────────────────────
# 7. BRIEFING GENERATOR (template-based, no LLM required)
# ─────────────────────────────────────────────────────────────
def identify_signals(inputs: dict, proba: float, threshold: float) -> list:
    """Return a list of triggered risk signal descriptions."""
    signals = []
    acwr = inputs["acute_load"] / max(inputs["chronic_load"], 1)

    if acwr > 1.5:
        signals.append(f"⚠️ ACWR = {acwr:.2f} (>1.5 — high load spike signal per protocol §2.3)")
    elif acwr > 1.3:
        signals.append(f"🟡 ACWR = {acwr:.2f} (1.3–1.5 — elevated signal per protocol §2.3)")

    if inputs["workload_increase_pct"] > 30:
        signals.append(f"⚠️ Workload increase = {inputs['workload_increase_pct']:.1f}% (>30% — medical review trigger per protocol §3.1)")
    elif inputs["workload_increase_pct"] > 20:
        signals.append(f"🟡 Workload increase = {inputs['workload_increase_pct']:.1f}% (21–30% — mandatory screening per protocol §3.1)")

    if inputs["sleep_hours"] < 4:
        signals.append(f"⚠️ Sleep = {inputs['sleep_hours']:.1f}h (<4h — critical sleep deficit per protocol §4.2)")
    elif inputs["sleep_hours"] < 6:
        signals.append(f"🟡 Sleep = {inputs['sleep_hours']:.1f}h (4–5.9h — poor sleep flag per protocol §4.2)")

    if inputs["soreness_score"] >= 9:
        signals.append(f"⚠️ Soreness = {inputs['soreness_score']}/10 (≥9 — severe; mandatory assessment per protocol §5.2)")
    elif inputs["soreness_score"] >= 7:
        signals.append(f"🟡 Soreness = {inputs['soreness_score']}/10 (7–8 — high; modified training per protocol §5.2)")

    if inputs["fatigue_score"] >= 8:
        signals.append(f"⚠️ Fatigue = {inputs['fatigue_score']}/10 (≥8 — severe fatigue per protocol §5.3)")
    elif inputs["fatigue_score"] >= 6:
        signals.append(f"🟡 Fatigue = {inputs['fatigue_score']}/10 (6–7 — moderate fatigue per protocol §5.3)")

    if inputs["wellness_score"] <= 3:
        signals.append(f"⚠️ Wellness = {inputs['wellness_score']}/10 (≤3 — poor; mandatory medical review per protocol §5.4)")
    elif inputs["wellness_score"] <= 6:
        signals.append(f"🟡 Wellness = {inputs['wellness_score']}/10 (4–6 — borderline per protocol §5.4)")

    if inputs["previous_injury"]:
        signals.append("🔴 Previous injury flag active — enhanced monitoring applies per protocol §6.1")

    # Composite flag
    composite_count = sum([
        inputs["soreness_score"] >= 6,
        inputs["fatigue_score"] >= 6,
        inputs["wellness_score"] <= 4,
        inputs["sleep_hours"] < 6,
    ])
    if composite_count >= 2:
        signals.append(f"🔴 COMPOSITE WELLNESS FLAG — {composite_count}/4 dimensions triggered (protocol §5.5)")

    if not signals:
        signals.append("✅ No individual threshold breaches detected in current inputs.")

    return signals

def detect_conflicts(inputs: dict, signals: list) -> list:
    """Identify potentially conflicting evidence."""
    conflicts = []
    acwr = inputs["acute_load"] / max(inputs["chronic_load"], 1)
    high_load = acwr > 1.3 or inputs["workload_increase_pct"] > 20
    good_wellness = (
        inputs["wellness_score"] >= 7
        and inputs["fatigue_score"] <= 3
        and inputs["soreness_score"] <= 3
        and inputs["sleep_hours"] >= 7
    )
    if high_load and good_wellness:
        conflicts.append(
            "⚡ CONFLICTING EVIDENCE: Load metrics are elevated but wellness/fatigue/sleep "
            "indicators are in the healthy range. Player may be adapting well or subjective "
            "data may be underreported. Requires clinical judgment."
        )
    low_load = acwr < 1.0 and inputs["workload_increase_pct"] < 10
    poor_wellness = inputs["wellness_score"] <= 4 or inputs["fatigue_score"] >= 7
    if low_load and poor_wellness:
        conflicts.append(
            "⚡ CONFLICTING EVIDENCE: Training load is low/normal but wellness indicators "
            "are poor. Consider non-training stressors: illness, sleep disorder, psychological "
            "factors. Protocol §4.3 & §5.5 may apply."
        )
    return conflicts

def generate_briefing(
    player_id: str,
    inputs: dict,
    proba: float,
    classification: str,
    threshold: float,
    retrieved_sections: list,
) -> str:
    """
    Generate a structured medical briefing using template + retrieved protocol text.

    NOTE: This is a TEMPLATE-BASED briefing, NOT an LLM output.
    It cites only retrieved protocol text; it does not generate
    clinical claims beyond what the protocol contains.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    acwr = inputs["acute_load"] / max(inputs["chronic_load"], 1)
    signals = identify_signals(inputs, proba, threshold)
    conflicts = detect_conflicts(inputs, signals)

    risk_color = "ELEVATED" if classification == "ELEVATED RISK" else "LOW"

    retrieved_text = ""
    if retrieved_sections:
        for sec in retrieved_sections:
            heading = sec["heading"]
            excerpt = sec["excerpt"][:400]
            retrieved_text += f"\n  — §{heading}: \"{excerpt}...\"\n"
    else:
        retrieved_text = "\n  [No protocol sections retrieved — evidence insufficient]\n"

    signals_text = "\n".join(f"  {s}" for s in signals)
    conflicts_text = ("\n".join(f"  {c}" for c in conflicts)) if conflicts else "  None identified."

    questions = [
        "Has the player self-reported any acute pain or discomfort?",
        "Is the reported sleep deficit due to a known temporary factor (travel, match schedule)?",
        "Have load targets been reviewed by the strength & conditioning team this week?",
        "Is the workload increase planned (pre-match preparation) or unplanned?",
        "For previous injury holders: has the graduated return-to-training protocol been followed?",
        "Are any non-training stressors (personal, travel, illness) known to the medical team?",
    ]
    questions_text = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))

    briefing = f"""
╔══════════════════════════════════════════════════════════════════╗
  ATHLETEGUARD AI — ADVISORY BRIEFING (TEMPLATE-BASED, NOT LLM)
  Generated: {now} | Player: {player_id}
╚══════════════════════════════════════════════════════════════════╝

⚠️  NOTICE: This is a SYNTHETIC DATA educational prototype.
    Risk scores are NOT diagnoses. All findings require qualified
    medical review before any training or clinical decision is made.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. RISK SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model risk probability : {proba:.1%}  (threshold: {threshold:.0%})
Risk classification    : {classification}
ACWR (computed)        : {acwr:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. TRIGGERED RISK SIGNALS (vs. fictional protocol thresholds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{signals_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. CONFLICTING EVIDENCE FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conflicts_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. RETRIEVED PROTOCOL EXCERPTS (source: club_protocol.md)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{retrieved_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. SUGGESTED QUESTIONS FOR MEDICAL REVIEW TEAM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{questions_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. MODEL & SYSTEM LIMITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • False positives: player may be flagged when actually fit — risk of
    unnecessary restriction; always verify with clinical assessment.
  • False negatives: player may not be flagged despite real risk — system
    cannot replace clinical observation and player reporting.
  • Model trained on SYNTHETIC data only — no real-world validation.
  • TF-IDF retrieval is keyword-based; may miss semantically relevant sections.
  • All protocol thresholds are FICTIONAL assumptions, not clinical standards.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⛔  DO NOT ACT ON THIS BRIEFING WITHOUT QUALIFIED MEDICAL REVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return briefing

# ─────────────────────────────────────────────────────────────
# 8. SQUAD OVERVIEW DATA (synthetic)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_squad_overview(seed: int = 99):
    """Generate a small synthetic squad snapshot for display."""
    rng = np.random.RandomState(seed)
    n = 22
    positions = ["GK", "DEF", "DEF", "DEF", "DEF", "DEF",
                 "MID", "MID", "MID", "MID", "MID", "MID",
                 "FWD", "FWD", "FWD", "FWD",
                 "GK", "DEF", "MID", "MID", "FWD", "FWD"]
    names = [
        "J. Santos", "M. Kovač", "T. Okafor", "R. Fischer", "A. Mbeki",
        "L. Pereira", "D. Müller", "C. Nkosi", "F. Hernandez", "B. Walsh",
        "G. Russo", "N. Patel", "K. Yamamoto", "E. Diallo", "S. Bergström",
        "P. O'Brien", "V. Andersen", "H. Ferreira", "I. Volkov", "J. Kim",
        "A. Nakamura", "T. Adeyemi",
    ]
    acute = rng.normal(1500, 400, n).clip(200, 3500).round(0)
    chronic = rng.normal(1300, 300, n).clip(300, 2800).round(0)
    sleep = rng.normal(7.0, 1.1, n).clip(3.5, 10.0).round(1)
    soreness = (rng.beta(2, 3, n) * 10).clip(0, 10).round(1)
    fatigue = (rng.beta(2, 3, n) * 10).clip(0, 10).round(1)
    wellness = (rng.beta(3, 2, n) * 10).clip(0, 10).round(1)
    acwr = (acute / chronic).round(2)

    risk_pts = (
        (acwr > 1.5).astype(int) * 2
        + (sleep < 5).astype(int) * 2
        + (soreness >= 8).astype(int)
        + (fatigue >= 8).astype(int)
        + (wellness <= 3).astype(int)
    )
    status = pd.cut(risk_pts, bins=[-1, 1, 2, 100],
                    labels=["🟢 Normal", "🟡 Monitor", "🔴 Flag"])

    df = pd.DataFrame({
        "Player": names[:n],
        "Pos": positions[:n],
        "Acute Load": acute,
        "Chronic Load": chronic,
        "ACWR": acwr,
        "Sleep (h)": sleep,
        "Soreness": soreness,
        "Fatigue": fatigue,
        "Wellness": wellness,
        "Status": status,
    })
    return df

# ─────────────────────────────────────────────────────────────
# 9. PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────
def plot_gauge(proba: float, classification: str):
    color = "#ef5350" if classification == "ELEVATED RISK" else "#66bb6a"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": "Risk Probability", "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155"},
            "bar": {"color": color},
            "bgcolor": "#0d1b2a",
            "borderwidth": 1,
            "bordercolor": "#1e3a5f",
            "steps": [
                {"range": [0, 40], "color": "#1b5e20"},
                {"range": [40, 60], "color": "#4a3000"},
                {"range": [60, 100], "color": "#7f0000"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.75,
                "value": proba * 100,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    labels = ["Low Risk", "Elevated Risk"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        title=title,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        title_font_size=13,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def plot_feature_importance(coef_dict: dict):
    items = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [v for _, v in items]
    colors = ["#ef5350" if v > 0 else "#66bb6a" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Logistic Regression Coefficients (scaled features)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        height=320,
        margin=dict(l=20, r=60, t=40, b=10),
        xaxis={"gridcolor": "#1e3a5f", "zerolinecolor": "#334155"},
        yaxis={"gridcolor": "rgba(0,0,0,0)"},
    )
    return fig

def plot_radar(inputs: dict):
    acwr = inputs["acute_load"] / max(inputs["chronic_load"], 1)
    cats = ["Soreness", "Fatigue", "Inv. Wellness", "Sleep Deficit", "ACWR Norm", "Workload Δ"]
    vals = [
        inputs["soreness_score"] / 10,
        inputs["fatigue_score"] / 10,
        1 - (inputs["wellness_score"] / 10),
        max(0, (8 - inputs["sleep_hours"]) / 8),
        min(acwr / 2, 1),
        min(max(inputs["workload_increase_pct"], 0) / 60, 1),
    ]
    vals_closed = vals + [vals[0]]
    cats_closed = cats + [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill="toself",
        fillcolor="rgba(239,83,80,0.2)",
        line={"color": "#ef5350", "width": 2},
    ))
    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 1], "gridcolor": "#1e3a5f", "linecolor": "#1e3a5f"},
            "angularaxis": {"gridcolor": "#1e3a5f", "linecolor": "#1e3a5f"},
            "bgcolor": "rgba(0,0,0,0)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        height=280,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────
# 10. COST-COMPARISON TABLE (illustrative assumptions only)
# ─────────────────────────────────────────────────────────────
def render_cost_comparison():
    st.markdown("""
<div class='section-header'>💰 Illustrative 3-Year Cost Comparison</div>
""", unsafe_allow_html=True)

    st.warning("""
⚠️ **HYPOTHETICAL ASSUMPTIONS ONLY** — Every figure below is a made-up illustrative assumption
for teaching purposes. None of these are vendor quotes, market research, or financial advice.
Real costs will differ significantly by organisation, geography, and vendor.
""")

    cols = st.columns(3)
    options = ["Build (Custom Dev)", "Buy (Off-Shelf SaaS)", "Rent (Managed Service)"]
    costs = {
        "Build (Custom Dev)": {
            "Y1": 120_000, "Y2": 45_000, "Y3": 45_000,
            "items": [
                "Dev team: 2 × engineers × 6 months (hypothetical ~£80k)",
                "Cloud infra setup: hypothetical ~£15k",
                "Data science/ML work: hypothetical ~£25k",
                "Ongoing maintenance Y2+: hypothetical ~£45k/yr",
            ],
            "pros": "Full control, customisable, no vendor lock-in",
            "cons": "High upfront cost, needs internal expertise, longer time-to-value",
        },
        "Buy (Off-Shelf SaaS)": {
            "Y1": 55_000, "Y2": 40_000, "Y3": 40_000,
            "items": [
                "Licence fee: hypothetical ~£30k/yr",
                "Implementation/onboarding: hypothetical ~£20k (Y1 only)",
                "Training: hypothetical ~£5k (Y1 only)",
                "Integration work: hypothetical ~£10k (Y1 only — estimate)",
            ],
            "pros": "Fast deployment, vendor support, regular updates",
            "cons": "Limited customisation, data privacy considerations, recurring costs",
        },
        "Rent (Managed Service)": {
            "Y1": 75_000, "Y2": 60_000, "Y3": 60_000,
            "items": [
                "Managed service fee: hypothetical ~£50k/yr",
                "Custom model tuning: hypothetical ~£20k (Y1)",
                "Data governance setup: hypothetical ~£5k (Y1)",
                "Support and SLA costs: included in hypothetical fee",
            ],
            "pros": "Fully managed, scales easily, dedicated support",
            "cons": "Highest recurring cost, reduced control, vendor dependency",
        },
    }

    for col, name in zip(cols, options):
        c = costs[name]
        total = c["Y1"] + c["Y2"] + c["Y3"]
        with col:
            st.markdown(f"""
<div class='metric-card' style='border-color:#1e88e5'>
<div class='metric-label'>{name}</div>
<div class='metric-value'>£{total:,.0f}</div>
<div class='metric-sub'>Hypothetical 3-yr total</div>
</div>""", unsafe_allow_html=True)
            st.markdown(f"**Y1:** £{c['Y1']:,} | **Y2:** £{c['Y2']:,} | **Y3:** £{c['Y3']:,}")
            for item in c["items"]:
                st.markdown(f"- {item}")
            st.markdown(f"✅ **Pros:** {c['pros']}")
            st.markdown(f"❌ **Cons:** {c['cons']}")

    st.markdown("""
**Business Value Context (Illustrative, Not Proven):**
- Reduced reactive injury management workload for medical staff
- Potential for earlier identification of fatigue accumulation patterns
- Structured documentation trail supporting medical governance
- Improved squad availability *may* reduce disruption to coaching plans — **no specific reduction figure is claimed**
- These are plausible operational benefits; actual ROI would require real deployment data and controlled evaluation

> "Any system claiming specific injury reduction percentages from synthetic or pilot data should be treated with extreme scepticism until independently validated."
""")

# ─────────────────────────────────────────────────────────────
# 11. SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
if "reviews" not in st.session_state:
    st.session_state.reviews = []
if "pending_review" not in st.session_state:
    st.session_state.pending_review = None

# Seed widget keys with Custom-Input defaults on the very first run.
# After that Streamlit owns these values; we only overwrite them via
# the on_change callback below.
_CUSTOM_DEFAULTS = {
    "pid": "PLY-001",
    "al":  1500.0,
    "cl":  1300.0,
    "sh":  7.0,
    "ss":  3,
    "fs":  3,
    "ws":  7,
    "wi":  10.0,
    "pi":  "No (0)",
}
for _k, _v in _CUSTOM_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _on_demo_change() -> None:
    """
    on_change callback for the demo-case selectbox.
    Runs BEFORE the sidebar widgets are instantiated on the rerun
    triggered by the selectbox change, so writing to widget keys
    here is safe and takes effect immediately.
    """
    choice = st.session_state["demo_selector"]
    if choice == "\u2014 Custom Input \u2014":
        # Custom input: leave existing widget values untouched.
        return
    case = DEMO_CASES[choice]
    st.session_state["pid"] = case["player_id"]
    st.session_state["al"]  = float(case["acute_load"])
    st.session_state["cl"]  = float(case["chronic_load"])
    st.session_state["sh"]  = float(case["sleep_hours"])
    st.session_state["ss"]  = int(case["soreness_score"])
    st.session_state["fs"]  = int(case["fatigue_score"])
    st.session_state["ws"]  = int(case["wellness_score"])
    st.session_state["wi"]  = float(case["workload_increase_pct"])
    st.session_state["pi"]  = "Yes (1)" if case["previous_injury"] == 1 else "No (0)"

# ─────────────────────────────────────────────────────────────
# 12. LOAD MODEL AND DATA
# ─────────────────────────────────────────────────────────────
with st.spinner("🔬 Initialising synthetic model and protocol index…"):
    pipeline, dummy, X_train, X_test, y_train, y_test, df_full = train_models()

# ─────────────────────────────────────────────────────────────
# 13. HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='banner'>
  <h1>🏥 AthleteGuard AI</h1>
  <p>Player Health & Injury Risk Monitoring System — Educational Prototype</p>
</div>
<div class='warning-banner'>
  ⛔ EDUCATIONAL PROTOTYPE — NOT MEDICAL ADVICE &nbsp;|&nbsp;
  Synthetic data only &nbsp;|&nbsp;
  Injury-risk scores are NOT diagnoses &nbsp;|&nbsp;
  All decisions require qualified medical review
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 14. SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Player Input Panel")
    st.markdown("---")

    # Demo case selector — on_change callback updates widget keys before
    # the widgets are rendered, so values reflect the preset immediately.
    st.selectbox(
        "Load Demo Case",
        ["\u2014 Custom Input \u2014"] + list(DEMO_CASES.keys()),
        key="demo_selector",
        on_change=_on_demo_change,
    )

    st.markdown("### 👤 Player")
    # No value= argument — session state is the single source of truth.
    player_id = st.text_input("Player ID (fictional)", key="pid")

    st.markdown("### 🏋️ Training Load")
    acute_load = st.slider(
        "Acute Load (7-day, AU)", 200.0, 4000.0,
        step=50.0, key="al"
    )
    chronic_load = st.slider(
        "Chronic Load (28-day, AU)", 300.0, 3000.0,
        step=50.0, key="cl"
    )
    workload_increase_pct = st.slider(
        "Workload Change vs Last Week (%)", -30.0, 60.0,
        step=1.0, key="wi"
    )

    st.markdown("### 😴 Recovery & Wellness")
    sleep_hours = st.slider(
        "Sleep Last Night (hours)", 3.0, 10.0,
        step=0.5, key="sh"
    )
    soreness_score = st.slider(
        "Soreness (0=none, 10=severe)", 0, 10,
        key="ss"
    )
    fatigue_score = st.slider(
        "Fatigue (0=fresh, 10=exhausted)", 0, 10,
        key="fs"
    )
    wellness_score = st.slider(
        "Wellness (0=very poor, 10=excellent)", 0, 10,
        key="ws"
    )

    st.markdown("### 🩺 Medical History")
    prev_inj = st.selectbox(
        "Previous Injury (last 8 weeks)?",
        ["No (0)", "Yes (1)"],
        key="pi"
    )
    previous_injury = 1 if "Yes" in prev_inj else 0

    st.markdown("### 🎚️ Model Threshold")
    threshold = st.slider(
        "Risk Probability Threshold", 0.1, 0.9, 0.5, 0.05, key="thr"
    )
    st.caption("Lower threshold = more sensitive (more flags). Higher = more specific (fewer flags).")

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyse Player", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 15. COMPUTE RESULTS
# ─────────────────────────────────────────────────────────────
inputs = {
    "acute_load": acute_load,
    "chronic_load": chronic_load,
    "sleep_hours": sleep_hours,
    "soreness_score": soreness_score,
    "fatigue_score": fatigue_score,
    "wellness_score": wellness_score,
    "workload_increase_pct": workload_increase_pct,
    "previous_injury": previous_injury,
}

acwr = acute_load / max(chronic_load, 1)
proba, classification = predict_player(pipeline, inputs, threshold)
signals = identify_signals(inputs, proba, threshold)
conflicts = detect_conflicts(inputs, signals)

query = build_query(inputs)
retrieved_sections = retrieve_protocol_sections(query, top_k=3)

briefing = generate_briefing(
    player_id=player_id,
    inputs=inputs,
    proba=proba,
    classification=classification,
    threshold=threshold,
    retrieved_sections=retrieved_sections,
)

# ─────────────────────────────────────────────────────────────
# 16. MAIN CONTENT TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏥 Player Analysis",
    "📊 Squad Overview",
    "🤖 Model Evaluation",
    "📋 Medical Reviewer",
    "💼 Business Case",
    "⚠️ Limitations",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — PLAYER ANALYSIS
# ════════════════════════════════════════════════════════════════
with tabs[0]:
    # Risk header
    badge_class = (
        "risk-high" if classification == "ELEVATED RISK"
        else "risk-borderline" if proba >= 0.35
        else "risk-low"
    )
    st.markdown(f"""
<div style='display:flex;align-items:center;gap:20px;margin-bottom:16px'>
  <div>
    <div style='font-size:0.8rem;color:#64748b'>Player</div>
    <div style='font-size:1.3rem;font-weight:700'>{player_id}</div>
  </div>
  <div class='{badge_class}'>{classification}</div>
  <div style='font-size:1.4rem;font-weight:700;color:#{"ef5350" if classification=="ELEVATED RISK" else "66bb6a"}'>{proba:.1%}</div>
</div>
""", unsafe_allow_html=True)

    col_gauge, col_radar, col_metrics = st.columns([1.2, 1.2, 1])

    with col_gauge:
        st.plotly_chart(plot_gauge(proba, classification), use_container_width=True, key="gauge")

    with col_radar:
        st.markdown("<div class='section-header'>Risk Radar</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_radar(inputs), use_container_width=True, key="radar")

    with col_metrics:
        st.markdown("<div class='section-header'>Key Metrics</div>", unsafe_allow_html=True)
        acwr_color = "#ef5350" if acwr > 1.5 else "#ffa726" if acwr > 1.3 else "#66bb6a"
        metrics_data = [
            ("ACWR", f"{acwr:.2f}", "Acute÷Chronic", acwr_color),
            ("Sleep", f"{sleep_hours:.1f}h", "Last night", "#66bb6a" if sleep_hours >= 7 else "#ef5350"),
            ("Soreness", f"{soreness_score}/10", "Self-reported", "#ef5350" if soreness_score >= 7 else "#66bb6a"),
            ("Fatigue", f"{fatigue_score}/10", "Self-reported", "#ef5350" if fatigue_score >= 7 else "#66bb6a"),
            ("Wellness", f"{wellness_score}/10", "Self-reported", "#66bb6a" if wellness_score >= 7 else "#ef5350"),
            ("Load Δ", f"{workload_increase_pct:+.1f}%", "Week-on-week", "#ef5350" if workload_increase_pct > 30 else "#66bb6a"),
        ]
        for label, value, sub, color in metrics_data:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-label'>{label}</div>
<div class='metric-value' style='color:{color};font-size:1.2rem'>{value}</div>
<div class='metric-sub'>{sub}</div>
</div>""", unsafe_allow_html=True)

    # ── Risk Signals
    st.markdown("<div class='section-header'>⚡ Risk Signals (vs. Fictional Protocol)</div>", unsafe_allow_html=True)
    for sig in signals:
        st.markdown(f"- {sig}")

    if conflicts:
        st.markdown("<div class='section-header'>⚡ Conflicting Evidence</div>", unsafe_allow_html=True)
        for conf in conflicts:
            st.info(conf)

    # ── Retrieved Protocol Sections
    st.markdown("<div class='section-header'>📖 Retrieved Protocol Excerpts</div>", unsafe_allow_html=True)
    st.caption("Retrieved via TF-IDF cosine similarity from club_protocol.md (fictional document)")

    if retrieved_sections:
        for sec in retrieved_sections:
            rel_pct = round(sec["score"] * 100, 1)
            st.markdown(f"""
<div class='protocol-box'>
<strong>§ {sec['heading']}</strong>
<span style='float:right;font-size:0.75rem;color:#546e7a'>Relevance: {rel_pct}%</span>
<br><br>{sec['excerpt']}
</div>""", unsafe_allow_html=True)
        if all(s["score"] < 0.05 for s in retrieved_sections):
            st.warning("⚠️ Retrieved sections have very low similarity scores. Protocol evidence may be insufficient for this specific combination of inputs.")
    else:
        st.warning("⚠️ No protocol sections retrieved. Ensure club_protocol.md is present in the working directory.")

    # ── Advisory Briefing
    st.markdown("<div class='section-header'>📝 Advisory Briefing (Template-Based)</div>", unsafe_allow_html=True)
    st.caption("⚠️ This is a TEMPLATE-BASED briefing, not an LLM output. It cites retrieved protocol text only.")
    st.markdown(f"<div class='briefing-box'><pre style='white-space:pre-wrap;font-family:monospace;font-size:0.8rem;color:#cfd8dc'>{briefing}</pre></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — SQUAD OVERVIEW
# ════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'>👥 Synthetic Squad Snapshot</div>", unsafe_allow_html=True)
    st.caption("⚠️ SYNTHETIC DATA ONLY — all player names and values are fictional")

    squad_df = generate_squad_overview()
    flagged = squad_df[squad_df["Status"] == "🔴 Flag"]
    monitor = squad_df[squad_df["Status"] == "🟡 Monitor"]
    normal = squad_df[squad_df["Status"] == "🟢 Normal"]

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Flagged", len(flagged), help="Illustrative count — fictional data")
    c2.metric("🟡 Monitor", len(monitor), help="Illustrative count — fictional data")
    c3.metric("🟢 Normal", len(normal), help="Illustrative count — fictional data")

    st.dataframe(
        squad_df.style.map(
            lambda v: "background-color:#3e0000;color:#ffcdd2" if v == "🔴 Flag"
            else "background-color:#3e2800;color:#fff3e0" if v == "🟡 Monitor"
            else "",
            subset=["Status"],
        ),
        use_container_width=True,
        height=500,
    )

    # ACWR distribution
    fig_scatter = px.scatter(
        squad_df, x="Acute Load", y="Chronic Load",
        color="Status",
        hover_data=["Player", "Pos", "ACWR", "Sleep (h)"],
        color_discrete_map={
            "🔴 Flag": "#ef5350",
            "🟡 Monitor": "#ffa726",
            "🟢 Normal": "#66bb6a",
        },
        title="Acute vs Chronic Load — Synthetic Squad",
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.8)",
        font={"color": "#e2e8f0"},
        height=350,
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="squad_scatter")

# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>🤖 Model Evaluation (Synthetic Data)</div>", unsafe_allow_html=True)
    st.warning("⚠️ All metrics below are computed on **synthetic data** generated by a rule-based process. These results illustrate model behaviour on artificial data and have NO predictive validity for real players.")

    m_main = compute_metrics(pipeline, X_test, y_test, threshold, "LogisticRegression")
    m_dummy = compute_metrics(dummy, X_test, y_test, threshold, "DummyClassifier (baseline)")

    col_m, col_d = st.columns(2)
    for col, m in [(col_m, m_main), (col_d, m_dummy)]:
        with col:
            label_color = "#1e88e5" if m["label"] == "LogisticRegression" else "#546e7a"
            st.markdown(f"<div style='color:{label_color};font-weight:700;font-size:1rem'>{m['label']}</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{m['precision']:.3f}")
            c2.metric("Recall", f"{m['recall']:.3f}")
            c3.metric("F1 Score", f"{m['f1']:.3f}")
            c4.metric("ROC-AUC", f"{m['auc']:.3f}" if not math.isnan(m["auc"]) else "N/A")
            st.plotly_chart(
                plot_confusion_matrix(m["cm"], f"Confusion Matrix — {m['label']}"),
                use_container_width=True,
                key=f"cm_{m['label']}",
            )

    # Feature importance
    coef_dict = get_feature_importance(pipeline)
    st.plotly_chart(plot_feature_importance(coef_dict), use_container_width=True, key="fi")

    st.markdown("""
### Business Interpretation of Errors

| Error Type | What It Means | Business Impact |
|---|---|---|
| **False Positive** | Model flags player as high-risk when they are actually fit | Unnecessary restriction, player frustration, coaching disruption. Lower operational cost than a false negative in most contexts. |
| **False Negative** | Model does NOT flag player who is actually at elevated risk | Player continues training with risk signals undetected. More serious consequence — requires human oversight as safety net. |

> **Threshold Advice:** In a precautionary health monitoring context, lowering the threshold (e.g., 0.35) increases recall (catches more true risks) at the cost of more false positives. The appropriate threshold must be decided by the medical team based on accepted false-positive rates, not by the AI system.

### Demo Cases — Expected Model Behaviour

| Demo Case | Expected Outcome |
|---|---|
| Low Risk (Player A) | Low probability, "LOW RISK" label, few/no signals |
| High Risk (Player B) | High probability, "ELEVATED RISK" label, multiple signals |
| Borderline (Player C) | Mid-range probability, possibly conflicting signals |

""")

    # Training data overview
    st.markdown("<div class='section-header'>📊 Synthetic Training Dataset</div>", unsafe_allow_html=True)
    st.caption(f"N={len(df_full)} synthetic observations | Elevated Risk prevalence: {df_full['elevated_risk'].mean():.1%}")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_dist = px.histogram(
            df_full, x="acute_load", color=df_full["elevated_risk"].map({0: "Low Risk", 1: "Elevated Risk"}),
            barmode="overlay", title="Acute Load Distribution by Label",
            color_discrete_map={"Low Risk": "#66bb6a", "Elevated Risk": "#ef5350"},
            nbins=30,
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,27,42,0.8)",
            font={"color": "#e2e8f0"}, height=280, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="hist_al")
    with col_h2:
        fig_dist2 = px.histogram(
            df_full, x="sleep_hours", color=df_full["elevated_risk"].map({0: "Low Risk", 1: "Elevated Risk"}),
            barmode="overlay", title="Sleep Hours Distribution by Label",
            color_discrete_map={"Low Risk": "#66bb6a", "Elevated Risk": "#ef5350"},
            nbins=25,
        )
        fig_dist2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,27,42,0.8)",
            font={"color": "#e2e8f0"}, height=280, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_dist2, use_container_width=True, key="hist_sh")

# ════════════════════════════════════════════════════════════════
# TAB 4 — MEDICAL REVIEWER
# ════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'>📋 Medical Reviewer Checkpoint</div>", unsafe_allow_html=True)
    st.info("""
**ℹ️ SESSION-STATE ONLY STORAGE**
Reviews are stored in browser session memory only and will be lost when the app restarts.
This is NOT a secure production medical record system. For real clinical use, a
compliant electronic health record system with authentication and audit logging would be required.
""")

    with st.form("reviewer_form"):
        st.markdown("#### Reviewer Details")
        rev_col1, rev_col2 = st.columns(2)
        reviewer_name = rev_col1.text_input("Reviewer Name / ID", placeholder="e.g. Dr. Smith / MED-42")
        reviewer_role = rev_col2.selectbox("Role", ["Club Doctor", "Head Physiotherapist", "Sports Scientist", "Strength & Conditioning"])

        st.markdown("#### Review of Current Player Assessment")
        st.markdown(f"**Patient:** {player_id} | **Risk Score:** {proba:.1%} | **Classification:** {classification}")

        decision = st.selectbox(
            "Decision",
            [
                "⏳ Pending — Awaiting further assessment",
                "✅ Clear for normal training activity",
                "🔧 Modify training — Reduced load / Modified session",
                "🚫 Hold — Pending full medical assessment",
            ]
        )

        rev_notes = st.text_area(
            "Clinical Notes",
            placeholder="Enter reviewer notes, observations, context, and rationale here…\n(Not stored securely — prototype only)",
            height=120,
        )

        st.markdown("---")
        st.markdown("**⚠️ Confirmation Required**")
        confirmed = st.checkbox(
            "I confirm I am a qualified medical professional reviewing this advisory output. "
            "I understand this system provides synthetic-data risk signals only and that "
            "clinical judgment supersedes any automated score.",
            value=False,
        )
        st.caption("No automated clearance or benching will occur. This form records a reviewer decision only.")

        submitted = st.form_submit_button("💾 Save Review Decision", use_container_width=True)

    if submitted:
        if not confirmed:
            st.error("❌ Please tick the confirmation checkbox before saving a review.")
        elif not reviewer_name.strip():
            st.error("❌ Please enter a reviewer name or ID.")
        else:
            review_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "player_id": player_id,
                "risk_proba": round(proba, 4),
                "classification": classification,
                "threshold_used": threshold,
                "reviewer": reviewer_name,
                "role": reviewer_role,
                "decision": decision,
                "notes": rev_notes,
                "acute_load": acute_load,
                "chronic_load": chronic_load,
                "sleep_hours": sleep_hours,
                "soreness_score": soreness_score,
                "fatigue_score": fatigue_score,
                "wellness_score": wellness_score,
                "workload_increase_pct": workload_increase_pct,
                "previous_injury": previous_injury,
            }
            st.session_state.reviews.append(review_record)
            st.success(f"✅ Review saved to session state. Total reviews this session: {len(st.session_state.reviews)}")

    # Review history
    if st.session_state.reviews:
        st.markdown("<div class='section-header'>📜 Review History (This Session)</div>", unsafe_allow_html=True)
        reviews_df = pd.DataFrame(st.session_state.reviews)
        st.dataframe(reviews_df[["timestamp", "player_id", "risk_proba", "classification", "reviewer", "decision"]], use_container_width=True)

        if st.button("🗑️ Clear Session Reviews"):
            st.session_state.reviews = []
            st.rerun()
    else:
        st.info("No reviews saved yet this session.")

# ════════════════════════════════════════════════════════════════
# TAB 5 — BUSINESS CASE
# ════════════════════════════════════════════════════════════════
with tabs[4]:
    render_cost_comparison()

    st.markdown("""
---
### Pre-Deployment Validation Requirements (Illustrative)

Before any version of this system could be considered for real deployment, the following would be required:

1. **Real Data Collection:** Replace synthetic data with real anonymised player wellness records (minimum 2–3 seasons, multiple players)
2. **Prospective Validation:** Compare model flags against independently assessed clinical outcomes
3. **Regulatory Review:** Assess applicability of medical device regulations (MDR/FDA) in jurisdiction
4. **Ethics & Privacy:** GDPR-compliant data handling, informed consent, player welfare officer sign-off
5. **Clinical Validation:** Independent review by sports medicine consultants
6. **Bias Audit:** Check for performance differences by position, age, nationality, playing style
7. **Human Factors Testing:** Ensure reviewers correctly understand and use (not blindly follow) the tool

> None of the above has been performed for this educational prototype.
""")

# ════════════════════════════════════════════════════════════════
# TAB 6 — LIMITATIONS
# ════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-header'>⚠️ System Limitations & Ethical Considerations</div>", unsafe_allow_html=True)

    limitations = {
        "🧪 Synthetic Data Only": (
            "The model was trained entirely on artificially generated data using a rule-based "
            "label function that mirrors the fictional protocol. No real player injury outcomes "
            "were used. Reported model metrics reflect performance on synthetic holdout data only "
            "and have no predictive validity for real players."
        ),
        "🔄 Circular Label Logic": (
            "The synthetic training labels were generated by a rule-based system closely aligned "
            "with the features the model learns. This creates a risk of circular validation — the "
            "model may appear to perform well simply because it re-learns the label-generation rules, "
            "not because it captures real injury patterns."
        ),
        "📉 No Temporal Structure": (
            "Real injury risk is a time-series problem. Player load accumulates over time, and the "
            "context of previous sessions matters critically. This system treats each observation "
            "as independent (i.i.d.) which ignores time-series dependencies."
        ),
        "⚖️ Bias and Fairness Risks": (
            "Any model trained on real data risks encoding historical biases in how injuries were "
            "reported, treated, or attributed across different player demographics, nationalities, "
            "or positions. Synthetic data avoids real bias but also provides no evidence that the "
            "system would be fair in deployment."
        ),
        "🚨 False Alarm Fatigue": (
            "Frequent false positives can cause reviewers to dismiss or override flags habitually "
            "('alert fatigue'), potentially causing a real risk signal to be ignored. Threshold "
            "calibration and false-positive rate management are critical in any deployment."
        ),
        "🤖 Hallucination Risk (if LLM added)": (
            "If an LLM is connected, there is a risk of confident-sounding but factually wrong "
            "medical claims. The current template-based system avoids this by generating only "
            "text derived from retrieved protocol sections and computed values."
        ),
        "🔒 Privacy and Security": (
            "Real player health data is highly sensitive. This prototype has no authentication, "
            "encryption, or audit logging. Production systems would require full GDPR/HIPAA-equivalent "
            "compliance, role-based access control, and a secure data storage layer."
        ),
        "👩‍⚕️ Human Oversight Requirement": (
            "Automation bias — over-reliance on AI scores — is a documented risk in clinical "
            "decision support. This system is designed as an advisory tool only. All clinical "
            "decisions must be made by qualified professionals who critically evaluate the output."
        ),
        "🏗️ Pre-Deployment Validation Gap": (
            "No prospective study, randomised controlled trial, or independent clinical validation "
            "has been performed. The system must not be used with real players without extensive "
            "real-world validation, ethical review, and regulatory compliance assessment."
        ),
    }

    for title, body in limitations.items():
        with st.expander(title, expanded=False):
            st.markdown(body)

    st.markdown("""
---
### References and Acknowledgements

- All sports science thresholds cited in the fictional protocol are **hypothetical** and inspired by general discussions in sports science literature. They are NOT reproductions of validated clinical guidelines.
- ACWR as a concept is discussed in sports science literature (e.g., Gabbett 2016), but its efficacy as a standalone injury predictor is contested. This system does not claim to implement any validated ACWR model.
- No real player data, club proprietary data, or clinical records were used.

*This prototype was built for educational purposes as a university project demonstration.*
""")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:1px solid #1e3a5f;margin:32px 0 12px'>
<div style='text-align:center;color:#475569;font-size:0.78rem'>
  AthleteGuard AI — Educational Prototype &nbsp;|&nbsp; Synthetic Data Only &nbsp;|&nbsp;
  NOT Medical Advice &nbsp;|&nbsp; University Project 2026
</div>
""", unsafe_allow_html=True)
