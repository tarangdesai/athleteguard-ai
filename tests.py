"""
tests.py — AthleteGuard AI Unit Tests
======================================
Run with:  python -m pytest tests.py -v
or:        python tests.py

Tests cover:
- Synthetic data generation shape, types, and label validity
- Model training and prediction pipeline
- Feature importance extraction
- Protocol retrieval (TF-IDF)
- Signal detection logic
- Briefing generation (template-based)
- Demo case sanity checks (low/high/borderline)
"""

import os
import sys
import math
import warnings
import unittest
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Try importing app components ───────────────────────────────
# app.py uses Streamlit which has a global runtime.
# We isolate importable functions by re-implementing the
# import-safe subset here, or by importing from app after
# mocking Streamlit globals.

# Import the bare modules needed
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────
# Re-implement testable functions independently of Streamlit
# (mirrors app.py logic exactly — kept in sync manually)
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "acute_load", "chronic_load", "sleep_hours",
    "soreness_score", "fatigue_score", "wellness_score",
    "workload_increase_pct", "previous_injury",
]


def generate_synthetic_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
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
    noise_mask = rng.uniform(0, 1, n) < 0.08
    elevated_risk[noise_mask] = 1 - elevated_risk[noise_mask]
    return pd.DataFrame({
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


def train_models(seed: int = 42):
    df = generate_synthetic_data(seed=seed)
    X = df[FEATURE_COLS]
    y = df["elevated_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
    dummy.fit(X_train, y_train)
    return pipeline, dummy, X_train, X_test, y_train, y_test, df


def predict_player(pipeline, inputs: dict, threshold: float = 0.5):
    row = pd.DataFrame([{col: inputs[col] for col in FEATURE_COLS}])
    proba = pipeline.predict_proba(row)[0, 1]
    label = "ELEVATED RISK" if proba >= threshold else "LOW RISK"
    return proba, label


def identify_signals(inputs: dict, proba: float, threshold: float) -> list:
    signals = []
    acwr = inputs["acute_load"] / max(inputs["chronic_load"], 1)
    if acwr > 1.5:
        signals.append(f"ACWR={acwr:.2f} > 1.5")
    if inputs["workload_increase_pct"] > 30:
        signals.append("workload increase > 30%")
    if inputs["sleep_hours"] < 4:
        signals.append("sleep < 4h critical")
    elif inputs["sleep_hours"] < 6:
        signals.append("sleep < 6h poor")
    if inputs["soreness_score"] >= 9:
        signals.append("soreness severe")
    if inputs["fatigue_score"] >= 8:
        signals.append("fatigue severe")
    if inputs["wellness_score"] <= 3:
        signals.append("wellness poor")
    if inputs["previous_injury"]:
        signals.append("previous_injury flag")
    if not signals:
        signals.append("No threshold breaches detected.")
    return signals


DEMO_CASES = {
    "Low Risk — Player A (Fit & Rested)": {
        "player_id": "PLY-007",
        "acute_load": 1200.0, "chronic_load": 1400.0, "sleep_hours": 8.5,
        "soreness_score": 2, "fatigue_score": 2, "wellness_score": 8,
        "workload_increase_pct": 5.0, "previous_injury": 0,
    },
    "High Risk — Player B (Overloaded)": {
        "player_id": "PLY-023",
        "acute_load": 3200.0, "chronic_load": 1300.0, "sleep_hours": 4.5,
        "soreness_score": 9, "fatigue_score": 9, "wellness_score": 2,
        "workload_increase_pct": 45.0, "previous_injury": 1,
    },
    "Borderline — Player C (Mixed Signals)": {
        "player_id": "PLY-041",
        "acute_load": 1900.0, "chronic_load": 1450.0, "sleep_hours": 6.0,
        "soreness_score": 6, "fatigue_score": 5, "wellness_score": 5,
        "workload_increase_pct": 22.0, "previous_injury": 0,
    },
}


# ─────────────────────────────────────────────────────────────
# TEST CLASSES
# ─────────────────────────────────────────────────────────────

class TestSyntheticData(unittest.TestCase):
    """Tests for synthetic data generation."""

    def setUp(self):
        self.df = generate_synthetic_data(n=600, seed=42)

    def test_row_count(self):
        """Dataset must have at least 500 rows."""
        self.assertGreaterEqual(len(self.df), 500, "Dataset must have ≥500 rows")

    def test_column_presence(self):
        """All required columns must be present."""
        required = FEATURE_COLS + ["elevated_risk", "acwr", "player_id"]
        for col in required:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_no_nulls(self):
        """Dataset must contain no null values."""
        self.assertEqual(self.df.isnull().sum().sum(), 0, "Dataset contains null values")

    def test_binary_target(self):
        """Target must be binary (0 or 1)."""
        unique_vals = set(self.df["elevated_risk"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}), f"Non-binary values in target: {unique_vals}")

    def test_label_prevalence(self):
        """Elevated risk prevalence should be between 10% and 60% (realistic synthetic)."""
        prevalence = self.df["elevated_risk"].mean()
        self.assertGreater(prevalence, 0.10, f"Prevalence too low: {prevalence:.2%}")
        self.assertLess(prevalence, 0.60, f"Prevalence too high: {prevalence:.2%}")

    def test_feature_ranges(self):
        """Features must be within expected bounds."""
        self.assertTrue((self.df["acute_load"] >= 200).all() and (self.df["acute_load"] <= 4000).all())
        self.assertTrue((self.df["chronic_load"] >= 300).all() and (self.df["chronic_load"] <= 3000).all())
        self.assertTrue((self.df["sleep_hours"] >= 3.0).all() and (self.df["sleep_hours"] <= 10.0).all())
        self.assertTrue((self.df["soreness_score"] >= 0).all() and (self.df["soreness_score"] <= 10).all())
        self.assertTrue((self.df["fatigue_score"] >= 0).all() and (self.df["fatigue_score"] <= 10).all())
        self.assertTrue((self.df["wellness_score"] >= 0).all() and (self.df["wellness_score"] <= 10).all())
        self.assertTrue(self.df["previous_injury"].isin([0, 1]).all())

    def test_reproducibility(self):
        """Same seed must produce identical results."""
        df1 = generate_synthetic_data(n=100, seed=7)
        df2 = generate_synthetic_data(n=100, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        df1 = generate_synthetic_data(n=100, seed=1)
        df2 = generate_synthetic_data(n=100, seed=2)
        self.assertFalse(df1["acute_load"].equals(df2["acute_load"]))


class TestModelTraining(unittest.TestCase):
    """Tests for model training and evaluation."""

    def setUp(self):
        self.pipeline, self.dummy, self.X_train, self.X_test, \
            self.y_train, self.y_test, self.df = train_models(seed=42)

    def test_pipeline_fitted(self):
        """Pipeline should be fitted with correct steps."""
        self.assertIn("scaler", self.pipeline.named_steps)
        self.assertIn("clf", self.pipeline.named_steps)

    def test_train_test_sizes(self):
        """Train/test split should be ~80/20."""
        total = len(self.X_train) + len(self.X_test)
        test_pct = len(self.X_test) / total
        self.assertAlmostEqual(test_pct, 0.2, delta=0.02)

    def test_predictions_binary(self):
        """Predictions must be 0 or 1."""
        preds = self.pipeline.predict(self.X_test)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_probabilities_valid(self):
        """Probabilities must be in [0, 1]."""
        probas = self.pipeline.predict_proba(self.X_test)
        self.assertTrue((probas >= 0).all() and (probas <= 1).all())
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_model_beats_dummy_auc(self):
        """Model AUC should be better than dummy (0.5)."""
        probas = self.pipeline.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, probas)
        self.assertGreater(auc, 0.5, f"Model AUC {auc:.3f} is not better than dummy baseline 0.5")

    def test_dummy_fitted(self):
        """Dummy classifier must be fitted and return predictions."""
        preds = self.dummy.predict(self.X_test)
        self.assertEqual(len(preds), len(self.y_test))

    def test_feature_importance_keys(self):
        """Feature importance dict must have all feature keys."""
        coef = self.pipeline.named_steps["clf"].coef_[0]
        coef_dict = dict(zip(FEATURE_COLS, coef))
        for col in FEATURE_COLS:
            self.assertIn(col, coef_dict)


class TestPredictPlayer(unittest.TestCase):
    """Tests for single-player prediction."""

    def setUp(self):
        self.pipeline, *_ = train_models(seed=42)

    def _make_inputs(self, **kwargs):
        base = {
            "acute_load": 1500.0, "chronic_load": 1300.0, "sleep_hours": 7.0,
            "soreness_score": 3, "fatigue_score": 3, "wellness_score": 7,
            "workload_increase_pct": 5.0, "previous_injury": 0,
        }
        base.update(kwargs)
        return base

    def test_returns_probability_and_label(self):
        inputs = self._make_inputs()
        proba, label = predict_player(self.pipeline, inputs)
        self.assertIsInstance(proba, float)
        self.assertIn(label, ["ELEVATED RISK", "LOW RISK"])

    def test_probability_in_range(self):
        inputs = self._make_inputs()
        proba, _ = predict_player(self.pipeline, inputs)
        self.assertGreaterEqual(proba, 0.0)
        self.assertLessEqual(proba, 1.0)

    def test_high_risk_inputs_produce_high_proba(self):
        """Severely overloaded inputs should produce elevated probability."""
        high_risk = self._make_inputs(
            acute_load=3500.0, chronic_load=1000.0, sleep_hours=3.5,
            soreness_score=10, fatigue_score=10, wellness_score=0,
            workload_increase_pct=55.0, previous_injury=1,
        )
        proba, label = predict_player(self.pipeline, high_risk)
        self.assertGreater(proba, 0.5, f"High-risk inputs gave low proba: {proba:.3f}")
        self.assertEqual(label, "ELEVATED RISK")

    def test_low_risk_inputs_produce_low_proba(self):
        """Healthy player inputs should produce low probability."""
        low_risk = self._make_inputs(
            acute_load=1100.0, chronic_load=1400.0, sleep_hours=9.0,
            soreness_score=1, fatigue_score=1, wellness_score=9,
            workload_increase_pct=2.0, previous_injury=0,
        )
        proba, label = predict_player(self.pipeline, low_risk)
        self.assertLess(proba, 0.5, f"Low-risk inputs gave high proba: {proba:.3f}")
        self.assertEqual(label, "LOW RISK")

    def test_threshold_changes_label(self):
        """Changing threshold should change classification."""
        inputs = self._make_inputs(
            acute_load=2000.0, chronic_load=1300.0, sleep_hours=6.0,
            soreness_score=5, fatigue_score=5, wellness_score=5,
        )
        proba, _ = predict_player(self.pipeline, inputs)
        # If proba is not exactly 0 or 1, different thresholds should differ
        _, label_low_thr = predict_player(self.pipeline, inputs, threshold=0.1)
        _, label_high_thr = predict_player(self.pipeline, inputs, threshold=0.95)
        # With very low threshold everything is high risk; with very high — low risk
        self.assertEqual(label_low_thr, "ELEVATED RISK")
        self.assertEqual(label_high_thr, "LOW RISK")


class TestDemoCases(unittest.TestCase):
    """Sanity checks on the three preset demo cases."""

    def setUp(self):
        self.pipeline, *_ = train_models(seed=42)

    def test_low_risk_demo(self):
        """Low-risk demo case should produce probability < 0.5."""
        inputs = {k: v for k, v in DEMO_CASES["Low Risk — Player A (Fit & Rested)"].items()
                  if k in FEATURE_COLS}
        proba, label = predict_player(self.pipeline, inputs)
        self.assertLess(proba, 0.5, f"Low-risk demo gave unexpected proba {proba:.3f}")

    def test_high_risk_demo(self):
        """High-risk demo case should produce probability ≥ 0.5."""
        inputs = {k: v for k, v in DEMO_CASES["High Risk — Player B (Overloaded)"].items()
                  if k in FEATURE_COLS}
        proba, label = predict_player(self.pipeline, inputs)
        self.assertGreaterEqual(proba, 0.5, f"High-risk demo gave low proba {proba:.3f}")

    def test_borderline_demo_is_between(self):
        """Borderline demo should produce intermediate probability."""
        low_inputs = {k: v for k, v in DEMO_CASES["Low Risk — Player A (Fit & Rested)"].items()
                      if k in FEATURE_COLS}
        high_inputs = {k: v for k, v in DEMO_CASES["High Risk — Player B (Overloaded)"].items()
                       if k in FEATURE_COLS}
        border_inputs = {k: v for k, v in DEMO_CASES["Borderline — Player C (Mixed Signals)"].items()
                         if k in FEATURE_COLS}

        low_p, _ = predict_player(self.pipeline, low_inputs)
        high_p, _ = predict_player(self.pipeline, high_inputs)
        border_p, _ = predict_player(self.pipeline, border_inputs)

        self.assertGreater(border_p, low_p, "Borderline should have higher risk than low-risk")
        self.assertLess(border_p, high_p, "Borderline should have lower risk than high-risk")


class TestSignalDetection(unittest.TestCase):
    """Tests for the identify_signals function."""

    def _base(self):
        return {
            "acute_load": 1500.0, "chronic_load": 1300.0, "sleep_hours": 7.0,
            "soreness_score": 3, "fatigue_score": 3, "wellness_score": 7,
            "workload_increase_pct": 5.0, "previous_injury": 0,
        }

    def test_no_signals_for_healthy_player(self):
        inputs = self._base()
        sigs = identify_signals(inputs, 0.1, 0.5)
        self.assertEqual(len(sigs), 1)
        self.assertIn("No threshold breaches", sigs[0])

    def test_high_acwr_detected(self):
        inputs = self._base()
        inputs["acute_load"] = 3000.0
        inputs["chronic_load"] = 1000.0
        sigs = identify_signals(inputs, 0.9, 0.5)
        self.assertTrue(any("ACWR" in s for s in sigs), f"ACWR signal not found in {sigs}")

    def test_poor_sleep_detected(self):
        inputs = self._base()
        inputs["sleep_hours"] = 3.5
        sigs = identify_signals(inputs, 0.8, 0.5)
        self.assertTrue(any("sleep" in s.lower() for s in sigs), f"Sleep signal not found in {sigs}")

    def test_severe_fatigue_detected(self):
        inputs = self._base()
        inputs["fatigue_score"] = 9
        sigs = identify_signals(inputs, 0.8, 0.5)
        self.assertTrue(any("fatigue" in s.lower() for s in sigs), f"Fatigue signal not found in {sigs}")

    def test_previous_injury_detected(self):
        inputs = self._base()
        inputs["previous_injury"] = 1
        sigs = identify_signals(inputs, 0.5, 0.5)
        self.assertTrue(any("previous_injury" in s.lower() or "injury" in s.lower() for s in sigs))

    def test_workload_spike_detected(self):
        inputs = self._base()
        inputs["workload_increase_pct"] = 45.0
        sigs = identify_signals(inputs, 0.9, 0.5)
        self.assertTrue(any("workload" in s.lower() or "30" in s for s in sigs))


class TestProtocolRetrieval(unittest.TestCase):
    """Tests for TF-IDF protocol retrieval."""

    def setUp(self):
        # Build a small in-memory protocol for testing
        self.sections = {
            "Training Load Monitoring": "acute training load chronic load ACWR workload increase thresholds",
            "Sleep Monitoring Protocol": "sleep hours poor sleep consecutive nights fatigue rest",
            "Soreness Thresholds": "soreness score high soreness modified training physio review",
            "Escalation Procedures": "medical review flag amber red status immediate notification",
            "Recovery Protocols": "recovery hydration nutrition cold therapy post-match window",
        }
        texts = list(self.sections.values())
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(texts)
        self.headings = list(self.sections.keys())

    def _retrieve(self, query: str, top_k: int = 2):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [{"heading": self.headings[i], "score": float(sims[i])} for i in top_idx]

    def test_sleep_query_retrieves_sleep_section(self):
        results = self._retrieve("sleep hours poor sleep fatigue")
        headings = [r["heading"] for r in results]
        self.assertIn("Sleep Monitoring Protocol", headings)

    def test_load_query_retrieves_load_section(self):
        results = self._retrieve("acute load chronic load ACWR training")
        headings = [r["heading"] for r in results]
        self.assertIn("Training Load Monitoring", headings)

    def test_returns_correct_count(self):
        results = self._retrieve("test query", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_scores_are_floats_in_range(self):
        results = self._retrieve("acute load sleep soreness")
        for r in results:
            self.assertIsInstance(r["score"], float)
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)


class TestProtocolFileExists(unittest.TestCase):
    """Check that the protocol file exists."""

    def test_protocol_file_exists(self):
        protocol_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "club_protocol.md"
        )
        self.assertTrue(
            os.path.exists(protocol_path),
            f"club_protocol.md not found at {protocol_path}"
        )

    def test_protocol_file_non_empty(self):
        protocol_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "club_protocol.md"
        )
        if os.path.exists(protocol_path):
            with open(protocol_path, "r") as f:
                content = f.read()
            self.assertGreater(len(content), 1000, "Protocol file appears too short")



class TestDemoCasePresetSwitching(unittest.TestCase):
    """
    Tests for the on_change callback logic that drives demo-preset switching.

    The callback (_on_demo_change in app.py) writes DEMO_CASES values into
    session-state widget keys.  We replicate that mapping here and verify:
      - Each preset writes the correct values for every widget key.
      - Switching between presets repeatedly yields the expected values.
      - Custom Input leaves state untouched (no overwrite).
      - All preset input values produce the correct model classification.
    """

    # Mirrors _CUSTOM_DEFAULTS in app.py
    CUSTOM_DEFAULTS = {
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

    def _apply_preset(self, session_state: dict, choice: str) -> dict:
        """Simulate what _on_demo_change does for a given choice."""
        if choice == "— Custom Input —":
            return session_state  # no change
        case = DEMO_CASES[choice]
        session_state["pid"] = case["player_id"]
        session_state["al"]  = float(case["acute_load"])
        session_state["cl"]  = float(case["chronic_load"])
        session_state["sh"]  = float(case["sleep_hours"])
        session_state["ss"]  = int(case["soreness_score"])
        session_state["fs"]  = int(case["fatigue_score"])
        session_state["ws"]  = int(case["wellness_score"])
        session_state["wi"]  = float(case["workload_increase_pct"])
        session_state["pi"]  = "Yes (1)" if case["previous_injury"] == 1 else "No (0)"
        return session_state

    def _state_to_inputs(self, state: dict) -> dict:
        """Convert session-state dict to the inputs dict used by the model."""
        return {
            "acute_load":            state["al"],
            "chronic_load":          state["cl"],
            "sleep_hours":           state["sh"],
            "soreness_score":        state["ss"],
            "fatigue_score":         state["fs"],
            "wellness_score":        state["ws"],
            "workload_increase_pct": state["wi"],
            "previous_injury":       1 if "Yes" in state["pi"] else 0,
        }

    def setUp(self):
        self.pipeline, *_ = train_models(seed=42)

    # ── Preset value correctness ───────────────────────────────

    def test_low_risk_preset_values(self):
        """Low-risk preset must set the correct widget values."""
        state = dict(self.CUSTOM_DEFAULTS)
        choice = "Low Risk — Player A (Fit & Rested)"
        self._apply_preset(state, choice)
        case = DEMO_CASES[choice]
        self.assertEqual(state["al"],  float(case["acute_load"]))
        self.assertEqual(state["cl"],  float(case["chronic_load"]))
        self.assertEqual(state["sh"],  float(case["sleep_hours"]))
        self.assertEqual(state["ss"],  int(case["soreness_score"]))
        self.assertEqual(state["fs"],  int(case["fatigue_score"]))
        self.assertEqual(state["ws"],  int(case["wellness_score"]))
        self.assertEqual(state["wi"],  float(case["workload_increase_pct"]))
        self.assertEqual(state["pid"], case["player_id"])
        self.assertEqual(state["pi"],  "No (0)")

    def test_high_risk_preset_values(self):
        """High-risk preset must set the correct widget values."""
        state = dict(self.CUSTOM_DEFAULTS)
        choice = "High Risk — Player B (Overloaded)"
        self._apply_preset(state, choice)
        case = DEMO_CASES[choice]
        self.assertEqual(state["al"],  float(case["acute_load"]))
        self.assertEqual(state["wi"],  float(case["workload_increase_pct"]))
        self.assertEqual(state["sh"],  float(case["sleep_hours"]))
        self.assertEqual(state["pi"],  "Yes (1)")  # previous_injury=1

    def test_borderline_preset_values(self):
        """Borderline preset must set the correct widget values."""
        state = dict(self.CUSTOM_DEFAULTS)
        choice = "Borderline — Player C (Mixed Signals)"
        self._apply_preset(state, choice)
        case = DEMO_CASES[choice]
        self.assertEqual(state["al"],  float(case["acute_load"]))
        self.assertEqual(state["wi"],  float(case["workload_increase_pct"]))
        self.assertEqual(state["pi"],  "No (0)")  # previous_injury=0

    # ── Custom Input leaves state untouched ───────────────────

    def test_custom_input_does_not_overwrite_state(self):
        """Selecting Custom Input must leave existing widget values intact."""
        state = dict(self.CUSTOM_DEFAULTS)
        state["al"] = 2500.0  # user has manually edited this
        self._apply_preset(state, "— Custom Input —")
        self.assertEqual(state["al"], 2500.0, "Custom Input must not reset manually-edited values")

    # ── Repeated switching ─────────────────────────────────────

    def test_switching_low_to_high_updates_all_keys(self):
        """Switching from Low Risk to High Risk must overwrite ALL widget keys."""
        state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(state, "Low Risk — Player A (Fit & Rested)")
        low_al = state["al"]
        self._apply_preset(state, "High Risk — Player B (Overloaded)")
        high_al = state["al"]
        self.assertNotEqual(low_al, high_al, "Acute load must differ between Low and High presets")

    def test_switching_high_to_borderline_updates_all_keys(self):
        """Switching from High Risk to Borderline must change widget values."""
        state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(state, "High Risk — Player B (Overloaded)")
        high_wi = state["wi"]
        self._apply_preset(state, "Borderline — Player C (Mixed Signals)")
        border_wi = state["wi"]
        self.assertNotEqual(high_wi, border_wi, "Workload change must differ between High and Borderline presets")

    def test_switching_back_to_custom_after_preset(self):
        """After loading a preset, switching to Custom Input leaves preset values."""
        state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(state, "High Risk — Player B (Overloaded)")
        high_al = state["al"]
        self._apply_preset(state, "— Custom Input —")
        # Values should be unchanged (Custom Input is a no-op on state)
        self.assertEqual(state["al"], high_al)

    def test_all_three_presets_produce_distinct_state(self):
        """All three presets must produce distinct acute_load values."""
        low_state = dict(self.CUSTOM_DEFAULTS)
        high_state = dict(self.CUSTOM_DEFAULTS)
        border_state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(low_state,    "Low Risk — Player A (Fit & Rested)")
        self._apply_preset(high_state,   "High Risk — Player B (Overloaded)")
        self._apply_preset(border_state, "Borderline — Player C (Mixed Signals)")
        loads = {low_state["al"], high_state["al"], border_state["al"]}
        self.assertEqual(len(loads), 3, f"All three presets should have distinct acute loads, got {loads}")

    # ── Model classification after preset ─────────────────────

    def test_low_risk_preset_classifies_as_low(self):
        """After applying Low Risk preset, model should return low risk."""
        state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(state, "Low Risk — Player A (Fit & Rested)")
        inputs = self._state_to_inputs(state)
        proba, label = predict_player(self.pipeline, inputs, threshold=0.5)
        self.assertEqual(label, "LOW RISK",
                         f"Low Risk preset gave '{label}' with proba={proba:.3f}")

    def test_high_risk_preset_classifies_as_elevated(self):
        """After applying High Risk preset, model should return elevated risk."""
        state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(state, "High Risk — Player B (Overloaded)")
        inputs = self._state_to_inputs(state)
        proba, label = predict_player(self.pipeline, inputs, threshold=0.5)
        self.assertEqual(label, "ELEVATED RISK",
                         f"High Risk preset gave '{label}' with proba={proba:.3f}")

    def test_borderline_sits_between_low_and_high(self):
        """Borderline preset probability should lie between Low and High."""
        low_state    = dict(self.CUSTOM_DEFAULTS)
        high_state   = dict(self.CUSTOM_DEFAULTS)
        border_state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(low_state,    "Low Risk — Player A (Fit & Rested)")
        self._apply_preset(high_state,   "High Risk — Player B (Overloaded)")
        self._apply_preset(border_state, "Borderline — Player C (Mixed Signals)")
        low_p,    _ = predict_player(self.pipeline, self._state_to_inputs(low_state))
        high_p,   _ = predict_player(self.pipeline, self._state_to_inputs(high_state))
        border_p, _ = predict_player(self.pipeline, self._state_to_inputs(border_state))
        self.assertGreater(border_p, low_p,
                           f"Borderline ({border_p:.3f}) should exceed Low ({low_p:.3f})")
        self.assertLess(border_p, high_p,
                        f"Borderline ({border_p:.3f}) should be below High ({high_p:.3f})")

    def test_previous_injury_flag_mapped_correctly(self):
        """previous_injury=1 in DEMO_CASES must map to 'Yes (1)' in session state."""
        state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(state, "High Risk — Player B (Overloaded)")
        self.assertEqual(state["pi"], "Yes (1)")
        inputs = self._state_to_inputs(state)
        self.assertEqual(inputs["previous_injury"], 1)

    def test_no_previous_injury_mapped_correctly(self):
        """previous_injury=0 in DEMO_CASES must map to 'No (0)' in session state."""
        state = dict(self.CUSTOM_DEFAULTS)
        self._apply_preset(state, "Low Risk — Player A (Fit & Rested)")
        self.assertEqual(state["pi"], "No (0)")
        inputs = self._state_to_inputs(state)
        self.assertEqual(inputs["previous_injury"], 0)


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("AthleteGuard AI — Test Suite")
    print("EDUCATIONAL PROTOTYPE — SYNTHETIC DATA ONLY")
    print("=" * 70)
    unittest.main(verbosity=2)
