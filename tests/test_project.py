from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disaster_uncertainty.calibration import IsotonicCalibrator, TemperatureScaler
from disaster_uncertainty.conformal import evaluate_conformal_prediction
from disaster_uncertainty.metrics import expected_calibration_error, rejection_curve
from disaster_uncertainty.modeling import (
    EmbeddingMLPConfig,
    LinearSVMConfig,
    LogisticRegressionConfig,
    TextEmbeddingMLP,
    TextLinearSVM,
    TextLogisticRegression,
    TextMultinomialNaiveBayes,
    sigmoid,
)


class MetricsTests(unittest.TestCase):
    def test_ece_is_zero_for_perfect_confident_predictions(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        probs = np.array([0.0, 1.0, 0.0, 1.0])

        self.assertAlmostEqual(expected_calibration_error(y_true, probs), 0.0)

    def test_rejection_curve_counts_answered_and_rejected_examples(self) -> None:
        y_true = np.array([0, 1, 1])
        probs = np.array([0.10, 0.60, 0.51])

        curve = rejection_curve(y_true, probs, thresholds=[0.70])

        self.assertEqual(int(curve.loc[0, "answered"]), 1)
        self.assertEqual(int(curve.loc[0, "rejected"]), 2)
        self.assertAlmostEqual(float(curve.loc[0, "coverage"]), 1 / 3)
        self.assertAlmostEqual(float(curve.loc[0, "selective_accuracy"]), 1.0)


class CalibrationTests(unittest.TestCase):
    def test_temperature_scaler_finds_positive_temperature(self) -> None:
        logits = np.array([-4.0, -1.0, 1.0, 4.0])
        y_true = np.array([0, 0, 1, 1])

        scaler = TemperatureScaler().fit(logits, y_true)
        calibrated = scaler.predict_proba(logits)

        self.assertGreater(scaler.temperature_, 0.0)
        self.assertTrue(np.all((calibrated > 0.0) & (calibrated < 1.0)))

    def test_isotonic_predictions_are_monotonic(self) -> None:
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        y_true = np.array([0, 1, 0, 1, 1])

        calibrator = IsotonicCalibrator().fit(probs, y_true)
        pred = calibrator.predict_proba(np.array([0.15, 0.25, 0.35, 0.95]))

        self.assertTrue(np.all(np.diff(pred) >= -1e-12))


class ConformalTests(unittest.TestCase):
    def test_conformal_prediction_returns_expected_columns(self) -> None:
        y_calibration = np.array([0, 1, 0, 1])
        calibration_probs = np.array([0.1, 0.9, 0.2, 0.8])
        y_test = np.array([0, 1])
        test_probs = np.array([0.3, 0.7])

        result = evaluate_conformal_prediction(
            "toy",
            y_calibration,
            calibration_probs,
            y_test,
            test_probs,
            alphas=[0.1],
        )

        self.assertEqual(result.loc[0, "model"], "toy")
        self.assertIn("empirical_coverage", result.columns)
        self.assertIn("avg_set_size", result.columns)
        self.assertGreaterEqual(float(result.loc[0, "empirical_coverage"]), 0.0)
        self.assertLessEqual(float(result.loc[0, "empirical_coverage"]), 1.0)


class ModelingTests(unittest.TestCase):
    def test_sigmoid_handles_scalar_and_array(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        arr = sigmoid(np.array([-1.0, 0.0, 1.0]))
        self.assertEqual(arr.shape, (3,))
        self.assertTrue(np.all((arr > 0.0) & (arr < 1.0)))

    def test_text_classifier_outputs_probabilities(self) -> None:
        texts = [
            "fire explosion rescue",
            "flood evacuation emergency",
            "coffee music movie",
            "homework meeting coffee",
        ]
        y = np.array([1, 1, 0, 0])
        model = TextLogisticRegression(
            min_df=1,
            max_features=100,
            config=LogisticRegressionConfig(epochs=20, learning_rate=0.25, seed=7),
        )

        model.fit(texts, y)
        probs = model.predict_proba(texts)

        self.assertEqual(probs.shape, (4,))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        self.assertGreater(probs[:2].mean(), probs[2:].mean())

    def test_naive_bayes_outputs_probabilities(self) -> None:
        texts = [
            "fire explosion rescue",
            "flood evacuation emergency",
            "coffee music movie",
            "homework meeting coffee",
        ]
        y = np.array([1, 1, 0, 0])
        model = TextMultinomialNaiveBayes(min_df=1, max_features=100)

        model.fit(texts, y)
        probs = model.predict_proba(texts)

        self.assertEqual(probs.shape, (4,))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        self.assertGreater(probs[:2].mean(), probs[2:].mean())

    def test_linear_svm_outputs_margin_based_probabilities(self) -> None:
        texts = [
            "fire explosion rescue",
            "flood evacuation emergency",
            "coffee music movie",
            "homework meeting coffee",
        ]
        y = np.array([1, 1, 0, 0])
        model = TextLinearSVM(
            min_df=1,
            max_features=100,
            config=LinearSVMConfig(epochs=12, learning_rate=0.15, seed=9),
        )

        model.fit(texts, y)
        probs = model.predict_proba(texts)

        self.assertEqual(probs.shape, (4,))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        self.assertGreater(probs[:2].mean(), probs[2:].mean())

    def test_embedding_mlp_outputs_probabilities(self) -> None:
        texts = [
            "fire explosion rescue",
            "flood evacuation emergency",
            "coffee music movie",
            "homework meeting coffee",
        ]
        y = np.array([1, 1, 0, 0])
        model = TextEmbeddingMLP(
            min_df=1,
            max_tokens=100,
            config=EmbeddingMLPConfig(
                embedding_dim=8,
                hidden_dim=8,
                epochs=8,
                learning_rate=0.05,
                seed=11,
            ),
        )

        model.fit(texts, y)
        probs = model.predict_proba(texts)

        self.assertEqual(probs.shape, (4,))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))


if __name__ == "__main__":
    unittest.main()
