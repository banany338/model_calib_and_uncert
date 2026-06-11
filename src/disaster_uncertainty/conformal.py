"""Simple split conformal prediction for binary classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import labels_from_proba


@dataclass
class ConformalThreshold:
    alpha: float
    qhat: float


def class_probability(probs: np.ndarray, klass: int) -> np.ndarray:
    """Return probability assigned to class 0 or class 1."""

    probs = np.asarray(probs, dtype=float)
    if klass == 1:
        return probs
    if klass == 0:
        return 1.0 - probs
    raise ValueError("klass must be 0 or 1")


def nonconformity_scores(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Score is low when the model assigns high probability to the true class."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    true_class_prob = np.where(y == 1, p, 1.0 - p)
    return 1.0 - true_class_prob


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("scores must not be empty")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])


def conformal_prediction_sets(probs: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean membership arrays for class 0 and class 1."""

    p = np.asarray(probs, dtype=float)
    include_0 = (1.0 - (1.0 - p)) <= qhat
    include_1 = (1.0 - p) <= qhat
    return include_0, include_1


def evaluate_conformal_prediction(
    model: str,
    y_calibration: np.ndarray,
    calibration_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    alphas: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Evaluate split conformal prediction sets for several alpha values."""

    calibration_scores = nonconformity_scores(y_calibration, calibration_probs)
    y = np.asarray(y_test, dtype=int)
    point_pred = labels_from_proba(test_probs)
    rows: list[dict[str, float | int | str]] = []

    for alpha in alphas:
        qhat = conformal_quantile(calibration_scores, float(alpha))
        include_0, include_1 = conformal_prediction_sets(test_probs, qhat)
        set_size = include_0.astype(int) + include_1.astype(int)
        contains_true = np.where(y == 1, include_1, include_0)
        singleton = set_size == 1
        empty = set_size == 0
        ambiguous = set_size == 2
        singleton_correct = singleton & (point_pred == y)

        rows.append(
            {
                "model": model,
                "alpha": float(alpha),
                "target_coverage": float(1.0 - alpha),
                "qhat": qhat,
                "empirical_coverage": float(contains_true.mean()),
                "avg_set_size": float(set_size.mean()),
                "singleton_rate": float(singleton.mean()),
                "ambiguous_rate": float(ambiguous.mean()),
                "empty_rate": float(empty.mean()),
                "singleton_accuracy": (
                    float(singleton_correct.sum() / singleton.sum())
                    if singleton.sum()
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)
