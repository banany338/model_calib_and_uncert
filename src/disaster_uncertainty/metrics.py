"""Evaluation metrics for accuracy and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import binary_log_loss, clip_probabilities


def labels_from_proba(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert positive-class probabilities to binary labels."""

    return (np.asarray(probs) >= threshold).astype(int)


def confidence_from_proba(probs: np.ndarray) -> np.ndarray:
    """Confidence of the chosen class, not just probability of class 1."""

    probs = np.asarray(probs, dtype=float)
    return np.maximum(probs, 1.0 - probs)


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall, F1, log-loss, Brier score and confusion cells."""

    y = np.asarray(y_true, dtype=int)
    p = clip_probabilities(probs)
    pred = labels_from_proba(p)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    accuracy = float((pred == y).mean())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = float(((p - y) ** 2).mean())

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "log_loss": binary_log_loss(y, p),
        "brier_score": brier,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def reliability_bins(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return bin-level confidence, accuracy and counts for a reliability diagram."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    pred = labels_from_proba(p)
    confidence = confidence_from_proba(p)
    correct = (pred == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for bin_idx in range(n_bins):
        left = edges[bin_idx]
        right = edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
        else:
            mean_confidence = float((left + right) / 2.0)
            accuracy = float("nan")

        rows.append(
            {
                "bin": bin_idx + 1,
                "left": float(left),
                "right": float(right),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Weighted average gap between confidence and empirical accuracy."""

    bins = reliability_bins(y_true, probs, n_bins=n_bins)
    total = bins["count"].sum()
    if total == 0:
        return 0.0

    gaps = (bins["accuracy"] - bins["mean_confidence"]).abs().fillna(0.0)
    return float((gaps * bins["count"]).sum() / total)


def high_confidence_errors(
    texts: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """Examples where the model is wrong but highly confident."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    pred = labels_from_proba(p)
    confidence = confidence_from_proba(p)
    mask = pred != y

    rows = []
    for idx in np.where(mask)[0]:
        rows.append(
            {
                "text": texts[int(idx)],
                "true_label": int(y[int(idx)]),
                "predicted_label": int(pred[int(idx)]),
                "confidence": float(confidence[int(idx)]),
                "prob_disaster": float(p[int(idx)]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("confidence", ascending=False).head(top_n).reset_index(drop=True)


def rejection_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Evaluate the 'I do not know' mechanism over confidence thresholds."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    pred = labels_from_proba(p)
    confidence = confidence_from_proba(p)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        answered = confidence > threshold
        answered_count = int(answered.sum())
        rejected_count = int((~answered).sum())
        coverage = answered_count / len(y) if len(y) else 0.0
        selective_accuracy = (
            float((pred[answered] == y[answered]).mean()) if answered_count else float("nan")
        )
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(coverage),
                "rejected": rejected_count,
                "answered": answered_count,
                "selective_accuracy": selective_accuracy,
            }
        )

    return pd.DataFrame(rows)
