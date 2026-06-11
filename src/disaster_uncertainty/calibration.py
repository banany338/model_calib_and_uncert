"""Calibration methods for binary probabilistic predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modeling import sigmoid


EPS = 1e-12


def clip_probabilities(probs: np.ndarray) -> np.ndarray:
    """Keep probabilities away from exact 0 and 1 for log-loss stability."""

    return np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)


def binary_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Binary negative log likelihood."""

    y = np.asarray(y_true, dtype=float)
    p = clip_probabilities(probs)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


@dataclass
class TemperatureScaler:
    """Temperature scaling: p = sigmoid(logit / T)."""

    temperature_: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        candidates = np.logspace(-1.2, 1.2, 121)
        losses = [
            binary_log_loss(y_true, sigmoid(logits / temperature))
            for temperature in candidates
        ]
        self.temperature_ = float(candidates[int(np.argmin(losses))])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return clip_probabilities(sigmoid(np.asarray(logits, dtype=float) / self.temperature_))


@dataclass
class PlattScaler:
    """Sigmoid/Platt scaling learned on the calibration split."""

    slope_: float = 1.0
    intercept_: float = 0.0
    epochs: int = 1_500
    learning_rate: float = 0.03
    l2: float = 1e-4

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        x = np.asarray(logits, dtype=float)
        y = np.asarray(y_true, dtype=float)
        slope = 1.0
        intercept = 0.0
        n = max(1, len(x))

        for step in range(self.epochs):
            z = slope * x + intercept
            p = sigmoid(z)
            error = p - y
            lr = self.learning_rate / np.sqrt(1.0 + step / 100.0)
            grad_slope = float((error * x).mean() + self.l2 * slope)
            grad_intercept = float(error.mean())
            slope -= lr * grad_slope
            intercept -= lr * grad_intercept

            if not np.isfinite(slope + intercept):
                slope, intercept = 1.0, 0.0
                break

            if n < 10:
                break

        self.slope_ = float(slope)
        self.intercept_ = float(intercept)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return clip_probabilities(
            sigmoid(self.slope_ * np.asarray(logits, dtype=float) + self.intercept_)
        )


class IsotonicCalibrator:
    """Monotonic non-parametric calibration via the pool adjacent violators algorithm."""

    def __init__(self) -> None:
        self.thresholds_: np.ndarray | None = None
        self.values_: np.ndarray | None = None

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        x = np.asarray(probs, dtype=float)
        y = np.asarray(y_true, dtype=float)
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        blocks: list[dict[str, float]] = []
        for xi, yi in zip(x_sorted, y_sorted):
            blocks.append({"sum_y": float(yi), "weight": 1.0, "max_x": float(xi)})
            while len(blocks) >= 2:
                prev = blocks[-2]["sum_y"] / blocks[-2]["weight"]
                curr = blocks[-1]["sum_y"] / blocks[-1]["weight"]
                if prev <= curr:
                    break
                merged = {
                    "sum_y": blocks[-2]["sum_y"] + blocks[-1]["sum_y"],
                    "weight": blocks[-2]["weight"] + blocks[-1]["weight"],
                    "max_x": blocks[-1]["max_x"],
                }
                blocks = blocks[:-2]
                blocks.append(merged)

        self.thresholds_ = np.array([block["max_x"] for block in blocks], dtype=float)
        self.values_ = np.array(
            [block["sum_y"] / block["weight"] for block in blocks],
            dtype=float,
        )
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None or self.values_ is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before prediction.")
        x = np.asarray(probs, dtype=float)
        idx = np.searchsorted(self.thresholds_, x, side="left")
        idx = np.clip(idx, 0, len(self.values_) - 1)
        return clip_probabilities(self.values_[idx])
