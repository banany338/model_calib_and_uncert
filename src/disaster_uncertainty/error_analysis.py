"""Heuristic error categorization for presentation-friendly model analysis."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import confidence_from_proba, labels_from_proba


KEYWORD_RE = re.compile(r"\bkeyword:\s*(.*)$", re.IGNORECASE)

FALSE_ALARM_PATTERNS = [
    "false alarm",
    "hoax",
    "averted",
    "drill",
    "test alarm",
    "not a disaster",
]
FICTION_ENTERTAINMENT_PATTERNS = [
    "movie",
    "film",
    "novel",
    "book",
    "chapter",
    "fan",
    "fans",
    "trailer",
    "episode",
    "game",
    "song",
    "music",
    "ibook",
    "bookboost",
    "fiction",
    "romance",
]
COMMERCE_PATTERNS = [
    "sale",
    "shop",
    "coupon",
    "handbag",
    "purse",
    "ebay",
    "amazon",
    "product",
    "deal",
    "buy",
    "free shipping",
]
METAPHOR_PATTERNS = [
    "on fire",
    "this is fire",
    "drowning in",
    "crushed",
    "blew up",
    "blown away",
    "exploded",
    "dead tired",
    "killed it",
    "disaster for my team",
    "crash again",
]
NEWS_CONTEXT_PATTERNS = [
    "http://",
    "https://",
    "t.co/",
    "via @",
    "reports",
    "news",
    "latest",
    "state actions",
    "potus",
    "policy",
    "verdict",
]
REAL_EVENT_PATTERNS = [
    "evacuated",
    "evacuation",
    "injured",
    "killed",
    "rescue",
    "warning",
    "wildfire",
    "earthquake",
    "flood",
    "hurricane",
    "tornado",
    "explosion",
    "crash",
    "derailment",
]


def extract_keyword(text: str) -> str:
    match = KEYWORD_RE.search(text)
    return match.group(1).strip().lower() if match else ""


def contains_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def categorize_error(text: str, true_label: int, predicted_label: int) -> str:
    """Assign a human-readable error type with simple transparent heuristics."""

    lowered = text.lower()
    keyword = extract_keyword(text)
    token_count = len(re.findall(r"[a-z0-9_#@']+", lowered))

    if true_label == 0 and predicted_label == 1:
        if contains_any(lowered, FALSE_ALARM_PATTERNS):
            return "false_alarm_or_averted"
        if contains_any(lowered, FICTION_ENTERTAINMENT_PATTERNS):
            return "fiction_entertainment"
        if contains_any(lowered, COMMERCE_PATTERNS):
            return "commerce_or_spam"
        if contains_any(lowered, METAPHOR_PATTERNS):
            return "metaphor_or_idiom"
        if contains_any(lowered, NEWS_CONTEXT_PATTERNS):
            return "news_link_or_policy_context"
        if keyword:
            return "keyword_trigger_false_positive"
        return "other_false_positive"

    if true_label == 1 and predicted_label == 0:
        if token_count <= 8:
            return "short_or_low_context_real_disaster"
        if contains_any(lowered, METAPHOR_PATTERNS):
            return "real_disaster_sounded_figurative"
        if contains_any(lowered, NEWS_CONTEXT_PATTERNS):
            return "linked_real_event_underrecognized"
        if contains_any(lowered, REAL_EVENT_PATTERNS):
            return "missed_real_event_terms"
        return "other_false_negative"

    return "correct"


def analyze_prediction_errors(
    model: str,
    texts: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    high_confidence_threshold: float = 0.90,
) -> pd.DataFrame:
    """Return one row per wrong prediction with category and confidence fields."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    pred = labels_from_proba(p)
    confidence = confidence_from_proba(p)
    wrong_indices = np.where(pred != y)[0]

    rows: list[dict[str, object]] = []
    for idx in wrong_indices:
        text = texts[int(idx)]
        true_label = int(y[int(idx)])
        predicted_label = int(pred[int(idx)])
        conf = float(confidence[int(idx)])
        rows.append(
            {
                "model": model,
                "category": categorize_error(text, true_label, predicted_label),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": conf,
                "prob_disaster": float(p[int(idx)]),
                "is_high_confidence": conf >= high_confidence_threshold,
                "keyword": extract_keyword(text),
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def summarize_error_categories(error_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error counts by model and heuristic category."""

    if error_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "category",
                "error_count",
                "high_confidence_error_count",
                "mean_confidence",
                "max_confidence",
            ]
        )

    grouped = (
        error_df.groupby(["model", "category"], as_index=False)
        .agg(
            error_count=("category", "size"),
            high_confidence_error_count=("is_high_confidence", "sum"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
        )
        .sort_values(["model", "high_confidence_error_count", "error_count"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    grouped["high_confidence_error_count"] = grouped["high_confidence_error_count"].astype(int)
    return grouped


def write_error_examples_report(
    error_df: pd.DataFrame,
    path: str | Path,
    max_examples_per_model: int = 8,
) -> None:
    """Write a short Markdown file with categorized high-confidence examples."""

    lines = [
        "# Analiza bledow wedlug typu tweeta",
        "",
        "Kategorie sa heurystyczne i sluza do interpretacji, nie do automatycznej oceny.",
        "",
    ]

    if error_df.empty:
        lines.append("Brak blednych predykcji do opisania.")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        return

    for model in error_df["model"].drop_duplicates():
        model_errors = (
            error_df[error_df["model"] == model]
            .sort_values("confidence", ascending=False)
            .head(max_examples_per_model)
        )
        lines.extend([f"## {model}", ""])
        for _, row in model_errors.iterrows():
            text = str(row["text"]).replace("\n", " ")
            if len(text) > 220:
                text = text[:217] + "..."
            lines.extend(
                [
                    f"- `{row['category']}` | true={row['true_label']} pred={row['predicted_label']} "
                    f"confidence={float(row['confidence']):.3f}",
                    f"  {text}",
                ]
            )
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
