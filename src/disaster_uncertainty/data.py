"""Data loading and splitting utilities for the Kaggle disaster tweets task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TEXT_COLUMN = "text"
TARGET_COLUMN = "target"


@dataclass(frozen=True)
class DatasetSplit:
    """Container returned by the project split function."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def load_disaster_tweets(csv_path: str | Path) -> pd.DataFrame:
    """Load Kaggle train.csv and validate the columns needed by this project."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku {path}. Pobierz train.csv z konkursu Kaggle "
            "'Natural Language Processing with Disaster Tweets' i podaj sciezke "
            "argumentem --train-csv albo uruchom wersje demonstracyjna --demo."
        )

    df = pd.read_csv(path)
    required = {TEXT_COLUMN, TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Brakuje wymaganych kolumn w {path}: {missing}")

    df = df.copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def build_text_input(df: pd.DataFrame) -> list[str]:
    """Build model input from the Kaggle columns without requiring all metadata."""

    text = df[TEXT_COLUMN].fillna("").astype(str)
    parts = [text]

    if "keyword" in df.columns:
        keyword = (
            df["keyword"]
            .fillna("")
            .astype(str)
            .str.replace("%20", " ", regex=False)
            .str.strip()
        )
        parts.append(" keyword: " + keyword)

    return pd.concat(parts, axis=1).agg(" ".join, axis=1).tolist()


def stratified_train_calibration_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    seed: int = 42,
) -> DatasetSplit:
    """Split data into train/calibration/test while preserving class ratios."""

    if train_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("train_fraction i calibration_fraction musza byc dodatnie")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("Suma train_fraction i calibration_fraction musi byc < 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    calibration_indices: list[int] = []
    test_indices: list[int] = []

    for _, class_rows in df.groupby(target_col, sort=True):
        indices = class_rows.index.to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_fraction))
        n_calibration = int(round(n * calibration_fraction))

        train_indices.extend(indices[:n_train])
        calibration_indices.extend(indices[n_train : n_train + n_calibration])
        test_indices.extend(indices[n_train + n_calibration :])

    def take(indices: list[int]) -> pd.DataFrame:
        out = df.loc[indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return out

    return DatasetSplit(
        train=take(train_indices),
        calibration=take(calibration_indices),
        test=take(test_indices),
    )


def make_demo_dataset(seed: int = 42, repeats: int = 45) -> pd.DataFrame:
    """Create a small tweet-like dataset for smoke tests when Kaggle data is absent."""

    rng = np.random.default_rng(seed)
    disaster_templates = [
        "Emergency crews respond to fire after explosion downtown",
        "Flood warning issued as river rises near homes",
        "Several injured after earthquake damages buildings",
        "Wildfire spreads quickly, residents ordered to evacuate",
        "Tornado sirens active, shelter now",
        "Train derailment reported, rescue teams on scene",
        "Bridge collapse leaves cars trapped in water",
        "Hurricane makes landfall with dangerous storm surge",
    ]
    non_disaster_templates = [
        "This new song is fire and I cannot stop replaying it",
        "My inbox exploded after the product launch",
        "Watching a movie about earthquakes tonight",
        "I am drowning in homework but coffee helps",
        "That football match was a total disaster for my team",
        "Need a rescue from this boring meeting",
        "The sale caused chaos at the mall",
        "My phone battery died and ruined my morning",
    ]

    rows: list[dict[str, object]] = []
    for i in range(repeats):
        disaster = disaster_templates[i % len(disaster_templates)]
        non_disaster = non_disaster_templates[i % len(non_disaster_templates)]
        rows.append({"id": len(rows), "keyword": "emergency", "text": disaster, "target": 1})
        rows.append({"id": len(rows), "keyword": "metaphor", "text": non_disaster, "target": 0})

    # Add a few deliberately ambiguous examples so calibration is not trivial.
    ambiguous = [
        ("fire in my heart after seeing the news", 0),
        ("sirens outside after crash near the station", 1),
        ("the market crash destroyed my portfolio", 0),
        ("smoke seen above airport, flights delayed", 1),
        ("this app just crashed again", 0),
        ("families evacuated after gas leak", 1),
    ]
    for text, target in ambiguous * max(2, repeats // 12):
        rows.append({"id": len(rows), "keyword": "", "text": text, "target": target})

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(
        drop=True
    )
