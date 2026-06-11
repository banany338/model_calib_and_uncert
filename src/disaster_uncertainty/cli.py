"""Command-line experiment runner for the disaster tweet uncertainty project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import IsotonicCalibrator, PlattScaler, TemperatureScaler
from .conformal import evaluate_conformal_prediction
from .data import (
    TARGET_COLUMN,
    build_text_input,
    load_disaster_tweets,
    make_demo_dataset,
    stratified_train_calibration_test_split,
)
from .error_analysis import (
    analyze_prediction_errors,
    summarize_error_categories,
    write_error_examples_report,
)
from .metrics import (
    classification_metrics,
    expected_calibration_error,
    high_confidence_errors,
    rejection_curve,
    reliability_bins,
)
from .modeling import (
    LinearSVMConfig,
    EmbeddingMLPConfig,
    LogisticRegressionConfig,
    TextEmbeddingMLP,
    TextLinearSVM,
    TextLogisticEnsemble,
    TextLogisticRegression,
    TextMultinomialNaiveBayes,
)
from .plots import (
    save_confusion_matrix_grid,
    save_conformal_plot,
    save_error_category_plot,
    save_rejection_curve,
    save_rejection_threshold_panels,
    save_reliability_diagram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration and uncertainty analysis for disaster tweet classification."
    )
    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Path to Kaggle train.csv with columns text and target.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where CSV summaries and SVG plots will be written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=15_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--mlp-epochs", type=int, default=24)
    parser.add_argument(
        "--skip-extra-models",
        action="store_true",
        help="Run only the original TF-IDF logistic regression family.",
    )
    parser.add_argument(
        "--include-platt",
        action="store_true",
        help="Also run Platt scaling. Useful, but not one of the examples listed in the assignment PDF.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a generated tweet-like dataset for smoke testing without Kaggle files.",
    )
    return parser.parse_args()


def evaluate_model(
    name: str,
    y_test: np.ndarray,
    probs: np.ndarray,
    test_texts: list[str],
    n_bins: int = 10,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute all project outputs for one model variant."""

    metrics = classification_metrics(y_test, probs)
    metrics["ece"] = expected_calibration_error(y_test, probs, n_bins=n_bins)
    metrics["model"] = name

    bins = reliability_bins(y_test, probs, n_bins=n_bins)
    bins.insert(0, "model", name)

    thresholds = np.round(np.arange(0.50, 0.96, 0.05), 2)
    curve = rejection_curve(y_test, probs, thresholds=thresholds)
    curve.insert(0, "model", name)

    errors = high_confidence_errors(test_texts, y_test, probs, top_n=12)
    if not errors.empty:
        errors.insert(0, "model", name)
    return metrics, bins, curve, errors


def write_markdown_report(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    best_ece_model: str,
    calibration_notes: list[str],
    plot_files: list[str],
) -> None:
    """Write a compact report that can be copied into presentation notes."""

    columns = ["model", "accuracy", "precision", "recall", "f1", "ece", "log_loss", "brier_score"]
    table = metrics_df[columns].copy()
    for col in columns[1:]:
        table[col] = table[col].map(lambda value: f"{value:.4f}")
    markdown_table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        markdown_table.append("| " + " | ".join(str(row[col]) for col in columns) + " |")

    lines = [
        "# Disaster Tweets: kalibracja i niepewnosc",
        "",
        "## Podsumowanie modeli",
        "",
        "\n".join(markdown_table),
        "",
        "## Parametry kalibratorow",
        "",
        *calibration_notes,
        f"- Najnizsze ECE na zbiorze testowym: **{best_ece_model}**",
        "",
        "## Pliki wynikowe",
        "",
        "- `metrics_summary.csv` - glowne metryki klasyfikacji i kalibracji.",
        "- `reliability_bins.csv` - dane do reliability diagram.",
        "- `high_confidence_errors.csv` - bledne predykcje o najwyzszej pewnosci.",
        "- `error_analysis.csv` - wszystkie bledne predykcje z kategoria bledu.",
        "- `error_category_summary.csv` - agregacja kategorii bledow.",
        "- `error_examples.md` - przyklady bledow wysokiej pewnosci do prezentacji.",
        "- `rejection_curves.csv` - coverage i accuracy dla progow 'nie wiem'.",
        "- `conformal_prediction.csv` - proste split conformal prediction dla surowych modeli.",
        "- `plots/` - osobne wykresy dla kazdej rodziny modelu.",
        "",
        "## Wykresy",
        "",
        *[f"- `{plot_file}`" for plot_file in plot_files],
        "",
    ]
    (output_dir / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def safe_plot_name(name: str) -> str:
    """Convert a model/family name into a filesystem-friendly SVG stem."""

    safe = name.replace("/", "_").replace(" ", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in safe)


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        df = make_demo_dataset(seed=args.seed)
        dataset_source = "generated demo dataset"
    else:
        df = load_disaster_tweets(args.train_csv)
        dataset_source = args.train_csv

    split = stratified_train_calibration_test_split(df, seed=args.seed)
    train_texts = build_text_input(split.train)
    calibration_texts = build_text_input(split.calibration)
    test_texts = build_text_input(split.test)
    y_train = split.train[TARGET_COLUMN].to_numpy(dtype=int)
    y_calibration = split.calibration[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = split.test[TARGET_COLUMN].to_numpy(dtype=int)

    model_probabilities: dict[str, np.ndarray] = {}
    conformal_candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    calibration_notes: list[str] = []

    def add_calibrated_family(
        family_name: str,
        calibration_scores: np.ndarray,
        calibration_probs: np.ndarray,
        test_scores: np.ndarray,
        test_probs: np.ndarray,
        raw_method_name: str = "raw",
    ) -> None:
        model_probabilities[f"{family_name}/{raw_method_name}"] = test_probs
        conformal_candidates[f"{family_name}/{raw_method_name}"] = (
            calibration_probs,
            test_probs,
        )

        temperature_scaler = TemperatureScaler().fit(calibration_scores, y_calibration)
        model_probabilities[f"{family_name}/temperature_scaling"] = (
            temperature_scaler.predict_proba(test_scores)
        )
        calibration_notes.append(
            f"- {family_name}: Temperature Scaling T = {temperature_scaler.temperature_:.4f}"
        )

        isotonic_calibrator = IsotonicCalibrator().fit(calibration_probs, y_calibration)
        model_probabilities[f"{family_name}/isotonic_regression"] = (
            isotonic_calibrator.predict_proba(test_probs)
        )

        if args.include_platt:
            platt_scaler = PlattScaler().fit(calibration_scores, y_calibration)
            model_probabilities[f"{family_name}/platt_scaling_extra"] = (
                platt_scaler.predict_proba(test_scores)
            )
            calibration_notes.append(
                f"- {family_name}: Platt Scaling slope = {platt_scaler.slope_:.4f}, "
                f"intercept = {platt_scaler.intercept_:.4f} (metoda dodatkowa spoza listy PDF)"
            )

    base_config = LogisticRegressionConfig(
        epochs=args.epochs,
        learning_rate=0.35,
        l2=1e-4,
        seed=args.seed,
    )
    logreg = TextLogisticRegression(
        min_df=args.min_df,
        max_features=args.max_features,
        config=base_config,
    )
    logreg.fit(train_texts, y_train)
    add_calibrated_family(
        "logreg_tfidf",
        calibration_scores=logreg.decision_function(calibration_texts),
        calibration_probs=logreg.predict_proba(calibration_texts),
        test_scores=logreg.decision_function(test_texts),
        test_probs=logreg.predict_proba(test_texts),
    )

    smoothing = 0.10
    y_train_smoothed = y_train * (1.0 - smoothing) + 0.5 * smoothing
    smooth_logreg = TextLogisticRegression(
        min_df=args.min_df,
        max_features=args.max_features,
        config=LogisticRegressionConfig(
            epochs=args.epochs,
            learning_rate=0.35,
            l2=1e-4,
            seed=args.seed + 11,
        ),
    )
    smooth_logreg.fit(train_texts, y_train_smoothed)
    model_probabilities[f"logreg_tfidf/label_smoothing_{smoothing:.2f}"] = (
        smooth_logreg.predict_proba(test_texts)
    )
    conformal_candidates[f"logreg_tfidf/label_smoothing_{smoothing:.2f}"] = (
        smooth_logreg.predict_proba(calibration_texts),
        smooth_logreg.predict_proba(test_texts),
    )

    if args.ensemble_size > 0:
        ensemble = TextLogisticEnsemble(
            size=args.ensemble_size,
            min_df=args.min_df,
            max_features=args.max_features,
            epochs=max(10, int(args.epochs * 0.75)),
            seed=args.seed + 100,
        )
        ensemble.fit(train_texts, y_train)
        model_probabilities["logreg_tfidf/bootstrap_ensemble"] = ensemble.predict_proba(
            test_texts
        )
        conformal_candidates["logreg_tfidf/bootstrap_ensemble"] = (
            ensemble.predict_proba(calibration_texts),
            ensemble.predict_proba(test_texts),
        )

    if not args.skip_extra_models:
        naive_bayes = TextMultinomialNaiveBayes(
            min_df=args.min_df,
            max_features=args.max_features,
            alpha=1.0,
        )
        naive_bayes.fit(train_texts, y_train)
        add_calibrated_family(
            "naive_bayes_tfidf",
            calibration_scores=naive_bayes.decision_function(calibration_texts),
            calibration_probs=naive_bayes.predict_proba(calibration_texts),
            test_scores=naive_bayes.decision_function(test_texts),
            test_probs=naive_bayes.predict_proba(test_texts),
        )

        svm = TextLinearSVM(
            min_df=args.min_df,
            max_features=args.max_features,
            config=LinearSVMConfig(
                epochs=max(15, int(args.epochs * 0.80)),
                learning_rate=0.18,
                l2=5e-4,
                seed=args.seed + 21,
            ),
        )
        svm.fit(train_texts, y_train)
        add_calibrated_family(
            "linear_svm_tfidf",
            calibration_scores=svm.decision_function(calibration_texts),
            calibration_probs=svm.predict_proba(calibration_texts),
            test_scores=svm.decision_function(test_texts),
            test_probs=svm.predict_proba(test_texts),
            raw_method_name="raw_margin_sigmoid",
        )

        mlp = TextEmbeddingMLP(
            min_df=args.min_df,
            max_tokens=min(args.max_features, 10_000),
            config=EmbeddingMLPConfig(
                epochs=args.mlp_epochs,
                embedding_dim=32,
                hidden_dim=48,
                learning_rate=0.035,
                l2=1e-5,
                seed=args.seed + 31,
            ),
        )
        mlp.fit(train_texts, y_train)
        add_calibrated_family(
            "embedding_mlp",
            calibration_scores=mlp.decision_function(calibration_texts),
            calibration_probs=mlp.predict_proba(calibration_texts),
            test_scores=mlp.decision_function(test_texts),
            test_probs=mlp.predict_proba(test_texts),
        )

    metric_rows: list[dict[str, float]] = []
    bin_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    analysis_error_frames: list[pd.DataFrame] = []

    reliability_for_plot: dict[str, pd.DataFrame] = {}
    rejection_for_plot: dict[str, pd.DataFrame] = {}

    for name, probs in model_probabilities.items():
        metrics, bins, curve, errors = evaluate_model(name, y_test, probs, test_texts)
        metric_rows.append(metrics)
        bin_frames.append(bins)
        curve_frames.append(curve)
        if not errors.empty:
            error_frames.append(errors)
        model_error_analysis = analyze_prediction_errors(name, test_texts, y_test, probs)
        if not model_error_analysis.empty:
            analysis_error_frames.append(model_error_analysis)
        reliability_for_plot[name] = bins
        rejection_for_plot[name] = curve

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df.sort_values(["ece", "log_loss"]).reset_index(drop=True)
    model_parts = metrics_df["model"].str.split("/", n=1, expand=True)
    metrics_df.insert(0, "base_model", model_parts[0])
    metrics_df.insert(1, "method", model_parts[1].fillna("raw"))
    bins_df = pd.concat(bin_frames, ignore_index=True)
    curves_df = pd.concat(curve_frames, ignore_index=True)
    errors_df = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames
        else pd.DataFrame(
            columns=["model", "text", "true_label", "predicted_label", "confidence", "prob_disaster"]
        )
    )
    error_analysis_df = (
        pd.concat(analysis_error_frames, ignore_index=True)
        if analysis_error_frames
        else pd.DataFrame(
            columns=[
                "model",
                "category",
                "true_label",
                "predicted_label",
                "confidence",
                "prob_disaster",
                "is_high_confidence",
                "keyword",
                "text",
            ]
        )
    )
    error_category_summary_df = summarize_error_categories(error_analysis_df)

    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    bins_df.to_csv(output_dir / "reliability_bins.csv", index=False)
    curves_df.to_csv(output_dir / "rejection_curves.csv", index=False)
    errors_df.to_csv(output_dir / "high_confidence_errors.csv", index=False)
    error_analysis_df.to_csv(output_dir / "error_analysis.csv", index=False)
    error_category_summary_df.to_csv(output_dir / "error_category_summary.csv", index=False)
    write_error_examples_report(error_analysis_df, output_dir / "error_examples.md")

    conformal_frames = [
        evaluate_conformal_prediction(
            model=name,
            y_calibration=y_calibration,
            calibration_probs=calibration_probs,
            y_test=y_test,
            test_probs=test_probs,
            alphas=[0.05, 0.10, 0.15, 0.20, 0.25],
        )
        for name, (calibration_probs, test_probs) in conformal_candidates.items()
    ]
    conformal_df = pd.concat(conformal_frames, ignore_index=True)
    conformal_df.to_csv(output_dir / "conformal_prediction.csv", index=False)

    for stale_name in ["reliability_diagram.svg", "rejection_curve.svg"]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_files: list[str] = []
    family_order = metrics_df["base_model"].drop_duplicates().tolist()
    for family in family_order:
        family_models = metrics_df.loc[metrics_df["base_model"] == family, "model"].tolist()
        reliability_subset = {
            name: reliability_for_plot[name]
            for name in family_models
            if name in reliability_for_plot
        }
        rejection_subset = {
            name: rejection_for_plot[name]
            for name in family_models
            if name in rejection_for_plot
        }
        family_metrics = metrics_df[metrics_df["base_model"] == family]
        safe_family = safe_plot_name(str(family))

        reliability_path = plot_dir / f"reliability_{safe_family}.svg"
        save_reliability_diagram(
            reliability_subset,
            reliability_path,
            title=f"{family}: reliability diagram",
        )
        plot_files.append(str(reliability_path.relative_to(output_dir)))

        rejection_path = plot_dir / f"rejection_{safe_family}.svg"
        save_rejection_curve(
            rejection_subset,
            rejection_path,
            title=f"{family}: mechanizm 'nie wiem'",
        )
        plot_files.append(str(rejection_path.relative_to(output_dir)))

        threshold_path = plot_dir / f"threshold_tradeoff_{safe_family}.svg"
        save_rejection_threshold_panels(
            rejection_subset,
            threshold_path,
            title=f"{family}: prog tau, coverage i odrzucenia",
        )
        plot_files.append(str(threshold_path.relative_to(output_dir)))

        confusion_path = plot_dir / f"confusion_{safe_family}.svg"
        save_confusion_matrix_grid(
            family_metrics,
            confusion_path,
            title=f"{family}: macierze TP/TN/FP/FN",
        )
        plot_files.append(str(confusion_path.relative_to(output_dir)))

        family_error_summary = error_category_summary_df[
            error_category_summary_df["model"].isin(family_models)
        ]
        error_category_path = plot_dir / f"error_categories_{safe_family}.svg"
        save_error_category_plot(
            family_error_summary,
            error_category_path,
            title=f"{family}: typy bledow wysokiej pewnosci",
        )
        plot_files.append(str(error_category_path.relative_to(output_dir)))

    conformal_plot_path = plot_dir / "conformal_prediction.svg"
    save_conformal_plot(
        conformal_df,
        conformal_plot_path,
        title="Split conformal prediction",
    )
    plot_files.append(str(conformal_plot_path.relative_to(output_dir)))

    write_markdown_report(
        output_dir=output_dir,
        metrics_df=metrics_df,
        best_ece_model=str(metrics_df.iloc[0]["model"]),
        calibration_notes=calibration_notes,
        plot_files=plot_files,
    )

    print(f"Dataset: {dataset_source}")
    print(f"Train/calibration/test: {len(split.train)}/{len(split.calibration)}/{len(split.test)}")
    print(f"Saved results to: {output_dir.resolve()}")
    print(metrics_df[["model", "accuracy", "f1", "ece", "log_loss"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
