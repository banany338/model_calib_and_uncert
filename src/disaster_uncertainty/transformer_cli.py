"""Optional Hugging Face Transformer experiment runner.

This module is intentionally separate from the lightweight TF-IDF pipeline.
It needs optional dependencies (`torch`, `transformers`) and may download model
weights from Hugging Face.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .calibration import IsotonicCalibrator, TemperatureScaler
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
from .modeling import sigmoid
from .plots import (
    save_confusion_matrix_grid,
    save_rejection_curve,
    save_rejection_threshold_panels,
    save_error_category_plot,
    save_reliability_diagram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional Transformer fine-tuning and calibration experiment."
    )
    parser.add_argument("--train-csv", default="nlp-getting-started/train.csv")
    parser.add_argument("--output-dir", default="reports_transformers/distilbert")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-calibration", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-large-model",
        action="store_true",
        help="Allow loading very large LLMs such as Llama 3 8B. Requires enough memory and access.",
    )
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def require_transformer_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Import optional heavy dependencies only when this runner is used."""

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Brakuje opcjonalnych zaleznosci do transformerow.\n"
            "Zainstaluj je poleceniem:\n"
            "  pip install -r requirements-transformers.txt\n"
            f"Szczegol techniczny: {exc}"
        ) from exc
    return torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def guard_large_model(model_name: str, allow_large_model: bool) -> None:
    lowered = model_name.lower()
    looks_like_llama_8b = "llama" in lowered and "8b" in lowered
    if looks_like_llama_8b and not allow_large_model:
        raise SystemExit(
            "Ten model wyglada na Llama 3 8B. To model generatywny z okolo 8 mld "
            "parametrow, wiec nie jest dobrym domyslnym wyborem do szybkiego "
            "fine-tuningu klasyfikatora tweetow na CPU. Jesli masz wystarczajacy "
            "sprzet, dostep do wag i swiadomie chcesz sprobowac, dodaj "
            "--allow-large-model."
        )


def limit_list(values: list[str], limit: int) -> list[str]:
    return values[:limit] if limit and limit > 0 else values


def limit_array(values: np.ndarray, limit: int) -> np.ndarray:
    return values[:limit] if limit and limit > 0 else values


def positive_class_probability(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return binary score and class-1 probability from Transformer logits."""

    if logits.ndim == 1 or logits.shape[1] == 1:
        scores = logits.reshape(-1)
        probs = sigmoid(scores)
        return scores, np.asarray(probs, dtype=float)

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs_all = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    scores = logits[:, 1] - logits[:, 0]
    return scores, probs_all[:, 1]


def make_dataset_class(torch: Any, Dataset: Any, tokenizer: Any, max_length: int) -> Any:
    class TweetDataset(Dataset):  # type: ignore[misc, valid-type]
        def __init__(self, texts: list[str], labels: np.ndarray | None = None) -> None:
            self.texts = texts
            self.labels = labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            encoding = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoding.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return item

    return TweetDataset


def collect_logits(model: Any, dataloader: Any, device: Any, torch: Any) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            if labels is not None:
                del labels
            batches.append(outputs.logits.detach().cpu().numpy())
    return np.concatenate(batches, axis=0)


def evaluate_variant(
    name: str,
    y_test: np.ndarray,
    probs: np.ndarray,
    test_texts: list[str],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = classification_metrics(y_test, probs)
    metrics["ece"] = expected_calibration_error(y_test, probs)
    metrics["model"] = name
    bins = reliability_bins(y_test, probs)
    bins.insert(0, "model", name)
    curve = rejection_curve(y_test, probs, thresholds=np.round(np.arange(0.50, 0.96, 0.05), 2))
    curve.insert(0, "model", name)
    errors = high_confidence_errors(test_texts, y_test, probs, top_n=12)
    if not errors.empty:
        errors.insert(0, "model", name)
    return metrics, bins, curve, errors


def run_transformer_experiment(args: argparse.Namespace) -> None:
    guard_large_model(args.model_name, args.allow_large_model)
    torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer = (
        require_transformer_dependencies()
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(torch, args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    if args.demo:
        df = make_demo_dataset(seed=args.seed)
    else:
        df = load_disaster_tweets(args.train_csv)

    split = stratified_train_calibration_test_split(df, seed=args.seed)
    train_texts = limit_list(build_text_input(split.train), args.limit_train)
    calibration_texts = limit_list(build_text_input(split.calibration), args.limit_calibration)
    test_texts = limit_list(build_text_input(split.test), args.limit_test)
    y_train = limit_array(split.train[TARGET_COLUMN].to_numpy(dtype=int), args.limit_train)
    y_calibration = limit_array(
        split.calibration[TARGET_COLUMN].to_numpy(dtype=int),
        args.limit_calibration,
    )
    y_test = limit_array(split.test[TARGET_COLUMN].to_numpy(dtype=int), args.limit_test)

    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        mode = "lokalnym cache" if args.local_files_only else "Hugging Face Hub"
        raise SystemExit(
            f"Nie udalo sie zaladowac tokenizera `{args.model_name}` z {mode}.\n"
            "Dla modeli publicznych sprawdz polaczenie z internetem. Dla modeli "
            "gated, takich jak Llama, potrzebny jest zaakceptowany dostep i HF_TOKEN.\n"
            f"Szczegol techniczny: {exc}"
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        mode = "lokalnego cache" if args.local_files_only else "Hugging Face Hub"
        raise SystemExit(
            f"Nie udalo sie zaladowac modelu `{args.model_name}` z {mode}.\n"
            "Jesli to Llama 3 8B, upewnij sie, ze masz zaakceptowany dostep, "
            "ustawiony HF_TOKEN i wystarczajaco pamieci/GPU.\n"
            f"Szczegol techniczny: {exc}"
        ) from exc
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    TweetDataset = make_dataset_class(torch, Dataset, tokenizer, args.max_length)
    train_loader = DataLoader(
        TweetDataset(train_texts, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    calibration_loader = DataLoader(
        TweetDataset(calibration_texts, y_calibration),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TweetDataset(test_texts, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            batch_size = int(batch["labels"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size
        print(f"epoch {epoch + 1}/{args.epochs}: loss={total_loss / max(1, seen):.4f}")

    calibration_logits = collect_logits(model, calibration_loader, device, torch)
    test_logits = collect_logits(model, test_loader, device, torch)
    calibration_scores, calibration_probs = positive_class_probability(calibration_logits)
    test_scores, raw_probs = positive_class_probability(test_logits)

    temperature_scaler = TemperatureScaler().fit(calibration_scores, y_calibration)
    isotonic_calibrator = IsotonicCalibrator().fit(calibration_probs, y_calibration)
    model_slug = args.model_name.replace("/", "_")
    variants = {
        f"{model_slug}/raw": raw_probs,
        f"{model_slug}/temperature_scaling": temperature_scaler.predict_proba(test_scores),
        f"{model_slug}/isotonic_regression": isotonic_calibrator.predict_proba(raw_probs),
    }

    metric_rows = []
    bin_frames = []
    curve_frames = []
    error_frames = []
    analysis_error_frames = []
    reliability_for_plot = {}
    rejection_for_plot = {}
    for name, probs in variants.items():
        metrics, bins, curve, errors = evaluate_variant(name, y_test, probs, test_texts)
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

    metrics_df = pd.DataFrame(metric_rows).sort_values(["ece", "log_loss"]).reset_index(drop=True)
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
    save_reliability_diagram(
        reliability_for_plot,
        plot_dir / "reliability_transformer.svg",
        title=f"{args.model_name}: reliability diagram",
    )
    save_rejection_curve(
        rejection_for_plot,
        plot_dir / "rejection_transformer.svg",
        title=f"{args.model_name}: mechanizm 'nie wiem'",
    )
    save_rejection_threshold_panels(
        rejection_for_plot,
        plot_dir / "threshold_tradeoff_transformer.svg",
        title=f"{args.model_name}: prog tau, coverage i odrzucenia",
    )
    save_confusion_matrix_grid(
        metrics_df,
        plot_dir / "confusion_transformer.svg",
        title=f"{args.model_name}: macierze TP/TN/FP/FN",
    )
    save_error_category_plot(
        error_category_summary_df,
        plot_dir / "error_categories_transformer.svg",
        title=f"{args.model_name}: typy bledow wysokiej pewnosci",
    )

    elapsed = time.time() - start
    summary = [
        f"# Transformer experiment: {args.model_name}",
        "",
        f"- Device: `{device}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Max length: `{args.max_length}`",
        f"- Runtime seconds: `{elapsed:.1f}`",
        f"- Temperature T: `{temperature_scaler.temperature_:.4f}`",
        "",
        metrics_df[["model", "accuracy", "precision", "recall", "f1", "ece", "log_loss"]]
        .to_string(index=False),
        "",
    ]
    (output_dir / "model_comparison.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"Saved transformer results to: {output_dir.resolve()}")
    print(metrics_df[["model", "accuracy", "f1", "ece", "log_loss"]].to_string(index=False))


def main() -> None:
    run_transformer_experiment(parse_args())


if __name__ == "__main__":
    main()
