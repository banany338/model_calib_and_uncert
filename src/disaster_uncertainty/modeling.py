"""A small dependency-light TF-IDF + logistic regression implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections import Counter
from typing import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9_#@']+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")

SparseRow = dict[int, float]


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""

    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    positive = arr >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_values = np.exp(arr[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    if np.isscalar(values):
        return float(out)
    return out


def tokenize(text: str) -> list[str]:
    """Normalize tweet text and return simple word-like tokens."""

    text = URL_RE.sub(" urltoken ", text.lower())
    text = USER_RE.sub(" usertoken ", text)
    return TOKEN_RE.findall(text)


class TfidfVectorizer:
    """Minimal TF-IDF vectorizer with word n-grams and sparse-row output."""

    def __init__(
        self,
        min_df: int = 2,
        max_features: int = 15_000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, texts: Iterable[str]) -> "TfidfVectorizer":
        texts = list(texts)
        document_frequency: Counter[str] = Counter()
        total_frequency: Counter[str] = Counter()

        for text in texts:
            terms = self._terms(text)
            total_frequency.update(terms)
            document_frequency.update(set(terms))

        candidates = [
            term for term, df in document_frequency.items() if df >= self.min_df
        ]
        candidates.sort(key=lambda term: (-total_frequency[term], term))
        if self.max_features:
            candidates = candidates[: self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(candidates)}
        n_docs = max(1, len(texts))
        idf = np.zeros(len(self.vocabulary_), dtype=float)
        for term, idx in self.vocabulary_.items():
            idf[idx] = math.log((1.0 + n_docs) / (1.0 + document_frequency[term])) + 1.0
        self.idf_ = idf
        return self

    def transform(self, texts: Iterable[str]) -> list[SparseRow]:
        if self.idf_ is None:
            raise RuntimeError("Vectorizer must be fitted before transform().")

        rows: list[SparseRow] = []
        for text in texts:
            counts: Counter[int] = Counter()
            for term in self._terms(text):
                idx = self.vocabulary_.get(term)
                if idx is not None:
                    counts[idx] += 1

            row: SparseRow = {}
            norm_sq = 0.0
            for idx, count in counts.items():
                value = (1.0 + math.log(count)) * float(self.idf_[idx])
                row[idx] = value
                norm_sq += value * value

            if norm_sq > 0:
                norm = math.sqrt(norm_sq)
                row = {idx: value / norm for idx, value in row.items()}
            rows.append(row)
        return rows

    def fit_transform(self, texts: Iterable[str]) -> list[SparseRow]:
        texts = list(texts)
        self.fit(texts)
        return self.transform(texts)

    def _terms(self, text: str) -> list[str]:
        tokens = tokenize(text)
        min_n, max_n = self.ngram_range
        terms: list[str] = []
        for n in range(min_n, max_n + 1):
            if n <= 0 or len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                terms.append(" ".join(tokens[i : i + n]))
        return terms


@dataclass
class LogisticRegressionConfig:
    epochs: int = 45
    learning_rate: float = 0.35
    l2: float = 1e-4
    seed: int = 42


class LogisticRegressionSGD:
    """Binary logistic regression trained with sparse stochastic gradient descent."""

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        self.config = config or LogisticRegressionConfig()
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, rows: list[SparseRow], y: np.ndarray, n_features: int) -> "LogisticRegressionSGD":
        rng = np.random.default_rng(self.config.seed)
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        y = np.asarray(y, dtype=float)
        indices = np.arange(len(rows))
        n = max(1, len(rows))

        for epoch in range(self.config.epochs):
            rng.shuffle(indices)
            lr = self.config.learning_rate / math.sqrt(1.0 + epoch)
            l2_step = lr * self.config.l2 / n

            for idx in indices:
                row = rows[int(idx)]
                logit = self._decision_one(row)
                error = float(sigmoid(logit) - y[int(idx)])

                for feature_idx, value in row.items():
                    assert self.weights_ is not None
                    self.weights_[feature_idx] -= lr * error * value

                if self.config.l2 > 0:
                    assert self.weights_ is not None
                    touched = list(row.keys())
                    self.weights_[touched] *= 1.0 - l2_step

                self.bias_ -= lr * error

        return self

    def decision_function(self, rows: list[SparseRow]) -> np.ndarray:
        return np.array([self._decision_one(row) for row in rows], dtype=float)

    def predict_proba(self, rows: list[SparseRow]) -> np.ndarray:
        return sigmoid(self.decision_function(rows))

    def _decision_one(self, row: SparseRow) -> float:
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        score = self.bias_
        for idx, value in row.items():
            score += self.weights_[idx] * value
        return float(score)


class TextLogisticRegression:
    """End-to-end text classifier: TF-IDF vectorizer followed by logistic regression."""

    def __init__(
        self,
        min_df: int = 2,
        max_features: int = 15_000,
        ngram_range: tuple[int, int] = (1, 2),
        config: LogisticRegressionConfig | None = None,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.model = LogisticRegressionSGD(config)

    def fit(self, texts: Iterable[str], y: np.ndarray) -> "TextLogisticRegression":
        texts = list(texts)
        rows = self.vectorizer.fit_transform(texts)
        self.model.fit(rows, np.asarray(y), n_features=len(self.vectorizer.vocabulary_))
        return self

    def decision_function(self, texts: Iterable[str]) -> np.ndarray:
        rows = self.vectorizer.transform(list(texts))
        return self.model.decision_function(rows)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        rows = self.vectorizer.transform(list(texts))
        return self.model.predict_proba(rows)


class MultinomialNaiveBayesSparse:
    """Multinomial Naive Bayes for positive sparse text features."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None

    def fit(
        self,
        rows: list[SparseRow],
        y: np.ndarray,
        n_features: int,
    ) -> "MultinomialNaiveBayesSparse":
        y = np.asarray(y, dtype=int)
        feature_sum = np.full((2, n_features), self.alpha, dtype=float)
        feature_total = np.full(2, self.alpha * n_features, dtype=float)
        class_count = np.bincount(y, minlength=2).astype(float)

        for row, label in zip(rows, y):
            for idx, value in row.items():
                feature_sum[label, idx] += value
                feature_total[label] += value

        self.class_log_prior_ = np.log((class_count + self.alpha) / (len(y) + 2 * self.alpha))
        self.feature_log_prob_ = np.log(feature_sum / feature_total[:, None])
        return self

    def decision_function(self, rows: list[SparseRow]) -> np.ndarray:
        if self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("Model must be fitted before prediction.")

        log_odds = self.class_log_prior_[1] - self.class_log_prior_[0]
        feature_log_odds = self.feature_log_prob_[1] - self.feature_log_prob_[0]
        scores = []
        for row in rows:
            score = float(log_odds)
            for idx, value in row.items():
                score += float(value * feature_log_odds[idx])
            scores.append(score)
        return np.array(scores, dtype=float)

    def predict_proba(self, rows: list[SparseRow]) -> np.ndarray:
        return sigmoid(self.decision_function(rows))


class TextMultinomialNaiveBayes:
    """End-to-end TF-IDF + Multinomial Naive Bayes text classifier."""

    def __init__(
        self,
        min_df: int = 2,
        max_features: int = 15_000,
        ngram_range: tuple[int, int] = (1, 2),
        alpha: float = 1.0,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.model = MultinomialNaiveBayesSparse(alpha=alpha)

    def fit(self, texts: Iterable[str], y: np.ndarray) -> "TextMultinomialNaiveBayes":
        texts = list(texts)
        rows = self.vectorizer.fit_transform(texts)
        self.model.fit(rows, np.asarray(y), n_features=len(self.vectorizer.vocabulary_))
        return self

    def decision_function(self, texts: Iterable[str]) -> np.ndarray:
        rows = self.vectorizer.transform(list(texts))
        return self.model.decision_function(rows)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        rows = self.vectorizer.transform(list(texts))
        return self.model.predict_proba(rows)


@dataclass
class LinearSVMConfig:
    epochs: int = 35
    learning_rate: float = 0.20
    l2: float = 5e-4
    seed: int = 42


class LinearSVMSGD:
    """Linear SVM trained with hinge loss; its margins are not probabilities."""

    def __init__(self, config: LinearSVMConfig | None = None) -> None:
        self.config = config or LinearSVMConfig()
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, rows: list[SparseRow], y: np.ndarray, n_features: int) -> "LinearSVMSGD":
        rng = np.random.default_rng(self.config.seed)
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        y_signed = np.where(np.asarray(y, dtype=int) == 1, 1.0, -1.0)
        indices = np.arange(len(rows))
        n = max(1, len(rows))

        for epoch in range(self.config.epochs):
            rng.shuffle(indices)
            lr = self.config.learning_rate / math.sqrt(1.0 + epoch)
            shrink = max(0.0, 1.0 - lr * self.config.l2 / n)
            for idx in indices:
                row = rows[int(idx)]
                assert self.weights_ is not None
                touched = list(row.keys())
                self.weights_[touched] *= shrink

                label = y_signed[int(idx)]
                margin = label * self._decision_one(row)
                if margin < 1.0:
                    for feature_idx, value in row.items():
                        self.weights_[feature_idx] += lr * label * value
                    self.bias_ += lr * label
        return self

    def decision_function(self, rows: list[SparseRow]) -> np.ndarray:
        return np.array([self._decision_one(row) for row in rows], dtype=float)

    def predict_proba(self, rows: list[SparseRow]) -> np.ndarray:
        return sigmoid(self.decision_function(rows))

    def _decision_one(self, row: SparseRow) -> float:
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        score = self.bias_
        for idx, value in row.items():
            score += self.weights_[idx] * value
        return float(score)


class TextLinearSVM:
    """End-to-end TF-IDF + linear SVM model."""

    def __init__(
        self,
        min_df: int = 2,
        max_features: int = 15_000,
        ngram_range: tuple[int, int] = (1, 2),
        config: LinearSVMConfig | None = None,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.model = LinearSVMSGD(config)

    def fit(self, texts: Iterable[str], y: np.ndarray) -> "TextLinearSVM":
        texts = list(texts)
        rows = self.vectorizer.fit_transform(texts)
        self.model.fit(rows, np.asarray(y), n_features=len(self.vectorizer.vocabulary_))
        return self

    def decision_function(self, texts: Iterable[str]) -> np.ndarray:
        rows = self.vectorizer.transform(list(texts))
        return self.model.decision_function(rows)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        rows = self.vectorizer.transform(list(texts))
        return self.model.predict_proba(rows)


@dataclass
class EmbeddingMLPConfig:
    embedding_dim: int = 32
    hidden_dim: int = 48
    epochs: int = 24
    learning_rate: float = 0.035
    l2: float = 1e-5
    seed: int = 42


class TextEmbeddingMLP:
    """Average trainable token embeddings followed by a one-hidden-layer MLP."""

    def __init__(
        self,
        min_df: int = 2,
        max_tokens: int = 10_000,
        config: EmbeddingMLPConfig | None = None,
    ) -> None:
        self.min_df = min_df
        self.max_tokens = max_tokens
        self.config = config or EmbeddingMLPConfig()
        self.vocabulary_: dict[str, int] = {}
        self.embeddings_: np.ndarray | None = None
        self.w1_: np.ndarray | None = None
        self.b1_: np.ndarray | None = None
        self.w2_: np.ndarray | None = None
        self.b2_: float = 0.0

    def fit(self, texts: Iterable[str], y: np.ndarray) -> "TextEmbeddingMLP":
        texts = list(texts)
        y = np.asarray(y, dtype=float)
        token_ids = self._fit_vocabulary_and_encode(texts)
        rng = np.random.default_rng(self.config.seed)
        vocab_size = max(1, len(self.vocabulary_))

        self.embeddings_ = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(vocab_size, self.config.embedding_dim),
        )
        self.w1_ = rng.normal(
            loc=0.0,
            scale=0.08,
            size=(self.config.embedding_dim, self.config.hidden_dim),
        )
        self.b1_ = np.zeros(self.config.hidden_dim, dtype=float)
        self.w2_ = rng.normal(loc=0.0, scale=0.08, size=self.config.hidden_dim)
        self.b2_ = 0.0

        indices = np.arange(len(token_ids))
        for epoch in range(self.config.epochs):
            rng.shuffle(indices)
            lr = self.config.learning_rate / math.sqrt(1.0 + epoch)
            for idx in indices:
                self._train_one(token_ids[int(idx)], float(y[int(idx)]), lr)
        return self

    def decision_function(self, texts: Iterable[str]) -> np.ndarray:
        encoded = self._encode_texts(list(texts))
        return np.array([self._forward(ids)[0] for ids in encoded], dtype=float)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        return sigmoid(self.decision_function(texts))

    def _fit_vocabulary_and_encode(self, texts: list[str]) -> list[list[int]]:
        document_frequency: Counter[str] = Counter()
        total_frequency: Counter[str] = Counter()
        tokenized = [tokenize(text) for text in texts]
        for tokens in tokenized:
            total_frequency.update(tokens)
            document_frequency.update(set(tokens))

        candidates = [
            token for token, df in document_frequency.items() if df >= self.min_df
        ]
        candidates.sort(key=lambda token: (-total_frequency[token], token))
        candidates = candidates[: self.max_tokens]
        self.vocabulary_ = {token: idx for idx, token in enumerate(candidates)}
        return [[self.vocabulary_[token] for token in tokens if token in self.vocabulary_] for tokens in tokenized]

    def _encode_texts(self, texts: list[str]) -> list[list[int]]:
        return [
            [self.vocabulary_[token] for token in tokenize(text) if token in self.vocabulary_]
            for text in texts
        ]

    def _document_embedding(self, ids: list[int]) -> np.ndarray:
        assert self.embeddings_ is not None
        if not ids:
            return np.zeros(self.config.embedding_dim, dtype=float)
        return self.embeddings_[ids].mean(axis=0)

    def _forward(self, ids: list[int]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        assert self.w1_ is not None and self.b1_ is not None and self.w2_ is not None
        doc = self._document_embedding(ids)
        hidden_pre = doc @ self.w1_ + self.b1_
        hidden = np.maximum(hidden_pre, 0.0)
        logit = float(hidden @ self.w2_ + self.b2_)
        return logit, doc, hidden_pre, hidden

    def _train_one(self, ids: list[int], target: float, lr: float) -> None:
        assert (
            self.embeddings_ is not None
            and self.w1_ is not None
            and self.b1_ is not None
            and self.w2_ is not None
        )
        logit, doc, hidden_pre, hidden = self._forward(ids)
        error = float(sigmoid(logit) - target)

        old_w1 = self.w1_.copy()
        old_w2 = self.w2_.copy()
        grad_w2 = hidden * error + self.config.l2 * self.w2_
        grad_b2 = error
        grad_hidden = old_w2 * error
        grad_hidden_pre = grad_hidden * (hidden_pre > 0)
        grad_w1 = np.outer(doc, grad_hidden_pre) + self.config.l2 * self.w1_
        grad_b1 = grad_hidden_pre
        grad_doc = old_w1 @ grad_hidden_pre

        self.w2_ -= lr * grad_w2
        self.b2_ -= lr * grad_b2
        self.w1_ -= lr * grad_w1
        self.b1_ -= lr * grad_b1

        if ids:
            embedding_step = lr * grad_doc / len(ids)
            for token_id in ids:
                self.embeddings_[token_id] -= embedding_step


class TextLogisticEnsemble:
    """Bootstrap ensemble used as a simple uncertainty-estimation baseline."""

    def __init__(
        self,
        size: int = 5,
        min_df: int = 2,
        max_features: int = 15_000,
        epochs: int = 35,
        seed: int = 42,
    ) -> None:
        self.size = size
        self.min_df = min_df
        self.max_features = max_features
        self.epochs = epochs
        self.seed = seed
        self.models_: list[TextLogisticRegression] = []

    def fit(self, texts: list[str], y: np.ndarray) -> "TextLogisticEnsemble":
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        y = np.asarray(y)
        for member in range(self.size):
            indices = rng.integers(0, len(texts), size=len(texts))
            sampled_texts = [texts[int(i)] for i in indices]
            sampled_y = y[indices]
            config = LogisticRegressionConfig(
                epochs=self.epochs,
                learning_rate=0.30,
                l2=2e-4,
                seed=self.seed + member + 1,
            )
            model = TextLogisticRegression(
                min_df=self.min_df,
                max_features=self.max_features,
                config=config,
            )
            model.fit(sampled_texts, sampled_y)
            self.models_.append(model)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("Ensemble must be fitted before prediction.")
        probs = np.vstack([model.predict_proba(texts) for model in self.models_])
        return probs.mean(axis=0)
