"""Train and evaluate sentiment classifiers without heavyweight ML packages."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


LABELS = ["negatif", "netral", "positif"]
STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "ini",
    "itu",
    "untuk",
    "dengan",
    "saya",
    "aku",
    "nya",
    "aja",
    "sih",
    "kok",
    "dong",
    "banget",
    "aplikasi",
    "app",
}
POSITIVE_TERMS = {
    "bagus",
    "baik",
    "mantap",
    "mudah",
    "cepat",
    "lancar",
    "suka",
    "puas",
    "membantu",
    "terbaik",
    "top",
    "keren",
    "recommended",
    "rekomendasi",
    "murah",
    "praktis",
    "aman",
    "nyaman",
    "ramah",
    "jelas",
    "bermanfaat",
    "hebat",
    "oke",
    "ok",
    "good",
    "great",
    "nice",
    "love",
    "excellent",
    "worth",
    "stabil",
}
NEGATIVE_TERMS = {
    "buruk",
    "jelek",
    "parah",
    "susah",
    "sulit",
    "lambat",
    "lemot",
    "error",
    "gagal",
    "kecewa",
    "rugi",
    "mahal",
    "ribet",
    "bohong",
    "tipu",
    "penipuan",
    "hilang",
    "hang",
    "crash",
    "bug",
    "macet",
    "lama",
    "buruknya",
    "payah",
    "zonk",
    "sampah",
    "mengecewakan",
    "aneh",
    "blokir",
    "tidak",
    "nggak",
    "ga",
    "gak",
    "kurang",
    "bad",
    "worst",
    "problem",
}


@dataclass(frozen=True)
class Example:
    text: str
    label: str


@dataclass(frozen=True)
class Experiment:
    name: str
    vectorizer: str
    ngram_range: tuple[int, int]
    split_ratio: float
    alpha: float
    remove_stopwords: bool


class TextVectorizer:
    def __init__(
        self,
        mode: str = "count",
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 2,
        max_features: int = 30000,
        remove_stopwords: bool = False,
    ) -> None:
        self.mode = mode
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.remove_stopwords = remove_stopwords
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def _tokens(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
        return tokens

    def _ngrams(self, tokens: list[str]) -> list[str]:
        features: list[str] = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            if len(tokens) < n:
                continue
            features.extend(" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
        features.extend(self._polarity_features(tokens))
        return features

    def _polarity_features(self, tokens: list[str]) -> list[str]:
        positive_score = sum(token in POSITIVE_TERMS for token in tokens)
        negative_score = sum(token in NEGATIVE_TERMS for token in tokens)
        features: list[str] = []

        if positive_score > negative_score:
            features.extend(["__polarity_positive__"] * (positive_score - negative_score))
        elif negative_score > positive_score:
            features.extend(["__polarity_negative__"] * (negative_score - positive_score))
        else:
            features.append("__polarity_neutral__")

        if positive_score:
            features.append("__has_positive_term__")
        if negative_score:
            features.append("__has_negative_term__")
        return features

    def fit(self, texts: list[str]) -> None:
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()

        for text in texts:
            features = self._ngrams(self._tokens(text))
            term_frequency.update(features)
            document_frequency.update(set(features))

        kept = [
            feature
            for feature, count in term_frequency.most_common()
            if document_frequency[feature] >= self.min_df
        ][: self.max_features]
        self.vocabulary = {feature: index for index, feature in enumerate(kept)}

        total_documents = len(texts)
        self.idf = {
            feature: math.log((1 + total_documents) / (1 + document_frequency[feature])) + 1
            for feature in self.vocabulary
        }

    def transform_one(self, text: str) -> dict[str, float]:
        counts = Counter(
            feature
            for feature in self._ngrams(self._tokens(text))
            if feature in self.vocabulary
        )
        if self.mode == "count":
            return dict(counts)

        total_terms = sum(counts.values()) or 1
        return {
            feature: (count / total_terms) * self.idf.get(feature, 1.0)
            for feature, count in counts.items()
        }


class MultinomialNaiveBayes:
    def __init__(self, labels: list[str], alpha: float = 1.0) -> None:
        self.labels = labels
        self.alpha = alpha
        self.class_log_prior: dict[str, float] = {}
        self.feature_log_prob: dict[str, dict[str, float]] = {}
        self.default_log_prob: dict[str, float] = {}

    def fit(self, vectors: list[dict[str, float]], y: list[str], vocabulary: dict[str, int]) -> None:
        class_counts = Counter(y)
        total_examples = len(y)
        vocabulary_size = max(1, len(vocabulary))
        feature_sums: dict[str, defaultdict[str, float]] = {
            label: defaultdict(float) for label in self.labels
        }
        totals = {label: 0.0 for label in self.labels}

        for vector, label in zip(vectors, y):
            for feature, value in vector.items():
                feature_sums[label][feature] += value
                totals[label] += value

        for label in self.labels:
            self.class_log_prior[label] = math.log((class_counts[label] + self.alpha) / (total_examples + self.alpha * len(self.labels)))
            denominator = totals[label] + self.alpha * vocabulary_size
            self.default_log_prob[label] = math.log(self.alpha / denominator)
            self.feature_log_prob[label] = {
                feature: math.log((feature_sums[label][feature] + self.alpha) / denominator)
                for feature in vocabulary
            }

    def predict_one(self, vector: dict[str, float]) -> str:
        scores: dict[str, float] = {}
        for label in self.labels:
            score = self.class_log_prior[label]
            for feature, value in vector.items():
                score += value * self.feature_log_prob[label].get(
                    feature,
                    self.default_log_prob[label],
                )
            scores[label] = score
        return max(scores, key=scores.get)

    def predict(self, vectors: list[dict[str, float]]) -> list[str]:
        return [self.predict_one(vector) for vector in vectors]


def load_dataset(path: Path) -> list[Example]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        examples = [
            Example(text=row["content"], label=row["sentiment"])
            for row in reader
            if row.get("content") and row.get("sentiment") in LABELS
        ]
    return examples


def stratified_split(
    examples: list[Example],
    train_ratio: float,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    grouped: dict[str, list[Example]] = {label: [] for label in LABELS}
    for example in examples:
        grouped[example.label].append(example)

    train: list[Example] = []
    test: list[Example] = []
    for label_examples in grouped.values():
        rng.shuffle(label_examples)
        split_index = int(len(label_examples) * train_ratio)
        train.extend(label_examples[:split_index])
        test.extend(label_examples[split_index:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    matrix = {label: {predicted: 0 for predicted in LABELS} for label in LABELS}
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def train_experiment(
    examples: list[Example],
    experiment: Experiment,
) -> dict[str, object]:
    train, test = stratified_split(examples, train_ratio=experiment.split_ratio)
    vectorizer = TextVectorizer(
        mode=experiment.vectorizer,
        ngram_range=experiment.ngram_range,
        remove_stopwords=experiment.remove_stopwords,
    )
    vectorizer.fit([example.text for example in train])

    x_train = [vectorizer.transform_one(example.text) for example in train]
    x_test = [vectorizer.transform_one(example.text) for example in test]
    y_train = [example.label for example in train]
    y_test = [example.label for example in test]

    model = MultinomialNaiveBayes(labels=LABELS, alpha=experiment.alpha)
    model.fit(x_train, y_train, vectorizer.vocabulary)
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    return {
        "experiment": experiment,
        "vectorizer": vectorizer,
        "model": model,
        "train_size": len(train),
        "test_size": len(test),
        "vocabulary_size": len(vectorizer.vocabulary),
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "confusion_matrix": confusion_matrix(y_test, test_predictions),
    }


def predict_sentiment(text: str, vectorizer: TextVectorizer, model: MultinomialNaiveBayes) -> str:
    return model.predict_one(vectorizer.transform_one(text))


def default_experiments() -> list[Experiment]:
    return [
        Experiment("Count unigram NB 80/20", "count", (1, 1), 0.80, 1.0, False),
        Experiment("TF-IDF unigram+bigram NB 80/20", "tfidf", (1, 2), 0.80, 0.5, True),
        Experiment("TF-IDF unigram+bigram NB 70/30", "tfidf", (1, 2), 0.70, 0.8, False),
    ]


def label_distribution(examples: list[Example]) -> dict[str, int]:
    counts = Counter(example.label for example in examples)
    return {label: counts[label] for label in LABELS}


def run_all(dataset_path: Path) -> list[dict[str, object]]:
    examples = load_dataset(dataset_path)
    return [train_experiment(examples, experiment) for experiment in default_experiments()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment experiments.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("google_play_reviews_sentiment.csv"),
        help="CSV dataset path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_dataset(args.dataset)
    print(f"Dataset rows: {len(examples)}")
    print(f"Label distribution: {label_distribution(examples)}")

    best_result: dict[str, object] | None = None
    for result in [train_experiment(examples, experiment) for experiment in default_experiments()]:
        experiment = result["experiment"]
        assert isinstance(experiment, Experiment)
        print(f"\n{experiment.name}")
        print(f"Train size: {result['train_size']}")
        print(f"Test size: {result['test_size']}")
        print(f"Vocabulary size: {result['vocabulary_size']}")
        print(f"Training accuracy: {result['train_accuracy']:.4f}")
        print(f"Testing accuracy: {result['test_accuracy']:.4f}")
        print(f"Confusion matrix: {result['confusion_matrix']}")

        if best_result is None or result["test_accuracy"] > best_result["test_accuracy"]:
            best_result = result

    if best_result is not None:
        sample_text = "Aplikasinya mudah dipakai, transaksi cepat, dan sangat membantu."
        prediction = predict_sentiment(
            sample_text,
            best_result["vectorizer"],
            best_result["model"],
        )
        print("\nInference example")
        print(f"Text: {sample_text}")
        print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
