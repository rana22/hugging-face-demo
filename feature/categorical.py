from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .doc_alignment import DocAlignmentModel
from schema import NodeSchema

MISSING_STRINGS = {"", "na", "n/a", "none", "null", "nan", "not reported"}
IDENTIFIER_HINTS = ("_id", "_record_id", "uuid", "crdc_id")

CATEGORICAL_WEIGHT_PROFILES = {
    "human_relevance": {
        "support": 0.001,
        "predictive_strength": 0.001,
        "determinism": 0.001,
        "stability": 0.001,
        "doc_alignment": 0.8,
    },
    "sample": {
        "support": 0.25,
        "predictive_strength": 0.30,
        "determinism": 0.20,
        "stability": 0.15,
        "doc_alignment": 0.10,
    },
    "diagnosis": {
        "support": 0.25,
        "predictive_strength": 0.30,
        "determinism": 0.20,
        "stability": 0.15,
        "doc_alignment": 0.10,
    },
    "file": {
        "support": 0.20,
        "predictive_strength": 0.20,
        "determinism": 0.20,
        "stability": 0.10,
        "doc_alignment": 0.20,
    },
    "study": {
        "support": 0.20,
        "predictive_strength": 0.20,
        "determinism": 0.15,
        "stability": 0.20,
        "doc_alignment": 0.25,
    },
}

def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and value.strip().lower() in MISSING_STRINGS


def normalize_value(value: Any) -> str:
    if is_missing(value):
        return ""

    # 🔥 handle list-like values
    if isinstance(value, list):
        return " ".join(str(v) for v in value)

    # 🔥 handle JSON-encoded lists (your case)
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return " ".join(str(x) for x in parsed)
            except Exception:
                pass
        return v

    return str(value).strip()

# keeps only columns a and b
# converts missing-like values to empty strings
# drops rows where either side is missing
def prepare_pair_frame(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    pair = df[[a, b]].copy()
    pair[a] = pair[a].map(normalize_value)
    pair[b] = pair[b].map(normalize_value)
    pair = pair[(pair[a] != "") & (pair[b] != "")]
    return pair.reset_index(drop=True)

# Intuition 
# It computes how well you can predict b from a by always choosing the most frequent b 
# for each value of a (a1, a2.., an), weighted by how often each a occurs.

# If each value of a maps to one consistent value of b → score ≈ 1.0
# If each a maps to many different b values → score ≈ low

# This is a soft functional dependency measure:
# 1.0 → a fully determines b
# ~0.7 → mostly determines
# ~0.5 → weak relationship
# low → independent

def conditional_determinism(pair: pd.DataFrame, a: str, b: str) -> float:
    if pair.empty:
        return 0.0

    total = len(pair)
    weighted_max_probs = []

    # all rows where a = some_value.
    for _, group in pair.groupby(a):
        # distribution of b
        # a1 = b1 (.67)
        # a1 = b2 (.33)
        counts = group[b].value_counts(normalize=True)

        if counts.empty:
            continue
        # Weight by how common this a is 
        weight = len(group) / total
        # counts.max() Take the most common b
        # Weight by how common this a (if a1 is more, the values matter more)
        weighted_max_probs.append(weight * float(counts.max()))

    # It is essentially:
    # Expected accuracy if you predict b using the most common value for each a (a1, a2..)
    return float(np.clip(sum(weighted_max_probs), 0.0, 1.0))

# Intuition
# baseline = how well you do by guessing the most common b
# accuracy = how well logistic regression does using a
# predictive_strength = how much better the model is than baseline, scaled to 0–1
# So
# 0.0 means a does not help predict b
# higher values mean stronger predictive relationship
# 1.0 would mean near-perfect predictive improvement over baseline

def predictive_strength_from_holdout(
    pair: pd.DataFrame,
    a: str,
    b: str,
    *,
    seed: int = 42,
) -> tuple[float, float, float, int, int]:
    # if there is no data, or either column has fewer than 2 unique values, 
    # it cannot learn anything useful, so it returns zeros. 
    if pair.empty or pair[a].nunique() < 2 or pair[b].nunique() < 2:
        return 0.0, 0.0, 0.0, 0, 0
    
    # prepare inputs for training 
    X = pair[[a]].astype(str)
    y = pair[b].astype(str)

    # If the dataset is very small:
    # it skips modeling and just uses the majority-class rate as 
    # both predictive strength and accuracy. That is a shortcut 
    # because a train/test split on tiny data would be unstable.
    if len(pair) < 8:
        baseline = float(y.value_counts(normalize=True).max())
        return baseline, baseline, 0.0, len(pair), 0

    # traning and test data
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, shuffle=True
        )

    # If the split or class balance is still not usable, it falls back again:
    if y_train.nunique() < 2 or y_test.empty:
        baseline = float(y_test.value_counts(normalize=True).max()) if len(y_test) else float(y.value_counts(normalize=True).max())
        return baseline, baseline, 0.0, len(X_train), len(X_test)

    # builds a very simple model
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), [a])],
        remainder="drop",
    )

    # pipeline - logistic regression model
    # 1. one-hot encode a
    # 2. fit logistic regression to predict b
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    # train 
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # evaluate test accuracy
    accuracy = float(accuracy_score(y_test, pred))

    # computes the baseline accuracy
    majority = y_train.value_counts().idxmax()
    baseline_pred = np.full(len(y_test), majority)
    baseline = float(accuracy_score(y_test, baseline_pred))

    #return result sets
    predictive_strength = 0.0 if baseline >= 1.0 else max(0.0, (accuracy - baseline) / max(1e-9, 1.0 - baseline))
    return predictive_strength, accuracy, baseline, len(X_train), len(X_test)


def stability_from_resamples(pair: pd.DataFrame, a: str, b: str, *, n_splits: int = 5) -> float:
    if pair.empty or len(pair) < 8:
        return 0.0

    scores: list[float] = []
    for seed in range(n_splits):
        score, _, _, _, _ = predictive_strength_from_holdout(pair, a, b, seed=seed)
        scores.append(score)

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    if mean_score <= 1e-9:
        return 0.0
    return float(np.clip(mean_score / (mean_score + std_score + 1e-9), 0.0, 1.0))

def get_categorical_weights(node_name: str) -> dict[str, float] | None:
    key = (node_name or "").lower()
    return CATEGORICAL_WEIGHT_PROFILES.get(key)

@dataclass
class CategoricalFeatureAnalyzer:
    node_schema: NodeSchema
    doc_model: DocAlignmentModel
    weights: dict[str, float] = field(init=False)

    @staticmethod
    def classify_strength(score: float) -> str:
        if score >= 0.9:
            return "functional"
        if score >= 0.7:
            return "strong"
        if score >= 0.45:
            return "conditional"
        if score >= 0.2:
            return "weak"
        return "independent"

    def __post_init__(self):
        object.__setattr__(self, "weights", get_categorical_weights(self.node_schema.name))

    def analyze(self, df: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
        if self.weights is None:
            return None
        pair = prepare_pair_frame(df, a, b)
        total_rows = len(df)
        row_count = len(pair)
        
        # How much of the original table actually has usable a/b pairs
        support = float(row_count / total_rows) if total_rows else 0.0
        # How well a predicts b using a simple holdout classifier
        predictive_strength, heldout_accuracy, baseline_accuracy, train_rows, test_rows = predictive_strength_from_holdout(pair, a, b)
        
        # “when a is known, how often is b basically fixed?”
        determinism = conditional_determinism(pair, a, b)
        
        stability = stability_from_resamples(pair, a, b)
        doc_alignment = float(self.doc_model.score(a, b))

        strength = (
            self.weights["support"] * support
            + self.weights["predictive_strength"] * predictive_strength
            + self.weights["determinism"] * determinism
            + self.weights["stability"] * stability
            + self.weights["doc_alignment"] * doc_alignment
        )

        return {
            "A": a,
            "B": b,
            "feature_type": "categorical",
            "support": support,
            "predictive_strength": predictive_strength,
            "determinism": determinism,
            "stability": stability,
            "doc_alignment": doc_alignment,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "classification": self.classify_strength(strength),
            "heldout_accuracy": heldout_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "row_count": row_count,
            "total_rows": total_rows,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "evidence": self._build_evidence(pair, a, b),
            "a_to_b_mapping": json.dumps(self._build_mapping(pair, a, b)),
        }

    def _build_evidence(self, pair: pd.DataFrame, a: str, b: str, limit: int = 5) -> list[dict[str, Any]]:
        if pair.empty:
            return []

        evidence: list[dict[str, Any]] = []
        for a_val, group in pair.groupby(a):
            counts = group[b].value_counts(normalize=True)
            if counts.empty:
                continue
            evidence.append(
                {
                    "A_value": a_val,
                    "count": int(len(group)),
                    "top_B_values": [
                        {"value": idx, "probability": float(prob)}
                        for idx, prob in counts.head(3).items()
                    ],
                }
            )

        evidence.sort(key=lambda item: item["count"], reverse=True)
        return evidence[:limit]

    def _build_mapping(self, pair: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for a_val, subset in pair.groupby(a):
            counts = subset[b].value_counts(normalize=True)
            if counts.empty:
                continue
            top_val = counts.index[0]
            top_prob = float(counts.iloc[0])
            mapping[str(a_val)] = str(top_val) if top_prob >= 0.95 else [str(v) for v in counts.index]
        return mapping