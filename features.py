from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from dotenv import load_dotenv

from docs import DocAlignmentModel
from schema import NodeSchema


load_dotenv()

MISSING_STRINGS = {"", "na", "n/a", "none", "null", "nan", "not reported"}
IDENTIFIER_HINTS = ("_id", "_record_id", "uuid", "crdc_id")


@dataclass(frozen=True)
class PairwiseFeatures:
    support: float
    predictive_strength: float
    determinism: float
    stability: float
    doc_alignment: float
    heldout_accuracy: float
    baseline_accuracy: float
    row_count: int
    total_rows: int
    train_rows: int
    test_rows: int


def build_conditional_map(df: pd.DataFrame, a: str, b: str, deterministic_threshold: float = 0.95) -> dict[str, Any]:
    mapping: dict[str, Any] = {}

    for a_val, subset in df.groupby(a):
        counts = subset[b].value_counts(normalize=True)
        if counts.empty:
            continue

        top_val = counts.index[0]
        top_prob = float(counts.iloc[0])

        if top_prob >= deterministic_threshold:
            mapping[str(a_val)] = str(top_val)
        else:
            mapping[str(a_val)] = [str(v) for v in counts.index]

    return mapping


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() in MISSING_STRINGS:
        return True
    return False


def normalize_value(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def is_identifier_like(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(hint) or lowered == hint.strip("_") for hint in IDENTIFIER_HINTS)


def prepare_pair_frame(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    pair = df[[a, b]].copy()
    pair[a] = pair[a].map(normalize_value)
    pair[b] = pair[b].map(normalize_value)
    pair = pair[(pair[a] != "") & (pair[b] != "")]
    return pair.reset_index(drop=True)


def conditional_determinism(pair: pd.DataFrame, a: str, b: str) -> float:
    if pair.empty:
        return 0.0

    total = len(pair)
    weighted_max_probs = []

    for _, group in pair.groupby(a):
        counts = group[b].value_counts(normalize=True)
        if counts.empty:
            continue
        weight = len(group) / total
        weighted_max_probs.append(weight * float(counts.max()))

    return float(np.clip(sum(weighted_max_probs), 0.0, 1.0))


def predictive_strength_from_holdout(
    pair: pd.DataFrame,
    a: str,
    b: str,
    *,
    seed: int = 42,
) -> tuple[float, float, float, int, int]:
    if pair.empty or pair[a].nunique() < 2 or pair[b].nunique() < 2:
        return 0.0, 0.0, 0.0, 0, 0

    X = pair[[a]].astype(str)
    y = pair[b].astype(str)

    if len(pair) < 8:
        baseline = float(y.value_counts(normalize=True).max())
        return baseline, baseline, 0.0, len(pair), 0

    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            shuffle=True,
        )

    if y_train.nunique() < 2 or y_test.empty:
        baseline = float(y_test.value_counts(normalize=True).max()) if len(y_test) else float(y.value_counts(normalize=True).max())
        return baseline, baseline, 0.0, len(X_train), len(X_test)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), [a]),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, pred))

    majority = y_train.value_counts().idxmax()
    baseline_pred = np.full(len(y_test), majority)
    baseline = float(accuracy_score(y_test, baseline_pred))

    if baseline >= 1.0:
        predictive_strength = 0.0
    else:
        predictive_strength = max(0.0, (accuracy - baseline) / max(1e-9, 1.0 - baseline))

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


def suffix_match_score(df: pd.DataFrame, a: str, b: str) -> float:
    total = 0
    match = 0

    for _, row in df.iterrows():
        av = normalize_value(row.get(a, "")).lower()
        bv = normalize_value(row.get(b, "")).lower()

        if not av or not bv:
            continue

        total += 1
        if bv.endswith(av):
            match += 1

    return match / total if total else 0.0


def substring_match_score(df: pd.DataFrame, a: str, b: str) -> float:
    total = 0
    match = 0

    for _, row in df.iterrows():
        av = normalize_value(row.get(a, "")).lower()
        bv = normalize_value(row.get(b, "")).lower()

        if not av or not bv:
            continue

        total += 1
        if av in bv:
            match += 1

    return match / total if total else 0.0


NODE_WEIGHT_PROFILES = {
    "default": {
        "support": 0.25,
        "predictive_strength": 0.30,
        "determinism": 0.20,
        "stability": 0.15,
        "doc_alignment": 0.10,
        "suffix_match": 0.0,
        "substring_match": 0.00,
    },
    "sample": {
        "support": 0.25,
        "predictive_strength": 0.30,
        "determinism": 0.20,
        "stability": 0.15,
        "doc_alignment": 0.10,
        "suffix_match": 0.0,
        "substring_match": 0.00,
    },
    "file": {
        "support": 0.30,
        "predictive_strength": 0.20,
        "determinism": 0.30,
        "stability": 0.10,
        "doc_alignment": 0.10,
        "suffix_match": 0.35,
        "substring_match": 0.35,
    },
    "study": {
        "support": 0.20,
        "predictive_strength": 0.20,
        "determinism": 0.15,
        "stability": 0.20,
        "doc_alignment": 0.25,
        "suffix_match": 0.0,
        "substring_match": 0.00,
    },
}


def get_node_weights(node_name: str) -> dict[str, float]:
    if not node_name:
        return NODE_WEIGHT_PROFILES["default"]
    return NODE_WEIGHT_PROFILES.get(node_name.lower(), NODE_WEIGHT_PROFILES["default"])


@dataclass
class PairwiseFeatureEngine:
    node_schema: NodeSchema
    doc_model: DocAlignmentModel
    weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            node_name = getattr(self.node_schema, "name", "") or ""
            object.__setattr__(self, "weights", get_node_weights(node_name))

    def get_skip_fields(self) -> set[str]:
        raw = os.getenv("SKIP_FIELDS", "")
        return {x.strip().lower() for x in raw.split(",") if x.strip()}

    def should_skip(self, col: str, schema: NodeSchema | None = None) -> bool:
        col_norm = col.strip().lower()

        if col_norm in self.get_skip_fields():
            return True

        if is_identifier_like(col_norm):
            return True

        if schema is not None:
            exclude_like = {str(x).strip().lower() for x in getattr(schema, "exclude_like", [])}
            if col_norm in exclude_like:
                return True

            props = getattr(schema, "properties", {})
            if col not in props:
                return True

        return False

    def get_model_columns(self, schema: NodeSchema, df: pd.DataFrame) -> list[str]:
        schema_columns = list(schema.properties.keys())
        return [c for c in schema_columns if c in df.columns and not self.should_skip(c, schema)]

    def evaluate_pair(self, df: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
        pair = prepare_pair_frame(df, a, b)
        total_rows = len(df)
        row_count = len(pair)
        support = float(row_count / total_rows) if total_rows else 0.0

        predictive_strength, heldout_accuracy, baseline_accuracy, train_rows, test_rows = predictive_strength_from_holdout(pair, a, b)
        determinism = conditional_determinism(pair, a, b)
        stability = stability_from_resamples(pair, a, b)
        doc_alignment = float(self.doc_model.score(a, b))
        suffix_match = suffix_match_score(pair, a, b)
        substring_match = substring_match_score(pair, a, b)

        strength = (
            self.weights["support"] * support
            + self.weights["predictive_strength"] * predictive_strength
            + self.weights["determinism"] * determinism
            + self.weights["stability"] * stability
            + self.weights["doc_alignment"] * doc_alignment
            + self.weights["suffix_match"] * suffix_match
            + self.weights["substring_match"] * substring_match
        )

        evidence = self._build_evidence(pair, a, b)
        classification = self.classify_strength(strength)
        a_to_b_map = build_conditional_map(pair, a, b)

        return {
        "A": a,
        "B": b,
        "support": support,
        "predictive_strength": predictive_strength,
        "determinism": determinism,
        "stability": stability,
        "doc_alignment": doc_alignment,
        "suffix_match": suffix_match,
        "substring_match": substring_match,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "classification": classification,
        "heldout_accuracy": heldout_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "row_count": row_count,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "evidence": evidence,
        "a_to_b_mapping": json.dumps(a_to_b_map),
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

    def evaluate_all_pairs(self, schema: NodeSchema, df: pd.DataFrame) -> pd.DataFrame:
        results: list[dict[str, Any]] = []
        columns = self.get_model_columns(schema, df)

        for a in columns:
            for b in columns:
                if a == b:
                    continue
                results.append(self.evaluate_pair(df, a, b))

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(
                ["strength", "predictive_strength", "support"],
                ascending=False,
            ).reset_index(drop=True)

        return results_df
