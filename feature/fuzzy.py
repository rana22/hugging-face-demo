from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import json, re

from .categorical import normalize_value, prepare_pair_frame
from .doc_alignment import DocAlignmentModel
from schema import NodeSchema

FUZZY_WEIGHT_PROFILES = {
    "cross_node_match": {
        "support": 0.15,
        "ratio": 0.10,
        "partial_ratio": 0.35,
        "token_sort_ratio": 0.20,
        "token_set_ratio": 0.15,
        "doc_alignment": 0.05,
    },
}

DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?[Zz]?",
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[Zz]",
    r"^\d{4}-\d{2}-\d{2}$",                           # 2021-08-13
    r"^\d{4}-\d{2}-\d{2}T",                           # 2021-08-13T...
    r"^\d{2}/\d{2}/\d{4}$",                           # 08/13/2021
    r"^\d{4}/\d{2}/\d{2}$",                           # 2021/08/13
]

ID_PATTERNS = [
    r"^[A-Za-z]{2,}\d{2,}",                           # OSA01, GLIOMA01
    r"^\d{4,}$",                                      # 000001, 12345
    r"^[A-Za-z0-9]+([-_][A-Za-z0-9]+)+$",             # GLIOMA01-i_2C4F-T1-A1-J05
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
]

def looks_like_date_column(series: pd.Series, sample_size: int = 50, threshold: float = 0.8) -> bool:
    values = (
        series.dropna()
        .astype(str)
        .map(str.strip)
    )
    if values.empty:
        return False

    sample = values.drop_duplicates().head(sample_size)
    if sample.empty:
        return False

    hits = sum(1 for v in sample if looks_like_date(v))
    return (hits / len(sample)) >= threshold

def looks_like_date(value: str) -> bool:
    s = str(value).strip().lower()
    return any(re.match(p, s) for p in DATE_PATTERNS)


def looks_like_id(value: str) -> bool:
    s = str(value).strip()
    return any(re.match(p, s, flags=re.IGNORECASE) for p in ID_PATTERNS)


def should_skip_fuzzy(av: str, bv: str) -> bool:
    av = str(av).strip()
    bv = str(bv).strip()

    if not av or not bv:
        return True

    # skip date vs id comparisons
    if looks_like_date(av) or looks_like_date(bv):
        return True

    return False


def get_fuzzy_weights(node_name: str) -> dict[str, float] | None:
    key = (node_name or "").lower()
    return FUZZY_WEIGHT_PROFILES.get(key)


def fuzzy_metrics(av: str, bv: str) -> dict[str, float]:
    av = normalize_value(av).lower()
    bv = normalize_value(bv).lower()

    if not av or not bv:
        return {
            "ratio": 0.0,
            "partial_ratio": 0.0,
            "token_sort_ratio": 0.0,
            "token_set_ratio": 0.0,
        }

    return {
        "ratio": fuzz.ratio(av, bv) / 100.0,
        "partial_ratio": fuzz.partial_ratio(av, bv) / 100.0,
        "token_sort_ratio": fuzz.token_sort_ratio(av, bv) / 100.0,
        "token_set_ratio": fuzz.token_set_ratio(av, bv) / 100.0,
    }


def fuzzy_best_match_type(scores: dict[str, float]) -> str:
    if not scores:
        return "no_match"
    return max(scores, key=scores.get)

def fuzzy_row_score(av: str, bv: str) -> dict[str, Any]:
    scores = fuzzy_metrics(av, bv)
    best_key = fuzzy_best_match_type(scores)
    best_score = scores.get(best_key, 0.0)
    scores["best_match_type"] = best_key
    scores["best_score"] = best_score
    return scores


def mean_score(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


@dataclass
class FuzzyFeatureAnalyzer:
    node_schema: NodeSchema | None
    doc_model: DocAlignmentModel | None
    weights: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        if self.node_schema is not None:
            key = self.node_schema.name
        else:
            key = "cross_node_match"

        weights = get_fuzzy_weights(key)
        if weights is None:
            weights = FUZZY_WEIGHT_PROFILES["cross_node_match"]

        object.__setattr__(self, "weights", weights)

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

    def analyze(self, df: pd.DataFrame, a: str, b: str) -> dict[str, Any] | None:
        if self.weights is None:
            return None

        pair = prepare_pair_frame(df, a, b)
        total_rows = len(df)
        row_count = len(pair)

        if total_rows == 0:
            return None

        support = float(row_count / total_rows)

        ratio_scores: list[float] = []
        partial_scores: list[float] = []
        token_sort_scores: list[float] = []
        token_set_scores: list[float] = []
        evidence: list[dict[str, Any]] = []
        mapping: dict[str, Any] = {}

        for _, row in pair.iterrows():
            av = str(row.get(a, ""))
            bv = str(row.get(b, ""))

            scores = fuzzy_row_score(av, bv)

            ratio_scores.append(float(scores["ratio"]))
            partial_scores.append(float(scores["partial_ratio"]))
            token_sort_scores.append(float(scores["token_sort_ratio"]))
            token_set_scores.append(float(scores["token_set_ratio"]))

            if len(evidence) < 20:
                evidence.append(
                    {
                        "A_value": av,
                        "B_value": bv,
                        "ratio": float(scores["ratio"]),
                        "partial_ratio": float(scores["partial_ratio"]),
                        "token_sort_ratio": float(scores["token_sort_ratio"]),
                        "token_set_ratio": float(scores["token_set_ratio"]),
                        "best_match_type": scores["best_match_type"],
                        "best_score": float(scores["best_score"]),
                    }
                )

        ratio = mean_score(ratio_scores)
        partial_ratio = mean_score(partial_scores)
        token_sort_ratio = mean_score(token_sort_scores)
        token_set_ratio = mean_score(token_set_scores)

        doc_alignment = float(self.doc_model.score(a, b)) if self.doc_model is not None else 0.0

        strength = (
            self.weights["support"] * support
            + self.weights["ratio"] * ratio
            + self.weights["partial_ratio"] * partial_ratio
            + self.weights["token_sort_ratio"] * token_sort_ratio
            + self.weights["token_set_ratio"] * token_set_ratio
            + self.weights["doc_alignment"] * doc_alignment
        )

        # build simple mapping from A -> best B values seen in pair rows
        for a_val, subset in pair.groupby(a):
            counts = subset[b].value_counts(normalize=True)
            if counts.empty:
                continue
            top_val = counts.index[0]
            top_prob = float(counts.iloc[0])
            mapping[str(a_val)] = str(top_val) if top_prob >= 0.95 else [str(v) for v in counts.index]

        return {
            "A": a,
            "B": b,
            "classification": self.classify_strength(strength),
            "a_to_b_mapping": json.dumps(mapping, default=str, indent=2),
            "feature_type": "fuzzy",
            "support": support,
            "ratio": ratio,
            "partial_ratio": partial_ratio,
            "token_sort_ratio": token_sort_ratio,
            "token_set_ratio": token_set_ratio,
            "doc_alignment": doc_alignment,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "row_count": row_count,
            "total_rows": total_rows,
            "evidence": evidence,
        }