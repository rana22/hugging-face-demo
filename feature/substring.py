from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .categorical import normalize_value, prepare_pair_frame
from .doc_alignment import DocAlignmentModel
from schema import NodeSchema

SUBSTRING_WEIGHT_PROFILES = {
    "human_relevance": {
        "support": 0.10,
        "prefix_match": 0.02,
        "suffix_match": 0.02,
        "substring_match": 0.05,
        "doc_alignment": 0.05,
    },
    "sample": {
        "support": 0.10,
        "prefix_match": 0.0,
        "suffix_match": 0.0,
        "substring_match": 0.0,
        "doc_alignment": 0.05,
    },
    "file": {
        "support": 0.15,
        "prefix_match": 0.25,
        "suffix_match": 0.30,
        "substring_match": 0.25,
        "doc_alignment": 0.05,
    },
    "study": {
        "support": 0.10,
        "prefix_match": 0.30,
        "suffix_match": 0.30,
        "substring_match": 0.25,
        "doc_alignment": 0.05,
    },
}

def smart_contains(av: str, bv: str) -> bool:
    tokens = re.split(r'[^a-z0-9]+', av.lower())
    bv = bv.lower()
    return any(t and t in bv for t in tokens)

def prefix_match_score(df: pd.DataFrame, a: str, b: str) -> float:
    total = 0
    match = 0
    for _, row in df.iterrows():
        av = normalize_value(row.get(a, "")).lower()
        bv = normalize_value(row.get(b, "")).lower()
        if not av or not bv:
            continue
        total += 1
        if bv.startswith(av):
            match += 1
    return match / total if total else 0.0


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
        if smart_contains(av, bv):
            match += 1
    return match / total if total else 0.0

def get_substring_weights(node_name: str) -> dict[str, float] | None:
    key = (node_name or "").lower()
    return SUBSTRING_WEIGHT_PROFILES.get(key)

@dataclass
class SubstringFeatureAnalyzer:
    node_schema: NodeSchema
    doc_model: DocAlignmentModel
    weights: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        if self.node_schema is not None:
            key = self.node_schema.name
        else:
            key = "cross_node_match"
        object.__setattr__(self, "weights", get_substring_weights(key))

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

    def analyze(self, df: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
        if self.weights is None:
            return None
        pair = prepare_pair_frame(df, a, b)
        total_rows = len(df)
        row_count = len(pair)
        support = float(row_count / total_rows) if total_rows else 0.0

        prefix = prefix_match_score(pair, a, b)
        suffix = suffix_match_score(pair, a, b)
        substring = substring_match_score(pair, a, b)
        doc_alignment = float(self.doc_model.score(a, b)) or 0.0

        strength = (
            self.weights["support"] * support
            + self.weights["prefix_match"] * prefix
            + self.weights["suffix_match"] * suffix
            + self.weights["substring_match"] * substring
            + self.weights["doc_alignment"] * doc_alignment
        )

        return {
            "A": a,
            "B": b,
            "feature_type": "substring",
            "support": support,
            "prefix_match": prefix,
            "suffix_match": suffix,
            "substring_match": substring,
            "doc_alignment": doc_alignment,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "classification": self.classify_strength(strength),
            "row_count": row_count,
            "total_rows": total_rows,
            "evidence": self._build_evidence(pair, a, b),
        }

    def _build_evidence(self, pair: pd.DataFrame, a: str, b: str, limit: int = 20) -> list[dict[str, Any]]:
        if pair.empty:
            return []

        evidence: list[dict[str, Any]] = []
        for _, row in pair.head(limit).iterrows():
            av = str(row[a])
            bv = str(row[b])
            relation = "contains" if av.lower() in bv.lower() else "no_match"
            if bv.lower().startswith(av.lower()):
                relation = "prefix"
            elif bv.lower().endswith(av.lower()):
                relation = "suffix"

            evidence.append({"A_value": av, "B_value": bv, "match_type": relation})

        return evidence


    
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