from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import re

from .categorical import normalize_value, prepare_pair_frame
from .doc_alignment import DocAlignmentModel
from schema import NodeSchema
from sentence_transformers import SentenceTransformer

# -----------------------------
# BIO TERM WEIGHT PROFILES
# -----------------------------
BIOTERM_WEIGHT_PROFILES = {
    "human_relevance": {
        "support": 0.10,
        "bio_term_overlap": 0.80,
        "doc_alignment": 0.10,
    },
    "study": {
        "support": 0.10,
        "bio_term_overlap": 0.80,
        "doc_alignment": 0.10,
    },
}


def get_bioterm_weights(node_name: str) -> dict[str, float] | None:
    key = (node_name or "").lower()
    return BIOTERM_WEIGHT_PROFILES.get(key)

_model = SentenceTransformer("all-MiniLM-L6-v2")
# -----------------------------
# BIO TERM EXTRACTION
# -----------------------------
BIO_TERM_PATTERN = re.compile(r"[A-Za-z0-9\-]+")


def extract_bio_terms(text: str) -> Set[str]:
    if not text:
        return set()

    tokens = BIO_TERM_PATTERN.findall(text.lower())

    # filter noise
    return {t for t in tokens if len(t) > 2}


# -----------------------------
# ROW-LEVEL OVERLAP
# -----------------------------
# def bio_term_overlap_row(av: str, bv: str) -> float:
#     terms_a = extract_bio_terms(av)
#     terms_b = extract_bio_terms(bv)

#     if not terms_a or not terms_b:
#         return 0.0

#     intersection = terms_a & terms_b
#     union = terms_a | terms_b

#     return len(intersection) / len(union)

def bio_term_embedding(text: str) -> np.ndarray:
    terms = extract_bio_terms(text)
    if not terms:
        return np.zeros(384)  # model dim

    # join terms into a compact representation
    term_string = " ".join(sorted(terms))
    return _model.encode(term_string)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -----------------------------
# DATAFRAME-LEVEL SCORE
# -----------------------------
# def bio_term_overlap_score(df: pd.DataFrame, a: str, b: str) -> float:
#     total = 0
#     score = 0.0

#     for _, row in df.iterrows():
#         av = normalize_value(row.get(a, ""))
#         bv = normalize_value(row.get(b, ""))

#         if not av or not bv:
#             continue

#         total += 1
#         score += bio_term_overlap_row(av, bv)

#     return score / total if total else 0.0

def bio_term_vector_score(df: pd.DataFrame, a: str, b: str) -> float:
    total = 0
    score = 0.0

    for _, row in df.iterrows():
        av = normalize_value(row.get(a, ""))
        bv = normalize_value(row.get(b, ""))

        if not av or not bv:
            continue

        emb_a = bio_term_embedding(av)
        emb_b = bio_term_embedding(bv)

        total += 1
        score += cosine_sim(emb_a, emb_b)

    return score / total if total else 0.0


@dataclass
class BioTermFeatureAnalyzer:
    node_schema: NodeSchema
    doc_model: DocAlignmentModel
    weights: dict[str, float] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "weights",
            get_bioterm_weights(self.node_schema.name)
        )
    
    def _build_evidence(self, pair: pd.DataFrame, a: str, b: str, limit: int = 5) -> list[dict[str, Any]]:
        if pair.empty:
            return []

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
        # ✅ guard (same as categorical)
        if self.weights is None:
            return None

        pair = prepare_pair_frame(df, a, b)

        total_rows = len(df)
        row_count = len(pair)
        support = float(row_count / total_rows) if total_rows else 0.0

        # ✅ compute vector similarity
        vector_score = bio_term_vector_score(pair, a, b)

        # ✅ doc alignment (restore it)
        doc_alignment = float(self.doc_model.score(a, b))

        # ✅ FIXED weight keys
        strength = (
            self.weights["support"] * support
            + self.weights["bio_term_overlap"] * vector_score
            + self.weights["doc_alignment"] * doc_alignment
        )

        return {
            "A": a,
            "B": b,
            "feature_type": "bio_term",
            "support": support,
            "bio_term_overlap": vector_score,
            "doc_alignment": doc_alignment,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "classification": self.classify_strength(strength),
            "row_count": row_count,
            "total_rows": total_rows,
            "evidence": self._build_evidence(pair, a, b),
        }