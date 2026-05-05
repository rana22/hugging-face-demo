from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import re
import requests
import os

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

NLP_URL = os.getenv("NLP_URL", "http://localhost:8000")

def call_bio_dataset_service(node_schema: NodeSchema, df: pd.DataFrame) -> list[dict]:
    try:
        if df.empty:
            return []

        df_sample = df.sample(min(len(df), 100), random_state=42)

        # 🔥 Extract schema-level text
        property_meta = {}
        for prop, spec in node_schema.properties.items():
            property_meta[prop] = {
                "description": spec.get("Desc", ""),
                "type": spec.get("Type", ""),
                "tags": spec.get("Tags", {}),
            }

        payload = {
            "node": node_schema.name,
            "node_description": node_schema.description or "",
            "properties": list(df_sample.columns),
            "property_metadata": property_meta,   # ✅ NEW
            "data": df_sample.astype(str).to_dict("records"),
        }

        resp = requests.post(
            f"{NLP_URL}/bio-analyze-dataset",
            json=payload,
            timeout=10,
        )

        if resp.status_code != 200:
            return []

        return resp.json().get("property_relations", [])

    except Exception as e:
        print(f"[BIO SERVICE ERROR] {e}")
        return []

@dataclass
class BioTermFeatureAnalyzer:
    node_schema: NodeSchema
    doc_model: DocAlignmentModel
    weights: dict[str, float] = field(init=False)
    _bio_cache: Dict[tuple, float] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", get_bioterm_weights(self.node_schema.name))

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

    # -----------------------------
    # Load dataset-level bio scores
    # -----------------------------
    def _ensure_cache(self, df: pd.DataFrame):
        if self.weights is None:
            return

        if self._bio_cache is not None:
            return

        results = call_bio_dataset_service(self.node_schema.name, df)

        cache: Dict[tuple, float] = {}
        for item in results:
            a = item.get("A")
            b = item.get("B")
            score = float(item.get("bio_overlap", 0.0))

            if a and b:
                cache[(a, b)] = score
                cache[(b, a)] = score  # symmetry

        self._bio_cache = cache

    # -----------------------------
    # Main analyzer
    # -----------------------------
    def analyze(self, df: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
        if self.weights is None:
            return None

        pair = prepare_pair_frame(df, a, b)
        total_rows = len(df)
        row_count = len(pair)

        support = float(row_count / total_rows) if total_rows else 0.0

        # ensure cache once
        self._ensure_cache(df)

        bio_score = 0.0
        if self._bio_cache:
            bio_score = self._bio_cache.get((a, b), 0.0)

        doc_alignment = float(self.doc_model.score(a, b))

        strength = (
            self.weights["support"] * support
            + self.weights["bio_term_overlap"] * bio_score
            + self.weights["doc_alignment"] * doc_alignment
        )

        return {
            "A": a,
            "B": b,
            "feature_type": "bio_term",
            "support": support,
            "bio_term_overlap": bio_score,
            "doc_alignment": doc_alignment,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "classification": self.classify_strength(strength),
            "row_count": row_count,
            "total_rows": total_rows,
            "evidence": self._build_evidence(pair, a, b),
        }

    # -----------------------------
    # Evidence (same style as substring)
    # -----------------------------
    def _build_evidence(
        self,
        pair: pd.DataFrame,
        a: str,
        b: str,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        if pair.empty:
            return []

        evidence = []
        for _, row in pair.head(limit).iterrows():
            evidence.append({
                "A_value": str(row[a]),
                "B_value": str(row[b]),
                "bio_score": self._bio_cache.get((a, b), 0.0) if self._bio_cache else 0.0
            })

        return evidence

