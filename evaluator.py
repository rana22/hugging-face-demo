from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from docs import DocAlignmentModel
from features import PairwiseFeatureEngine
from schema import NodeSchema

@dataclass(frozen=True)
class RelationshipResult:
    A: str
    B: str
    strength: float
    classification: str
    support: float
    predictive_strength: float
    determinism: float
    stability: float
    doc_alignment: float
    suffix_match: float | None = None
    substring_match: float | None = None
    heldout_accuracy: float
    baseline_accuracy: float
    row_count: int
    total_rows: int
    train_rows: int
    test_rows: int
    evidence: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RelationshipResult":
        return cls(**row)


class PairwiseRelationshipEvaluator(PairwiseFeatureEngine):
    """Backward-compatible alias for the MVP feature engine."""

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.evaluate_all_pairs(df)


def build_evaluator(node_schema: NodeSchema) -> PairwiseRelationshipEvaluator:
    return PairwiseRelationshipEvaluator(node_schema=node_schema, doc_model=DocAlignmentModel().fit(node_schema))
