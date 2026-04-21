from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from feature.doc_alignment import DocAlignmentModel
from feature.base import FeatureBase
from schema import NodeSchema
from feature.categorical import CategoricalFeatureAnalyzer
from feature.substring import SubstringFeatureAnalyzer
from feature.cluster import ClusteringFeatureAnalyzer

@dataclass(frozen=True)
class RelationshipResult:
    A: str
    B: str
    feature_type: str
    strength: float
    classification: str
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
    suffix_match: float = 0.0
    substring_match: float = 0.0
    evidence: list[dict[str, Any]] = None
    a_to_b_mapping: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RelationshipResult":
        return cls(**row)


class PairwiseRelationshipEvaluator(FeatureBase):
    def __init__(self, node_schema: NodeSchema):
        self.node_schema = node_schema
        self.doc_model = DocAlignmentModel().fit(node_schema)
        self.categorical = CategoricalFeatureAnalyzer(
            node_schema=self.node_schema,
            doc_model=self.doc_model,
        )
        self.substring = SubstringFeatureAnalyzer(
            node_schema=self.node_schema,
            doc_model=self.doc_model,
        )
        self.cluster = ClusteringFeatureAnalyzer(
            node_schema=self.node_schema,
            doc_model=self.doc_model,
        )
        # self.semantic = SemanticFeatureAnalyzer(
        #     node_schema=self.node_schema,
        # )

    def get_model_columns(self, df: pd.DataFrame) -> list[str]:
        schema_columns = list(getattr(self.node_schema, "properties", {}).keys())
        return [
            c for c in schema_columns if c in df.columns and not self.should_skip(c, self.node_schema)
        ]

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.evaluate_all_pairs(df)

    def evaluate_all_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        results: list[dict[str, Any]] = []
        columns = self.get_model_columns(df)

        for a in columns:
            for b in columns:
                if a == b:
                    continue
                cat_result = self.categorical.analyze(df, a, b)
                if cat_result is not None:
                    results.append(cat_result)
                string_result = self.substring.analyze(df, a, b)
                if string_result is not None:
                    results.append(string_result)
                cluster_result = self.cluster.analyze(df, a, b)
                if cluster_result is not None:
                    results.append(cluster_result)
                

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(
                ["strength", "predictive_strength", "support"],
                ascending=False,
            ).reset_index(drop=True)

        return results_df

    def evaluate_all_pairs2(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        results_by_feature: dict[str, list[dict[str, Any]]] = {}
        columns = self.get_model_columns(df)

        for a in columns:
            feature_results: list[dict[str, Any]] = []

            for b in columns:
                if a == b:
                    continue

                cat_result = self.categorical.analyze(df, a, b)
                cat_result["analysis_type"] = "categorical"
                cat_result["feature"] = a
                cat_result["compared_with"] = b
                feature_results.append(cat_result)

                sub_result = self.substring.analyze(df, a, b)
                sub_result["analysis_type"] = "substring"
                sub_result["feature"] = a
                sub_result["compared_with"] = b
                feature_results.append(sub_result)

            results_by_feature[a] = feature_results

        results_dfs: dict[str, pd.DataFrame] = {}

        for feature, results in results_by_feature.items():
            result_df = pd.DataFrame(results)

            if not result_df.empty:
                sort_cols = [c for c in ["analysis_type", "strength", "predictive_strength", "support"] if c in result_df.columns]
                if sort_cols:
                    asc = [True] + [False] * (len(sort_cols) - 1)
                    result_df = result_df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

            results_dfs[feature] = result_df

        return results_dfs


def build_evaluator(node_schema: NodeSchema) -> PairwiseRelationshipEvaluator:
    return PairwiseRelationshipEvaluator(node_schema=node_schema)