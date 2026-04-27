from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .categorical import normalize_value, prepare_pair_frame
from .doc_alignment import DocAlignmentModel
from schema import NodeSchema

CLUSTERING_WEIGHT_PROFILES = {
    "file": {
        "support": 0.20,
        "separation": 0.60,
        "doc_alignment": 0.20,
    },
}

def get_clustering_weights(node_name: str) -> dict[str, float]:
    key = (node_name or "").lower()
    return CLUSTERING_WEIGHT_PROFILES.get(key)

@dataclass
class ClusteringFeatureAnalyzer:
    node_schema: NodeSchema
    doc_model: DocAlignmentModel
    weights: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", get_clustering_weights(self.node_schema.name))

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

    def _is_applicable(self, df: pd.DataFrame, a: str, b: str) -> bool:
        if a not in df.columns or b not in df.columns:
            return False

        a_series = df[a].map(normalize_value)
        b_numeric = pd.to_numeric(df[b], errors="coerce")

        if a_series.replace("", np.nan).dropna().nunique() < 2:
            return False

        numeric_coverage = float(b_numeric.notna().mean()) if len(b_numeric) else 0.0
        if numeric_coverage < 0.75:
            return False

        positive_coverage = float((b_numeric.dropna() > 0).mean()) if b_numeric.notna().any() else 0.0
        if positive_coverage < 0.75:
            return False

        return True

    @staticmethod
    def _safe_log_values(values: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        numeric = numeric[numeric > 0]
        if numeric.empty:
            return np.array([], dtype=float)
        return np.log(numeric.to_numpy(dtype=float))

    @staticmethod
    def _kmeans_1d(values: np.ndarray, k: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray, float]:
        if len(values) == 0:
            return np.array([], dtype=int), np.array([], dtype=float), 0.0

        if k <= 1 or len(values) <= 1:
            centroid = np.array([float(values.mean())], dtype=float)
            inertia = float(((values - centroid[0]) ** 2).sum())
            labels = np.zeros(len(values), dtype=int)
            return labels, centroid, inertia

        k = min(k, len(values))
        centroids = np.quantile(values, np.linspace(0, 1, k + 2)[1:-1]).astype(float)

        if np.unique(centroids).size < k:
            centroids = np.linspace(values.min(), values.max(), k, dtype=float)

        labels = np.zeros(len(values), dtype=int)

        for _ in range(max_iter):
            distances = np.abs(values[:, None] - centroids[None, :])
            new_labels = distances.argmin(axis=1)

            new_centroids = centroids.copy()
            for idx in range(k):
                mask = new_labels == idx
                if mask.any():
                    new_centroids[idx] = float(values[mask].mean())

            if np.allclose(new_centroids, centroids):
                labels = new_labels
                centroids = new_centroids
                break

            labels = new_labels
            centroids = new_centroids

        inertia = 0.0
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                inertia += float(((values[mask] - centroids[idx]) ** 2).sum())

        return labels, centroids, inertia

    def _cluster_series(self, values: pd.Series, max_clusters: int = 3) -> dict[str, Any]:
        log_values = self._safe_log_values(values)
        if len(log_values) == 0:
            return {
                "cluster_count": 0,
                "separation": 0.0,
                "clusters": [],
                "dominant_center": None,
                "dominant_spread": None,
            }

        if len(log_values) < 4:
            mean = float(log_values.mean())
            std = float(log_values.std()) if len(log_values) > 1 else 0.0
            return {
                "cluster_count": 1,
                "separation": 0.0,
                "clusters": [
                    {
                        "center_log": mean,
                        "spread_log": std,
                        "count": int(len(log_values)),
                        "share": 1.0,
                    }
                ],
                "dominant_center": mean,
                "dominant_spread": std,
            }

        total_sse = float(((log_values - log_values.mean()) ** 2).sum())
        if total_sse <= 1e-12:
            mean = float(log_values.mean())
            return {
                "cluster_count": 1,
                "separation": 0.0,
                "clusters": [
                    {
                        "center_log": mean,
                        "spread_log": 0.0,
                        "count": int(len(log_values)),
                        "share": 1.0,
                    }
                ],
                "dominant_center": mean,
                "dominant_spread": 0.0,
            }

        best = None
        best_score = -np.inf
        best_k = 1

        for k in range(1, min(max_clusters, len(log_values)) + 1):
            labels, centroids, inertia = self._kmeans_1d(log_values, k)
            fit = max(0.0, 1.0 - (inertia / total_sse))
            penalty = 0.04 * (k - 1)
            score = fit - penalty

            if score > best_score:
                best_score = score
                best = (labels, centroids, inertia, fit)
                best_k = k

        if best is None:
            mean = float(log_values.mean())
            std = float(log_values.std()) if len(log_values) > 1 else 0.0
            return {
                "cluster_count": 1,
                "separation": 0.0,
                "clusters": [
                    {
                        "center_log": mean,
                        "spread_log": std,
                        "count": int(len(log_values)),
                        "share": 1.0,
                    }
                ],
                "dominant_center": mean,
                "dominant_spread": std,
            }

        labels, centroids, inertia, fit = best
        clusters: list[dict[str, Any]] = []

        for idx in range(best_k):
            mask = labels == idx
            if not mask.any():
                continue

            cluster_vals = log_values[mask]
            clusters.append(
                {
                    "center_log": float(centroids[idx]),
                    "spread_log": float(cluster_vals.std()) if len(cluster_vals) > 1 else 0.0,
                    "count": int(mask.sum()),
                    "share": float(mask.mean()),
                    "min_value": float(np.exp(cluster_vals.min())),
                    "max_value": float(np.exp(cluster_vals.max())),
                }
            )

        clusters.sort(key=lambda x: x["count"], reverse=True)

        dominant_center = clusters[0]["center_log"] if clusters else float(log_values.mean())
        dominant_spread = clusters[0]["spread_log"] if clusters else float(log_values.std())

        # Separation grows when the best fit is good and the cluster centers are distinct.
        if len(clusters) > 1:
            centers = np.array([c["center_log"] for c in clusters], dtype=float)
            separation = float(np.tanh(np.std(centers) + fit))
        else:
            separation = float(max(0.0, fit * 0.5))

        return {
            "cluster_count": int(len(clusters)),
            "separation": separation,
            "clusters": clusters,
            "dominant_center": float(dominant_center),
            "dominant_spread": float(dominant_spread),
        }

    def _build_evidence(self, pair: pd.DataFrame, a: str, b: str, limit: int = 5) -> list[dict[str, Any]]:
        if pair.empty:
            return []

        work = pair[[a, b]].copy()
        work[a] = work[a].map(normalize_value)
        work[b] = pd.to_numeric(work[b], errors="coerce")
        work = work[(work[a] != "") & work[b].notna() & (work[b] > 0)]

        evidence: list[dict[str, Any]] = []

        for category, group in work.groupby(a):
            cluster_summary = self._cluster_series(group[b])

            evidence.append(
                {
                    "category_value": str(category),
                    "count": int(len(group)),
                    "cluster_count": cluster_summary["cluster_count"],
                    "separation": cluster_summary["separation"],
                    "clusters": cluster_summary["clusters"][:limit],
                }
            )

        return evidence

    def _cluster_separation_score(self, pair: pd.DataFrame, a: str, b: str) -> float:
        if pair.empty:
            return 0.0

        work = pair[[a, b]].copy()
        work[a] = work[a].map(normalize_value)
        work[b] = pd.to_numeric(work[b], errors="coerce")
        work = work[(work[a] != "") & work[b].notna() & (work[b] > 0)]

        if work.empty:
            return 0.0

        category_scores: list[float] = []
        dominant_centers: list[float] = []

        for _, group in work.groupby(a):
            summary = self._cluster_series(group[b])
            if summary["cluster_count"] == 0:
                continue

            category_scores.append(float(summary["separation"]))
            if summary["dominant_center"] is not None:
                dominant_centers.append(float(summary["dominant_center"]))

        if not category_scores:
            return 0.0

        within = float(np.mean(category_scores))
        between = float(np.tanh(np.std(dominant_centers))) if len(dominant_centers) > 1 else 0.0

        return float(np.clip(0.65 * within + 0.35 * between, 0.0, 1.0))

    def analyze(self, df: pd.DataFrame, a: str, b: str) -> dict[str, Any] | None:
        if not self._is_applicable(df, a, b):
            return None

        pair = prepare_pair_frame(df, a, b)
        total_rows = len(df)
        row_count = len(pair)
        support = float(row_count / total_rows) if total_rows else 0.0

        separation = self._cluster_separation_score(pair, a, b)
        doc_alignment = float(self.doc_model.score(a, b))

        strength = (
            self.weights["support"] * support
            + self.weights["separation"] * separation
            + self.weights["doc_alignment"] * doc_alignment
        )

        return {
            "A": a,
            "B": b,
            "feature_type": "cluster",
            "support": support,
            "separation": separation,
            "predictive_strength": separation,
            "doc_alignment": doc_alignment,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "classification": self.classify_strength(strength),
            "row_count": row_count,
            "total_rows": total_rows,
            "evidence": self._build_evidence(pair, a, b),
        }