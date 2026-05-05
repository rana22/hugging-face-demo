from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from collections import deque
from itertools import combinations
from feature.base import FeatureBase
from feature.fuzzy import FuzzyFeatureAnalyzer, should_skip_fuzzy, looks_like_date_column
from schema import NodeSchema
from feature.categorical import normalize_value

# --------------------------------------
# Helper: Build Cross Pair Frame
# --------------------------------------
def build_cross_pair_frame(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    a: str,
    b: str,
    max_rows: int = 200,
) -> pd.DataFrame:
    parent_vals = parent_df[a].dropna().astype(str).unique()
    child_vals = child_df[b].dropna().astype(str).head(max_rows)

    rows = []

    for pv in parent_vals:
        for cv in child_vals:
            pv_norm = normalize_value(pv)
            cv_norm = normalize_value(cv)

            if not pv_norm or not cv_norm:
                continue

            rows.append({a: pv_norm, b: cv_norm})

    if not rows:
        return pd.DataFrame(columns=[a, b])

    return pd.DataFrame(rows)

# --------------------------------------
# Helper: Infer Relationship Type
# --------------------------------------
def infer_cardinality(parent_df: pd.DataFrame, child_df: pd.DataFrame, a: str, b: str) -> str:
    try:
        parent_unique = parent_df[a].nunique()
        child_unique = child_df[b].nunique()

        if parent_unique == len(parent_df) and child_unique > parent_unique:
            return "one_to_many"

        if parent_unique > 1 and child_unique > 1:
            return "many_to_many"

        return "unknown"
    except Exception:
        return "unknown"

def build_graph(df):
    graph = {}
    for _, row in df.iterrows():
        graph.setdefault(row["parent"], []).append(row["child"])
    return graph


def find_path(graph, start, target):
    def dfs(node, path):
        if node == target:
            return path

        for nxt in graph.get(node, []):
            if nxt not in path:
                res = dfs(nxt, path + [nxt])
                if res:
                    return res
        return None

    return dfs(start, [start])


def find_selected_path(df, selected_nodes):
    graph = build_graph(df)

    a, b = selected_nodes

    path = find_path(graph, a, b)
    if path:
        return [path]

    # try reverse
    path = find_path(graph, b, a)
    if path:
        return [path[::-1]]

    return []


def extract_edges_from_path(df, path):
    path_set = set(path)

    edges = []

    for _, row in df.iterrows():
        p = row["parent"]
        c = row["child"]

        if p in path_set and c in path_set:
            edges.append((p, c))

    return edges

def shortest_path(graph, start, target):
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == target:
            return path

        if node in visited:
            continue

        visited.add(node)

        for nxt in graph.get(node, []):
            queue.append(path + [nxt])

    return None

# def find_selected_edges(df, selected_nodes):
#     graph = build_graph(df)

#     # get shortest path (better than DFS)
#     path = shortest_path(graph, selected_nodes[0], selected_nodes[1])

#     if not path:
#         path = shortest_path(graph, selected_nodes[1], selected_nodes[0])
#         if path:
#             path = path[::-1]

#     if not path:
#         return []
#     print(path)
#     edges = extract_edges_from_path(df, path)

#     return edges

# def find_selected_edges(df, selected_nodes):
#     graph = build_graph(df)
#     all_paths = []
#     seen_paths = set()
#     for start, end in combinations(selected_nodes, 2):
#         path = shortest_path(graph, start, end)

#         if not path:
#             path = shortest_path(graph, end, start)
#             if path:
#                 path = path[::-1]

#         if not path:
#             continue

#         path_key = tuple(path)
#         if path_key in seen_paths:
#             continue
#         seen_paths.add(path_key)

#         edges = extract_edges_from_path(df, path)
        
#         # debuging
#         all_paths.append({
#             "start": start,
#             "end": end,
#             "path": path,
#             "edges": edges,
#         })
#     # print(all_paths)
#     return all_paths

def find_selected_edges(df, selected_nodes):
    graph = build_graph(df)

    unique_edges = set()

    for start, end in combinations(selected_nodes, 2):
        path = shortest_path(graph, start, end)

        if not path:
            path = shortest_path(graph, end, start)
            if path:
                path = path[::-1]

        if not path:
            continue

        # convert path → edges
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            unique_edges.add((parent, child))

    return list(unique_edges)

class _NullDocAlignmentModel:
    def score(self, a: str, b: str) -> float:
        return 0.0


class _CrossNodeSchema:
    name = "cross_node_match"


def infer_cardinality(parent_df: pd.DataFrame, child_df: pd.DataFrame, a: str, b: str) -> str:
    try:
        parent_unique = parent_df[a].nunique()
        child_unique = child_df[b].nunique()

        if parent_unique == len(parent_df) and child_unique > parent_unique:
            return "one_to_many"

        if parent_unique == 1 and child_unique == 1:
            return "one_to_one"

        if parent_unique > 1 and child_unique > 1:
            return "many_to_many"

        return "unknown"
    except Exception:
        return "unknown"

def _sample_values(values, max_n=40):
    values = pd.Series(values).dropna().astype(str).str.lower().drop_duplicates()
    if len(values) > max_n:
        values = values.sample(n=max_n, random_state=0)
    return values.tolist()

class CrossNodeRelationshipEvaluator(FeatureBase):
    def __init__(
        self,
        node_schemas: Optional[Dict[str, NodeSchema]] = None,
    ):
        self.node_schemas = node_schemas or {}

        self.fuzzy = FuzzyFeatureAnalyzer(
            node_schema=_CrossNodeSchema(),
            doc_model=_NullDocAlignmentModel(),
        )

    def analyze(
        self,
        node_dfs: Dict[str, pd.DataFrame],
        edges: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Analyze cross-node relationships for an explicit list of edges.

        Example:
            edges = [("parent", "child1"), ("child1", "child2")]
        """
        results: list[dict[str, Any]] = []

        if not edges:
            return pd.DataFrame()

        seen_edges = set()

        for parent_node, child_node in edges:
            print(f"Parent -> Child: {parent_node} -> {child_node}")
            edge_key = (parent_node, child_node)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            parent_df = node_dfs.get(parent_node)
            child_df = node_dfs.get(child_node)

            if parent_df is None or child_df is None:
                continue

            pair_results = self._evaluate_pair(
                parent_node=parent_node,
                child_node=child_node,
                parent_df=parent_df,
                child_df=child_df,
            )
            results.extend(pair_results)
        print("comvert to df")
        df = pd.DataFrame(results)

        if not df.empty:
            sort_cols = [c for c in ["strength", "support"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

        return df

    def _evaluate_pair(
        self,
        parent_node: str,
        child_node: str,
        parent_df: pd.DataFrame,
        child_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        parent_cols = list(parent_df.columns)
        child_cols = list(child_df.columns)

        threshold = 0.80

        # remove columns with date pattern
        parent_cols = [
            c for c in parent_df.columns
            if not looks_like_date_column(parent_df[c])
        ]
        child_cols = [
            c for c in child_df.columns
            if not looks_like_date_column(child_df[c])
        ]

        # Precompute child values once per child column.
        child_values_by_col = {
            b: child_df[b].dropna().astype(str).str.lower().unique()[:200]
            for b in child_cols
        }

        for a in parent_cols:
            found_strong = False

            parent_values = parent_df[a].dropna().astype(str).str.lower().tolist()
            if not parent_values:
                continue

            for b in child_cols:
                if a == b:
                    continue

                child_values = child_values_by_col.get(b, [])
                if len(child_values) == 0:
                    continue
                
                max_parent_vals = 40
                max_child_vals = 40
                max_pairs_per_column = 1000
                parent_values = _sample_values(parent_df[a], max_parent_vals)
                child_values = _sample_values(child_df[b], max_child_vals)

                # Build a small pair frame for the reusable fuzzy analyzer.
                pair_rows: list[dict[str, Any]] = []
                # for av in parent_values:
                #     for cv in child_values:
                #         # avoid date pattern
                #         # if should_skip_fuzzy(av, cv):
                #         #     continue
                #         pair_rows.append({
                #             a: av,
                #             b: cv,
                #             "type": "cross_node_match",
                #             "name": "cross_node_match",
                #         })
                for av in parent_values:
                    for cv in child_values:
                        pair_rows.append({
                            a: av,
                            b: cv,
                            "type": "cross_node_match",
                            "name": "cross_node_match",
                        })
                        if len(pair_rows) >= max_pairs_per_column:
                            break
                    if len(pair_rows) >= max_pairs_per_column:
                        break

                if not pair_rows:
                    continue
                
                pair_df = pd.DataFrame(pair_rows)

                result = self.fuzzy.analyze(pair_df, a, b)
                if result is None:
                    continue

                strength = float(result.get("strength", 0.0))
                relation_type = infer_cardinality(parent_df, child_df, a, b)

                result.update({
                    "parent_node": parent_node,
                    "child_node": child_node,
                    "A": f"{parent_node}.{a}",
                    "B": f"{child_node}.{b}",
                    "feature_type": "cross_node_fuzzy",
                    "relation_type": relation_type,
                })

                results.append(result)

                # Stop once a strong likely join-like relationship is found.
                if strength >= threshold and relation_type in ("one_to_many", "one_to_one"):
                    found_strong = True
                    break

            if found_strong:
                break

        return results