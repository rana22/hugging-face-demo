from __future__ import annotations

import os, copy
import pandas as pd
from pathlib import Path
from typing import Any
from collections import deque
import requests
import yaml

from node_relation import extract_relationships, relationships_to_dataframe, enrich_relationships_with_properties

from graph.DFS import build_graph, bfs_layers, extract_clusters, cluster_with_relations_df, deduplicate

DEFAULT_SKIP = {"sample_id", "crdc_id", "comment", "uuid", "created", "updated"}

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def remove_yaml_anchors(data: Any) -> Any:
    return copy.deepcopy(data)

def load_yaml_from_url(url: str) -> dict[str, Any]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = yaml.safe_load(resp.text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object from {url}, got {type(data).__name__}")
    return data


def _get_nodes(node_model: dict[str, Any]) -> dict[str, Any]:
    nodes = node_model.get("Nodes") or node_model.get("nodes") or {}
    if not isinstance(nodes, dict):
        raise ValueError("Node model does not contain a valid Nodes mapping.")
    return nodes


def _get_prop_defs(prop_model: dict[str, Any]) -> dict[str, Any]:
    props = prop_model.get("PropDefinitions") or prop_model.get("properties") or {}
    if not isinstance(props, dict):
        raise ValueError("Property model does not contain a valid PropDefinitions mapping.")
    return props


def _normalize_node_spec(node_name: str, node_spec: dict[str, Any], prop_defs: dict[str, Any]) -> dict[str, Any]:
    props = node_spec.get("Props") or node_spec.get("props") or []
    if not isinstance(props, list):
        raise ValueError(f"Node '{node_name}' Props must be a list.")

    merged_props: dict[str, Any] = {}
    for prop_name in props:
        if prop_name not in prop_defs:
            continue

        spec = prop_defs[prop_name]
        if not isinstance(spec, dict):
            continue

        # Keep the full property spec, but normalize common keys so the
        # generated YAML is easy to consume downstream.
        merged_props[prop_name] = {
            "Desc": spec.get("Desc", spec.get("description", "")),
            "Src": spec.get("Src", spec.get("source", "")),
            "Type": spec.get("Type", spec.get("type")),
            "Enum": spec.get("Enum", spec.get("enum", [])) or [],
            "Req": spec.get("Req", spec.get("required")),
            "Tags": spec.get("Tags", spec.get("tags", {})) or {},
        }

        # Preserve optional/source flags when present
        for extra_key in ("Key", "Auto", "Deprecated", "Term"):
            if extra_key in spec:
                merged_props[prop_name][extra_key] = spec[extra_key]

    exclude_like = [p for p in DEFAULT_SKIP if p in merged_props]

    out: dict[str, Any] = {
        "node": node_name,
        "description": node_spec.get("Desc", node_spec.get("description", "")),
        "tags": node_spec.get("Tags", node_spec.get("tags", {})) or {},
        "props": props,
        "exclude_like": exclude_like,
        "properties": merged_props,
    }

    # Keep the raw node spec too, so nothing is lost.
    out["node_spec"] = node_spec
    return out

# def build_relation_schemas():
#     node_url = os.getenv("NODE_MODEL_URL", "")
#     prop_url = os.getenv("PROP_MODEL_URL", "")
#     node_model = load_yaml_from_url(node_url)
#     prop_model = load_yaml_from_url(prop_url)

#     nodes = _get_nodes(node_model)
#     prop_defs = _get_prop_defs(prop_model)

#     node_schemas = {}
#     try:
#         for node_name, node_spec in nodes.items():
#             if not isinstance(node_spec, dict):
#                 continue

#             merged = _normalize_node_spec(node_name, node_spec, prop_defs)
#             merged = remove_yaml_anchors(merged)
#             node_schemas[node_name] = merged
#         relationships = extract_relationships(node_model)
#         relationships = enrich_relationships_with_properties(relationships, node_schemas)
#         rel_df = relationships_to_dataframe(relationships)
#         return node_schemas, rel_df, ""
#     except Exception as e:
#         return {}, None, f"Error on building relation \n {str(e)}"

def filter_clusters_by_nodes(clusters, node_list):
    node_set = set(node_list)

    filtered = []

    for cluster in clusters:
        if all(node in node_set for node in cluster):
            filtered.append(cluster)

    return filtered

def build_join_paths_df(paths, rel_df):
    rows = []

    for path_id, path in enumerate(paths):
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]

            rel = rel_df[
                (rel_df["parent"] == parent) &
                (rel_df["child"] == child)
            ]

            for _, r in rel.iterrows():
                rows.append({
                    "path_id": path_id,
                    "step": i,
                    "parent": parent,
                    "child": child,
                    "type": r["type"],
                    "parent_prop": r["parent_prop"],
                    "child_prop": r["child_prop"],
                    "path": " → ".join(path)
                })

    return pd.DataFrame(rows)

def build_relation_schemas(node_list_state):
    import pandas as pd

    node_url = os.getenv("NODE_MODEL_URL", "")
    prop_url = os.getenv("PROP_MODEL_URL", "")

    try:
        node_model = load_yaml_from_url(node_url)
        prop_model = load_yaml_from_url(prop_url)

        nodes = _get_nodes(node_model)
        prop_defs = _get_prop_defs(prop_model)

        rows = []
        node_schemas = {}

        # ✅ Build schemas (same as before)
        for node_name, node_spec in nodes.items():
            if not isinstance(node_spec, dict):
                continue

            merged = _normalize_node_spec(node_name, node_spec, prop_defs)
            merged = remove_yaml_anchors(merged)
            node_schemas[node_name] = merged

        # ✅ Process relationships directly → rows
        relationships = node_model.get("Relationships", {})
        # avoid circular dependencies
        explored_src_node = {}

        for rel_name, rel_spec in relationships.items():
            default_mul = rel_spec.get("Mul", "many_to_one")

            for end in rel_spec.get("Ends", []):
                src = end.get("Src")
                dst = end.get("Dst")

                if src == dst:
                    continue

                if not src or not dst:
                    continue

                mul = (end.get("Mul") or default_mul).lower()

                # --- resolve direction ---
                if mul == "many_to_one":
                    parent, child, rel_type = dst, src, "one_to_many"
                elif mul == "one_to_many":
                    parent, child, rel_type = src, dst, "one_to_many"
                elif mul == "one_to_one":
                    parent, child, rel_type = dst, src, "one_to_one"
                elif mul == "many_to_many":
                    parent, child, rel_type = src, dst, "many_to_many"
                else:
                    parent, child, rel_type = dst, src, mul

                parent_props = node_schemas.get(parent, {}).get("properties", {})
                child_props = node_schemas.get(child, {}).get("properties", {})

                matches = []

                # ✅ simple inline matching (no extra function)
                for p in parent_props:
                    for c in child_props:
                        if p == c or (p in c or c in p):
                            # optional filter → only id-like
                            if "id" in p or "id" in c:
                                matches.append((p, c))

                # ✅ write rows directly
                if not matches:
                    rows.append({
                        "relation": rel_name,
                        "parent": parent,
                        "child": child,
                        "type": rel_type,
                        "parent_prop": None,
                        "child_prop": None,
                    })
                else:
                    for p, c in matches:
                        rows.append({
                            "relation": rel_name,
                            "parent": parent,
                            "child": child,
                            "type": rel_type,
                            "parent_prop": f"{parent}.{p}",
                            "child_prop": f"{child}.{c}",
                        })

        df = pd.DataFrame(rows).sort_values(
            by=["parent", "child"]
        ).reset_index(drop=True)
        node_tree_df = df.copy()

        # build node tree
        graph = build_graph(df)
        paths = bfs_layers(graph)
        valid_paths = [p for p in paths if 2 <= len(p) <= 4]
        valid_paths = filter_paths_by_nodes(valid_paths, node_list_state)
        valid_paths = valid_paths[:10]

        join_df = build_join_paths_df(valid_paths, df)
        return node_schemas, df, "", node_tree_df, join_df

    except Exception as e:
        return {}, None, f"Error on building relation\n{str(e)}"

# def build_relation_schemas(node_list_state):
#     import pandas as pd

#     node_url = os.getenv("NODE_MODEL_URL", "")
#     prop_url = os.getenv("PROP_MODEL_URL", "")

#     try:
#         node_model = load_yaml_from_url(node_url)
#         prop_model = load_yaml_from_url(prop_url)

#         nodes = _get_nodes(node_model)
#         prop_defs = _get_prop_defs(prop_model)

#         rows = []
#         node_schemas = {}

#         # ✅ Build schemas (same as before)
#         for node_name, node_spec in nodes.items():
#             if not isinstance(node_spec, dict):
#                 continue

#             merged = _normalize_node_spec(node_name, node_spec, prop_defs)
#             merged = remove_yaml_anchors(merged)
#             node_schemas[node_name] = merged

#         # ✅ Process relationships directly → rows
#         relationships = node_model.get("Relationships", {})
#         # avoid circular dependencies
#         explored_src_node = {}

#         for rel_name, rel_spec in relationships.items():
#             default_mul = rel_spec.get("Mul", "many_to_one")

#             for end in rel_spec.get("Ends", []):
#                 src = end.get("Src")
#                 dst = end.get("Dst")

#                 if src == dst:
#                     continue

#                 if not src or not dst:
#                     continue

#                 mul = (end.get("Mul") or default_mul).lower()

#                 # --- resolve direction ---
#                 if mul == "many_to_one":
#                     parent, child, rel_type = dst, src, "one_to_many"
#                 elif mul == "one_to_many":
#                     parent, child, rel_type = src, dst, "one_to_many"
#                 elif mul == "one_to_one":
#                     parent, child, rel_type = dst, src, "one_to_one"
#                 elif mul == "many_to_many":
#                     parent, child, rel_type = src, dst, "many_to_many"
#                 else:
#                     parent, child, rel_type = dst, src, mul

#                 parent_props = node_schemas.get(parent, {}).get("properties", {})
#                 child_props = node_schemas.get(child, {}).get("properties", {})

#                 matches = []

#                 # ✅ simple inline matching (no extra function)
#                 for p in parent_props:
#                     for c in child_props:
#                         if p == c or (p in c or c in p):
#                             # optional filter → only id-like
#                             if "id" in p or "id" in c:
#                                 matches.append((p, c))

#                 # ✅ write rows directly
#                 if not matches:
#                     rows.append({
#                         "relation": rel_name,
#                         "parent": parent,
#                         "child": child,
#                         "type": rel_type,
#                         "parent_prop": None,
#                         "child_prop": None,
#                     })
#                 else:
#                     for p, c in matches:
#                         rows.append({
#                             "relation": rel_name,
#                             "parent": parent,
#                             "child": child,
#                             "type": rel_type,
#                             "parent_prop": f"{parent}.{p}",
#                             "child_prop": f"{child}.{c}",
#                         })

#         df = pd.DataFrame(rows).sort_values(
#             by=["parent", "child"]
#         ).reset_index(drop=True)

#         # build node tree
#         graph = build_graph(df)
#         path = bfs_layers(graph)
#         clusters = extract_clusters(path)
#         dedup_clusters = deduplicate(clusters)

#         filtered_clusters = filter_clusters_by_nodes(
#             dedup_clusters,
#             node_list_state
#         )
#         dedup_clusters = filtered_clusters[:10]
#         cluster_df = cluster_with_relations_df(dedup_clusters, df)

#         return node_schemas, df, "", cluster_df

#     except Exception as e:
#         return {}, None, f"Error on building relation\n{str(e)}"

def filter_paths_by_nodes(paths, node_list):
    node_set = set(node_list)
    return [
        p for p in paths
        if set(p).issubset(node_set)
    ]

def find_paths(graph, start, target):
    queue = deque([(start, [start])])
    paths = []

    while queue:
        node, path = queue.popleft()

        if node == target:
            paths.append(path)
            continue

        for neighbor in graph.get(node, []):
            if neighbor not in path:  # avoid cycles
                queue.append((neighbor, path + [neighbor]))

    return paths




    