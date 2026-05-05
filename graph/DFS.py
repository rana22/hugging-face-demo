from collections import deque
import pandas as pd

def build_graph(rel_df):
    if not isinstance(rel_df, pd.DataFrame):
        rel_df = pd.DataFrame(rel_df)

    graph = {}

    for _, row in rel_df.iterrows():
        parent = row["parent"]
        child = row["child"]

        graph.setdefault(parent, []).append(child)

    return graph

def bfs_layers(graph, root="program"):
    visited = set()
    queue = deque([(root, [root])])

    paths = []

    while queue:
        node, path = queue.popleft()

        if node in visited:
            continue

        visited.add(node)
        paths.append(path)

        for child in graph.get(node, []):
            if child not in path:
                queue.append((child, path + [child]))

    return paths

def extract_clusters(paths, min_size=2, max_size=4):
    clusters = []

    for path in paths:
        for i in range(len(path)):
            for j in range(i + min_size, min(i + max_size + 1, len(path) + 1)):
                cluster = path[i:j]
                clusters.append(cluster)

    return clusters

def deduplicate(clusters):
    seen = set()
    result = []

    for c in clusters:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            result.append(c)

    return result

def prioritize_clusters(clusters):
    priority_nodes = {"study", "case"}

    return sorted(
        clusters,
        key=lambda c: sum(1 for n in c if n in priority_nodes),
        reverse=True
    )

def get_cluster_relations(cluster, rel_df):
    nodes = set(cluster)

    return rel_df[
        (rel_df["parent"].isin(nodes)) &
        (rel_df["child"].isin(nodes))
    ]

def clusters_to_df(clusters):
    rows = []

    for i, cluster in enumerate(clusters):
        rows.append({
            "cluster_id": i,
            "nodes": " -> ".join(cluster),
            "size": len(cluster)
        })

    return pd.DataFrame(rows)

def clusters_to_long_df(clusters):
    rows = []

    for i, cluster in enumerate(clusters):
        for step, node in enumerate(cluster):
            rows.append({
                "cluster_id": i,
                "step": step,
                "node": node,
                "cluster_size": len(cluster)
            })

    return pd.DataFrame(rows)

def cluster_with_relations_df(clusters, rel_df):
    rows = []

    for i, cluster in enumerate(clusters):
        nodes = set(cluster)

        sub_rel = rel_df[
            (rel_df["parent"].isin(nodes)) &
            (rel_df["child"].isin(nodes))
        ]

        for _, r in sub_rel.iterrows():
            rows.append({
                "cluster_id": i,
                "cluster": " -> ".join(cluster),
                "parent": r["parent"],
                "child": r["child"],
                "type": r["type"],
                "parent_prop": r["parent_prop"],
                "child_prop": r["child_prop"],
            })

    return pd.DataFrame(rows)