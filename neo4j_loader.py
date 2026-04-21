from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional
import json
import pandas as pd
from dotenv import dotenv_values, load_dotenv
from neo4j import GraphDatabase
from pathlib import Path
from schema import NodeSchema

load_dotenv()
print("SKIP_FIELDS =", os.getenv("SKIP_FIELDS"))

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str

@dataclass
class Neo4jClient:
    uri: str
    user: str
    password: str

    def __post_init__(self) -> None:
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self.driver.close()

    def run_query(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self.driver.session() as session:
            result = session.run(query, params or {})
            # return pd.DataFrame([record.data() for record in result])
            records = [record.data() for record in result]

        if not records:
            return pd.DataFrame()

        if set(records[0].keys()) == {"row"} and isinstance(records[0]["row"], dict):
            return pd.DataFrame([record["row"] for record in records])

        return pd.DataFrame(records)

def load_neo4j_config(env_path: str | Path | None = None) -> Neo4jConfig:
    """Load Neo4j settings from environment variables or a .env file."""
    if env_path is not None:
        load_dotenv(env_path, override=False)
        env = dotenv_values(env_path)
    else:
        load_dotenv(override=False)
        env = dotenv_values()

    uri = env.get("NEO4J_URI") or os.getenv("NEO4J_URI")
    user = env.get("NEO4J_USER") or os.getenv("NEO4J_USER")
    password = env.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")
    database = env.get("NEO4J_DATABASE") or os.getenv("NEO4J_DATABASE") or None

    missing = [name for name, value in (("NEO4J_URI", uri), ("NEO4J_USER", user), ("NEO4J_PASSWORD", password)) if not value]
    if missing:
        raise ValueError(
            "Missing Neo4j configuration in environment: " + ", ".join(missing)
            + ". Set them in .env or export them before running the command."
        )

    return Neo4jConfig(uri=str(uri), user=str(user), password=str(password))

def _format_label(label: str) -> str:
    label = str(label).strip()
    if not label:
        raise ValueError("Node label cannot be empty")
    return label

def build_neo4j_query(schema: NodeSchema, study_id: str | None = None) -> str:
    """
    Build a Neo4j query dynamically from the node schema.

    - Uses the schema node name as the label
    - Returns only schema-defined properties
    - Optionally filters by study_id for study-scoped extraction
    """
    node_label = schema.name
    props = list(schema.properties.keys())
    projection = ", ".join(f".{p}" for p in props)

    # study node: direct match
    if node_label.lower() == "study":
        if study_id:
            return f"""
            MATCH (n:study)
            WHERE n.clinical_study_designation = $study_id
            RETURN n {{ {projection} }} AS row
            """
        return f"""
        MATCH (n:study)
        RETURN n {{ {projection} }} AS row
        """

    # case / sample / other downstream nodes: study-scoped traversal
    depth_by_node = {
        "case": 2,
        "sample": 3,
        "file": 3,
    }

    depth = depth_by_node.get(node_label.lower(), 3)
    if study_id:
        return f"""
        MATCH (s:study)<-[*1..{depth}]-(n:{node_label})
        WHERE s.clinical_study_designation = $study_id
        RETURN n {{ {projection} }} AS row
        """

    return f"""
        MATCH (n:{node_label})
        RETURN n {{ {projection} }} AS row
    """

# def build_node_projection_query(
#     node_label: str,
#     properties: Iterable[str] | None = None,
#     *,
#     alias: str = "n",
#     limit: int | None = None,
#     skip_fields: set[str] | None = None,
# ) -> str:
#     """Build a Cypher query that works for any node type.

#     Example:
#         MATCH (n:sample)
#         RETURN n { .sample_site, .general_sample_pathology } AS row
#     """
#     label = _format_label(node_label)
#     DEFAULT_SKIP_FIELDS = os.getenv("SKIP_FIELDS", "")
#     skip = DEFAULT_SKIP_FIELDS | (skip_fields or set())
#     props = [str(p).strip() for p in (properties or []) if str(p).strip() and str(p).strip() not in skip]
#     if props:
#         projection = ", ".join(f".{p}" for p in props)
#         query = f"MATCH ({alias}:{label}) RETURN {alias} {{ {projection} }} AS row"
#     else:
#         query = f"MATCH ({alias}:{label}) RETURN {alias} {{ .* }} AS row"
#     if limit is not None and int(limit) > 0:
#         query += f" LIMIT {int(limit)}"
#     return query

# def fetch_rows_from_neo4j(
#     node_label: str,
#     properties: Iterable[str] | None = None,
#     *,
#     params: dict[str, Any] | None = None,
#     limit: int | None = None,
#     env_path: str | Path | None = None,
# ) -> pd.DataFrame:
#     config = load_neo4j_config(env_path)
#     client = Neo4jClient(config.uri, config.user, config.password)
#     try:
#         cypher = build_neo4j_query(node_label, properties)
#         return client.run_query(cypher, params)
#     finally:
#         client.close()

def fetch_rows_from_neo4j(
    schema: NodeSchema,
    *,
    study_id: str | None = None,
    params: dict[str, Any] | None = None,
    limit: int | None = None,
    env_path: str | Path | None = None,
) -> pd.DataFrame:
    config = load_neo4j_config(env_path)
    client = Neo4jClient(config.uri, config.user, config.password)
    try:
        cypher = build_neo4j_query(schema, study_id=study_id)
        query_params = dict(params or {})
        if study_id is not None:
            query_params.setdefault("study_id", study_id)
        return client.run_query(cypher, query_params)
    finally:
        client.close()

def fetch_sample_node_rows(client: Neo4jClient, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a Cypher query that returns flattened sample-node rows."""
    return client.run_query(query, params)

def export_query_to_json(
    client: Neo4jClient,
    query: str,
    params: dict[str, Any] | None = None,
    output_file: str = None,
    root_key: str | None = None,
) -> None:
    with client.driver.session() as session:
        result = session.run(query, params or {})
        rows = [record.data() for record in result]

    if root_key is not None:
        rows = [row[root_key] for row in rows]

    if output_file is not None:
        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()

        output_path = base_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

        print(f"Saved to: {output_path}")

    return rows


if __name__ == "__main__":
    uri="bolt://localhost:11011"
    user="neo4j"
    password="root"
    client = Neo4jClient(uri, user, password)

    study_query = """
        MATCH (s:study)
        WHERE s.clinical_study_designation = $study_id
        RETURN s
        """
    study_id = "OSA01"
    study_param = {"study_id": study_id}

    sample_query = """
        MATCH (s:study)<-[*]-(smpl:sample)
        WHERE s.clinical_study_designation = $study_id
        RETURN smpl { .* } AS smpl
    """

    try:
        export_query_to_json(
            client,
            sample_query,
            study_param,
            output_file="sample_123.json",
            root_key="smpl"
        )

    except Exception as e:
        print("An error occurred:", e)
    finally:
        client.close()
