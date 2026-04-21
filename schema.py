from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd
import yaml, os

from dotenv import dotenv_values, load_dotenv

load_dotenv()
print("SKIP_FIELDS =", os.getenv("SKIP_FIELDS"))

@dataclass(frozen=True)
class PropertySchema:
    name: str
    description: str = ""
    type: str | None = None
    enum: list[str] = field(default_factory=list)
    required: bool | str | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts = [self.name.replace("_", " ")]
        if self.description:
            parts.append(self.description)
        if self.enum:
            # parts.append(" ".join(self.enum[:20]))
            parts.append(" ".join(str(v) for v in self.enum[:20]))
        return "\n".join(parts)


@dataclass(frozen=True)
class NodeSchema:
    name: str
    description: str = ""
    properties: dict[str, PropertySchema] = field(default_factory=dict)
    exclude_like: list[str] = field(default_factory=list)

    def property_names(self) -> list[str]:
        return list(self.properties.keys())

    def property_texts(self) -> dict[str, str]:
        return {name: prop.text for name, prop in self.properties.items()}



def load_node_schema(path: str | Path, nodes = []) -> NodeSchema:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    node_name = raw.get("node", "sample")
    description = raw.get("description", "")
    exclude_like = raw.get("exclude_like", []) or []

    properties: dict[str, PropertySchema] = {}
    for prop_name, spec in (raw.get("properties", {}) or {}).items():
        properties[prop_name] = PropertySchema(
            name=prop_name,
            description=spec.get("description", "") or "",
            type=spec.get("type"),
            enum=list(spec.get("enum", []) or []),
            required=spec.get("required"),
            tags=dict(spec.get("tags", {}) or {}),
        )

    return NodeSchema(
        name=node_name,
        description=description,
        properties=properties,
        exclude_like=[str(x) for x in exclude_like],
    )

def load_nodes_schema(nodes: list[str]) -> list[NodeSchema]:
    schemas: list[NodeSchema] = []
    base_path = Path(os.environ["ICDC_SCHEMA_OUTPUT_DIR"])

    for node in nodes:
        rp_node = f"{node}".replace(" ", "_")
        path = base_path / f"{rp_node}_node.yaml"

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        node_name = raw.get("node", "sample")
        description = raw.get("description", "")
        exclude_like = raw.get("exclude_like", []) or []

        properties: dict[str, PropertySchema] = {}
        for prop_name, spec in (raw.get("properties", {}) or {}).items():
            properties[prop_name] = PropertySchema(
                name=prop_name,
                description=spec.get("description", "") or "",
                type=spec.get("type"),
                enum=list(spec.get("enum", []) or []),
                required=spec.get("required"),
                tags=dict(spec.get("tags", {}) or {}),
            )

        schemas.append(NodeSchema(
            name=node_name,
            description=description,
            properties=properties,
            exclude_like=[str(x) for x in exclude_like],
        ))
    return schemas

def node_schemas_to_df(schemas: list[NodeSchema]) -> pd.DataFrame:
    rows = []
    for schema in schemas:
        rows.append(
            {
                "node": schema.name,
                "description": schema.description,
                "num_properties": len(schema.properties),
                "properties": ", ".join(schema.property_names()) if hasattr(schema, "property_names") else ", ".join(schema.properties.keys()),
                "exclude_like": ", ".join(schema.exclude_like),
            }
        )
    return pd.DataFrame(rows)

def build_node_schema_json(node_name, nodes, props):
    node_spec = nodes[node_name]
    merged = {
        "node": node_name,
        "description": node_spec.get("Desc", "") or "",
        "exclude_like": ["sample_id", "crdc_id", "comment", "uuid", "created", "updated"],
        "properties": {},
    }

    for p in node_spec.get("Props", []) or node_spec.get("props", []):
        if p in props:
            spec = props[p]
            merged["properties"][p] = {
                "description": spec.get("Desc", "") or "",
                "type": spec.get("Type", spec.get("type")),
                "enum": spec.get("Enum", spec.get("enum", [])) or [],
                "required": spec.get("Req", spec.get("required")),
                "tags": spec.get("Tags", spec.get("tags", {})) or {},
            }
    return merged

def json_to_node_schema(data: dict[str, Any]) -> NodeSchema:
    properties: dict[str, PropertySchema] = {}

    for prop_name, spec in (data.get("properties", {}) or {}).items():
        properties[prop_name] = PropertySchema(
            name=prop_name,
            description=spec.get("description", "") or "",
            type=spec.get("type"),
            enum=[str(v) for v in (spec.get("enum", []) or [])],
            required=spec.get("required"),
            tags=dict(spec.get("tags", {}) or {}),
        )

    return NodeSchema(
        name=data.get("node", "sample"),
        description=data.get("description", "") or "",
        properties=properties,
        exclude_like=[str(x) for x in (data.get("exclude_like", []) or [])],
    )

def load_schemas_from_models(nodes, properties) -> list[NodeSchema]:
    """
    Fetch node model + property model from URLs found in env text/upload,
    merge matching node/property definitions, and return list[NodeSchema].
    """
    schemas: list[NodeSchema] = []
    try:
        for node_name in nodes.keys():
            merged = build_node_schema_json(node_name, nodes, properties)
            # print(merged)
            schemas.append(json_to_node_schema(merged))
        return schemas, ""
    except Exception as e:
        return [],  f"error while loading schema \n {str(e)}"

def node_schemas_to_markdown(schemas: list[NodeSchema]) -> str:
    parts = []

    for s in schemas:
        parts.append(f"## {s.name}")

        if s.description:
            parts.append(f"**Description:** {s.description}")

        parts.append(f"**Properties ({len(s.properties)})**")

        for name, p in s.properties.items():
            parts.append(f"- **{name}**")
            parts.append(f"  - type: {p.type or 'N/A'}")
            parts.append(f"  - required: {p.required if p.required is not None else 'N/A'}")

            if p.description:
                parts.append(f"  - description: {p.description}")

            if p.enum:
                parts.append(f"  - enum: {', '.join(map(str, p.enum))}")

            if p.tags:
                tag_str = ", ".join(f"{k}={v}" for k, v in p.tags.items())
                parts.append(f"  - tags: {tag_str}")

        if s.exclude_like:
            parts.append(f"**Exclude like:** {', '.join(s.exclude_like)}")

        parts.append("")  # spacing

    return "\n".join(parts)