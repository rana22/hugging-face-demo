from __future__ import annotations

import os, copy
from pathlib import Path
from typing import Any

import requests
import yaml


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


def build_all_node_schemas(
    *,
    node_model_url: str,
    prop_model_url: str,
    output_dir: str | Path,
) -> list[Path]:
    node_model = load_yaml_from_url(node_model_url)
    prop_model = load_yaml_from_url(prop_model_url)

    nodes = _get_nodes(node_model)
    prop_defs = _get_prop_defs(prop_model)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for node_name, node_spec in nodes.items():
        if not isinstance(node_spec, dict):
            continue

        merged = _normalize_node_spec(node_name, node_spec, prop_defs)
        merged = remove_yaml_anchors(merged)

        out_path = out_dir / f"{node_name}_node.yaml"
        out_path.write_text(
            yaml.dump(
                merged,
                Dumper=NoAliasDumper,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        written.append(out_path)
    return written

def main() -> None:
    node_url = os.environ["NODE_MODEL_URL"]
    prop_url = os.environ["PROP_MODEL_URL"]

    output_dir = os.environ.get("SCHEMA_OUTPUT_DIR")

    paths = build_all_node_schemas(
        node_model_url=node_url,
        prop_model_url=prop_url,
        output_dir=output_dir,
    )

    for path in paths:
        print(f"Wrote {path.resolve()}")

if __name__ == "__main__":
    main()