from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features import MISSING_STRINGS, normalize_value, is_identifier_like
from util import load_json_rows, save_dataframe, rows_to_json
from schema import NodeSchema, load_node_schema, PropertySchema
from neo4j_loader import fetch_rows_from_neo4j


CLASSIFICATION_PRIORITY = {
    "functional": 3,
    "strong": 2,
    "conditional": 1,
}

RETAINED_CLASSIFICATIONS = set(CLASSIFICATION_PRIORITY)

DEFAULT_SKIP = {
    "sample_id",
    "crdc_id",
    "comment",
    "uuid",
    "created",
    "updated",
}


@dataclass(frozen=True)
class Relationship:
    A: str
    B: str
    strength: float
    classification: str
    evidence: dict[str, Any]

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _clean_field_name(value: str) -> str:
    return str(value).strip()


def _load_json_column_map(value: Any) -> dict[str, dict[str, float]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(k): {str(kk): float(vv) for kk, vv in (inner or {}).items()}
            for k, inner in value.items()
        }
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "{}":
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return _load_json_column_map(parsed)
        except Exception:
            return {}
    return {}


def _pick_top_k(prob_map: dict[str, float], top_p: float = 0.95, temperature: float = 1.0) -> dict[str, float]:
    if not prob_map:
        return {}
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    total = float(sum(v for _, v in items))
    if total <= 0:
        return {}

    normed = [(k, v / total) for k, v in items]
    kept: list[tuple[str, float]] = []
    running = 0.0
    for k, p in normed:
        kept.append((k, p))
        running += p
        if running >= top_p:
            break

    if not kept:
        kept = normed[:1]

    probs = np.array([p for _, p in kept], dtype=float)
    if temperature and temperature != 1.0:
        probs = np.power(probs, 1.0 / max(1e-9, temperature))
    probs = probs / probs.sum()
    return {k: float(p) for (k, _), p in zip(kept, probs, strict=False)}


def weighted_choice(prob_map: dict[str, float], rng: np.random.Generator) -> str:
    if not prob_map:
        return ""
    keys = list(prob_map.keys())
    probs = np.array([prob_map[k] for k in keys], dtype=float)
    total = probs.sum()
    if total <= 0:
        return str(rng.choice(keys))
    probs = probs / total
    return str(rng.choice(keys, p=probs))

def _mapping_value_set(value: Any) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, list):
        return {str(v).strip() for v in value if str(v).strip()}

    if isinstance(value, str):
        text = value.strip()
        if not text or text == "{}":
            return set()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return {str(v).strip() for v in parsed if str(v).strip()}
            if isinstance(parsed, str):
                return {parsed.strip()} if parsed.strip() else set()
        except Exception:
            return {text}
        return set()

    return {str(value).strip()} if str(value).strip() else set()

def _parse_cluster_evidence(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            value = json.loads(text)
        except Exception:
            return []

    if not isinstance(value, list):
        return []

    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
    return out

def _parse_a_to_b_mapping(value: Any) -> dict[str, set[str]]:
    if value is None:
        return {}

    if isinstance(value, str):
        text = value.strip()
        if not text or text == "{}":
            return {}
        try:
            value = json.loads(text)
        except Exception:
            return {}

    if not isinstance(value, dict):
        return {}

    out: dict[str, set[str]] = {}
    for a_val, b_vals in value.items():
        out[str(a_val)] = _mapping_value_set(b_vals)
    return out

class SyntheticDataGenerator:
    def __init__(
        self,
        real_rows: pd.DataFrame,
        relationships: pd.DataFrame,
        schema: NodeSchema,
        *,
        seed: int = 42,
        top_p: float = 0.95,
        temperature_strong: float = 1.0,
        temperature_conditional: float = 1.1,
        include_low_value_fields: bool = False,
        max_validation_rounds: int = 5,
    ) -> None:
        self.real_rows = real_rows.copy()
        self.relationships = relationships.copy()
        self.schema = schema
        self.rng = np.random.default_rng(seed)
        self.top_p = top_p
        self.temperature_strong = temperature_strong
        self.temperature_conditional = temperature_conditional
        self.include_low_value_fields = include_low_value_fields
        self.max_validation_rounds = max_validation_rounds

        self.allowed_fields = self._build_allowed_fields()
        self.relationship_list = self._build_relationship_list()
        self.relationship_sort_order = sorted(
            self.relationship_list,
            key=lambda r: (
                CLASSIFICATION_PRIORITY.get(r.classification, 0),
                r.strength,
            ),
            reverse=True,
        )
        self.parent_for_child = self._build_parent_map()
        self.conditional_maps = self._build_conditional_maps()
        self.generation_order = self._build_generation_order()
        self.marginals = self._build_marginals()

    def _build_allowed_fields(self) -> list[str]:
        fields: list[str] = []
        for name, prop in self.schema.properties.items():
            if name in DEFAULT_SKIP:
                continue
            if is_identifier_like(name):
                continue
            if name in self.schema.exclude_like:
                continue
            if name not in self.real_rows.columns:
                continue

            series = self.real_rows[name].map(normalize_value)
            non_empty = float((series != "").mean()) if len(series) else 0.0

            if not self.include_low_value_fields:
                if non_empty < 0.20:
                    continue

            fields.append(name)
        return fields

    # def _row_is_valid(self, row: dict[str, Any]) -> bool:
    #     for rel in self.relationship_list:
    #         a_val = normalize_value(row.get(rel.A))
    #         b_val = normalize_value(row.get(rel.B))

    #         if not a_val or not b_val:
    #             return False

    #         raw_map = rel.evidence.get("a_to_b_mapping")
    #         mapping = _parse_a_to_b_mapping(raw_map)

    #         # If we have no mapping for this pair, the row cannot be validated safely.
    #         if not mapping:
    #             return False

    #         allowed = mapping.get(a_val)
    #         if allowed is None:
    #             return False

    #         if allowed and b_val not in allowed:
    #             return False

    #     return True

    def _cluster_value_in_range(self, row: dict[str, Any], rel: Relationship, value: Any) -> bool:
        a_val = normalize_value(row.get(rel.A))
        if not a_val:
            return False

        raw_evidence = rel.evidence.get("evidence")
        cluster_rows = _parse_cluster_evidence(raw_evidence)
        if not cluster_rows:
            return False

        matched_bucket = None
        for item in cluster_rows:
            category_value = normalize_value(
                item.get("category_value", item.get("category", ""))
            )
            if category_value == a_val:
                matched_bucket = item
                break

        if matched_bucket is None:
            return False

        try:
            b_val = float(pd.to_numeric(value, errors="coerce"))
        except Exception:
            return False

        if pd.isna(b_val):
            return False

        clusters = matched_bucket.get("clusters", [])
        if isinstance(clusters, str):
            try:
                clusters = json.loads(clusters)
            except Exception:
                clusters = []

        if not isinstance(clusters, list) or not clusters:
            return False

        for c in clusters:
            if not isinstance(c, dict):
                continue

            min_v = c.get("min_value")
            max_v = c.get("max_value")
            if min_v is None or max_v is None:
                continue

            try:
                min_v = float(min_v)
                max_v = float(max_v)
            except Exception:
                continue

            pad = max(1e-9, 0.05 * (max_v - min_v))
            if (min_v - pad) <= b_val <= (max_v + pad):
                return True

        return False

    def _row_is_valid(self, row: dict[str, Any]) -> bool:
        for rel in self.relationship_list:
            a_val = normalize_value(row.get(rel.A))
            b_val = normalize_value(row.get(rel.B))

            if not a_val or not b_val:
                return False

            feature_type = str(rel.evidence.get("feature_type", "")).strip().lower()

            raw_map = rel.evidence.get("a_to_b_mapping")
            mapping = _parse_a_to_b_mapping(raw_map)

            # 1) strict mapping validation
            if mapping:
                allowed = mapping.get(a_val)
                if allowed is None:
                    return False
                if allowed and b_val not in allowed:
                    return False
                continue

            # 2) substring validation
            if feature_type == "substring":
                if a_val not in b_val and b_val not in a_val:
                    return False
                continue

            # 3) clustering → DO NOT validate (handled in repair)
            if feature_type == "cluster":
                continue

            # 4) everything else without mapping → skip
            continue

        return True

    def validate_rows(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df is None or df.empty:
            return df, df

        mask = df.apply(lambda r: self._row_is_valid(r.to_dict()), axis=1)
        valid_df = df[mask].reset_index(drop=True)
        invalid_df = df[~mask].reset_index(drop=True)
        return valid_df, invalid_df

    def _build_relationship_list(self) -> list[Relationship]:
        if self.relationships.empty:
            return []

        rels: list[Relationship] = []
        for _, row in self.relationships.iterrows():
            a = _clean_field_name(row.get("A", ""))
            b = _clean_field_name(row.get("B", ""))
            if not a or not b:
                continue
            if a not in self.allowed_fields or b not in self.allowed_fields:
                continue

            classification = str(row.get("classification", "")).strip().lower()
            if classification not in RETAINED_CLASSIFICATIONS:
                continue

            strength = _as_float(row.get("strength", 0.0), 0.0)
            evidence = row.to_dict()
            rels.append(Relationship(A=a, B=b, strength=strength, classification=classification, evidence=evidence))
        return rels

    def _build_parent_map(self) -> dict[str, Relationship]:
        best: dict[str, Relationship] = {}
        for rel in self.relationship_list:
            current = best.get(rel.B)
            if current is None:
                best[rel.B] = rel
                continue
            if rel.strength > current.strength:
                best[rel.B] = rel
                continue
            if rel.strength == current.strength:
                if CLASSIFICATION_PRIORITY.get(rel.classification, 0) > CLASSIFICATION_PRIORITY.get(current.classification, 0):
                    best[rel.B] = rel
        return best

    def _build_conditional_maps(self) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
        maps: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        for rel in self.relationship_list:
            key = (rel.A, rel.B)
            if key in maps:
                continue

            pair = self.real_rows[[rel.A, rel.B]].copy()
            pair[rel.A] = pair[rel.A].map(normalize_value)
            pair[rel.B] = pair[rel.B].map(normalize_value)
            pair = pair[(pair[rel.A] != "") & (pair[rel.B] != "")]
            if pair.empty:
                maps[key] = {}
                continue

            conditional: dict[str, dict[str, float]] = {}
            for a_val, group in pair.groupby(rel.A):
                counts = group[rel.B].value_counts(normalize=True)
                conditional[str(a_val)] = {str(k): float(v) for k, v in counts.items()}
            maps[key] = conditional
        return maps

    def _build_generation_order(self) -> list[str]:
        fields = list(self.allowed_fields)
        indegree = {f: 0 for f in fields}
        children: dict[str, set[str]] = {f: set() for f in fields}

        for child, rel in self.parent_for_child.items():
            parent = rel.A
            if parent in indegree and child in indegree and parent != child:
                indegree[child] += 1
                children[parent].add(child)

        queue = [f for f in fields if indegree[f] == 0]
        order: list[str] = []
        seen = set()

        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            order.append(node)
            for child in sorted(children.get(node, set())):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        for field in fields:
            if field not in seen:
                order.append(field)
        return order

    def _build_marginals(self) -> dict[str, dict[str, float]]:
        marginals: dict[str, dict[str, float]] = {}
        for field in self.allowed_fields:
            series = self.real_rows[field].map(normalize_value)
            series = series[series != ""]
            if series.empty:
                marginals[field] = {}
                continue
            counts = series.value_counts(normalize=True)
            marginals[field] = {str(k): float(v) for k, v in counts.items()}
        return marginals

    def _sample_from_relation(self, rel: Relationship, parent_value: str) -> str:
        conditional = self.conditional_maps.get((rel.A, rel.B), {})
        probs = conditional.get(parent_value, {})
        if not probs:
            return ""
        if rel.classification == "functional":
            return max(probs.items(), key=lambda kv: kv[1])[0]
        top_p = self.top_p
        temperature = self.temperature_strong if rel.classification == "strong" else self.temperature_conditional
        filtered = _pick_top_k(probs, top_p=top_p, temperature=temperature)
        if filtered:
            return weighted_choice(filtered, self.rng)
        return max(probs.items(), key=lambda kv: kv[1])[0]

    def _apply_relation_reconciliation(self, row: dict[str, Any]) -> bool:
        changed = False
        for rel in self.relationship_sort_order:
            parent_value = normalize_value(row.get(rel.A))
            if not parent_value:
                continue

            current_child = normalize_value(row.get(rel.B))
            conditional = self.conditional_maps.get((rel.A, rel.B), {})
            if not conditional:
                continue

            allowed = set(conditional.get(parent_value, {}).keys())
            if not allowed:
                continue

            if rel.classification == "functional":
                desired = self._sample_from_relation(rel, parent_value)
                if desired and current_child != desired:
                    row[rel.B] = desired
                    changed = True
                continue
            
            # For strong/conditional, repair missing or incompatible children.
            if not current_child or current_child not in allowed:
                desired = self._sample_from_relation(rel, parent_value)
                if desired and current_child != desired:
                    row[rel.B] = desired
                    changed = True

        return changed

    def _validate_enum_values(self, row: dict[str, Any]) -> bool:
        """Ensure enum-backed fields always take allowed values."""
        changed = False
        for field in self.allowed_fields:
            prop: PropertySchema | None = self.schema.properties.get(field)
            if not prop or not prop.enum:
                continue
            value = normalize_value(row.get(field))
            if value and value not in {str(x) for x in prop.enum}:
                # Try the most likely value from the learned marginal.
                probs = self.marginals.get(field, {})
                best = ""
                for candidate in probs.keys():
                    if candidate in {str(x) for x in prop.enum}:
                        best = candidate
                        break
                if not best:
                    best = str(prop.enum[0]) if prop.enum else ""
                if best and best != value:
                    row[field] = best
                    changed = True
        return changed
    
    def _apply_cluster_repair(self, row: dict[str, Any]) -> bool:
        changed = False

        for rel in self.relationship_sort_order:
            feature_type = str(rel.evidence.get("feature_type", "")).strip().lower()
            if feature_type != "clustering":
                continue

            parent_value = normalize_value(row.get(rel.A))
            if not parent_value:
                continue

            current_child = normalize_value(row.get(rel.B))
            if current_child and self._cluster_value_in_range(row, rel, current_child):
                continue

            target = self._cluster_target_value(row, rel)
            if target is not None:
                row[rel.B] = float(np.exp(target))
                changed = True

        return changed

    def _repair_row(self, row: dict[str, Any]) -> dict[str, Any]:
        for _ in range(self.max_validation_rounds):
            changed = False
            changed |= self._apply_relation_reconciliation(row)
            changed |= self._validate_enum_values(row)
            changed |= self._apply_cluster_repair(row)
            if not changed:
                break
        return row

    def _sample_field(self, field: str, context: dict[str, Any]) -> str:
        parent_rel = self.parent_for_child.get(field)
        if parent_rel and parent_rel.A in context:
            parent_value = normalize_value(context[parent_rel.A])
            conditional = self.conditional_maps.get((parent_rel.A, field), {})
            if parent_value in conditional:
                probs = conditional[parent_value]
                if parent_rel.classification == "functional":
                    return max(probs.items(), key=lambda kv: kv[1])[0]
                top_p = self.top_p
                temperature = self.temperature_strong if parent_rel.classification == "strong" else self.temperature_conditional
                filtered = _pick_top_k(probs, top_p=top_p, temperature=temperature)
                if filtered:
                    return weighted_choice(filtered, self.rng)

        probs = self.marginals.get(field, {})
        if probs:
            return weighted_choice(probs, self.rng)
        return ""

    # def generate_row(self) -> dict[str, Any]:
    #     row: dict[str, Any] = {}
    #     for field in self.generation_order:
    #         row[field] = self._sample_field(field, row)

    #     # A whole-row repair pass makes the output coherent across the node,
    #     # not only pairwise.
    #     return self._repair_row(row)

    # def generate(self, n: int) -> pd.DataFrame:
    #     rows = [self.generate_row() for _ in range(n)]
    #     return pd.DataFrame(rows, columns=self.generation_order)
    def generate_row(self) -> dict[str, Any] | None:
        row: dict[str, Any] = {}
        for field in self.generation_order:
            row[field] = self._sample_field(field, row)
        row = self._repair_row(row)

        # drop invalid rows here
        if not self._row_is_valid(row):
            return None

        return row


    def generate(self, n: int) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = max(n * 20, 100)

        while len(rows) < n and attempts < max_attempts:
            attempts += 1
            row = self.generate_row()
            if row is not None:
                rows.append(row)

        return pd.DataFrame(rows, columns=self.generation_order)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic sample-node rows from pairwise relationships")
    parser.add_argument("--data", required=False, help="Path to the real flattened sample JSON array")
    parser.add_argument("--schema", required=True, help="Path to the YAML schema for the sample node")
    parser.add_argument("--relationships", required=True, help="Path to pairwise relationships CSV")
    parser.add_argument("--study", required=True, help="Optional study ID for Neo4j extraction")
    parser.add_argument("--env-path", required=False, default=None, help="Optional path to .env file")
    parser.add_argument("--neo4j-limit", type=int, default=None, help="Optional limit for Neo4j rows")
    parser.add_argument("--output", default="outputs/synthetic_rows.csv", help="Output file (.csv or .json)")
    parser.add_argument("--n", type=int, default=50, help="Number of synthetic rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p cutoff for conditional sampling")
    parser.add_argument("--temp-strong", type=float, default=1.0, help="Sampling temperature for strong relations")
    parser.add_argument("--temp-conditional", type=float, default=1.1, help="Sampling temperature for conditional relations")
    parser.add_argument("--include-low-value-fields", action="store_true", help="Keep lower-value fields if present in the data")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    schema = load_node_schema(args.schema)
    rel_df = pd.read_csv(args.relationships)

    real_df = fetch_rows_from_neo4j(
        schema,
        study_id=args.study,
        limit=args.neo4j_limit,
        env_path=args.env_path,
    )

    if args.data:
        real_df = load_json_rows(args.data)

    generator = SyntheticDataGenerator(
        real_rows=real_df,
        relationships=rel_df,
        schema=schema,
        seed=args.seed,
        top_p=args.top_p,
        temperature_strong=args.temp_strong,
        temperature_conditional=args.temp_conditional,
        include_low_value_fields=args.include_low_value_fields,
    )
    synthetic_df = generator.generate(args.n)
    valid_df, invalid_df = generator.validate_rows(synthetic_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(valid_df, output_path)

    # Also save JSON next to CSV for easy inspection.
    json_path = output_path.with_suffix(".json")
    rows_to_json(valid_df.to_dict(orient="records"), json_path)


if __name__ == "__main__":
    main()
