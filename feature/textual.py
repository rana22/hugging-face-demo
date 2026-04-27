# from __future__ import annotations

# from dataclasses import dataclass, field
# from itertools import combinations
# from typing import Any, Callable

# import json
# import pandas as pd
# import numpy as np

# from schema import NodeSchema
# from prompt import build_textual_relation_messages

# MISSING_STRINGS = {"", "na", "n/a", "none", "null", "nan", "not reported"}
# TEXTUAL_RELATION_LABELS = {"matching", "related", "complementary", "unrelated"}

# def is_missing(value: Any) -> bool:
#     if value is None:
#         return True
#     try:
#         if pd.isna(value):
#             return True
#     except Exception:
#         pass
#     return isinstance(value, str) and value.strip().lower() in MISSING_STRINGS


# def normalize_value(value: Any) -> str:
#     if is_missing(value):
#         return ""
#     return str(value).strip()


# def _compact_json(obj: Any) -> str:
#     return json.dumps(obj, ensure_ascii=False, default=str, indent=2)


# def _sample_non_empty_values(df: pd.DataFrame, col: str, n: int = 5) -> list[str]:
#     if df is None or df.empty or col not in df.columns:
#         return []

#     vals = []
#     for v in df[col].dropna().astype(str).tolist():
#         v = v.strip()
#         if v and v.lower() not in {"na", "n/a", "null", "none", "nan"}:
#             vals.append(v)
#         if len(vals) >= n:
#             break
#     return vals

# def _extract_property_schema_text(node_schema: NodeSchema, prop_name: str) -> str:
#     """
#     Tries to pull human-readable text for a property from NodeSchema.
#     Adjust attribute names here if your schema object uses different field names.
#     """
#     prop_obj = None
#     props = getattr(node_schema, "properties", None) or getattr(node_schema, "Props", None) or []
#     if isinstance(props, dict):
#         prop_obj = props.get(prop_name)
#     else:
#         for p in props:
#             pname = getattr(p, "name", None) or getattr(p, "Name", None)
#             if pname == prop_name:
#                 prop_obj = p
#                 break

#     node_name = getattr(node_schema, "name", "") or getattr(node_schema, "Name", "")
#     node_desc = getattr(node_schema, "description", "") or getattr(node_schema, "Desc", "")

#     prop_desc = ""
#     prop_type = ""
#     prop_required = ""
#     prop_enum = ""

#     if prop_obj is not None:
#         prop_desc = getattr(prop_obj, "description", "") or getattr(prop_obj, "Desc", "") or ""
#         prop_type = getattr(prop_obj, "type", "") or getattr(prop_obj, "Type", "") or ""
#         prop_required = getattr(prop_obj, "required", "") or getattr(prop_obj, "Req", "") or ""
#         prop_enum = getattr(prop_obj, "enum", None) or getattr(prop_obj, "Enum", None) or []
    
#     return (
#         f"Node name: {node_name}\n"
#         f"Node description: {node_desc}\n"
#         f"Property name: {prop_name}\n"
#         f"Property description: {prop_desc}\n"
#         f"Property type: {prop_type}\n"
#         f"Property required: {prop_required}\n"
#         f"Property enum: {prop_enum}"
#     ).strip()

# def _sample_property_values(
#     df: pd.DataFrame,
#     col: str,
#     n: int = 5,
#     max_chars: int = 200,
# ) -> list[str]:
#     if df is None or df.empty or col not in df.columns:
#         return []

#     values: list[str] = []
#     seen: set[str] = set()

#     for v in df[col].tolist():
#         text = normalize_value(v)
#         if not text:
#             continue

#         if len(text) > max_chars:
#             text = text[:max_chars].rstrip() + "..."

#         if text in seen:
#             continue

#         seen.add(text)
#         values.append(text)

#         if len(values) >= n:
#             break

#     return values


# def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
#     if df is None or df.empty:
#         return pd.DataFrame()

#     rows: list[dict[str, Any]] = []

#     for a, b in self._candidate_pairs(df):
#         key = self._cache_key(a, b)

#         if key in self.cache:
#             result = self.cache[key]
#         else:
#             try:
#                 prop_a_values = _sample_property_values(df, a, self.max_value_samples)
#                 prop_b_values = _sample_property_values(df, b, self.max_value_samples)

#                 prompt_messages = build_textual_relation_messages(
#                     self.node_schema,
#                     a,
#                     b,
#                     prop_a_values,
#                     prop_b_values,
#                 )

#                 result = self.relation_model(prompt_messages)

#                 if not isinstance(result, dict):
#                     raise ValueError("Model did not return a dict")

#                 self.cache[key] = result

#             except Exception as e:
#                 result = {
#                     "label": "error",
#                     "reason": f"Model failure: {str(e)}",
#                     "raw": "",
#                 }

#         label = (result.get("label") or "unrelated").lower()
#         reason = result.get("reason", "")
#         raw = result.get("raw", "")

#         rows.append(
#             {
#                 "A": a,
#                 "B": b,
#                 "feature_type": "textual",
#                 "label": label,
#                 "classification": self.classify_strength(label),
#                 "reasoning": reason,
#                 "model_raw": raw,
#                 "strength": (
#                     1.0 if label == "matching"
#                     else 0.75 if label == "related"
#                     else 0.5 if label == "complementary"
#                     else 0.0
#                 ),
#                 "node_name": getattr(self.node_schema, "name", "") or getattr(self.node_schema, "Name", ""),
#                 "node_description": getattr(self.node_schema, "description", "") or getattr(self.node_schema, "Desc", ""),
#                 "property_a_text": _extract_property_schema_text(self.node_schema, a),
#                 "property_b_text": _extract_property_schema_text(self.node_schema, b),
#                 "sample_a": _sample_property_values(df, a, self.max_value_samples),
#                 "sample_b": _sample_property_values(df, b, self.max_value_samples),
#             }
#         )

#     out = pd.DataFrame(rows)

#     if not out.empty:
#         out = out[out["label"].isin({"matching", "related", "complementary"})].reset_index(drop=True)

#     return out

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import json
import pandas as pd
import numpy as np

from schema import NodeSchema
from feature.prompt import build_textual_relation_messages

MISSING_STRINGS = {"", "na", "n/a", "none", "null", "nan", "not reported"}
TEXTUAL_RELATION_LABELS = {"matching", "related", "complementary", "unrelated"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and value.strip().lower() in MISSING_STRINGS


def normalize_value(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def _sample_property_values(
    df: pd.DataFrame,
    col: str,
    n: int = 5,
    max_chars: int = 200,
) -> list[str]:
    if df is None or df.empty or col not in df.columns:
        return []

    values: list[str] = []
    seen: set[str] = set()

    for v in df[col].tolist():
        text = normalize_value(v)
        if not text:
            continue

        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."

        if text in seen:
            continue

        seen.add(text)
        values.append(text)

        if len(values) >= n:
            break

    return values


def _extract_property_schema_text(node_schema: NodeSchema, prop_name: str) -> str:
    prop_obj = None
    props = getattr(node_schema, "properties", None) or getattr(node_schema, "Props", None) or []
    if isinstance(props, dict):
        prop_obj = props.get(prop_name)
    else:
        for p in props:
            pname = getattr(p, "name", None) or getattr(p, "Name", None)
            if pname == prop_name:
                prop_obj = p
                break

    node_name = getattr(node_schema, "name", "") or getattr(node_schema, "Name", "")
    node_desc = getattr(node_schema, "description", "") or getattr(node_schema, "Desc", "")

    prop_desc = ""
    prop_type = ""
    prop_required = ""
    prop_enum = ""

    if prop_obj is not None:
        prop_desc = getattr(prop_obj, "description", "") or getattr(prop_obj, "Desc", "") or ""
        prop_type = getattr(prop_obj, "type", "") or getattr(prop_obj, "Type", "") or ""
        prop_required = getattr(prop_obj, "required", "") or getattr(prop_obj, "Req", "") or ""
        prop_enum = getattr(prop_obj, "enum", None) or getattr(prop_obj, "Enum", None) or []

    return (
        f"Node name: {node_name}\n"
        f"Node description: {node_desc}\n"
        f"Property name: {prop_name}\n"
        f"Property description: {prop_desc}\n"
        f"Property type: {prop_type}\n"
        f"Property required: {prop_required}\n"
        f"Property enum: {prop_enum}"
    ).strip()


@dataclass
class TextualFeatureAnalyzer:
    node_schema: NodeSchema
    relation_model: Callable[[list[dict[str, str]]], dict[str, Any]]
    max_value_samples: int = 5
    cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def classify_strength(label: str) -> str:
        label = (label or "").lower()
        if label == "matching":
            return "functional"
        if label == "related":
            return "strong"
        if label == "complementary":
            return "conditional"
        return "independent"

    def _cache_key(self, a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b)))

    def _candidate_pairs(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        schema_props = getattr(self.node_schema, "properties", None) or getattr(self.node_schema, "Props", None) or []
        prop_names: list[str] = []

        if isinstance(schema_props, dict):
            prop_names = [str(k) for k in schema_props.keys()]
        else:
            for p in schema_props:
                pname = getattr(p, "name", None) or getattr(p, "Name", None)
                if pname:
                    prop_names.append(str(pname))

        prop_names = [p for p in prop_names if p in df.columns]
        if not prop_names:
            prop_names = [c for c in df.columns if c not in {"type", "type_"}]

        usable = []
        for p in prop_names:
            s = df[p].dropna().astype(str)
            non_empty = s[s.str.strip() != ""]
            if len(non_empty) >= 2:
                usable.append(p)

        return list(combinations(usable, 2))

    # def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
    #     if df is None or df.empty:
    #         return pd.DataFrame()

    #     rows: list[dict[str, Any]] = []

    #     for a, b in self._candidate_pairs(df):
    #         key = self._cache_key(a, b)

    #         if key in self.cache:
    #             result = self.cache[key]
    #         else:
    #             try:
    #                 prop_a_values = _sample_property_values(df, a, self.max_value_samples)
    #                 prop_b_values = _sample_property_values(df, b, self.max_value_samples)

    #                 prompt_messages = build_textual_relation_messages(
    #                     self.node_schema,
    #                     a,
    #                     b,
    #                     prop_a_values,
    #                     prop_b_values,
    #                 )

    #                 result = self.relation_model(prompt_messages)

    #                 if not isinstance(result, dict):
    #                     raise ValueError("Model did not return a dict")

    #                 self.cache[key] = result

    #             except Exception as e:
    #                 result = {
    #                     "label": "error",
    #                     "reason": f"Model failure: {str(e)}",
    #                     "raw": "",
    #                 }

    #         label = (result.get("label") or "unrelated").lower()
    #         reason = result.get("reason", "")
    #         raw = result.get("raw", "")

    #         rows.append(
    #             {
    #                 "A": a,
    #                 "B": b,
    #                 "feature_type": "textual",
    #                 "label": label,
    #                 "classification": self.classify_strength(label),
    #                 "reasoning": reason,
    #                 "model_raw": raw,
    #                 "strength": (
    #                     1.0 if label == "matching"
    #                     else 0.75 if label == "related"
    #                     else 0.5 if label == "complementary"
    #                     else 0.0
    #                 ),
    #                 "node_name": getattr(self.node_schema, "name", "") or getattr(self.node_schema, "Name", ""),
    #                 "node_description": getattr(self.node_schema, "description", "") or getattr(self.node_schema, "Desc", ""),
    #                 "property_a_text": _extract_property_schema_text(self.node_schema, a),
    #                 "property_b_text": _extract_property_schema_text(self.node_schema, b),
    #                 "sample_a": _sample_property_values(df, a, self.max_value_samples),
    #                 "sample_b": _sample_property_values(df, b, self.max_value_samples),
    #             }
    #         )

    #     out = pd.DataFrame(rows)

    #     if not out.empty:
    #         out = out[out["label"].isin({"matching", "related", "complementary"})].reset_index(drop=True)

    #     return out
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []

        candidate_pairs = self._candidate_pairs(df)
        if not candidate_pairs:
            return pd.DataFrame()

        a, b = candidate_pairs[0]   # only the first pair

        key = self._cache_key(a, b)

        if key in self.cache:
            result = self.cache[key]
        else:
            try:
                prop_a_values = _sample_property_values(df, a, self.max_value_samples)
                prop_b_values = _sample_property_values(df, b, self.max_value_samples)

                prompt_messages = build_textual_relation_messages(
                    self.node_schema,
                    a,
                    b,
                    prop_a_values,
                    prop_b_values,
                )

                result = self.relation_model(prompt_messages)

                if not isinstance(result, dict):
                    raise ValueError("Model did not return a dict")

                self.cache[key] = result

            except Exception as e:
                result = {
                    "label": "error",
                    "reason": f"Model failure: {str(e)}",
                    "raw": "",
                }

        label = (result.get("label") or "unrelated").lower()
        reason = result.get("reason", "")
        raw = result.get("raw", "")

        rows.append(
            {
                "A": a,
                "B": b,
                "feature_type": "textual",
                "label": label,
                "classification": self.classify_strength(label),
                "reasoning": reason,
                "model_raw": raw,
                "strength": (
                    1.0 if label == "matching"
                    else 0.75 if label == "related"
                    else 0.5 if label == "complementary"
                    else 0.0
                ),
                "node_name": getattr(self.node_schema, "name", "") or getattr(self.node_schema, "Name", ""),
                "node_description": getattr(self.node_schema, "description", "") or getattr(self.node_schema, "Desc", ""),
                "property_a_text": _extract_property_schema_text(self.node_schema, a),
                "property_b_text": _extract_property_schema_text(self.node_schema, b),
                "sample_a": _sample_property_values(df, a, self.max_value_samples),
                "sample_b": _sample_property_values(df, b, self.max_value_samples),
            }
        )

        return pd.DataFrame(rows)