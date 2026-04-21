# features/base.py
from __future__ import annotations

import os
from typing import Any

import pandas as pd

MISSING_STRINGS = {"", "na", "n/a", "none", "null", "nan", "not reported"}
IDENTIFIER_HINTS = ("_id", "_record_id", "uuid", "crdc_id")


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

def is_identifier_like(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(hint) or lowered == hint.strip("_") for hint in IDENTIFIER_HINTS)


def prepare_pair_frame(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    pair = df[[a, b]].copy()
    pair[a] = pair[a].map(normalize_value)
    pair[b] = pair[b].map(normalize_value)
    pair = pair[(pair[a] != "") & (pair[b] != "")]
    return pair.reset_index(drop=True)


class FeatureBase:
    def get_skip_fields(self) -> set[str]:
        raw = os.getenv("SKIP_FIELDS", "")
        return {x.strip().lower() for x in raw.split(",") if x.strip()}

    def should_skip(self, col: str, schema: Any | None = None) -> bool:
        col_norm = col.strip().lower()

        if col_norm in self.get_skip_fields():
            return True

        if is_identifier_like(col_norm):
            return True

        if schema is not None:
            exclude_like = {str(x).strip().lower() for x in getattr(schema, "exclude_like", [])}
            if col_norm in exclude_like:
                return True

            props = getattr(schema, "properties", {})
            if col not in props:
                return True

        return False