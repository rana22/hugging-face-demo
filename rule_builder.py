# build_rules_from_pairs.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


# --- Configuration ---

DEFAULT_OUTPUT = "learned_sample_rules.yaml"

# tune these to your dataset later
STRONG_THRESHOLD = 0.80
FUNCTIONAL_THRESHOLD = 0.93

LABEL_ALIASES = {
    "condition": "condition",
    "conditional": "condition",
    "conditional_distribution": "condition",
    "strong": "strong",
    "strong_relation": "strong",
    "strong_dependency": "strong",
    "functional": "functional",
    "functional_dependency": "functional",
    "constrain": "functional",
}


@dataclass
class RuleRow:
    A: str
    B: str
    relation_label: str
    strength: float
    confidence: float | None
    support: float | None
    predictive_strength: float | None
    determinism: float | None
    stability: float | None
    doc_alignment: float | None


def _as_float(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return default


def _norm_label(label: Any) -> str:
    if label is None:
        return ""
    return LABEL_ALIASES.get(str(label).strip().lower(), str(label).strip().lower())


def _pick_col(df: pd.DataFrame, *candidates: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def load_pairs_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df


def normalize_pairs_df(df: pd.DataFrame) -> list[RuleRow]:
    a_col = _pick_col(df, "A", "a", "source", "from", "parent")
    b_col = _pick_col(df, "B", "b", "target", "to", "child")
    label_col = _pick_col(df, "label", "relation_label", "relation_type", "strength_label", "kind")
    strength_col = _pick_col(df, "strength", "score")
    conf_col = _pick_col(df, "confidence", "conf")
    support_col = _pick_col(df, "support")
    pred_col = _pick_col(df, "predictive_strength", "predictive", "prediction_strength")
    det_col = _pick_col(df, "determinism")
    stab_col = _pick_col(df, "stability")
    doc_col = _pick_col(df, "doc_alignment", "documentation_alignment")

    if not a_col or not b_col or not strength_col:
        raise ValueError(
            "CSV must contain at least A/B and strength columns. "
            f"Found columns: {list(df.columns)}"
        )

    rows: list[RuleRow] = []
    for _, r in df.iterrows():
        label = _norm_label(r[label_col]) if label_col else ""
        rows.append(
            RuleRow(
                A=str(r[a_col]).strip(),
                B=str(r[b_col]).strip(),
                relation_label=label,
                strength=_as_float(r[strength_col], 0.0) or 0.0,
                confidence=_as_float(r[conf_col]) if conf_col else None,
                support=_as_float(r[support_col]) if support_col else None,
                predictive_strength=_as_float(r[pred_col]) if pred_col else None,
                determinism=_as_float(r[det_col]) if det_col else None,
                stability=_as_float(r[stab_col]) if stab_col else None,
                doc_alignment=_as_float(r[doc_col]) if doc_col else None,
            )
        )
    return rows


def choose_best_direction(rows: Iterable[RuleRow]) -> list[RuleRow]:
    """
    If both A->B and B->A exist, keep the stronger one.
    Direction matters for rule generation.
    """
    best: dict[tuple[str, str], RuleRow] = {}
    for row in rows:
        key = (row.A, row.B)
        reverse = (row.B, row.A)

        # keep the stronger direction if both are present
        if reverse in best:
            if row.strength > best[reverse].strength:
                del best[reverse]
                best[key] = row
        elif key not in best:
            best[key] = row
        else:
            if row.strength > best[key].strength:
                best[key] = row

    return list(best.values())


def infer_rule_type(row: RuleRow) -> str:
    """
    Map the pairwise label/score to a YAML rule type.
    """
    if row.relation_label == "functional" or row.strength >= FUNCTIONAL_THRESHOLD:
        return "functional"
    if row.relation_label == "strong" or row.strength >= STRONG_THRESHOLD:
        return "strong"
    if row.relation_label == "condition":
        return "condition"
    return "skip"


def build_rule_entry(row: RuleRow) -> dict[str, Any] | None:
    rule_kind = infer_rule_type(row)
    if rule_kind == "skip":
        return None

    evidence = {
        "strength": round(row.strength, 4),
    }
    if row.confidence is not None:
        evidence["confidence"] = round(row.confidence, 4)
    if row.support is not None:
        evidence["support"] = round(row.support, 4)
    if row.predictive_strength is not None:
        evidence["predictive_strength"] = round(row.predictive_strength, 4)
    if row.determinism is not None:
        evidence["determinism"] = round(row.determinism, 4)
    if row.stability is not None:
        evidence["stability"] = round(row.stability, 4)
    if row.doc_alignment is not None:
        evidence["doc_alignment"] = round(row.doc_alignment, 4)

    # Functional = hard rule
    if rule_kind == "functional":
        return {
            "type": "constrain",
            "A": row.A,
            "B": row.B,
            "rule": "functional_dependency",
            "evidence": evidence,
        }

    # Strong and condition = conditional sampling with different confidence
    sampling = {
        "top_p": 0.95,
        "temperature": 1.0 if rule_kind == "strong" else 1.1,
    }

    min_strength = {
        "strength": round(row.strength, 4),
    }
    if row.confidence is not None:
        min_strength["confidence"] = round(row.confidence, 4)

    return {
        "type": "conditional_distribution",
        "A": row.A,
        "B": row.B,
        "min_strength": min_strength,
        "sampling": sampling,
        "evidence": evidence,
    }


def build_columns_section(rule_rows: list[RuleRow], preferred_root: str | None = None) -> list[dict[str, str]]:
    """
    Root = preferred_root if supplied, else a field that only appears as A
    in the retained rules; everything else is dependent.
    """
    if not rule_rows:
        return []

    all_a = {r.A for r in rule_rows}
    all_b = {r.B for r in rule_rows}

    root_candidates = []
    if preferred_root and preferred_root in (all_a | all_b):
        root_candidates = [preferred_root]
    else:
        root_candidates = sorted(all_a - all_b)

    used = {r.A for r in rule_rows} | {r.B for r in rule_rows}
    cols = []

    if root_candidates:
        cols.append({"name": root_candidates[0], "role": "root"})
        used.discard(root_candidates[0])

    for name in sorted(used):
        cols.append({"name": name, "role": "dependent"})

    return cols


def build_yaml_manifest(
    df: pd.DataFrame,
    preferred_root: str | None = None,
    node_name: str = "sample",
) -> dict[str, Any]:
    rows = normalize_pairs_df(df)
    rows = choose_best_direction(rows)

    dependencies = []
    for row in rows:
        entry = build_rule_entry(row)
        if entry is not None:
            dependencies.append(entry)

    columns = build_columns_section(rows, preferred_root=preferred_root)

    return {
        "version": 1,
        "node": node_name,
        "columns": columns,
        "dependencies": dependencies,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build YAML-style rules from pairwise relationships.")
    parser.add_argument("csv_path", type=Path, help="Path to pairwise_relationships.csv")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--root", type=str, default=None, help="Preferred root field, e.g. specific_sample_pathology")
    parser.add_argument("--node", type=str, default="sample", help="Node name for manifest")
    args = parser.parse_args()

    df = load_pairs_csv(args.csv_path)
    manifest = build_yaml_manifest(df, preferred_root=args.root, node_name=args.node)

    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()