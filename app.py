from __future__ import annotations

import json
import os, re 
import tempfile
from pathlib import Path
from typing import Any
import requests

import gradio as gr
import pandas as pd
import yaml
import html as html_lib
import tempfile

# These imports assume your package is included in the Space repo.
# If some functions are named differently in your current branch,
# adjust the import lines only.
from generator import SyntheticDataGenerator
from neo4j_loader import fetch_rows_from_neo4j
from reporting import write_markdown_report
from schema import load_node_schema, PropertySchema, load_schemas_from_models, node_schemas_to_markdown
from schema_builder import build_relation_schemas
from viz import generate_visual_report
from evaluator import PairwiseRelationshipEvaluator
from cross_evaluator import CrossNodeRelationshipEvaluator, find_selected_path, find_selected_edges
# from feature.model_wrapper import relation_model_wrapper
# from feature.textual import TextualFeatureAnalyzer

BASE_DIR = Path(__file__).resolve().parent
TABLE_JS = (BASE_DIR / "static" / "table.js").read_text(encoding="utf-8")
CUSTOM_CSS = (BASE_DIR / "static" / "styles.css").read_text(encoding="utf-8")

AG_GRID_HEAD = f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-quartz.css">
<script src="https://cdn.jsdelivr.net/npm/ag-grid-community/dist/ag-grid-community.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script>
  {TABLE_JS}
</script>
"""

DISPLAY_COLUMNS = {
    "categorical": [
        "A",
        "B",
        "support",
        "predictive_strength",
        "determinism",
        "stability",
        "doc_alignment",
        "strength",
        "classification",
        "evidence",
        "a_to_b_mapping",
    ],
    "substring": [
        "A",
        "B",
        "support",
        "prefix_match",
        "suffix_match",
        "substring_match",
        "doc_alignment",
        "strength",
        "classification",
        "evidence",
    ],
    "default": [
        "A",
        "B",
        "support",
        "strength",
        "classification",
        "evidence",
    ],
}

def _json_safe(obj) -> str:
    return json.dumps(obj, default=str).replace("</", "<\\/")

def df_to_html(curr_node_relations: pd.DataFrame):

    if curr_node_relations is None or curr_node_relations.empty:
        row_data = [["Alice", 25, "true"], ["Bob", 30, "false"]]
        col_defs = [{"field":"Name"}, {"field":"Age"}, {"field":"Active"}]
    else:
        display_df = curr_node_relations.copy()

        # Optional: hide or simplify huge text columns
        for col in ["evidence", "a_to_b_mapping"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(str)

        display_df = display_df.reset_index(drop=False)
        display_df.columns = [str(c) for c in display_df.columns]

        curr_row_data = display_df.fillna("").to_dict(orient="records")
        col_defs = [{"field": str(col)} for col in display_df.columns]
        
        row_data =[["Alice", 24, "true"]]
        return f"""
            <div id="table-root"
                style="height: 300px; width: 100%;"
                data-row-data='{_json_safe(curr_row_data)}'
                data-col-defs='{_json_safe(col_defs)}'>
                {_json_safe(col_defs)}
                </div>
            """

    data = _json_safe(row_data)
    cols = _json_safe(col_defs)

    return f"""
    <div id="table-root"
        style="height: 300px; width: 100%;"
        data-row-data='{data}'
        data-col-defs='{cols}'>
    </div>
    """

def df_to_html2(curr_node_relations: pd.DataFrame):
    if curr_node_relations is None or curr_node_relations.empty:
        display_df = pd.DataFrame(
            []
        )
    else:
        display_df = curr_node_relations.copy()

        for col in ["evidence", "a_to_b_mapping"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(str)

        display_df = display_df.reset_index(drop=False)
        display_df.columns = [str(c) for c in display_df.columns]

    row_json = json.dumps(display_df.fillna("").to_dict(orient="records"), ensure_ascii=False)
    col_json = json.dumps([{"field": str(c)} for c in display_df.columns], ensure_ascii=False)

    return f"""
    <input
        id="table-search"
        type="text"
        placeholder="Search table..."
        style="margin-bottom: 8px; width: 300px; padding: 6px;"
    />
    <div id="table-root" class="ag-theme-quartz" style="height: 700px; width: 100%;"></div>
    <script type="application/json" id="table-root-data">{row_json}</script>
    <script type="application/json" id="table-root-cols">{col_json}</script>
    """

def df_to_simple_html(df):
    if df is None or df.empty:
        return "<p>No data</p>"

    return df.head(100).to_html(index=False, escape=True)

def _parse_env_text(env_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (env_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def toggle_schema_fields(mode):
    return gr.update(visible=(mode == "upload"))

def _build_status_md(node_name: str, study_id: str | None, results: pd.DataFrame) -> str:
    lines = [f"# Analysis for `{node_name}`"]
    lines.append(f"- Study: `{study_id or 'all'}`")
    lines.append(f"- Pairs analyzed: `{len(results)}`")
    if not results.empty and "classification" in results.columns:
        cls_counts = results["classification"].fillna("unknown").value_counts().to_dict()
        lines.append("\n## Relationship counts")
        for k, v in cls_counts.items():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)

def load_env_to_text(env_file):
    if env_file is None:
        return ""

    try:
        if isinstance(env_file, (str, Path)):
            path = Path(env_file)
        else:
            path = Path(env_file.name)

        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"# Failed to read file: {e}"

# def _read_single_file(file_path: str) -> pd.DataFrame:
#     _, ext = os.path.splitext(file_path)
#     ext = ext.lower()

#     if ext in [".xlsx", ".xls"]:
#         df = pd.read_excel(file_path)
#     elif ext == ".json":
#         # Supports JSON array or JSON object
#         with open(file_path, "r", encoding="utf-8-sig") as f:
#             data = json.load(f)

#         if isinstance(data, list):
#             df = pd.DataFrame(data)
#         elif isinstance(data, dict):
#             # If the JSON is a single object, wrap it as one row
#             df = pd.DataFrame([data])
#         else:
#             raise ValueError(f"Unsupported JSON structure in {file_path}")
#     else:
#         raise ValueError(f"Unsupported file type: {ext}")

#     if "type" not in df.columns and "type_" not in df.columns:
#         raise ValueError(f"Missing required column/key 'type' in {file_path}. Type refers to node (sample, study, file, case..)")

#     return df

def _read_single_file(file_path: str) -> dict[str, pd.DataFrame]:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            raise ValueError(f"Unsupported JSON structure in {file_path}")

        df = pd.DataFrame(data)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # normalize type column
    if "type_" in df.columns:
        df.rename(columns={"type_": "type"}, inplace=True)

    if "type" not in df.columns:
        raise ValueError(f"Missing 'type' field in {file_path}")

    # ✅ split by node type
    node_dfs = {}

    for node_type, group in df.groupby("type"):
        # drop irrelevant columns (all-null columns)
        clean_df = group.drop(columns=["type"]).dropna(axis=1, how="all")

        node_dfs[node_type] = clean_df.reset_index(drop=True)

    return node_dfs

def get_excel_or_json_data(files):
    try:
        if not files:
            return {}, gr.update(choices=[], value=None), pd.DataFrame(), ""

        if not isinstance(files, list):
            files = [files]

        grouped_data: dict[str, pd.DataFrame] = {}

        # for file_path in files:
        #     df = _read_single_file(file_path)
        #     if "type" in df.columns:
        #         df["_node_type"] = df["type"]
        #     elif "type_" in df.columns:
        #         df["_node_type"] = df["type_"]

        #     for node_name, node_df in df.groupby("_node_type", dropna=False):
        #         node_key = str(node_name)

        #         if node_key in grouped_data:
        #             grouped_data[node_key] = pd.concat(
        #                 [grouped_data[node_key], node_df.copy()],
        #                 ignore_index=True
        #             )
        #         else:
        #             grouped_data[node_key] = node_df.copy()
        for file_path in files:
            node_dfs = _read_single_file(file_path)  # now returns dict

            for node_name, node_df in node_dfs.items():
                node_key = str(node_name)

                if node_key in grouped_data:
                    grouped_data[node_key] = pd.concat(
                        [grouped_data[node_key], node_df.copy()],
                        ignore_index=True
                    )
                else:
                    grouped_data[node_key] = node_df.copy()

        node_list = sorted(grouped_data.keys())
        first_node = node_list[0] if node_list else None
        preview_df = grouped_data[first_node] if first_node else pd.DataFrame()

        return grouped_data, gr.update(choices=node_list, value=first_node), preview_df, node_list, ""

    except Exception as e:
        return {}, gr.update(choices=[], value=None), pd.DataFrame(), [], f"<div style='color:red;font-weight:700'>Error: {e}</div>"

def display_selected_node(node, grouped_data):
    if not node or node not in grouped_data:
        return pd.DataFrame()
    return grouped_data[node]

def _safe_excel_sheet_name(name: str, used: set[str]) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {" ", "_", "-"} else "_" for ch in name).strip()
    cleaned = cleaned.replace(" ", "_")[:31] or "Sheet"
    candidate = cleaned
    i = 2
    while candidate in used:
        suffix = f"_{i}"
        candidate = f"{cleaned[:31 - len(suffix)]}{suffix}"
        i += 1
    used.add(candidate)
    return candidate

def run_analysis(schema_state, data_state, selected_node):
    try:
        if not schema_state:
            raise gr.Error("Load schema first.")

        if not selected_node:
            raise gr.Error("Select a node.")

        schema = next((s for s in schema_state if s.name == selected_node), None)
        if schema is None:
            raise gr.Error(f"Schema for node '{selected_node}' was not found.")

        df = data_state.get(selected_node)
        if df is None or df.empty:
            return {}, {}, f"No data available for node `{selected_node}`.", pd.DataFrame(),

        engine = PairwiseRelationshipEvaluator(schema)
        results = engine.evaluate_all_pairs(df)

        summary_md = _build_status_md(schema.name, None, results)

        features_dfs = dict(tuple(results.groupby('feature_type')))
        # Optional: if you want a markdown report table in the UI
        # report_html = df_to_html2(results)

        results_by_node = {selected_node: results.copy()}
        relationship_by_node = {selected_node: results.copy()}

        return results_by_node, relationship_by_node, summary_md, features_dfs

    except Exception as e:
        return {}, {}, f"<div style='color:red;font-weight:700'>Error: {e}</div>", pd.DataFrame(),


def extract_node_branches(rel_df: pd.DataFrame, selected_nodes: list[str]) -> pd.DataFrame:
    if rel_df.empty:
        return pd.DataFrame()

    return rel_df[
        rel_df["parent_node"].isin(selected_nodes) &
        rel_df["child_node"].isin(selected_nodes)
    ].reset_index(drop=True)

def extract_node_branches(rel_df: pd.DataFrame, selected_nodes: list[str]) -> pd.DataFrame:
    if rel_df.empty:
        return pd.DataFrame()

    return rel_df[
        rel_df["parent_node"].isin(selected_nodes) &
        rel_df["child_node"].isin(selected_nodes)
    ].reset_index(drop=True)

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def run_cross_analysis(schema_state, node_data_state, node_tree, selected_nodes):
    try:
        if not schema_state:
            raise gr.Error("Load schema first.")

        if not selected_nodes or len(selected_nodes) < 2:
            raise gr.Error("Select at least two nodes for cross-node analysis.")

        if len(node_data_state) < 2:
            raise gr.Error("Not enough valid node data for cross analysis.")

        # hard cap: avoid huge selections
        selected_nodes = list(dict.fromkeys(selected_nodes))[:8]

        edges = find_selected_edges(node_tree, selected_nodes)

        if not edges:
            return {}, {}, "No edges found for selected nodes.", {}, pd.DataFrame(), ""

        cross_eval = CrossNodeRelationshipEvaluator(node_schemas=schema_state)

        all_results = []
        batch_size = 5

        for edge_batch in chunked(edges, batch_size):
            batch_df = cross_eval.analyze(
                node_dfs=node_data_state,
                edges=edge_batch
            )
            if batch_df is not None and not batch_df.empty:
                all_results.append(batch_df)

        if not all_results:
            return {}, {}, "No cross-node relationships detected.", {}, pd.DataFrame(), ""

        cross_results = pd.concat(all_results, ignore_index=True)

        cross_results = cross_results.sort_values(
            "strength", ascending=False
        ).reset_index(drop=True)

        html_table = _build_sortable_table(
            cross_results,
            "cross_node_analysis_html_table",
            "fuzzy search"
        )

        summary_md = _build_status_md("Cross Node", None, cross_results)

        if "feature_type" in cross_results.columns:
            features_dfs = dict(tuple(cross_results.groupby("feature_type")))
        else:
            features_dfs = {}

        results_by_node = {"cross_node": cross_results.copy()}
        relationship_by_node = {"cross_node": cross_results.copy()}

        return (
            results_by_node,
            relationship_by_node,
            summary_md,
            features_dfs,
            cross_results,
            html_table
        )

    except Exception as e:
        print(f"[CROSS ANALYSIS ERROR] {e}")
        return {}, {}, f"<div style='color:red;font-weight:700'>Error: {e}</div>", {}, pd.DataFrame(), ""

# def run_cross_analysis(
#     schema_state, 
#     node_data_state, 
#     node_tree,
#     selected_nodes
# ):
#     try:
#         if not schema_state:
#             raise gr.Error("Load schema first.")

#         if not selected_nodes or len(selected_nodes) < 2:
#             raise gr.Error("Select at least two nodes for cross-node analysis.")

#         if len(node_data_state) < 2:
#             raise gr.Error("Not enough valid node data for cross analysis.")

#         print("CrossNodeRelationshipEvaluator")
#         # --------------------------------------
#         # Run evaluator (node_tree-aware)
#         # --------------------------------------
#         print("cross_results")
#         print(selected_nodes)

#         # print(node_tree)
#         edges = find_selected_edges(node_tree, selected_nodes)
#         print(edges)
#         # get the node tree with the selcted nodes

#         cross_eval = CrossNodeRelationshipEvaluator(
#             node_schemas=schema_state
#         )
#         # cross_results = cross_eval.analyze_edges(
#         #     node_dfs=node_data_state,
#         #     edges=edges
#         # )

#         cross_results = cross_eval.analyze(
#             node_dfs=node_data_state,
#             edges=edges
#         )
#         print(cross_results)
#         # print(node_dfs)

#         if cross_results.empty:
#             return {}, {}, "No cross-node relationships detected.", {}, pd.DataFrame()

#         # --------------------------------------
#         # Sort results
#         # --------------------------------------
#         cross_results = cross_results.sort_values(
#             "strength", ascending=False
#         ).reset_index(drop=True)

#         html_table = _build_sortable_table(
#             cross_results,
#             "cross_node_analysis_html_table",
#             "fuzzy search"
#         )

#         # --------------------------------------
#         # Summary
#         # --------------------------------------
#         summary_md = _build_status_md("Cross Node", None, cross_results)

#         # --------------------------------------
#         # Feature grouping
#         # --------------------------------------
#         if "feature_type" in cross_results.columns:
#             features_dfs = dict(tuple(cross_results.groupby("feature_type")))
#         else:
#             features_dfs = {}

#         results_by_node = {"cross_node": cross_results.copy()}
#         relationship_by_node = {"cross_node": cross_results.copy()}

#         return (
#             results_by_node,
#             relationship_by_node,
#             summary_md,
#             features_dfs,
#             cross_results,
#             html_table
#         )

#     except Exception as e:
#         print(f"[CROSS ANALYSIS ERROR] {e}")
#         return {}, {}, f"<div style='color:red;font-weight:700'>Error: {e}</div>", {}, pd.DataFrame(), ""

def render_generated_tables(analysis_df: pd.DataFrame, valid_df: pd.DataFrame, invalid_df: pd.DataFrame) -> tuple[str, str, str]:
    analysis_html = (
         _build_sortable_table(format_display_df(analysis_df), "property-relation-table", "Analyze property (all features)")
        if analysis_df is not None and not analysis_df.empty
        else "<p>No Analysis data</p>"
    )

    valid_html = (
        _build_sortable_table(format_display_df(valid_df), "generated-table", "Generated Data")
        if valid_df is not None and not valid_df.empty
        else "<p>No generated data</p>"
    )

    invalid_html = (
        _build_sortable_table(format_display_df(invalid_df), "invalid-table", "Invalid Data")
        if invalid_df is not None and not invalid_df.empty
        else "<p>No invalid rows</p>"
    )

    return analysis_html, valid_html, invalid_html, ""

def show_node_sumamry_tables(
    selected_node: str,
    results_state: dict,
    generated_state: dict,
    summary_state: dict,
):
    results_df = results_state.get(selected_node)
    generated_df = generated_state.get(selected_node)
    summary_text = summary_state.get(selected_node, "No summary available.")

    if results_df is None:
        results_df = pd.DataFrame()

    if generated_df is None:
        generated_df = pd.DataFrame()


    return summary_text, generated_df, df_to_html2(results_df)

def filter_df(df, query):
    if df is None or query is None or query.strip() == "":
        return df

    query = query.lower()

    mask = df.apply(
        lambda row: row.astype(str).str.lower().str.contains(query).any(),
        axis=1
    )

    return df[mask]

def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def truncate_cell(x):
        if isinstance(x, (list, dict)):
            x = json.dumps(x, default=str)
        x = "" if x is None else str(x)
        return html_lib.escape(x, quote=False)

    if "evidence" in out.columns:
        out["evidence"] = out["evidence"].apply(
            lambda x: json.dumps(x, indent=2, default=str) if isinstance(x, (list, dict)) else x
        ).apply(truncate_cell)

    if "a_to_b_mapping" in out.columns and not out["a_to_b_mapping"].astype(str).eq("").all():
        out["a_to_b_mapping"] = out["a_to_b_mapping"].astype(str).apply(truncate_cell)
    return out

# load schema
def load_and_display_schema(
    env_text,
    node_list
):
    print("load schema")
    env_values: dict[str, str] = {}
    env_values.update(_parse_env_text(env_text))
    for k, v in env_values.items():
        os.environ[k] = v
    try:
        # env var 
        node_url = os.getenv("NODE_MODEL_URL", "")
        prop_url = os.getenv("PROP_MODEL_URL", "")

        if not node_url or not prop_url:
            raise gr.Error("Set NODE_MODEL_URL and PROP_MODEL_URL in the environment.")
        # print(node_url)
        # print(prop_url)
        node_model = yaml.safe_load(requests.get(node_url, timeout=30).text)
        prop_model = yaml.safe_load(requests.get(prop_url, timeout=30).text)
        nodes = node_model.get("Nodes", {}) or node_model.get("nodes", {}) or {}
        props = prop_model.get("PropDefinitions", {}) or prop_model.get("properties", {}) or {}
        
        whole_schema, error = load_schemas_from_models(nodes, props)
        if error:
            return [], error
        md = node_schemas_to_markdown(whole_schema)
        return md, whole_schema, ""

    except Exception as e:
        return "", [], f"error while loading schema - app\n {str(e)}"

def generate_data(
    schema_state,
    data_state,
    selected_node,
    num_rows
):
    try:
        if not schema_state:
            raise gr.Error("Load schema first.")

        if not selected_node:
            raise gr.Error("Select a node.")

        # get schema
        schema = next((s for s in schema_state if s.name == selected_node), None)
        if schema is None:
            raise gr.Error(f"Schema not found for node '{selected_node}'.")

        # get real data
        df = data_state.get(selected_node)
        print(f"selected node {selected_node}, data length {len(df)}")
        if df is None or df.empty:
            raise gr.Error(f"No data found for node '{selected_node}'.")

        # run relationship analysis first
        engine = PairwiseRelationshipEvaluator(schema)
        results = engine.evaluate_all_pairs(df)

        if results.empty:
            raise gr.Error("No relationships found. Cannot generate synthetic data.")

        # generate synthetic data
        gen = SyntheticDataGenerator(
            real_rows=df,
            relationships=results,
            schema=schema,
        )
        synth_df = gen.generate(int(num_rows))
        valid_df, invalid_df = gen.validate_rows(synth_df)
        print(f"synthe sis {synth_df.shape[0]}, analysis {results.shape[0]}")
        
        return render_generated_tables(results, valid_df, invalid_df)

    except Exception as e:
        return [], [], [], f"error while generating data - app\n {str(e)}"


def get_display_df(df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
    cols = DISPLAY_COLUMNS.get(feature_type, DISPLAY_COLUMNS["default"])
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()

# def _cell_html(value) -> str:
#     if value is None:
#         text = ""
#     elif isinstance(value, float) and pd.isna(value):
#         text = ""
#     elif isinstance(value, (list, dict)):
#         text = json.dumps(value, ensure_ascii=False, default=str)
#     else:
#         text = str(value)

#     escaped = html_lib.escape(text)

#     if len(text) <= 120:
#         return escaped

#     return f"""
#     <div class="expandable-cell collapsed" onclick="toggleExpand(this)" title="Click to expand">
#         {escaped}
#     </div>
#     """

def _cell_html(value, max_len: int = 120) -> str:
    if value is None:
        text = ""
    elif isinstance(value, float) and pd.isna(value):
        text = ""
    elif isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)

    escaped_full = html_lib.escape(text)

    if len(text) <= max_len:
        return escaped_full

    escaped_short = html_lib.escape(text[:max_len] + "…")

    return (
        f'<span class="expandable-cell" '
        f'data-full-value="{escaped_full}" '
        f'title="Click to expand">'
        f"{escaped_short}"
        f"</span>"
    )

def _col_to_class(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "-", col)  # replace non-alphanumeric with -
    return f"col-{col}"

def _build_sortable_table(df: pd.DataFrame, table_id: str, title: str) -> str:
    if df is None or df.empty:
        return f"""
        <div class="mb-4">
            <h4 class="text-primary">{html_lib.escape(title)}</h4>
            <p>No data</p>
        </div>
        """

    df = df.round(3)
    headers = []
    for idx, col in enumerate(df.columns):
        col_class = _col_to_class(col)
        headers.append(
            f'<th  class="{col_class}" onclick="sortHtmlTable(\'{table_id}\', {idx})">'
            f'{html_lib.escape(str(col))}<span class="sort-indicator"></span>'
            f'</th>'
        )

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            cells.append(f"<td>{_cell_html(row[col])}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
    <div class="mb-4">
        <h4 class="text-primary">{html_lib.escape(title)}</h4>
        <div class="table-wrap">
            <table id="{table_id}" class="sortable-table table table-striped table-bordered table-hover table-sm"
                   data-sort-col="" data-sort-dir="asc">
                <thead>
                    <tr>{''.join(headers)}</tr>
                </thead>
                <tbody>
                    {''.join(body_rows)}
                </tbody>
            </table>
        </div>
    </div>
    """

def download_relations(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name

def update_node_selector(node_list):
    return gr.update(choices=node_list)

with gr.Blocks(
    title="ICDC Synthetic Data Demo"
) as demo:
    gr.Markdown("# ICDC Synthetic Data Demo\nAnalyze learned property relationships, visualize them, and generate synthetic rows.")
    
    with gr.Accordion("Before you start", open=True):
        gr.Markdown(
            """
            1. PROVIDE NODE_MODEL_URL and PROP_MODEL_URL in Environment variables input text.
            2. Upload a JSON or Excel file of the data specific to study and node. 
            ## Data Format Requirements (Make sure the file has `type` or `type_`, for target node)

            ### Supported file types
            - JSON (`.json`)
            - Excel (`.xlsx`, `.xls`)

            ---

            ## Required structure

            Each row must represent a single entity and include a node type.

            ### Required field
            - `type` **or** `type_` → defines the node (e.g., sample, file, study)

            ---

            ## JSON format (recommended)

            ### Option 1: Array of objects
            ```json
            [
            {
                "type": "file",
                "file_name": "example.bam",
                "file_type": "bam",
                "file_size": 1048576
            },
            {
                "type": "file",
                "file_name": "example2.bai",
                "file_type": "bai",
                "file_size": 2048
            }
            ]
            ```
            3. Select the node from the dropdown if multiple files for multiple nodes are uploaded.
            4. Click **Load Schema**.
            5. Click **Run Property analysis**, to view the analysis (change parameter for weights)
            6. Review the analysis table.
            7. Click **Analyze and Generate Synthetic Data**.
            """
        )

    with gr.Row():
        env_text = gr.Textbox(
            label="Environment variables (.env text)",
            lines=8,
            placeholder="NODE_MODEL_URL=...\nPROP_MODEL_URL=...\nNEO4J_URI=...\nNEO4J_USER=...\nNEO4J_PASSWORD=...",
        )
        
        env_upload = gr.File(
            label="Upload .txt file",
            file_types=[".env", ".txt"],
            type="filepath",
        )
    env_upload.change(
        load_env_to_text,
        inputs=[env_upload],
        outputs=[env_text]
    )

    with gr.Row():
        data_upload = gr.File(
            label="Upload data",
            file_types=[".json", ".xlsx"],
            type="filepath",
            file_count="multiple",
        )
    # 👇 PLACE ERROR BOX IMMEDIATELY AFTER
    error_box = gr.HTML(value="")

    with gr.Row():
        selected_node_table = gr.Dropdown(label="Select node", choices=[], interactive=True)
        # view_data_file_select = gr.Dropdown(label="Select file", choices=[])
    
    view_data_table = gr.Dataframe(label="Selected Node data", interactive=False)
    node_data_state = gr.State({})
    node_list_state = gr.State([])

    selected_node_table.change(
        fn=display_selected_node,
        inputs=[selected_node_table, node_data_state],
        outputs=[view_data_table],
    )

    # load schema - node documents
    load_schema_btn = gr.Button("Load Schema")
    schema_state = gr.State([])
    # node_table = gr.DataFrame(label="Node Summary")
    # prop_table = gr.DataFrame(label="Property Details")

    schema_markdown = gr.Markdown(label="Schema View",  height=500)
    load_schema_btn.click(
        fn=load_and_display_schema,
        inputs=[env_text, node_list_state],
        outputs=[schema_markdown, schema_state, error_box],
    )

    load_relations_btn = gr.Button("View Node Relations")
    schema_markdown_1 = gr.JSON(label="Schema View")
    view_relation_table = gr.Dataframe(label="Relation", interactive=False)
    
    download_btn = gr.Button("Download Relations CSV")
    download_file = gr.File(label="Download file")
    rel_df_state = gr.State()
    node_tree_state = gr.State(pd.DataFrame())
    node_tree_table = gr.Dataframe(label="Node Tree", interactive=False)

    load_relations_btn.click(
        fn=build_relation_schemas,
        inputs=[node_list_state],
        outputs=[
            schema_markdown_1,
            view_relation_table,
            error_box,
            node_tree_state,
            node_tree_table
        ],
    )

    download_btn.click(
        fn=download_relations,
        inputs=view_relation_table,
        outputs=download_file,
    )
    
    # NodeSchema
    with gr.Row():
        run_analysis_btn = gr.Button("Run Property analysis")
        textual_analyze_btn = gr.Button("Textual Analyze")
    # run_analysis_btn = gr.Button("Run Property analysis")

    analysis_state = gr.State({})
    relationship_state = gr.State({})
    analysis_summary = gr.Markdown()
    analysis_dfs = gr.State({})

    feature_tables_html = gr.HTML()

    def render_tables(dfs):
        if not isinstance(dfs, dict) or not dfs:
            return "<p>No grouped tables to display.</p>"

        html_parts = []
        for i, (feature_type, df) in enumerate(dfs.items(), start=1):
            display_df = format_display_df(get_display_df(df, feature_type))
            table_id = f"feature_table_{i}"
            html_parts.append(_build_sortable_table(display_df, table_id, str(feature_type)))

        return "".join(html_parts)
    
    

    run_analysis_btn.click(
        run_analysis,
        inputs=[
            schema_state, 
            node_data_state, 
            selected_node_table
        ],
        outputs=[
            analysis_state, 
            relationship_state, 
            analysis_summary, 
            analysis_dfs
        ]
    ).then(
        render_tables,
        inputs=analysis_dfs,
        outputs=feature_tables_html
    )

    # textual analysis summary
    textual_analysis_summary = gr.Markdown()
    textual_analysis_table = gr.HTML()
    textual_analysis_state = gr.State(pd.DataFrame())

    # generate data
    num_rows_input = gr.Number(
        label="Number of rows to generate",
        value=60,
        precision=0
    )
    generate_btn = gr.Button("Analyze and Generate Synthetic Data")
    # generated_table = gr.Dataframe(label="Generated Data")
    # invalid_data_table = gr.Dataframe(label="Invalid Data")
    analysis_table = gr.HTML()
    generated_table = gr.HTML()
    invalid_data_table = gr.HTML()

    generate_btn.click(
        fn=generate_data,
        inputs=[
            schema_state,
            node_data_state,
            selected_node_table,
            num_rows_input
        ],
        outputs=[analysis_table, generated_table, invalid_data_table, error_box],
    )

    # cross node validation
    gr.Markdown("## Cross Node Analysis")
    select_all_btn = gr.Button("Select All Nodes")
    node_selector = gr.CheckboxGroup(
        label="Select Nodes for Cross Analysis",
        choices=[],   # ✅ must be list, not State
    )
    cross_btn = gr.Button("Run Cross Node Analysis")

    def select_all_nodes(nodes):
        return nodes

    select_all_btn.click(
        fn=select_all_nodes,
        inputs=[node_list_state],
        outputs=node_selector,
    )

    gr.Markdown("### View Node Analysis")
    cross_node_analysis_state = gr.State({})
    cross_node_relationship_state = gr.State({})
    cross_node_analysis_summary = gr.Markdown()
    feature_table = gr.Dataframe()
    cross_node_analysis_dfs = gr.Dataframe(label="Cross Node Relation", interactive=False)
    cross_node_analysis_html_table = gr.HTML()

    cross_btn.click(
        fn=run_cross_analysis,
        inputs=[
            schema_state,
            node_data_state,
            node_tree_state,
            node_selector
        ],
        outputs=[
            cross_node_analysis_state, 
            cross_node_relationship_state, 
            cross_node_analysis_summary, 
            feature_table,
            cross_node_analysis_dfs,
            cross_node_analysis_html_table
        ]
    )

    # data upload
    data_upload.change(
        fn=get_excel_or_json_data,
        inputs=[data_upload],
        outputs=[
            node_data_state,
            selected_node_table,
            view_data_table,
            node_list_state,
            error_box
        ],
    ).then(
        fn=update_node_selector,
        inputs=node_list_state,
        outputs=node_selector
    )


def main() -> None:
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7867")),
        head=AG_GRID_HEAD + f"<style>{CUSTOM_CSS}</style>"
    )
if __name__ == "__main__":
    main()
