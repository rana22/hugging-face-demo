# Load schema
# Load relationships CSV
# Load real data (Neo4j or input file)

#         ↓

# Build allowed field list
# Build relationship list
# Build parent-child map
# Build conditional probability maps
# Build generation order
# Build marginals

#         ↓

# For each synthetic row:
#     For each field in generation order:
#         If parent exists and parent value is known:
#             sample from conditional distribution
#         Else:
#             sample from marginal distribution

#     Repair row:
#         - reconcile parent/child constraints
#         - enforce enum validity
#         - repeat up to max rounds

#     Validate row:
#         - check every relationship mapping
#         - reject if any constraint fails

#         ↓
# Keep row only if valid

#         ↓

# Stop when n valid rows are collected or max attempts reached

#         ↓

# Save CSV and JSON


from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
import requests

import gradio as gr
import pandas as pd
import yaml
import html as html_lib

# These imports assume your package is included in the Space repo.
# If some functions are named differently in your current branch,
# adjust the import lines only.
from icdc_data_generator_mvp.docs import DocAlignmentModel
from icdc_data_generator_mvp.features import PairwiseFeatureEngine
from icdc_data_generator_mvp.generator import SyntheticDataGenerator
from icdc_data_generator_mvp.io import load_json_rows
from icdc_data_generator_mvp.neo4j_loader import fetch_rows_from_neo4j
from icdc_data_generator_mvp.reporting import write_markdown_report
from icdc_data_generator_mvp.schema import load_node_schema
from icdc_data_generator_mvp.viz import generate_visual_report
from .evaluator2 import PairwiseRelationshipEvaluator

BASE_DIR = Path(__file__).resolve().parent
TABLE_JS = (BASE_DIR / "static" / "table.js").read_text(encoding="utf-8")

AG_GRID_HEAD = f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-quartz.css">
<script src="https://cdn.jsdelivr.net/npm/ag-grid-community/dist/ag-grid-community.min.js"></script>
<script>
  {TABLE_JS}
</script>
"""

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
            [{"Name": "Alice", "Age": 25, "Active": True},
             {"Name": "Bob", "Age": 30, "Active": False}]
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

def show_node_summary_tables(selected_node: str, results_state: dict, generated_state: dict, summary_state: dict):
    results_df = results_state.get(selected_node, pd.DataFrame())
    generated_df = generated_state.get(selected_node, pd.DataFrame())
    summary_text = summary_state.get(selected_node, "No summary available.")

    return (
        summary_text,
        dataframe_to_ag_grid_html(results_df, f"relationships_{selected_node}"),
        dataframe_to_ag_grid_html(generated_df, f"generated_{selected_node}"),
    )

def _parse_env_text(env_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (env_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _load_env_upload(env_file) -> dict[str, str]:
    if env_file is None:
        return {}

    if isinstance(env_file, (str, Path)):
        path = Path(env_file)
    else:
        path = Path(env_file.name)

    text = path.read_text(encoding="utf-8")
    return _parse_env_text(text)

def _fetch_node_names() -> list[str]:
    node_url = os.getenv("NODE_MODEL_URL", "")
    if not node_url:
        return []

    resp = requests.get(node_url, timeout=30)
    resp.raise_for_status()
    data = yaml.safe_load(resp.text)
    nodes = data.get("Nodes", {}) or data.get("nodes", {}) or {}
    return sorted(nodes.keys())

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

def run_pipeline(
    env_text: str,
    env_upload,
    schema_mode: str,
    node_name,
    schema_yaml_text: str,
    study_id: str,
    neo4j_limit: int,
    output_rows: int,
    use_neo4j: bool,
    viz: bool,
) -> tuple[str, str | None, str | None, list[tuple[str, str]], str]:
    env_values: dict[str, str] = {}
    env_values.update(_parse_env_text(env_text))
    env_values.update(_load_env_upload(env_upload))
    for k, v in env_values.items():
        os.environ[k] = v

    selected_nodes = node_name if isinstance(node_name, list) else [node_name]
    selected_nodes = [n for n in selected_nodes if n]

    results_by_node: dict[str, pd.DataFrame] = {}
    generated_by_node: dict[str, pd.DataFrame] = {}
    summary_by_node: dict[str, str] = {}
    relationship_by_node: dict[str, pd.DataFrame] = {}

    if not selected_nodes:
        raise gr.Error("Choose at least one node.")

    if schema_mode == "upload" and len(selected_nodes) != 1:
        raise gr.Error("Upload mode currently supports exactly one selected node.")

    if schema_mode == "generated":
        node_url = os.getenv("NODE_MODEL_URL", "")
        prop_url = os.getenv("PROP_MODEL_URL", "")
        if not node_url or not prop_url:
            raise gr.Error("Set NODE_MODEL_URL and PROP_MODEL_URL in the environment.")

        node_model = yaml.safe_load(requests.get(node_url, timeout=30).text)
        prop_model = yaml.safe_load(requests.get(prop_url, timeout=30).text)
        nodes = node_model.get("Nodes", {}) or node_model.get("nodes", {}) or {}
        props = prop_model.get("PropDefinitions", {}) or prop_model.get("properties", {}) or {}
    else:
        if not schema_yaml_text.strip():
            raise gr.Error("Please paste or upload a node schema YAML.")
        schema_data = yaml.safe_load(schema_yaml_text)
        nodes = schema_data.get("Nodes", {}) or schema_data.get("nodes", {}) or {}
        props = {}

    output_dir = Path(tempfile.mkdtemp(prefix="icdc_demo_"))
    excel_path = output_dir / f"{study_id or 'all'}_results.xlsx"
    combined_report_path = output_dir / f"{study_id or 'all'}_strong_relationships.md"

    summary_parts: list[str] = []
    chart_items: list[tuple[str, str]] = []
    synth_messages: list[str] = []

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for selected_node in selected_nodes:
            if schema_mode == "generated":
                if selected_node not in nodes:
                    raise gr.Error(f"Node '{selected_node}' not found in node model.")

                node_spec = nodes[selected_node]
                merged = {
                    "node": selected_node,
                    "description": node_spec.get("Desc", ""),
                    "exclude_like": ["sample_id", "crdc_id", "comment", "uuid", "created", "updated"],
                    "properties": {},
                }

                for p in node_spec.get("Props", []) or node_spec.get("props", []):
                    if p in props:
                        spec = props[p]
                        merged["properties"][p] = {
                            "description": spec.get("Desc", ""),
                            "type": spec.get("Type", spec.get("type")),
                            "enum": spec.get("Enum", spec.get("enum", [])) or [],
                            "required": spec.get("Req", spec.get("required")),
                            "tags": spec.get("Tags", spec.get("tags", {})) or {},
                        }

                with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
                    yaml.safe_dump(merged, tmp, sort_keys=False)
                    schema_path = tmp.name
            else:
                # Upload mode assumes the pasted YAML already describes the selected node.
                schema_path = None
                with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
                    yaml.safe_dump(schema_data, tmp, sort_keys=False)
                    schema_path = tmp.name

            schema = load_node_schema(schema_path, nodes)

            if use_neo4j:
                df = fetch_rows_from_neo4j(
                    schema,
                    study_id=study_id or None,
                    limit=neo4j_limit or None,
                )
            else:
                raise gr.Error(
                    "This demo Space is configured for Neo4j-backed loading. "
                    "Turn on the Neo4j toggle and provide env values."
                )

            # engine = PairwiseFeatureEngine(schema, doc_model)
            engine = PairwiseRelationshipEvaluator(schema)
            results = engine.evaluate_all_pairs(df)

            # summarize for each node
            summary_by_node.setdefault(schema.name, _build_status_md(schema.name, study_id or None, results))
            sheet_name = _safe_excel_sheet_name(schema.name, set(writer.book.sheetnames))
            results.to_excel(writer, sheet_name=sheet_name, index=False)

            node_report_path = output_dir / f"{sheet_name}_strong_relationships.md"
            write_markdown_report(results, node_report_path)

            # summary_parts.append(_build_status_md(schema.name, study_id or None, results))

            if viz:
                node_chart_dir = output_dir / sheet_name
                node_chart_dir.mkdir(parents=True, exist_ok=True)
                chart_paths = generate_visual_report(results, node_chart_dir, top_n=15)
                for p in chart_paths:
                    chart_items.append((str(p), f"{schema.name}: {p.name}"))

            node_key = selected_node
            summary_by_node[node_key] = _build_status_md(node_key, study_id or None, results)
            if not results.empty:
                results_by_node[node_key] = results.copy()
                columns = list(df.columns)
                gen = SyntheticDataGenerator(
                    real_rows=df,
                    relationships=results,
                    schema=schema,
                )
                synth_df = gen.generate(output_rows)
                print(f"generated {len(synth_df)}")

                generated_by_node[node_key] = synth_df.copy()
                synth_csv = output_dir / f"{sheet_name}_synthetic_rows.csv"
                synth_json = output_dir / f"{sheet_name}_synthetic_rows.json"
                synth_df.to_csv(synth_csv, index=False)
                synth_df.to_json(synth_json, orient="records", indent=2)
                synth_messages.append(f"{node_key}: generated {len(synth_df)} synthetic rows")
            else:
                synth_messages.append(f"{schema.name}: synthetic generation skipped (no relationships)")

    combined_report_path.write_text("\n\n---\n\n".join(summary_parts), encoding="utf-8")

    # summary = "\n\n---\n\n".join(summary_parts)
    synth_msg = "\n".join(synth_messages) if synth_messages else "Synthetic generation skipped."

    return (
        summary_by_node,
        chart_items,
        synth_msg,
        results_by_node,
        generated_by_node,
        relationship_by_node,
        gr.update(choices=selected_nodes, value=selected_nodes[0]),
    )

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

with gr.Blocks(
    title="ICDC Synthetic Data Demo",
    head=AG_GRID_HEAD,
) as demo:
    gr.Markdown("# ICDC Synthetic Data Demo\nAnalyze learned property relationships, visualize them, and generate synthetic rows.")

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
    with gr.Row():
        study_id = gr.Textbox(label="Study ID", placeholder="OSA01")
    with gr.Row():
        node_names = gr.CheckboxGroup(choices=[], label="Nodes")
    with gr.Row():
        schema_mode = gr.Radio(["generated", "upload"], value="generated", label="Schema mode")
        # node_name = gr.Dropdown(choices=[], label="Node", allow_custom_value=True)

    schema_yaml_text = gr.Textbox(
        label="Node schema YAML",
        lines=14,
        visible=False,
    )
    use_neo4j = gr.Checkbox(value=True, label="Load data from Neo4j")
    viz = gr.Checkbox(value=True, label="Generate visualizations")

    with gr.Row():
        neo4j_limit = gr.Number(value=200, precision=0, label="Neo4j row limit")
        output_rows = gr.Number(value=50, precision=0, label="Synthetic rows to generate")

    btn = gr.Button("Run analysis")

    # pairwise_csv = gr.File(label="Pairwise relationships CSV")
    # pairwise_md = gr.File(label="Strong relationships report")
    # charts = gr.Gallery(label="Visualizations", columns=2, height=320)
    synth_status = gr.Markdown(label="Synthetic generation")

    node_view = gr.Dropdown(choices=[], label="View node")
    summary_md = gr.Markdown()

    relationships_grid = gr.HTML()
    generated_grid = gr.HTML()

    # relationships_table = gr.Dataframe(
    #     label="Relationships",
    #     interactive=False,
    # )

    search_box = gr.Textbox(label="Search", placeholder="Type to filter...")
    generated_table = gr.Dataframe()

    charts = gr.Gallery(label="Visualizations", columns=2, height=320)

    results_state = gr.State({})
    generated_state = gr.State({})
    summary_state = gr.State({})
    relationship_state = gr.State({})

    def refresh_nodes(_env_text: str, _upload):
        env_values = _parse_env_text(_env_text)
        for k, v in env_values.items():
            os.environ[k] = v
        try:
            return gr.update(choices=_fetch_node_names(), value=[])
        except Exception:
            return gr.update(choices=[], value=[])

    env_text.change(refresh_nodes, inputs=[env_text, env_upload], outputs=[node_names])
    env_upload.change(
        load_env_to_text,
        inputs=[env_upload],
        outputs=[env_text]
    )

    schema_mode.change(
        toggle_schema_fields,
        inputs=[schema_mode],
        outputs=[schema_yaml_text],
    )

    generated_table = gr.Dataframe(
        label="Generated data",
        interactive=False,
    )
    relationships_grid = gr.HTML()
    # gr.HTML(df_to_html(None), js_on_load=js)
    node_view.change(
        show_node_sumamry_tables,
        inputs=[node_view, results_state, generated_state, summary_state],
        outputs=[summary_md, generated_table, relationships_grid], 
    )

    search_box.change(
        fn=filter_df,
        inputs=[generated_table, search_box],
        outputs=generated_table
    )
    
    print(f"sutdy {study_id}")
    btn.click(
        run_pipeline,
        inputs=[
            env_text, 
            env_upload, 
            schema_mode, 
            node_names, 
            schema_yaml_text, 
            study_id, 
            neo4j_limit, 
            output_rows,
            use_neo4j,
            viz
        ],
        outputs=[
            summary_state,
            charts,
            synth_status,
            results_state,
            generated_state,
            relationship_state,
            node_view,
        ],
    )


def main() -> None:
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))

if __name__ == "__main__":
    main()
