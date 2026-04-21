from __future__ import annotations

import argparse, os
from pathlib import Path
from datetime import datetime
from numpy import True_
import pandas as pd

from dotenv import dotenv_values, load_dotenv

from docs import DocAlignmentModel
from features import PairwiseFeatureEngine
from io import load_json_rows, save_dataframe
from reporting import write_markdown_report
from neo4j_loader import fetch_rows_from_neo4j
from schema import load_node_schema, load_nodes_schema

load_dotenv()
print("SKIP_FIELDS =", os.getenv("SKIP_FIELDS"))
# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Analyze pairwise sample-node property relationships")
#     parser.add_argument(
#         "--data",
#         required=False,
#         help="Optional path to JSON array of flattened rows. If omitted, rows are loaded from Neo4j.",
#     )
#     parser.add_argument("--schema", required=True, help="Path to YAML schema for the sample node")
#     parser.add_argument("--output-dir", default="outputs", help="Directory for analysis outputs")
#     parser.add_argument("--neo4j-limit", type=int, default=None, help="Optional row limit when loading from Neo4j")
#     return parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze pairwise sample-node property relationships"
    )
    parser.add_argument(
        "--data",
        required=False,
        help="Optional path to JSON array of flattened rows. If omitted, rows are loaded from Neo4j.",
    )
    parser.add_argument(
        "--nodes",
        required=True,
        help="Provide list of nodes to generate data for",
    )
    parser.add_argument(
        "--schema",
        required=False,
        help="Path to YAML schema for the node",
    )
    parser.add_argument(
        "--study",
        dest="study_id",
        required=False,
        help="Study ID for Neo4j filtering",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--neo4j-limit",
        type=int,
        default=None,
        help="Optional row limit when loading from Neo4j",
    )
    return parser

# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     schema = load_node_schema(args.schema)
#     if args.data:
#         df = load_json_rows(args.data)
#     else:
#         df = fetch_rows_from_neo4j(
#             schema,
#             study_id=schema.name,
#             properties=schema.property_names(),
#             limit=args.neo4j_limit,
#         )
#     # df = load_json_rows(args.data)
#     # schema = load_node_schema(args.schema)
#     doc_model = DocAlignmentModel().fit(schema)
#     engine = PairwiseFeatureEngine(schema, doc_model)

#     results = engine.evaluate_all_pairs(schema, df)

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     save_dataframe(results, output_dir / "pairwise_relationships.csv")
#     save_dataframe(results, output_dir / "pairwise_relationships.json")
#     write_markdown_report(results, output_dir / "strong_relationships.md")

#     if results.empty:
#         print("No pairwise relationships were computed.")
#         return

#     display_cols = [
#         "A",
#         "B",
#         "classification",
#         "strength",
#         "predictive_strength",
#         "support",
#         "determinism",
#         "stability",
#         "doc_alignment",
#         "heldout_accuracy",
#         "baseline_accuracy",
#     ]
#     print(results[display_cols].head(15).to_string(index=False))
#     print(f"\nSaved outputs to {output_dir.resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # schema = load_nodes_schema(args.schema, args.nodes)
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    schemas = load_nodes_schema(nodes)

    for schema in schemas:
        if args.data:
            df = load_json_rows(args.data)
        else:
            df = fetch_rows_from_neo4j(
                schema,
                study_id=args.study_id,
                limit=args.neo4j_limit,
            )

        doc_model = DocAlignmentModel().fit(schema)
        engine = PairwiseFeatureEngine(schema, doc_model)
        results = engine.evaluate_all_pairs(schema, df)
        
        output_dir = Path(os.environ["OUTPUT_DIR"]) 
        if args.output_dir:
            output_dir = Path(args.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        node_name = schema.name
        study_id = args.study_id if args.study_id else "all"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        file_prefix = f"{node_name}_{study_id}_{timestamp}"

        save_dataframe(results, output_dir / node_name / f"{file_prefix}_pairwise_relationships.csv")
        save_dataframe(results, output_dir / node_name /f"{file_prefix}_pairwise_relationships.json")
        write_markdown_report(results, output_dir / node_name / f"{file_prefix}_strong_relationships.md")

    if results.empty:
        print("No pairwise relationships were computed.")
        return

    display_cols = [
        "A",
        "B",
        "classification",
        "strength",
        "predictive_strength",
        "support",
        "determinism",
        "stability",
        "doc_alignment",
        "suffix_match",
        "substring_match",
        "heldout_accuracy",
        "baseline_accuracy",
    ]

    
    print(results[display_cols].head(15).to_string(index=False))
    results = results.rename(columns={"heldout_accuracy": "heldout_accuracy (test)"})
    print(f"\nSaved outputs to {output_dir.resolve()}")

if __name__ == "__main__":
    main()
