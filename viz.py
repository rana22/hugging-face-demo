from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _safe_filename(value: str) -> str:
    cleaned = []
    for ch in value.lower().strip():
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    return out or "chart"


def _save_line_chart(
    x: pd.Series,
    y_series: dict[str, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> Path:
    plt.figure(figsize=(10, 6))
    for label, y in y_series.items():
        plt.plot(x, y, marker="o", label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def plot_evaluator_learning_curve(
    results: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "evaluator_learning_curve.png",
) -> Path | None:
    """
    Plots evaluator performance against evidence size.
    This is not epoch-based learning; it shows whether performance improves
    as the amount of evidence increases.
    """
    if results is None or results.empty:
        return None

    df = results.copy()

    x_col = _first_existing_column(df, ["train_rows", "row_count", "support", "total_rows"])
    y_col = _first_existing_column(df, ["heldout_accuracy", "predictive_strength", "strength"])

    if not x_col or not y_col:
        return None

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col]).sort_values(x_col).reset_index(drop=True)

    if df.empty:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    # Smooth a bit so the trend is easier to read
    window = max(1, len(df) // 10)
    y_smooth = df[y_col].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], y_smooth, marker="o", label=f"{y_col} (smoothed)")

    if "baseline_accuracy" in df.columns:
        baseline = pd.to_numeric(df["baseline_accuracy"], errors="coerce")
        if baseline.notna().any():
            plt.plot(df[x_col], baseline.rolling(window=window, min_periods=1).mean(),
                     marker="o", label="baseline_accuracy (smoothed)")

    plt.title("Evaluator Performance vs Evidence Size")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved evaluator curve → {out_path}")
    return out_path

def plot_loss_error_curve(
    results: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "loss_error_curve.png",
) -> Path | None:
    """
    Generates and saves a loss/error curve.
    Returns the saved file path, or None if required data is missing.
    """
    if results is None or results.empty:
        return None

    df = results.copy()

    strength_col = _first_existing_column(df, ["strength", "score", "confidence", "weight"])
    if not strength_col or not pd.api.types.is_numeric_dtype(df[strength_col]):
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    df = df.sort_values(strength_col, ascending=False).reset_index(drop=True)

    strength = pd.to_numeric(df[strength_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    loss = 1.0 - strength

    pred_col = _first_existing_column(df, ["predictive_strength"])
    if pred_col and pd.api.types.is_numeric_dtype(df[pred_col]):
        predictive_strength = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        error = 1.0 - predictive_strength
    else:
        error = loss.copy()

    window = max(1, len(df) // 10)
    loss_smooth = loss.rolling(window=window, min_periods=1).mean()
    error_smooth = error.rolling(window=window, min_periods=1).mean()

    x = range(1, len(df) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, loss_smooth, marker="o", label="Model Loss (1 - strength)")
    plt.plot(x, error_smooth, marker="o", label="Prediction Error")

    plt.title("Model Learning Behavior (Loss & Error)")
    plt.xlabel("Relationships ranked from strongest to weakest")
    plt.ylabel("Error (lower is better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved loss curve → {out_path}")
    return out_path

# def plot_loss_error_curve(results: pd.DataFrame, output_dir: str | Path, filename: str = "loss_error_curve.png") -> Path | None:
#     """
#     Demo loss/error curve.

#     If the table has real training columns, uses them.
#     Otherwise derives a proxy curve from relationship strength.
#     """
#     if results is None or results.empty:
#         return None

#     df = results.copy()

#     # Prefer explicit loss columns if they exist.
#     train_loss_col = _first_existing_column(df, ["train_loss", "loss", "error", "train_error"])
#     test_loss_col = _first_existing_column(df, ["test_loss", "validation_loss", "val_loss", "test_error", "validation_error"])

#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     out_path = output_dir / filename

#     if train_loss_col or test_loss_col:
#         x = pd.Series(range(1, len(df) + 1), name="step")
#         y_series: dict[str, pd.Series] = {}

#         if train_loss_col and pd.api.types.is_numeric_dtype(df[train_loss_col]):
#             y_series["Train loss"] = df[train_loss_col].fillna(method="ffill").fillna(method="bfill")
#         if test_loss_col and pd.api.types.is_numeric_dtype(df[test_loss_col]):
#             y_series["Test loss"] = df[test_loss_col].fillna(method="ffill").fillna(method="bfill")

#         if y_series:
#             return _save_line_chart(
#                 x=x,
#                 y_series=y_series,
#                 title="Loss / Error Curve",
#                 xlabel="Step",
#                 ylabel="Loss / Error",
#                 out_path=out_path,
#             )

#     # Proxy curve from pairwise relationship strength.
#     strength_col = _first_existing_column(df, ["strength", "score", "confidence", "weight"])
#     if not strength_col or not pd.api.types.is_numeric_dtype(df[strength_col]):
#         return None

#     df = df.sort_values(strength_col, ascending=False).reset_index(drop=True)
#     x = pd.Series(range(1, len(df) + 1), name="rank")

#     strength = df[strength_col].fillna(0.0).clip(lower=0.0, upper=1.0)
#     loss_proxy = 1.0 - strength

#     pred_col = _first_existing_column(df, ["predictive_strength", "prediction_strength"])
#     if pred_col and pd.api.types.is_numeric_dtype(df[pred_col]):
#         error_proxy = 1.0 - df[pred_col].fillna(0.0).clip(lower=0.0, upper=1.0)
#     else:
#         error_proxy = loss_proxy

#     # Smooth slightly so the chart looks better in a demo.
#     window = max(1, len(df) // 10)
#     loss_smooth = loss_proxy.rolling(window=window, min_periods=1).mean()
#     error_smooth = error_proxy.rolling(window=window, min_periods=1).mean()

#     return _save_line_chart(
#         x=x,
#         y_series={
#             "Loss proxy": loss_smooth,
#             "Error proxy": error_smooth,
#         },
#         title="Loss / Error Curve",
#         xlabel="Ranked relationship",
#         ylabel="Loss / Error",
#         out_path=out_path,
#     )


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _save_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path) -> Path:
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def _save_histogram(values: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path) -> Path:
    plt.figure(figsize=(10, 6))
    plt.hist(values.dropna(), bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def _save_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path) -> Path:
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def generate_visual_report(
    results: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 15,
) -> list[Path]:
    """
    Generate lightweight PNG visualizations for a pairwise relationship table.

    Returns a list of saved chart paths.
    The function is intentionally defensive so it works across slightly different
    result schemas.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_paths: list[Path] = []

    if results is None or results.empty:
        empty_path = output_dir / "no_results.png"
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No relationships found", ha="center", va="center", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(empty_path, dpi=160, bbox_inches="tight")
        plt.close()
        return [empty_path]

    df = results.copy()

    # 1) Relationship/classification counts
    classification_col = _first_existing_column(
        df,
        ["classification", "relationship_type", "label", "category", "class"],
    )
    if classification_col and df[classification_col].notna().any():
        counts = (
            df[classification_col]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .head(top_n)
        )
        out_path = output_dir / f"{_safe_filename(classification_col)}_counts.png"
        chart_paths.append(
            _save_bar_chart(
                counts,
                title="Relationship counts",
                xlabel=classification_col,
                ylabel="Count",
                out_path=out_path,
            )
        )

    # 2) Top relationships by score/confidence if present
    score_col = _first_existing_column(
        df,
        ["score", "confidence", "probability", "similarity", "weight"],
    )
    if score_col and pd.api.types.is_numeric_dtype(df[score_col]):
        top_scores = (
            df[[score_col]]
            .dropna()
            .sort_values(score_col, ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        top_scores.index = [f"#{i+1}" for i in range(len(top_scores))]
        out_path = output_dir / f"{_safe_filename(score_col)}_top.png"
        chart_paths.append(
            _save_bar_chart(
                top_scores[score_col],
                title=f"Top {min(top_n, len(top_scores))} rows by {score_col}",
                xlabel="Row",
                ylabel=score_col,
                out_path=out_path,
            )
        )

        hist_path = output_dir / f"{_safe_filename(score_col)}_histogram.png"
        chart_paths.append(
            _save_histogram(
                df[score_col],
                title=f"Distribution of {score_col}",
                xlabel=score_col,
                ylabel="Frequency",
                out_path=hist_path,
            )
        )

    # 3) Relationship pair counts, if we can infer source/target columns
    source_col = _first_existing_column(
        df,
        ["source", "source_property", "left", "left_property", "property_a", "prop_a", "from"],
    )
    target_col = _first_existing_column(
        df,
        ["target", "target_property", "right", "right_property", "property_b", "prop_b", "to"],
    )
    if source_col and target_col:
        pair_series = (
            df[[source_col, target_col]]
            .fillna("")
            .astype(str)
            .agg(" -> ".join, axis=1)
            .value_counts()
            .head(top_n)
        )
        out_path = output_dir / "top_pairs.png"
        chart_paths.append(
            _save_bar_chart(
                pair_series,
                title=f"Top {min(top_n, len(pair_series))} pairs",
                xlabel="Pair",
                ylabel="Count",
                out_path=out_path,
            )
        )

    # 4) If we have two numeric columns, create a simple scatter for the first pair
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[:2]
        scatter_path = output_dir / f"{_safe_filename(x_col)}_vs_{_safe_filename(y_col)}.png"
        chart_paths.append(
            _save_scatter(
                df[x_col],
                df[y_col],
                title=f"{x_col} vs {y_col}",
                xlabel=x_col,
                ylabel=y_col,
                out_path=scatter_path,
            )
        )

    # 5) Loss / error curve
    loss_path = plot_loss_error_curve(df, output_dir)
    if loss_path:
        chart_paths.append(loss_path)

    # 6 learning
    loss_err = plot_evaluator_learning_curve(df, output_dir)
    if loss_err:
        chart_paths.append(loss_err)

    return chart_paths


__all__ = ["generate_visual_report"]