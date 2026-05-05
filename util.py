from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

def load_json_rows(path: str | Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of row objects")
    return pd.DataFrame(data)

def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".json":
        path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def rows_to_json(records: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")
