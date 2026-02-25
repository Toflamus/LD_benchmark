from __future__ import annotations

"""Result writing utilities for LD benchmark notebooks.

The benchmark workflow writes small CSV artifacts that are easy to load from
`DataProcessingScript.ipynb`:
- `summary.csv`: one row per run, appended over time
- `traj.csv`: a compact representation of the external-variable trajectory
"""

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if needed, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_row_csv(path: Path, row: Mapping[str, object], fieldnames: Sequence[str]) -> None:
    """Append one row to a CSV file, creating it with a header if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def write_rows_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write all rows to a CSV file, overwriting any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_traj_csv(path: Path, traj_points: Sequence[Sequence[int]]) -> None:
    """Write trajectory points to CSV with columns `step,e1..eN`.

    The number of `e*` columns is inferred as the maximum dimension seen in
    `traj_points`. If `traj_points` is empty, the file is still created with a
    minimal header (`step`) so downstream code can detect "no trajectory".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not traj_points:
        # Still write an empty file with a minimal header for tooling.
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step"])
        return

    dim = max(len(p) for p in traj_points)
    fieldnames = ["step"] + [f"e{i}" for i in range(1, dim + 1)]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, p in enumerate(traj_points, start=1):
            row = {"step": idx}
            for j, v in enumerate(p, start=1):
                row[f"e{j}"] = int(v)
            writer.writerow(row)


def write_json(path: Path, data: object) -> None:
    """Write `data` to JSON, overwriting any existing file.

    Uses `default=str` so common non-JSON types (e.g., Path, enums) get a
    reasonable string representation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
