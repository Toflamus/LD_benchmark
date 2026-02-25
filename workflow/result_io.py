from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_row_csv(path: Path, row: Mapping[str, object], fieldnames: Sequence[str]) -> None:
    """Append a single row to a CSV file; create with header if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def write_rows_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_traj_csv(path: Path, traj_points: Sequence[Sequence[int]]) -> None:
    """Write trajectory points to CSV with columns: step, e1..eN."""
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
