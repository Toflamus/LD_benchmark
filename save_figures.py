#!/usr/bin/env python3
"""Save LD_benchmark plots to figure/<YYYYMMDD>/...

This script mirrors `workflow/DataProcessingScript.ipynb`, but writes plots to
files instead of displaying them. It is intended for Linux/CI or batch runs.

Outputs
-------
Creates:
    LD_benchmark/figure/<date>/...

Examples
--------
Save all figures for today's date:
    python save_figures.py

Save figures for a specific date:
    python save_figures.py --date 20260225

Save only LDSDA figures:
    python save_figures.py --algorithms gdpopt.ldsda

Notes
-----
- Uses a non-interactive Matplotlib backend (Agg).
- Expects benchmark artifacts under results/<date>/... (traj.csv, summary.csv).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _ld_root() -> Path:
    # This file lives in LD_benchmark/
    return Path(__file__).resolve().parent


def _default_date_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def _normalize_algo_tag(name: str) -> str:
    x = str(name).strip().lower()
    if x in {"ldsda", "gdpopt.ldsda"}:
        return "LDSDA"
    if x in {"ldbd", "gdpopt.ldbd"}:
        return "LDBD"
    return str(name).strip().upper()


def _algo_folder(tag: str) -> str:
    return tag.lower()


def _selected_algorithms(raw: list[str]) -> list[str]:
    algos = [_normalize_algo_tag(a) for a in raw]
    out: list[str] = []
    for a in algos:
        if a not in out:
            out.append(a)
    return out


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Save LD_benchmark figures to disk.")

    p.add_argument(
        "--date",
        default=_default_date_stamp(),
        help="Results date stamp (YYYYMMDD). Default: today.",
    )

    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["gdpopt.ldsda", "gdpopt.ldbd"],
        help="Algorithms to plot (e.g. gdpopt.ldsda gdpopt.ldbd).",
    )

    p.add_argument(
        "--results-dir",
        default=None,
        help="Override results directory. Default: LD_benchmark/results/<date>/",
    )

    p.add_argument(
        "--figures-dir",
        default=None,
        help="Override figures directory. Default: LD_benchmark/figure/<date>/",
    )

    p.add_argument(
        "--nt",
        type=int,
        default=30,
        help="NT value for the CSTR 2D + heatmap overlay figures. Default: 30.",
    )

    return p


def load_traj(traj_csv: Path) -> list[tuple[int, ...]]:
    """Load a trajectory saved by the benchmark runner (`traj.csv`)."""
    if not traj_csv.exists():
        return []

    import pandas as pd

    df = pd.read_csv(traj_csv)
    e_cols = [c for c in df.columns if c.startswith("e")]
    points: list[tuple[int, ...]] = []
    for _, row in df.iterrows():
        coords = tuple(int(row[c]) for c in e_cols if c in row and pd.notna(row[c]))
        if coords:
            points.append(coords)
    return points


def load_summaries_under(results_day_dir: Path, subdir: str):
    """Load all summary.csv files beneath results/<date>/<subdir>/ (recursive)."""
    import pandas as pd

    base = results_day_dir / subdir
    if not base.exists():
        return pd.DataFrame()

    paths = sorted(base.rglob("summary.csv"))
    if not paths:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        d["__file"] = str(p)
        dfs.append(d)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def filter_by_algorithms(df, selected: set[str]):
    if df.empty:
        return df
    if not selected:
        return df
    if "algorithm" not in df.columns:
        return df

    alg_col = df["algorithm"].map(_normalize_algo_tag)
    return df[alg_col.isin(selected)].copy()


def _style_fixed_2d_axes(
    ax,
    *,
    xlim,
    ylim,
    xticks,
    yticks,
    xlabel="e1",
    ylabel="e2",
):
    ax.tick_params(axis="both", which="both", direction="in", labelsize=16, width=2, length=6, top=True, right=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    axis_thickness = 2.5
    for spine in ax.spines.values():
        spine.set_linewidth(axis_thickness)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(list(xticks))
    ax.set_yticks(list(yticks))

    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold")
    ax.grid(True, which="major", linestyle=":", alpha=0.6, lw=1.5)


def plot_point_sequence(
    points: list[tuple[int, ...]],
    *,
    save_path: Path,
    title: str,
    xlim,
    ylim,
    xticks,
    yticks,
):
    import matplotlib.pyplot as plt

    if not points:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(len(points) - 1):
        ax.annotate(
            "",
            xy=points[i + 1],
            xytext=points[i],
            arrowprops=dict(arrowstyle="->", color="royalblue", lw=2, mutation_scale=20),
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, color="crimson", s=120, zorder=5)
    for i, (x, y) in enumerate(points, start=1):
        ax.text(x, y + 0.15, str(i), fontsize=16, color="darkred", fontweight="bold", ha="center", va="bottom")

    _style_fixed_2d_axes(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.set_title(title, fontsize=18, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_2d_paths(
    points_by_alg: dict[str, list[tuple[int, ...]]],
    *,
    save_path: Path,
    title: str,
    xlim,
    ylim,
    xticks,
    yticks,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    style = {
        "LDSDA": {"color": "red", "marker": "o"},
        "LDBD": {"color": "black", "marker": "s"},
    }

    for alg, pts in points_by_alg.items():
        if not pts:
            continue
        st = style.get(alg, {"color": "gray", "marker": "o"})

        for i in range(len(pts) - 1):
            ax.annotate(
                "",
                xy=pts[i + 1],
                xytext=pts[i],
                arrowprops=dict(arrowstyle="->", color=st["color"], lw=2, mutation_scale=18),
            )

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=st["color"], lw=2, marker=st["marker"], label=alg)

        y_off = 0.15 if alg == "LDSDA" else -0.25
        for k, (x, y) in enumerate(pts, start=1):
            ax.text(x, y + y_off, str(k), fontsize=12, color=st["color"], fontweight="bold", ha="center", va="bottom")

    _style_fixed_2d_axes(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_small_batch_3d(enum_csv: Path, traj: list[tuple[int, ...]], *, save_path: Path, title: str):
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.ticker import MaxNLocator

    df = pd.read_csv(enum_csv)

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df["x"],
        df["y"],
        df["z"],
        c=df["Objective"],
        cmap="viridis",
        s=100,
        label="Optimal Points",
        edgecolors="k",
        zorder=5,
    )

    all_coords = {(i, j, k) for i in range(1, 5) for j in range(1, 5) for k in range(1, 5)}
    existing_coords = set(zip(df["x"], df["y"], df["z"]))
    missing_coords = list(all_coords - existing_coords)
    if missing_coords:
        mx, my, mz = zip(*missing_coords)
        ax.scatter(
            mx,
            my,
            mz,
            marker="^",
            facecolors="none",
            edgecolors="gray",
            s=60,
            alpha=0.3,
            label="Infeasible/Missing",
        )

    seq = [(int(p[0]), int(p[1]), int(p[2])) for p in (traj or []) if len(p) >= 3]
    for i in range(len(seq) - 1):
        start = seq[i]
        end = seq[i + 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="blue", lw=2, zorder=10)
        ax.quiver(
            start[0],
            start[1],
            start[2],
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
            length=0.5,
            color="blue",
            normalize=True,
            arrow_length_ratio=0.3,
        )

    for idx, (px, py, pz) in enumerate(seq, start=1):
        ax.text(px, py, pz + 0.1, str(idx), color="red", fontsize=16, fontweight="bold")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlim(0.1, 4.1)
    ax.set_ylim(0.1, 4.1)
    ax.set_zlim(0.1, 4.1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_zticks([1, 2, 3, 4])

    ax.tick_params(axis="both", which="major", direction="in", labelsize=12, pad=5)
    ax.set_xlabel("Number of Mixers ($Z_E$)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Reactors ($Z_E$)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_zlabel("Number of Centrifuges ($Z_E$)", fontsize=14, fontweight="bold", labelpad=10)

    fig.colorbar(scatter, ax=ax, label="Objective Value", pad=0.1, shrink=0.6)
    ax.set_title(title, fontsize=18, fontweight="bold")
    plt.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_cstr_perf(df_cstr, *, save_path: Path):
    import matplotlib.pyplot as plt
    import pandas as pd

    if df_cstr.empty:
        return

    # Extract NT from instance='NT30'
    if "instance" in df_cstr.columns:
        nt = df_cstr["instance"].astype(str).str.extract(r"NT(\d+)")[0]
        df_cstr["NT"] = pd.to_numeric(nt, errors="coerce")
    else:
        df_cstr["NT"] = None

    df_cstr["cpu_time"] = df_cstr["cpu_time_s"].where(df_cstr["cpu_time_s"].notna(), df_cstr.get("wall_time_s"))
    df_cstr = df_cstr.dropna(subset=["NT", "objective_value"])
    df_cstr["NT"] = df_cstr["NT"].astype(int)

    df_cstr = df_cstr.sort_values(["algorithm", "NT", "__file"]).groupby(["algorithm", "NT"], as_index=False).tail(1)

    style = {
        "LDSDA": {"color": "red", "marker": "o", "label": "LD-SDA"},
        "LDBD": {"color": "black", "marker": "s", "label": "LD-BD"},
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for alg in sorted(df_cstr["algorithm"].unique()):
        tag = _normalize_algo_tag(alg)
        sub = df_cstr[df_cstr["algorithm"].map(_normalize_algo_tag) == tag].sort_values("NT")
        st = style.get(tag, {"color": "gray", "marker": "o", "label": tag})
        ax1.plot(sub["NT"], sub["objective_value"], color=st["color"], marker=st["marker"], lw=2, label=st["label"])
        ax2.plot(sub["NT"], sub["cpu_time"], color=st["color"], marker=st["marker"], lw=2, label=st["label"])

    ax1.set_ylabel("Objective")
    ax1.set_title("CSTR: objective vs NT")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.set_xlabel("NT")
    ax2.set_ylabel("CPU time [s]")
    ax2.set_title("CSTR: CPU time vs NT")
    ax2.grid(True, linestyle=":", alpha=0.6)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_cstr_heatmap_overlay(enum_csv: Path, cstr_points: dict[str, list[tuple[int, ...]]], *, save_path: Path, title: str):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    df_enum = pd.read_csv(enum_csv)
    df_enum = df_enum.dropna(subset=["x", "y", "Objective"]).copy()
    df_enum["x"] = df_enum["x"].astype(int)
    df_enum["y"] = df_enum["y"].astype(int)
    df_enum["Objective"] = df_enum["Objective"].astype(float)

    pivot_df = df_enum.pivot_table(index="y", columns="x", values="Objective", aggfunc="min")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        pivot_df,
        origin="lower",
        cmap="viridis_r",
        aspect="equal",
        extent=[pivot_df.columns.min() - 0.5, pivot_df.columns.max() + 0.5, pivot_df.index.min() - 0.5, pivot_df.index.max() + 0.5],
    )

    style = {"LDSDA": {"color": "red", "marker": "o"}, "LDBD": {"color": "black", "marker": "s"}}
    for alg, pts in cstr_points.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        st = style.get(alg, {"color": "white", "marker": "o"})
        ax.plot(xs, ys, color=st["color"], lw=2, marker=st["marker"], label=f"{alg} traj")
        for k, (x, y) in enumerate(pts, start=1):
            ax.text(x, y, str(k), color=st["color"], fontsize=8, fontweight="bold")

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    ax.set_xlabel("External variable x (e1)")
    ax.set_ylabel("External variable y (e2)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Objective")
    ax.legend(loc="upper left")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_column_perf(df_col, *, save_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    if df_col.empty:
        return

    df_col["cpu_time"] = df_col["cpu_time_s"].where(df_col["cpu_time_s"].notna(), df_col.get("wall_time_s"))
    df_col = df_col.dropna(subset=["objective_value"])

    if "initial_point_key" in df_col.columns:
        df_col = df_col.sort_values(["algorithm", "initial_point_key"])

    style_map = {
        "LDSDA": {"marker": "*", "color": "red", "label": "LD-SDA"},
        "LDBD": {"marker": "s", "color": "black", "label": "LD-BD", "fillstyle": "none"},
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    def _plot_scatter(ax, y_col, include_legend=False):
        for alg in sorted(df_col["algorithm"].unique()):
            tag = _normalize_algo_tag(alg)
            subset = df_col[df_col["algorithm"].map(_normalize_algo_tag) == tag]
            style = style_map.get(tag, {"marker": "o", "color": "gray", "label": str(alg)})
            ax.scatter(
                subset["initial_point_key"],
                subset[y_col],
                marker=style["marker"],
                color=style["color"],
                label=style.get("label") if include_legend else None,
                s=80,
                facecolors=style["color"] if "fillstyle" not in style else "none",
            )

    _plot_scatter(ax1, "objective_value", include_legend=True)
    _plot_scatter(ax2, "cpu_time")

    max_key = int(df_col["initial_point_key"].max()) if "initial_point_key" in df_col.columns else 0
    xmax = max(10, max_key + 1)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)
        ax.set_xlim(0, xmax)

    ax1.tick_params(axis="y", direction="in", labelsize=12)
    ax1.set_ylabel("Objective", fontsize=12)
    ax1.set_title("Column: objective vs init key", loc="left", fontweight="bold")
    ax1.legend(loc="upper right", ncol=2)

    ax2.tick_params(axis="x", direction="in", labelsize=12)
    ax2.tick_params(axis="y", direction="in", labelsize=12)
    ax2.set_ylabel("CPU time [s]", fontsize=12)
    ax2.set_xlabel("Initial point key", fontsize=12)
    ax2.set_title("Column: CPU time vs init key", loc="left", fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Ensure we never try to display plots
    import matplotlib

    matplotlib.use("Agg")

    ld_root = _ld_root()

    # Optional: prefer local Pyomo source tree if present (repo layout: <workspace>/pyomo/pyomo/...)
    pyomo_src = ld_root.parent / "pyomo"
    if (pyomo_src / "pyomo").is_dir() and str(pyomo_src) not in sys.path:
        sys.path.insert(0, str(pyomo_src))

    selected = _selected_algorithms(args.algorithms)

    results_day_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else (ld_root / "results" / args.date)
    figures_day_dir = Path(args.figures_dir).expanduser().resolve() if args.figures_dir else (ld_root / "figure" / args.date)

    assets_dir = ld_root / "assets"
    sb_enum_csv = assets_dir / "compl_enum_small_batch_baron.csv"
    cstr_enum_csv = assets_dir / "compl_enum_cstr_30_baron_dantzig.csv"

    # --- Toy (math) ---
    toy_dir = _ensure_dir(figures_day_dir / "toy")
    toy_points: dict[str, list[tuple[int, ...]]] = {}
    for alg in selected:
        traj_csv = results_day_dir / "toy" / _algo_folder(alg) / "traj.csv"
        toy_points[alg] = load_traj(traj_csv)

    # Fixed 1..5 axes for toy
    toy_xlim = (0.5, 5.5)
    toy_ylim = (0.5, 5.5)
    toy_ticks = [1, 2, 3, 4, 5]

    for alg, pts in toy_points.items():
        plot_point_sequence(
            pts,
            save_path=toy_dir / f"toy_{alg.lower()}_traj.png",
            title=f"Toy Trajectory ({alg})",
            xlim=toy_xlim,
            ylim=toy_ylim,
            xticks=toy_ticks,
            yticks=toy_ticks,
        )

    plot_2d_paths(
        toy_points,
        save_path=toy_dir / "toy_combined.png",
        title="Toy Trajectory (combined)",
        xlim=toy_xlim,
        ylim=toy_ylim,
        xticks=toy_ticks,
        yticks=toy_ticks,
    )

    # --- Small-batch ---
    if sb_enum_csv.exists():
        sb_dir = _ensure_dir(figures_day_dir / "small_batch")
        for alg in selected:
            traj_csv = results_day_dir / "small_batch" / _algo_folder(alg) / "traj.csv"
            pts = load_traj(traj_csv)
            plot_small_batch_3d(
                sb_enum_csv,
                pts,
                save_path=sb_dir / f"small_batch_{alg.lower()}_traj.png",
                title=f"Small Batch Trajectory ({alg})",
            )

    # --- Column objective + CPU vs init key ---
    df_col = load_summaries_under(results_day_dir, "column")
    df_col = filter_by_algorithms(df_col, set(selected))
    if not df_col.empty and "model" in df_col.columns:
        df_col = df_col[df_col["model"] == "column"]

    if not df_col.empty:
        plot_column_perf(df_col, save_path=_ensure_dir(figures_day_dir / "column") / "column_perf.png")

    # --- CSTR objective + CPU vs NT ---
    df_cstr = load_summaries_under(results_day_dir, "cstr")
    df_cstr = filter_by_algorithms(df_cstr, set(selected))
    if not df_cstr.empty and "model" in df_cstr.columns:
        df_cstr = df_cstr[df_cstr["model"] == "cstr"]

    if not df_cstr.empty:
        plot_cstr_perf(df_cstr, save_path=_ensure_dir(figures_day_dir / "cstr") / "cstr_perf_vs_nt.png")

    # --- CSTR NT=<nt>: 2D trajectory ---
    cstr_dir = _ensure_dir(figures_day_dir / "cstr" / f"NT{int(args.nt)}")
    cstr_points: dict[str, list[tuple[int, ...]]] = {}
    for alg in selected:
        traj_csv = results_day_dir / "cstr" / f"NT{int(args.nt)}" / _algo_folder(alg) / "traj.csv"
        cstr_points[alg] = load_traj(traj_csv)

    # Fixed 1..NT axes for CSTR
    nt = int(args.nt)
    cstr_xlim = (0.5, nt + 0.5)
    cstr_ylim = (0.5, nt + 0.5)
    cstr_ticks = list(range(1, nt + 1))

    for alg, pts in cstr_points.items():
        plot_point_sequence(
            pts,
            save_path=cstr_dir / f"cstr_NT{nt}_{alg.lower()}_traj.png",
            title=f"CSTR NT={nt} Trajectory ({alg})",
            xlim=cstr_xlim,
            ylim=cstr_ylim,
            xticks=cstr_ticks,
            yticks=cstr_ticks,
        )

    plot_2d_paths(
        cstr_points,
        save_path=cstr_dir / f"cstr_NT{nt}_combined.png",
        title=f"CSTR NT={nt} Trajectory (combined)",
        xlim=cstr_xlim,
        ylim=cstr_ylim,
        xticks=cstr_ticks,
        yticks=cstr_ticks,
    )

    # --- CSTR heatmap overlay (NT=30 assets) ---
    if cstr_enum_csv.exists() and nt == 30:
        plot_cstr_heatmap_overlay(
            cstr_enum_csv,
            cstr_points,
            save_path=cstr_dir / "cstr_NT30_heatmap_overlay.png",
            title="CSTR NT=30 landscape (assets) with trajectory overlay",
        )

    print(f"Saved figures under: {figures_day_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
