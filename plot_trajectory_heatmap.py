#!/usr/bin/env python3
"""Plot LD-BD / LD-SDA trajectory on a heatmap of the enumerated objective landscape.

Background
----------
- Feasible points from the complete enumeration CSV are shown as a coloured
  heatmap (green = low/good objective, red = high/bad objective).
- Infeasible points (NaN objective in the enumeration) and grid cells outside
  the enumerated region are shown as red crosses.
- The algorithm's search trajectory is overlaid with numbered arrows.

Output directories
------------------
- Single trajectories:       figure/<date>_column_traj/<ALGO>_init_NNN_trajectory.png
- Two-init comparison:       figure/<date>_column_traj/<ALGO>_init_NNN_vs_init_MMM.png
- Two-date comparison:       figure/<date>-vs-<date2>_column_traj/<ALGO>_init_NNN_comparison.png

All output directories can be overridden with ``--output-dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ld_root() -> Path:
    return Path(__file__).resolve().parent


def load_enum(csv_path: Path) -> pd.DataFrame:
    """Load the complete-enumeration CSV and clean it."""
    df = pd.read_csv(csv_path)
    # Drop the 'Final' summary row if present
    if "Status" in df.columns:
        df = df[df["Status"] != "Final"].copy()
    df["x"] = df["x"].astype(int)
    df["y"] = df["y"].astype(int)
    return df


def load_traj(traj_csv: Path) -> list[tuple[int, int]]:
    """Load a trajectory CSV (step, e1, e2)."""
    if not traj_csv.exists():
        return []
    df = pd.read_csv(traj_csv)
    points: list[tuple[int, int]] = []
    for _, row in df.iterrows():
        points.append((int(row["e1"]), int(row["e2"])))
    return points


def build_landscape(df_enum: pd.DataFrame, x_range: range, y_range: range):
    """Build objective grid and infeasible-point list from enumeration data.

    Returns
    -------
    obj_grid : 2D np.ndarray (y, x) with NaN for infeasible/missing
    feasible_pts : list of (x, y, obj)
    infeasible_pts : list of (x, y)
    """
    # Index enumeration by (x, y)
    enum_dict: dict[tuple[int, int], float] = {}
    for _, row in df_enum.iterrows():
        key = (int(row["x"]), int(row["y"]))
        enum_dict[key] = row["Objective"]  # NaN if infeasible

    feasible_pts: list[tuple[int, int, float]] = []
    infeasible_pts: list[tuple[int, int]] = []

    obj_grid = np.full((len(y_range), len(x_range)), np.nan)

    for iy, y in enumerate(y_range):
        for ix, x in enumerate(x_range):
            val = enum_dict.get((x, y), np.nan)
            if np.isfinite(val):
                obj_grid[iy, ix] = val
                feasible_pts.append((x, y, val))
            else:
                infeasible_pts.append((x, y))

    return obj_grid, feasible_pts, infeasible_pts


def plot_single(
    df_enum: pd.DataFrame,
    traj: list[tuple[int, int]],
    *,
    save_path: Path,
    title: str,
    x_range: range,
    y_range: range,
):
    """Plot one trajectory on the heatmap background."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    obj_grid, feasible_pts, infeasible_pts = build_landscape(
        df_enum, x_range, y_range
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    # -- Colored scatter for feasible points (blue-green palette) --
    if feasible_pts:
        feas_x, feas_y, feas_obj = zip(*feasible_pts)
        sc = ax.scatter(
            feas_x, feas_y,
            c=feas_obj,
            cmap="GnBu",
            s=250,
            marker="o",
            edgecolors="black",
            linewidths=1,
            zorder=3,
            label="Feasible",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Objective Value", fontsize=14, fontweight="bold")

    # -- Red crosses for infeasible / unenumerated points --
    if infeasible_pts:
        inf_x, inf_y = zip(*infeasible_pts)
        ax.scatter(
            inf_x,
            inf_y,
            marker="x",
            c="red",
            s=120,
            linewidths=2.5,
            zorder=3,
            label="Infeasible",
        )

    # -- Trajectory arrows --
    if traj:
        for i in range(len(traj) - 1):
            x0, y0 = traj[i]
            x1, y1 = traj[i + 1]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="royalblue",
                    lw=2.5,
                    mutation_scale=18,
                    connectionstyle="arc3,rad=0.1",
                ),
                zorder=6,
            )

        # Numbered dots along the trajectory
        traj_x = [p[0] for p in traj]
        traj_y = [p[1] for p in traj]
        ax.scatter(traj_x, traj_y, color="white", edgecolors="royalblue",
                   linewidths=2, s=200, zorder=7)
        for k, (px, py) in enumerate(traj, start=1):
            ax.text(
                px,
                py,
                str(k),
                fontsize=11,
                color="royalblue",
                fontweight="bold",
                ha="center",
                va="center",
                zorder=8,
            )

        # Highlight start and end
        ax.scatter(
            [traj[0][0]], [traj[0][1]],
            color="lime", edgecolors="black", linewidths=2, s=300, zorder=9,
            marker="o", label="Start",
        )
        ax.text(traj[0][0], traj[0][1], "1", fontsize=11, color="black",
                fontweight="bold", ha="center", va="center", zorder=10)
        ax.scatter(
            [traj[-1][0]], [traj[-1][1]],
            color="gold", edgecolors="black", linewidths=2, s=300, zorder=9,
            marker="*", label="End",
        )

    # -- Axis styling --
    ax.set_xlim(x_range.start - 0.5, x_range.stop - 0.5)
    ax.set_ylim(y_range.start - 0.5, y_range.stop - 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(
        axis="both", which="both", direction="in",
        labelsize=13, width=2, length=5, top=True, right=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_xlabel("$e_1$ (Number of trays — rectifying)", fontsize=15, fontweight="bold")
    ax.set_ylabel("$e_2$ (Number of trays — stripping)", fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, which="major", linestyle=":", alpha=0.4, lw=1)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_comparison(
    df_enum: pd.DataFrame,
    traj_a: list[tuple[int, int]],
    traj_b: list[tuple[int, int]],
    *,
    save_path: Path,
    title_a: str,
    title_b: str,
    suptitle: str,
    x_range: range,
    y_range: range,
):
    """Plot two trajectories side-by-side on the same heatmap background."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    obj_grid, feasible_pts, infeasible_pts = build_landscape(
        df_enum, x_range, y_range
    )

    # Shared colour limits from feasible points
    if feasible_pts:
        feas_obj_vals = [v for _, _, v in feasible_pts]
        vmin, vmax = min(feas_obj_vals), max(feas_obj_vals)
    else:
        vmin, vmax = 0, 1

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)

    sc = None
    for ax, traj, subtitle in [(ax_a, traj_a, title_a), (ax_b, traj_b, title_b)]:
        # -- Colored scatter for feasible points (blue-green palette) --
        if feasible_pts:
            feas_x, feas_y, feas_obj = zip(*feasible_pts)
            sc = ax.scatter(
                feas_x, feas_y, c=feas_obj, cmap="GnBu",
                s=250, marker="o", edgecolors="black", linewidths=1,
                zorder=3, label="Feasible", vmin=vmin, vmax=vmax,
            )
        if infeasible_pts:
            inf_x, inf_y = zip(*infeasible_pts)
            ax.scatter(inf_x, inf_y, marker="x", c="red", s=120,
                       linewidths=2.5, zorder=3, label="Infeasible")

        if traj:
            for i in range(len(traj) - 1):
                x0, y0 = traj[i]
                x1, y1 = traj[i + 1]
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color="royalblue", lw=2.5,
                        mutation_scale=18, connectionstyle="arc3,rad=0.1",
                    ),
                    zorder=6,
                )
            traj_x = [p[0] for p in traj]
            traj_y = [p[1] for p in traj]
            ax.scatter(traj_x, traj_y, color="white", edgecolors="royalblue",
                       linewidths=2, s=200, zorder=7)
            for k, (px, py) in enumerate(traj, start=1):
                ax.text(px, py, str(k), fontsize=11, color="royalblue",
                        fontweight="bold", ha="center", va="center", zorder=8)
            ax.scatter([traj[0][0]], [traj[0][1]], color="lime",
                       edgecolors="black", linewidths=2, s=300, zorder=9,
                       marker="o", label="Start")
            ax.text(traj[0][0], traj[0][1], "1", fontsize=11, color="black",
                    fontweight="bold", ha="center", va="center", zorder=10)
            ax.scatter([traj[-1][0]], [traj[-1][1]], color="gold",
                       edgecolors="black", linewidths=2, s=300, zorder=9,
                       marker="*", label="End")

        ax.set_xlim(x_range.start - 0.5, x_range.stop - 0.5)
        ax.set_ylim(y_range.start - 0.5, y_range.stop - 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(axis="both", which="both", direction="in",
                       labelsize=12, width=2, length=5, top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.set_xlabel("$e_1$", fontsize=14, fontweight="bold")
        ax.set_ylabel("$e_2$", fontsize=14, fontweight="bold")
        ax.set_title(subtitle, fontsize=14, fontweight="bold")
        ax.grid(True, which="major", linestyle=":", alpha=0.4, lw=1)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    if sc is not None:
        fig.colorbar(sc, ax=[ax_a, ax_b], fraction=0.02, pad=0.02, label="Objective Value")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot LD-BD / LD-SDA trajectory on enumerated objective heatmap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--date", required=True,
        help="Results date stamp (folder name under results/), e.g. 20260311_ipopt_preprocess_off",
    )
    p.add_argument(
        "--date2", default=None,
        help="Second date stamp for cross-date side-by-side comparison (same init).",
    )
    p.add_argument(
        "--algorithm", default="ldbd", choices=["ldbd", "ldsda"],
        help="Algorithm subfolder. Default: ldbd.",
    )
    p.add_argument(
        "--init", type=int, default=None,
        help="Init index for single-trajectory plot (e.g. 38). If omitted, plots all available inits.",
    )
    p.add_argument(
        "--compare", type=int, nargs=2, metavar=("INIT_A", "INIT_B"),
        help="Compare two initializations side by side, e.g. --compare 1 38.",
    )
    p.add_argument(
        "--enum-csv", default=None,
        help="Path to complete-enumeration CSV. Default: assets/compl_enum_column_17_optimal_baron_bigm.csv",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Override output directory.",
    )
    p.add_argument(
        "--x-range", default="1,16", help="e1 range as start,stop (Python range). Default: 1,16",
    )
    p.add_argument(
        "--y-range", default="1,16", help="e2 range as start,stop (Python range). Default: 1,16",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root = _ld_root()

    # Enumeration data
    if args.enum_csv:
        enum_csv = Path(args.enum_csv).expanduser().resolve()
    else:
        enum_csv = root / "assets" / "compl_enum_column_17_optimal_baron_bigm.csv"
    if not enum_csv.exists():
        print(f"ERROR: Enumeration CSV not found: {enum_csv}", file=sys.stderr)
        return 1
    df_enum = load_enum(enum_csv)

    # Axis ranges
    xr = [int(v) for v in args.x_range.split(",")]
    yr = [int(v) for v in args.y_range.split(",")]
    x_range = range(xr[0], xr[1])
    y_range = range(yr[0], yr[1])

    # Results directory for the primary date
    results_dir = root / "results" / args.date / "column" / args.algorithm
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    algo_upper = args.algorithm.upper()
    n_plotted = 0

    # ---- Mode 1: Compare two initializations from the same date ----
    if args.compare is not None:
        init_a, init_b = args.compare
        traj_a = load_traj(results_dir / f"init_{init_a:03d}" / "traj.csv")
        traj_b = load_traj(results_dir / f"init_{init_b:03d}" / "traj.csv")
        if not traj_a:
            print(f"ERROR: No trajectory for init_{init_a:03d}", file=sys.stderr)
            return 1
        if not traj_b:
            print(f"ERROR: No trajectory for init_{init_b:03d}", file=sys.stderr)
            return 1

        if args.output_dir:
            out_dir = Path(args.output_dir).expanduser().resolve()
        else:
            out_dir = root / "figure" / f"{args.date}_column_traj"

        fname = f"{algo_upper}_init_{init_a:03d}_vs_init_{init_b:03d}.png"
        plot_comparison(
            df_enum, traj_a, traj_b,
            save_path=out_dir / fname,
            title_a=f"init {init_a} ({algo_upper})",
            title_b=f"init {init_b} ({algo_upper})",
            suptitle=f"{algo_upper} — init {init_a} vs init {init_b}  [{args.date}]",
            x_range=x_range,
            y_range=y_range,
        )
        n_plotted += 1

    # ---- Mode 2: Cross-date comparison (same init) ----
    elif args.date2 is not None:
        results_dir2 = root / "results" / args.date2 / "column" / args.algorithm
        if not results_dir2.exists():
            print(f"ERROR: Second results directory not found: {results_dir2}", file=sys.stderr)
            return 1

        if args.output_dir:
            out_dir = Path(args.output_dir).expanduser().resolve()
        else:
            out_dir = root / "figure" / f"{args.date}-vs-{args.date2}_column_traj"

        # Gather init directories
        if args.init is not None:
            init_dirs = [results_dir / f"init_{args.init:03d}"]
        else:
            init_dirs = sorted(
                d for d in results_dir.iterdir()
                if d.is_dir() and d.name.startswith("init_")
            )

        for init_d in init_dirs:
            init_name = init_d.name
            traj = load_traj(init_d / "traj.csv")
            if not traj:
                continue
            traj2 = load_traj(results_dir2 / init_name / "traj.csv")

            fname = f"{algo_upper}_{init_name}_comparison.png"
            plot_comparison(
                df_enum, traj, traj2,
                save_path=out_dir / fname,
                title_a=f"{args.date} ({algo_upper})",
                title_b=f"{args.date2} ({algo_upper})",
                suptitle=f"{algo_upper} {init_name}: {args.date} vs {args.date2}",
                x_range=x_range,
                y_range=y_range,
            )
            n_plotted += 1

    # ---- Mode 3: Single trajectory plots ----
    else:
        if args.output_dir:
            out_dir = Path(args.output_dir).expanduser().resolve()
        else:
            out_dir = root / "figure" / f"{args.date}_column_traj"

        if args.init is not None:
            init_dirs = [results_dir / f"init_{args.init:03d}"]
            if not init_dirs[0].exists():
                print(f"ERROR: Init directory not found: {init_dirs[0]}", file=sys.stderr)
                return 1
        else:
            init_dirs = sorted(
                d for d in results_dir.iterdir()
                if d.is_dir() and d.name.startswith("init_")
            )

        for init_d in init_dirs:
            init_name = init_d.name
            traj = load_traj(init_d / "traj.csv")
            if not traj:
                continue
            fname = f"{algo_upper}_{init_name}_trajectory.png"
            title = f"{algo_upper} {init_name} — {args.date}"
            plot_single(
                df_enum, traj,
                save_path=out_dir / fname,
                title=title,
                x_range=x_range,
                y_range=y_range,
            )
            n_plotted += 1

    print(f"\nDone. {n_plotted} figure(s) generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
