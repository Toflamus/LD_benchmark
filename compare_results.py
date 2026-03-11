#!/usr/bin/env python3
"""Compare column benchmark results between two result directories.

Creates side-by-side comparison plots and saves them to
    figure/<dir1>-vs-<dir2>/

Usage
-----
    python compare_results.py 20260310 20260310_preprocess_on
    python compare_results.py 20260310 20260310_preprocess_on --algorithms gdpopt.ldsda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


def _ld_root() -> Path:
    return Path(__file__).resolve().parent


def _normalize_algo_tag(name: str) -> str:
    x = str(name).strip().lower()
    if x in {"ldsda", "gdpopt.ldsda"}:
        return "LDSDA"
    if x in {"ldbd", "gdpopt.ldbd"}:
        return "LDBD"
    return str(name).strip().upper()


def _algo_folder(tag: str) -> str:
    return tag.lower()


def load_column_summary(results_dir: Path) -> pd.DataFrame:
    """Load all column summary.csv files from a results directory."""
    col_dir = results_dir / "column"
    if not col_dir.exists():
        return pd.DataFrame()

    dfs = []
    for csv_path in sorted(col_dir.rglob("summary.csv")):
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["algorithm_tag"] = combined["algorithm"].map(_normalize_algo_tag)
    # Mark feasibility: objective_value is NaN means infeasible
    combined["feasible"] = combined["objective_value"].notna()
    return combined


def load_run_configs(results_dir: Path) -> dict:
    """Load the resolved benchmark config if available."""
    cfg_path = results_dir / "benchmark_config.resolved.json"
    if cfg_path.exists():
        import json
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def plot_objective_comparison(df1, df2, label1, label2, algo_tag, save_dir):
    """Scatter plot: objective value vs initial_point_key for both runs."""
    s1 = df1[df1["algorithm_tag"] == algo_tag].sort_values("initial_point_key")
    s2 = df2[df2["algorithm_tag"] == algo_tag].sort_values("initial_point_key")

    if s1.empty and s2.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot feasible points
    f1 = s1[s1["feasible"]]
    f2 = s2[s2["feasible"]]
    ax.scatter(f1["initial_point_key"], f1["objective_value"],
               marker="o", color="blue", s=60, label=f"{label1} (feasible)", alpha=0.7)
    ax.scatter(f2["initial_point_key"], f2["objective_value"],
               marker="s", color="red", s=60, label=f"{label2} (feasible)", alpha=0.7)

    # Plot infeasible points as markers on the x-axis area
    inf1 = s1[~s1["feasible"]]
    inf2 = s2[~s2["feasible"]]
    if not inf1.empty:
        y_min = ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else 0
        ax.scatter(inf1["initial_point_key"], [y_min] * len(inf1),
                   marker="x", color="blue", s=80, label=f"{label1} (infeasible)", zorder=5)
    if not inf2.empty:
        y_min = ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else 0
        ax.scatter(inf2["initial_point_key"], [y_min] * len(inf2),
                   marker="x", color="red", s=80, label=f"{label2} (infeasible)", zorder=5)

    ax.set_xlabel("Initial Point Key", fontsize=12)
    ax.set_ylabel("Objective Value", fontsize=12)
    ax.set_title(f"Column {algo_tag}: Objective Value Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_dir / f"column_{algo_tag.lower()}_objective.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_comparison(df1, df2, label1, label2, algo_tag, save_dir):
    """Scatter plot: CPU time vs initial_point_key for both runs."""
    s1 = df1[df1["algorithm_tag"] == algo_tag].sort_values("initial_point_key")
    s2 = df2[df2["algorithm_tag"] == algo_tag].sort_values("initial_point_key")

    if s1.empty and s2.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    f1 = s1[s1["feasible"]]
    f2 = s2[s2["feasible"]]
    inf1 = s1[~s1["feasible"]]
    inf2 = s2[~s2["feasible"]]

    ax.scatter(f1["initial_point_key"], f1["cpu_time_s"],
               marker="o", color="blue", s=60, label=f"{label1} (feasible)", alpha=0.7)
    ax.scatter(f2["initial_point_key"], f2["cpu_time_s"],
               marker="s", color="red", s=60, label=f"{label2} (feasible)", alpha=0.7)

    if not inf1.empty:
        ax.scatter(inf1["initial_point_key"], inf1["cpu_time_s"],
                   marker="x", color="blue", s=80, label=f"{label1} (infeasible)", zorder=5)
    if not inf2.empty:
        ax.scatter(inf2["initial_point_key"], inf2["cpu_time_s"],
                   marker="x", color="red", s=80, label=f"{label2} (infeasible)", zorder=5)

    ax.set_xlabel("Initial Point Key", fontsize=12)
    ax.set_ylabel("CPU Time [s]", fontsize=12)
    ax.set_title(f"Column {algo_tag}: CPU Time Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_dir / f"column_{algo_tag.lower()}_cputime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feasibility_comparison(df1, df2, label1, label2, algo_tag, save_dir):
    """Bar chart showing feasibility status per initial point."""
    s1 = df1[df1["algorithm_tag"] == algo_tag].set_index("initial_point_key")["feasible"]
    s2 = df2[df2["algorithm_tag"] == algo_tag].set_index("initial_point_key")["feasible"]

    all_keys = sorted(set(s1.index) | set(s2.index))
    if not all_keys:
        return

    # Categories: both feasible, only run1 feasible, only run2 feasible, both infeasible
    categories = []
    for k in all_keys:
        f1 = s1.get(k, False)
        f2 = s2.get(k, False)
        if f1 and f2:
            categories.append("both_feasible")
        elif f1 and not f2:
            categories.append("only_run1")
        elif not f1 and f2:
            categories.append("only_run2")
        else:
            categories.append("both_infeasible")

    color_map = {
        "both_feasible": "green",
        "only_run1": "blue",
        "only_run2": "red",
        "both_infeasible": "gray",
    }
    colors = [color_map[c] for c in categories]

    fig, ax = plt.subplots(figsize=(16, 4))
    bars = ax.bar(range(len(all_keys)), [1] * len(all_keys), color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels(all_keys, rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("Initial Point Key", fontsize=12)
    ax.set_title(f"Column {algo_tag}: Feasibility Comparison", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", label="Both feasible"),
        Patch(facecolor="blue", label=f"Only {label1} feasible"),
        Patch(facecolor="red", label=f"Only {label2} feasible"),
        Patch(facecolor="gray", label="Both infeasible"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_dir / f"column_{algo_tag.lower()}_feasibility.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_difference(df1, df2, label1, label2, algo_tag, save_dir):
    """Bar chart: time difference (run2 - run1) for each init point."""
    s1 = df1[df1["algorithm_tag"] == algo_tag].set_index("initial_point_key")
    s2 = df2[df2["algorithm_tag"] == algo_tag].set_index("initial_point_key")

    common_keys = sorted(set(s1.index) & set(s2.index))
    # Only compare points that are feasible in both
    keys = [k for k in common_keys if s1.loc[k, "feasible"] and s2.loc[k, "feasible"]]
    if not keys:
        return

    diffs = [s2.loc[k, "cpu_time_s"] - s1.loc[k, "cpu_time_s"] for k in keys]
    colors = ["red" if d > 0 else "green" for d in diffs]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(keys)), diffs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=7)
    ax.set_xlabel("Initial Point Key", fontsize=12)
    ax.set_ylabel(f"Time Diff [s] ({label2} - {label1})", fontsize=12)
    ax.set_title(f"Column {algo_tag}: CPU Time Difference (positive = {label2} slower)",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_dir / f"column_{algo_tag.lower()}_time_diff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison_csv(df1, df2, label1, label2, algo_tag, save_dir):
    """Save merged comparison data as CSV."""
    s1 = df1[df1["algorithm_tag"] == algo_tag][
        ["initial_point_key", "starting_point", "objective_value", "cpu_time_s",
         "termination_condition", "feasible"]
    ].copy()
    s2 = df2[df2["algorithm_tag"] == algo_tag][
        ["initial_point_key", "starting_point", "objective_value", "cpu_time_s",
         "termination_condition", "feasible"]
    ].copy()

    merged = pd.merge(
        s1, s2,
        on=["initial_point_key", "starting_point"],
        how="outer",
        suffixes=(f"_{label1}", f"_{label2}"),
    )
    merged = merged.sort_values("initial_point_key")
    merged.to_csv(save_dir / f"column_{algo_tag.lower()}_comparison.csv", index=False)
    return merged


def print_summary(df1, df2, label1, label2, algo_tag):
    """Print a text summary of the comparison."""
    s1 = df1[df1["algorithm_tag"] == algo_tag]
    s2 = df2[df2["algorithm_tag"] == algo_tag]

    n1_feas = s1["feasible"].sum()
    n2_feas = s2["feasible"].sum()
    n1_total = len(s1)
    n2_total = len(s2)

    print(f"\n{'='*60}")
    print(f"  {algo_tag} Summary")
    print(f"{'='*60}")
    print(f"  {label1}: {n1_feas}/{n1_total} feasible")
    print(f"  {label2}: {n2_feas}/{n2_total} feasible")

    # Find differing feasibility
    idx1 = s1.set_index("initial_point_key")["feasible"]
    idx2 = s2.set_index("initial_point_key")["feasible"]
    common = sorted(set(idx1.index) & set(idx2.index))

    only1 = [k for k in common if idx1[k] and not idx2[k]]
    only2 = [k for k in common if not idx1[k] and idx2[k]]

    if only1:
        print(f"\n  Feasible only in {label1} (not {label2}):")
        for k in only1:
            sp = s1[s1["initial_point_key"] == k]["starting_point"].iloc[0]
            print(f"    init_{k:03d}: {sp}")

    if only2:
        print(f"\n  Feasible only in {label2} (not {label1}):")
        for k in only2:
            sp = s2[s2["initial_point_key"] == k]["starting_point"].iloc[0]
            print(f"    init_{k:03d}: {sp}")

    # Time comparison for commonly feasible points
    both_feas = [k for k in common if idx1[k] and idx2[k]]
    if both_feas:
        t1 = s1.set_index("initial_point_key").loc[both_feas, "cpu_time_s"]
        t2 = s2.set_index("initial_point_key").loc[both_feas, "cpu_time_s"]
        print(f"\n  Time stats (both feasible, n={len(both_feas)}):")
        print(f"    {label1}: mean={t1.mean():.1f}s, median={t1.median():.1f}s, total={t1.sum():.1f}s")
        print(f"    {label2}: mean={t2.mean():.1f}s, median={t2.median():.1f}s, total={t2.sum():.1f}s")
        ratio = t2.mean() / t1.mean() if t1.mean() > 0 else float("inf")
        print(f"    Ratio ({label2}/{label1}): {ratio:.2f}x")

    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare column benchmark results between two runs.")
    parser.add_argument("dir1", help="First results directory name (e.g. 20260310)")
    parser.add_argument("dir2", help="Second results directory name (e.g. 20260310_preprocess_on)")
    parser.add_argument("--algorithms", nargs="+", default=["gdpopt.ldsda", "gdpopt.ldbd"],
                        help="Algorithms to compare.")
    parser.add_argument("--results-base", default=None,
                        help="Base results directory. Default: LD_benchmark/results/")
    args = parser.parse_args(argv)

    ld_root = _ld_root()
    results_base = Path(args.results_base) if args.results_base else ld_root / "results"

    dir1 = results_base / args.dir1
    dir2 = results_base / args.dir2

    if not dir1.exists():
        print(f"Error: {dir1} does not exist.")
        return 1
    if not dir2.exists():
        print(f"Error: {dir2} does not exist.")
        return 1

    label1 = args.dir1
    label2 = args.dir2

    # Output directory
    save_dir = ld_root / "figure" / f"{args.dir1}-vs-{args.dir2}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df1 = load_column_summary(dir1)
    df2 = load_column_summary(dir2)

    if df1.empty:
        print(f"Warning: No column summary data found in {dir1}")
    if df2.empty:
        print(f"Warning: No column summary data found in {dir2}")

    selected = [_normalize_algo_tag(a) for a in args.algorithms]

    for algo_tag in selected:
        has1 = not df1.empty and algo_tag in df1["algorithm_tag"].values
        has2 = not df2.empty and algo_tag in df2["algorithm_tag"].values

        if not has1 and not has2:
            print(f"Skipping {algo_tag}: no data in either run.")
            continue

        sub1 = df1[df1["algorithm_tag"] == algo_tag] if has1 else pd.DataFrame(columns=df1.columns if not df1.empty else [])
        sub2 = df2[df2["algorithm_tag"] == algo_tag] if has2 else pd.DataFrame(columns=df2.columns if not df2.empty else [])

        # Print summary
        if has1 and has2:
            print_summary(df1, df2, label1, label2, algo_tag)

        # Generate plots
        plot_objective_comparison(df1 if has1 else pd.DataFrame(), df2 if has2 else pd.DataFrame(),
                                 label1, label2, algo_tag, save_dir)
        plot_time_comparison(df1 if has1 else pd.DataFrame(), df2 if has2 else pd.DataFrame(),
                             label1, label2, algo_tag, save_dir)
        plot_feasibility_comparison(df1 if has1 else pd.DataFrame(), df2 if has2 else pd.DataFrame(),
                                    label1, label2, algo_tag, save_dir)
        plot_time_difference(df1 if has1 else pd.DataFrame(), df2 if has2 else pd.DataFrame(),
                             label1, label2, algo_tag, save_dir)

        # Save comparison CSV
        if has1 and has2:
            save_comparison_csv(df1, df2, label1, label2, algo_tag, save_dir)

    print(f"Figures and data saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
