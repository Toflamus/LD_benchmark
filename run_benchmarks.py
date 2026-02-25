#!/usr/bin/env python3
"""Command-line runner for LD_benchmark.

This script mirrors `workflow/ModelTestingScript.ipynb`, but is easier to use on
Linux/CI and supports running *one* benchmark at a time.

Examples
--------
Run toy for both algorithms:
    python3 run_benchmarks.py --test toy

Run only LDSDA for CSTR NT=25:
    python3 run_benchmarks.py --test cstr --algorithms gdpopt.ldsda --nt 25

Run column for keys 1..10 and clear today's column outputs first:
    python3 run_benchmarks.py --test column --column-keys 1-10 --clear

Run everything and clear today's whole results folder first:
    python3 run_benchmarks.py --test all --clear
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _add_sys_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def _ld_root() -> Path:
    # This file lives in LD_benchmark/
    return Path(__file__).resolve().parent


def _default_date_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def _parse_int_list(values: list[str] | None) -> list[int] | None:
    """Parse int lists from argv.

    Supported formats:
    - repeated ints:  --nt 5 10 25
    - comma list:     --nt 5,10,25
    - range token:    --column-keys 1-10

    Returns None if `values` is None.
    """

    if values is None:
        return None

    tokens: list[str] = []
    for v in values:
        tokens.extend([t for t in v.split(",") if t.strip()])

    out: list[int] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if "-" in t:
            a_str, b_str = t.split("-", 1)
            a, b = int(a_str), int(b_str)
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(a, b - 1, -1)))
        else:
            out.append(int(t))

    # De-dup while preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _clear_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LD_benchmark experiments.")

    p.add_argument(
        "--test",
        choices=("toy", "small_batch", "cstr", "column", "all"),
        default="all",
        help="Which benchmark to run.",
    )

    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["gdpopt.ldsda", "gdpopt.ldbd"],
        help="One or two full GDPopt solver names.",
    )

    p.add_argument(
        "--date",
        default=_default_date_stamp(),
        help="Results date stamp (YYYYMMDD). Default: today.",
    )

    p.add_argument(
        "--results-dir",
        default=None,
        help="Override results output directory. Default: LD_benchmark/results/<date>/",
    )

    p.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Clear existing outputs before running. If --test=all, clears the whole date folder; "
            "otherwise clears only that test subfolder."
        ),
    )

    # Common config
    p.add_argument("--tee", action="store_true", help="Enable solver tee output.")
    p.add_argument("--time-limit", type=int, default=900, help="Time limit in seconds.")
    p.add_argument("--direction-norm", default="Linf", help="Direction norm passed to GDPopt.")
    p.add_argument("--nlp-solver", default="ipopt", help="NLP solver name.")
    p.add_argument("--mip-solver", default="gurobi", help="MIP solver name.")
    p.add_argument("--separation-solver", default="gurobi", help="Separation solver name.")
    p.add_argument("--minlp-solver", default="gams", help="MINLP solver name (or empty to disable).")

    # CSTR
    p.add_argument(
        "--nt",
        nargs="+",
        default=None,
        help="CSTR stage counts. Examples: --nt 5 10 25  or  --nt 5,10,25",
    )

    # Column
    p.add_argument(
        "--column-keys",
        nargs="+",
        default=None,
        help="Column initialization keys. Examples: --column-keys 1-10  or  --column-keys 1,2,3",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ld_root = _ld_root()
    _add_sys_path(ld_root)

    # Optional: prefer local Pyomo source tree if present (repo layout: <workspace>/pyomo/pyomo/...)
    pyomo_src = ld_root.parent / "pyomo"
    if (pyomo_src / "pyomo").is_dir():
        _add_sys_path(pyomo_src)

    from workflow.benchmark_suite import (
        CommonConfig,
        run_all,
        run_column_random_init,
        run_cstr,
        run_small_batch,
        run_toy,
    )

    algorithms = tuple(args.algorithms)

    # Resolve output directory
    if args.results_dir:
        results_day_dir = Path(args.results_dir).expanduser().resolve()
    else:
        results_day_dir = (ld_root / "results" / str(args.date)).resolve()

    results_day_dir.mkdir(parents=True, exist_ok=True)

    common = CommonConfig(
        tee=bool(args.tee),
        time_limit=int(args.time_limit) if args.time_limit else None,
        direction_norm=str(args.direction_norm),
        nlp_solver=str(args.nlp_solver),
        mip_solver=str(args.mip_solver),
        separation_solver=str(args.separation_solver),
        minlp_solver=(str(args.minlp_solver) if str(args.minlp_solver).strip() else None),
    )

    # Parse optional lists
    nt_list = _parse_int_list(args.nt)
    if nt_list is None:
        nt_list = [5, 10, 15, 20, 25, 30]

    column_keys = _parse_int_list(args.column_keys)

    # Optional clear
    if args.clear:
        if args.test == "all":
            _clear_path(results_day_dir)
            results_day_dir.mkdir(parents=True, exist_ok=True)
        else:
            _clear_path(results_day_dir / args.test)

    # Run
    if args.test == "toy":
        run_toy(date_results_dir=results_day_dir, common=common, algorithms=algorithms)
    elif args.test == "small_batch":
        run_small_batch(date_results_dir=results_day_dir, common=common, algorithms=algorithms)
    elif args.test == "cstr":
        run_cstr(
            date_results_dir=results_day_dir,
            common=common,
            NT_list=nt_list,
            algorithms=algorithms,
        )
    elif args.test == "column":
        run_column_random_init(
            date_results_dir=results_day_dir,
            common=common,
            algorithms=algorithms,
            initial_point_keys=column_keys,
        )
    elif args.test == "all":
        run_all(
            date_results_dir=results_day_dir,
            common=common,
            cstr_NT_list=nt_list,
            column_keys=column_keys,
        )
    else:
        raise AssertionError(f"Unhandled test: {args.test}")

    print("Done. Results written under:", results_day_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
