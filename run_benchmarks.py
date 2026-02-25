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

Run everything for only one algorithm (e.g., LDSDA):
    python3 run_benchmarks.py --test all --algorithms gdpopt.ldsda
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _add_sys_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def _ld_root() -> Path:
    # This file lives in LD_benchmark/
    return Path(__file__).resolve().parent


def _default_date_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config JSON root must be an object: {path}")
    return data


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def _get(d: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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
        "--config",
        default=None,
        help=(
            "Path to a JSON config file. If omitted, uses LD_benchmark/benchmark_config.json when present. "
            "CLI flags override values loaded from JSON."
        ),
    )

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

    # Load JSON defaults (optional)
    config_path: Path | None
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        candidate = ld_root / "benchmark_config.json"
        config_path = candidate if candidate.exists() else None

    cfg: dict[str, Any] = {}
    if config_path is not None:
        cfg = _load_json(config_path)

    # Resolve run/test defaults
    cfg_test = _get(cfg, ["run", "test"], None)
    cfg_algs = _get(cfg, ["run", "algorithms"], None)
    cfg_nt = _get(cfg, ["run", "cstr_nt_list"], None)
    cfg_col_keys = _get(cfg, ["run", "column_keys"], None)

    # If user didn't pass CLI values, take from config
    if args.test == "all" and cfg_test in ("toy", "small_batch", "cstr", "column", "all"):
        args.test = cfg_test

    if args.algorithms == ["gdpopt.ldsda", "gdpopt.ldbd"] and isinstance(cfg_algs, list) and cfg_algs:
        args.algorithms = [str(x) for x in cfg_algs]

    if args.nt is None and isinstance(cfg_nt, list) and cfg_nt:
        args.nt = [str(int(x)) for x in cfg_nt]

    if args.column_keys is None and (cfg_col_keys is None or isinstance(cfg_col_keys, list)):
        if cfg_col_keys is None:
            args.column_keys = None
        else:
            args.column_keys = [str(int(x)) for x in cfg_col_keys]

    # Resolve outputs defaults
    cfg_date = _get(cfg, ["outputs", "date"], None)
    cfg_results_dir = _get(cfg, ["outputs", "results_dir"], None)
    cfg_clear = _get(cfg, ["outputs", "clear"], None)

    if args.date == _default_date_stamp() and isinstance(cfg_date, str) and cfg_date:
        if cfg_date.lower() == "today":
            args.date = _default_date_stamp()
        else:
            args.date = cfg_date

    if args.results_dir is None and cfg_results_dir:
        args.results_dir = str(cfg_results_dir)

    if args.clear is False and isinstance(cfg_clear, bool):
        args.clear = cfg_clear

    # Resolve common config defaults
    def _cfg_common(name: str, default: Any) -> Any:
        return _get(cfg, ["common", name], default)

    if args.tee is False:
        args.tee = bool(_cfg_common("tee", args.tee))

    if args.time_limit == 900:
        args.time_limit = int(_cfg_common("time_limit", args.time_limit))

    if args.direction_norm == "Linf":
        args.direction_norm = str(_cfg_common("direction_norm", args.direction_norm))

    if args.nlp_solver == "ipopt":
        args.nlp_solver = str(_cfg_common("nlp_solver", args.nlp_solver))

    if args.mip_solver == "gurobi":
        args.mip_solver = str(_cfg_common("mip_solver", args.mip_solver))

    if args.separation_solver == "gurobi":
        args.separation_solver = str(_cfg_common("separation_solver", args.separation_solver))

    if args.minlp_solver == "gams":
        args.minlp_solver = str(_cfg_common("minlp_solver", args.minlp_solver))

    minlp_solver_args = _cfg_common("minlp_solver_args", None)
    if minlp_solver_args is not None and not isinstance(minlp_solver_args, dict):
        raise TypeError(
            "common.minlp_solver_args must be an object/dict (or null) in the JSON config; "
            f"got {type(minlp_solver_args).__name__}"
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
        minlp_solver_args=minlp_solver_args,
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

    # Persist the resolved configuration for reproducibility.
    resolved_cfg = {
        "schema_version": 1,
        "timestamp_local": datetime.now().isoformat(),
        "source_config_path": str(config_path) if config_path is not None else None,
        "loaded_config": cfg,
        "resolved": {
            "test": args.test,
            "algorithms": list(algorithms),
            "date": str(args.date),
            "results_day_dir": str(results_day_dir),
            "clear": bool(args.clear),
            "cstr_nt_list": nt_list,
            "column_keys": column_keys,
            "common": common.as_dict(),
        },
    }
    _write_json(results_day_dir / "benchmark_config.resolved.json", resolved_cfg)

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
            algorithms=algorithms,
        )
    else:
        raise AssertionError(f"Unhandled test: {args.test}")

    if config_path is not None:
        print("Config:", config_path)
    print("Done. Results written under:", results_day_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
