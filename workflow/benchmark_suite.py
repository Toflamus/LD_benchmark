from __future__ import annotations

"""Model-specific benchmark wrappers for LD-SDA / LD-BD.

This module is the orchestration layer that:
- Imports/constructs each benchmark model.
- Chooses model-specific `logical_constraint_list` and default `starting_point`.
- Calls `workflow.benchmark_runner.run_gdpopt_case` to execute and record runs.

All outputs are written beneath a caller-provided `date_results_dir`, typically
`LD_benchmark/results/<YYYYMMDD>/`.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .benchmark_runner import find_ld_benchmark_root, run_gdpopt_case
from .result_io import ensure_dir


def _add_sys_path(path: Path) -> None:
    """Prepend `path` to `sys.path` if it is not already present."""
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass(frozen=True)
class CommonConfig:
    """Shared GDPopt and subsolver configuration.

    This configuration is intentionally used for *both* `gdpopt.ldsda` and
    `gdpopt.ldbd` so benchmarks compare algorithm behavior under consistent
    solver settings.
    """
    # GDPopt common
    tee: bool = False
    time_limit: int | None = 900
    direction_norm: str = "Linf"

    # Subsolvers
    
    nlp_solver: str = "ipopt"
    nlp_solver_args: dict[str, Any] | None = None
    mip_solver: str = "gurobi"
    mip_solver_args: dict[str, Any] | None = None
    separation_solver: str = "gurobi"
    separation_solver_args: dict[str, Any] | None = None

    # Optional (kept for compatibility with existing scripts)
    # Set a default to allow consistent MINLP subproblem solving across runs.
    minlp_solver: str | None = "gams"
    minlp_solver_args: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert to the keyword-argument dict expected by `run_gdpopt_case`."""
        return {
            "tee": self.tee,
            "time_limit": self.time_limit,
            "direction_norm": self.direction_norm,
            "nlp_solver": self.nlp_solver,
            "nlp_solver_args": self.nlp_solver_args or {},
            "mip_solver": self.mip_solver,
            "mip_solver_args": self.mip_solver_args or {},
            "separation_solver": self.separation_solver,
            "separation_solver_args": self.separation_solver_args or {},
            "minlp_solver": self.minlp_solver,
            "minlp_solver_args": self.minlp_solver_args or {},
        }


def run_toy(
    *,
    date_results_dir: Path,
    common: CommonConfig,
    algorithms: Sequence[str] = ("gdpopt.ldsda", "gdpopt.ldbd"),
    starting_point: Sequence[int] = (5, 1),
) -> None:
    """Run the 2D Linan toy problem for the requested algorithms."""
    root = find_ld_benchmark_root(date_results_dir)
    model_dir = root / "models" / "linan_toy"
    _add_sys_path(model_dir)

    from toy_problem import build_model  # type: ignore

    for alg in algorithms:
        m = build_model()
        algo_dir = ensure_dir(date_results_dir / "toy" / alg.split(".")[-1])
        run_gdpopt_case(
            model_name="toy",
            instance="default",
            model=m,
            algorithm=alg,
            starting_point=starting_point,
            logical_constraint_list=[m.oneY1, m.oneY2],
            results_dir=algo_dir,
            common_config=common.as_dict(),
        )


def run_small_batch(
    *,
    date_results_dir: Path,
    common: CommonConfig,
    algorithms: Sequence[str] = ("gdpopt.ldsda", "gdpopt.ldbd"),
    starting_point: Sequence[int] = (3, 3, 3),
) -> None:
    """Run the small-batch process benchmark for the requested algorithms."""
    root = find_ld_benchmark_root(date_results_dir)
    model_dir = root / "models" / "small_batch"
    _add_sys_path(model_dir)

    from gdp_small_batch import build_model  # type: ignore

    for alg in algorithms:
        m = build_model()
        algo_dir = ensure_dir(date_results_dir / "small_batch" / alg.split(".")[-1])
        run_gdpopt_case(
            model_name="small_batch",
            instance="default",
            model=m,
            algorithm=alg,
            starting_point=starting_point,
            logical_constraint_list=[m.lim["mixer"], m.lim["reactor"], m.lim["centrifuge"]],
            results_dir=algo_dir,
            common_config=common.as_dict(),
        )


def run_cstr(
    *,
    date_results_dir: Path,
    common: CommonConfig,
    NT_list: Sequence[int] = (5, 10, 15, 20, 25, 30),
    algorithms: Sequence[str] = ("gdpopt.ldsda", "gdpopt.ldbd"),
    starting_point: Sequence[int] = (1, 1),
) -> None:
    """Run the CSTR benchmark for a list of stage counts (`NT_list`)."""
    root = find_ld_benchmark_root(date_results_dir)
    model_dir = root / "models" / "cstr_testing"
    _add_sys_path(model_dir)

    from cstr import build_model  # type: ignore

    for NT in NT_list:
        for alg in algorithms:
            m = build_model(NT)
            algo_dir = ensure_dir(date_results_dir / "cstr" / f"NT{NT}" / alg.split(".")[-1])
            run_gdpopt_case(
                model_name="cstr",
                instance=f"NT{NT}",
                model=m,
                algorithm=alg,
                starting_point=starting_point,
                logical_constraint_list=[m.one_unreacted_feed, m.one_recycle],
                results_dir=algo_dir,
                common_config=common.as_dict(),
            )


def run_column_random_init(
    *,
    date_results_dir: Path,
    common: CommonConfig,
    algorithms: Sequence[str] = ("gdpopt.ldsda", "gdpopt.ldbd"),
    initial_point_keys: Iterable[int] | None = None,
) -> None:
    """Run the column benchmark over a set of randomized initial points.

    Parameters
    ----------
    initial_point_keys:
        Keys from `models/column/column_initial_test.POINT_DICT`.
        If None, runs all available initializations.

    Notes
    -----
    Each initialization writes its own `solver.log` and `traj.csv` under
    `column/<algo>/init_###/` while all runs append to a single
    `column/<algo>/summary.csv`.
    """

    root = find_ld_benchmark_root(date_results_dir)
    model_dir = root / "models" / "column"
    _add_sys_path(model_dir)

    from gdp_column import build_column  # type: ignore
    from initialize import initialize  # type: ignore
    from column_initial_test import POINT_DICT, tray_point_to_starting_point  # type: ignore

    # Default: all available keys
    keys = list(POINT_DICT.keys()) if initial_point_keys is None else list(initial_point_keys)

    model_args = {"min_trays": 8, "max_trays": 17, "xD": 0.95, "xB": 0.95}

    for key in keys:
        tray_point = POINT_DICT[key]
        starting_point = tray_point_to_starting_point(tray_point)

        for alg in algorithms:
            m = build_column(**model_args)
            initialize(m)
            algo_root = ensure_dir(date_results_dir / "column" / alg.split(".")[-1])
            run_dir = ensure_dir(algo_root / f"init_{key:03d}")
            run_gdpopt_case(
                model_name="column",
                instance="NT17",
                model=m,
                algorithm=alg,
                starting_point=starting_point,
                logical_constraint_list=[m.one_reflux, m.one_boilup],
                results_dir=run_dir,
                summary_dir=algo_root,
                common_config=common.as_dict(),
                initial_point_key=key,
            )


def run_all(
    *,
    date_results_dir: Path,
    common: CommonConfig,
    cstr_NT_list: Sequence[int] = (5, 10, 15, 20, 25, 30),
    column_keys: Iterable[int] | None = None,
) -> None:
    """Run toy, small-batch, CSTR sweep, and column random-init benchmarks."""
    run_toy(date_results_dir=date_results_dir, common=common)
    run_small_batch(date_results_dir=date_results_dir, common=common)
    run_cstr(date_results_dir=date_results_dir, common=common, NT_list=cstr_NT_list)
    run_column_random_init(date_results_dir=date_results_dir, common=common, initial_point_keys=column_keys)
