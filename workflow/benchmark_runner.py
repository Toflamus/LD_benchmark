from __future__ import annotations

import io
import logging
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .result_io import append_row_csv, ensure_dir, write_traj_csv


_TUPLE_RE = re.compile(r"\((?:\s*\d+\s*,)*\s*\d+\s*\)")


@dataclass(frozen=True)
class RunResult:
    model: str
    algorithm: str
    instance: str
    starting_point: tuple[int, ...]
    objective_value: float | None
    wall_time_s: float
    cpu_time_s: float | None
    termination_condition: str | None
    solver_status: str | None
    traj: list[tuple[int, ...]]


def find_ld_benchmark_root(start: Path | None = None) -> Path:
    """Find the LD_benchmark root (folder containing models/ and results/)."""
    if start is None:
        start = Path.cwd()

    # 1) If cwd is already the root
    if (start / "models").is_dir() and (start / "results").is_dir():
        return start

    # 2) If workspace root contains LD_benchmark/
    candidate = start / "LD_benchmark"
    if (candidate / "models").is_dir() and (candidate / "results").is_dir():
        return candidate

    # 3) Walk up
    for parent in [start] + list(start.parents):
        if (parent / "models").is_dir() and (parent / "results").is_dir():
            return parent
        candidate = parent / "LD_benchmark"
        if (candidate / "models").is_dir() and (candidate / "results").is_dir():
            return candidate

    raise FileNotFoundError("Could not locate LD_benchmark root from: %s" % start)


def _setup_run_logger(log_path: Path, name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid stacking handlers across repeated runs in a notebook.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _parse_traj_from_log_text(text: str) -> list[tuple[int, ...]]:
    traj: list[tuple[int, ...]] = []
    for m in _TUPLE_RE.finditer(text):
        raw = m.group(0)
        # raw is like "(5, 1)" -> parse ints
        nums = tuple(int(x.strip()) for x in raw.strip("()").split(","))
        if not traj or traj[-1] != nums:
            traj.append(nums)
    return traj


def _try_get_ldbd_path(solver: Any) -> list[tuple[int, ...]] | None:
    path = getattr(solver, "_path", None)
    if path and isinstance(path, list):
        try:
            return [tuple(map(int, p)) for p in path]
        except Exception:
            return None
    return None


def run_gdpopt_case(
    *,
    model_name: str,
    instance: str,
    model: Any,
    algorithm: str,
    starting_point: Sequence[int],
    logical_constraint_list: Sequence[Any],
    results_dir: Path,
    summary_dir: Path | None = None,
    common_config: dict[str, Any],
    initial_point_key: int | None = None,
) -> RunResult:
    """Run a single (model, algorithm) benchmark and write log/traj/summary files."""

    ensure_dir(results_dir)
    if summary_dir is None:
        summary_dir = results_dir
    ensure_dir(summary_dir)
    algo_tag = algorithm.split(".")[-1].lower()

    run_id = f"{model_name}-{instance}-{algo_tag}"
    log_path = results_dir / "solver.log"
    logger = _setup_run_logger(log_path, name=f"ld_benchmark.{run_id}")

    # Lazy import so notebooks can add the local Pyomo source tree to sys.path.
    from pyomo.environ import SolverFactory, value  # type: ignore

    solver = SolverFactory(algorithm)

    # Capture any stray stdout/stderr from subsolvers as well.
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf), redirect_stderr(buf):
        results = solver.solve(
            model,
            # common config
            tee=bool(common_config.get("tee", False)),
            time_limit=common_config.get("time_limit", None),
            logger=logger,
            # discrete
            starting_point=list(starting_point),
            logical_constraint_list=list(logical_constraint_list),
            direction_norm=common_config.get("direction_norm", "Linf"),
            # solvers
            nlp_solver=common_config.get("nlp_solver", "ipopt"),
            nlp_solver_args=common_config.get("nlp_solver_args", {}),
            mip_solver=common_config.get("mip_solver", "gurobi"),
            mip_solver_args=common_config.get("mip_solver_args", {}),
            separation_solver=common_config.get("separation_solver", "gurobi"),
            separation_solver_args=common_config.get("separation_solver_args", {}),
            minlp_solver=common_config.get("minlp_solver", None),
            minlp_solver_args=common_config.get("minlp_solver_args", {}),
        )
    wall_time_s = time.perf_counter() - t0

    # Objective
    try:
        obj = float(value(model.obj))
    except Exception:
        obj = None

    # CPU time (not always populated)
    cpu_time_s = getattr(getattr(results, "solver", None), "user_time", None)

    termination = getattr(getattr(results, "solver", None), "termination_condition", None)
    status = getattr(getattr(results, "solver", None), "status", None)

    # Trajectory: prefer LDBD internal path if available; otherwise parse log.
    traj = _try_get_ldbd_path(solver)
    if traj is None:
        # Combine explicit log + captured stdout/stderr buffer.
        log_text = ""
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        log_text = log_text + "\n" + buf.getvalue()
        traj = _parse_traj_from_log_text(log_text)

    traj_path = results_dir / "traj.csv"
    write_traj_csv(traj_path, traj)

    summary_path = summary_dir / "summary.csv"
    fieldnames = [
        "model",
        "instance",
        "algorithm",
        "initial_point_key",
        "starting_point",
        "objective_value",
        "wall_time_s",
        "cpu_time_s",
        "termination_condition",
        "solver_status",
    ]
    append_row_csv(
        summary_path,
        {
            "model": model_name,
            "instance": instance,
            "algorithm": algo_tag.upper(),
            "initial_point_key": initial_point_key,
            "starting_point": tuple(starting_point),
            "objective_value": obj,
            "wall_time_s": wall_time_s,
            "cpu_time_s": cpu_time_s,
            "termination_condition": str(termination) if termination is not None else None,
            "solver_status": str(status) if status is not None else None,
        },
        fieldnames,
    )

    return RunResult(
        model=model_name,
        algorithm=algo_tag.upper(),
        instance=instance,
        starting_point=tuple(starting_point),
        objective_value=obj,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        termination_condition=str(termination) if termination is not None else None,
        solver_status=str(status) if status is not None else None,
        traj=list(traj),
    )
