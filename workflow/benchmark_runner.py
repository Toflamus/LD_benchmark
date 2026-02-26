from __future__ import annotations

"""Single-case runner for LD-SDA / LD-BD benchmark experiments.

This module provides:
- `run_gdpopt_case(...)`: runs one (model, algorithm) configuration and writes
    artifacts under a results directory.
- Standardized artifacts: `solver.log`, `traj.csv`, and an appended `summary.csv`.

Notes
-----
Pyomo is imported lazily inside `run_gdpopt_case` so that notebooks can inject
the local Pyomo source tree into `sys.path` at runtime.
"""

import io
import logging
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .result_io import append_row_csv, ensure_dir, write_json, write_traj_csv


_TUPLE_RE = re.compile(r"\((?:\s*\d+\s*,)*\s*\d+\s*\)")


@dataclass(frozen=True)
class RunResult:
    model: str
    algorithm: str
    # Full solver string passed to SolverFactory, e.g. "gdpopt.ldsda".
    algorithm_full: str
    instance: str
    starting_point: tuple[int, ...]
    objective_value: float | None
    wall_time_s: float
    cpu_time_s: float | None
    termination_condition: str | None
    solver_status: str | None
    traj: list[tuple[int, ...]]


def find_ld_benchmark_root(start: Path | None = None) -> Path:
    """Find the LD_benchmark root (folder containing `models/` and `results/`).

    Parameters
    ----------
    start:
        Starting directory for discovery. Defaults to the current working
        directory.

    Returns
    -------
    pathlib.Path
        Path to the LD_benchmark root.
    """
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
    """Create a dedicated file logger for a single benchmark run.

    The logger is configured to write to `log_path` and to avoid stacking
    handlers across repeated notebook executions.
    """
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
    """Extract a trajectory (sequence of integer tuples) from log text.

    This is a best-effort fallback used when the solver object does not expose
    an internal path (e.g., `_path`). The parser searches the full log/console
    text for tuple-like substrings such as `(5, 1)` or `(3, 2, 4)`.
    """
    traj: list[tuple[int, ...]] = []
    for m in _TUPLE_RE.finditer(text):
        raw = m.group(0)
        # raw is like "(5, 1)" -> parse ints
        nums = tuple(int(x.strip()) for x in raw.strip("()").split(","))
        if not traj or traj[-1] != nums:
            traj.append(nums)
    return traj


def _try_get_ldbd_path(solver: Any) -> list[tuple[int, ...]] | None:
    """Return the LDBD internal trajectory path if exposed by the solver."""
    path = getattr(solver, "_path", None)
    if path and isinstance(path, list):
        try:
            return [tuple(map(int, p)) for p in path]
        except Exception:
            return None
    return None


def store_run_config(results_dir: Path, config: dict[str, Any]) -> Path:
    """Persist a per-run configuration snapshot.

    Writes `run_config.json` into `results_dir`.
    """
    path = results_dir / "run_config.json"
    write_json(path, config)
    return path


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
    """Run one benchmark case and write standardized artifacts.

    Parameters
    ----------
    model_name:
        Short model identifier (e.g., "toy", "cstr").
    instance:
        Instance identifier (e.g., "default", "NT25").
    model:
        A fully constructed Pyomo model (typically GDP).
    algorithm:
        GDPopt solver name, e.g. `"gdpopt.ldsda"` or `"gdpopt.ldbd"`.
    starting_point:
        Initial external-variable vector in the solver's expected index space.
    logical_constraint_list:
        Logical constraints used to define the external variables.
    results_dir:
        Directory that receives the per-run `solver.log` and `traj.csv`.
    summary_dir:
        Directory where the aggregated `summary.csv` should be appended.
        Defaults to `results_dir`.
    common_config:
        Dictionary of shared solver configuration used for both algorithms.
    initial_point_key:
        Optional stable identifier for an initial point (useful for sweeps).

    Returns
    -------
    RunResult
        In-memory summary of the run, including objective value and trajectory.
    """

    ensure_dir(results_dir)
    if summary_dir is None:
        summary_dir = results_dir
    ensure_dir(summary_dir)
    algo_tag = algorithm.split(".")[-1].lower()

    run_id = f"{model_name}-{instance}-{algo_tag}"
    log_path = results_dir / "solver.log"
    logger = _setup_run_logger(log_path, name=f"ld_benchmark.{run_id}")

    # Lazy import so notebooks can add the local Pyomo source tree to sys.path.
    import pyomo  # type: ignore
    from pyomo.environ import SolverFactory, value  # type: ignore

    solver = SolverFactory(algorithm)

    solve_kwargs: dict[str, Any] = {
        # common config
        "tee": bool(common_config.get("tee", False)),
        "time_limit": common_config.get("time_limit", None),
        "logger": logger,
        # discrete
        "starting_point": list(starting_point),
        "logical_constraint_list": list(logical_constraint_list),
        "direction_norm": common_config.get("direction_norm", "Linf"),
        # subsolvers
        "nlp_solver": common_config.get("nlp_solver", "ipopt"),
        "nlp_solver_args": common_config.get("nlp_solver_args", {}),
        "mip_solver": common_config.get("mip_solver", "gurobi"),
        "mip_solver_args": common_config.get("mip_solver_args", {}),
    }

    # Optional MINLP subproblem solver configuration.
    # Only pass these kwargs when requested to avoid triggering unexpected-kw
    # errors on GDPopt variants that don't accept MINLP options.
    minlp_solver = common_config.get("minlp_solver", None)
    if minlp_solver is not None and str(minlp_solver).strip():
        solve_kwargs["minlp_solver"] = minlp_solver

        minlp_solver_args = common_config.get("minlp_solver_args", None)
        # Accept dict-like args; ignore None / empty dict.
        if isinstance(minlp_solver_args, dict) and minlp_solver_args:
            solve_kwargs["minlp_solver_args"] = minlp_solver_args

    # Optional: some GDPopt solvers (notably current LDSDA) do not accept these.
    if algo_tag != "ldsda" and common_config.get("separation_solver", None) is not None:
        solve_kwargs["separation_solver"] = common_config.get("separation_solver")
        solve_kwargs["separation_solver_args"] = common_config.get(
            "separation_solver_args", {}
        )

    def _write_run_config() -> None:
        # Keep this small and JSON-friendly.
        cfg = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "instance": instance,
            "algorithm_tag": algo_tag.upper(),
            "algorithm_full": algorithm,
            "initial_point_key": initial_point_key,
            "starting_point": list(map(int, starting_point)),
            "common_config": dict(common_config),
            "solve_kwargs": {
                # Exclude non-serializable / very large objects.
                k: v
                for k, v in solve_kwargs.items()
                if k not in {"logger", "logical_constraint_list"}
            },
            "logical_constraint_count": len(logical_constraint_list),
            "pyomo_version": getattr(pyomo, "__version__", None),
        }
        store_run_config(results_dir, cfg)

    # Write the config before solve so it's available even if solve fails.
    _write_run_config()

    def _solve_with_fallback() -> Any:
        """Solve with best-effort retries when kwargs are rejected.

        Some GDPopt implementations (especially LDSDA) may reject optional
        keywords such as separation/MINLP kwargs. This tries progressively
        simpler kwarg sets rather than failing the whole benchmark.
        """

        nonlocal solve_kwargs

        def _mk_variant(drop: Iterable[str]) -> dict[str, Any]:
            return {k: v for k, v in solve_kwargs.items() if k not in set(drop)}

        variants: list[tuple[str, dict[str, Any]]] = [("full", dict(solve_kwargs))]
        # Remove separation kwargs
        if "separation_solver" in solve_kwargs or "separation_solver_args" in solve_kwargs:
            variants.append(("no_separation", _mk_variant(["separation_solver", "separation_solver_args"])))
        # Remove MINLP args
        if "minlp_solver_args" in solve_kwargs:
            variants.append(("no_minlp_args", _mk_variant(["minlp_solver_args"])))
        # Remove MINLP entirely
        if "minlp_solver" in solve_kwargs or "minlp_solver_args" in solve_kwargs:
            variants.append(("no_minlp", _mk_variant(["minlp_solver", "minlp_solver_args"])))

        seen: set[tuple[str, ...]] = set()
        last_exc: Exception | None = None

        for tag, kwargs in variants:
            key_sig = tuple(sorted(kwargs.keys()))
            if key_sig in seen:
                continue
            seen.add(key_sig)

            if kwargs.keys() != solve_kwargs.keys():
                logger.warning("Retrying solve variant '%s' for solver '%s'.", tag, algorithm)
                # Update persisted config to reflect the retried kwargs.
                solve_kwargs = kwargs
                _write_run_config()

            try:
                return solver.solve(model, **kwargs)
            except (TypeError, ValueError) as e:
                last_exc = e
                logger.warning("Solve variant '%s' failed for '%s': %s", tag, algorithm, e)
                continue

        assert last_exc is not None
        raise last_exc

    # Capture any stray stdout/stderr from subsolvers as well.
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            results = _solve_with_fallback()
        except (TypeError, ValueError) as e:
            # Do not crash the whole suite; record a failed run and return.
            logger.error("Solve failed for '%s' (%s/%s): %s", algorithm, model_name, instance, e)
            wall_time_s = time.perf_counter() - t0

            traj_path = results_dir / "traj.csv"
            write_traj_csv(traj_path, [])

            summary_path = summary_dir / "summary.csv"
            fieldnames = [
                "model",
                "instance",
                "algorithm",
                "algorithm_full",
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
                    "algorithm_full": algorithm,
                    "initial_point_key": initial_point_key,
                    "starting_point": tuple(starting_point),
                    "objective_value": None,
                    "wall_time_s": wall_time_s,
                    "cpu_time_s": None,
                    "termination_condition": "EXCEPTION",
                    "solver_status": "ERROR",
                },
                fieldnames,
            )

            return RunResult(
                model=model_name,
                algorithm=algo_tag.upper(),
                algorithm_full=algorithm,
                instance=instance,
                starting_point=tuple(starting_point),
                objective_value=None,
                wall_time_s=wall_time_s,
                cpu_time_s=None,
                termination_condition="EXCEPTION",
                solver_status="ERROR",
                traj=[],
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

    # Optional: report objective values for specific CSTR points if visited.
    if model_name == "cstr":
        point_info = getattr(getattr(solver, "data_manager", None), "point_info", None)
        if isinstance(point_info, dict):
            for pt in ((2, 1), (2, 2)):
                info = point_info.get(tuple(pt))
                if info is not None:
                    logger.info(
                        "Visited point %s: objective=%s, feasible=%s",
                        pt,
                        info.get("objective"),
                        info.get("feasible"),
                    )

    traj_path = results_dir / "traj.csv"
    write_traj_csv(traj_path, traj)

    summary_path = summary_dir / "summary.csv"
    fieldnames = [
        "model",
        "instance",
        "algorithm",
        "algorithm_full",
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
            "algorithm_full": algorithm,
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
        algorithm_full=algorithm,
        instance=instance,
        starting_point=tuple(starting_point),
        objective_value=obj,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        termination_condition=str(termination) if termination is not None else None,
        solver_status=str(status) if status is not None else None,
        traj=list(traj),
    )
