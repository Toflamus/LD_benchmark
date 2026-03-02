import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pyomo.contrib.gdpopt.discrete_search_enums import SearchPhase
from pyomo.contrib.gdpopt.ldbd import GDP_LDBD_Solver

from toy_problem import build_model


@dataclass
class SeparationRecord:
    iteration: int
    anchor: tuple
    sep_obj: float
    alpha: float
    p_vals: tuple


@dataclass
class MasterRecord:
    iteration: int
    point: tuple
    z_lb: float | None


class LDBDSeparationLogger(GDP_LDBD_Solver):
    def __init__(self):
        super().__init__()
        self.separation_records = []
        self.master_records = []

    def _solve_separation_lp(self, anchor_point, config):
        p_vals, alpha_val = super()._solve_separation_lp(anchor_point, config)
        if p_vals is None:
            return p_vals, alpha_val

        anchor_point = tuple(anchor_point)
        sep_obj = sum(p_vals[i] * anchor_point[i] for i in range(len(p_vals))) + alpha_val
        self.separation_records.append(
            SeparationRecord(
                iteration=int(getattr(self, "iteration", 0)),
                anchor=anchor_point,
                sep_obj=float(sep_obj),
                alpha=float(alpha_val),
                p_vals=tuple(float(v) for v in p_vals),
            )
        )
        return p_vals, alpha_val

    def _solve_master(self, config):
        z_lb, next_point = super()._solve_master(config)
        if next_point is not None:
            self.master_records.append(
                MasterRecord(
                    iteration=int(getattr(self, "iteration", 0)),
                    point=tuple(next_point),
                    z_lb=float(z_lb) if z_lb is not None else None,
                )
            )
        return z_lb, next_point


def write_separation_csv(records, csv_path):
    if not records:
        return

    max_dim = max(len(r.p_vals) for r in records)
    fieldnames = [
        "iteration",
        "anchor",
        "anchor_e1",
        "anchor_e2",
        "sep_obj",
        "alpha",
    ] + [f"p_{i}" for i in range(max_dim)]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "iteration": r.iteration,
                "anchor": str(r.anchor),
                "anchor_e1": r.anchor[0] if len(r.anchor) > 0 else None,
                "anchor_e2": r.anchor[1] if len(r.anchor) > 1 else None,
                "sep_obj": r.sep_obj,
                "alpha": r.alpha,
            }
            for i, val in enumerate(r.p_vals):
                row[f"p_{i}"] = val
            writer.writerow(row)


def plot_separation_3d(records, point_info, output_path):
    if not records:
        raise RuntimeError("No separation LP records available for plotting.")

    anchor_x = [r.anchor[0] for r in records]
    anchor_y = [r.anchor[1] for r in records]
    anchor_z = [r.sep_obj for r in records]

    neighbor_x = []
    neighbor_y = []
    neighbor_z = []
    for point, info in (point_info or {}).items():
        if info.get("source") != str(SearchPhase.NEIGHBOR_EVAL):
            continue
        neighbor_x.append(point[0])
        neighbor_y.append(point[1])
        neighbor_z.append(float(info.get("objective")))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        anchor_x,
        anchor_y,
        anchor_z,
        color="green",
        s=50,
        label="Anchors (sep LP)",
        depthshade=True,
    )
    if neighbor_x:
        ax.scatter(
            neighbor_x,
            neighbor_y,
            neighbor_z,
            color="magenta",
            s=35,
            label="Neighbors (objective)",
            depthshade=True,
        )

    ax.set_xlabel("e1")
    ax.set_ylabel("e2")
    ax.set_zlabel("separation LP objective")
    ax.set_title("LD-BD Separation LP at Anchors")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def plot_iteration_planes(records, master_records, point_info, external_bounds, output_dir):
    if not records or not master_records:
        raise RuntimeError("Missing separation LP or master records for plotting.")

    if len(external_bounds) < 2:
        raise RuntimeError("Expected at least two external variables for 3D plotting.")

    lb1, ub1 = external_bounds[0]
    lb2, ub2 = external_bounds[1]

    grid_x = np.linspace(lb1, ub1, 21)
    grid_y = np.linspace(lb2, ub2, 21)

    records_by_iter = {}
    for record in records:
        records_by_iter.setdefault(record.iteration, []).append(record)
    master_by_iter = {m.iteration: m for m in master_records}

    for iteration in sorted(records_by_iter.keys() & master_by_iter.keys()):
        iter_records = records_by_iter[iteration]
        master = master_by_iter[iteration]

        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

        for idx, rec in enumerate(iter_records):
            p0 = rec.p_vals[0]
            p1 = rec.p_vals[1]
            alpha = rec.alpha
            plane_z = p0 * mesh_x + p1 * mesh_y + alpha
            ax.plot_surface(
                mesh_x,
                mesh_y,
                plane_z,
                color="salmon",
                alpha=0.5,
                linewidth=0,
                antialiased=True,
                label="Cut planes" if idx == 0 else None,
            )

        anchor_x = []
        anchor_y = []
        anchor_z = []
        neighbor_x = []
        neighbor_y = []
        neighbor_z = []
        for point, info in (point_info or {}).items():
            if info.get("iteration_found") != iteration:
                continue
            source = info.get("source")
            if source == str(SearchPhase.ANCHOR):
                anchor_x.append(point[0])
                anchor_y.append(point[1])
                anchor_z.append(float(info.get("objective")))
            elif source == str(SearchPhase.NEIGHBOR_EVAL):
                neighbor_x.append(point[0])
                neighbor_y.append(point[1])
                neighbor_z.append(float(info.get("objective")))

        if anchor_x:
            ax.scatter(
                anchor_x,
                anchor_y,
                anchor_z,
                color="blue",
                s=45,
                label="Anchors",
            )
        if neighbor_x:
            ax.scatter(
                neighbor_x,
                neighbor_y,
                neighbor_z,
                color="#7ec8ff",
                s=35,
                label="Neighbors",
            )

        point_x = master.point[0]
        point_y = master.point[1]
        point_z = (
            master.z_lb
            if master.z_lb is not None
            else iter_records[0].sep_obj
        )
        ax.scatter(
            [point_x],
            [point_y],
            [point_z],
            color="purple",
            s=55,
            label="Master point",
        )

        ax.set_xlabel("e1")
        ax.set_ylabel("e2")
        ax.set_zlabel("cut plane")
        ax.set_title(f"Iteration {iteration}")
        ax.legend(loc="upper left")
        fig.tight_layout()

        output_path = output_dir / f"separation_plane_iter_{iteration:02d}.png"
        fig.savefig(output_path, dpi=300)


def main():
    output_dir = Path(__file__).resolve().parent
    csv_path = output_dir / "sep_lp.csv"
    fig_path = output_dir / "separation_lp_3d.png"

    model = build_model()
    solver = LDBDSeparationLogger()

    results = solver.solve(
        model,
        starting_point=[5, 1],
        direction_norm="Linf",
        logical_constraint_list=[model.oneY1, model.oneY2],
        subproblem_solver="gams",
        mip_solver="gurobi",
        separation_solver="gurobi",
        tee=True,
    )

    write_separation_csv(solver.separation_records, csv_path)
    plot_separation_3d(
        solver.separation_records,
        getattr(solver.data_manager, "point_info", None),
        fig_path,
    )

    external_info = getattr(solver.data_manager, "external_var_info_list", []) or []
    external_bounds = [(info.LB, info.UB) for info in external_info]
    plot_iteration_planes(
        solver.separation_records,
        solver.master_records,
        getattr(solver.data_manager, "point_info", None),
        external_bounds,
        output_dir,
    )

    print("Solver Status:", results.solver.status)
    print("Solver Termination Condition:", results.solver.termination_condition)
    print("Separation LP CSV:", csv_path)
    print("Figure:", fig_path)


if __name__ == "__main__":
    main()
