import argparse
import os
import sys

import pyomo.environ as pe
from pyomo.core.base.misc import display
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory

# m = build_model(NT=6)

# # Solve MINLP
# pe.TransformationFactory('core.logical_to_linear').apply_to(m)
# pe.TransformationFactory('gdp.hull').apply_to(m)
# opt = SolverFactory('gams', solver='knitro')
# opt.solve(m, tee=True)

# # Solve with LDSDA
# for NT in range(6,7):
#     m = build_model(NT)
#     pe.SolverFactory('gdpopt.ldbd').solve(
#                                 m,
#                                 minlp_solver='gams',
#                                 nlp_solver='baron',    
#                                 # nlp_solver_args=dict(add_options=["option optcr=1e-4;"]),
#                                 # minlp_solver_args=dict(solver='gams',add_options=["option optcr=0.001;"]),#, tee=False, keepfiles=False), # TODO
#                                 starting_point=[1,1],
#                                 logical_constraint_list=[m.one_unreacted_feed,
#                                 m.one_recycle
#                                 ],
#                                 direction_norm='Linf',
#                                 time_limit=900,
#                                 tee=True,
#                             )
#     print('NT:', NT)
#     print('Objective:', pe.value(m.obj))

def _ensure_local_imports() -> None:
    """Allow `from cstr import build_model` when running from any cwd."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GDPopt-LDBD benchmark for the CSTR model."
    )
    parser.add_argument(
        "NT",
        type=int,
        help="Number of trays/time steps/etc (model-specific).",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=900,
        help="Solver wall time limit in seconds (default: 900).",
    )
    parser.add_argument(
        "--subproblem-solver",
        default="gams",
        help="Subproblem solver interface/name (default: gams).",
    )
    parser.add_argument(
        "--direction-norm",
        default="Linf",
        help="Direction norm for LDSDA/LDBD (default: Linf).",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Stream solver output to console.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    _ensure_local_imports()
    from cstr import build_model

    m = build_model(args.NT)
    mip_opts = {
        "Threads": 1,
        "Seed": 1,
        "ConcurrentMIP": 1,
        # optional: reduces “creative” behavior (can slow down)
        "Heuristics": 0.0,
        # optional: more stable numerics
        "NumericFocus": 3, ## This is the key
    }

    pe.SolverFactory("gdpopt.ldbd").solve(
        m,
        subproblem_solver=args.subproblem_solver,
        starting_point=[1, 1],
        logical_constraint_list=[
            m.one_unreacted_feed,
            m.one_recycle,
        ],
        mip_solver_args={"options": mip_opts},
        separation_solver="gurobi",
        # separation_solver_args={"options": mip_opts},
        direction_norm=args.direction_norm,
        time_limit=args.time_limit,
        tee=args.tee,
    )

    print(f"NT: {args.NT}")
    print(f"Objective: {pe.value(m.obj)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
