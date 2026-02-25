# LD_benchmark

Benchmark harness for comparing **LD-SDA** vs **LD-BD** on a small suite of GDP test problems.

This repo provides two Jupyter notebooks:

- `workflow/ModelTestingScript.ipynb` — runs the benchmarks and writes standardized artifacts.
- `workflow/DataProcessingScript.ipynb` — loads those artifacts and generates plots.

The GDPopt solver **full names** used for execution are:

- `gdpopt.ldsda`
- `gdpopt.ldbd`

## What gets benchmarked

The suite currently includes four benchmarks:

1. **Toy (2D)**: simple illustrative GDP with 2 external variables.
2. **Small-batch (3D)**: process superstructure with 3 external variables.
3. **CSTR sweep**: runs multiple sizes `NT ∈ {5,10,15,20,25,30}`.
4. **Distillation column (random init)**: runs multiple randomized initializations (keys in `models/column/column_initial_test.py`).
	- By default, the harness runs **all** available keys. Use `--column-keys ...` (CLI) or `run.column_keys` (JSON config) to restrict.

All benchmarks are run for both algorithms using a shared configuration object (`workflow/benchmark_suite.py::CommonConfig`).

## Requirements

At minimum you need a Python environment with:

- `pyomo`
- `pandas`
- `matplotlib`
- `openpyxl` (used by the distillation column initializer)

You also need working solver backends matching your configuration. The default notebook config is:

- NLP solver: `ipopt`
- MIP solver: `gurobi`
- MINLP solver: `gams`

If you use `gams` for MINLP subproblems, the *actual* MINLP solver used by GAMS can be selected via `minlp_solver_args`.
For example, to run BARON through GAMS and tighten the optimality tolerance:

```json
{
	"common": {
		"minlp_solver": "gams",
		"minlp_solver_args": {
			"solver": "baron",
			"add_options": ["option optcr=0.0001;"]
		}
	}
}
```

Make sure these are installed/licensed and discoverable in your environment.

## Running benchmarks (ModelTestingScript)

Open `workflow/ModelTestingScript.ipynb` and run cells top-to-bottom.

The notebook is intentionally structured so you can rerun *only one* failing benchmark:

- Run Toy
- Run Small-batch
- Run CSTR sweep
- Run Column random-init
- (Optional) Run everything

### Rerun behavior (overwrite vs append)

When you rerun a benchmark **without clearing anything**:

- Per-run files `solver.log` and `traj.csv` are **overwritten** in that run folder.
- Aggregated `summary.csv` is **appended** to.
	- This means rerunning the same case will create duplicate rows in `summary.csv`.

If you want a clean rerun for *one benchmark*, set the corresponding `CLEAR_*_FIRST = True` flag in that benchmark cell.
This deletes that benchmark subfolder for the current day before running.

## Running benchmarks (CLI / Linux)

For Linux (or CI), you can run benchmarks without notebooks using:

`run_benchmarks.py`

### JSON configuration (no long command lines)

The CLI runner automatically loads defaults from:

`benchmark_config.json`

So you can usually just:

```bash
python3 run_benchmarks.py
```

Edit `benchmark_config.json` to change solvers, which test(s) to run, algorithms, and sweep lists.

To run the full suite for only one algorithm by default, set:

- `run.algorithms` to a 1-element list (e.g. `["gdpopt.ldsda"]`)

To explore **all** column initial points by default, set:

- `run.column_keys` to `null` (use a list like `[1,2,3]` to restrict)

Examples:

```bash
# Run toy for both algorithms
python3 run_benchmarks.py --test toy

# Run only LDSDA for CSTR NT=25
python3 run_benchmarks.py --test cstr --algorithms gdpopt.ldsda --nt 25

# Run column for keys 1..10, clearing today's column folder first
python3 run_benchmarks.py --test column --column-keys 1-10 --clear

# Run everything and clear today's entire results folder first
python3 run_benchmarks.py --test all --clear

# Run everything for only one algorithm (e.g., LDSDA)
python3 run_benchmarks.py --test all --algorithms gdpopt.ldsda
```

## Results layout

All outputs are written under a date-stamped folder:

`results/<YYYYMMDD>/...`

Standard artifacts:

- `solver.log` — solver / subsolver console and GDPopt logs
- `traj.csv` — external-variable trajectory (path)
- `summary.csv` — one row per run with objective and timing
- `run_config.json` — per-run configuration snapshot (common config + effective solve kwargs)

When using the CLI runner, the date folder also includes:

- `benchmark_config.resolved.json` — the resolved configuration for that invocation (JSON defaults + CLI overrides)

Example structure:

```text
results/20260225/
	toy/
		ldsda/
			solver.log
			traj.csv
			summary.csv
		ldbd/
			solver.log
			traj.csv
			summary.csv
	cstr/
		NT25/
			ldsda/
				solver.log
				traj.csv
				summary.csv
			ldbd/
				solver.log
				traj.csv
				summary.csv
	column/
		ldsda/
			summary.csv
			init_001/
				solver.log
				traj.csv
			init_002/
				solver.log
				traj.csv
		ldbd/
			summary.csv
			init_001/
				solver.log
				traj.csv
```

Notes for the column benchmark:

- Each initialization writes its own `solver.log` / `traj.csv` under `init_###/`.
- All initializations append into a single `column/<algo>/summary.csv`.

## Processing and plots (DataProcessingScript)

Open `workflow/DataProcessingScript.ipynb` and run cells to:

- Load saved trajectories from `traj.csv` and plot them.
- Load `summary.csv` for timing/objective plots (notably for the column random-init and selected CSTR sizes).

This notebook assumes you have already run the corresponding benchmark(s) for the same `today` date stamp.

## Saving figures (CLI)

If you want to generate and save all figures **without** opening the plotting notebook, use:

`save_figures.py`

This reads from:

`results/<YYYYMMDD>/...`

and writes PNGs to:

`figure/<YYYYMMDD>/...`

Examples:

```bash
# Save all figures for today's date
python3 save_figures.py

# Save figures for a specific date
python3 save_figures.py --date 20260225

# Save only LDSDA figures
python3 save_figures.py --date 20260225 --algorithms gdpopt.ldsda

# Save CSTR NT=25 trajectory plots (and combined) for that date
python3 save_figures.py --date 20260225 --nt 25
```

Notes:

- `save_figures.py` expects the corresponding benchmark results to already exist for the date you request.
- The CSTR heatmap overlay figure is currently generated only for `--nt 30` (because the landscape CSV is stored in `assets/compl_enum_cstr_30_baron_dantzig.csv`).

## Troubleshooting

- **`init.xlsx` not found (column init)**: the column initializer reads `models/column/init.xlsx`.
	If you still see an error, verify the file exists and that `openpyxl` is installed.

- **Keyword-argument mismatch for LDSDA**: some Pyomo GDPopt solvers may not accept certain optional keywords
	(e.g., a separation solver). The runner is written to treat `separation_solver` as best-effort.

## Version control

Generated outputs are ignored by default via `LD_benchmark/.gitignore`:

- `results/`
- `figure/`
- `__pycache__/`, `*.py[cod]`
- `.ipynb_checkpoints/`

If your repo already has tracked `*.pyc` files, `.gitignore` will not hide diffs for them until you remove them from tracking.
