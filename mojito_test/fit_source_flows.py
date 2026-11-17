"""Fit and save one coppuccino copula flow per source chain of a Mojito galactic-binary submission.

Each submission directory holds one posterior chain per source, ``posterior_chains/GF_xxxxx_posterior.feather``. This
script fits a flow to every chain (or to one slice of them, for a SLURM job array) in the PETRA coordinates below and
writes, under OUT_DIR/<submission>/flows/:

    GF_xxxxx_flow.pkl     the fitted flow; read it with ``coppuccino.load_flow``
    GF_xxxxx_flow.json    what it was fitted on: coordinates, settings, sample count, timing, versions

A source whose files already exist is skipped, so a job that times out can simply be resubmitted. A source whose fit
fails is recorded in ``failed_<task>.csv`` and the job moves on to the next one.

The flows live in the coordinates of ``to_petra_coords``; PETRA and PETRA2 must evaluate them in the same ones.

Usage:
    python fit_source_flows.py DATA_DIR/SUBMISSION OUT_DIR [--task-id I --tasks N] [--f-min-mhz F --f-max-mhz F]
    python fit_source_flows.py DATA_DIR/SUBMISSION OUT_DIR --count        # number of sources to fit, then exit

With ``--tasks N`` the source list (sorted by name) is split into N contiguous slices and task I (0-based, default
$SLURM_ARRAY_TASK_ID) fits slice I.
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

COORDS = ["f [uHz]", "fdot [1e-18 Hz/s]", "log10 A", "cos inc", "sin dec", "RA"]
# Same flow settings as the PETRA2 workflow (FLOW in mojito_petra2_workflow.py).
FIT_SETTINGS = dict(knots=8, flow_layers=4, max_epochs=150, patience=20, learning_rate=1e-3)


def to_petra_coords(df: pd.DataFrame) -> np.ndarray:
    """The 6 PETRA coordinates of a submission's posterior table; phase and polarization are left out.

    Frequency is absolute (in uHz), so every source of every submission shares one coordinate system; coppuccino
    fits each marginal from the data quantiles and runs in float64, so the large offset costs no precision.
    """
    return np.column_stack([df["Frequency"].to_numpy() * 1e6,
                            df["Frequency Derivative"].to_numpy() * 1e18,
                            np.log10(df["Amplitude"].to_numpy()),
                            np.cos(df["Inclination"].to_numpy()),
                            np.sin(df["Dec"].to_numpy()),
                            df["RA"].to_numpy()])


def source_names(submission: Path, f_min: float | None, f_max: float | None) -> list[str]:
    """Names of the source chains, sorted, optionally restricted to a band of posterior-median frequency [Hz]."""
    names = sorted(p.name.removesuffix("_posterior.feather")
                   for p in (submission / "posterior_chains").glob("GF_*_posterior.feather"))
    if f_min is None and f_max is None:
        return names
    medians = pd.read_feather(submission / "posterior_medians.feather").set_index("Name")["Frequency"]
    lo, hi = f_min if f_min is not None else -np.inf, f_max if f_max is not None else np.inf
    keep = set(medians[(medians >= lo) & (medians < hi)].index)
    return [n for n in names if n in keep]


def task_slice(names: list[str], task_id: int, tasks: int) -> list[str]:
    """Contiguous slice `task_id` of `tasks` near-equal slices (neighbouring sources go to the same task)."""
    bounds = np.linspace(0, len(names), tasks + 1).round().astype(int)
    return names[bounds[task_id]:bounds[task_id + 1]]


def fit_one(chain_path: Path, flow_path: Path, seed: int, max_samples: int | None) -> dict:
    from coppuccino import normalizing_flows_fit, save_flow

    df = pd.read_feather(chain_path)
    x = to_petra_coords(df)
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    if max_samples is not None and len(x) > max_samples:
        x = x[np.random.default_rng(seed).choice(len(x), max_samples, replace=False)]
    t = time.time()
    flow = normalizing_flows_fit(x, rng_seed=seed, **FIT_SETTINGS)
    fit_seconds = time.time() - t

    # Write to a temporary name and rename, so an interrupted job never leaves a truncated flow behind.
    tmp = flow_path.with_suffix(".pkl.tmp")
    save_flow(flow, tmp)
    os.replace(tmp, flow_path)
    return dict(n_rows=int(len(df)), n_dropped_nonfinite=int((~finite).sum()), n_fit=int(len(x)),
                fit_seconds=round(fit_seconds, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path, help="submission directory, holding posterior_chains/")
    parser.add_argument("out", type=Path, help="output root; flows go to OUT/<submission name>/flows/")
    parser.add_argument("--tasks", type=int, default=1, help="number of slices the sources are split into")
    parser.add_argument("--task-id", type=int, default=None, help="slice to fit (default: $SLURM_ARRAY_TASK_ID or 0)")
    parser.add_argument("--f-min-mhz", type=float, default=None, help="only sources with median f >= this [mHz]")
    parser.add_argument("--f-max-mhz", type=float, default=None, help="only sources with median f < this [mHz]")
    parser.add_argument("--max-samples", type=int, default=None, help="subsample longer chains to this many rows")
    parser.add_argument("--seed", type=int, default=1, help="base seed; source GF_n uses seed + n")
    parser.add_argument("--count", action="store_true", help="print the number of sources and exit")
    args = parser.parse_args()

    f_min = None if args.f_min_mhz is None else args.f_min_mhz * 1e-3
    f_max = None if args.f_max_mhz is None else args.f_max_mhz * 1e-3
    names = source_names(args.submission, f_min, f_max)
    if args.count:
        print(len(names))
        return
    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if not 0 <= task_id < args.tasks:
        raise SystemExit(f"task id {task_id} is outside 0..{args.tasks - 1}")
    mine = task_slice(names, task_id, args.tasks)

    import jax
    from importlib.metadata import version
    from petra.copula_flows import quiet_external_fit

    dest = args.out / args.submission.name / "flows"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"task {task_id}/{args.tasks}: {len(mine)} of {len(names)} sources "
          f"({mine[0] if mine else '-'} .. {mine[-1] if mine else '-'}), jax devices {jax.devices()}", flush=True)

    failed = []
    for i, name in enumerate(mine, start=1):
        flow_path, meta_path = dest / f"{name}_flow.pkl", dest / f"{name}_flow.json"
        if flow_path.exists() and meta_path.exists():
            print(f"[{i}/{len(mine)}] {name}: already done", flush=True)
            continue
        seed = args.seed + int(name.split("_")[1])
        try:
            with quiet_external_fit(True):          # coppuccino 1.0 has no switch for its progress bar
                info = fit_one(args.submission / "posterior_chains" / f"{name}_posterior.feather", flow_path, seed,
                               args.max_samples)
        except Exception as err:                    # one bad chain must not stop the slice
            failed.append(dict(Name=name, error=repr(err)))
            print(f"[{i}/{len(mine)}] {name}: FAILED {err!r}", flush=True)
            traceback.print_exc()
            continue
        meta = dict(Name=name, submission=args.submission.name, coords=COORDS, settings=FIT_SETTINGS, seed=seed,
                    max_samples=args.max_samples, **info, coppuccino=version("coppuccino"), jax=jax.__version__,
                    host=platform.node(), slurm_job=os.environ.get("SLURM_JOB_ID"))
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"[{i}/{len(mine)}] {name}: {info['n_fit']} rows, {info['fit_seconds']} s", flush=True)

    if failed:
        pd.DataFrame(failed).to_csv(dest / f"failed_{task_id:04d}.csv", index=False)
        print(f"{len(failed)} fits failed; see {dest / f'failed_{task_id:04d}.csv'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
