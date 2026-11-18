"""PETRA relabeling of a Mojito galactic-binary submission with the per-source flows of fit_source_flows.py.

The submission is already a labeling: one chain per source (GF_xxxxx). This script takes that labeling as PETRA's
starting point and the flows fitted to it as the auxiliary densities q_alpha, and performs the PETRA assignment step
(paper Eqs. 14 and 17): for every posterior sample, the entries are assigned one-to-one to catalog sources by
maximizing sum_alpha log[Q*_alpha q_alpha(theta)] over present sources plus log(1 - Q*_alpha) over absent ones.
Entries whose own source wins stay where they are; the others are relabeled.

Joint samples. The submissions provide one chain per source, not joint global-fit samples, so sample s is taken to
be row s of every chain. A chain with fewer rows than the longest one is absent from the remaining samples, which
gives Q*_alpha = n_rows_alpha / max n_rows (erebor's detection_statistic; 1 for every geemoo source).

Locality. Sources only compete with sources whose chains overlap in frequency. The sources are grouped into
clusters: connected components of overlapping [q, 1 - q] frequency-quantile intervals (``--quantile``, padded by
``--pad-uhz``). Assignments are solved per cluster; a cluster of one source cannot change and is copied unchanged.

This is one PETRA iteration with frozen flows. After it, the flows of the sources whose chains changed no longer
describe them exactly; refitting those flows on the new chains and rerunning this script is the next iteration
(PETRA's fixed point is reached when a pass moves nothing).

Output, under OUT_DIR/<submission>/:
    posterior_chains/GF_xxxxx_posterior.feather   the relabeled chain of each source, in the input's columns, plus
                                                   origin_GF and origin_row (where each row came from) and sample
    relabel_summary_<task>.csv                    per source: rows before and after, kept, lost, gained, from where
    clusters_<task>.csv                           per source: its cluster, cluster size, frequency interval
Both outputs keep the input's layout, so fit_source_flows.py can be pointed at OUT_DIR/<submission> to refit.

Usage:
    python petra_relabel_frozen.py SUBMISSION_DIR FLOWS_DIR OUT_DIR [--tasks N --task-id I]
    python petra_relabel_frozen.py SUBMISSION_DIR FLOWS_DIR OUT_DIR --combine     # merge the per-task summaries

SUBMISSION_DIR holds posterior_chains/ (as for fit_source_flows.py); FLOWS_DIR holds the GF_xxxxx_flow.pkl files.
With --tasks N the clusters, in frequency order, are split into N contiguous slices of near-equal source count.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_source_flows import COORDS, to_petra_coords  # noqa: E402  (same coordinates as the flows)

LOG_PROB_FLOOR = -50.0      # petra.copula_flows.DEFAULT_LOG_PROB_FLOOR: a non-finite flow density becomes this
CLEAR_CACHE_EVERY = 100     # flows evaluated between jax.clear_caches() calls (each flow compiles its own kernel)


# --------------------------------------------------------------------------------------------------------------------
# Sources and clusters
# --------------------------------------------------------------------------------------------------------------------
def load_ranges(submission: Path, quantile: float) -> pd.DataFrame:
    """Per source: number of rows and the [q, 1 - q] quantile interval of its frequency [uHz], sorted by frequency."""
    rows = []
    for path in sorted((submission / "posterior_chains").glob("GF_*_posterior.feather")):
        f = pd.read_feather(path, columns=["Frequency"])["Frequency"].to_numpy() * 1e6
        f = f[np.isfinite(f)]
        rows.append(dict(Name=path.name.removesuffix("_posterior.feather"), n_rows=len(f),
                         f_lo=np.quantile(f, quantile), f_hi=np.quantile(f, 1 - quantile), f_med=np.median(f)))
    return pd.DataFrame(rows).sort_values("f_lo", ignore_index=True)


def make_clusters(ranges: pd.DataFrame, pad: float) -> pd.DataFrame:
    """Label connected components of overlapping frequency intervals (a sweep over intervals sorted by f_lo)."""
    cluster, current, reach = np.empty(len(ranges), dtype=int), -1, -np.inf
    for i, (lo, hi) in enumerate(zip(ranges.f_lo - pad, ranges.f_hi + pad)):
        if lo > reach:
            current += 1
            reach = hi
        else:
            reach = max(reach, hi)
        cluster[i] = current
    out = ranges.assign(cluster=cluster)
    return out.assign(cluster_size=out.groupby("cluster").Name.transform("size"))


def task_clusters(clusters: pd.DataFrame, task_id: int, tasks: int) -> np.ndarray:
    """Cluster ids of slice `task_id`: contiguous in frequency, near-equal numbers of sources per slice."""
    ids = clusters.cluster.unique()
    first_row = clusters.groupby("cluster").cumcount().eq(0)
    start = np.flatnonzero(first_row.to_numpy())                    # row where each cluster starts
    cut = np.linspace(0, len(clusters), tasks + 1)
    owner = np.searchsorted(cut, start, side="right") - 1          # slice holding each cluster's first source
    return ids[owner == task_id]


# --------------------------------------------------------------------------------------------------------------------
# PETRA assignment within one cluster
# --------------------------------------------------------------------------------------------------------------------
def relabel_cluster(names, chains, flows, presence, n_max, eps, evaluated):
    """Relabel the entries of one cluster. Returns, per slot, the (source index, row) of the entry it holds per sample.

    chains[k]: (n_k, 6) PETRA coordinates of source k; row s is sample s. flows[k]: its coppuccino flow.
    """
    import coppuccino
    import jax

    K = len(names)
    for name, x in zip(names, chains):
        if not np.isfinite(x).all():
            raise SystemExit(f"{name}: non-finite PETRA coordinates (e.g. zero amplitude); clean the chain first")
    present = np.zeros((n_max, K), dtype=bool)
    for k, x in enumerate(chains):
        present[:len(x), k] = True
    points = np.concatenate(chains)
    offsets = np.concatenate([[0], np.cumsum([len(x) for x in chains])])

    # log q_alpha(theta) of every entry of the cluster under every source of the cluster.
    L = np.empty((K, len(points)))
    for a in range(K):
        lp = np.asarray(coppuccino.log_prob(flows[a], points), dtype=float)
        L[a] = np.where(np.isfinite(lp), lp, LOG_PROB_FLOOR)
        evaluated[0] += 1
        if evaluated[0] % CLEAR_CACHE_EVERY == 0:
            jax.clear_caches()

    q = np.clip(presence, eps, 1 - eps)
    log_q, log_not_q = np.log(q), np.log1p(-q)
    # entry index (into points) of source b in sample s; -1 where absent
    entry = np.full((n_max, K), -1)
    for k in range(K):
        rows = np.flatnonzero(present[:, k])
        entry[rows, k] = offsets[k] + rows

    # R[s, alpha, beta] = log Q_alpha + log q_alpha(theta_beta^s) for a present entry beta, log(1 - Q_alpha) if absent
    R = np.where(present[:, None, :], log_q[None, :, None] + L[:, np.clip(entry, 0, None)].transpose(1, 0, 2),
                 log_not_q[None, :, None])
    # Skip the assignment where the submission's labeling is provably optimal. With the gain G = R - log(1 - Q_alpha)
    # every null entry is worth 0 in every row, so any assignment totals sum_alpha log(1 - Q_alpha) plus the gains of
    # the present entries, which is at most the sum of their column maxima. The identity reaches that bound when every
    # present entry gains most in its own row.
    G = R - log_not_q[None, :, None]
    own = np.einsum("saa->sa", G)
    keep = np.all(~present | (own >= G.max(axis=1)), axis=1)

    holder = np.tile(np.arange(K), (n_max, 1))                    # holder[s, alpha] = entry slot beta it receives
    for s in np.flatnonzero(~keep):
        rows, cols = linear_sum_assignment(R[s], maximize=True)
        holder[s, rows] = cols
    # (source, row) held by each slot in each sample; (-1, -1) for an empty slot
    held_present = np.take_along_axis(present, holder, axis=1)
    src = np.where(held_present, holder, -1)
    row = np.where(held_present, np.arange(n_max)[:, None], -1)
    return src, row, int((~keep).sum())


# --------------------------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------------------------
def run_task(args) -> None:
    from coppuccino import load_flow

    submission, flows_dir = args.submission, args.flows
    sub_name = args.name or submission.name
    dest = args.out / sub_name
    (dest / "posterior_chains").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    clusters = make_clusters(load_ranges(submission, args.quantile), args.pad_uhz)
    n_max = int(clusters.n_rows.max())
    presence = clusters.set_index("Name").n_rows / n_max
    mine = task_clusters(clusters, args.task_id, args.tasks)
    sel = clusters[clusters.cluster.isin(mine)]
    sizes = sel.groupby("cluster").size()
    print(f"{sub_name}: {len(clusters)} sources in {clusters.cluster.nunique()} clusters "
          f"(largest {clusters.cluster_size.max()}); task {args.task_id}/{args.tasks}: {len(sel)} sources in "
          f"{len(mine)} clusters, {int((sizes > 1).sum())} with more than one source "
          f"(largest {sizes.max() if len(sizes) else 0}); ranges in {time.time() - t0:.0f} s", flush=True)
    sel.to_csv(dest / f"clusters_{args.task_id:04d}.csv", index=False)

    summary, evaluated, n_solved = [], [0], 0
    for cid, group in sel.groupby("cluster", sort=False):
        names = list(group.Name)
        tables = [pd.read_feather(submission / "posterior_chains" / f"{n}_posterior.feather") for n in names]
        if len(names) == 1:
            src = np.zeros((len(tables[0]), 1), dtype=int)
            row = np.arange(len(tables[0]))[:, None]
        else:
            flows = []
            for n in names:
                meta = json.loads((flows_dir / f"{n}_flow.json").read_text())
                if meta["coords"] != COORDS:
                    raise SystemExit(f"{n}: flow fitted in coordinates {meta['coords']}, expected {COORDS}")
                flows.append(load_flow(flows_dir / f"{n}_flow.pkl"))
            chains = [to_petra_coords(t) for t in tables]
            src, row, solved = relabel_cluster(names, chains, flows, presence[names].to_numpy(), n_max,
                                               args.eps, evaluated)
            n_solved += solved

        stacked = pd.concat(tables, ignore_index=True)
        offsets = np.concatenate([[0], np.cumsum([len(t) for t in tables])])
        for a, name in enumerate(names):
            held = src[:, a] >= 0
            s_idx, k_idx, r_idx = np.flatnonzero(held), src[held, a], row[held, a]
            out = stacked.iloc[offsets[k_idx] + r_idx].reset_index(drop=True)
            out["origin_GF"] = [names[k] for k in k_idx]
            out["origin_row"] = r_idx
            out["sample"] = s_idx
            out.to_feather(dest / "posterior_chains" / f"{name}_posterior.feather")

            own = (k_idx == a)
            gained = pd.Series([names[k] for k in k_idx[~own]]).value_counts()
            n_before = len(tables[a])
            summary.append(dict(Name=name, cluster=cid, cluster_size=len(names), n_before=n_before,
                                n_after=int(held.sum()), kept=int(own.sum()), lost=n_before - int(own.sum()),
                                gained=int((~own).sum()), presence_before=presence[name],
                                presence_after=held.sum() / n_max,
                                gained_from="; ".join(f"{g}:{c}" for g, c in gained.items())))

    summary = pd.DataFrame(summary)
    summary.to_csv(dest / f"relabel_summary_{args.task_id:04d}.csv", index=False)
    moved = summary.lost.sum()
    print(f"task {args.task_id}: {len(summary)} sources, {evaluated[0]} flow evaluations, assignment solved in "
          f"{n_solved} cluster-samples; {moved} of {summary.n_before.sum()} entries relabeled "
          f"({moved / max(summary.n_before.sum(), 1):.3%}); {int((summary.lost > 0).sum())} sources changed; "
          f"{time.time() - t0:.0f} s", flush=True)


def combine(args) -> None:
    dest = args.out / (args.name or args.submission.name)
    summary = pd.concat([pd.read_csv(p) for p in sorted(dest.glob("relabel_summary_*.csv"))], ignore_index=True)
    clusters = pd.concat([pd.read_csv(p) for p in sorted(dest.glob("clusters_*.csv"))], ignore_index=True)
    summary.sort_values("Name").to_csv(dest / "relabel_summary.csv", index=False)
    clusters.sort_values("f_lo").to_csv(dest / "clusters.csv", index=False)
    changed = summary[summary.lost + summary.gained > 0].sort_values("lost", ascending=False)
    (dest / "changed_sources.txt").write_text("\n".join(changed.Name) + ("\n" if len(changed) else ""))
    print(f"{len(summary)} sources; {summary.lost.sum()} of {summary.n_before.sum()} entries relabeled "
          f"({summary.lost.sum() / summary.n_before.sum():.3%}); {len(changed)} sources changed "
          f"(listed in changed_sources.txt)")
    if len(changed):
        print(changed.head(20).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path, help="submission directory, holding posterior_chains/")
    parser.add_argument("flows", type=Path, help="directory holding GF_xxxxx_flow.pkl / .json")
    parser.add_argument("out", type=Path, help="output root; results go to OUT/<submission name>/")
    parser.add_argument("--name", default=None, help="submission name for the output folder (default: its folder name)")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--task-id", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID or 0")
    parser.add_argument("--quantile", type=float, default=0.005,
                        help="frequency interval of a source: its [q, 1-q] quantiles (default 0.005)")
    parser.add_argument("--pad-uhz", type=float, default=0.0, help="widen every interval by this much [uHz]")
    parser.add_argument("--eps", type=float, default=1e-6, help="clip Q* to [eps, 1-eps] (petra's default)")
    parser.add_argument("--combine", action="store_true", help="merge the per-task summaries and exit")
    args = parser.parse_args()
    if args.combine:
        combine(args)
        return
    args.task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if not 0 <= args.task_id < args.tasks:
        raise SystemExit(f"task id {args.task_id} is outside 0..{args.tasks - 1}")
    run_task(args)


if __name__ == "__main__":
    main()
