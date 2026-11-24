"""The continuous reaction field on the Grambow reaction set (plan item PO3).

Stage 1 only: the construction, on 2,000 reactions drawn with a fixed seed.  No model is fitted
here.  The stage answers four questions before the full run is allowed:

* How many reactions can be read and turned into fields, and why are the others refused?
* How far apart are the two sides after the superposition, under each of the two rules?
* Do the two identities of `cmfqsar/reaction.py` hold on real reactions, to rounding?
* What do the kernels and the invariant rows cost in time and memory, and what does that say
  about the full set?

``python benchmarks/prepare_reactions.py --download`` has to be run first.  The results go to
``benchmarks/results/reactions/stage1.json`` and ``table.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402

import prepare_reactions  # noqa: E402
from cmfqsar.fields import FieldSet  # noqa: E402
from cmfqsar.reaction import (PARTS, REACTION_SPEC, SUPERPOSE_ON, bond_change_counts,  # noqa: E402
                              bond_changes, channel_norms, invariant_rows, overlap_kernels,
                              reaction_fields, read_reaction)

RESULTS = os.path.join(HERE, "results", "reactions")
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
ALPHA = 0.5
SAMPLE = 2000
SEED = 0


def _say(text):
    print(text, flush=True)


def peak_memory_gb() -> float:
    """The largest memory this process has held so far, in GB, or nan where it cannot be read."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 2 ** 30
    except Exception:
        return float("nan")


def sample(n: int = SAMPLE, seed: int = SEED):
    rows = prepare_reactions.load()
    if n and n < len(rows):
        keep = np.sort(np.random.RandomState(seed).choice(len(rows), n, replace=False))
        rows = [rows[i] for i in keep]
    return rows


def build(rows, fieldset, on):
    """(reactions, their fields, reasons for the ones that could not be built)."""
    reactions, items, refused = [], [], {}
    for row in rows:
        rx = read_reaction(row["rsmi"], row["psmi"], row["r"], row["p"], f"rxn{row['idx']:06d}")
        try:
            it = reaction_fields(rx, fieldset, on=on)
        except Exception as err:                                  # a field that cannot be built
            why = f"{type(err).__name__}: {str(err)[:60]}"
            refused[why] = refused.get(why, 0) + 1
            continue
        reactions.append(rx)
        items.append(it)
    return reactions, items, refused


def _quantiles(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), q90=float(np.quantile(v, 0.9)),
                largest=float(v.max()))


def stage1(n: int = SAMPLE, seed: int = SEED, results: str = RESULTS) -> dict:
    RDLogger.DisableLog("rdApp.*")
    os.makedirs(results, exist_ok=True)
    fieldset = FieldSet.create(physchem=list(FIELDS), alpha=ALPHA)
    rows = sample(n, seed)
    out = dict(sample=len(rows), seed=seed, fields=list(FIELDS), alpha=ALPHA,
               spec=dict(d_max=REACTION_SPEC.d_max, step=REACTION_SPEC.step,
                         sigma=REACTION_SPEC.sigma))
    built = {}
    for on in SUPERPOSE_ON:
        t0 = time.time()
        reactions, items, refused = build(rows, fieldset, on)
        built[on] = (reactions, items)
        out[f"build_{on}"] = dict(
            built=len(items), refused=refused, seconds=time.time() - t0,
            rmsd_fitted_atoms=_quantiles([it.rmsd for it in items]),
            # how far every atom of the product is from its own place in the reactant
            displacement_all_atoms=_quantiles([
                float(np.sqrt(np.mean(np.sum((it.product.coords - it.reactant.coords) ** 2,
                                             axis=1)))) for it in items]))
        _say(f"{on}: {len(items)} built in {time.time() - t0:.0f} s, refused {refused}")

    reactions, items = built["heavy"]
    n_atoms = np.array([rx.n_atoms for rx in reactions])
    changed = np.array([len(bond_changes(rx)) for rx in reactions])
    counts, labels = bond_change_counts(reactions)
    order = np.argsort(-counts.sum(axis=0))
    out["reactions"] = dict(
        atoms=_quantiles(n_atoms), changed_bonds=_quantiles(changed),
        without_a_changed_bond=int(np.sum(changed == 0)),
        count_columns=len(labels),
        most_frequent_changes=[[labels[i], int(counts[:, i].sum())] for i in order[:12]])

    # the overlap kernels, their cost, and identity (1)
    t0 = time.time()
    K = overlap_kernels(items, fieldset)
    seconds = time.time() - t0
    sides = K["reactant"] + K["product"]
    channels = 2.0 * K["conventional"] + 0.5 * K["dynamic"]
    scale = float(np.abs(sides).max())
    out["overlap"] = dict(
        seconds=seconds, peak_memory_gb=peak_memory_gb(),
        bytes_per_part=int(K["dynamic"].nbytes),
        identity_one_largest_deviation=float(np.abs(sides - channels).max() / scale))
    _say(f"overlap kernels of {len(items)} reactions: {seconds:.0f} s, "
         f"peak {out['overlap']['peak_memory_gb']:.2f} GB, identity (1) to "
         f"{out['overlap']['identity_one_largest_deviation']:.1e}")
    # how much of the two sides is left in the dynamic channel, per field
    dyn = np.array([np.diag(K["dynamic"][f]) for f in range(len(FIELDS))])
    side = np.array([np.diag(sides[f]) for f in range(len(FIELDS))])
    share = dyn / np.where(side > 0, side, 1.0)
    out["dynamic_share_of_the_sides"] = {
        FIELDS[f]: _quantiles(share[f]) for f in range(len(FIELDS))}
    norms = channel_norms(items[:50], fieldset)
    out["norms_agree_with_the_kernel_diagonal"] = bool(
        np.allclose(norms ** 2, dyn[:, :50].T, rtol=1e-8, atol=1e-10))

    # identity (2) on the first 200 reactions against all
    back = [it.reversed() for it in items[:200]]
    Kb = overlap_kernels(back, fieldset, others=items, parts=("dynamic", "conventional", "bonds"))
    out["overlap"]["identity_two_largest_deviation"] = float(max(
        np.abs(Kb["dynamic"] + K["dynamic"][:, :200]).max(),
        np.abs(Kb["conventional"] - K["conventional"][:, :200]).max(),
        np.abs(Kb["bonds"] + K["bonds"][:200]).max()) / scale)
    del K, Kb, sides, channels

    # the invariant rows, their cost, identity (1), and what the superposition changes
    t0 = time.time()
    rows_h = invariant_rows(items)
    seconds = time.time() - t0
    rows_c = invariant_rows(built["centre"][1])
    scale = float(np.abs(rows_h["reactant"]).max())
    same = len(built["centre"][1]) == len(items)
    out["invariant"] = dict(
        seconds=seconds, columns=int(rows_h["dynamic"].shape[1]),
        bond_columns=int(rows_h["bonds"].shape[1]),
        identity_one_largest_deviation=float(np.abs(
            rows_h["reactant"] + rows_h["product"] - 2.0 * rows_h["conventional"]
            - 0.5 * rows_h["dynamic"]).max() / scale),
        odd_row_same_under_both_rules=bool(same and np.allclose(
            rows_h["odd"], rows_c["odd"], rtol=1e-8, atol=1e-8 * scale)),
        # correlation of the dynamic rows of the two superposition rules, over reactions
        dynamic_rows_correlation_between_rules=float(np.corrcoef(
            rows_h["dynamic"].ravel(), rows_c["dynamic"].ravel())[0, 1]) if same else None,
        peak_memory_gb=peak_memory_gb())
    _say(f"invariant rows: {seconds:.0f} s, {out['invariant']['columns']} columns, identity "
         f"(1) to {out['invariant']['identity_one_largest_deviation']:.1e}")

    # what the full set would need
    full = len(prepare_reactions.load())
    ratio = (full / max(len(items), 1)) ** 2
    out["full_set"] = dict(
        reactions=full,
        gb_per_part_all_fields=out["overlap"]["bytes_per_part"] * ratio / 2 ** 30,
        gb_per_part_one_field=out["overlap"]["bytes_per_part"] * ratio / len(FIELDS) / 2 ** 30,
        overlap_hours=out["overlap"]["seconds"] * ratio / 3600.0,
        invariant_minutes=out["invariant"]["seconds"] * full / max(len(items), 1) / 60.0)
    with open(os.path.join(results, "stage1.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def report(results: str = RESULTS) -> str:
    with open(os.path.join(results, "stage1.json")) as f:
        d = json.load(f)
    L = ["# The continuous reaction field, stage 1: the construction (plan item PO3)", "",
         f"{d['sample']} reactions of the Grambow set (wB97X-D3/def2-TZVP), drawn with seed "
         f"{d['seed']}. Fields: {', '.join(d['fields'])}; attenuation {d['alpha']} per square "
         f"angstrom. No model is fitted at this stage.", "",
         "## Superposition of the product on the reactant", "",
         "| Atoms the fit uses | Reactions built | RMSD over the fitted atoms, median (A) | "
         "90 % below (A) | Displacement of all atoms, median (A) | 90 % below (A) |",
         "|---|---|---|---|---|---|"]
    names = {"heavy": "all heavy atoms", "centre": "atoms of the changed bonds"}
    for on in ("heavy", "centre"):
        b = d[f"build_{on}"]
        L.append(f"| {names[on]} | {b['built']} | {b['rmsd_fitted_atoms']['median']:.3f} | "
                 f"{b['rmsd_fitted_atoms']['q90']:.3f} | "
                 f"{b['displacement_all_atoms']['median']:.3f} | "
                 f"{b['displacement_all_atoms']['q90']:.3f} |")
    r = d["reactions"]
    L += ["", "## The reactions", "",
          f"Atoms per reaction: median {r['atoms']['median']:.0f}, largest "
          f"{r['atoms']['largest']:.0f}. Changed bonds per reaction: median "
          f"{r['changed_bonds']['median']:.0f}, largest {r['changed_bonds']['largest']:.0f}; "
          f"{r['without_a_changed_bond']} reactions have none. The count reference has "
          f"{r['count_columns']} columns.", "",
          "| Bond change | Number in the sample |", "|---|---|"]
    L += [f"| {name} | {n} |" for name, n in r["most_frequent_changes"]]
    L += ["", "## Share of the two sides that is left in the dynamic channel", "",
          "The self-overlap of the dynamic channel divided by the sum of the self-overlaps of "
          "the two sides. Zero means that nothing changes. 1 means that the two sides do not "
          "overlap at all. A value above 1 means that the two sides overlap with opposite "
          "signs, which a field with signed weights allows.", "",
          "| Field | Median | 90 % below | Largest |", "|---|---|---|---|"]
    for name, q in d["dynamic_share_of_the_sides"].items():
        L.append(f"| {name} | {q['median']:.3f} | {q['q90']:.3f} | {q['largest']:.3f} |")
    o, i, f = d["overlap"], d["invariant"], d["full_set"]
    L += ["", "## The two identities, and the cost", "",
          "| Quantity | Value |", "|---|---|",
          f"| Identity (1), overlap kernel: largest deviation, relative to the largest entry | "
          f"{o['identity_one_largest_deviation']:.1e} |",
          f"| Identity (2), overlap kernel, 200 reversed reactions: largest deviation | "
          f"{o['identity_two_largest_deviation']:.1e} |",
          f"| Identity (1), invariant rows: largest deviation | "
          f"{i['identity_one_largest_deviation']:.1e} |",
          f"| The odd invariant row is the same under both superposition rules | "
          f"{'yes' if i['odd_row_same_under_both_rules'] else 'no'} |",
          f"| Correlation of the dynamic invariant rows between the two rules | "
          f"{i['dynamic_rows_correlation_between_rules']:.3f} |",
          f"| Overlap kernels of five parts, time (s) | {o['seconds']:.0f} |",
          f"| Invariant rows of five parts, time (s) | {i['seconds']:.0f} |",
          f"| Invariant columns per part | {i['columns']} |",
          f"| Largest memory held by the process (GB) | {i['peak_memory_gb']:.2f} |",
          f"| Full set of {f['reactions']} reactions: one part, all fields, in memory (GB) | "
          f"{f['gb_per_part_all_fields']:.1f} |",
          f"| Full set: overlap kernels of five parts, estimated time (h) | "
          f"{f['overlap_hours']:.1f} |",
          f"| Full set: invariant rows, estimated time (min) | {f['invariant_minutes']:.0f} |"]
    text = "\n".join(L) + "\n"
    with open(os.path.join(results, "table.md"), "w", newline="\n") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", type=int, default=SAMPLE, help="reactions drawn for the stage")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--report", action="store_true", help="only write the table again")
    args = ap.parse_args()
    if not args.report:
        stage1(args.sample, args.seed)
    _say(report())


if __name__ == "__main__":
    main()
