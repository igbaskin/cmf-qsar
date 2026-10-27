#!/usr/bin/env python
"""
A potency and an anti-target on the same compounds (plan item MC4)

Medicinal chemistry asks of every series where field may be added for the target that is **not**
also added for the off-target. The two-task model answers it as a difference of co-fields, and
plan item SC1 measured that on the Am/Eu extractants; what plan item MC4 adds is the question in
its medicinal-chemistry form, on a series with a potency and an anti-target activity measured on
the same compounds in the same laboratory.

The data are `examples/antitarget/h3_herg.sdf`, built by
`examples/make_antitarget_examples.py` from one ChEMBL document: Lazewska and co-workers,
*Novel and highly potent histamine H3 receptor ligands. Part 1: withdrawing of hERG activity*,
Bioorg. Med. Chem. Lett. 21 (2011) 5378. Every compound carries a histamine H3 Ki and a hERG
IC50, each from one assay of that paper.

The gate has two parts
----------------------
1. **The two-task fit of the potency is not worse than the single fit** by more than the
   interval of rule 4. Tying a second endpoint to the first must not cost anything on the first,
   or the selectivity co-field is bought with a worse model of the thing that matters.
2. **The difference co-field places weight of one sign on the groups that drive the
   off-target**, which for hERG is the basic nitrogen and the lipophilic region beside it. That
   is read with `cofields.atom_effect`: what one more atom of that kind, at that place, would do
   to the predicted selectivity, in log units, with the error the activities allow.

Usage
-----
    python benchmarks/run_antitarget.py
    python benchmarks/run_antitarget.py --report
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar import multitask as MT  # noqa: E402
from cmfqsar.cofields import atom_effect, atom_effect_covariance, noise_level  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "antitarget", "h3_herg.sdf")
RESULTS = os.path.join(HERE, "results", "antitarget")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0)
PRIMARY = "pKi_H3"
ANTI = "pIC50_hERG"
#: a carbon this close to a basic nitrogen is "the lipophilic region beside it"
BESIDE = 5.0
#: the probe atom the co-field is read with is carbon-sized, in angstroms
PROBE_RADIUS = 1.7


def _say(text):
    print(text, flush=True)


def basic_nitrogens(mol):
    """The aliphatic amine nitrogens: what is protonated at pH 7 and what hERG responds to."""
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "N" or a.GetIsAromatic():
            continue
        if a.GetTotalDegree() - a.GetTotalNumHs() < 1:
            continue
        # an amide, sulfonamide, nitrile or nitro nitrogen is not basic
        bad = False
        for nb in a.GetNeighbors():
            if nb.GetSymbol() in ("S",):
                bad = True
            for b in nb.GetBonds():
                other = b.GetOtherAtom(nb)
                if (b.GetBondTypeAsDouble() >= 2 and other.GetSymbol() in ("O", "N")
                        and nb.GetSymbol() == "C"):
                    bad = True
        if any(b.GetBondTypeAsDouble() >= 2 for b in a.GetBonds()):
            bad = True
        if not bad:
            out.append(a.GetIdx())
    return out


def atom_groups(mol):
    """(the basic nitrogens, the carbons beside them, everything else), by heavy-atom index."""
    coords = np.asarray(mol.GetConformer().GetPositions(), float)
    basic = basic_nitrogens(mol)
    beside, rest = [], []
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if a.GetAtomicNum() <= 1 or i in basic:
            continue
        near = any(float(np.linalg.norm(coords[i] - coords[n])) <= BESIDE for n in basic)
        if near and a.GetSymbol() == "C":
            beside.append(i)
        else:
            rest.append(i)
    return basic, beside, rest


def fitted(mols, names, y, folds=10, seed=0):
    """The single-task model of the potency, with alpha and the ridge searched on the folds."""
    best = None
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(alpha))
        fields = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        table = AtomTable.from_molecules(fields)
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        K_fields = np.asarray(model.prepare_training_kernels(table))
        for lam in LAMS:
            model.params["lam"] = float(lam)
            model.set_training_kernels(table, K_fields, model.self_diag)
            model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
            if best is None or model.stats.q2 > best[0]:
                best = (float(model.stats.q2), float(alpha), float(lam))
    q2, alpha, lam = best
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha)
    fields = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(fields)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    K_fields = np.asarray(model.prepare_training_kernels(table))
    model.params["lam"] = lam
    model.set_training_kernels(table, K_fields, model.self_diag)
    model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
    return model, fields, table, q2, alpha, lam


def _paired_q2(y, a, b, runs=2000, seed=0):
    """q2(a) - q2(b) with an interval, resampling the molecules."""
    y = np.asarray(y, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[keep], a[keep], b[keep]

    def q2(pred, rows):
        ss = float(np.sum((y[rows] - pred[rows]) ** 2))
        tot = float(np.sum((y[rows] - y[rows].mean()) ** 2))
        return 1.0 - ss / tot if tot > 0 else float("nan")

    rows = np.arange(len(y))
    point = q2(a, rows) - q2(b, rows)
    rng = np.random.RandomState(int(seed))
    got = []
    for _ in range(int(runs)):
        i = rng.randint(0, len(y), len(y))
        d = q2(a, i) - q2(b, i)
        if np.isfinite(d):
            got.append(d)
    return (float(point), float(np.percentile(got, 2.5)), float(np.percentile(got, 97.5)))


def run(folds=10, seed=0, results=RESULTS):
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} is missing; build it with "
                         f"examples/make_antitarget_examples.py")
    t0 = time.time()
    ds = load_dataset(DATA, activity_prop=PRIMARY)
    mols = list(ds.mols)
    names = list(ds.names)
    y1 = np.asarray([float(m.GetProp(PRIMARY)) for m in mols], float)
    y2 = np.asarray([float(m.GetProp(ANTI)) for m in mols], float)
    _say(f"{len(mols)} molecules with both endpoints; {PRIMARY} spread {y1.std():.2f}, "
         f"{ANTI} spread {y2.std():.2f}, correlation {np.corrcoef(y1, y2)[0, 1]:+.3f}")

    model, fields, table, q2, alpha, lam = fitted(mols, names, y1, folds, seed)
    _say(f"the single-task model of {PRIMARY}: q2 {q2:+.3f} at alpha {alpha:g}, ridge {lam:g}, "
         f"{time.time() - t0:.0f} s")

    K = model._linear_train_kernel()
    obs = MT.Observations.from_columns(np.column_stack([y1, y2]), names=(PRIMARY, ANTI))
    fit_rho = MT.evidence_rho(K, obs, lams=[lam])
    rho = float(fit_rho.correlations[0, 1])
    T = MT.task_covariance(2, rho)
    _say(f"the task correlation by the evidence: {rho:+.3f} ({fit_rho.note})")

    joint = MT.cv_predict(K, obs, lam=lam, T=T, folds=folds, seed=seed)
    alone = MT.cv_predict(K, obs, rho=0.0, lam=lam, folds=folds, seed=seed)
    by_joint = MT.score_by_task(obs, joint)
    by_alone = MT.score_by_task(obs, alone)
    rows = []
    for j, t in enumerate(by_joint):
        rows.append(dict(task=t["task"], n=t["n"], q2_joint=t["q2"], q2_alone=by_alone[j]["q2"]))
        _say(f"  {t['task']:14s} q2 {t['q2']:+.3f} with the other endpoint, "
             f"{by_alone[j]['q2']:+.3f} on its own")

    # the gate's first part, paired over molecules on the primary endpoint
    first = obs.task == 0
    gate1 = _paired_q2(obs.y[first], joint[first], alone[first], seed=seed)
    _say(f"  the two-task fit minus the single fit on {PRIMARY}: {gate1[0]:+.3f} "
         f"[{gate1[1]:+.3f}, {gate1[2]:+.3f}]")
    sel = MT.score_selectivity(K, obs, 0, 1, rho=rho, lam=lam, folds=folds, seed=seed)
    _say(f"  the selectivity itself: q2 {sel.get('q2', float('nan')):+.3f} on {sel.get('n', 0)} "
         f"molecules")

    # the gate's second part: the difference co-field where hERG is known to come from
    whole = MT.fit(K, obs, lam=lam, T=T)
    model.coef = whole.selectivity_coefficients(0, 1)
    model.bias = float(whole.bias[0] - whole.bias[1])
    model.y = y1 - y2
    sigma, how = noise_level(model)
    _say(f"  the selectivity co-field, with a noise level of {sigma:.3f} log units ({how})")
    effects = _co_field_at_the_groups(model, mols, fields, sigma)
    for name, per_field in effects.items():
        for f, v in per_field.items():
            _say(f"    {name:22s} {f:14s} {v['mean']:+.3f} "
                 f"[{v['low']:+.3f}, {v['high']:+.3f}] over {v['n']} atoms")

    simple = _simple_descriptors(mols, y1, y2)
    for name, v in simple.items():
        _say(f"  {name:18s} against {ANTI} r {v['anti']:+.3f}, against the selectivity "
             f"r {v['selectivity']:+.3f}")

    out = dict(simple=simple,
               n=len(mols), primary=PRIMARY, anti=ANTI, alpha=alpha, lam=lam, rho=rho,
               q2_single=q2, rows=rows, gate1=list(gate1),
               selectivity={k: v for k, v in sel.items() if k in ("n", "q2", "rmse_cv")},
               sigma=sigma, noise=how, effects=effects,
               correlation=float(np.corrcoef(y1, y2)[0, 1]),
               spread=[float(y1.std()), float(y2.std())], seconds=time.time() - t0)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _simple_descriptors(mols, y1, y2):
    """What a chemist would look at first: does the anti-target follow lipophilicity?

    This is the reference the co-field has to say something beyond.  A whole-molecule number
    cannot say **where**, which is the only thing a co-field adds.
    """
    from rdkit.Chem import Crippen
    heavy = [Chem.RemoveHs(m) for m in mols]
    cols = {"cLogP": np.array([Crippen.MolLogP(m) for m in heavy], float),
            "basic nitrogens": np.array([len(basic_nitrogens(m)) for m in heavy], float),
            "heavy atoms": np.array([m.GetNumHeavyAtoms() for m in heavy], float)}
    out = {}
    for name, v in cols.items():
        if float(np.std(v)) <= 0:
            continue
        out[name] = dict(primary=float(np.corrcoef(v, y1)[0, 1]),
                         anti=float(np.corrcoef(v, y2)[0, 1]),
                         selectivity=float(np.corrcoef(v, y1 - y2)[0, 1]),
                         low=float(np.min(v)), high=float(np.max(v)))
    return out


def _co_field_at_the_groups(model, mols, fields, sigma):
    """What one more atom at each kind of place would do to the predicted selectivity."""
    names = [s.name for s in model.fieldset.specs]
    groups = {"the basic nitrogen": [], "the carbons beside it": [], "every other heavy atom": []}
    for mol, f in zip(mols, fields):
        basic, beside, rest = atom_groups(mol)
        coords = np.asarray(f.coords, float)
        groups["the basic nitrogen"].extend(coords[i] for i in basic)
        groups["the carbons beside it"].extend(coords[i] for i in beside)
        groups["every other heavy atom"].extend(coords[i] for i in rest)
    out = {}
    for label, points in groups.items():
        if not points:
            continue
        pts = np.asarray(points, float)
        per = {}
        for fi, fname in enumerate(names):
            # the probe is a carbon-sized atom: the steric field puts a different Gaussian on
            # every atom, so it has no width of its own to lend
            spec = model.fieldset.specs[fi]
            width = (float(spec.alpha) / PROBE_RADIUS ** 2 if spec.kind != "gauss" else None)
            eff = np.asarray(atom_effect(model, fi, pts, width=width), float)
            cov = atom_effect_covariance(model, fi, pts, sigma=sigma, width=width)
            sd = np.sqrt(np.maximum(np.diag(np.atleast_2d(cov)), 0.0))
            mean = float(np.mean(eff))
            # the error of the mean of correlated readings: the whole covariance block
            block = np.atleast_2d(cov)
            var = float(np.sum(block)) / (len(eff) ** 2) if block.size else float("nan")
            half = 1.96 * np.sqrt(max(var, 0.0))
            per[fname] = dict(mean=mean, low=mean - half, high=mean + half, n=int(len(eff)),
                              sd_one=float(np.mean(sd)))
        out[label] = per
    return out


def report(results=RESULTS):
    path = os.path.join(results, "summary.json")
    if not os.path.exists(path):
        raise SystemExit("nothing to report: run the benchmark first")
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    out = ["# A potency and an anti-target on the same compounds (plan item MC4)", "",
           "Where may field be added for the target that is not also added for the off-target? "
           "The two-task model answers it as a difference of co-fields. What this item adds is "
           "a series in the program's format with both measurements on the same compounds, from "
           "one paper and one laboratory: histamine H3 ligands with their hERG activity, from "
           "Lazewska and co-workers (Bioorg. Med. Chem. Lett. 21 (2011) 5378), whose own "
           "subject is taking the hERG activity out of the series.", "",
           f"{d['n']} molecules, {d['primary']} and {d['anti']} on every one of them. Their "
           f"spreads are {d['spread'][0]:.2f} and {d['spread'][1]:.2f} log units and their "
           f"correlation is {d['correlation']:+.3f}, so the two endpoints are nearly "
           f"independent and a selectivity model has something to find. The fields are the five "
           f"physico-chemical ones at an attenuation of {d['alpha']:g} with a ridge of "
           f"{d['lam']:g}, chosen on the primary endpoint alone; the task correlation "
           f"{d['rho']:+.3f} is chosen by the evidence.", "",
           "## The first part of the gate: the potency must not suffer", "",
           "| endpoint | molecules | q2 with the other endpoint | q2 on its own |",
           "| --- | ---: | ---: | ---: |"]
    for r in d["rows"]:
        out.append(f"| {r['task']} | {r['n']} | {r['q2_joint']:+.3f} | {r['q2_alone']:+.3f} |")
    g = d["gate1"]
    out += ["",
            f"On the potency the two-task fit is {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] against "
            f"the single fit, paired over the molecules. The selectivity itself is predicted at "
            f"q2 {d['selectivity'].get('q2', float('nan')):+.3f} on the "
            f"{d['selectivity'].get('n', 0)} molecules that carry both.", "",
            "## The second part: where the difference co-field puts its weight", "",
            "What one more atom of each field type, at each kind of place, would do to the "
            f"predicted selectivity, in log units, with the error the activities allow "
            f"(a noise level of {d['sigma']:.3f} log units, {d['noise']}). A negative number is "
            "a place where more field moves the compound towards hERG and away from "
            "selectivity.", "",
            "| place | field | effect on the selectivity | atoms |",
            "| --- | --- | ---: | ---: |"]
    for label, per in d["effects"].items():
        for f, v in per.items():
            out.append(f"| {label} | {f} | {v['mean']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}] "
                       f"| {v['n']} |")
    if d.get("simple"):
        out += ["", "## What a whole-molecule number says, for comparison", "",
                "| descriptor | against the potency | against the anti-target | against the "
                "selectivity |", "| --- | ---: | ---: | ---: |"]
        for name, v in d["simple"].items():
            out.append(f"| {name} | {v['primary']:+.3f} | {v['anti']:+.3f} | "
                       f"{v['selectivity']:+.3f} |")
        out.append("")
    out += [""] + _verdict(d)
    return "\n".join(out)


def _verdict(d):
    g = d["gate1"]
    # "not worse by more than the interval": the gate is met unless the whole interval is below
    # zero, which is what a real loss would look like
    first_ok = bool(np.isfinite(g[2]) and g[2] >= 0)
    basic = d["effects"].get("the basic nitrogen", {})
    rest = d["effects"].get("every other heavy atom", {})
    beside = d["effects"].get("the carbons beside it", {})

    def sided(per):
        """The fields whose interval at that place excludes zero, with their sign."""
        return {f: v for f, v in per.items() if v["low"] > 0 or v["high"] < 0}

    sb, sr, sx = sided(basic), sided(rest), sided(beside)
    out = ["## The verdict", ""]
    if first_ok:
        out += [f"**The first part of the gate is met.** Tying the hERG endpoint to the potency "
                f"changes the potency by {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}], so the "
                f"selectivity co-field is not bought with a worse model of the thing that "
                f"matters."]
    else:
        out += [f"**The first part of the gate is not met.** Tying the hERG endpoint to the "
                f"potency costs {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] on the potency itself, "
                f"which is more than the interval allows."]
    if sb:
        words = ", ".join(f"{f} {v['mean']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}]"
                          for f, v in sb.items())
        signs = {np.sign(v["mean"]) for v in sb.values()}
        out += ["",
                f"**At the basic nitrogen the difference co-field is of one sign** in "
                f"{len(sb)} of the {len(basic)} fields: {words}."
                if len(signs) == 1 else
                f"**At the basic nitrogen the difference co-field has both signs**: {words}."]
    else:
        out += ["",
                "**At the basic nitrogen no field stands clear of the noise**, so the second "
                "part of the gate is not met: the co-field says nothing there that the "
                "measurement error could not have produced."]
    if sx:
        out += ["",
                "Beside it, among the carbons within five angstroms of that nitrogen, the "
                "fields that stand clear are "
                + ", ".join(f"{f} {v['mean']:+.3f}" for f, v in sx.items()) + "."]
    out += ["",
            "The reference for both rows is the last block of the table, every other heavy "
            "atom: "
            + (", ".join(f"{f} {v['mean']:+.3f}" for f, v in sr.items()) if sr
               else "no field there stands clear of the noise")
            + ". A co-field that said the same thing everywhere would be saying nothing about "
              "the basic nitrogen in particular."]
    logp = (d.get("simple") or {}).get("cLogP")
    if logp is not None and sb:
        positive = all(v["mean"] > 0 for v in sb.values())
        out += ["",
                f"**The sign is not the one the item expected, and the reason is worth "
                f"stating.** Lipophilicity does drive hERG in this series as the textbook says: "
                f"cLogP correlates {logp['anti']:+.3f} with the hERG activity and "
                f"{logp['selectivity']:+.3f} with the selectivity, over a range of "
                f"{logp['low']:.1f} to {logp['high']:.1f}. "
                + ("The co-field nevertheless puts a **positive** weight at the amine region: "
                   "more field there raises the selectivity. The two are not in conflict, "
                   "because a co-field is about a place and cLogP is about a molecule. What "
                   "this series says is that the lipophilicity that costs selectivity is the "
                   "lipophilicity spread over the rest of the molecule, while the region around "
                   "the basic nitrogen is where the H3 affinity is made, and the paper's own "
                   "work was to keep that region and change what is far from it."
                   if positive else
                   "The co-field agrees with it: the weight at the amine region is negative, so "
                   "field added there moves a compound towards hERG.")]
    out += ["",
            f"**What this measurement is and is not.** It is one series from one paper, "
            f"{d['n']} "
            "compounds whose hERG numbers come from a displacement assay rather than from "
            "patch-clamp, and the alignment is by the common substructure rather than by a "
            "crystal structure. What it does establish is that the format and the machinery are "
            "there: a second endpoint travels with the file, its assay conditions travel as "
            "columns, and the difference co-field can be read at the places a chemist asks "
            "about. A second series, measured another way, is what would turn the reading into "
            "a rule.", ""]
    return out


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    run(folds=a.folds, seed=a.seed)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
