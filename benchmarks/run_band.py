#!/usr/bin/env python
"""
A vibrational band as a local target: the carbonyl stretch of substituted benzoic acids (plan
item SP2)

Plan item LP2 made the sample an atom and its surroundings -- the molecule's fields under a
Gaussian envelope about the atom -- and measured that description on a shift and on an acidity.
This script measures it on a vibrational band: the carbonyl stretch of the substituted benzoic
acids that Hammett defined his constants on, computed with ORCA at r2SCAN-3c by
`examples/make_band_examples.py` (`examples/spectra/benzoic_acids_co.sdf`, one number on the
carbonyl carbon of each acid).  A band is a property of a bond, so the question is the one LP2
asked of the shift: does the field around the atom predict it better than a count of what is
around the atom, and does a co-field around the atom say where the substituent acts?

Six things are measured, on folds that hold out whole molecules.

1. **The band against the Hammett constant.**  Before any field: the computed band against
   sigma, as a straight line, which is what a physical organic chemist would plot first.  The
   slope of that line for the meta and for the para acids is the picture the co-field is asked to
   reproduce, and the cross-validated q2 of the line is a second reference beside the count.
2. **The invariant local kernel** of LP2 (`cmfqsar/localenv.py`) with the envelope width
   scanned, against the atom-centred shell count of LP2, the reference of rule 1.  The plan's
   expectation is a best width near 1 A, as for the shift.
3. **The posed overlap kernel** of LP3 (`cmfqsar/localalign.py`) at the same widths.  A carbonyl
   carbon has one carbon and two oxygens of different types bonded to it, so its frame is unique
   and needs no averaging.
4. **The co-field around the carbonyl carbon**, read at the ring positions (ipso, ortho, meta,
   para, taken from the posed geometries themselves), with the error the noise in the bands
   allows and a y-scramble null, and drawn in the plane of the carboxyl group.  It is read at
   three widths, because the substituent is 4 to 7 A from the carbonyl carbon and a narrow
   envelope does not keep it.
5. **The length of the C=O bond as a predictor.**  The geometries in the data file are the ones
   the quantum optimisation produced, and a shorter bond has a higher stretching frequency
   (Badger's rule).  A straight line on this one length is therefore a control: a field
   description that is no better than this line has read the bond length, which the quantum
   calculation supplied, and not the substituent.
6. **Everything again on the start geometries.**  The same atoms, bonds and bands, with the
   coordinates of `examples/spectra/benzoic_acids.sdf`: the force-field geometries the quantum
   run started from, which hold no information from that run.  This is the measurement that
   answers the question of the item, whether the structure predicts the band.

Usage
-----
    python benchmarks/run_band.py                 # everything, into benchmarks/results/band/
    python benchmarks/run_band.py --rho 0.1 0.25  # a shorter width scan
    python benchmarks/run_band.py --structures none   # without the start geometries
    python benchmarks/run_band.py --report        # the table from the stored results
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
from rdkit.Geometry import Point3D  # noqa: E402

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.cofields import cofield_at_points, contrast_at  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localalign import overlap_kernel, posed_environments, well_posed  # noqa: E402
from cmfqsar.localenv import (LocalResult, choose_lam, count_features, count_kernel,  # noqa: E402
                              envelope_width, environment, local_features, local_kernel,
                              paired_interval, types_of)

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_co.sdf")
#: the geometries the quantum run started from (force field), for the control of part 6
STRUCTURES = os.path.join(ROOT, "examples", "spectra", "benzoic_acids.sdf")
RESULTS = os.path.join(HERE, "results", "band")
TARGET = "co_stretch"
#: envelope widths tried, in A^-2: from 10 A (the whole acid with almost equal weights) down to
#: 0.7 A (the carbonyl group and little else).  1.0 is where plan item LP2 found the shift best
#: predicted.
RHOS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
#: the widths the co-field is read at.  The para carbon is 4.3 A from the carbonyl carbon and
#: the first atom of a para substituent 5.8 A: the envelope there is 0.83 and 0.71 at rho 0.01,
#: 0.40 and 0.19 at rho 0.05, 0.16 and 0.03 at rho 0.1 (the width plan item LP3 read its ring
#: co-field at, where the positions were at most 2.8 A away).
COFIELD_RHOS = (0.01, 0.05, 0.1)
POSITIONS = ("ipso", "ortho", "meta", "para")
SCRAMBLES = 50
FOLDS = 10


def _say(text):
    print(text)
    sys.stdout.flush()


def q2_of(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def rmse_of(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2)))


# --------------------------------------------------------------------------- #
# the data
# --------------------------------------------------------------------------- #

def with_start_geometries(mols, names, path):
    """Copies of `mols` with the coordinates of the molecules of the same name in the SD file
    `path`: the same atoms, bonds and bands on the geometries the quantum run started from."""
    start = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(path, removeHs=False)
             if m is not None}
    out = []
    for m, name in zip(mols, names):
        s = start.get(name)
        if s is None:
            raise SystemExit(f"{path} has no structure named {name}")
        if [a.GetAtomicNum() for a in s.GetAtoms()] != [a.GetAtomicNum() for a in m.GetAtoms()]:
            raise SystemExit(f"{name}: the atoms of {path} are in another order")
        c = Chem.Mol(m)
        conf, pos = c.GetConformer(), s.GetConformer().GetPositions()
        for i in range(c.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(float(pos[i][0]), float(pos[i][1]),
                                            float(pos[i][2])))
        out.append(c)
    return out


def read(path=DATA, structures=None):
    """The acids, their fields, the per-atom target and the substituent table.  With
    `structures`, the coordinates are taken from that SD file (:func:`with_start_geometries`)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run examples/make_band_examples.py, then the "
                         f"same script with --orca")
    ds = load_dataset(path, atom_activity=TARGET)
    mols, names = list(ds.mols), list(ds.names)
    if structures is not None:
        mols = with_start_geometries(mols, names, structures)
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    targets = AtomTargets.from_molecules(mols, TARGET, names=names)
    if len(targets) != len(mols):
        raise SystemExit(f"{len(targets)} bands on {len(mols)} acids: every acid should carry "
                         f"one band on its carbonyl carbon")
    table = {"position": [], "sigma": [], "substituent": [], "anti": [], "imaginary": []}
    for r in range(len(targets)):
        m = mols[int(targets.molecule[r])]
        table["position"].append(m.GetProp("position") if m.HasProp("position") else "")
        table["sigma"].append(float(m.GetProp("sigma")) if m.HasProp("sigma") else float("nan"))
        table["substituent"].append(m.GetProp("substituent") if m.HasProp("substituent")
                                    else names[int(targets.molecule[r])])
        table["anti"].append(m.HasProp("carboxyl_after_opt")
                             and m.GetProp("carboxyl_after_opt") != "syn")
        table["imaginary"].append(int(m.GetProp("n_imaginary")) if m.HasProp("n_imaginary")
                                  else 0)
    return mols, names, fields, targets, fieldset, table


# --------------------------------------------------------------------------- #
# 1. the band against the Hammett constant
# --------------------------------------------------------------------------- #

def hammett_lines(targets, table):
    """The straight line band = a + rho * sigma, for every acid together and for the meta and
    para acids apart: slope, intercept, correlation and count."""
    y = np.asarray(targets.y, float)
    sigma = np.asarray(table["sigma"], float)
    pos = np.asarray(table["position"])
    out = {}
    for label, mask in (("all", np.isfinite(sigma)),
                        ("meta", (pos == "meta") | (pos == "none")),
                        ("para", (pos == "para") | (pos == "none"))):
        mask = mask & np.isfinite(sigma)
        if mask.sum() < 3 or np.std(sigma[mask]) <= 0:
            continue
        slope, intercept = np.polyfit(sigma[mask], y[mask], 1)
        r = float(np.corrcoef(sigma[mask], y[mask])[0, 1])
        out[label] = dict(slope=float(slope), intercept=float(intercept), r=r,
                          n=int(mask.sum()))
    return out


def _as_result(name, pred, targets, folds, note=""):
    y = np.asarray(targets.y, float)
    return LocalResult(name=name, predicted=np.asarray(pred, float), q2=q2_of(y, pred),
                       error=rmse_of(y, pred), mae=float(np.mean(np.abs(y - pred))), lam=0.0,
                       n_samples=len(y), n_molecules=targets.n_measured_molecules, folds=folds,
                       note=note, molecule=np.asarray(targets.molecule, int))


def hammett_reference(targets, table, folds=FOLDS, seed=0, two_lines=False):
    """The Hammett line as a predictor, out of fold: one line over sigma, or one per position."""
    y = np.asarray(targets.y, float)
    sigma = np.asarray(table["sigma"], float)
    pos = np.asarray(table["position"])
    pred = np.full(len(y), np.nan)
    for te in targets.folds(folds, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        groups = (("meta", "para") if two_lines else ("all",))
        for g in groups:
            in_g = np.ones(len(y), bool) if g == "all" else ((pos == g) | (pos == "none"))
            fit_rows = tr[in_g[tr] & np.isfinite(sigma[tr])]
            if len(fit_rows) < 3:
                continue
            slope, intercept = np.polyfit(sigma[fit_rows], y[fit_rows], 1)
            rows = te[in_g[te]]
            if g != "all":
                # the unsubstituted acid sits on both lines: give it the mean of the two
                both = rows[pos[rows] == "none"]
                rows = rows[pos[rows] != "none"]
                for b in both:
                    v = intercept + slope * sigma[b]
                    pred[b] = v if not np.isfinite(pred[b]) else 0.5 * (pred[b] + v)
            pred[rows] = intercept + slope * sigma[rows]
    pred = np.where(np.isfinite(pred), pred, np.nanmean(y))
    name = "two Hammett lines (meta, para)" if two_lines else "one Hammett line"
    return _as_result(name, pred, targets, folds, note="band = a + rho * sigma, fitted in fold")


# --------------------------------------------------------------------------- #
# 5. the length of the C=O bond as a predictor
# --------------------------------------------------------------------------- #

def carbonyl_lengths(mols, targets):
    """The length of the C=O bond of every sample, in A: from the target carbon to its oxygen
    neighbour that has no other bond.  NaN where the carbon has no such neighbour."""
    out = np.full(len(targets), np.nan)
    for r in range(len(targets)):
        mol, c = mols[int(targets.molecule[r])], int(targets.atom[r])
        o = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()
             if n.GetSymbol() == "O" and n.GetDegree() == 1]
        if len(o) == 1:
            p = mol.GetConformer().GetPositions()
            out[r] = float(np.linalg.norm(p[c] - p[o[0]]))
    return out


def bond_length_reference(mols, targets, folds=FOLDS, seed=0):
    """The band as a straight line on the length of the C=O bond, out of fold.

    A shorter bond has a higher stretching frequency (Badger's rule).  On geometries from a
    quantum optimisation this line is a control and not a competitor: the length comes from
    the calculation that also gives the band."""
    y = np.asarray(targets.y, float)
    d = carbonyl_lengths(mols, targets)
    ok = np.isfinite(d)
    pred = np.full(len(y), np.nan)
    for te in targets.folds(folds, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        tr = tr[ok[tr]]
        if len(tr) < 3 or np.std(d[tr]) <= 0:
            continue
        slope, intercept = np.polyfit(d[tr], y[tr], 1)
        pred[te] = intercept + slope * d[te]
    pred = np.where(np.isfinite(pred), pred, np.nanmean(y))
    r = float(np.corrcoef(d[ok], y[ok])[0, 1]) if ok.sum() > 2 and np.std(d[ok]) > 0 else 0.0
    return _as_result("the length of the C=O bond (one line)", pred, targets, folds,
                      note=f"band = a + b * length, fitted in fold; r {r:+.3f}; lengths "
                           f"{np.nanmin(d):.4f} to {np.nanmax(d):.4f} A, spread (sd) "
                           f"{np.nanstd(d):.4f} A")


# --------------------------------------------------------------------------- #
# 4. the ring positions in the frame of the carbonyl carbon
# --------------------------------------------------------------------------- #

def ring_points(mols, targets, envs):
    """Where the ring sits in the posed frame: mean coordinates of the ipso, ortho, meta and
    para carbons over the samples, the two ortho and the two meta carbons told apart by the side
    of the carboxyl they lie on (the carbonyl oxygen's side or the hydroxyl's).

    Returns the labelled mean points and, for the contrasts, the four-position points with the
    two sides averaged.
    """
    sums, counts = {}, {}
    side_of_o = []
    for r in range(len(targets)):
        k, c = int(targets.molecule[r]), int(targets.atom[r])
        mol, env = mols[k], envs[r]
        back = {int(a): e for e, a in enumerate(np.asarray(env.member_atom_map, int))}
        dist = Chem.GetDistanceMatrix(mol)
        atom_c = mol.GetAtomWithIdx(c)
        ipso = [n.GetIdx() for n in atom_c.GetNeighbors() if n.GetIsAromatic()]
        o_carbonyl = [n.GetIdx() for n in atom_c.GetNeighbors()
                      if n.GetSymbol() == "O" and n.GetDegree() == 1]
        if len(ipso) != 1 or len(o_carbonyl) != 1 or o_carbonyl[0] not in back:
            continue
        ring = next((set(rg) for rg in mol.GetRingInfo().AtomRings()
                     if ipso[0] in rg and len(rg) == 6), None)
        if ring is None:
            continue
        y_o = float(env.coords[back[o_carbonyl[0]]][1])
        side_of_o.append(np.sign(y_o) or 1.0)
        for a in ring:
            d = int(dist[c, a])
            label = {1: "ipso", 2: "ortho", 3: "meta", 4: "para"}.get(d)
            if label is None or a not in back:
                continue
            p = np.asarray(env.coords[back[a]], float)
            if label in ("ortho", "meta"):
                label += " (carbonyl side)" if p[1] * y_o > 0 else " (hydroxyl side)"
            sums[label] = sums.get(label, 0.0) + p
            counts[label] = counts.get(label, 0) + 1
    means = {k: (sums[k] / counts[k]) for k in sums}
    four = {}
    for name in POSITIONS:
        parts = [v for k, v in means.items() if k == name or k.startswith(name + " ")]
        if parts:
            four[name] = np.mean(np.array(parts), axis=0)
    return means, four, {k: int(v) for k, v in counts.items()}


def para_over_meta(values):
    a, b = abs(values["para"]), abs(values["meta"])
    return float((a - b) / (a + b)) if (a + b) > 0 else 0.0


def alternation(values):
    a = 0.5 * (abs(values["ortho"]) + abs(values["para"]))
    b = abs(values["meta"])
    return float((a - b) / (a + b)) if (a + b) > 0 else 0.0


NULLED = ("para_over_meta", "alternation")


def cofield_reading(model, table, y, lam, names, means, four, sigma_noise, runs=SCRAMBLES,
                    seed=0):
    """The co-field of every field at the ring positions, its error from the noise in the bands,
    and a y-scramble null for the position statistics."""
    labels = list(means.keys())
    pts_all = np.array([means[k] for k in labels])
    pts_four = np.array([four[k] for k in POSITIONS])
    model.params = dict(model.params)
    model.params["lam"] = float(lam)

    def read():
        out = {}
        for f, name in enumerate(names):
            v4 = dict(zip(POSITIONS, cofield_at_points(model, f, pts_four, include_scale=True)))
            vals = dict(zip(labels, cofield_at_points(model, f, pts_all, include_scale=True)))
            out[name] = {"at": {k: float(v) for k, v in vals.items()},
                         "four": {k: float(v) for k, v in v4.items()},
                         "para_over_meta": para_over_meta(v4), "alternation": alternation(v4)}
        return out

    model.fit(table, y, cv_folds=5, cv_seed=0, reuse_kernels=True)
    out = read()
    for f, name in enumerate(names):
        errs = {}
        for k in POSITIONS:
            w = np.zeros(len(POSITIONS))
            w[POSITIONS.index(k)] = 1.0
            _v, sd = contrast_at(model, f, pts_four, w, sigma=sigma_noise)
            errs["sd_" + k] = float(sd)
            errs["z_" + k] = float(abs(out[name]["four"][k]) / sd) if sd > 0 else float("nan")
        s = {k: float(np.sign(out[name]["four"][k]) or 1.0) for k in POSITIONS}
        w = np.zeros(len(POSITIONS))
        w[POSITIONS.index("para")] = s["para"]
        w[POSITIONS.index("meta")] = -s["meta"]
        v, sd = contrast_at(model, f, pts_four, w, sigma=sigma_noise)
        errs["para_minus_meta"] = float(v)
        errs["para_minus_meta_sd"] = float(sd)
        errs["para_minus_meta_z"] = float(v / sd) if sd > 0 else float("nan")
        out[name].update(errs)
    rng = np.random.RandomState(int(seed))
    null = {name: {k: [] for k in NULLED} for name in names}
    for _ in range(int(runs)):
        model.fit(table, y[rng.permutation(len(y))], cv_folds=5, cv_seed=0, reuse_kernels=True)
        got = read()
        for name in names:
            for k in NULLED:
                null[name][k].append(got[name][k])
    for name in names:
        for k in NULLED:
            a = np.asarray([x for x in null[name][k] if np.isfinite(x)], float)
            mean = float(a.mean()) if len(a) else float("nan")
            sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
            out[name][k + "_null"] = mean
            out[name][k + "_null_sd"] = sd
            out[name][k + "_z"] = (float((out[name][k] - mean) / sd)
                                   if sd and np.isfinite(sd) and sd > 0 else float("nan"))
    model.fit(table, y, cv_folds=5, cv_seed=0, reuse_kernels=True)
    return out


def plot_cofield(model, names, means, path, fields=("electrostatic", "steric")):
    """The co-field in the plane of the carboxyl group, with the mean atom positions drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(-3.0, 6.6, 0.1)
    ys = np.arange(-4.0, 4.0, 0.1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    show = [f for f in fields if f in names]
    fig, axes = plt.subplots(1, len(show), figsize=(4.8 * len(show), 4.2), squeeze=False)
    for j, fname in enumerate(show):
        f = list(names).index(fname)
        v = cofield_at_points(model, f, pts, include_scale=True).reshape(X.shape)
        vmax = float(np.abs(v).max()) or 1.0
        ax = axes[0, j]
        im = ax.imshow(v, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="equal")
        ax.scatter([0.0], [0.0], s=60, facecolor="none", edgecolor="black", lw=1.6, zorder=4)
        for label, p in means.items():
            ax.scatter([p[0]], [p[1]], s=18, color="black", zorder=3)
            short = label.split(" ")[0]
            ax.annotate(short, (p[0], p[1] + 0.25), ha="center", va="bottom", fontsize=7,
                        color="dimgray")
        ax.set_title(f"{model.fieldset.specs[f].label}, z = 0 A", fontsize=10)
        ax.set_xlabel("x / A (ring on +x)", fontsize=8)
        ax.set_ylabel("y / A", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle("Co-field around the carbonyl carbon (red: raises the band per unit of field). "
                 "The carbonyl carbon is at the origin.", fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def _res(r):
    return dict(name=r.name, q2=r.q2, error=r.error, mae=r.mae, lam=r.lam, note=r.note)


def measure(mols, fields, targets, fieldset, hammett, rhos, cofield_rhos, folds, seed,
            scrambles, results, prefix=""):
    """The references that depend on the geometry (the count and the bond length), the width
    scan of the two kernels and the co-field at the reading widths, for one set of geometries.
    `hammett` holds the two Hammett references, which do not depend on the geometry; `prefix`
    starts the names of the figures."""
    y = np.asarray(targets.y, float)
    one = hammett["one_line"]
    types = types_of(fields, min_atoms=10)
    ref = choose_lam(count_kernel(count_features(fields, targets, types)), targets, folds=folds,
                     seed=seed, name="atom-centred counts")
    ref.note = f"{len(types)} atom types in 6 shells"
    _say("  " + ref.summary())
    bond = bond_length_reference(mols, targets, folds, seed)
    d = paired_interval(bond, ref, targets, runs=2000, seed=seed)
    bond.note += f"; against the counts {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
    _say("  " + bond.summary())

    rows = []
    best = None
    for rho in rhos:
        t0 = time.time()
        plain = [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
                 for r in range(len(targets))]
        inv = choose_lam(local_kernel(local_features(plain)), targets, folds=folds, seed=seed,
                         name=f"invariant local, rho {rho:g}")
        envs = posed_environments(fields, targets, rho=rho)
        model, atable, K = overlap_kernel(envs, fieldset)
        aligned = choose_lam(K, targets, folds=folds, seed=seed,
                             name=f"posed overlap, rho {rho:g}")
        d_i = paired_interval(inv, ref, targets, runs=2000, seed=seed)
        d_a = paired_interval(aligned, ref, targets, runs=2000, seed=seed)
        d_ai = paired_interval(aligned, inv, targets, runs=2000, seed=seed)
        d_i1 = paired_interval(inv, one, targets, runs=2000, seed=seed)
        d_a1 = paired_interval(aligned, one, targets, runs=2000, seed=seed)
        d_ib = paired_interval(inv, bond, targets, runs=2000, seed=seed)
        d_ab = paired_interval(aligned, bond, targets, runs=2000, seed=seed)
        rows.append(dict(rho=float(rho), width=float(envelope_width(rho)),
                         atoms=float(np.mean([e.n_atoms for e in envs])),
                         invariant=dict(q2=inv.q2, error=inv.error, mae=inv.mae, lam=inv.lam),
                         aligned=dict(q2=aligned.q2, error=aligned.error, mae=aligned.mae,
                                      lam=aligned.lam),
                         against_counts=dict(invariant=d_i, aligned=d_a),
                         against_one_line=dict(invariant=d_i1, aligned=d_a1),
                         against_bond_length=dict(invariant=d_ib, aligned=d_ab),
                         aligned_against_invariant=d_ai, seconds=time.time() - t0))
        _say(f"  rho {rho:g} (width {envelope_width(rho):.2f} A, {rows[-1]['atoms']:.1f} atoms)")
        _say("    " + inv.summary() + f"; against the counts {d_i[0]:+.3f} "
             f"[{d_i[1]:+.3f}, {d_i[2]:+.3f}]")
        _say("    " + aligned.summary() + f"; against the counts {d_a[0]:+.3f} "
             f"[{d_a[1]:+.3f}, {d_a[2]:+.3f}]; against the invariant {d_ai[0]:+.3f} "
             f"[{d_ai[1]:+.3f}, {d_ai[2]:+.3f}]")
        if best is None or aligned.q2 > best[1]:
            best = (float(rho), aligned.q2, aligned.lam, aligned.error)

    cofields = {}
    names_f = [s.name for s in fieldset.specs]
    for rho in sorted({float(r) for r in cofield_rhos}):
        envs = posed_environments(fields, targets, rho=rho)
        model, atable, K = overlap_kernel(envs, fieldset)
        aligned = choose_lam(K, targets, folds=folds, seed=seed, name=f"posed, rho {rho:g}")
        means, four, counts = ring_points(mols, targets, envs)
        if len(four) < 4:
            _say(f"  rho {rho:g}: the ring positions are not all kept ({sorted(four)}); "
                 f"no co-field read at this width")
            continue
        reading = cofield_reading(model, atable, y, aligned.lam, names_f, means, four,
                                  sigma_noise=aligned.error, runs=scrambles, seed=seed)
        tag = prefix + f"cofield_rho_{rho:g}".replace(".", "p")
        plot_cofield(model, names_f, means, os.path.join(results, f"{tag}.png"))
        cofields[f"{rho:g}"] = dict(rho=rho, width=float(envelope_width(rho)), q2=aligned.q2,
                                    lam=aligned.lam, error=aligned.error,
                                    positions={k: [float(x) for x in v] for k, v in means.items()},
                                    four={k: [float(x) for x in v] for k, v in four.items()},
                                    counts=counts, fields=reading, figure=f"{tag}.png")
        e = reading.get("electrostatic", {})
        _say(f"  co-field at rho {rho:g}: electrostatic at ipso/ortho/meta/para "
             + ", ".join(f"{e['four'][k]:+.3g}" for k in POSITIONS)
             + f"; para over meta {e['para_over_meta']:+.2f} (null {e['para_over_meta_null']:+.2f}"
             f" +/- {e['para_over_meta_null_sd']:.2f}, z {e['para_over_meta_z']:+.1f}); "
             f"para minus meta {e['para_minus_meta']:+.3g} +/- {e['para_minus_meta_sd']:.2g} "
             f"(z {e['para_minus_meta_z']:+.1f})")
    return dict(references={"counts": _res(ref), "bond_length": _res(bond)}, rows=rows,
                best=dict(rho=best[0], q2=best[1], lam=best[2], error=best[3]),
                cofields=cofields)


def run(rhos=RHOS, cofield_rhos=COFIELD_RHOS, folds=FOLDS, seed=0, scrambles=SCRAMBLES,
        results=RESULTS, data=DATA, structures=STRUCTURES):
    t_all = time.time()
    os.makedirs(results, exist_ok=True)       # the co-field pictures are written before the table
    mols, names, fields, targets, fieldset, table = read(data)
    y = np.asarray(targets.y, float)
    n_anti, n_imag = int(sum(table["anti"])), int(sum(1 for k in table["imaginary"] if k))
    _say(f"=== the carbonyl band of {len(targets)} substituted benzoic acids: "
         f"{y.min():.1f} to {y.max():.1f} cm^-1, spread (sd) {y.std():.1f}; "
         f"{n_anti} ended with an anti carboxyl, {n_imag} with an imaginary frequency")
    lines = hammett_lines(targets, table)
    for k, v in lines.items():
        _say(f"  band against sigma, {k}: slope {v['slope']:+.1f} cm^-1 per sigma unit, "
             f"r {v['r']:+.3f}, n {v['n']}")
    poses = well_posed(fields, targets, rho=min(cofield_rhos))
    _say(f"  poses per sample at the carbonyl carbon: {dict(sorted(poses.items()))}")
    hammett = {"one_line": hammett_reference(targets, table, folds, seed, two_lines=False),
               "two_lines": hammett_reference(targets, table, folds, seed, two_lines=True)}
    for r in hammett.values():
        _say("  " + r.summary())

    _say("--- the geometries of the quantum optimisation")
    main_part = measure(mols, fields, targets, fieldset, hammett, rhos, cofield_rhos, folds,
                        seed, scrambles, results)
    start_part = None
    if structures and os.path.exists(structures):
        _say(f"--- the start geometries ({os.path.relpath(structures, ROOT)})")
        mols_s, _n, fields_s, targets_s, _fs, _t = read(data, structures=structures)
        start_part = measure(mols_s, fields_s, targets_s, fieldset, hammett, rhos, cofield_rhos,
                             folds, seed, scrambles, results, prefix="start_")
        start_part["structures"] = os.path.relpath(structures, ROOT).replace(os.sep, "/")
    else:
        _say("--- no start geometries: that part is left out")

    references = dict(main_part["references"])
    references.update({k: _res(r) for k, r in hammett.items()})
    out = dict(n_samples=len(targets), n_molecules=targets.n_measured_molecules, folds=folds,
               seed=seed, target=TARGET,
               y=dict(low=float(y.min()), high=float(y.max()), spread=float(y.std()),
                      anti=n_anti, imaginary=n_imag),
               hammett_lines=lines, poses={str(k): v for k, v in sorted(poses.items())},
               references=references, rows=main_part["rows"], best=main_part["best"],
               cofields=main_part["cofields"], start_geometry=start_part,
               seconds=time.time() - t_all)
    with io.open(os.path.join(results, "band.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    _say(f"  {out['seconds'] / 60:.1f} min")
    return out


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #

def _iv(d):
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def _reference_table(refs, keys):
    lines = ["| descriptor | q2 | RMSE, cm^-1 | note |", "|---|---:|---:|---|"]
    for k in keys:
        r = refs.get(k)
        if r is not None:
            lines.append(f"| {r['name']} | {r['q2']:+.3f} | {r['error']:.2f} | {r['note']} |")
    return lines


def _scan_table(rows):
    lines = ["| rho, 1/A^2 | width, A | atoms | invariant local q2 | against counts | "
             "against one Hammett line | against the bond length | posed overlap q2 | "
             "against counts | against one Hammett line | against the bond length | "
             "posed against invariant |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        b = row.get("against_bond_length")
        lines.append(f"| {row['rho']:g} | {row['width']:.2f} | {row['atoms']:.1f} | "
                     f"{row['invariant']['q2']:+.3f} | {_iv(row['against_counts']['invariant'])} | "
                     f"{_iv(row['against_one_line']['invariant'])} | "
                     f"{_iv(b['invariant']) if b else ''} | "
                     f"{row['aligned']['q2']:+.3f} | {_iv(row['against_counts']['aligned'])} | "
                     f"{_iv(row['against_one_line']['aligned'])} | "
                     f"{_iv(b['aligned']) if b else ''} | "
                     f"{_iv(row['aligned_against_invariant'])} |")
    return lines


def _cofield_tables(cofields, where):
    lines = []
    for key, c in cofields.items():
        lines += ["", f"## The co-field at the ring positions, {where}, width "
                      f"{envelope_width(c['rho']):.2f} A (rho {c['rho']:g}, posed q2 "
                      f"{c['q2']:+.3f})", "",
                  f"Figure: `{c['figure']}`.", "",
                  "| field | ipso | ortho | meta | para | para minus meta (sd, z) | "
                  "para over meta | null | z |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for fname, e in c["fields"].items():
            f4 = e["four"]
            lines.append(f"| {fname} | {f4['ipso']:+.3g} ({e['sd_ipso']:.2g}) | "
                         f"{f4['ortho']:+.3g} ({e['sd_ortho']:.2g}) | "
                         f"{f4['meta']:+.3g} ({e['sd_meta']:.2g}) | "
                         f"{f4['para']:+.3g} ({e['sd_para']:.2g}) | "
                         f"{e['para_minus_meta']:+.3g} ({e['para_minus_meta_sd']:.2g}, "
                         f"{e['para_minus_meta_z']:+.1f}) | {e['para_over_meta']:+.2f} | "
                         f"{e['para_over_meta_null']:+.2f} +/- {e['para_over_meta_null_sd']:.2f} | "
                         f"{e['para_over_meta_z']:+.1f} |")
    return lines


def report(results=RESULTS):
    path = os.path.join(results, "band.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run this script first")
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    s = d.get("start_geometry")
    lines = ["# The carbonyl band of substituted benzoic acids as a local target (plan item SP2)",
             "",
             f"{d['n_samples']} acids (the 95 substituents of the Hammett set, meta and para to "
             f"the carboxyl group), the carbonyl stretch computed with ORCA at r2SCAN-3c and "
             f"scaled by 0.97, from {d['y']['low']:.0f} to {d['y']['high']:.0f} cm^-1 with a "
             f"spread (standard deviation) of {d['y']['spread']:.1f} cm^-1; "
             f"{d['y']['anti']} acids ended with an anti carboxyl and {d['y']['imaginary']} with "
             f"an imaginary frequency.  {d['folds']} folds holding out whole molecules; the "
             f"ridge chosen on the same folds for every descriptor.  Differences in q2 carry "
             f"their 95 % bootstrap interval over the molecules.",
             "",
             "The measurement is made twice: on the geometries of the quantum optimisation, "
             "which is the calculation that also gives the band, and on the force-field "
             "geometries that calculation started from.  Only the second says what the "
             "structure alone predicts.",
             ""]
    lines += ["## The band against the Hammett constant", "",
              "| acids | n | slope, cm^-1 per sigma unit | r |", "|---|---:|---:|---:|"]
    for k, v in d["hammett_lines"].items():
        lines.append(f"| {k} | {v['n']} | {v['slope']:+.1f} | {v['r']:+.3f} |")
    lines += ["", "## The references, on the geometries of the quantum optimisation", ""]
    lines += _reference_table(d["references"], ("counts", "one_line", "two_lines", "bond_length"))
    lines += ["", "## The field around the carbonyl carbon by envelope width, on the geometries "
                  "of the quantum optimisation", ""]
    lines += _scan_table(d["rows"])
    if s:
        lines += ["", "## The references, on the start geometries (force field)", ""]
        lines += _reference_table(s["references"], ("counts", "bond_length"))
        lines += ["", "## The field around the carbonyl carbon by envelope width, on the start "
                      "geometries (force field)", ""]
        lines += _scan_table(s["rows"])
    lines += ["", "## How the co-field tables are read", "",
              "Values of h_f c_f C_f at the mean position of each ring carbon in the frame of "
              "the carbonyl carbon: what one unit of the field at that place adds to the band, "
              "in cm^-1.  The number in brackets is the standard deviation that the noise in "
              "the bands (taken as the cross-validation error) allows.  *Para minus meta* is "
              "|para| - |meta| with its standard deviation and z-score from the same noise.  "
              "*Para over meta* is (|para| - |meta|) / (|para| + |meta|), with the mean and "
              "standard deviation of a y-scramble null and the z-score against it."]
    lines += _cofield_tables(d["cofields"], "geometries of the quantum optimisation")
    if s:
        lines += _cofield_tables(s["cofields"], "start geometries")
    lines += ["", _verdict(d), ""]
    return "\n".join(lines)


def _best(rows, kind):
    return max(rows, key=lambda r: r[kind]["q2"])


def _verdict(d):
    refs = d["references"]
    ref, one, bond = refs["counts"], refs["one_line"], refs.get("bond_length")
    bi, ba = _best(d["rows"], "invariant"), _best(d["rows"], "aligned")
    di, da = bi["against_counts"]["invariant"], ba["against_counts"]["aligned"]
    parts = [f"**Reading.** One Hammett line predicts the band with q2 {one['q2']:+.3f}. On the "
             f"geometries of the quantum optimisation the count reaches q2 {ref['q2']:+.3f}; the "
             f"invariant local kernel is best at a width of {bi['width']:.2f} A (q2 "
             f"{bi['invariant']['q2']:+.3f}, against the count {_iv(di)}), the posed overlap at "
             f"{ba['width']:.2f} A (q2 {ba['aligned']['q2']:+.3f}, against the count {_iv(da)})."]
    top = max(bi["invariant"]["q2"], ba["aligned"]["q2"])
    if bond is not None:
        parts.append(f"A straight line on the length of the C=O bond of the same geometries "
                     f"reaches q2 {bond['q2']:+.3f}.")
        if bond["q2"] >= top - 0.02 and ba.get("against_bond_length"):
            parts.append(f"The posed overlap at its best width is "
                         f"{_iv(ba['against_bond_length']['aligned'])} against that line, so "
                         f"nearly all of what the narrow field predicts "
                         f"is one bond length. That length comes from the quantum optimisation, "
                         f"the calculation that also gives the band, so this is not a "
                         f"prediction from the structure.")
    s = d.get("start_geometry")
    gate_rows, where = d["rows"], "the geometries of the quantum optimisation"
    cof = d["cofields"]
    if s:
        rs = s["references"]["counts"]
        si, sa = _best(s["rows"], "invariant"), _best(s["rows"], "aligned")
        dsi, dsa = si["against_counts"]["invariant"], sa["against_counts"]["aligned"]
        narrow = [max(r["invariant"]["q2"], r["aligned"]["q2"]) for r in s["rows"]
                  if r["width"] <= 1.0] or [float("nan")]
        parts.append(f"On the start geometries, which hold nothing from the quantum "
                     f"calculation, the count reaches q2 {rs['q2']:+.3f}; the invariant local "
                     f"kernel is best at a width of {si['width']:.2f} A (q2 "
                     f"{si['invariant']['q2']:+.3f}, against the count {_iv(dsi)}, against one "
                     f"Hammett line {_iv(si['against_one_line']['invariant'])}), the posed "
                     f"overlap at {sa['width']:.2f} A (q2 {sa['aligned']['q2']:+.3f}, against "
                     f"the count {_iv(dsa)}). At a width of 1.00 A or less neither kernel is "
                     f"above q2 {max(narrow):+.3f} there.")
        gate_rows, where, cof = s["rows"], "the start geometries", s["cofields"]
    gi, ga = _best(gate_rows, "invariant"), _best(gate_rows, "aligned")
    gate1 = gi["against_counts"]["invariant"][1] > 0 or ga["against_counts"]["aligned"][1] > 0
    low_i, low_a = gi["against_counts"]["invariant"][1], ga["against_counts"]["aligned"][1]
    parts.append(f"The first half of the gate, the local field beating the count by more than "
                 f"the interval, is read on {where} and is " + ("met" if gate1 else "not met")
                 + f": the lower end of the interval is {low_i:+.3f} for the invariant kernel "
                 f"and {low_a:+.3f} for the posed overlap, each at a width chosen as the best "
                 f"of the scan.")
    for c in cof.values():
        e = c["fields"].get("electrostatic")
        if e is not None:
            parts.append(f"At a width of {envelope_width(c['rho']):.2f} A (posed q2 "
                         f"{c['q2']:+.3f}) the electrostatic co-field is "
                         f"{e['four']['meta']:+.3g} at meta and {e['four']['para']:+.3g} at para, "
                         f"in cm^-1 per unit of field; para minus meta in absolute value is "
                         f"{e['para_minus_meta']:+.3g} with a z of {e['para_minus_meta_z']:+.1f} "
                         f"against the noise and {e['para_over_meta_z']:+.1f} against the "
                         f"scramble.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rho", nargs="*", type=float, default=list(RHOS),
                    help="envelope width parameters of the scan, in 1/A^2")
    ap.add_argument("--cofield-rho", nargs="*", type=float, default=list(COFIELD_RHOS),
                    help="width parameters the co-field is read at")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scrambles", type=int, default=SCRAMBLES)
    ap.add_argument("--data", default=DATA,
                    help="the SD file with the bands on the optimised geometries")
    ap.add_argument("--structures", default=STRUCTURES,
                    help="the SD file with the start geometries of the same acids; 'none' "
                         "leaves that part out")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    if a.report is None:
        structures = None if str(a.structures).lower() == "none" else a.structures
        run(rhos=a.rho, cofield_rhos=a.cofield_rho, folds=a.folds, seed=a.seed,
            scrambles=a.scrambles, results=a.results, data=a.data, structures=structures)
    text = report(a.results)
    if a.report in (None, "-"):
        path = os.path.join(a.results, "table.md")
    else:
        path = a.report
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    if a.report == "-":
        print(text)
    else:
        print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
