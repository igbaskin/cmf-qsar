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

Eight things are measured, the first six on folds that hold out whole molecules.

1. **The band against the Hammett constant.**  Before any field: the computed band against
   sigma, as a linear regression (the straight line band = a + rho * sigma, fitted by least
   squares), which is what a physical organic chemist would plot first.  The slope of that
   regression for the meta and for the para acids is the picture the co-field is asked to
   reproduce, and its cross-validated q2 is a second reference beside the count.
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
   (Badger's rule).  A linear regression on this one length is therefore a control: a field
   description that is no better than this regression has read the bond length, which the quantum
   calculation supplied, and not the substituent.
6. **Everything again on the start geometries.**  The same atoms, bonds and bands, with the
   coordinates of `examples/spectra/benzoic_acids.sdf`: the force-field geometries the quantum
   run started from, which hold no information from that run.  This is the measurement that
   answers the question of the item, whether the structure predicts the band.
7. **Indicator fields** (plan item LP3) on the start geometries: one Gaussian field per
   Sybyl atom type, alone and together with the five physico-chemical fields.  The five
   fields do not tell the atom types apart, and the Hammett regressions for meta and para
   separately were not reached with them.  This part repeats the cross-validation with
   other folds, gives every number also for folds that hold out whole substituents (the
   series holds each substituent twice), and reads what one atom of a type adds to the
   band at the place of the first atom of the substituent.  It writes `indicator.json`.
8. **Quantum charges** on the start geometries.  The default charges (Gasteiger) are computed
   from the bonds alone and do not hold the effect of a substituent through the pi electrons
   of the ring (resonance).  This part takes the charge of every atom from one GFN2-xTB
   calculation on the start geometry, without optimisation
   (`examples/spectra/benzoic_acids_xtb.sdf`), and builds the electrostatic field from it.  It
   also adds two fields from the Fukui indices of the same calculation.  The cross-validation
   is the one of part 7.  The width of the envelope and the ridge are chosen inside every
   training part, because the width decides the result.  The references include the
   regressions on the quantum charges of the carboxyl group, which show what the charges
   hold without any field.  It writes `charges.json`.

Usage
-----
    python benchmarks/run_band.py                 # everything, into benchmarks/results/band/
    python benchmarks/run_band.py --rho 0.1 0.25  # a shorter width scan
    python benchmarks/run_band.py --structures none   # without the start geometries
    python benchmarks/run_band.py --only indicator    # part 7 alone
    python benchmarks/run_band.py --only charges      # part 8 alone
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
from cmfqsar.atomtypes import assign_atom_types  # noqa: E402
from cmfqsar.cofields import (atom_effect, atom_effect_covariance,  # noqa: E402
                              cofield_at_points, contrast_at, dual_covariance_factor)
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import (FieldSet, atom_property, atom_type_counts,  # noqa: E402
                            build_fields, cif_types_in)
from cmfqsar.localalign import overlap_kernel, posed_environments, well_posed  # noqa: E402
from cmfqsar.localenv import (LAMS, LocalResult, choose_lam, count_features,  # noqa: E402
                              count_kernel, cross_validate_lams, envelope_width, environment,
                              local_features, local_kernel, paired_interval, types_of)
from cmfqsar.model import krr_fit  # noqa: E402

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
    """The linear regression on sigma as a predictor, out of fold: one equation for all acids,
    or one equation per position."""
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
    name = ("linear regression on the Hammett constant, meta and para separately" if two_lines
            else "linear regression on the Hammett constant, all acids")
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
    """The band as a linear regression on the length of the C=O bond, out of fold.

    A shorter bond has a higher stretching frequency (Badger's rule).  On geometries from a
    quantum optimisation this regression is a control and not a competitor: the length comes from
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
    return _as_result("linear regression on the length of the C=O bond", pred, targets, folds,
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
# 7. indicator fields: one field per atom type (plan item LP3)
# --------------------------------------------------------------------------- #

#: the envelope widths of the indicator part.  The first atom of a substituent is 5.0 A (meta)
#: or 5.8 A (para) from the carbonyl carbon, so only the wide envelopes of the scan keep it.
INDICATOR_RHOS = (0.01, 0.02, 0.05, 0.1)
#: the width the conclusions of the part are stated at, and the width the effect of one atom is
#: read at: 10 A, where the five fields were best on the start geometries.  It is fixed here and
#: not chosen as the best of the indicator scan.
INDICATOR_READ_RHO = 0.01
#: an atom type gets an indicator field when it occurs in this many acids (as in plan item LP3)
INDICATOR_MIN_ACIDS = 10
#: the lower limit of the control that gives a field to nearly every atom type of the series
INDICATOR_ALL_MIN_ACIDS = 2
#: how many times the cross-validation is repeated, each time with other folds
REPEATS = 10
#: the ridge values of the indicator part: `localenv.LAMS` and two larger values, because the
#: posed overlap of the indicator fields chooses the largest value of the shorter list
WIDE_LAMS = tuple(LAMS) + (10.0, 100.0)
#: where the first atom of a substituent can be, in the frame of the carbonyl carbon
EFFECT_PLACES = ("meta (carbonyl side)", "meta (hydroxyl side)", "para")
#: an atom type and a place enter the comparison when this many acids have such a first atom
MIN_FIRST = 2
INDICATOR_SETS = ("five", "indicator", "both", "both_all")


def substituent_targets(targets, table):
    """A copy of `targets` whose unit is the substituent and not the acid.

    The series holds each substituent twice, once meta and once para to the carboxyl group.
    The folds of `targets` hold out whole acids, so the other acid of the same substituent is
    usually in the training set.  In the copy, the array `molecule` holds the number of the
    substituent.  Its folds therefore hold out both acids of a substituent together, and a
    bootstrap over its `molecule` draws substituents.

    The copy is for cross-validation and intervals only.  Kernels and counts are built with
    the original `targets`, because they use `molecule` to find the acid.
    """
    subs = list(table["substituent"])
    ids = {s: i for i, s in enumerate(sorted(set(subs)))}
    return AtomTargets(molecule=np.array([ids[s] for s in subs], int),
                       atom=np.asarray(targets.atom, int), y=np.asarray(targets.y, float),
                       n_mol=len(ids), name=targets.name,
                       per_molecule=list(targets.per_molecule))


def twin_correlation(targets, table):
    """How alike the meta acid and the para acid of one substituent are: the correlation of
    their bands and of their Hammett constants, over the substituents that occur twice."""
    y = np.asarray(targets.y, float)
    sigma = np.asarray(table["sigma"], float)
    pairs = twin_pairs(table)
    if len(pairs) < 3:
        return dict(n=len(pairs), r_band=float("nan"), r_sigma=float("nan"))
    m, p = pairs.T
    return dict(n=len(pairs), r_band=float(np.corrcoef(y[m], y[p])[0, 1]),
                r_sigma=float(np.corrcoef(sigma[m], sigma[p])[0, 1]))


def squared_errors(results, y):
    """The squared error of every sample, as the mean over repeated cross-validations."""
    y = np.asarray(y, float)
    return np.mean([(y - np.asarray(r.predicted, float)) ** 2 for r in results], axis=0)


def repeated_q2(results, y):
    """(mean q2, standard deviation of q2 over the repeats, root mean square error)."""
    y = np.asarray(y, float)
    q = np.array([q2_of(y, r.predicted) for r in results], float)
    sd = float(q.std(ddof=1)) if len(q) > 1 else 0.0
    return float(q.mean()), sd, float(np.sqrt(squared_errors(results, y).mean()))


def paired_repeats(a, b, targets, runs=2000, seed=0):
    """(difference, low, high) of q2 for the descriptor `a` against `b`, where `a` and `b` are
    lists of results of repeated cross-validations and a q2 is the mean over the repeats.

    The interval is the 95 % bootstrap interval over the units of `targets.molecule`: acids,
    or substituents for the copy made by :func:`substituent_targets`.  One draw of units is
    used for every repeat, so the interval describes the sample of acids and not the folds.
    """
    y = np.asarray(targets.y, float)
    gain = squared_errors(b, y) - squared_errors(a, y)
    unit = np.asarray(targets.molecule, int)
    which = np.unique(unit)
    rows = {int(k): np.where(unit == int(k))[0] for k in which}

    def stat(idx):
        return float(gain[idx].sum() / np.sum((y[idx] - y[idx].mean()) ** 2))

    point = stat(np.arange(len(y)))
    if len(which) < 4:
        return (point, float("nan"), float("nan"))
    rng = np.random.RandomState(int(seed))
    out = []
    for _ in range(int(runs)):
        drawn = which[rng.randint(0, len(which), len(which))]
        idx = np.concatenate([rows[int(k)] for k in drawn])
        if np.std(y[idx]) > 0:
            out.append(stat(idx))
    if not out:
        return (point, float("nan"), float("nan"))
    return (point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def kernel_agreement(a, b):
    """The correlation between the off-diagonal elements of two kernels, each centred first.
    Near 1, the two descriptions order the pairs of samples alike and the second one cannot
    add much to the first."""
    n = len(a)
    h = np.eye(n) - 1.0 / n
    upper = np.triu_indices(n, 1)
    return float(np.corrcoef((h @ a @ h)[upper], (h @ b @ h)[upper])[0, 1])


def first_atoms(mols, targets, envs):
    """The first atom of the substituent of every acid: its Sybyl type, its place (para, or
    meta on the side of the carbonyl oxygen or of the hydroxyl oxygen) and its coordinates in
    the posed frame.  `None` for an acid without a substituent."""
    out = []
    for r in range(len(targets)):
        k, c = int(targets.molecule[r]), int(targets.atom[r])
        mol, env = mols[k], envs[r]
        back = {int(a): e for e, a in enumerate(np.asarray(env.member_atom_map, int))}
        atom_c = mol.GetAtomWithIdx(c)
        ipso = [n.GetIdx() for n in atom_c.GetNeighbors() if n.GetIsAromatic()]
        o_carbonyl = [n.GetIdx() for n in atom_c.GetNeighbors()
                      if n.GetSymbol() == "O" and n.GetDegree() == 1]
        found = None
        ring = None
        if len(ipso) == 1 and len(o_carbonyl) == 1 and o_carbonyl[0] in back:
            ring = next((set(rg) for rg in mol.GetRingInfo().AtomRings()
                         if ipso[0] in rg and len(rg) == 6), None)
        if ring is not None:
            dist = Chem.GetDistanceMatrix(mol)
            types = assign_atom_types(mol, "sybyl")
            y_o = float(env.coords[back[o_carbonyl[0]]][1])
            for a in sorted(ring):
                place = {3: "meta", 4: "para"}.get(int(dist[c, a]))
                if place is None:
                    continue
                for n in mol.GetAtomWithIdx(a).GetNeighbors():
                    j = n.GetIdx()
                    if j in ring or n.GetAtomicNum() == 1 or j not in back:
                        continue
                    p = np.asarray(env.coords[back[j]], float)
                    side = "" if place == "para" else (
                        " (carbonyl side)" if p[1] * y_o > 0 else " (hydroxyl side)")
                    found = (types[j], place + side, p)
        out.append(found)
    return out


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def type_effects(model, atable, y, lam, fieldset, rho, sigma_noise, first, table):
    """What one atom of each indicator type adds to the band at the place of the first atom of
    a substituent, in cm^-1 (`cofields.atom_effect`, as in plan item LP3).

    The atom enters with the weight the envelope gives it at that distance.  Beside each effect
    stand the acids that have an atom of that type as the first atom of the substituent at that
    place: how many, their mean band shift against the unsubstituted acid, and their mean
    Hammett constant.  The model was fitted on these bands, so the agreement with the mean shift
    is a check that the model has stored the substituent effects by atom type and place.  It is
    not a prediction.
    """
    y = np.asarray(y, float)
    where = {}
    for f in first:
        if f is not None:
            where.setdefault(f[1], []).append(f[2])
    places = [p for p in EFFECT_PLACES if p in where]
    if not places:
        return None
    pts = np.array([np.mean(where[p], axis=0) for p in places])
    damp = np.exp(-float(rho) * np.sum(pts ** 2, axis=1))
    pos = list(table["position"])
    sigma = np.asarray(table["sigma"], float)
    plain = [r for r in range(len(y)) if pos[r] == "none"]
    band0 = float(y[plain].mean()) if plain else float(y.mean())
    seen = {}
    for r, f in enumerate(first):
        if f is not None:
            seen.setdefault((f[0], f[1]), []).append(r)
    model.params = dict(model.params)
    model.params["lam"] = float(lam)
    model.fit(atable, y, cv_folds=5, cv_seed=0, reuse_kernels=True)
    factor = dual_covariance_factor(model)
    types = {}
    for f, name in enumerate(fieldset.names):
        if not name.startswith("cif:"):
            continue
        t = name[4:]
        row = {}
        for k, place in enumerate(places):
            v = float(atom_effect(model, f, pts[k:k + 1], weight=float(damp[k]))[0])
            cov = atom_effect_covariance(model, f, pts[k:k + 1], weight=float(damp[k]),
                                         sigma=sigma_noise, factor=factor)
            rows = seen.get((t, place), [])
            row[place] = dict(effect=v, sd=float(np.sqrt(max(float(cov[0, 0]), 0.0))),
                              n=len(rows),
                              shift=float(np.mean(y[rows] - band0)) if rows else None,
                              sigma=float(np.nanmean(sigma[rows])) if rows else None)
        types[t] = row
    cells = [e for row in types.values() for e in row.values() if e["n"] >= MIN_FIRST]
    return dict(rho=float(rho), lam=float(lam), sigma_noise=float(sigma_noise),
                unsubstituted_band=band0, min_first=MIN_FIRST,
                places={p: dict(point=[float(x) for x in pts[k]], n=len(where[p]),
                                distance=float(np.linalg.norm(pts[k])), envelope=float(damp[k]))
                        for k, p in enumerate(places)},
                types=types, pairs=len(cells),
                r_shift=_corr([e["effect"] for e in cells], [e["shift"] for e in cells]),
                r_sigma=_corr([e["effect"] for e in cells], [e["sigma"] for e in cells]))


def _repeat(K, tg, folds, seeds, name):
    return [choose_lam(K, tg, lams=WIDE_LAMS, folds=folds, seed=s, name=name) for s in seeds]


def run_indicator(rhos=INDICATOR_RHOS, folds=FOLDS, seed=0, repeats=REPEATS, results=RESULTS,
                  data=DATA, structures=STRUCTURES, read_rho=INDICATOR_READ_RHO, runs=2000):
    """The indicator fields of plan item LP3 on this series, on the start geometries.

    The five physico-chemical fields do not tell the atom types apart.  A continuous indicator
    field of a type is one Gaussian on every atom of that type, so it does.  Three field sets
    are compared with each of the two kernels: the five fields, the indicator fields alone, and
    both together (and, as a control, both together with a field for nearly every type).

    Two things differ from parts 1 to 6.  The cross-validation is repeated `repeats` times with
    other folds, because the q2 of one set of folds moves by about 0.02 with the folds, which is
    as much as the differences in question.  And every number is given for two kinds of folds:
    folds that hold out whole acids, and folds that hold out whole substituents
    (:func:`substituent_targets`).
    """
    t_all = time.time()
    if not structures or not os.path.exists(structures):
        raise SystemExit("the indicator part is measured on the start geometries: give the SD "
                         "file with --structures")
    os.makedirs(results, exist_ok=True)
    mols, names, fields5, targets, fs5, table = read(data, structures=structures)
    y = np.asarray(targets.y, float)
    units = {"acids": targets, "substituents": substituent_targets(targets, table)}
    seeds = [int(seed) + k for k in range(int(repeats))]
    rhos = sorted({float(r) for r in rhos} | {float(read_rho)})
    counts = atom_type_counts(mols, "sybyl")
    cif = cif_types_in(mols, "sybyl", min_molecules=INDICATOR_MIN_ACIDS)
    cif_all = cif_types_in(mols, "sybyl", min_molecules=INDICATOR_ALL_MIN_ACIDS)
    sets = {"five": ("the five physico-chemical fields", fs5),
            "indicator": (f"{len(cif)} indicator fields",
                          FieldSet.create(physchem=(), cif_types=cif)),
            "both": (f"the five fields and {len(cif)} indicator fields",
                     FieldSet.create(cif_types=cif)),
            "both_all": (f"the five fields and {len(cif_all)} indicator fields",
                         FieldSet.create(cif_types=cif_all))}
    twins = twin_correlation(targets, table)
    _say(f"=== indicator fields on {len(targets)} acids ({units['substituents'].n_mol} "
         f"substituents), start geometries: {len(cif)} atom types in at least "
         f"{INDICATOR_MIN_ACIDS} acids, {len(cif_all)} in at least {INDICATOR_ALL_MIN_ACIDS}; "
         f"{len(seeds)} repeats of {folds} folds")
    _say(f"  the meta and the para acid of one substituent: r of the bands "
         f"{twins['r_band']:+.2f}, r of the Hammett constants {twins['r_sigma']:+.2f}, "
         f"{twins['n']} substituents")

    count_k = count_kernel(count_features(fields5, targets, types_of(fields5, min_atoms=10)))
    refs = {}
    for unit, tg in units.items():
        refs[unit] = {
            "counts": _repeat(count_k, tg, folds, seeds, "atom-centred counts"),
            "one_line": [hammett_reference(tg, table, folds, s, two_lines=False)
                         for s in seeds],
            "two_lines": [hammett_reference(tg, table, folds, s, two_lines=True)
                          for s in seeds]}
        for k, v in refs[unit].items():
            q, sd, _e = repeated_q2(v, y)
            _say(f"  whole {unit} held out, {v[0].name}: q2 {q:+.3f} (sd over the repeats "
                 f"{sd:.3f})")

    res, kern, keep = {}, {}, None
    for key, (label, fs) in sets.items():
        t0 = time.time()
        fields = fields5 if key == "five" else [build_fields(m, fs, name=n)
                                                for m, n in zip(mols, names)]
        for rho in rhos:
            plain = [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
                     for r in range(len(targets))]
            k_inv = local_kernel(local_features(plain))
            envs = posed_environments(fields, targets, rho=rho)
            model, atable, k_pos = overlap_kernel(envs, fs)
            kern[key, rho] = {"invariant": k_inv, "aligned": k_pos}
            for unit, tg in units.items():
                for kind, K in kern[key, rho].items():
                    res[key, rho, unit, kind] = _repeat(K, tg, folds, seeds,
                                                        f"{kind}, {label}, rho {rho:g}")
            if key == "both" and rho == float(read_rho):
                keep = (model, atable, envs, fs)
        _say(f"  {label}: {fs.n_fields} fields, {time.time() - t0:.0f} s")

    held = {}
    for unit, tg in units.items():
        rows = []
        for key, (label, fs) in sets.items():
            for rho in rhos:
                row = dict(fields=key, label=label, n_fields=int(fs.n_fields), rho=rho,
                           width=float(envelope_width(rho)))
                for kind in ("invariant", "aligned"):
                    rr = res[key, rho, unit, kind]
                    q, sd, err = repeated_q2(rr, y)
                    cell = dict(q2=q, sd=sd, error=err,
                                q2_by_repeat=[float(r.q2) for r in rr],
                                lam=float(np.median([r.lam for r in rr])))
                    for other, ref in refs[unit].items():
                        cell["against_" + other] = paired_repeats(rr, ref, tg, runs, seed)
                    if key != "five":
                        cell["against_five"] = paired_repeats(
                            rr, res["five", rho, unit, kind], tg, runs, seed)
                    row[kind] = cell
                rows.append(row)
                if rho == float(read_rho):
                    i, a = row["invariant"], row["aligned"]
                    _say(f"  whole {unit} held out, {label}, width {row['width']:.2f} A: "
                         f"invariant q2 {i['q2']:+.3f} ({i['sd']:.3f}), against the separate "
                         f"Hammett regressions {_iv(i['against_two_lines'])}; posed q2 "
                         f"{a['q2']:+.3f} ({a['sd']:.3f})")
        ref_out = {}
        for k, v in refs[unit].items():
            q, sd, err = repeated_q2(v, y)
            ref_out[k] = dict(name=v[0].name, q2=q, sd=sd, error=err,
                              q2_by_repeat=[float(q2_of(y, r.predicted)) for r in v])
        held[unit] = dict(n_units=int(len(np.unique(tg.molecule))), references=ref_out,
                          rows=rows)

    agreement = [dict(rho=rho, width=float(envelope_width(rho)),
                      invariant=kernel_agreement(kern["five", rho]["invariant"],
                                                 kern["indicator", rho]["invariant"]),
                      aligned=kernel_agreement(kern["five", rho]["aligned"],
                                               kern["indicator", rho]["aligned"]))
                 for rho in rhos]
    for a in agreement:
        _say(f"  width {a['width']:.2f} A: the kernel of the indicator fields against the "
             f"kernel of the five fields, r {a['invariant']:+.2f} (invariant), "
             f"{a['aligned']:+.2f} (posed)")

    effects = None
    if keep is not None:
        model, atable, envs, fs = keep
        rr = res["both", float(read_rho), "acids", "aligned"]
        q, sd, err = repeated_q2(rr, y)
        effects = type_effects(model, atable, y, float(np.median([r.lam for r in rr])), fs,
                               float(read_rho), err, first_atoms(mols, targets, envs), table)
        if effects is not None:
            effects.update(q2=q, fields="both", width=float(envelope_width(read_rho)))
            _say(f"  the effect of one atom of a type at the place of the first atom of the "
                 f"substituent, against the mean observed shift: r {effects['r_shift']:+.2f} "
                 f"over {effects['pairs']} pairs; against the mean Hammett constant: r "
                 f"{effects['r_sigma']:+.2f}")

    out = dict(n_samples=len(targets), n_substituents=int(units["substituents"].n_mol),
               folds=int(folds), seed=int(seed), repeats=len(seeds), runs=int(runs),
               lams=[float(x) for x in WIDE_LAMS], target=TARGET,
               structures=os.path.relpath(structures, ROOT).replace(os.sep, "/"),
               read_rho=float(read_rho), read_width=float(envelope_width(read_rho)),
               min_acids=INDICATOR_MIN_ACIDS, min_acids_all=INDICATOR_ALL_MIN_ACIDS,
               types=list(cif), types_all=list(cif_all),
               type_counts={t: int(n) for t, n in counts.items()},
               twins=twins, held_out=held, kernel_agreement=agreement, effects=effects,
               seconds=time.time() - t_all)
    with io.open(os.path.join(results, "indicator.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    _say(f"  {out['seconds'] / 60:.1f} min")
    return out


# --------------------------------------------------------------------------- #
# 8. quantum charges on the start geometries
# --------------------------------------------------------------------------- #

#: the start geometries with the GFN2-xTB charge and the Fukui functions of every atom, from
#: one calculation per acid without an optimisation (`examples/make_band_examples.py --xtb`)
QUANTUM = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_xtb.sdf")
QUANTUM_CHARGE = "q_xtb"
QUANTUM_FUKUI = ("fukui_plus", "fukui_minus")
#: the widths of the charge part: 10, 4.5, 2 and 1 A.  The conclusions are stated at 10 A, as
#: in the indicator part.  At 1 A the envelope keeps the carboxyl group and little else.
CHARGE_RHOS = (0.01, 0.05, 0.25, 1.0)
CARBOXYL_ATOMS = ("carbonyl carbon", "carbonyl oxygen", "hydroxyl oxygen", "hydroxyl hydrogen")
CHARGE_SETS = ("five", "es", "es_q", "five_q", "five_q_fukui", "five_q_indicator")
CHARGE_REFERENCES = ("counts", "one_line", "two_lines", "oxygen_charge", "carboxyl_charges")
#: the number of folds inside a training part, for the choice of the width and of the ridge
INNER_FOLDS = 5


def with_quantum_charges(mols, names, path, charge=QUANTUM_CHARGE, extra=QUANTUM_FUKUI,
                         tolerance=1e-3):
    """Copies of `mols` whose atoms carry the charges of the SD file `path`.

    The charge of the file (atom property `charge`) is set on every atom as `partial_charge`,
    which is the property a field set with `charge_model="file"` reads.  The properties
    `extra` are set on every atom under their own names.

    The file must hold the same atoms in the same order at the same places, to `tolerance`
    in A.  A charge computed on another geometry would bring information about that geometry
    into the description, which is the fault part 5 measures for the bond length.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run examples/make_band_examples.py --xtb")
    source = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(path, removeHs=False)
              if m is not None}
    out = []
    for m, name in zip(mols, names):
        s = source.get(name)
        if s is None:
            raise SystemExit(f"{path} has no structure named {name}")
        if [a.GetAtomicNum() for a in s.GetAtoms()] != [a.GetAtomicNum() for a in m.GetAtoms()]:
            raise SystemExit(f"{name}: the atoms of {path} are in another order")
        moved = float(np.abs(s.GetConformer().GetPositions()
                             - m.GetConformer().GetPositions()).max())
        if moved > tolerance:
            raise SystemExit(f"{name}: the geometry of {path} differs by {moved:.3f} A from "
                             f"the geometry the fields are built on")
        values = {p: atom_property(s, p, mol_name=name) for p in (charge,) + tuple(extra)}
        c = Chem.Mol(m)
        for a in c.GetAtoms():
            a.SetDoubleProp("partial_charge", float(values[charge][a.GetIdx()]))
            for p in extra:
                a.SetDoubleProp(p, float(values[p][a.GetIdx()]))
        out.append(c)
    return out


def carboxyl_atoms(mol, carbon):
    """The numbers of the carbonyl carbon, the carbonyl oxygen, the hydroxyl oxygen and the
    hydroxyl hydrogen of the carboxyl group at `carbon`, or None when it is not such a group."""
    atom = mol.GetAtomWithIdx(int(carbon))
    oxygens = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
    one = [n for n in oxygens if n.GetDegree() == 1]
    two = [n for n in oxygens if n.GetDegree() == 2]
    if len(one) != 1 or len(two) != 1:
        return None
    h = [n for n in two[0].GetNeighbors() if n.GetSymbol() == "H"]
    if len(h) != 1:
        return None
    return int(carbon), one[0].GetIdx(), two[0].GetIdx(), h[0].GetIdx()


def carboxyl_charges(mols, targets, charges):
    """The charges of the four atoms of the carboxyl group of every sample, in the order of
    `CARBOXYL_ATOMS`.  `charges` holds one array per molecule.  NaN where the target atom is
    not the carbon of a carboxyl group."""
    out = np.full((len(targets), len(CARBOXYL_ATOMS)), np.nan)
    for r in range(len(targets)):
        k = int(targets.molecule[r])
        idx = carboxyl_atoms(mols[k], targets.atom[r])
        if idx is not None:
            out[r] = np.asarray(charges[k], float)[list(idx)]
    return out


def regression_reference(X, targets, folds=FOLDS, seed=0, name="linear regression", note=""):
    """The band as a linear regression on the columns of `X`, fitted by least squares on the
    training acids of every fold and predicted for the acids the fold holds out.  With one
    column it is a simple linear regression, with several a multiple linear regression."""
    y = np.asarray(targets.y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    ok = np.all(np.isfinite(X), axis=1)
    pred = np.full(len(y), np.nan)
    for te in targets.folds(folds, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        tr = tr[ok[tr]]
        if len(tr) < X.shape[1] + 2:
            continue
        mean, scale = X[tr].mean(axis=0), X[tr].std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        coef = np.linalg.lstsq(np.c_[np.ones(len(tr)), (X[tr] - mean) / scale], y[tr],
                               rcond=None)[0]
        rows = te[ok[te]]
        pred[rows] = np.c_[np.ones(len(rows)), (X[rows] - mean) / scale] @ coef
    pred = np.where(np.isfinite(pred), pred, np.nanmean(y))
    return _as_result(name, pred, targets, folds, note=note)


def nested_choice(kernels, targets, lams=WIDE_LAMS, folds=FOLDS, seed=0, inner=INNER_FOLDS,
                  name="kernel"):
    """Out-of-fold predictions with the kernel and the ridge chosen inside every training part.

    `kernels` maps a key (here the decay of the envelope, which sets its width) to a kernel
    over the same samples.  For every fold, the training part is divided again into `inner`
    folds over the same unit as the outer folds (acids, or substituents).  The kernel and the
    ridge with the lowest error on these inner folds are chosen, the model is fitted on the
    whole training part, and the held-out samples are predicted.  The held-out samples take
    no part in the choice, so the q2 is not raised by the search, as it is in
    :func:`cmfqsar.localenv.choose_lam`.

    Returns the result and the list of the choices, one (key, ridge) per fold.
    """
    y = np.asarray(targets.y, float)
    keys, lams = list(kernels), [float(x) for x in lams]
    pred = np.full(len(y), np.nan)
    picks = []
    for f, te in enumerate(targets.folds(folds, seed)):
        tr = np.setdiff1d(np.arange(len(y)), te)
        sub = targets.subset(tr)
        best = None
        for key in keys:
            K = np.asarray(kernels[key], float)[np.ix_(tr, tr)]
            for r in cross_validate_lams(K, sub, lams, inner, 1000 * (int(seed) + 1) + f):
                if np.isfinite(r.q2) and (best is None or r.q2 > best[0]):
                    best = (float(r.q2), key, float(r.lam))
        if best is None:
            continue
        _q, key, lam = best
        K = np.asarray(kernels[key], float)
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam)
        pred[te] = K[np.ix_(te, tr)] @ a + b
        picks.append((key, lam))
    pred = np.where(np.isfinite(pred), pred, np.nanmean(y))
    return _as_result(name, pred, targets, folds,
                      note=f"the width and the ridge chosen on {inner} folds inside every "
                           f"training part"), picks


def twin_pairs(table):
    """The rows of the meta acid and of the para acid of every substituent that occurs twice,
    as an array with one pair per line."""
    pos = list(table["position"])
    rows = {}
    for r, s in enumerate(table["substituent"]):
        if pos[r] in ("meta", "para"):
            rows.setdefault(s, {})[pos[r]] = r
    pairs = [(v["meta"], v["para"]) for v in rows.values() if len(v) == 2]
    return np.array(pairs, int).reshape(-1, 2)


def _number(x):
    """A float for the result file, or None where the value is not defined."""
    return float(x) if x is not None and np.isfinite(x) else None


def twin_difference(results, targets, table):
    """How well a model predicts the difference between the para acid and the meta acid of one
    substituent.

    The two acids hold the same atoms.  They differ in the place of the substituent, and the
    band differs because the effect of a substituent through the pi electrons of the ring
    (resonance) reaches the para place and hardly the meta place.  The prediction of an acid
    is its mean over the repeated cross-validations in `results`.  Returns the number of
    substituents, the correlation between the predicted and the computed difference, the root
    mean square error of the predicted difference, the q2 of the difference, and the standard
    deviation of the computed difference.
    """
    y = np.asarray(targets.y, float)
    pred = np.mean([np.asarray(r.predicted, float) for r in results], axis=0)
    pairs = twin_pairs(table)
    if len(pairs) < 3:
        return dict(n=int(len(pairs)), r=None, error=None, q2=None, spread=None)
    m, p = pairs.T
    seen, got = y[p] - y[m], pred[p] - pred[m]
    return dict(n=int(len(pairs)), r=_number(_corr(seen, got)), error=rmse_of(seen, got),
                q2=_number(q2_of(seen, got)), spread=float(np.std(seen)))


def position_errors(results, y, table):
    """The root mean square error over the meta acids and over the para acids, from the
    squared errors of the repeated cross-validations."""
    err = squared_errors(results, y)
    pos = np.asarray(table["position"])
    return {p: float(np.sqrt(err[pos == p].mean())) if np.any(pos == p) else None
            for p in ("meta", "para")}


def largest_errors(results, y, names, n=3):
    """The `n` samples with the largest squared error, as the mean over repeated
    cross-validations: the name, the root mean square error of the sample, and the share of
    the sample in the sum of the squared errors of all samples."""
    err = squared_errors(results, y)
    total = float(err.sum())
    return [dict(name=str(names[r]), error=float(np.sqrt(err[r])),
                 share=float(err[r] / total) if total > 0 else 0.0)
            for r in np.argsort(-err)[:int(n)]]


def without_rows(results, targets, rows, folds=FOLDS):
    """The results and the targets over the samples that are not in `rows`.

    Nothing is fitted again.  The predictions are the ones the cross-validation made with
    every sample in its place, and only the score leaves the rows out.  This shows how much
    of a q2 is decided by a few samples.
    """
    keep = np.setdiff1d(np.arange(len(targets)), np.asarray(rows, int))
    sub = targets.subset(keep)
    return [_as_result(r.name, np.asarray(r.predicted, float)[keep], sub, folds)
            for r in results], sub


def charge_statistics(Q, y, table):
    """What the charges of the carboxyl group say about the band, atom by atom: mean and
    standard deviation of the charge over the acids, its correlation with the band and with
    the Hammett constant (all acids, meta acids, para acids), and the correlation of the
    difference between the para acid and the meta acid of one substituent with the same
    difference of the bands."""
    y = np.asarray(y, float)
    sigma = np.asarray(table["sigma"], float)
    pos = np.asarray(table["position"])
    pairs = twin_pairs(table)
    out = {}
    for j, label in enumerate(CARBOXYL_ATOMS):
        q = Q[:, j]
        ok = np.isfinite(q) & np.isfinite(sigma)
        row = dict(mean=_number(np.nanmean(q)), sd=_number(np.nanstd(q)),
                   r_band=_number(_corr(q[np.isfinite(q)], y[np.isfinite(q)])),
                   r_sigma=_number(_corr(q[ok], sigma[ok])))
        for p in ("meta", "para"):
            row["r_sigma_" + p] = _number(_corr(q[ok & (pos == p)], sigma[ok & (pos == p)]))
        if len(pairs) >= 3:
            m, p = pairs.T
            row["r_twin"] = _number(_corr(q[p] - q[m], y[p] - y[m]))
        else:
            row["r_twin"] = None
        out[label] = row
    return out


def run_charges(rhos=CHARGE_RHOS, folds=FOLDS, seed=0, repeats=REPEATS, results=RESULTS,
                data=DATA, structures=STRUCTURES, quantum=QUANTUM,
                read_rho=INDICATOR_READ_RHO, runs=2000):
    """The electrostatic field from quantum charges, on the start geometries.

    The electrostatic field of the five fields is built from Gasteiger charges.  That scheme
    moves charge along the sigma bonds only, so it does not hold the effect of a substituent
    through the pi electrons of the ring (resonance).  Here the charges are taken from one
    GFN2-xTB calculation per acid on the start geometry.  The geometry is not optimised, so
    the description holds nothing about the geometry of the ORCA run that gives the band.

    The charge of the carbonyl oxygen is one computed number that could hold the whole answer,
    as the length of the C=O bond did in part 5.  The references therefore include the linear
    regression on that charge and the multiple linear regression on the charges of the four
    atoms of the carboxyl group.  A field with quantum charges that is no better than these
    regressions has used the charges of the carboxyl group and not the field of the acid.

    As in part 7, the cross-validation is repeated with other folds, and every number is given
    for folds that hold out whole acids and for folds that hold out whole substituents.

    The width of the envelope decides the result here, so it is not chosen by the result.  The
    rows at one width keep the convention of the other parts (the ridge with the best q2).  A
    second set of rows chooses the width and the ridge inside every training part
    (:func:`nested_choice`), and these are the rows that the verdict reads.
    """
    t_all = time.time()
    if not structures or not os.path.exists(structures):
        raise SystemExit("the charge part is measured on the start geometries: give the SD "
                         "file with --structures")
    os.makedirs(results, exist_ok=True)
    mols, names, fields5, targets, fs5, table = read(data, structures=structures)
    qmols = with_quantum_charges(mols, names, quantum)
    y = np.asarray(targets.y, float)
    units = {"acids": targets, "substituents": substituent_targets(targets, table)}
    seeds = [int(seed) + k for k in range(int(repeats))]
    rhos = sorted({float(r) for r in rhos} | {float(read_rho)})
    cif = cif_types_in(mols, "sybyl", min_molecules=INDICATOR_MIN_ACIDS)
    es = ("electrostatic",)
    sets = {"five": ("the five fields, Gasteiger charges", fs5, mols),
            "es": ("the electrostatic field alone, Gasteiger charges",
                   FieldSet.create(physchem=es), mols),
            "es_q": ("the electrostatic field alone, quantum charges",
                     FieldSet.create(physchem=es, charge_model="file"), qmols),
            "five_q": ("the five fields, quantum charges",
                       FieldSet.create(charge_model="file"), qmols),
            "five_q_fukui": ("the five fields with quantum charges and the two Fukui fields",
                             FieldSet.create(charge_model="file",
                                             atom_properties=QUANTUM_FUKUI), qmols),
            "five_q_indicator": (f"the five fields with quantum charges and {len(cif)} "
                                 f"indicator fields",
                                 FieldSet.create(charge_model="file", cif_types=cif), qmols)}
    built = {"five": fields5}
    for key, (label, fs, src) in sets.items():
        if key not in built:
            built[key] = [build_fields(m, fs, name=n) for m, n in zip(src, names)]
        if fs.charge_model == "file":
            bad = [n for n, f in zip(names, built[key]) if f.charge_source != "file"]
            if bad:
                raise SystemExit(f"{label}: the charges of {bad[0]} were not read from the "
                                 f"file ({len(bad)} acids)")
    q_g = carboxyl_charges(mols, targets, [f.charges for f in built["five"]])
    q_x = carboxyl_charges(qmols, targets, [f.charges for f in built["five_q"]])
    carboxyl = dict(gasteiger=charge_statistics(q_g, y, table),
                    quantum=charge_statistics(q_x, y, table))
    X4 = np.c_[np.ones(len(y)), q_x]
    fit = y - X4 @ np.linalg.lstsq(X4, y, rcond=None)[0]
    carboxyl["fit_r2_four_quantum"] = float(1.0 - np.sum(fit ** 2) / np.sum((y - y.mean()) ** 2))
    twins = twin_correlation(targets, table)
    _say(f"=== quantum charges on {len(targets)} acids ({units['substituents'].n_mol} "
         f"substituents), start geometries, charges from "
         f"{os.path.relpath(quantum, ROOT)}; {len(seeds)} repeats of {folds} folds")
    for label in CARBOXYL_ATOMS:
        x, g = carboxyl["quantum"][label], carboxyl["gasteiger"][label]
        _say(f"  {label}: quantum charge {x['mean']:+.3f} (sd {x['sd']:.4f}), r with the band "
             f"{_r(x['r_band'])}, with the Hammett constant {_r(x['r_sigma'])}; Gasteiger "
             f"charge {g['mean']:+.3f} (sd {g['sd']:.4f}), r with the band {_r(g['r_band'])}")

    count_k = count_kernel(count_features(fields5, targets, types_of(fields5, min_atoms=10)))
    oxygen = CARBOXYL_ATOMS.index("carbonyl oxygen")
    refs = {}
    for unit, tg in units.items():
        refs[unit] = {
            "counts": _repeat(count_k, tg, folds, seeds, "atom-centred counts"),
            "one_line": [hammett_reference(tg, table, folds, s, two_lines=False)
                         for s in seeds],
            "two_lines": [hammett_reference(tg, table, folds, s, two_lines=True)
                          for s in seeds],
            "oxygen_charge": [regression_reference(
                q_x[:, oxygen], tg, folds, s,
                "linear regression on the quantum charge of the carbonyl oxygen")
                for s in seeds],
            "carboxyl_charges": [regression_reference(
                q_x, tg, folds, s,
                "multiple linear regression on the quantum charges of the four atoms of the "
                "carboxyl group") for s in seeds]}
        for k, v in refs[unit].items():
            q, sd, _e = repeated_q2(v, y)
            _say(f"  whole {unit} held out, {v[0].name}: q2 {q:+.3f} (sd over the repeats "
                 f"{sd:.3f})")

    res, kern = {}, {}
    for key, (label, fs, _src) in sets.items():
        t0 = time.time()
        fields = built[key]
        for rho in rhos:
            plain = [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
                     for r in range(len(targets))]
            envs = posed_environments(fields, targets, rho=rho)
            kern[key, rho] = {"invariant": local_kernel(local_features(plain)),
                              "aligned": overlap_kernel(envs, fs)[2]}
            for unit, tg in units.items():
                for kind, K in kern[key, rho].items():
                    res[key, rho, unit, kind] = _repeat(K, tg, folds, seeds,
                                                        f"{kind}, {label}, rho {rho:g}")
        _say(f"  {label}: {fs.n_fields} fields, {time.time() - t0:.0f} s")

    t0 = time.time()
    nest = {}
    for key, (label, _fs, _src) in sets.items():
        for unit, tg in units.items():
            for kind in ("invariant", "aligned"):
                by_rho = {rho: kern[key, rho][kind] for rho in rhos}
                nest[key, unit, kind] = [nested_choice(by_rho, tg, WIDE_LAMS, folds, s,
                                                       name=f"{kind}, {label}")
                                         for s in seeds]
    _say(f"  the width and the ridge chosen inside the folds: {time.time() - t0:.0f} s")

    # The substituent that holds the largest part of the squared error of the five fields
    # with quantum charges.  Every nested row and every reference is also scored without
    # its acids, so that the table shows what one substituent decides.
    sample = [names[int(m)] for m in targets.molecule]
    subst = np.asarray([str(s) for s in table["substituent"]])
    err = squared_errors([r for r, _p in nest["five_q", "substituents", "invariant"]], y)
    share = {s: float(err[subst == s].sum() / err.sum()) for s in np.unique(subst)}
    worst = max(share, key=share.get)
    out_rows = np.where(subst == worst)[0]
    top = np.array([float(np.max(f.charges)) for f in built["five_q"]])
    top = top[np.asarray(targets.molecule, int)]
    rest = np.setdiff1d(np.arange(len(y)), out_rows)
    outlier = dict(substituent=worst, acids=[sample[r] for r in out_rows],
                   share=share[worst], largest_charge=float(top[out_rows].max()),
                   largest_charge_of_the_others=float(top[rest].max()),
                   of_the_acid=sample[int(rest[np.argmax(top[rest])])])
    _say(f"  the substituent with the largest part of the squared error of the five fields "
         f"with quantum charges (whole substituents held out, width chosen inside the "
         f"folds): {worst}, {100 * share[worst]:.0f} %; its largest quantum charge is "
         f"{outlier['largest_charge']:+.2f} e, the largest of the other acids "
         f"{outlier['largest_charge_of_the_others']:+.2f} e ({outlier['of_the_acid']})")

    def score_without(rr, tg, ref=None):
        cut, sub = without_rows(rr, tg, out_rows, folds)
        q, sd, e = repeated_q2(cut, sub.y)
        cell = dict(q2=q, sd=sd, error=e)
        for other in (ref or ()):
            cell["against_" + other] = paired_repeats(
                cut, without_rows(refs_of[other], tg, out_rows, folds)[0], sub, runs, seed)
        return cell

    #: what each field set is compared with, besides the references
    base = {"es_q": ("es",), "five_q": ("five",), "five_q_fukui": ("five", "five_q"),
            "five_q_indicator": ("five", "five_q"), "es": ("five",)}
    held = {}
    for unit, tg in units.items():
        refs_of = refs[unit]
        rows = []
        for key, (label, fs, _src) in sets.items():
            for rho in rhos:
                row = dict(fields=key, label=label, n_fields=int(fs.n_fields), rho=rho,
                           width=float(envelope_width(rho)))
                for kind in ("invariant", "aligned"):
                    rr = res[key, rho, unit, kind]
                    q, sd, err = repeated_q2(rr, y)
                    cell = dict(q2=q, sd=sd, error=err, by_position=position_errors(rr, y, table),
                                q2_by_repeat=[float(r.q2) for r in rr],
                                lam=float(np.median([r.lam for r in rr])),
                                twins=twin_difference(rr, tg, table))
                    for other, ref in refs[unit].items():
                        cell["against_" + other] = paired_repeats(rr, ref, tg, runs, seed)
                    for other in base.get(key, ()):
                        cell["against_" + other] = paired_repeats(
                            rr, res[other, rho, unit, kind], tg, runs, seed)
                    row[kind] = cell
                rows.append(row)
                if rho == float(read_rho):
                    i, a = row["invariant"], row["aligned"]
                    _say(f"  whole {unit} held out, {label}, width {row['width']:.2f} A: "
                         f"invariant q2 {i['q2']:+.3f} ({i['sd']:.3f}), against the separate "
                         f"Hammett regressions {_iv(i['against_two_lines'])}; posed q2 "
                         f"{a['q2']:+.3f} ({a['sd']:.3f})")
        chosen = []
        for key, (label, fs, _src) in sets.items():
            row = dict(fields=key, label=label, n_fields=int(fs.n_fields))
            for kind in ("invariant", "aligned"):
                rr = [r for r, _p in nest[key, unit, kind]]
                picks = [p for _r, pp in nest[key, unit, kind] for p in pp]
                q, sd, err = repeated_q2(rr, y)
                cell = dict(q2=q, sd=sd, error=err, by_position=position_errors(rr, y, table),
                            q2_by_repeat=[float(r.q2) for r in rr],
                            lam=float(np.median([p[1] for p in picks])),
                            widths={f"{envelope_width(rho):.2f}":
                                    int(sum(1 for p in picks if p[0] == rho)) for rho in rhos},
                            twins=twin_difference(rr, tg, table),
                            largest=largest_errors(rr, y, sample),
                            without=score_without(rr, tg, ("two_lines", "carboxyl_charges")))
                for other, ref in refs[unit].items():
                    cell["against_" + other] = paired_repeats(rr, ref, tg, runs, seed)
                for other in base.get(key, ()):
                    cell["against_" + other] = paired_repeats(
                        rr, [r for r, _p in nest[other, unit, kind]], tg, runs, seed)
                row[kind] = cell
            chosen.append(row)
            i, a = row["invariant"], row["aligned"]
            _say(f"  whole {unit} held out, {label}, width chosen inside the folds: "
                 f"invariant q2 {i['q2']:+.3f} ({i['sd']:.3f}), against the separate Hammett "
                 f"regressions {_iv(i['against_two_lines'])}, widths {i['widths']}; posed q2 "
                 f"{a['q2']:+.3f} ({a['sd']:.3f}), {_iv(a['against_two_lines'])}")
            w = i["without"]
            _say(f"    largest error {i['largest'][0]['name']} "
                 f"({i['largest'][0]['error']:.1f} cm^-1, {100 * i['largest'][0]['share']:.0f} "
                 f"% of the squared error); scored without {worst}: q2 {w['q2']:+.3f} "
                 f"({w['sd']:.3f}), against the separate Hammett regressions "
                 f"{_iv(w['against_two_lines'])}; posed, largest error "
                 f"{a['largest'][0]['name']} ({a['largest'][0]['error']:.1f} cm^-1)")
        ref_out = {}
        for k, v in refs[unit].items():
            q, sd, err = repeated_q2(v, y)
            ref_out[k] = dict(name=v[0].name, q2=q, sd=sd, error=err,
                              by_position=position_errors(v, y, table),
                              q2_by_repeat=[float(q2_of(y, r.predicted)) for r in v],
                              twins=twin_difference(v, tg, table),
                              largest=largest_errors(v, y, sample),
                              without=score_without(v, tg))
        held[unit] = dict(n_units=int(len(np.unique(tg.molecule))), references=ref_out,
                          rows=rows, chosen=chosen)

    agreement = []
    for rho in rhos:
        row = dict(rho=rho, width=float(envelope_width(rho)))
        for name, (a, b) in (("electrostatic", ("es_q", "es")), ("five", ("five_q", "five"))):
            for kind in ("invariant", "aligned"):
                row[f"{name}_{kind}"] = kernel_agreement(kern[a, rho][kind], kern[b, rho][kind])
        agreement.append(row)
        _say(f"  width {row['width']:.2f} A: the kernel with quantum charges against the kernel "
             f"with Gasteiger charges, electrostatic field alone r "
             f"{row['electrostatic_invariant']:+.2f}, five fields r "
             f"{row['five_invariant']:+.2f} (invariant)")

    sigma = np.asarray(table["sigma"], float)
    pairs = twin_pairs(table)
    if len(pairs) >= 3:
        twins["r_sigma_band"] = _number(_corr(sigma[pairs[:, 1]] - sigma[pairs[:, 0]],
                                              y[pairs[:, 1]] - y[pairs[:, 0]]))
        twins["spread"] = float(np.std(y[pairs[:, 1]] - y[pairs[:, 0]]))
        twins["mean"] = float(np.mean(y[pairs[:, 1]] - y[pairs[:, 0]]))
    first = next(iter(Chem.SDMolSupplier(quantum, removeHs=False)))
    level = first.GetProp("qm_level") if first is not None and first.HasProp("qm_level") else ""
    out = dict(n_samples=len(targets), n_substituents=int(units["substituents"].n_mol),
               folds=int(folds), seed=int(seed), repeats=len(seeds), runs=int(runs),
               lams=[float(x) for x in WIDE_LAMS], target=TARGET,
               structures=os.path.relpath(structures, ROOT).replace(os.sep, "/"),
               quantum=os.path.relpath(quantum, ROOT).replace(os.sep, "/"),
               quantum_level=level, charge=QUANTUM_CHARGE, fukui=list(QUANTUM_FUKUI),
               read_rho=float(read_rho), read_width=float(envelope_width(read_rho)),
               widths=[float(envelope_width(r)) for r in rhos], inner_folds=INNER_FOLDS,
               types=list(cif), carboxyl=carboxyl, twins=twins, outlier=outlier,
               held_out=held,
               kernel_agreement=agreement, seconds=time.time() - t_all)
    with io.open(os.path.join(results, "charges.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    _say(f"  {out['seconds'] / 60:.1f} min")
    return out


def _r(value):
    """A correlation for a message or a table: the number, or a dash where it is not defined
    (a quantity that does not vary has no correlation)."""
    return "-" if value is None else f"{value:+.3f}"


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
             "against the Hammett regression | against the bond length | posed overlap q2 | "
             "against counts | against the Hammett regression | against the bond length | "
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


def report(results=RESULTS, table=None):
    """The table of the stored result as Markdown text.  The text is also written to `table`,
    by default `table.md` in the results folder, so that `run_all.py` can rebuild the table
    from `band.json` and compare it with the committed one.  The part on the indicator
    fields is added when `indicator.json` is in the results folder, and the part on the
    quantum charges when `charges.json` is there."""
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
    extra = os.path.join(results, "indicator.json")
    if os.path.exists(extra):
        with io.open(extra, encoding="utf-8") as fh:
            lines += _indicator_section(json.load(fh)) + [""]
    extra = os.path.join(results, "charges.json")
    if os.path.exists(extra):
        with io.open(extra, encoding="utf-8") as fh:
            lines += _charge_section(json.load(fh)) + [""]
    text = "\n".join(lines)
    with io.open(table or os.path.join(results, "table.md"), "w", encoding="utf-8",
                 newline="\n") as fh:
        fh.write(text)
    return text


def _best(rows, kind):
    return max(rows, key=lambda r: r[kind]["q2"])


def _verdict(d):
    refs = d["references"]
    ref, one, bond = refs["counts"], refs["one_line"], refs.get("bond_length")
    bi, ba = _best(d["rows"], "invariant"), _best(d["rows"], "aligned")
    di, da = bi["against_counts"]["invariant"], ba["against_counts"]["aligned"]
    parts = [f"**Reading.** A linear regression on the Hammett constant, with one equation for "
             f"all acids (the Hammett regression), predicts the band with q2 "
             f"{one['q2']:+.3f}. On the "
             f"geometries of the quantum optimisation the count reaches q2 {ref['q2']:+.3f}; the "
             f"invariant local kernel is best at a width of {bi['width']:.2f} A (q2 "
             f"{bi['invariant']['q2']:+.3f}, against the count {_iv(di)}), the posed overlap at "
             f"{ba['width']:.2f} A (q2 {ba['aligned']['q2']:+.3f}, against the count {_iv(da)})."]
    top = max(bi["invariant"]["q2"], ba["aligned"]["q2"])
    if bond is not None:
        parts.append(f"A linear regression on the length of the C=O bond of the same geometries "
                     f"reaches q2 {bond['q2']:+.3f}.")
        if bond["q2"] >= top - 0.02 and ba.get("against_bond_length"):
            parts.append(f"The posed overlap at its best width is "
                         f"{_iv(ba['against_bond_length']['aligned'])} against that regression, so "
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
                     f"{si['invariant']['q2']:+.3f}, against the count {_iv(dsi)}, against the "
                     f"Hammett regression {_iv(si['against_one_line']['invariant'])}), the posed "
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


def _q(cell):
    return f"{cell['q2']:+.3f} ({cell['sd']:.3f})"


def _indicator_rows(rows):
    lines = ["| fields | number of fields | width, A | invariant local q2 (sd over the repeats) | "
             "against the atom-centred counts | against the five fields | against the Hammett "
             "regression, all acids | against the Hammett regressions, meta and para "
             "separately | posed overlap q2 (sd over the repeats) | against the five fields | "
             "against the Hammett regressions, meta and para separately |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        i, a = row["invariant"], row["aligned"]
        lines.append(f"| {row['label']} | {row['n_fields']} | {row['width']:.2f} | {_q(i)} | "
                     f"{_iv(i['against_counts'])} | "
                     f"{_iv(i['against_five']) if 'against_five' in i else ''} | "
                     f"{_iv(i['against_one_line'])} | {_iv(i['against_two_lines'])} | {_q(a)} | "
                     f"{_iv(a['against_five']) if 'against_five' in a else ''} | "
                     f"{_iv(a['against_two_lines'])} |")
    return lines


def _row_at(held, fields, rho):
    return next(r for r in held["rows"] if r["fields"] == fields and r["rho"] == rho)


def _indicator_section(d):
    """The part of the table that `indicator.json` gives."""
    n15, n25 = len(d["types"]), len(d["types_all"])
    extra = [t for t in d["types_all"] if t not in d["types"]]
    lines = ["## Indicator fields: one field per atom type (plan item LP3), on the start "
                 "geometries", "",
             f"A continuous indicator field of an atom type is one Gaussian on every atom of "
             f"that type.  {n15} Sybyl types occur in at least {d['min_acids']} acids "
             f"({', '.join(d['types'])}) and each gets a field.  The last field set of each "
             f"table is a control with a field for every type that occurs in at least "
             f"{d['min_acids_all']} acids ({n25} types; it adds {', '.join(extra)}).",
             "",
             f"The cross-validation is repeated {d['repeats']} times with other folds.  A q2 "
             f"is the mean over the repeats, and the number in brackets is its standard "
             f"deviation over the repeats.  The ridge is chosen on the folds of each repeat, "
             f"for every descriptor alike.  A difference in q2 carries its 95 % bootstrap "
             f"interval over the units that the folds hold out.",
             "",
             f"The series holds each substituent twice, meta and para to the carboxyl group "
             f"({d['n_substituents']} substituents in {d['n_samples']} acids).  Over the "
             f"{d['twins']['n']} substituents that occur twice, the band of the meta acid and "
             f"the band of the para acid correlate with r {d['twins']['r_band']:+.2f}, and "
             f"their Hammett constants with r {d['twins']['r_sigma']:+.2f}.  A fold that holds "
             f"out one acid therefore usually keeps the other acid of the same substituent in "
             f"the training set.  The tables are given for both kinds of folds."]
    by = d["held_out"]["acids"]["references"]["counts"].get("q2_by_repeat")
    if by:
        five = _row_at(d["held_out"]["acids"], "five",
                       d["read_rho"])["invariant"]["q2_by_repeat"]
        lines += ["", f"The first repeat uses the folds of the tables above.  On those folds "
                      f"the atom-centred counts reach q2 {by[0]:+.3f}; over the {len(by)} "
                      f"repeats they lie between {min(by):+.3f} and {max(by):+.3f}.  The "
                      f"invariant kernel of the five fields at the width of "
                      f"{d['read_width']:.2f} A reaches {five[0]:+.3f} on the first repeat "
                      f"and lies between {min(five):+.3f} and {max(five):+.3f}.  The margin "
                      f"of the five fields over the counts in the tables above is read on "
                      f"that one set of folds.  The column *against the atom-centred "
                      f"counts* below gives it over all repeats."]
    for unit, title in (("acids", "whole acids held out"),
                        ("substituents", "whole substituents held out (both acids of a "
                                         "substituent together)")):
        h = d["held_out"][unit]
        lines += ["", f"### Indicator fields, {title}", "",
                  "| reference | q2 (sd over the repeats) | RMSE, cm^-1 |", "|---|---:|---:|"]
        for k in ("counts", "one_line", "two_lines"):
            r = h["references"][k]
            lines.append(f"| {r['name']} | {_q(r)} | {r['error']:.2f} |")
        lines += [""] + _indicator_rows(h["rows"])
    lines += ["", "### How alike the kernel of the indicator fields and the kernel of the five "
                  "fields are", "",
              "The correlation between the off-diagonal elements of the two kernels, each "
              "centred first.  Near 1, the indicator fields would order the pairs of acids as "
              "the five fields do and could add little.", "",
              "| width, A | invariant local kernel | posed overlap kernel |", "|---:|---:|---:|"]
    for a in d["kernel_agreement"]:
        lines.append(f"| {a['width']:.2f} | {a['invariant']:+.3f} | {a['aligned']:+.3f} |")
    e = d.get("effects")
    if e:
        places = list(e["places"])
        where = "; ".join(f"{p}, {v['distance']:.2f} A from the carbonyl carbon, {v['n']} "
                          f"acids, envelope {v['envelope']:.2f}" for p, v in e["places"].items())
        lines += ["", f"### What one atom of a type adds to the band at the place of the first "
                      f"atom of the substituent, width {e['width']:.2f} A", "",
                  f"The posed overlap model with the five fields and the {n15} indicator fields "
                  f"(q2 {e['q2']:+.3f}, ridge {e['lam']:g}).  The places are the mean positions "
                  f"of the first atom of the substituent in the frame of the carbonyl carbon: "
                  f"{where}.  *Effect* is what one atom of the type at that place adds to the "
                  f"band, in cm^-1, with the standard deviation that the noise in the bands "
                  f"allows.  *Acids* is the number of acids whose substituent starts with an "
                  f"atom of that type at that place, and *mean shift* is their mean band minus "
                  f"the band of the unsubstituted acid ({e['unsubstituted_band']:.1f} cm^-1).  "
                  f"The model was fitted on these bands, so the comparison is a check of what "
                  f"the model has stored and not a prediction.", "",
                  "| atom type | " + " | ".join(f"effect, {p} | acids | mean shift, cm^-1"
                                                for p in places) + " |",
                  "|---|" + "---:|---:|---:|" * len(places)]
        for t, row in e["types"].items():
            cells = []
            for p in places:
                c = row[p]
                cells.append(f"{c['effect']:+.2f} ({c['sd']:.2f}) | {c['n']} | "
                             + (f"{c['shift']:+.1f}" if c["shift"] is not None else ""))
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
        lines += ["", f"Over the {e['pairs']} pairs of a type and a place with at least "
                      f"{e['min_first']} acids, the effects correlate with the mean shifts "
                      f"with r {e['r_shift']:+.2f} and with the mean Hammett constants of those "
                      f"acids with r {e['r_sigma']:+.2f}."]
    lines += ["", _indicator_verdict(d)]
    return lines


def _against(iv):
    """How an interval of a difference in q2 reads: above, below or not separated."""
    if iv[1] > 0:
        return "above"
    if iv[2] < 0:
        return "below"
    return "not separated from"


def _indicator_verdict(d):
    rho = d["read_rho"]
    a, s = d["held_out"]["acids"], d["held_out"]["substituents"]
    parts = [f"**Reading of the indicator fields.** The statements are made at the width of "
             f"{d['read_width']:.2f} A, fixed before the measurement as the width where the "
             f"five fields were best, and for the invariant local kernel."]
    for h, what in ((a, "whole acids"), (s, "whole substituents")):
        five, ind, both = (_row_at(h, k, rho)["invariant"] for k in ("five", "indicator", "both"))
        two, cnt = h["references"]["two_lines"], h["references"]["counts"]
        parts.append(f"With {what} held out the five fields reach q2 {five['q2']:+.3f}, the "
                     f"indicator fields alone {ind['q2']:+.3f} and both together "
                     f"{both['q2']:+.3f}. Both together are {_iv(both['against_five'])} against "
                     f"the five fields ({_against(both['against_five'])} them) and "
                     f"{_iv(both['against_two_lines'])} against the Hammett regressions for "
                     f"meta and para separately, which reach {two['q2']:+.3f} "
                     f"({_against(both['against_two_lines'])} them). Against the atom-centred "
                     f"counts ({cnt['q2']:+.3f}) the five fields are "
                     f"{_iv(five['against_counts'])} ({_against(five['against_counts'])} "
                     f"them) and both together {_iv(both['against_counts'])} "
                     f"({_against(both['against_counts'])} them).")
    pa, p5 = _row_at(a, "both", rho)["aligned"], _row_at(a, "five", rho)["aligned"]
    parts.append(f"The posed overlap with both field sets reaches q2 {pa['q2']:+.3f} with whole "
                 f"acids held out, against {p5['q2']:+.3f} with the five fields "
                 f"({_iv(pa['against_five'])}, {_against(pa['against_five'])} them).")
    ag = next(x for x in d["kernel_agreement"] if x["rho"] == rho)
    parts.append(f"The kernel of the indicator fields correlates with the kernel of the five "
                 f"fields with r {ag['invariant']:+.2f} (invariant) and {ag['aligned']:+.2f} "
                 f"(posed), so the indicator fields are a different description and not a copy "
                 f"of the five fields.")
    drop5 = _row_at(a, "five", rho)["invariant"]["q2"] - _row_at(s, "five", rho)["invariant"]["q2"]
    dropb = _row_at(a, "both", rho)["invariant"]["q2"] - _row_at(s, "both", rho)["invariant"]["q2"]
    parts.append(f"Holding out whole substituents lowers the q2 of the five fields by "
                 f"{drop5:.3f} and of both field sets by {dropb:.3f}. It does not change the "
                 f"Hammett regressions, because a Hammett constant is a measured value of the "
                 f"substituent, and a field model has to learn a substituent from the other "
                 f"substituents.")
    e = d.get("effects")
    if e:
        parts.append(f"The effects of single atoms follow the mean observed shifts by the type "
                     f"and the place of the first atom (r {e['r_shift']:+.2f} over "
                     f"{e['pairs']} pairs), so the model has stored the substituent effects by "
                     f"atom type and place.")
    ba, bs = (_row_at(h, "both", rho)["invariant"]["against_two_lines"] for h in (a, s))
    parts.append(f"**Answer to the question of plan item SP2.** The five fields with the "
                 f"indicator fields {'pass' if ba[1] > 0 and bs[1] > 0 else 'do not pass'} "
                 f"the Hammett regressions for meta and para separately: the difference in "
                 f"q2 is {_iv(ba)} with whole acids held out and {_iv(bs)} with whole "
                 f"substituents held out.")
    return " ".join(parts)


def _opt(cell, key):
    return _iv(cell[key]) if key in cell else ""


def _picked(cell, widths):
    return " / ".join(str(cell["widths"].get(f"{w:.2f}", 0)) for w in widths)


def _largest(cell):
    top = cell["largest"][0]
    return f"{top['name']} ({top['error']:.1f}; {100 * top['share']:.0f} %)"


def _charge_references(refs, worst):
    lines = [f"| reference | q2 (sd over the repeats) | RMSE, cm^-1 | RMSE of the meta acids, "
             f"cm^-1 | RMSE of the para acids, cm^-1 | acid with the largest error (its RMSE, "
             f"cm^-1; its part of the squared error) | q2 scored without the acids of {worst} "
             f"(sd over the repeats) |", "|---|---:|---:|---:|---:|---|---:|"]
    for k in CHARGE_REFERENCES:
        r, p = refs[k], refs[k]["by_position"]
        lines.append(f"| {r['name']} | {_q(r)} | {r['error']:.2f} | {p['meta']:.2f} | "
                     f"{p['para']:.2f} | {_largest(r)} | {_q(r['without'])} |")
    return lines


def _charge_chosen(chosen, widths, kind):
    w = " / ".join(f"{x:.2f}" for x in widths)
    lines = [f"| fields | number of fields | q2 (sd over the repeats) | number of folds that "
             f"chose the width {w} A | RMSE of the meta acids, cm^-1 | RMSE of the para acids, "
             f"cm^-1 | against the five fields with Gasteiger charges | against the five "
             f"fields with quantum charges | against the Hammett regressions, meta and para "
             f"separately | against the regression on the four charges of the carboxyl group |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in chosen:
        c = row[kind]
        p = c["by_position"]
        lines.append(f"| {row['label']} | {row['n_fields']} | {_q(c)} | {_picked(c, widths)} | "
                     f"{p['meta']:.2f} | {p['para']:.2f} | {_opt(c, 'against_five')} | "
                     f"{_opt(c, 'against_five_q')} | {_iv(c['against_two_lines'])} | "
                     f"{_iv(c['against_carboxyl_charges'])} |")
    return lines


def _charge_without(chosen, worst, kind):
    lines = [f"| fields | acid with the largest error (its RMSE, cm^-1; its part of the "
             f"squared error) | q2 scored without the acids of {worst} (sd over the repeats) | "
             f"without them, against the Hammett regressions, meta and para separately | "
             f"without them, against the regression on the four charges of the carboxyl "
             f"group |", "|---|---|---:|---:|---:|"]
    for row in chosen:
        c = row[kind]
        w = c["without"]
        lines.append(f"| {row['label']} | {_largest(c)} | {_q(w)} | "
                     f"{_iv(w['against_two_lines'])} | {_iv(w['against_carboxyl_charges'])} |")
    return lines


def _charge_scan(rows):
    lines = ["| fields | width, A | invariant local q2 (sd over the repeats) | against the "
             "five fields with Gasteiger charges | against the Hammett regressions, meta and "
             "para separately | against the regression on the four charges of the carboxyl "
             "group | posed overlap q2 (sd over the repeats) | posed overlap against the "
             "Hammett regressions, meta and para separately |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        i, a = row["invariant"], row["aligned"]
        lines.append(f"| {row['label']} | {row['width']:.2f} | {_q(i)} | "
                     f"{_opt(i, 'against_five')} | {_iv(i['against_two_lines'])} | "
                     f"{_iv(i['against_carboxyl_charges'])} | {_q(a)} | "
                     f"{_iv(a['against_two_lines'])} |")
    return lines


def _chosen_row(held, fields):
    return next(r for r in held["chosen"] if r["fields"] == fields)


def _charge_section(d):
    """The part of the table that `charges.json` gives."""
    widths = sorted(d["widths"], reverse=True)
    cq, cg = d["carboxyl"]["quantum"], d["carboxyl"]["gasteiger"]
    tw, worst = d["twins"], d["outlier"]["substituent"]
    lines = ["## Quantum charges: the electrostatic field from GFN2-xTB charges, on the start "
             "geometries", "",
             f"The electrostatic field of the five fields is built from Gasteiger charges.  "
             f"That scheme computes the charges from the bonds alone and moves charge along "
             f"the sigma bonds only.  It does not hold the effect of a substituent through "
             f"the pi electrons of the ring (resonance).  In this part the charge of every "
             f"atom is taken from one calculation with the semi-empirical quantum method "
             f"{d['quantum_level'] or 'GFN2-xTB'} (program xtb) on the start geometry.  The "
             f"geometry is not optimised, so the description holds nothing about the geometry "
             f"of the ORCA run that gives the band.  The charges are in `{d['quantum']}`.",
             "",
             f"The same calculation gives the two Fukui indices of every atom: the change of "
             f"the charge of the atom when one electron is added to the molecule, and when "
             f"one electron is removed.  One field set adds a field for each of the two "
             f"indices ({', '.join(d['fukui'])}).  Another adds the {len(d['types'])} "
             f"indicator fields of the part above.",
             "",
             f"The cross-validation is repeated {d['repeats']} times with other folds, with "
             f"whole acids held out and with whole substituents held out, as in the part "
             f"above.  A difference in q2 carries its 95 % bootstrap interval over the units "
             f"that the folds hold out.",
             "",
             "### The charges of the four atoms of the carboxyl group", "",
             "| atom | quantum charge, mean (sd over the acids), e | quantum charge: r with the "
             "band | quantum charge: r with the Hammett constant | Gasteiger charge, mean (sd "
             "over the acids), e | Gasteiger charge: r with the band |",
             "|---|---:|---:|---:|---:|---:|"]
    for atom in CARBOXYL_ATOMS:
        x, g = cq[atom], cg[atom]
        lines.append(f"| {atom} | {x['mean']:+.3f} ({x['sd']:.4f}) | {_r(x['r_band'])} | "
                     f"{_r(x['r_sigma'])} | {g['mean']:+.3f} ({g['sd']:.4f}) | "
                     f"{_r(g['r_band'])} |")
    lines += ["", f"A multiple linear regression of the band on the four quantum charges, "
                  f"fitted on all acids, has r2 {d['carboxyl']['fit_r2_four_quantum']:.3f}.  "
                  f"The charges of the carboxyl group are computed numbers that could hold the "
                  f"whole answer, as the length of the C=O bond did on the optimised "
                  f"geometries.  The references below therefore include the linear regression "
                  f"on the charge of the carbonyl oxygen and the multiple linear regression on "
                  f"the four charges, both fitted inside the folds.  A field with quantum "
                  f"charges that is no better than these regressions has used the charges of "
                  f"the carboxyl group and not the field of the acid."]
    for unit, title in (("acids", "whole acids held out"),
                        ("substituents", "whole substituents held out (both acids of a "
                                         "substituent together)")):
        h = d["held_out"][unit]
        lines += ["", f"### Quantum charges, {title}: the references", ""]
        lines += _charge_references(h["references"], worst)
        lines += ["", f"### Quantum charges, {title}: the width and the ridge chosen inside the "
                      f"folds, invariant local kernel", "",
                  f"Every training part is divided again into {d['inner_folds']} folds over "
                  f"the same unit.  The width of the envelope (one of "
                  f"{', '.join(f'{w:.2f}' for w in widths)} A) and the ridge are the pair with "
                  f"the best q2 on these inner folds, so the held-out acids take no part in "
                  f"the choice.  These are the rows that the reading below uses.", ""]
        lines += _charge_chosen(h["chosen"], widths, "invariant")
        lines += ["", f"Where the error of these rows is.  The last three columns score the "
                      f"same predictions without the acids of the substituent {worst}; nothing "
                      f"is fitted again.", ""]
        lines += _charge_without(h["chosen"], worst, "invariant")
        lines += ["", f"### Quantum charges, {title}: the width and the ridge chosen inside the "
                      f"folds, posed overlap kernel", ""]
        lines += _charge_chosen(h["chosen"], widths, "aligned")
        lines += [""]
        lines += _charge_without(h["chosen"], worst, "aligned")
        lines += ["", f"### Quantum charges, {title}: every width", "",
                  "Here the width is fixed and the ridge is the one with the best q2 on the "
                  "folds of each repeat, as in the tables of the other parts.  A row of this "
                  "table shows what a width holds.  The best row of a field set is not a "
                  "result, because it was chosen by its q2.", ""]
        lines += _charge_scan(h["rows"])
    a, s = d["held_out"]["acids"], d["held_out"]["substituents"]
    lines += ["", "### The difference between the para acid and the meta acid of one "
                  "substituent", "",
              f"Resonance is the part of a substituent effect that differs between the para "
              f"and the meta position.  Over the {tw['n']} substituents that occur twice, the "
              f"band of the para acid minus the band of the meta acid has a mean of "
              f"{tw['mean']:+.1f} cm^-1 and a standard deviation of {tw['spread']:.1f} cm^-1, "
              f"and it correlates with the difference of the two Hammett constants with r "
              f"{_r(tw['r_sigma_band'])}.  The table gives how well each description predicts "
              f"this difference from its out-of-fold predictions of the two acids (invariant "
              f"local kernel, width and ridge chosen inside the folds).  A q2 below zero "
              f"means that the description predicts the difference worse than its mean does.",
              "",
              "| description | whole acids held out: q2 of the predicted difference | whole "
              "acids held out: RMSE of the difference, cm^-1 | whole substituents held out: "
              "q2 of the predicted difference | whole substituents held out: RMSE of the "
              "difference, cm^-1 |",
              "|---|---:|---:|---:|---:|"]
    for k in CHARGE_REFERENCES:
        ta, ts = a["references"][k]["twins"], s["references"][k]["twins"]
        lines.append(f"| {a['references'][k]['name']} | {ta['q2']:+.2f} | {ta['error']:.2f} | "
                     f"{ts['q2']:+.2f} | {ts['error']:.2f} |")
    for k in CHARGE_SETS:
        ra, rs = _chosen_row(a, k), _chosen_row(s, k)
        ta, ts = ra["invariant"]["twins"], rs["invariant"]["twins"]
        lines.append(f"| {ra['label']} | {ta['q2']:+.2f} | {ta['error']:.2f} | "
                     f"{ts['q2']:+.2f} | {ts['error']:.2f} |")
    lines += ["", "### How alike the kernel with quantum charges and the kernel with Gasteiger "
                  "charges are", "",
              "The correlation between the off-diagonal elements of the two kernels, each "
              "centred first.  Near 1, the change of the charges does not change which acids "
              "the description finds alike.", "",
              "| width, A | electrostatic field alone, invariant local kernel | electrostatic "
              "field alone, posed overlap kernel | five fields, invariant local kernel | five "
              "fields, posed overlap kernel |", "|---:|---:|---:|---:|---:|"]
    for g in d["kernel_agreement"]:
        lines.append(f"| {g['width']:.2f} | {g['electrostatic_invariant']:+.3f} | "
                     f"{g['electrostatic_aligned']:+.3f} | {g['five_invariant']:+.3f} | "
                     f"{g['five_aligned']:+.3f} |")
    lines += ["", _charge_verdict(d)]
    return lines


def _passes(*intervals):
    """True when every difference has its whole 95 % interval above zero."""
    return all(iv[1] > 0 for iv in intervals)


def _charge_verdict(d):
    a, s = d["held_out"]["acids"], d["held_out"]["substituents"]
    cq, cg = d["carboxyl"]["quantum"], d["carboxyl"]["gasteiger"]
    out = d["outlier"]
    ox = "carbonyl oxygen"
    sd_g = max(cg[k]["sd"] for k in CARBOXYL_ATOMS)
    parts = [f"**Reading of the quantum charges.** The Gasteiger charges of the four atoms of "
             f"the carboxyl group are the same in every acid (the largest standard deviation "
             f"is {sd_g:.4f} e), so the default electrostatic field holds a substituent only "
             f"where the substituent is. The quantum charge of the {ox} varies (standard "
             f"deviation {cq[ox]['sd']:.4f} e) and correlates with the band with r "
             f"{_r(cq[ox]['r_band'])} and with the Hammett constant with r "
             f"{_r(cq[ox]['r_sigma'])}."]
    wide = max(d["widths"])

    def at(held, key, width):
        return next(r for r in held["rows"]
                    if r["fields"] == key and abs(r["width"] - width) < 1e-9)["invariant"]

    def best(held, key):
        return max((r for r in held["rows"] if r["fields"] == key),
                   key=lambda r: r["invariant"]["q2"])

    bq, bg = best(a, "es_q"), best(a, "es")
    narrow = bq["width"]
    e1 = bq["invariant"]
    parts.append(f"This changes what the width of the envelope does. With Gasteiger charges "
                 f"the electrostatic field alone reaches q2 {at(a, 'es', narrow)['q2']:+.3f} "
                 f"at {narrow:.2f} A and {at(a, 'es', wide)['q2']:+.3f} at {wide:.2f} A (whole "
                 f"acids held out, ridge chosen by the q2), and its best width is "
                 f"{bg['width']:.2f} A: the envelope has to reach the substituent. With "
                 f"quantum charges it reaches {e1['q2']:+.3f} at {narrow:.2f} A and "
                 f"{at(a, 'es_q', wide)['q2']:+.3f} at {wide:.2f} A. The regression on the "
                 f"four charges of the carboxyl group reaches "
                 f"{a['references']['carboxyl_charges']['q2']:+.3f}, and the difference of "
                 f"the field at {narrow:.2f} A from it is "
                 f"{_iv(e1['against_carboxyl_charges'])}. The quantum calculation has "
                 f"carried the effect of the substituent to the atoms of the carboxyl group, "
                 f"and a narrow field reads it there. The width therefore decides the "
                 f"result. It is chosen inside the folds, and the statements below are made "
                 f"for those rows and for the invariant local kernel.")
    for h, what in ((a, "whole acids"), (s, "whole substituents")):
        five, eg, eq, fq, ff, fi = (
            _chosen_row(h, k)["invariant"]
            for k in ("five", "es", "es_q", "five_q", "five_q_fukui", "five_q_indicator"))
        two, four = h["references"]["two_lines"], h["references"]["carboxyl_charges"]
        parts.append(f"With {what} held out, the Hammett regressions for meta and para "
                     f"separately reach q2 {two['q2']:+.3f} and the regression on the four "
                     f"charges of the carboxyl group reaches {four['q2']:+.3f}. The "
                     f"electrostatic field alone reaches {eg['q2']:+.3f} with Gasteiger "
                     f"charges and {eq['q2']:+.3f} with quantum charges (difference "
                     f"{_iv(eq['against_es'])}). With quantum charges it is "
                     f"{_iv(eq['against_two_lines'])} against the separate Hammett "
                     f"regressions ({_against(eq['against_two_lines'])} them) and "
                     f"{_iv(eq['against_carboxyl_charges'])} against the regression on the "
                     f"four charges ({_against(eq['against_carboxyl_charges'])} it). The five "
                     f"fields reach {five['q2']:+.3f} with Gasteiger charges and "
                     f"{fq['q2']:+.3f} with quantum charges, with a standard deviation over "
                     f"the repeats of {fq['sd']:.3f} (see the paragraph on one substituent "
                     f"below). With the two Fukui fields added the q2 is {ff['q2']:+.3f}: "
                     f"{_iv(ff['against_five'])} against the five fields with Gasteiger "
                     f"charges ({_against(ff['against_five'])} them), "
                     f"{_iv(ff['against_two_lines'])} against the separate Hammett regressions "
                     f"({_against(ff['against_two_lines'])} them) and "
                     f"{_iv(ff['against_carboxyl_charges'])} against the regression on the "
                     f"four charges ({_against(ff['against_carboxyl_charges'])} it). With the "
                     f"indicator fields added the q2 is {fi['q2']:+.3f} "
                     f"({_iv(fi['against_five_q'])} against the five fields with quantum "
                     f"charges, {_against(fi['against_five_q'])} them).")
    qa, qs = (_chosen_row(h, "five_q")["invariant"] for h in (a, s))
    top = qs["largest"][0]
    wa, ws = (_chosen_row(h, "five_q_fukui")["invariant"]["without"] for h in (a, s))
    posed = [row["aligned"]["largest"][0]["name"].endswith("-" + out["substituent"])
             for h in (a, s) for row in h["chosen"]]
    parts.append(f"**One substituent decides the result of the five fields with quantum "
                 f"charges.** The acids of the substituent {out['substituent']} "
                 f"({', '.join(out['acids'])}) hold {100 * out['share']:.0f} % of the squared "
                 f"error of that row with whole substituents held out. The acid with the "
                 f"largest error is {top['name']}, with a root mean square error of "
                 f"{top['error']:.1f} cm^-1 over the repeats. The largest quantum charge of an "
                 f"atom in these acids is {out['largest_charge']:+.2f} e. The largest in all "
                 f"other acids is {out['largest_charge_of_the_others']:+.2f} e "
                 f"({out['of_the_acid']}). When these acids are held out together, the "
                 f"training part holds no acid like them. The inner folds then choose a "
                 f"narrow width with a small ridge, and that model extrapolates far. Scored "
                 f"without these acids, the same predictions give q2 "
                 f"{qa['without']['q2']:+.3f} with whole acids held out and "
                 f"{qs['without']['q2']:+.3f} with whole substituents held out "
                 f"({_iv(qa['without']['against_two_lines'])} and "
                 f"{_iv(qs['without']['against_two_lines'])} against the separate Hammett "
                 f"regressions scored in the same way). The set with the two Fukui fields, "
                 f"scored in the same way, gives {_iv(wa['against_two_lines'])} and "
                 f"{_iv(ws['against_two_lines'])}. With the posed overlap kernel, an acid of "
                 f"this substituent has the largest error in {sum(posed)} of the "
                 f"{len(posed)} rows of the two tables. The substituent was found by its "
                 f"error, after the result was known, so the scores without it describe the "
                 f"failure and are not results. The tables with a fixed width do not show "
                 f"this failure, because there the ridge is chosen by the q2 of the folds "
                 f"that hold these acids out. A field from quantum charges therefore needs a "
                 f"check that the charges of a new molecule are inside the range of the "
                 f"training set. A tabulated Hammett constant needs no such check.")
    ta, ts = (_chosen_row(h, "five")["invariant"]["twins"] for h in (a, s))
    ea, es_ = (_chosen_row(h, "es_q")["invariant"]["twins"] for h in (a, s))
    fa_, fs_ = (_chosen_row(h, "five_q_fukui")["invariant"]["twins"] for h in (a, s))
    ha, hs = (h["references"]["two_lines"]["twins"] for h in (a, s))
    ca, cs = (h["references"]["carboxyl_charges"]["twins"] for h in (a, s))
    parts.append(f"**Resonance.** The difference between the para acid and the meta acid of "
                 f"one substituent is the part of the band that resonance decides. The five "
                 f"fields with Gasteiger charges predict it with q2 {ta['q2']:+.2f} (whole "
                 f"acids held out) and {ts['q2']:+.2f} (whole substituents held out). The "
                 f"electrostatic field alone with quantum charges reaches {ea['q2']:+.2f} and "
                 f"{es_['q2']:+.2f}, and the five fields with quantum charges and the two "
                 f"Fukui fields reach {fa_['q2']:+.2f} and {fs_['q2']:+.2f}. The separate "
                 f"Hammett regressions reach {ha['q2']:+.2f} and {hs['q2']:+.2f}, and the "
                 f"regression on the four charges of the carboxyl group {ca['q2']:+.2f} and "
                 f"{cs['q2']:+.2f}.")
    names = (("es_q", "The electrostatic field alone with quantum charges"),
             ("five_q_fukui", "The five fields with quantum charges and the two Fukui fields"),
             ("five_q", "The five fields with quantum charges alone"))
    passed = [label for key, label in names
              if _passes(_chosen_row(a, key)["invariant"]["against_two_lines"],
                         _chosen_row(s, key)["invariant"]["against_two_lines"])]
    answer = "**Answer.** With the width and the ridge chosen inside the folds, "
    if passed:
        answer += (f"these field sets pass the Hammett regressions for meta and para "
                   f"separately on both kinds of folds: {'; '.join(passed).lower()}.")
    else:
        answer += ("no field set with quantum charges passes the Hammett regressions for meta "
                   "and para separately on both kinds of folds.")
    for key, label in names:
        x, z = (_chosen_row(h, key)["invariant"]["against_two_lines"] for h in (a, s))
        verb = "is" if key == "es_q" else "are"
        answer += (f" {label} {verb} {_iv(x)} against them with whole acids held out "
                   f"({_against(x)} them) and {_iv(z)} with whole substituents held out "
                   f"({_against(z)} them).")
    ba, bs = best(a, "five_q_fukui"), best(s, "five_q_fukui")
    four = [_chosen_row(h, key)["invariant"]["against_carboxyl_charges"]
            for h in (a, s) for key in ("es_q", "five_q_fukui")]
    answer += (f" The interval of the last set is wide because of the substituent "
               f"{out['substituent']}. In the table with a fixed width, the best width of "
               f"the set with the Fukui fields ({ba['width']:.2f} A with whole acids held "
               f"out, {bs['width']:.2f} A with whole substituents held out) gives "
               f"{_iv(ba['invariant']['against_two_lines'])} and "
               f"{_iv(bs['invariant']['against_two_lines'])} against the separate Hammett "
               f"regressions. The difference between these numbers and the numbers above is "
               f"what a choice of the width and the ridge by the result adds. Quantum "
               f"charges do what "
               f"Gasteiger charges cannot do on this series: they bring the fields to the "
               f"level of the separate Hammett regressions, and they predict the difference "
               f"between the para and the meta acid.")
    if any(_passes(iv) for iv in four):
        answer += (" At least one of the two best field sets is above the regression on the "
                   "four charges of the carboxyl group on one kind of folds (see the tables).")
    else:
        answer += (" They do not bring the q2 of the fields above the regression on the "
                   "four charges of the carboxyl group: every interval of that difference "
                   "holds zero. So most of the gain is in the charges of the carboxyl group "
                   "themselves. On the difference between the para and the meta acid the two "
                   "best field sets reach a higher q2 than this regression (see the "
                   "paragraph on resonance); no interval was computed for that difference.")
    answer += (" The quantum charges come from a quantum calculation, as the band does. This "
               "part therefore measures how much of a costly quantum result a quantum "
               "calculation of a few seconds per acid already holds. It does not measure "
               "what the structure alone predicts.")
    parts.append(answer)
    return "\n\n".join(parts)


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
    ap.add_argument("--only", choices=("all", "band", "indicator", "charges"), default="all",
                    help="what to run: parts 1 to 6 (band), the indicator fields of part 7 "
                         "(indicator), the quantum charges of part 8 (charges), or "
                         "everything (all)")
    ap.add_argument("--quantum", default=QUANTUM,
                    help="the SD file with the start geometries and the quantum charge of "
                         "every atom, for part 8")
    ap.add_argument("--repeats", type=int, default=REPEATS,
                    help="how many times the cross-validation of the indicator part is "
                         "repeated with other folds")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    if a.report is None:
        structures = None if str(a.structures).lower() == "none" else a.structures
        if a.only in ("all", "band"):
            run(rhos=a.rho, cofield_rhos=a.cofield_rho, folds=a.folds, seed=a.seed,
                scrambles=a.scrambles, results=a.results, data=a.data,
                structures=structures)
        if a.only == "indicator" or (a.only == "all" and structures):
            run_indicator(folds=a.folds, seed=a.seed, repeats=a.repeats, results=a.results,
                          data=a.data, structures=structures)
        if a.only == "charges" or (a.only == "all" and structures
                                   and os.path.exists(a.quantum)):
            run_charges(folds=a.folds, seed=a.seed, repeats=a.repeats, results=a.results,
                        data=a.data, structures=structures, quantum=a.quantum)
    if a.report in (None, "-"):
        path = os.path.join(a.results, "table.md")
    else:
        path = a.report
    text = report(a.results, path)
    if a.report == "-":
        print(text)
    else:
        print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
