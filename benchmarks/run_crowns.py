#!/usr/bin/env python
"""
A host series in the frame of its cavity: plan item SC5.

``examples/crowns/`` holds 79 crown ethers and cryptands with the stability constants (log K) of
their sodium and potassium complexes in methanol at 25 degrees C, entered by hand from the
tabulation of Izatt et al., Chem. Rev. 85 (1985) 271, and each host wrapped around each ion
(GFN1-xTB in implicit methanol, with every donor of the ring held at its contact distance from
the ion), in the frame of its cavity (``cmfqsar/cavity.py``): the origin at the centroid of the
donor atoms of the macrocycle, the z axis along the unique axis of those donors.
``examples/make_crown_examples.py`` describes the rules of the table and of the geometries, and
why a geometry is the host as far around the ion as it can go and not a minimum of the energy.

What is measured
----------------
**1. The data and the geometries** (``--stage data``).  How many donors each ion touches, how
far the donors are from their contact distance, the radius of the donor ring and how far the ion
is from the centroid of the donors.

**2. Where the probe says the ion sits** (``--stage probe``).  The deepest point of the ion probe
field of the host alone against the position xtb gave the ion.  The reference is the origin of
the frame, the centroid of the donors, which needs no field.

**3. The constants and their difference** (``--stage models``).  Nested cross-validation, the
ridge and every other setting chosen inside each fold, on three targets: log K of potassium
(hosts in the conformation of the potassium complex), log K of sodium (sodium complex) and the
selectivity log K(K+) - log K(Na+) (both conformations).  Two fold designs: five random folds,
repeated ten times, and one family of hosts held out at a time (nine families), which asks for a
ring size or a ring type the model has not seen.  The descriptions, on the same folds:

    ring counts            the ring size, the donors of the ring by element, the fused aromatic
                           rings, the donors on side arms, and whether the host is bicyclic:
                           numbers read from the formula
    complex shape          read from the complex: how many ring donors the ion touches, how
                           far they are from the contact distance, the radius of the donor
                           ring, and the distance of the ion from the donor centroid
    ring counts + complex shape
    fingerprint            ECFP4 counts, Tanimoto kernel
    electrostatic field    the overlap kernel of one field, in the cavity frame
    ion probe field        the potassium or sodium probe field, in the cavity frame
    five fields            electrostatic, steric, hydrophobic, acceptor and donor
    five fields + probe
    five fields, ion side  the same frame with z turned to the side of the ion
    five fields, no frame  the alignment-free kernel of the same fields
    five fields, amounts kept   the overlap kernel without the division by the size of each
                           field, so that a host with more donors has more acceptor field
    five fields + ring counts

The references get a Gaussian kernel whose width is chosen inside each fold, so that they have
the nonlinear fit the fields have.  The grids of the field width, of that kernel width and of the
ridge were widened twice after the first runs chose values at their ends; with the grids below
the width of the aligned fields is chosen inside the grid (2 to 8), and two choices still sit at
an end: the alignment-free kernel takes the widest field offered in every fold, and the ridge of
the selectivity models is the smallest offered in about half of the folds.

**4. The selectivity by three routes** (``--stage select``): a model of the difference, the
difference of two separate models, and the two-task model of plan item SC1, in which each ion is
an endpoint of one kernel and their correlation is chosen inside each fold.

**5. The co-field** (``--stage cofield``), which is the gate of the item: "the co-field of the K+
constant names the cavity size and the donor positions".  The five-field model is fitted on all
hosts and read in cylinder coordinates of the frame: the distance rho from the z axis, the
height z, the turn phi about the axis.  A co-field is the gain per unit of field at a point,
which is not a quantity one can compare with a structure, so the acceptor, hydrophobic and
electrostatic co-fields are read as **the effect of one atom**: the overlap of the co-field with
the Gaussian of one atom placed at the point (one acceptor atom, one hydrophobic atom, one
negative unit charge), in units of log K.  The steric field has a width per element and is
read as the co-field itself.  The readings:

* the radius, in the plane of the donors (|z| <= 0.5 A, averaged over phi), at which an atom
  gains most, and the radius at which the gain changes sign, both with an interval from
  bootstrap samples of the hosts.  They are compared with the radii of the donor rings;
* the point (rho, z) of the largest gain anywhere, compared with where the hosts have donors
  outside the plane (the bridgehead nitrogens of the cryptands);
* how the gain varies with phi, and how the donors of the hosts are spread over phi, which
  says whether positions around the ring can be named at all.

Two checks of whether a map is more than noise: two random halves of the hosts give two maps
that are compared point by point; and the same after the constants are given to the hosts at
random.  The size of a map does not separate the two cases, because a ridge fit gives shuffled
constants a map as well; the agreement of the halves does.

Usage
-----
    python benchmarks/run_crowns.py                 # every stage, then the table (3 minutes)
    python benchmarks/run_crowns.py --stage models
    python benchmarks/run_crowns.py --report
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

from rdkit import Chem, DataStructs, RDLogger                               # noqa: E402
from rdkit.Chem import AllChem                                              # noqa: E402
from rdkit.Geometry import Point3D                                          # noqa: E402

from cmfqsar import cavity as CV                                            # noqa: E402
from cmfqsar import multitask as MT                                         # noqa: E402
from cmfqsar.cofields import field_volume, make_grid, per_molecule_fields   # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields                # noqa: E402
from cmfqsar.model import CMFModel                                          # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "crowns")
DATA = os.path.join(ROOT, "examples", "crowns")

SEED = 0
OUTER, INNER, REPEATS = 5, 5, 10
LAMS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
ALPHAS = (0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
WIDTHS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)   # of a Gaussian kernel, in median distances
SHARES = (0.25, 0.5, 0.75)         # of the first kernel in a sum of two
RHOS = (0.0, 0.3, 0.5, 0.7, 0.9)   # correlation of the two endpoints in the two-task model
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
FIELD_LABELS = {"electrostatic": "electrostatic", "steric": "steric", "hydrophobic": "hydrophobic",
                "hba": "acceptor", "hbd": "donor"}
IONS = ("Na", "K")
PROBES = {"Na": "sodium", "K": "potassium"}
TARGETS = ("K", "Na", "selectivity")
TARGET_LABELS = {"K": "log K of K+", "Na": "log K of Na+",
                 "selectivity": "log K(K+) - log K(Na+)"}
DESIGNS = ("random", "family")
MODELS = ("ring counts", "complex shape", "ring counts + complex shape", "fingerprint",
          "electrostatic field", "ion probe field", "five fields", "five fields + probe",
          "five fields, ion side", "five fields, no frame", "five fields, amounts kept",
          "five fields + ring counts")
COUNT_NAMES = ("ring size", "ring donors", "ring nitrogens", "ring sulfurs", "fused aromatic rings",
               "side-arm donors", "bicyclic")
#: the cylinder grid the co-field is read on, angstrom and number of turns
RHO = np.arange(0.0, 7.01, 0.25)
ZED = np.arange(-4.0, 4.01, 0.25)
N_PHI = 24
PLANE = 0.5                        # |z| at most this is "the plane of the donors"
HALVES, PERMUTATIONS, BOOTSTRAPS = 20, 20, 200
GRID_MARGIN, GRID_STEP = 2.0, 0.4


# ------------------------------------------------------------------------------------ the data
def load():
    """The hosts both files hold, in the order of the table."""
    got = {}
    for ion in IONS:
        path = os.path.join(DATA, f"hosts_{ion}.sdf")
        got[ion] = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(path, removeHs=False)
                    if m is not None}
    names = [n for n in got["K"] if n in got["Na"]]
    d = {"names": names, "mols": {i: [got[i][n] for n in names] for i in IONS}}
    first = d["mols"]["K"]
    d["y"] = {"Na": np.array([float(m.GetProp("logK_Na")) for m in first]),
              "K": np.array([float(m.GetProp("logK_K")) for m in first])}
    d["y"]["selectivity"] = d["y"]["K"] - d["y"]["Na"]
    d["family"] = [m.GetProp("family") for m in first]
    d["smiles"] = [m.GetProp("smiles") for m in first]
    d["ion_xyz"] = {i: np.array([[float(v) for v in m.GetProp("ion_xyz").split()]
                                 for m in d["mols"][i]]) for i in IONS}
    return d


def ring_counts(smiles):
    """The numbers of COUNT_NAMES, read from the formula."""
    mol = Chem.MolFromSmiles(smiles)
    ri = mol.GetRingInfo()
    macro = [r for r in ri.AtomRings() if len(r) >= CV.MIN_MACROCYCLE]
    inside = set(a for r in macro for a in r)
    donors = CV.donor_atoms(mol)
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    benzo = sum(1 for r in ri.AtomRings() if len(r) == 6
                and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)
                and len(inside.intersection(r)) >= 2)
    arm = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in CV.DONOR_ELEMENTS
              and a.GetIdx() not in inside and not (a.GetSymbol() == "O" and a.GetDegree() == 1
                                                    and a.GetTotalNumHs() == 0))
    return [min(len(r) for r in macro), len(donors), sum(sym[a] == "N" for a in donors),
            sum(sym[a] == "S" for a in donors), benzo, arm, float(len(macro) > 1)]


SHAPE_NAMES = ("ring donors in contact", "misfit of the donor distances",
               "radius of the donor ring", "ion to donor centroid")


def shape_numbers(mol, ion_xyz, ion):
    """The numbers of SHAPE_NAMES, read from the complex."""
    xyz = mol.GetConformer().GetPositions()
    donors = CV.donor_atoms(mol)
    return [float(CV.ion_contacts(xyz, donors, ion_xyz, ion)),
            CV.donor_misfit(xyz, donors, ion_xyz, ion), donor_radius(mol),
            float(np.linalg.norm(ion_xyz - xyz[donors].mean(axis=0)))]


def donor_radius(mol):
    xyz = mol.GetConformer().GetPositions()
    p = xyz[CV.donor_atoms(mol)]
    return float(np.linalg.norm(p - p.mean(axis=0), axis=1).mean())


def turned_to_ion(mols, ion_xyz):
    """Copies with z on the side of the ion: a half turn about x where the ion is below."""
    out = []
    for m, p in zip(mols, ion_xyz):
        c = Chem.Mol(m)
        if p[2] < 0:
            conf = c.GetConformer()
            for i, q in enumerate(m.GetConformer().GetPositions()):
                conf.SetAtomPosition(i, Point3D(float(q[0]), float(-q[1]), float(-q[2])))
        out.append(c)
    return out


# --------------------------------------------------------------------------------- the kernels
def field_kernels(mols, names, fields=FIELDS, probes=(), alpha=1.0, normalize=True):
    """(fields, n, n) overlap kernels, cosine-normalised or not, with the model and the table."""
    fs = FieldSet.create(physchem=list(fields), probes=list(probes), alpha=float(alpha),
                         charge_model="gasteiger")
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fs, method="krr", normalize=bool(normalize))
    return np.asarray(model.prepare_training_kernels(table)), model, table


def amounts_kernel(mols, names, alpha):
    """The mean of the five overlap kernels with the amounts kept: each field's kernel is
    divided by its mean diagonal only, so a host with more of a field has a longer row."""
    Kf = field_kernels(mols, names, alpha=alpha, normalize=False)[0]
    scale = np.array([np.mean(np.diag(k)) for k in Kf])
    keep = scale > 0
    return np.tensordot(1.0 / scale[keep], Kf[keep], axes=(0, 0)) / len(FIELDS)


def free_kernel(mols, names, alpha):
    """The alignment-free kernel of the five fields, with a unit diagonal."""
    fs = FieldSet.create(physchem=list(FIELDS), alpha=float(alpha), charge_model="gasteiger")
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    K = np.asarray(CMFModel(fieldset=fs, method="krr",
                            kernel_source="autocorrelation").training_kernel(table))
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / np.outer(d, d)


def gaussian_kernels(X):
    """One Gaussian kernel per width of WIDTHS on the standardised columns of X."""
    X = np.asarray(X, float)
    sd = X.std(axis=0)
    Z = (X[:, sd > 0] - X[:, sd > 0].mean(axis=0)) / sd[sd > 0]
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
    med = float(np.median(d2[np.triu_indices(len(Z), 1)]))
    return {w: np.exp(-d2 / (2.0 * w * w * max(med, 1e-12))) for w in WIDTHS}


def tanimoto_kernel(smiles):
    fps = [AllChem.GetHashedMorganFingerprint(Chem.MolFromSmiles(s), 2, nBits=2048)
           for s in smiles]
    return np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])


def build_kernels(d, verbose=True):
    """Every candidate kernel of every description, per geometry ("Na", "K" or "both")."""
    t0 = time.time()
    names = d["names"]
    counts = np.array([ring_counts(s) for s in d["smiles"]])
    contacts = {i: np.array([shape_numbers(m, p, i) for m, p in
                             zip(d["mols"][i], d["ion_xyz"][i])]) for i in IONS}
    contacts["both"] = np.hstack([contacts["Na"], contacts["K"]])
    fp = tanimoto_kernel(d["smiles"])
    base = {}
    for ion in IONS:
        mols = d["mols"][ion]
        turned = turned_to_ion(mols, d["ion_xyz"][ion])
        b = {"five": {}, "electrostatic": {}, "side": {}, "free": {}, "raw": {}}
        for a in ALPHAS:
            Kf = field_kernels(mols, names, alpha=a)[0]
            b["five"][a] = Kf.mean(axis=0)
            b["electrostatic"][a] = Kf[0]
            b["side"][a] = field_kernels(turned, names, alpha=a)[0].mean(axis=0)
            b["free"][a] = free_kernel(mols, names, a)
            b["raw"][a] = amounts_kernel(mols, names, a)
        b["probe"] = field_kernels(mols, names, (), (PROBES[ion],))[0][0]
        base[ion] = b
        if verbose:
            print(f"  kernels of the {ion}+ geometries ({time.time() - t0:.0f} s)", flush=True)
    both = {k: {a: 0.5 * (base["Na"][k][a] + base["K"][k][a]) for a in ALPHAS}
            for k in ("five", "electrostatic", "side", "free", "raw")}
    both["probe"] = 0.5 * (base["Na"]["probe"] + base["K"]["probe"])
    base["both"] = both
    out = {}
    for geo in ("Na", "K", "both"):
        b = base[geo]
        gc = gaussian_kernels(counts)
        cand = {
            "ring counts": {("w", w): K for w, K in gc.items()},
            "complex shape": {("w", w): K for w, K in gaussian_kernels(contacts[geo]).items()},
            "ring counts + complex shape": {("w", w): K for w, K in gaussian_kernels(
                np.hstack([counts, contacts[geo]])).items()},
            "fingerprint": {("fp",): fp},
            "electrostatic field": {("a", a): b["electrostatic"][a] for a in ALPHAS},
            "ion probe field": {("probe",): b["probe"]},
            "five fields": {("a", a): b["five"][a] for a in ALPHAS},
            "five fields + probe": {("a", a, "s", s): s * b["five"][a] + (1 - s) * b["probe"]
                                    for a in ALPHAS for s in SHARES},
            "five fields, ion side": {("a", a): b["side"][a] for a in ALPHAS},
            "five fields, no frame": {("a", a): b["free"][a] for a in ALPHAS},
            "five fields, amounts kept": {("a", a): b["raw"][a] for a in ALPHAS},
            "five fields + ring counts": {
                ("a", a, "s", s, "w", w): s * b["five"][a] + (1 - s) * gc[w]
                for a in ALPHAS for s in SHARES for w in WIDTHS},
        }
        out[geo] = cand
    return out, {"counts": counts, "shape": contacts}


# ------------------------------------------------------------------------- the cross-validation
def ridge(K, y, lam):
    """Kernel ridge regression with a bias: the dual coefficients and the bias."""
    n = len(y)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = K + lam * np.eye(n)
    A[:n, n] = A[n, :n] = 1.0
    sol = np.linalg.solve(A, np.concatenate([y, [0.0]]))
    return sol[:n], float(sol[n])


def k_folds(n, k, seed):
    order = np.random.RandomState(seed).permutation(n)
    return [np.sort(g) for g in np.array_split(order, k)]


def design_folds(design, families, repeat=0):
    """The test sets of one pass over the hosts."""
    if design == "family":
        fam = np.asarray(families)
        return [np.where(fam == f)[0] for f in dict.fromkeys(families)]
    return k_folds(len(families), OUTER, SEED + repeat)


def choose(cands, y, rows, seed):
    """(key, lam) with the lowest inner cross-validation error on `rows`."""
    inner = k_folds(len(rows), INNER, seed)
    best = None
    for key, K in cands.items():
        Kr = K[np.ix_(rows, rows)]
        for lam in LAMS:
            p = np.zeros(len(rows))
            for te in inner:
                tr = np.setdiff1d(np.arange(len(rows)), te)
                a, b = ridge(Kr[np.ix_(tr, tr)], y[rows][tr], lam)
                p[te] = Kr[np.ix_(te, tr)] @ a + b
            e = float(np.mean((p - y[rows]) ** 2))
            if best is None or e < best[0] - 1e-12:
                best = (e, key, lam)
    return best[1], best[2]


def nested(cands, y, folds, seed):
    """Out-of-fold predictions, the training mean each host was scored against, the choices."""
    n = len(y)
    pred, ref, chosen = np.zeros(n), np.zeros(n), []
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        key, lam = choose(cands, y, tr, seed)
        K = cands[key]
        a, b = ridge(K[np.ix_(tr, tr)], y[tr], lam)
        pred[te] = K[np.ix_(te, tr)] @ a + b
        ref[te] = y[tr].mean()
        chosen.append([list(key), lam])
    return pred, ref, chosen


def q2_of(y, pred, ref):
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - ref) ** 2))


def paired_interval(a, b, seed=SEED, draws=2000):
    """The paired difference of two models' absolute errors, with a bootstrap interval."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(diff)), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


GEOMETRY = {"K": "K", "Na": "Na", "selectivity": "both"}


# ------------------------------------------------------------------------------ stage 1: data
def data_stage(d):
    out = {"hosts": len(d["names"]), "families": {}, "targets": {}, "geometry": {}}
    for f in d["family"]:
        out["families"][f] = out["families"].get(f, 0) + 1
    for t in TARGETS:
        y = d["y"][t]
        out["targets"][t] = {"min": float(y.min()), "max": float(y.max()),
                             "mean": float(y.mean()), "sd": float(y.std())}
    out["correlation_K_Na"] = float(np.corrcoef(d["y"]["K"], d["y"]["Na"])[0, 1])
    for ion in IONS:
        c = np.array([shape_numbers(m, p, ion) for m, p in zip(d["mols"][ion],
                                                                d["ion_xyz"][ion])])
        n_donors = np.array([len(CV.donor_atoms(m)) for m in d["mols"][ion]])
        fam = d["family"]
        out["geometry"][ion] = {
            "radius_sum": CV.ION_RADII[ion] + CV.DONOR_RADIUS,
            "contact_limit": CV.contact_distance(ion),
            "all_donors_in_contact": int(np.sum(c[:, 0] == n_donors)),
            "not_all_in_contact": [n for n, k, m in zip(d["names"], c[:, 0], n_donors) if k < m],
            "mean_misfit": float(c[:, 1].mean()), "max_misfit": float(c[:, 1].max()),
            "mean_ion_to_centroid": float(c[:, 3].mean()),
            "by_family": {f: {"donor_radius": float(np.mean(c[[g == f for g in fam], 2])),
                              "ion_to_centroid": float(np.mean(c[[g == f for g in fam], 3]))}
                          for f in dict.fromkeys(fam)},
            "correlations_with_log_K": {
                name: (float(np.corrcoef(c[:, j], d["y"][ion])[0, 1]) if c[:, j].std() > 0
                       else None) for j, name in enumerate(SHAPE_NAMES)}}
    counts = np.array([ring_counts(s) for s in d["smiles"]])
    out["count_correlations"] = {t: {n: (float(np.corrcoef(counts[:, j], d["y"][t])[0, 1])
                                         if counts[:, j].std() > 0 else None)
                                     for j, n in enumerate(COUNT_NAMES)} for t in TARGETS}
    return out


# ----------------------------------------------------------------------------- stage 2: probe
def probe_stage(d):
    """Does the probe field of the host put the ion where xtb put it?"""
    out = {}
    for ion in IONS:
        fs = FieldSet.create(physchem=[], probes=[PROBES[ion]], alpha=1.0,
                             charge_model="gasteiger")
        d_probe, d_centre = [], []
        for m, n, p in zip(d["mols"][ion], d["names"], d["ion_xyz"][ion]):
            mf = build_fields(m, fs, n)
            donors = m.GetConformer().GetPositions()[CV.donor_atoms(m)]
            grid = make_grid(donors, margin=GRID_MARGIN, spacing=GRID_STEP)
            vol = field_volume(mf, fs.specs[0], 0, grid).ravel()
            d_probe.append(float(np.linalg.norm(grid.points()[int(np.argmin(vol))] - p)))
            d_centre.append(float(np.linalg.norm(p)))
        dp, dc = np.array(d_probe), np.array(d_centre)
        out[ion] = {"probe": PROBES[ion], "grid_step": GRID_STEP,
                    "probe_minimum": {"mean": float(dp.mean()), "median": float(np.median(dp)),
                                      "within_1A": float(np.mean(dp <= 1.0))},
                    "donor_centroid": {"mean": float(dc.mean()), "median": float(np.median(dc)),
                                       "within_1A": float(np.mean(dc <= 1.0))},
                    "probe_closer": float(np.mean(dp < dc))}
    return out


# ---------------------------------------------------------------------------- stage 3: models
def models_stage(d, kernels):
    out = {}
    t0 = time.time()
    for target in TARGETS:
        y = d["y"][target]
        cands = kernels[GEOMETRY[target]]
        for design in DESIGNS:
            repeats = 1 if design == "family" else REPEATS
            got = {}
            for model in MODELS:
                q2, preds, errs, picks = [], [], [], []
                for r in range(repeats):
                    folds = design_folds(design, d["family"], r)
                    pred, ref, chosen = nested(cands[model], y, folds, SEED + r)
                    q2.append(q2_of(y, pred, ref))
                    preds.append(pred.tolist())
                    errs.append(np.abs(pred - y))
                    picks.extend(chosen)
                rmse = float(np.sqrt(np.mean([(np.array(p) - y) ** 2 for p in preds])))
                got[model] = {"q2": float(np.mean(q2)), "q2_sd": float(np.std(q2)),
                              "rmse": rmse, "abs_errors": np.mean(errs, axis=0).tolist(),
                              "predictions": preds, "chosen": picks}
                print(f"  {target:11s} {design:6s} {model:28s} q2 {got[model]['q2']:+.3f} "
                      f"rmse {rmse:.3f} ({time.time() - t0:.0f} s)", flush=True)
            pairs = {}
            for a, b in (("five fields", "ring counts"), ("five fields", "complex shape"),
                         ("five fields", "ring counts + complex shape"),
                         ("five fields", "fingerprint"),
                         ("five fields", "five fields, no frame"),
                         ("five fields", "five fields, ion side"),
                         ("five fields", "five fields, amounts kept"),
                         ("five fields, amounts kept", "ring counts"),
                         ("five fields", "electrostatic field"),
                         ("five fields", "ion probe field"),
                         ("ion probe field", "electrostatic field"),
                         ("five fields + probe", "five fields"),
                         ("five fields + ring counts", "ring counts"),
                         ("five fields + ring counts", "five fields")):
                pairs[f"{a} | {b}"] = paired_interval(got[a]["abs_errors"],
                                                      got[b]["abs_errors"])
            out[f"{target}|{design}"] = {"models": got, "pairs": pairs, "n": len(y),
                                         "repeats": repeats}
    return out


# --------------------------------------------------------------------------- stage 4: selectivity
def two_task(K_by_alpha, d, folds, seed):
    """The two-task model: alpha, the ridge and the correlation chosen on the training hosts."""
    n = len(d["names"])
    obs = MT.Observations.from_columns(np.column_stack([d["y"]["K"], d["y"]["Na"]]),
                                       names=("K", "Na"))
    pred = np.zeros(n)
    chosen = []
    for te in folds:
        held = set(int(i) for i in te)
        tr = np.array([p for p in range(len(obs)) if int(obs.molecule[p]) not in held])
        sub = obs.subset(tr)
        best = None
        for a, K in K_by_alpha.items():
            for rho in RHOS:
                for lam in LAMS:
                    p = MT.cv_predict(K, sub, rho=rho, lam=lam, folds=INNER, seed=seed)
                    e = float(np.nanmean((p - sub.y) ** 2))
                    if best is None or e < best[0] - 1e-12:
                        best = (e, a, rho, lam)
        _e, a, rho, lam = best
        f = MT.fit(K_by_alpha[a], sub, rho=rho, lam=lam)
        w = f.selectivity_coefficients(0, 1)
        for m in te:
            pred[m] = float(K_by_alpha[a][m] @ w + f.bias[0] - f.bias[1])
        chosen.append([a, rho, lam])
    return pred, chosen


def select_stage(d, kernels, models):
    y = d["y"]["selectivity"]
    out = {}
    t0 = time.time()
    for design in DESIGNS:
        repeats = 1 if design == "family" else REPEATS
        rows = {}
        for model in ("ring counts", "complex shape", "five fields"):
            direct = models[f"selectivity|{design}"]["models"][model]
            pk = np.array(models[f"K|{design}"]["models"][model]["predictions"])
            pn = np.array(models[f"Na|{design}"]["models"][model]["predictions"])
            q2s, errs = [], []
            for r in range(repeats):
                folds = design_folds(design, d["family"], r)
                ref = np.zeros(len(y))
                for te in folds:
                    ref[te] = y[np.setdiff1d(np.arange(len(y)), te)].mean()
                q2s.append(q2_of(y, pk[r] - pn[r], ref))
                errs.append(np.abs(pk[r] - pn[r] - y))
            rows[f"{model}, a model of the difference"] = {
                "q2": direct["q2"], "q2_sd": direct["q2_sd"], "rmse": direct["rmse"],
                "abs_errors": direct["abs_errors"]}
            rows[f"{model}, two separate models"] = {
                "q2": float(np.mean(q2s)), "q2_sd": float(np.std(q2s)),
                "rmse": float(np.sqrt(np.mean((pk - pn - y) ** 2))),
                "abs_errors": np.mean(errs, axis=0).tolist()}
        K_by_alpha = {a: kernels["both"]["five fields"][("a", a)] for a in ALPHAS}
        q2s, errs, preds, picks = [], [], [], []
        for r in range(repeats):
            folds = design_folds(design, d["family"], r)
            pred, chosen = two_task(K_by_alpha, d, folds, SEED + r)
            ref = np.zeros(len(y))
            for te in folds:
                ref[te] = y[np.setdiff1d(np.arange(len(y)), te)].mean()
            q2s.append(q2_of(y, pred, ref))
            errs.append(np.abs(pred - y))
            preds.append(pred)
            picks.extend(chosen)
            print(f"  two-task {design} repeat {r + 1}/{repeats}: q2 {q2s[-1]:+.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        rows["five fields, the two-task model"] = {
            "q2": float(np.mean(q2s)), "q2_sd": float(np.std(q2s)),
            "rmse": float(np.sqrt(np.mean((np.array(preds) - y) ** 2))),
            "abs_errors": np.mean(errs, axis=0).tolist(), "chosen": picks,
            "mean_rho": float(np.mean([c[1] for c in picks]))}
        pairs = {}
        a = "five fields, the two-task model"
        for b in ("five fields, a model of the difference", "five fields, two separate models",
                  "ring counts, a model of the difference",
                  "complex shape, a model of the difference"):
            pairs[f"{a} | {b}"] = paired_interval(rows[a]["abs_errors"], rows[b]["abs_errors"])
        out[design] = {"rows": rows, "pairs": pairs}
    return out


# ------------------------------------------------------------------------------ stage 5: co-field
def cylinder_points():
    phi = 2.0 * np.pi * np.arange(N_PHI) / N_PHI
    r, z, p = np.meshgrid(RHO, ZED, phi, indexing="ij")
    return np.stack([r * np.cos(p), r * np.sin(p), z], axis=-1).reshape(-1, 3)


def plane_profile(C):
    """A map averaged over the turn and over |z| <= PLANE, as a function of rho."""
    cyl = C.reshape(len(RHO), len(ZED), N_PHI)
    return cyl[:, np.abs(ZED) <= PLANE + 1e-9, :].mean(axis=(1, 2))


def peak(profile, sign=1.0):
    """(rho, value) of the largest value of sign * profile."""
    k = int(np.argmax(sign * profile))
    return float(RHO[k]), float(profile[k])


def sign_change(profile):
    """The radius at which a profile turns from negative to positive, going outwards from its
    lowest point: linear between the two grid radii.  NaN when it does not turn."""
    k = int(np.argmin(profile))
    for i in range(k, len(profile) - 1):
        if profile[i] < 0 <= profile[i + 1]:
            w = -profile[i] / (profile[i + 1] - profile[i])
            return float(RHO[i] + w * (RHO[i + 1] - RHO[i]))
    return float("nan")


#: what one atom is, per field read as the effect of an atom: (weight, what the table calls it)
ATOM = {"hba": (1.0, "one acceptor atom"), "hydrophobic": (1.0, "one hydrophobic atom"),
        "electrostatic": (-1.0, "one negative unit charge")}


def reading_basis(model, f, pts):
    """(hosts, points): what each host's field f contributes at each point, per unit of its
    dual coefficient.  For a field of ATOM it is the overlap with one atom placed at the
    point, for another field the field itself."""
    from cmfqsar.cofields import probe_basis
    name = FIELDS[f]
    if name not in ATOM:
        return per_molecule_fields(model, f, pts) / len(FIELDS)
    centres, coeffs, widths, index = model.cofield_basis(f)
    c, k, w = probe_basis(centres, coeffs, widths, float(model.fieldset.specs[f].alpha),
                          ATOM[name][0])
    return per_molecule_fields(model, f, pts, basis=(c, k, w, index)) / len(FIELDS)


def cofield_stage(d, verbose=True):
    out = {"grid": {"rho_step": float(RHO[1] - RHO[0]), "rho_max": float(RHO[-1]),
                    "z_max": float(ZED[-1]), "turns": N_PHI, "plane": PLANE},
           "fields": [FIELD_LABELS[f] for f in FIELDS],
           "read_as": {FIELD_LABELS[f]: (ATOM[f][1] if f in ATOM else "the co-field itself")
                       for f in FIELDS}}
    pts = cylinder_points()
    maps = {}
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    mono = np.array([g != "cryptand" for g in d["family"]])
    # every host, and the monocyclic hosts alone.  The second reading was added after the
    # first: with the cryptands in, the largest contrast of the set is between a ring and a
    # cage, and the map shows that contrast before it shows the size of a ring.
    for ion, (label, keep) in ((i, s) for i in IONS
                               for s in (("", np.ones(len(mono), bool)),
                                         (", monocyclic hosts", mono))):
        y = d["y"][ion][keep]
        n = len(y)
        mols = [m for m, k in zip(d["mols"][ion], keep) if k]
        names = [m for m, k in zip(d["names"], keep) if k]
        # the width and the ridge by five-fold cross-validation on these hosts
        per_alpha = {a: field_kernels(mols, names, alpha=a) for a in ALPHAS}
        a, lam = choose({("a", a): v[0].mean(axis=0) for a, v in per_alpha.items()}, y,
                        np.arange(n), SEED)
        alpha = a[1]
        Kf, model, _table = per_alpha[alpha]
        K = Kf.mean(axis=0)
        coef = ridge(K, y, lam)[0]
        radii = np.array([donor_radius(m) for m in mols])
        top = np.argsort(-y)[:10]
        fam = [g for g, k in zip(d["family"], keep) if k]
        res = {"alpha": alpha, "lam": lam, "hosts": n, "ion": ion,
               "radius_sum": CV.ION_RADII[ion] + CV.DONOR_RADIUS,
               "donor_radius": {"all": float(radii.mean()),
                                "ten strongest hosts": float(radii[top].mean()),
                                "by_family": {
                                    f: float(np.mean([r for r, g in zip(radii, fam) if g == f]))
                                    for f in dict.fromkeys(fam)}},
               "fields": {}}

        def fitted(rows, values):
            return ridge(K[np.ix_(rows, rows)], values, lam)[0]

        halves = [rng.permutation(n) for _ in range(HALVES)]
        perms = [rng.permutation(n) for _ in range(PERMUTATIONS)]
        boots = [rng.randint(0, n, n) for _ in range(BOOTSTRAPS)]
        for f, name in enumerate(FIELDS):
            Xf = reading_basis(model, f, pts)
            # a field that fewer than ten hosts carry has no map worth reading: the donor
            # field exists only on the hosts with a hydroxyl or an N-H group
            if int(np.sum(np.abs(Xf).sum(axis=1) > 0)) < 10:
                continue
            C = coef @ Xf
            prof = plane_profile(C)
            rho_star, value = peak(prof)
            cyl = C.reshape(len(RHO), len(ZED), N_PHI)
            mean_rz = cyl.mean(axis=2)
            # the largest gain anywhere in (rho, z), and how the gain varies with the turn at
            # the radius of the largest gain in the plane
            k = np.unravel_index(int(np.argmax(mean_rz)), mean_rz.shape)
            ring = cyl[int(np.argmax(prof)), np.abs(ZED) <= PLANE + 1e-9, :].mean(axis=0)
            used = np.abs(Xf).mean(axis=0) > 1e-3 * np.abs(Xf).mean(axis=0).max()
            # bootstrap samples of the hosts: an interval for the two radii
            b_rho, b_zero = [], []
            for ix in boots:
                u = np.unique(ix)
                w = np.bincount(ix, minlength=n)[u].astype(float)
                # a host drawn twice counts twice: its ridge is divided by its count
                A = np.zeros((len(u) + 1, len(u) + 1))
                A[:-1, :-1] = K[np.ix_(u, u)] + lam * np.diag(1.0 / w)
                A[:-1, -1] = A[-1, :-1] = 1.0
                ab = np.linalg.solve(A, np.concatenate([y[u], [0.0]]))[:-1]
                pb = plane_profile(ab @ Xf[u])
                b_rho.append(peak(pb)[0])
                b_zero.append(sign_change(pb))
            b_zero = np.array(b_zero)
            turned = np.isfinite(b_zero)
            # two halves of the hosts, with the constants and with shuffled constants
            r_half, r_null = [], []
            for order in halves:
                h1, h2 = order[: n // 2], order[n // 2:]
                c1, c2 = fitted(h1, y[h1]) @ Xf[h1], fitted(h2, y[h2]) @ Xf[h2]
                r_half.append(float(np.corrcoef(c1[used], c2[used])[0, 1]))
            for order, shuffle in zip(halves, perms):
                ys = y[shuffle]
                h1, h2 = order[: n // 2], order[n // 2:]
                c1, c2 = fitted(h1, ys[h1]) @ Xf[h1], fitted(h2, ys[h2]) @ Xf[h2]
                r_null.append(float(np.corrcoef(c1[used], c2[used])[0, 1]))
            res["fields"][FIELD_LABELS[name]] = {
                "read_as": ATOM[name][1] if name in ATOM else "the co-field itself",
                "plane_radius": rho_star, "plane_value": value,
                "plane_profile": prof.tolist(),
                "radius_interval": [float(np.percentile(b_rho, 2.5)),
                                    float(np.percentile(b_rho, 97.5))],
                "sign_change": sign_change(prof),
                "sign_change_interval": ([float(np.percentile(b_zero[turned], 2.5)),
                                          float(np.percentile(b_zero[turned], 97.5))]
                                         if turned.sum() >= 20 else [None, None]),
                "sign_change_found": float(turned.mean()),
                "largest": {"rho": float(RHO[k[0]]), "z": float(ZED[k[1]]),
                            "value": float(mean_rz[k])},
                "lowest": {"rho": float(RHO[np.unravel_index(int(np.argmin(mean_rz)),
                                                             mean_rz.shape)[0]]),
                           "z": float(ZED[np.unravel_index(int(np.argmin(mean_rz)),
                                                           mean_rz.shape)[1]]),
                           "value": float(mean_rz.min())},
                "turn_sd_over_mean": float(ring.std() / max(abs(ring.mean()), 1e-300)),
                "halves_r": float(np.mean(r_half)), "halves_r_sd": float(np.std(r_half)),
                "null_r": float(np.mean(r_null)), "null_r_sd": float(np.std(r_null))}
            maps[f"{ion}{'_mono' if label else ''}_{name}"] = mean_rz
            if name == "hba":
                # which hosts have an acceptor atom at the point of the largest gain
                has = []
                for m in mols:
                    xyz = m.GetConformer().GetPositions()
                    het = [a.GetIdx() for a in m.GetAtoms()
                           if a.GetSymbol() in CV.DONOR_ELEMENTS]
                    gap = np.hypot(np.hypot(xyz[het, 0], xyz[het, 1]) - RHO[k[0]],
                                   xyz[het, 2] - ZED[k[1]])
                    has.append(bool(np.any(gap <= 1.5)))
                has = np.array(has)
                cage = np.array([g == "cryptand" for g in fam])
                res["at_largest_gain"] = {
                    "within": 1.5, "cryptands_with": int(has[cage].sum()),
                    "cryptands": int(cage.sum()), "others_with": int(has[~cage].sum()),
                    "others": int((~cage).sum()),
                    "mean_log_K_with": float(y[has].mean()) if has.any() else None,
                    "mean_log_K_without": float(y[~has].mean()) if (~has).any() else None}
            if verbose:
                e = res["fields"][FIELD_LABELS[name]]
                print(f"  {ion}+{label} {FIELD_LABELS[name]:13s} largest gain in the plane at "
                      f"{rho_star:.2f} A [{e['radius_interval'][0]:.2f}, "
                      f"{e['radius_interval'][1]:.2f}], sign change at {e['sign_change']:.2f}, "
                      f"halves r {e['halves_r']:+.2f} (shuffled {e['null_r']:+.2f}) "
                      f"({time.time() - t0:.0f} s)", flush=True)
        # the donors themselves: where the frame puts them
        donors = np.vstack([m.GetConformer().GetPositions()[CV.donor_atoms(m)] for m in mols])
        rho_d = np.hypot(donors[:, 0], donors[:, 1])
        phi_d = np.arctan2(donors[:, 1], donors[:, 0])
        res["donors"] = {"n": int(len(donors)), "mean_rho": float(rho_d.mean()),
                         "share_within_plane": float(np.mean(np.abs(donors[:, 2]) <= PLANE)),
                         "sd_z": float(donors[:, 2].std()),
                         "turn_resultant": float(np.abs(np.mean(np.exp(1j * phi_d))))}
        out[f"{ion}+{label}"] = res
    np.savez_compressed(os.path.join(RESULTS, "cofield_maps.npz"), rho=RHO, z=ZED, **maps)
    return out


# ------------------------------------------------------------------------------------ the table
def _f(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def _pair(p):
    return (f"{p['mean']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]"
            + ("" if p["excludes_zero"] else " (includes zero)"))


def write_table(results=None):
    """The table of the five stages, from the JSON each one wrote in the folder `results`."""
    results = results or RESULTS
    path = os.path.join(results, "table.md")

    def read(name):
        p = os.path.join(results, f"{name}.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    data, probe, models, select, cof = (read(n) for n in ("data", "probe", "models", "select",
                                                          "cofield"))
    L = ["# Crown ethers and cryptands in the frame of the cavity (plan item SC5)", ""]
    if data:
        L += [f"{data['hosts']} hosts in {len(data['families'])} families: "
              + ", ".join(f"{k} {v}" for k, v in data["families"].items()) + ".", ""]
        L += ["| Target | Lowest | Highest | Mean | Standard deviation |", "|---|---|---|---|---|"]
        for t in TARGETS:
            s = data["targets"][t]
            L.append(f"| {TARGET_LABELS[t]} | {s['min']:.2f} | {s['max']:.2f} | {s['mean']:.2f} "
                     f"| {s['sd']:.2f} |")
        L += ["", f"The correlation of the two constants is {data['correlation_K_Na']:+.3f}.", "",
              "## The geometries of the complexes", "",
              "A geometry is the host wrapped around the ion: every donor of the ring was held "
              "at the sum of the ionic radii from the ion while the complex was optimised. The "
              "misfit is the root-mean-square difference between the ion-donor distances and "
              "that sum.", "",
              "| Ion | Sum of the ionic radii (A) | Hosts with every ring donor in contact | "
              "Mean misfit (A) | Largest misfit (A) | Mean distance of the ion from the donor "
              "centroid (A) |", "|---|---|---|---|---|---|"]
        for ion in IONS:
            g = data["geometry"][ion]
            L.append(f"| {ion}+ | {g['radius_sum']:.2f} | {g['all_donors_in_contact']} of "
                     f"{data['hosts']} | {g['mean_misfit']:.2f} | {g['max_misfit']:.2f} | "
                     f"{g['mean_ion_to_centroid']:.2f} |")
        L += ["", "| Family | Radius of the donor ring with Na+ (A) | With K+ (A) | Distance of "
              "Na+ from the donor centroid (A) | Of K+ (A) |", "|---|---|---|---|---|"]
        for f in data["families"]:
            a, b = (data["geometry"][i]["by_family"][f] for i in IONS)
            L.append(f"| {f} | {a['donor_radius']:.2f} | {b['donor_radius']:.2f} | "
                     f"{a['ion_to_centroid']:.2f} | {b['ion_to_centroid']:.2f} |")
        L += ["", "| Number read from the complex | Correlation with log K of Na+ | With log K "
              "of K+ |", "|---|---|---|"]
        for name in SHAPE_NAMES:
            L.append(f"| {name} | " + " | ".join(
                _f(data["geometry"][i]["correlations_with_log_K"][name], 2) for i in IONS) + " |")
        L += ["", "| Number read from the formula | Correlation with log K of K+ | With log K of "
              "Na+ | With the selectivity |", "|---|---|---|---|"]
        for name in COUNT_NAMES:
            L.append(f"| {name} | " + " | ".join(
                _f(data["count_correlations"][t][name], 2) for t in TARGETS) + " |")
        L.append("")
    if probe:
        L += ["## Where the probe field puts the ion", "",
              "| Ion | Deepest point of the probe field: mean distance from the ion (A) | Within "
              "1 A | Donor centroid: mean distance from the ion (A) | Within 1 A | Probe closer "
              "than the centroid |", "|---|---|---|---|---|---|"]
        for ion in IONS:
            p = probe[ion]
            L.append(f"| {ion}+ | {p['probe_minimum']['mean']:.2f} | "
                     f"{100 * p['probe_minimum']['within_1A']:.0f} % | "
                     f"{p['donor_centroid']['mean']:.2f} | "
                     f"{100 * p['donor_centroid']['within_1A']:.0f} % | "
                     f"{100 * p['probe_closer']:.0f} % |")
        L.append("")
    if models:
        for design, title in (("random", "five random folds, ten repeats"),
                              ("family", "one family of hosts held out at a time")):
            L += [f"## Cross-validated q2, {title}", "",
                  "| Description | " + " | ".join(TARGET_LABELS[t] for t in TARGETS) + " |",
                  "|---|---|---|---|"]
            for m in MODELS:
                cells = []
                for t in TARGETS:
                    r = models[f"{t}|{design}"]["models"][m]
                    cells.append(_f(r["q2"]) + (f" ± {r['q2_sd']:.3f}" if design == "random"
                                                else "") + f" ({r['rmse']:.2f})")
                L.append(f"| {m} | " + " | ".join(cells) + " |")
            L += ["", "In brackets: the root-mean-square error in log K units.", ""]
            L += [f"### Paired differences of the mean absolute error, {title}", "",
                  "A negative value means the first description has the smaller error. The "
                  "interval is a 95 % bootstrap interval over the hosts.", "",
                  "| First description, second description | "
                  + " | ".join(TARGET_LABELS[t] for t in TARGETS) + " |", "|---|---|---|---|"]
            for key in models[f"K|{design}"]["pairs"]:
                L.append(f"| {key.replace(' | ', ', ')} | " + " | ".join(
                    _pair(models[f"{t}|{design}"]["pairs"][key]) for t in TARGETS) + " |")
            L.append("")
    if select:
        L += ["## The selectivity by three routes", "",
              "| Route | q2, random folds | q2, families held out |", "|---|---|---|"]
        for row in select["random"]["rows"]:
            a, b = select["random"]["rows"][row], select["family"]["rows"][row]
            L.append(f"| {row} | {_f(a['q2'])} ± {a['q2_sd']:.3f} ({a['rmse']:.2f}) | "
                     f"{_f(b['q2'])} ({b['rmse']:.2f}) |")
        tt = select["random"]["rows"]["five fields, the two-task model"]
        L += ["", f"The correlation of the two endpoints chosen inside the folds is "
              f"{tt['mean_rho']:.2f} on average.", "",
              "| Two-task model against | Random folds | Families held out |", "|---|---|---|"]
        for key in select["random"]["pairs"]:
            L.append(f"| {key.split(' | ')[1]} | {_pair(select['random']['pairs'][key])} | "
                     f"{_pair(select['family']['pairs'][key])} |")
        L.append("")
    if cof:
        def rng_(v):
            return "n/a" if v[0] is None else f"{v[0]:.2f} to {v[1]:.2f}"

        L += ["## The co-field of the five-field model, read in the frame", "",
              "The acceptor, hydrophobic and electrostatic co-fields are read as the effect of "
              "one atom placed at a point (one acceptor atom, one hydrophobic atom, one "
              "negative unit charge), in log K units; the steric co-field is read as it is. "
              "The plane is the set of heights of at most "
              f"{cof['grid']['plane']:g} A, and every value is averaged over the turn about z.",
              "",
              "| Ion and hosts | Field | Radius of the largest gain in the plane (A) | 95 % "
              "interval "
              "from bootstrap samples of the hosts | Gain there (log K) | Radius at which the "
              "gain turns from negative to positive (A) | 95 % interval | Correlation of the "
              "maps of two halves of the hosts | The same with shuffled constants |",
              "|---|---|---|---|---|---|---|---|---|"]
        sets = [k for k in cof if k not in ("grid", "fields", "read_as")]
        for ion in sets:
            for name, e in cof[ion]["fields"].items():
                z0 = e["sign_change"]
                L.append(f"| {ion} | {name} | {e['plane_radius']:.2f} | "
                         f"{rng_(e['radius_interval'])} | {e['plane_value']:+.3f} | "
                         + ("none" if z0 is None or not np.isfinite(z0) else f"{z0:.2f}")
                         + f" | {rng_(e['sign_change_interval'])} | "
                         f"{e['halves_r']:+.2f} ± {e['halves_r_sd']:.2f} | "
                         f"{e['null_r']:+.2f} ± {e['null_r_sd']:.2f} |")
        L += ["", "| Ion and hosts | Field | Largest gain anywhere: distance from the axis "
              "(A) | Height "
              "(A) | Gain | Largest loss anywhere: distance from the axis (A) | Height (A) | "
              "Loss | Variation of the gain with the turn, in the plane (sd / mean) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for ion in sets:
            for name, e in cof[ion]["fields"].items():
                g, lo = e["largest"], e["lowest"]
                L.append(f"| {ion} | {name} | {g['rho']:.2f} | {g['z']:+.2f} | "
                         f"{g['value']:+.3f} | {lo['rho']:.2f} | {lo['z']:+.2f} | "
                         f"{lo['value']:+.3f} | {e['turn_sd_over_mean']:.2f} |")
        L += ["", "| Ion and hosts | Hosts | Field width alpha | Sum of the ionic radii (A) | "
              "Mean radius of the donor ring (A) | Ten strongest hosts (A) | Donors within the "
              "plane | Concentration of the donors in the turn (0 none, 1 complete) |",
              "|---|---|---|---|---|---|---|---|"]
        for ion in sets:
            r, dn = cof[ion]["donor_radius"], cof[ion]["donors"]
            L.append(f"| {ion} | {cof[ion]['hosts']} | {cof[ion]['alpha']:g} | "
                     f"{cof[ion]['radius_sum']:.2f} | {r['all']:.2f} | "
                     f"{r['ten strongest hosts']:.2f} | "
                     f"{100 * dn['share_within_plane']:.0f} % | {dn['turn_resultant']:.2f} |")
        L += ["", "| Ion and hosts | Cryptands with an O, N or S atom within 1.5 A of the "
              "largest acceptor gain | Other hosts with one | Mean log K with | Without |",
              "|---|---|---|---|---|"]
        for ion in sets:
            g = cof[ion]["at_largest_gain"]
            if not g["cryptands"]:
                continue
            L.append(f"| {ion} | {g['cryptands_with']} of {g['cryptands']} | "
                     f"{g['others_with']} of {g['others']} | {_f(g['mean_log_K_with'], 2)} | "
                     f"{_f(g['mean_log_K_without'], 2)} |")
        L += ["", "Mean radius of the donor ring by family (A): "
              + "; ".join(f"{ion}+ " + ", ".join(f"{k} {v:.2f}" for k, v in
                          cof[f"{ion}+"]["donor_radius"]["by_family"].items())
                          for ion in IONS) + ".", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    return path


STAGES = ("data", "probe", "models", "select", "cofield")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=STAGES, action="append")
    ap.add_argument("--report", action="store_true", help="only rewrite the table")
    a = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    if not a.report:
        stages = a.stage or list(STAGES)
        d = load()
        print(f"{len(d['names'])} hosts", flush=True)

        def save(name, obj):
            with open(os.path.join(RESULTS, f"{name}.json"), "w", encoding="utf-8",
                      newline="\n") as fh:
                json.dump(obj, fh, indent=1)

        kernels = None
        if "data" in stages:
            save("data", data_stage(d))
        if "probe" in stages:
            save("probe", probe_stage(d))
        if "models" in stages or "select" in stages:
            kernels = build_kernels(d)[0]
        if "models" in stages:
            save("models", models_stage(d, kernels))
        if "select" in stages:
            models = json.load(open(os.path.join(RESULTS, "models.json"), encoding="utf-8"))
            save("select", select_stage(d, kernels, models))
        if "cofield" in stages:
            save("cofield", cofield_stage(d))
    print("wrote", os.path.relpath(write_table(), ROOT))


if __name__ == "__main__":
    main()
