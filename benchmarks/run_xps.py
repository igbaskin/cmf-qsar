#!/usr/bin/env python
"""
Does the field around a carbon atom predict its C 1s binding energy? (plan item LP6)

X-ray photoelectron spectroscopy (XPS) measures the energy needed to remove a core electron.
For carbon this is the C 1s binding energy, near 290 eV, and it shifts by up to about 10 eV with
the chemical surroundings of the atom.  The value belongs to one atom, so the sample is the atom
and its surroundings: the atom-centred environment of `cmfqsar/localenv.py` (plan item LP2).

The data are the two sets of the AugerNet release (Fouda, Zenodo 10.5281/zenodo.19688196,
CC BY 4.0).  `--fetch` downloads them into `benchmarks/cache/xps/augernet_data/`; nothing is
copied into the repository.

* **computed**: 2,116 molecules from QM9 with 1 to 9 heavy atoms, a computed binding energy on
  each of their 8,637 carbon atoms;
* **measured**: 113 molecules with 2 to 23 heavy atoms, a measured gas-phase binding energy on
  570 carbon atoms.  The release splits them into 63 validation and 50 evaluation molecules.

Each molecule is one `.xyz` file and one `_out.txt` file with a value per atom in the same order
(`-1` where there is none).  Bonds are perceived from the geometry.

How it is measured
------------------
* **Inside the computed set**: cross-validation with whole molecules held out.
* **On the measured set**: every model is fitted on all computed atoms and predicts the measured
  atoms.  The measured molecules are larger than the computed ones, so this also tests
  extrapolation in size.  A setting is chosen on the validation molecules and scored on the
  evaluation molecules, as the AugerNet preprint does (arXiv 2604.27070: a mean absolute error of
  0.27 eV on the validation and 0.33 eV on the evaluation molecules, from all 2,116 computed
  molecules).
* **References**, on the same samples and folds:
  - the mean of the training set;
  - a lookup by bonded environment (Morgan environments, the mean of the training atoms that
    share the deepest environment);
  - an atom-centred count: the type of the atom and the atom types in distance shells, out to
    8.5 A and out to 1.8, 2.8 and 3.8 A, with a linear and with a Gaussian kernel;
  - the potential model of XPS shifts: a linear regression on the partial charge of the atom and
    the electrostatic potential at its nucleus from the charges of the other atoms, and the same
    two numbers with a Gaussian kernel.  The charges are Gasteiger charges or GFN2-xTB Mulliken
    charges at the given geometry (one xtb call per molecule, about 0.1 s);
  - the counts with the two xtb numbers added.
* **The field model**: the five fields of the molecule under a Gaussian envelope about the atom,
  the rotation-invariant autocorrelation of that environment, kernel ridge regression; alone and
  with the two xtb numbers added.
* **Intervals** are bootstrapped over molecules, not atoms.

Usage
-----
    python benchmarks/run_xps.py --fetch                 # download the two archives
    python benchmarks/run_xps.py --max-molecules 200     # a quick look
    python benchmarks/run_xps.py --max-molecules 1000    # the stored result (about 45 minutes)
    python benchmarks/run_xps.py                         # all 2,116 molecules (needs about 3 GB)
    python benchmarks/run_xps.py --report                # write the table from the stored result
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
from rdkit.Chem import rdDetermineBonds  # noqa: E402

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localenv import (count_features, count_kernel, cross_validate_lams,  # noqa: E402
                              environment, local_features, local_kernel, types_of, _ridge_path)
from run_local import as_local_result, environment_ids, hose_lookup  # noqa: E402

DATA = os.path.join(HERE, "cache", "xps", "augernet_data")
RESULTS = os.path.join(HERE, "results", "xps")
RHOS = (0.25, 0.5, 1.0, 2.0, 4.0)
#: shell edges of the short count references, in A: bonded neighbours, then one and two more shells
SHORT_SHELLS = ((1.8,), (1.8, 2.8), (1.8, 2.8, 3.8))
#: the two raw archives of the release, with their SHA-256
ARCHIVES = {
    "calc_cebe.tar.gz": "2f71327d144c20179e2945f71532886061babd89a9a7a0943a8cd84c9e1aa279",
    "exp_cebe.tar.gz": "e6e8b8923ec46593456044d6e5f4f7435d1ef4137a48711c26f9a84d15feeaf5",
}
RECORD = "https://zenodo.org/api/records/19688196/files/{name}/content"
#: what the stored result was run with
STORED = 1000
LAMS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
#: width of a Gaussian kernel, as a multiple of the median distance between samples
WIDTHS = (0.5, 1.0, 2.0, 4.0, 8.0)
HOSE_DEPTH = 4


def _say(text):
    print(text, flush=True)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def fetch():
    """Download the two raw archives, check their SHA-256 and unpack them into the cache."""
    import hashlib
    import tarfile
    import urllib.request

    os.makedirs(DATA, exist_ok=True)
    for name, digest in ARCHIVES.items():
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            _say(f"downloading {name}")
            urllib.request.urlretrieve(RECORD.format(name=name), path)
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != digest:
            raise SystemExit(f"{name}: SHA-256 {got} is not the recorded {digest}")
        with tarfile.open(path) as tar:
            tar.extractall(DATA, filter="data")
        _say(f"{name}: checked and unpacked")


def read_xyz(path):
    """Symbols and coordinates of one file; the computed files are in the QM9 layout."""
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    sym, xyz = [], []
    for line in lines[2:2 + n]:
        p = line.replace("*^", "e").split()
        sym.append(p[0])
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return sym, np.asarray(xyz, float)


def to_mol(sym, xyz):
    """An RDKit molecule with bonds perceived from the geometry, atoms in the order of the file."""
    block = f"{len(sym)}\n\n" + "\n".join(f"{s} {x:.8f} {y:.8f} {z:.8f}"
                                          for s, (x, y, z) in zip(sym, xyz))
    mol = Chem.MolFromXYZBlock(block)
    how = "bond orders"
    try:
        full = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(full, charge=0)
        mol = full
    except Exception:
        how = "connectivity only"
        rdDetermineBonds.DetermineConnectivity(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
    return mol, how


def read_set(name, max_molecules=0):
    """Molecules, their names and one value array per molecule (not a number = no value)."""
    d = os.path.join(DATA, name)
    if not os.path.isdir(d):
        raise SystemExit(f"{d} is missing. Run benchmarks/run_xps.py --fetch first.")
    stems = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".xyz") and not f.startswith("._"))
    if max_molecules:
        stems = stems[::max(1, len(stems) // max_molecules)][:max_molecules]
    mols, names, values, how = [], [], [], {}
    for s in stems:
        sym, xyz = read_xyz(os.path.join(d, s + ".xyz"))
        v = np.loadtxt(os.path.join(d, s + "_out.txt"), ndmin=1)
        if len(v) != len(sym):
            raise ValueError(f"{s}: {len(v)} values for {len(sym)} atoms")
        v = np.where((v > 0) & (np.asarray(sym) == "C"), v, np.nan)
        mol, h = to_mol(sym, xyz)
        how[h] = how.get(h, 0) + 1
        mols.append(mol)
        names.append(s)
        values.append(v)
    return mols, names, values, how


# --------------------------------------------------------------------------- #
# Descriptors that are not fields
# --------------------------------------------------------------------------- #

def xtb_charges(mols, names, tag, jobs=4):
    """GFN2-xTB Mulliken charges of every molecule at the geometry of its file, one array each.

    One xtb call per molecule in `benchmarks/cache/xps/xtb/<tag>/<name>/`; a finished job is
    reused.  The atoms keep the order of the file.
    """
    from concurrent.futures import ThreadPoolExecutor
    from cmfqsar.xtb import Config, run_job

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = Config(opt=False, fukui=False)
    root = os.path.join(HERE, "cache", "xps", "xtb", tag)

    def work(k):
        d = os.path.join(root, names[k])
        os.makedirs(d, exist_ok=True)
        m = mols[k]
        xyz = m.GetConformer().GetPositions()
        with open(os.path.join(d, "geom.xyz"), "w", newline="\n") as fh:
            fh.write(f"{m.GetNumAtoms()}\n{names[k]}\n" + "\n".join(
                f"{a.GetSymbol()} {x:.8f} {y:.8f} {z:.8f}"
                for a, (x, y, z) in zip(m.GetAtoms(), xyz)) + "\n")
        q = run_job(d, cfg, 0, 1).charges
        if len(q) != m.GetNumAtoms():
            raise ValueError(f"{names[k]}: xtb returned {len(q)} charges")
        return np.asarray(q, float)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(work, range(len(mols))))


def potential_rows(mols, targets, charges=None):
    """Per measured atom: its partial charge and the potential at its nucleus, in e and e/A.

    The potential is the sum of q_j / r_ij over the other atoms.  These two numbers are the
    classical potential model of XPS chemical shifts.  The charges are Gasteiger charges, or the
    arrays given in `charges` (one per molecule).
    """
    out = np.zeros((len(targets), 2))
    cache = {}
    for r in range(len(targets)):
        k, i = int(targets.molecule[r]), int(targets.atom[r])
        if k not in cache:
            m = Chem.Mol(mols[k])
            if charges is None:
                Chem.rdPartialCharges.ComputeGasteigerCharges(m)
                q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()], float)
                q = np.where(np.isfinite(q), q, 0.0)
            else:
                q = np.asarray(charges[k], float)
            xyz = m.GetConformer().GetPositions()
            dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
            np.fill_diagonal(dist, np.inf)
            cache[k] = (q, (q[None, :] / dist).sum(axis=1))
        q, v = cache[k]
        out[r] = q[i], v[i]
    return out


def gaussian_kernel(A, B, width):
    a2 = (A * A).sum(axis=1)[:, None]
    b2 = (B * B).sum(axis=1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)
    return np.exp(-d2 / (2.0 * width * width))


def median_distance(X, seed=0, n=1500):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), size=min(n, len(X)), replace=False)
    S = X[rows]
    d2 = np.maximum((S * S).sum(1)[:, None] + (S * S).sum(1)[None, :] - 2.0 * S @ S.T, 0.0)
    d = np.sqrt(d2[np.triu_indices(len(S), 1)])
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def standardise(X, Xt):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    keep = sd > 1e-12
    return (X[:, keep] - mu[keep]) / sd[keep], (Xt[:, keep] - mu[keep]) / sd[keep]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def scores(y, p):
    e = p - y
    return dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e * e))),
                offset=float(np.mean(e)), mae_centred=float(np.mean(np.abs(e - e.mean()))),
                r2=float(1.0 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)),
                r=float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else float("nan"))


def boot_mae_difference(y, pa, pb, molecule, runs=2000, seed=0):
    """Mean absolute error of `a` minus that of `b`, with an interval over molecules."""
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    ids = np.unique(molecule)
    sa = np.array([ea[molecule == k].sum() for k in ids])
    sb = np.array([eb[molecule == k].sum() for k in ids])
    n = np.array([(molecule == k).sum() for k in ids], float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        pick = rng.integers(0, len(ids), len(ids))
        out.append((sa[pick].sum() - sb[pick].sum()) / n[pick].sum())
    return [float(ea.mean() - eb.mean()), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5))]


def cross_validated(name, K, targets, folds, seed):
    """The best ridge of the grid for one kernel on the computed set."""
    return max(cross_validate_lams(K, targets, LAMS, folds, seed, name=name), key=lambda r: r.q2)


def kernel_model(name, K, Kx, targets, y_ext, folds, seed, note="", res=None):
    """Cross-validate a kernel on the computed set, then fit on all of it and predict `Kx` rows."""
    t0 = time.time()
    if res is None:
        res = cross_validated(name, K, targets, folds, seed)
    (a, b), = _ridge_path(K, targets.y, [res.lam])
    ext = Kx @ a + b
    return dict(name=name, note=note, lam=float(res.lam), cv=scores(targets.y, res.predicted),
                external=scores(y_ext, ext), cv_pred=res.predicted, ext_pred=ext,
                seconds=round(time.time() - t0))


def best_width(name, X, Xx, targets, y_ext, folds, seed, note=""):
    """The same with a Gaussian kernel, the width chosen by the cross-validation."""
    A, B = standardise(X, Xx)
    m = median_distance(A, seed)
    best, tried = None, {}
    for w in WIDTHS:
        res = cross_validated(name, gaussian_kernel(A, A, w * m), targets, folds, seed)
        tried[w] = float(res.mae)
        if best is None or res.mae < best[1].mae:
            best = (w, res)
    w, res = best
    got = kernel_model(name, gaussian_kernel(A, A, w * m), gaussian_kernel(B, A, w * m), targets,
                       y_ext, folds, seed, note=f"{note}width {w:g} x median", res=res)
    got["width"], got["widths_tried"] = w, tried
    return got


def run(max_molecules=0, folds=5, seed=0, rhos=RHOS, results=RESULTS):
    RDLogger.DisableLog("rdApp.*")
    t_all = time.time()
    mols, names, values, how = read_set("calc_cebe", max_molecules)
    xmols, xnames, xvalues, xhow = read_set("exp_cebe", 0)
    targets = AtomTargets.from_arrays(values, name="C 1s")
    xt = AtomTargets.from_arrays(xvalues, name="C 1s")
    _say(f"computed: {len(targets)} carbon atoms in {targets.n_measured_molecules} molecules; "
         f"bond perception {how}")
    _say(f"measured: {len(xt)} carbon atoms in {xt.n_measured_molecules} molecules; "
         f"bond perception {xhow}")
    y, yx = targets.y, xt.y
    fl = targets.folds(folds, seed)
    rows = []

    def add(row):
        rows.append(row)
        c, e = row["cv"], row["external"]
        _say(f"  {row['name']:44s} computed MAE {c['mae']:.3f} r2 {c['r2']:+.3f} | measured MAE "
             f"{e['mae']:.3f} (offset {e['offset']:+.2f}, centred {e['mae_centred']:.3f}) "
             f"r {e['r']:+.3f}  [{row.get('note', '')}]")

    # the mean
    pm = np.zeros(len(y))
    for te in fl:
        pm[te] = y[np.setdiff1d(np.arange(len(y)), te)].mean()
    add(dict(name="mean of the training set", cv=scores(y, pm),
             external=scores(yx, np.full(len(yx), y.mean())), cv_pred=pm,
             ext_pred=np.full(len(yx), y.mean()), note=""))

    # the lookup by bonded environment
    ids = environment_ids(mols, targets, depth=HOSE_DEPTH)
    idx = environment_ids(xmols, xt, depth=HOSE_DEPTH)
    pred, _ = hose_lookup(ids, y, fl)
    both = np.vstack([ids, idx])
    pe, depth = hose_lookup(both, np.concatenate([y, np.zeros(len(yx))]),
                            [np.arange(len(y), len(y) + len(yx))])
    add(dict(name=f"lookup by bonded environment, {HOSE_DEPTH} bonds", cv=scores(y, pred),
             external=scores(yx, pe[len(y):]), cv_pred=pred, ext_pred=pe[len(y):],
             note=f"measured atoms read at depth: "
                  f"{ {int(d): int((depth[len(y):] == d).sum()) for d in np.unique(depth[len(y):])} }"))

    # the potential model, with Gasteiger charges and with GFN2-xTB charges
    t0 = time.time()
    q_xtb, qx_xtb = xtb_charges(mols, names, "calc"), xtb_charges(xmols, xnames, "exp")
    _say(f"  xtb charges in {time.time() - t0:.0f} s")
    P, Px = potential_rows(mols, targets), potential_rows(xmols, xt)
    Q, Qx = potential_rows(mols, targets, q_xtb), potential_rows(xmols, xt, qx_xtb)
    for label, A, Ax in (("Gasteiger", P, Px), ("xtb", Q, Qx)):
        po = np.zeros(len(y))
        design = np.column_stack([np.ones(len(y)), A])
        for te in fl:
            tr = np.setdiff1d(np.arange(len(y)), te)
            coef = np.linalg.lstsq(design[tr], y[tr], rcond=None)[0]
            po[te] = design[te] @ coef
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        px = np.column_stack([np.ones(len(yx)), Ax]) @ coef
        add(dict(name=f"linear regression on {label} charge and potential", cv=scores(y, po),
                 external=scores(yx, px), cv_pred=po, ext_pred=px,
                 note=f"E = {coef[0]:.2f} {coef[1]:+.2f} q {coef[2]:+.2f} V"))
        add(best_width(f"Gaussian kernel on {label} charge and potential", A, Ax, targets, yx,
                       folds, seed))

    # fields and counts
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    xfields = [build_fields(m, fieldset, name=n) for m, n in zip(xmols, xnames)]
    if fields[0].n_atoms != mols[0].GetNumAtoms():
        raise SystemExit("the fields do not keep every atom of the file in its order")
    types = types_of(fields, min_atoms=10)
    C, Cx = count_features(fields, targets, types), count_features(xfields, xt, types)
    add(kernel_model("atom-centred counts, linear", count_kernel(C), count_kernel(C, Cx),
                     targets, yx, folds, seed, note=f"{len(types)} atom types in 6 shells"))
    add(best_width("atom-centred counts, Gaussian kernel", C, Cx, targets, yx, folds, seed))
    CQ, CQx = np.hstack([C, Q]), np.hstack([Cx, Qx])
    add(kernel_model("counts with xtb charge and potential, linear", count_kernel(CQ),
                     count_kernel(CQ, CQx), targets, yx, folds, seed))
    add(best_width("counts with xtb charge and potential, Gaussian", CQ, CQx, targets, yx,
                   folds, seed))
    # the same counts with short shells: a short envelope is compared with a short count
    for edges in SHORT_SHELLS:
        S, Sx = count_features(fields, targets, types, edges), count_features(xfields, xt, types,
                                                                              edges)
        tag = f"counts to {edges[-1]:g} A"
        add(kernel_model(f"{tag}, linear", count_kernel(S), count_kernel(S, Sx), targets, yx,
                         folds, seed))
        add(best_width(f"{tag}, Gaussian kernel", S, Sx, targets, yx, folds, seed))
        add(best_width(f"{tag}, with xtb charge and potential, Gaussian", np.hstack([S, Q]),
                       np.hstack([Sx, Qx]), targets, yx, folds, seed))

    for rho in rhos:
        t0 = time.time()
        envs = [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
                for r in range(len(targets))]
        xenvs = [environment(xfields[int(xt.molecule[r])], int(xt.atom[r]), rho)
                 for r in range(len(xt))]
        X, Xx = local_features(envs), local_features(xenvs)
        note = (f"{X.shape[1]} channels, {np.mean([e.n_atoms for e in envs]):.1f} atoms per "
                f"environment, features {time.time() - t0:.0f} s")
        add(kernel_model(f"local field, rho {rho:g}, linear", local_kernel(X),
                         local_kernel(X, Xx), targets, yx, folds, seed, note=note))
        add(best_width(f"local field, rho {rho:g}, Gaussian kernel", X, Xx, targets, yx, folds,
                       seed))
        add(best_width(f"local field, rho {rho:g}, with xtb charge and potential, Gaussian",
                       np.hstack([X, Q]), np.hstack([Xx, Qx]), targets, yx, folds, seed))

    # differences against the strongest reference on the measured set
    refs = [r for r in rows if not r["name"].startswith("local field")]
    ref = min(refs, key=lambda r: r["external"]["mae"])
    ref_cv = min(refs, key=lambda r: r["cv"]["mae"])
    _say(f"strongest reference on the measured set: {ref['name']}; inside the computed set: "
         f"{ref_cv['name']}")
    small = np.array([xmols[int(k)].GetNumHeavyAtoms() <= 9 for k in xt.molecule])
    lists = {}
    for part in ("val", "eval"):
        with io.open(os.path.join(DATA, "exp_cebe", f"mol_list_{part}.txt")) as fh:
            lists[part] = {line.strip() for line in fh if line.strip()}
    in_val = np.array([xnames[int(k)] in lists["val"] for k in xt.molecule])
    in_eval = np.array([xnames[int(k)] in lists["eval"] for k in xt.molecule])
    _say(f"published split of the measured set: validation {len(lists['val'])} molecules, "
         f"{int(in_val.sum())} atoms; evaluation {len(lists['eval'])} molecules, "
         f"{int(in_eval.sum())} atoms; in neither {int((~in_val & ~in_eval).sum())} atoms")
    for r in rows:
        r["validation"] = scores(yx[in_val], r["ext_pred"][in_val])
        r["evaluation"] = scores(yx[in_eval], r["ext_pred"][in_eval])
    _say("  mean absolute error on the published validation / evaluation molecules, eV")
    for r in rows:
        _say(f"    {r['name']:66s} {r['validation']['mae']:.3f} / {r['evaluation']['mae']:.3f}")
    families = (("field", [r for r in rows if r["name"].startswith("local field")]),
                ("reference", [r for r in rows if not r["name"].startswith("local field")]))
    chosen = {}
    for family, members in families:
        pick = min(members, key=lambda r: r["validation"]["mae"])
        pick_cv = min(members, key=lambda r: r["cv"]["mae"])
        chosen[family] = pick
        _say(f"  {family} model chosen on the validation molecules: {pick['name']}, evaluation "
             f"MAE {pick['evaluation']['mae']:.3f}; chosen by the cross-validation: "
             f"{pick_cv['name']}, evaluation MAE {pick_cv['evaluation']['mae']:.3f}")
    gap = boot_mae_difference(yx[in_eval], chosen["field"]["ext_pred"][in_eval],
                              chosen["reference"]["ext_pred"][in_eval], xt.molecule[in_eval],
                              seed=seed)
    _say(f"  chosen field model minus chosen reference on the evaluation molecules: "
         f"{gap[0]:+.3f} [{gap[1]:+.3f}, {gap[2]:+.3f}]")
    gap_all = boot_mae_difference(yx, chosen["field"]["ext_pred"], chosen["reference"]["ext_pred"],
                                  xt.molecule, seed=seed)
    _say(f"  the same on all measured atoms: {gap_all[0]:+.3f} [{gap_all[1]:+.3f}, "
         f"{gap_all[2]:+.3f}]")
    for r in rows:
        r["external_against_reference"] = boot_mae_difference(yx, r["ext_pred"], ref["ext_pred"],
                                                              xt.molecule, seed=seed)
        r["cv_against_reference"] = boot_mae_difference(y, r["cv_pred"], ref_cv["cv_pred"],
                                                        targets.molecule, seed=seed)
        r["external_small"] = scores(yx[small], r["ext_pred"][small])
        r["external_large"] = scores(yx[~small], r["ext_pred"][~small])
        if r["name"].startswith("local field"):
            d, lo, hi = r["external_against_reference"]
            c, clo, chi = r["cv_against_reference"]
            _say(f"  {r['name']}: measured MAE minus reference {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
                 f"computed {c:+.3f} [{clo:+.3f}, {chi:+.3f}]")
    out = dict(folds=folds, seed=seed, n_computed=len(y), n_computed_molecules=len(mols),
               n_measured=len(yx), n_measured_molecules=len(xmols),
               n_measured_small=int(small.sum()), bond_perception=dict(computed=how, measured=xhow),
               spread=dict(computed=float(np.std(y)), measured=float(np.std(yx))),
               chosen={k: v["name"] for k, v in chosen.items()}, chosen_gap=gap, chosen_gap_all=gap_all,
               max_molecules=int(max_molecules),
               reference_external=ref["name"], reference_cv=ref_cv["name"],
               rows=[{k: v for k, v in r.items() if k not in ("cv_pred", "ext_pred")}
                     for r in rows],
               minutes=round((time.time() - t_all) / 60.0, 1))
    os.makedirs(results, exist_ok=True)
    tag = "c1s" if max_molecules == STORED else f"c1s_{max_molecules or 'all'}"
    with io.open(os.path.join(results, f"{tag}.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    _say(f"done in {out['minutes']} min")
    if max_molecules == STORED:
        write_table(results)
    return out


def report(results=RESULTS):
    """The table of the stored result, as text."""
    path = os.path.join(results, "c1s.json")
    if not os.path.exists(path):
        return "No stored result. Run benchmarks/run_xps.py --max-molecules 1000 first.\n"
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    L = ["# C 1s binding energies of carbon atoms (plan item LP6)", "",
         f"Training: {d['n_computed']} carbon atoms of {d['n_computed_molecules']} computed "
         f"molecules of the AugerNet release ({d['folds']} folds, whole molecules held out). "
         f"Test: {d['n_measured']} measured values in {d['n_measured_molecules']} molecules, "
         f"split by the release into validation and evaluation molecules. "
         f"Standard deviation of the values: {d['spread']['computed']:.2f} eV computed, "
         f"{d['spread']['measured']:.2f} eV measured.", "",
         "Mean absolute error in eV. \"Small\" and \"large\" are the measured molecules with at "
         "most 9 and with more than 9 heavy atoms; the computed molecules have at most 9.", "",
         "| Model | Computed set, cross-validation | Validation molecules | Evaluation molecules "
         "| Small molecules | Large molecules | Setting |",
         "|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        setting = r.get("note", "") or ""
        if r.get("lam") is not None:
            setting = (setting + "; " if setting else "") + f"ridge {r['lam']:g}"
        L.append(f"| {r['name']} | {r['cv']['mae']:.3f} | {r['validation']['mae']:.3f} | "
                 f"{r['evaluation']['mae']:.3f} | {r['external_small']['mae']:.3f} | "
                 f"{r['external_large']['mae']:.3f} | {setting} |")
    g, ga = d["chosen_gap"], d["chosen_gap_all"]
    L += ["",
          f"Field model chosen on the validation molecules: {d['chosen']['field']}. "
          f"Reference chosen on the validation molecules: {d['chosen']['reference']}.", "",
          f"Field model minus reference, mean absolute error in eV with a 95 % interval over "
          f"molecules: {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] on the evaluation molecules, "
          f"{ga[0]:+.3f} [{ga[1]:+.3f}, {ga[2]:+.3f}] on all measured atoms.", "",
          "Published network on the same split (arXiv 2604.27070, all 2,116 computed molecules): "
          "0.27 eV on the validation and 0.33 eV on the evaluation molecules.", ""]
    return "\n".join(L)


def write_table(results=RESULTS):
    text = report(results)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-molecules", type=int, default=0,
                    help="use only this many computed molecules (0 = all)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, nargs="*", default=list(RHOS),
                    help="envelope width parameters of the local field, in 1/A^2")
    ap.add_argument("--fetch", action="store_true", help="download and unpack the two archives")
    ap.add_argument("--report", action="store_true",
                    help="write the table of the stored result and print it")
    a = ap.parse_args()
    if a.fetch:
        return fetch()
    if a.report:
        return print(write_table())
    run(a.max_molecules, a.folds, a.seed, a.rho)


if __name__ == "__main__":
    main()
