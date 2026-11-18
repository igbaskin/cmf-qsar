#!/usr/bin/env python
"""
The glass transition temperature of homopolymers from oligomers built by the program, against
the descriptors published with the data set.

The set is ``examples/polymer_tg/tg.csv``, made by ``benchmarks/prepare_polymer_tg.py`` from the
curated PolyMetriX set (Kunchapu and Jablonka, npj Comput. Mater. 2025; CC BY 4.0): 7,316
homopolymers, each a repeat unit written as SMILES with two attachment points, with its Tg in
kelvin and one of 22 polymer classes.

The question, fixed before the measurement
------------------------------------------
Can the program, with the oligomer builder of :mod:`cmfqsar.polymer`, predict a physical property
of polymers as well as the descriptors made for polymers?  The reference is the published one:
the 100 hierarchical descriptors of the PolyMetriX library (side chain, backbone, full polymer)
with gradient boosting at its default settings, the model of the paper (its Fig. 6 gives a mean
absolute error of about 33.5 K on random 5-fold splits and about 41 K on cluster splits).  The
main comparison is

    field kernel of the central repeat unit  minus  PolyMetriX descriptors with gradient boosting,

the difference of cross-validated q2 with a 95 % paired bootstrap interval, on random folds and on
folds that hold out whole polymer classes.  The other models say where a difference comes from.

The full run of 18 November 2026 answered it: the field alone is 0.06 to 0.07 behind; the field
plus the eight counts equals the published model (+0.009 on random folds, +0.001 with classes
held out); with the counts, the 3D distances equal the bond graph; ECFP4 in a Tanimoto kernel is
the best model on both tests.

The sample
----------
Every repeat unit becomes a trimer capped with hydrogen, built by :func:`cmfqsar.polymer.build`:
the stereocentres the repeat unit leaves open fixed once and copied into every unit
("isotactic"), the most extended of 4 ETKDG conformations, optimised with MMFF94.  The fields are
the five physicochemical fields of the property benchmarks (Gasteiger electrostatic, steric,
hydrophobic, acceptor, donor).

The models
----------
* **invariant, oligomer** -- the cosine-normalised field autocorrelation of the whole trimer, caps
  included: what ``cmf build --kernel autocorrelation`` fits on the SDF of ``cmf polymers``.
* **invariant, central unit** -- the same autocorrelation counting only the atom pairs whose first
  atom lies in the middle repeat unit (:func:`cmfqsar.polymer.autocorrelation_rows`): the field
  environment of one repeat unit inside a chain, without the ends.
* **topological, central unit** -- the same channels with the distance counted in bonds instead
  of angstrom, from 0 bonds up to a maximum chosen by the evidence: the graph alone, no conformer.
* **eight counts** -- a Gaussian kernel on the eight counts of the property benchmarks (molecular
  weight, donors, acceptors, rotatable bonds, rings, aromatic rings, polar surface area, fraction
  of sp3 carbons), computed on the repeat unit capped with hydrogen.
* **PolyMetriX descriptors, kernel** -- a Gaussian kernel on the 100 PolyMetriX descriptors,
  standardised.
* **PolyMetriX descriptors, gradient boosting** -- scikit-learn's GradientBoostingRegressor with
  its default settings on the same 100 descriptors: the published model.  It is not chosen by
  the evidence; it has no setting to choose.
* **ECFP4 of the repeat unit** -- the Tanimoto kernel on 2048-bit Morgan fingerprints of radius 2
  of the PSMILES, attachment points included: the Morgan reference of the paper.
* **central unit + eight counts**, **central unit + PolyMetriX descriptors** and **topological +
  eight counts** -- a sum of two kernels, the second scaled by h; the setting of each part is the
  one its single model chose on the same training block, and only h and the ridge are chosen here.

How the numbers are made
------------------------
Kernel ridge regression with a bias; in every block the setting and the ridge are chosen by the
log evidence on the training block (the eigendecomposition route of ``run_logp.py``, which checks
itself against ``log_evidence`` and ``krr_fit``).  Two fold schemes: five random folds repeated
three times, and five folds of whole polymer classes repeated three times (the 22 classes are
dealt to the five folds in a random order, each to the fold that is smallest at that moment).
The paired bootstrap runs over polymers for the random folds and over classes for the class
folds, since a class is the unit held out there.  The channel scale and the standardisation of
the descriptors use all polymers and no Tg value.

Usage
-----
    python benchmarks/run_polymer_tg.py --build      # the trimers and their features (cached)
    python benchmarks/run_polymer_tg.py              # the models; builds what is missing
    python benchmarks/run_polymer_tg.py --quick 600  # a smoke test on 600 polymers, no cache
    python benchmarks/run_polymer_tg.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import run_logp as L  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "polymer_tg", "tg.csv")
RESULTS = os.path.join(HERE, "results", "polymer_tg")
CACHE = os.path.join(HERE, "cache", "polymer_tg")
RAW = os.path.join(CACHE, "raw", "tg.csv")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
UNITS, CONFORMERS, BUILD_SEED = 3, 4, 0
#: the logP benchmark chose 0.03-0.05 A; the first full run here chose 0.02, the narrowest then
#: offered, in every block, so the list goes down to 0.005
SIGMAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0)
D_MAX = 12.0
TOPO_MAX = (2, 4, 6, 8, 12, 20, 30, 40)
COUNT_WIDTHS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
#: the first full run chose h = 10, the largest then offered, for the central unit plus the
#: PolyMetriX descriptors in every block, so the list goes on to 100
HS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LAMS = L.LAMS
FOLDS, REPEATS, SEED = 5, 3, 0
SCHEMES = ("random", "class")
MODELS = ("invariant, oligomer", "invariant, central unit", "topological, central unit",
          "eight counts", "PolyMetriX descriptors, kernel",
          "PolyMetriX descriptors, gradient boosting", "ECFP4 of the repeat unit",
          "central unit + eight counts", "central unit + PolyMetriX descriptors",
          "topological + eight counts")
MAIN = ("invariant, central unit", "PolyMetriX descriptors, gradient boosting")
#: the published mean absolute errors read off Fig. 6 of the paper, in kelvin
PUBLISHED = {"PolyMetriX descriptors (all three levels), gradient boosting":
             {"random 5-fold": 33.5, "5 clusters": 41.0},
             "Morgan fingerprint, gradient boosting": {"random 5-fold": 37.0, "5 clusters": 47.0}}

log = L.log


# --------------------------------------------------------------------------- the data

def load_table(quick: int = 0):
    with open(DATA, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if quick:
        keep = np.linspace(0, len(rows) - 1, min(quick, len(rows))).astype(int)
        rows = [rows[i] for i in keep]
    return rows


def polymetrix_features(rows) -> np.ndarray:
    """The 100 PolyMetriX descriptors of each polymer, from the cached download, by PSMILES."""
    import prepare_polymer_tg as P
    raw = P.read_raw()
    cols = P.feature_columns(raw[0])
    by_smiles = {r["PSMILES"]: [float(r[c]) for c in cols] for r in raw}
    return np.array([by_smiles[r["psmiles"]] for r in rows])


# --------------------------------------------------------------------------- the build

def _specs():
    from cmfqsar.invariant import AutocorrelationSpec
    return [AutocorrelationSpec(d_max=D_MAX, step=s, sigma=s) for s in SIGMAS]


def topological_rows(mol: Chem.Mol, W: np.ndarray, centre: np.ndarray, d_max: int) -> np.ndarray:
    """The field autocorrelation with the distance in bonds, counting pairs that start in `centre`.

    The channel of field pair (f, g) at k bonds is S_fg + S_gf for f != g and S_ff for f = g, with
    S_fg = sum over atoms a in the centre and atoms b k bonds from a of W_af W_bg: the convention of
    :func:`cmfqsar.polymer.autocorrelation_rows` with a count of bonds in place of a Gaussian.
    """
    from cmfqsar.invariant import field_pairs
    D = Chem.GetDistanceMatrix(mol)
    pairs = field_pairs(W.shape[1])
    Wc = W * np.asarray(centre, float)[:, None]
    out = np.zeros((len(pairs), d_max + 1))
    for k in range(d_max + 1):
        S = Wc.T @ (D == k).astype(float) @ W
        for p, (f, g) in enumerate(pairs):
            out[p, k] = S[f, g] + (S[g, f] if f != g else 0.0)
    return out.ravel()


def _features_one(task):
    """Worker: one repeat unit to its trimer, fields, rows and counts."""
    k, psmiles = task
    try:
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
        from cmfqsar.fields import FieldSet, build_fields
        from cmfqsar.polymer import (autocorrelation_rows, build, central_unit, oligomer,
                                     parse_repeat_unit)
        RDLogger.DisableLog("rdApp.*")
        fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                             param_scheme="rdkit")
        mol, info = build(psmiles, n=UNITS, conformers=CONFORMERS, seed=BUILD_SEED)
        centre = central_unit(mol)
        mf = build_fields(mol, fs, f"p{k}")
        specs = _specs()
        whole = autocorrelation_rows(mf, specs)
        middle = autocorrelation_rows(mf, specs, centre=centre)
        topo = topological_rows(mol, np.asarray(mf.weights, float), centre, max(TOPO_MAX))
        mono = Chem.RemoveHs(oligomer(parse_repeat_unit(psmiles), n=1, add_hydrogens=False))
        counts = [Descriptors.MolWt(mono), Lipinski.NumHDonors(mono), Lipinski.NumHAcceptors(mono),
                  rdMolDescriptors.CalcNumRotatableBonds(mono), rdMolDescriptors.CalcNumRings(mono),
                  rdMolDescriptors.CalcNumAromaticRings(mono), rdMolDescriptors.CalcTPSA(mono),
                  rdMolDescriptors.CalcFractionCSP3(mono)]
        return (k, [w.astype(np.float32) for w in whole], [m.astype(np.float32) for m in middle],
                topo.astype(np.float32), np.array(counts), Chem.MolToMolBlock(mol), info, "")
    except Exception as exc:                     # noqa: BLE001 - reported per polymer
        return k, None, None, None, None, "", {}, f"{type(exc).__name__}: {exc}"


def build_features(rows, jobs: int, use_cache: bool, tag: str = "full"):
    """Trimers, rows of both autocorrelations at every width, topological rows, eight counts.

    Cached as one npz per width and kind under ``benchmarks/cache/polymer_tg/`` with the PSMILES
    of every row, so that a later run reads them instead of building again.
    """
    ids = np.array([r["psmiles"] for r in rows])
    meta = os.path.join(CACHE, f"build_{tag}.npz")
    if use_cache and os.path.isfile(meta):
        z = np.load(meta, allow_pickle=False)
        if z["ids"].shape == ids.shape and np.all(z["ids"] == ids):
            log(f"features read from the cache ({tag})")
            return {"ok": z["ok"], "topo": z["topo"], "counts": z["counts"],
                    "span": z["span"], "atoms": z["atoms"]}
    from multiprocessing import get_context
    specs = _specs()
    n = len(rows)
    widths = [15 * s.n_bins for s in specs]
    whole = [np.zeros((n, w), np.float32) for w in widths]
    middle = [np.zeros((n, w), np.float32) for w in widths]
    topo = np.zeros((n, 15 * (max(TOPO_MAX) + 1)), np.float32)
    counts = np.zeros((n, 8))
    ok = np.zeros(n, bool)
    span = np.full(n, np.nan)
    atoms = np.zeros(n, int)
    errors = {}
    sdf_path = os.path.join(CACHE, f"trimers_{tag}.sdf")
    os.makedirs(CACHE, exist_ok=True)
    blocks = [""] * n
    t0 = time.time()
    tasks = [(k, r["psmiles"]) for k, r in enumerate(rows)]
    with get_context("spawn").Pool(int(jobs)) as pool:
        for done, res in enumerate(pool.imap_unordered(_features_one, tasks, chunksize=2)):
            k, wh, mi, tp, ct, block, info, err = res
            if err:
                errors[rows[k]["id"]] = err
            else:
                for j in range(len(specs)):
                    whole[j][k] = wh[j]
                    middle[j][k] = mi[j]
                topo[k], counts[k], ok[k] = tp, ct, True
                span[k], atoms[k] = info.get("span", np.nan), int(info.get("atoms", 0))
                blocks[k] = block
            if (done + 1) % 250 == 0 or done + 1 == n:
                log(f"  built {done + 1}/{n} ({time.time() - t0:.0f} s, {len(errors)} failed)")
    with open(sdf_path, "w", encoding="utf-8", newline="\n") as fh:
        for r, b in zip(rows, blocks):
            if b:
                # every column of the table goes into the record, so that another benchmark
                # (run_polymer_packing.py) can share this function with its own columns
                props = "".join(f"> <{k}>\n{v}\n\n" for k, v in r.items() if k != "id")
                fh.write(r["id"] + b[b.index("\n"):] + props + "$$$$\n")
    if use_cache:
        for s, w, m in zip(SIGMAS, whole, middle):
            np.save(os.path.join(CACHE, f"whole_{tag}_{s:g}.npy"), w)
            np.save(os.path.join(CACHE, f"central_{tag}_{s:g}.npy"), m)
        np.savez(meta, ids=ids, ok=ok, topo=topo, counts=counts, span=span, atoms=atoms)
        with open(os.path.join(CACHE, f"errors_{tag}.json"), "w", encoding="utf-8") as fh:
            json.dump(errors, fh, indent=1)
    out = {"ok": ok, "topo": topo, "counts": counts, "span": span, "atoms": atoms,
           "errors": errors}
    if not use_cache:
        out["whole"] = dict(zip(SIGMAS, whole))
        out["central"] = dict(zip(SIGMAS, middle))
    return out


def autocorr_rows(feat, kind: str, sigma: float, tag: str, keep: np.ndarray) -> np.ndarray:
    if kind in feat:
        return np.asarray(feat[kind][sigma], np.float32)[keep]
    return np.load(os.path.join(CACHE, f"{kind}_{tag}_{sigma:g}.npy"))[keep]


def _rows_one(task):
    """Worker: the autocorrelation rows of one stored trimer at the widths asked for."""
    k, block, psmiles, sigmas = task
    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.invariant import AutocorrelationSpec
    from cmfqsar.polymer import autocorrelation_rows, oligomer, unit_labels
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromMolBlock(block, removeHs=False)
    labels = unit_labels(oligomer(psmiles, n=UNITS))
    centre = labels == UNITS // 2
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = build_fields(mol, fs, f"p{k}")
    specs = [AutocorrelationSpec(d_max=D_MAX, step=s, sigma=s) for s in sigmas]
    whole = autocorrelation_rows(mf, specs)
    middle = autocorrelation_rows(mf, specs, centre=centre)
    return k, [w.astype(np.float32) for w in whole], [m.astype(np.float32) for m in middle]


def extend_widths(rows, jobs: int, tag: str = "full") -> None:
    """Rows at the widths of :data:`SIGMAS` that the cache does not hold yet, from the trimers.

    The trimers are read back from ``trimers_<tag>.sdf``, so that a width added to the list
    later describes exactly the structures the other widths describe.  The unit labels are not
    in that file; they are those of the same oligomer built again, whose atoms come in the same
    order (checked for all 7,302 trimers on 18 November 2026).
    """
    from cmfqsar.invariant import AutocorrelationSpec
    missing = [s for s in SIGMAS
               if not os.path.isfile(os.path.join(CACHE, f"central_{tag}_{s:g}.npy"))]
    if not missing:
        return
    z = np.load(os.path.join(CACHE, f"build_{tag}.npz"))
    ok = z["ok"]
    sup = Chem.SDMolSupplier(os.path.join(CACHE, f"trimers_{tag}.sdf"), removeHs=False,
                             sanitize=False)
    blocks = {}
    for m in sup:
        name = m.GetProp("_Name")
        blocks[name] = Chem.MolToMolBlock(m)
    ids = [r["id"] for r in rows]
    tasks = [(k, blocks[ids[k]], rows[k]["psmiles"], missing)
             for k in range(len(rows)) if ok[k]]
    n = len(rows)
    nb = [15 * AutocorrelationSpec(d_max=D_MAX, step=s, sigma=s).n_bins for s in missing]
    whole = [np.zeros((n, w), np.float32) for w in nb]
    middle = [np.zeros((n, w), np.float32) for w in nb]
    from multiprocessing import get_context
    t0 = time.time()
    with get_context("spawn").Pool(int(jobs)) as pool:
        for done, (k, wh, mi) in enumerate(pool.imap_unordered(_rows_one, tasks, chunksize=8)):
            for j in range(len(missing)):
                whole[j][k], middle[j][k] = wh[j], mi[j]
            if (done + 1) % 1000 == 0 or done + 1 == len(tasks):
                log(f"  widths {missing}: {done + 1}/{len(tasks)} ({time.time() - t0:.0f} s)")
    for s, w, m in zip(missing, whole, middle):
        np.save(os.path.join(CACHE, f"whole_{tag}_{s:g}.npy"), w)
        np.save(os.path.join(CACHE, f"central_{tag}_{s:g}.npy"), m)


def field_kernel(F: np.ndarray, sigma: float) -> np.ndarray:
    """The cosine kernel of ``cmfqsar.invariant.autocorrelation_kernel`` with its channel scale.

    The rows stay in single precision and only the kernel is double: at 0.005 A a row has 36,000
    channels, and the double-precision copies of the library function would need 6 GB.  The
    block scales are the root mean squares of :func:`cmfqsar.invariant.channel_scale`, summed in
    double precision.
    """
    from cmfqsar.invariant import AutocorrelationSpec, block_sizes
    spec = AutocorrelationSpec(d_max=D_MAX, step=sigma, sigma=sigma)
    A = np.asarray(F, np.float32)
    start = 0
    rms = []
    for width in block_sizes(spec, len(FIELDS)):
        blk = A[:, start:start + width].astype(np.float64)
        rms.append(float(np.sqrt(np.mean(blk ** 2))))
        start += width
    floor = max(1e-300, max(rms) * 1e-9)
    start = 0
    for width, r in zip(block_sizes(spec, len(FIELDS)), rms):
        A[:, start:start + width] *= np.float32(1.0 / r if r > floor else 0.0)
        start += width
    K = (A @ A.T).astype(np.float64)
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / np.outer(d, d)


def topo_kernel(T: np.ndarray, d_max: int) -> np.ndarray:
    """The cosine kernel on the topological rows up to `d_max` bonds, scaled per field pair."""
    from cmfqsar.invariant import autocorrelation_kernel, channel_scale
    full = max(TOPO_MAX) + 1
    X = T.reshape(len(T), 15, full)[:, :, :d_max + 1].reshape(len(T), -1)
    blocks = [d_max + 1] * 15
    sc = channel_scale(X, d_max + 1, blocks=blocks)
    return autocorrelation_kernel(X, scale=sc, n_bins=d_max + 1, blocks=blocks)


# --------------------------------------------------------------------------- the folds

def class_folds(classes: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    """Whole classes dealt in a random order, each to the fold that is smallest at that moment."""
    rng = np.random.RandomState(seed)
    names = sorted(set(classes.tolist()))
    order = rng.permutation(len(names))
    sizes = np.zeros(k)
    fold_of = {}
    for i in order:
        c = names[i]
        f = int(np.argmin(sizes + 1e-9 * np.arange(k)))
        fold_of[c] = f
        sizes[f] += np.sum(classes == c)
    return [np.where(np.array([fold_of[c] == f for c in classes]))[0] for f in range(k)]


def make_blocks(n: int, classes: np.ndarray):
    """(scheme, repeat, training indices, test indices) for every block."""
    out = []
    for rep in range(REPEATS):
        for te in L.RW.folds(n, FOLDS, SEED + rep):
            out.append(("random", rep, np.setdiff1d(np.arange(n), te), te))
    for rep in range(REPEATS):
        for te in class_folds(classes, FOLDS, SEED + rep):
            out.append(("class", rep, np.setdiff1d(np.arange(n), te), te))
    return out


class Lazy:
    """A kernel made on first use and kept, so that a fully cached setting never builds it."""

    def __init__(self, make):
        self.make, self.value = make, None

    def __call__(self):
        if self.value is None:
            self.value = self.make()
        return self.value


class Model:
    """The candidate settings of one model, each evaluated on every block, chosen per block.

    Every evaluated setting is kept in ``benchmarks/cache/polymer_tg/settings/`` under a key made
    of the model, the setting, the Tg values and the blocks, so that a run with a wider grid
    evaluates only the settings it adds.  `K` may be a function, which is then called only when
    the setting is not in the cache.
    """

    cache_dir: Optional[str] = None

    def __init__(self, name: str):
        self.name = name
        self.labels: List[str] = []
        self.cands = []
        self.fixed = None                       # predictions of a model with nothing to choose

    def _key(self, label, y, blocks, only) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(f"{self.name}|{label}|{sorted(only) if only is not None else 'all'}|".encode())
        h.update(np.asarray(y, np.float64).tobytes())
        h.update(np.asarray(LAMS, np.float64).tobytes())
        for scheme, rep, tr, te in blocks:
            h.update(f"{scheme}{rep}".encode())
            h.update(np.asarray(te, np.int64).tobytes())
        return h.hexdigest()[:32]

    def add(self, label, K, y, blocks, only=None):
        self.labels.append(label)
        path = None
        if self.cache_dir is not None:
            path = os.path.join(self.cache_dir, self._key(label, y, blocks, only) + ".npz")
            if os.path.isfile(path):
                z = np.load(path)
                self.cands.append({"ev": z["ev"], "pred": [z[f"p{k}"] if f"p{k}" in z else None
                                                           for k in range(len(blocks))]})
                return
        cand = L.evaluate(K() if callable(K) else K, y, blocks, only)
        self.cands.append(cand)
        if path is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez(path, ev=cand["ev"], **{f"p{k}": p for k, p in enumerate(cand["pred"])
                                            if p is not None})

    def best_label(self, k: int) -> str:
        ev = np.array([c["ev"][k] for c in self.cands])
        return self.labels[np.unravel_index(int(np.nanargmax(ev)), ev.shape)[0]]

    def select(self, n: int, blocks):
        pred = {s: np.full((REPEATS, n), np.nan) for s in SCHEMES}
        chosen = {s: {} for s in SCHEMES}
        for k, (scheme, rep, _, te) in enumerate(blocks):
            if self.fixed is not None:
                pred[scheme][rep, te] = self.fixed[k]
                continue
            ev = np.array([c["ev"][k] for c in self.cands])
            c, j = np.unravel_index(int(np.nanargmax(ev)), ev.shape)
            pred[scheme][rep, te] = self.cands[c]["pred"][k][j]
            key = f"{self.labels[c]}, ridge {LAMS[j]:g}"
            chosen[scheme][key] = chosen[scheme].get(key, 0) + 1
        return pred, chosen


def gradient_boosting(X: np.ndarray, y: np.ndarray, blocks) -> List[np.ndarray]:
    from sklearn.ensemble import GradientBoostingRegressor
    out = []
    for _, _, tr, te in blocks:
        m = GradientBoostingRegressor(random_state=0).fit(X[tr], y[tr])
        out.append(m.predict(X[te]))
    return out


def paired_groups(y, a, b, groups, draws: int = 2000, seed: int = SEED):
    """q2 of a minus q2 of b, with a bootstrap interval that resamples whole groups."""
    from cmfqsar.model import r2_score
    rng = np.random.RandomState(seed)
    names = np.array(sorted(set(groups.tolist())))
    members = [np.where(groups == g)[0] for g in names]
    diffs = []
    for _ in range(draws):
        idx = np.concatenate([members[i] for i in rng.randint(0, len(names), len(names))])
        diffs.append(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(r2_score(y, a) - r2_score(y, b)), "lo": float(lo), "hi": float(hi)}


# --------------------------------------------------------------------------- the run

def run(quick: int = 0, jobs: int = 16, build_only: bool = False) -> Optional[dict]:
    from cmfqsar.descriptors import fingerprint_kernel
    from cmfqsar.model import r2_score, rmse

    t0 = time.time()
    L.selfcheck()
    rows = load_table(quick)
    tag = f"quick{quick}" if quick else "full"
    feat = build_features(rows, jobs, use_cache=not quick, tag=tag)
    keep = np.where(feat["ok"])[0]
    log(f"{len(keep)} of {len(rows)} trimers built ({time.time() - t0:.0f} s); median "
        f"{np.median(feat['atoms'][keep]):.0f} atoms, end-to-end distance median "
        f"{np.nanmedian(feat['span'][keep]):.1f} A")
    if build_only:
        return None
    if not quick:
        extend_widths(rows, jobs, tag)
        Model.cache_dir = os.path.join(CACHE, "settings")
    rows = [rows[i] for i in keep]
    y = np.array([float(r["tg_k"]) for r in rows])
    classes = np.array([r["polymer_class"] for r in rows])
    n = len(rows)
    blocks = make_blocks(n, classes)
    log(f"{n} polymers, {len(blocks)} blocks, Tg {y.mean():.1f} +- {y.std():.1f} K")

    models = {m: Model(m) for m in MODELS}
    Zc = L.standardise(feat["counts"][keep])
    Xp = polymetrix_features(rows)
    Zp = L.standardise(Xp[:, Xp.std(axis=0) > 0])
    T = np.asarray(feat["topo"][keep], float)
    for w in COUNT_WIDTHS:
        models["eight counts"].add(f"width {w:g}", lambda w=w: L.gaussian_at(Zc, w), y, blocks)
        models["PolyMetriX descriptors, kernel"].add(f"width {w:g}",
                                                     lambda w=w: L.gaussian_at(Zp, w), y, blocks)
    log(f"count and descriptor kernels done ({time.time() - t0:.0f} s)")
    models["PolyMetriX descriptors, gradient boosting"].fixed = gradient_boosting(Xp, y, blocks)
    log(f"gradient boosting done ({time.time() - t0:.0f} s)")
    units = [Chem.MolFromSmiles(r["psmiles"]) for r in rows]
    models["ECFP4 of the repeat unit"].add("Tanimoto", lambda: fingerprint_kernel(units), y,
                                           blocks)
    for dm in TOPO_MAX:
        models["topological, central unit"].add(f"up to {dm} bonds",
                                                lambda dm=dm: topo_kernel(T, dm), y, blocks)
    log(f"topological kernels done ({time.time() - t0:.0f} s)")
    for s in SIGMAS:
        for kind, name in (("whole", "invariant, oligomer"), ("central", "invariant, central unit")):
            models[name].add(f"sigma {s:g}",
                             lambda kind=kind, s=s: field_kernel(
                                 autocorr_rows(feat, kind, s, tag, keep), s), y, blocks)
        log(f"sigma {s:g} done ({time.time() - t0:.0f} s)")

    width = [models["invariant, central unit"].best_label(k) for k in range(len(blocks))]
    cwidth = [models["eight counts"].best_label(k) for k in range(len(blocks))]
    pwidth = [models["PolyMetriX descriptors, kernel"].best_label(k) for k in range(len(blocks))]
    reach = [models["topological, central unit"].best_label(k) for k in range(len(blocks))]
    for s in SIGMAS:
        here = {k for k, w in enumerate(width) if w == f"sigma {s:g}"}
        if not here:
            continue
        Kf = Lazy(lambda s=s: field_kernel(autocorr_rows(feat, "central", s, tag, keep), s))
        for w in COUNT_WIDTHS:
            only = {k for k in here if cwidth[k] == f"width {w:g}"}
            if only:
                Kg = Lazy(lambda w=w: L.gaussian_at(Zc, w))
                for h in HS:
                    models["central unit + eight counts"].add(
                        f"sigma {s:g}, width {w:g}, h {h:g}", lambda h=h, Kg=Kg: Kf() + h * Kg(),
                        y, blocks, only)
            only = {k for k in here if pwidth[k] == f"width {w:g}"}
            if only:
                Kp = Lazy(lambda w=w: L.gaussian_at(Zp, w))
                for h in HS:
                    models["central unit + PolyMetriX descriptors"].add(
                        f"sigma {s:g}, width {w:g}, h {h:g}", lambda h=h, Kp=Kp: Kf() + h * Kp(),
                        y, blocks, only)
        del Kf
        log(f"combined models at sigma {s:g}, {len(here)} blocks ({time.time() - t0:.0f} s)")
    for dm in TOPO_MAX:
        here = {k for k, w in enumerate(reach) if w == f"up to {dm} bonds"}
        if not here:
            continue
        Kt = Lazy(lambda dm=dm: topo_kernel(T, dm))
        for w in COUNT_WIDTHS:
            only = {k for k in here if cwidth[k] == f"width {w:g}"}
            if only:
                Kg = Lazy(lambda w=w: L.gaussian_at(Zc, w))
                for h in HS:
                    models["topological + eight counts"].add(
                        f"up to {dm} bonds, width {w:g}, h {h:g}",
                        lambda h=h, Kg=Kg, Kt=Kt: Kt() + h * Kg(), y, blocks, only)
    log(f"combined models done ({time.time() - t0:.0f} s)")

    res, preds = {}, {}
    for name, m in models.items():
        pred, chosen = m.select(n, blocks)
        r = {}
        for scheme in SCHEMES:
            p = pred[scheme]
            q2 = [float(r2_score(y, p[rep])) for rep in range(REPEATS)]
            mae = [float(np.mean(np.abs(y - p[rep]))) for rep in range(REPEATS)]
            r[scheme] = {"q2_mean": float(np.mean(q2)), "q2_by_repeat": q2,
                         "mae_mean": float(np.mean(mae)), "mae_by_repeat": mae,
                         "rmse_mean": float(np.mean([rmse(y, p[rep]) for rep in range(REPEATS)])),
                         "chosen": chosen[scheme]}
            preds[(name, scheme)] = p.mean(axis=0)
        res[name] = r
        log(f"  {name:42s} random q2 {r['random']['q2_mean']:+.3f} MAE "
            f"{r['random']['mae_mean']:5.1f} K | class q2 {r['class']['q2_mean']:+.3f} MAE "
            f"{r['class']['mae_mean']:5.1f} K")

    pairs = [MAIN,
             ("central unit + PolyMetriX descriptors", "PolyMetriX descriptors, gradient boosting"),
             ("central unit + eight counts", "PolyMetriX descriptors, gradient boosting"),
             ("invariant, central unit", "invariant, oligomer"),
             ("invariant, central unit", "topological, central unit"),
             ("invariant, central unit", "PolyMetriX descriptors, kernel"),
             ("invariant, central unit", "ECFP4 of the repeat unit"),
             ("central unit + eight counts", "eight counts"),
             ("central unit + eight counts", "invariant, central unit"),
             ("central unit + eight counts", "topological + eight counts"),
             ("central unit + PolyMetriX descriptors", "PolyMetriX descriptors, kernel"),
             ("PolyMetriX descriptors, kernel", "PolyMetriX descriptors, gradient boosting")]
    comps = []
    for a, b in pairs:
        for scheme in SCHEMES:
            pa, pb = preds[(a, scheme)], preds[(b, scheme)]
            g = (L.RS.paired(y, pa, pb) if scheme == "random"
                 else paired_groups(y, pa, pb, classes))
            comps.append({"model": a, "against": b, "scheme": scheme, **g})
            log(f"  {a} - {b} [{scheme}]: {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    by_class = {}
    for name in MAIN + ("central unit + PolyMetriX descriptors",):
        p = preds[(name, "class")]
        by_class[name] = {c: {"n": int(np.sum(classes == c)),
                              "mae": float(np.mean(np.abs(y - p)[classes == c]))}
                          for c in sorted(set(classes.tolist()))}
    payload = {"polymers": n, "refused_by_builder": feat.get("errors") or _errors(tag),
               "quick": quick, "tg": {"mean": float(y.mean()), "sd": float(y.std()),
                                      "min": float(y.min()), "max": float(y.max())},
               "atoms_median": float(np.median(feat["atoms"][keep])),
               "span_median": float(np.nanmedian(feat["span"][keep])),
               "models": res, "comparisons": comps, "class_mae": by_class,
               "published": PUBLISHED, "sigmas": list(SIGMAS), "topo_max": list(TOPO_MAX),
               "count_widths": list(COUNT_WIDTHS), "hs": list(HS), "lams": list(LAMS),
               "folds": FOLDS, "repeats": REPEATS, "units": UNITS, "conformers": CONFORMERS,
               "seconds": round(time.time() - t0, 1)}
    L.jdump(payload, os.path.join(RESULTS, "quick.json" if quick else "polymer_tg.json"))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, f"predictions_{tag}.npz"), ids=np.array([r["id"] for r in rows]),
             y=y, **{f"{m}|{s}": p for (m, s), p in preds.items()})
    if not quick:
        report(payload)
    return payload


def _errors(tag: str) -> Dict[str, str]:
    path = os.path.join(CACHE, f"errors_{tag}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "polymer_tg.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    t = payload["tg"]
    lines = [f"Glass transition temperature of {payload['polymers']} homopolymers (PolyMetriX, "
             f"Tg {t['mean']:.0f} +- {t['sd']:.0f} K). Each repeat unit is built as a trimer "
             f"capped with hydrogen. Five random folds and five folds of whole polymer classes, "
             f"each repeated {payload['repeats']} times; the setting and the ridge are chosen by "
             f"the evidence on the training block. MAE is the mean absolute error.", "",
             "| Model | q², random folds | MAE, random folds (K) | q², class folds | "
             "MAE, class folds (K) |", "|---|---:|---:|---:|---:|"]
    for name in MODELS:
        r = payload["models"][name]
        lines.append(f"| {name} | {r['random']['q2_mean']:+.3f} | {r['random']['mae_mean']:.1f} | "
                     f"{r['class']['q2_mean']:+.3f} | {r['class']['mae_mean']:.1f} |")
    lines += ["", "Published mean absolute errors (Fig. 6 of the PolyMetriX paper, read off the "
              "chart; gradient boosting; random 5-fold and 5 structural clusters, which are not "
              "the class folds above):", "",
              "| Model | MAE, random 5-fold (K) | MAE, 5 clusters (K) |", "|---|---:|---:|"]
    for name, v in payload["published"].items():
        lines.append(f"| {name} | {v['random 5-fold']:.1f} | {v['5 clusters']:.1f} |")
    lines += ["", "| Difference of q² | Folds | Mean [95 % interval] |", "|---|---|---|"]
    for c in payload["comparisons"]:
        lines.append(f"| {c['model']} minus {c['against']} | {c['scheme']} | "
                     f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] |")
    text = "\n".join(lines) + "\n"
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0,
                    help="use this many polymers and do not touch the cache")
    ap.add_argument("--jobs", type=int, default=16, help="processes for building the trimers")
    ap.add_argument("--build", action="store_true", help="build the trimers and features only")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.quick, a.jobs, a.build)
    return 0


if __name__ == "__main__":
    sys.exit(main())
