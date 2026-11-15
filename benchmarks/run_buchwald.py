"""A reaction yield as the activity: the Buchwald-Hartwig screen with its components in frames.

The data are the 3,955 palladium-catalysed C-N couplings of Ahneman, Estrada, Lin, Dreher and
Doyle (Science 360 (2018) 186), built by ``examples/make_buchwald_examples.py``: 15 aryl
halides, 4 ligands, 3 bases and 22 isoxazole additives in every combination, and the yield of
each. The question is the one plan item PO1 asked of a selectivity, now of a yield: does a
component described by its fields in the frame of the bond it reacts or interferes with
predict better than the same component described without a frame, or by a fingerprint, or by a
few descriptors, or only by its name?

**The model.** One kernel over reactions, the ANOVA product of one kernel per component,

    K(r, s) = prod_c (1 + K_c(c_r, c_s)),

which holds a constant, the four main effects and every interaction among the components. The
ligand and the base enter by name (an identity kernel): with four ligands and three bases a
model can only tell them apart. The aryl halide and the additive enter in eight ways, which
are the methods of the comparison:

* one-hot, main effects: the sum of four identity kernels, the additive model of the yield;
* one-hot, all interactions: the ANOVA product of four identity kernels, which is what a model
  that knows the components only by name can do;
* fingerprints: the Tanimoto kernel of Morgan fingerprints (radius 2, 2048 bits);
* descriptors: a Gaussian kernel over a few RDKit descriptors of the whole molecule and the
  Gasteiger charges of the atoms of the reacting bond (C and X, or N and O);
* fields, no frame: the rotation-invariant autocorrelation of the five fields, its width chosen;
* fields in the frame: the overlap of the five fields, the aryl halide in the frame of its C-X
  bond and the additive in the frame of its N-O bond, the attenuation and the normalisation
  chosen;
* the same two field kernels with the Gaussian layer of `CMFModel(kernel_layer="gauss")`,
  exp(-gamma d^2) over the distance d between two fields, its width chosen as well.

The Gaussian layer is there for a reason the first run found. A field is a sum of atom
Gaussians, so in a shared frame the field of a molecule is close to the sum of the fields of its
parts. The 15 aryl halides are every combination of 5 arenes and 3 halogens, and their linear
frame kernel has rank 7 to within 1e-3: a linear model on it is a Free-Wilson model in three
dimensions and cannot give 2-pyridyl iodide an effect of its own. The rotation-invariant kernel
and the fingerprints are not linear in the parts and have no such limit, so the fair comparison
of frame and no frame is the one with the same layer on both.

Every choice of a setting and of the ridge is made inside the training part of every split, by
a three-fold cross-validation of the same kind as the split (random folds for the random
splits, whole additives for the additive splits, whole aryl halides for the aryl halide
splits), so no method sees the reactions it is scored on.

**The names of the parts (added 15 November 2026).** The 15 aryl halides are every combination of
5 arenes and 3 halogens, so a held-out aryl halide is not unknown: its arene and its halogen
are in the training part with other partners. The reference that says what a structure adds
over that is the aryl halide described by the names of its two parts, with an identity kernel
for the arene and one for the halogen. Two forms are given and both are reported: the two
added (a linear model in the parts) and with their product as well. The additive enters by
name. Two designs: each aryl halide out (15 folds, compared with the stored methods), and each
arene out (5 folds, all three halides of an arene together), where the arene is new and only a
structure can tell it from the others. ``--parts`` runs it; the folds of the new design are
chosen inside the training part by whole arenes, as the other designs choose by whole
additives or aryl halides.

**The splits.** The ten random 70/30 splits and the four additive splits published with the
data and used by every later paper; leave one additive out over all 22; leave one aryl halide
out over all 15. The score is R^2, the coefficient of determination on the test part, as the
published papers report it; the differences between two methods on the leave-one-out designs
carry a 95% interval from a bootstrap over the held-out components.

**The co-field.** The linear model with the frame is linear in the field of each component in its
main-effect term, so the main effect of the additive (and of the aryl halide) has a co-field
in the frame of the N-O (C-X) bond: the place where more of a field raises the yield. It is
read at named places of the ring and drawn in the ring plane.

Usage
-----
    python benchmarks/run_buchwald.py                  # every method that has no stored result, then the table
    python benchmarks/run_buchwald.py --methods frame  # one method again
    python benchmarks/run_buchwald.py --report         # the table and figures from the stored results
    python benchmarks/run_buchwald.py --parts          # the names of the parts and the arene-out design
    python benchmarks/run_buchwald.py --quick          # a few folds of every design, to a scratch directory
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.cofields import evaluate_gaussians  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, median_gamma_scale, r2_score, rmse  # noqa: E402

DATA = os.path.join(ROOT, "examples", "buchwald")
OUT_DIR = os.path.join(HERE, "results", "buchwald")

ROLES = ("ligand", "base", "aryl_halide", "additive")
BUILT = ("aryl_halide", "additive")
SDF = {"aryl_halide": "aryl_halides.sdf", "additive": "additives.sdf"}
PHYSCHEM = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
#: the grids. A first run with narrower grids chose their sharpest end for every method (the
#: attenuation 2, the width 0.15 A, the factor 0.25), so each was extended past the end it chose
#: until the best setting of a scan on three splits lay inside it.
ALPHAS = [0.8, 2.0, 5.0]
GAUSS_ALPHAS = [0.8, 2.0, 5.0, 8.0]
GAMMAS = [0.5, 1.0, 2.0, 4.0, 8.0]
AC_WIDTHS = [0.07, 0.1, 0.15, 0.25, 0.35, 0.75]
GAUSS_WIDTHS = [0.1, 0.35]
AC_D_MAX = 16.0
RBF_FACTORS = [0.06, 0.125, 0.25, 0.5, 1.0]
LAMS = [float(x) for x in np.logspace(-5, 2, 15)]
INNER_FOLDS = 3
RUNS = 1000
SEED = 0

#: method key -> (label, kernel form, settings)
METHODS: Dict[str, Tuple[str, str, List[dict]]] = {
    "main": ("one-hot, main effects", "sum", [{}]),
    "onehot": ("one-hot, all interactions", "anova", [{}]),
    "fingerprint": ("Morgan fingerprints", "anova", [{}]),
    "descriptor": ("RDKit descriptors", "anova", [{"factor": f} for f in RBF_FACTORS]),
    "invariant": ("fields, no frame", "anova", [{"width": w} for w in AC_WIDTHS]),
    "frame": ("fields in the frame", "anova",
              [{"alpha": a, "cosine": c} for a in ALPHAS for c in (False, True)]),
    "invariant_gauss": ("fields, no frame, Gaussian layer", "anova",
                        [{"width": w, "gamma": g} for w in GAUSS_WIDTHS for g in GAMMAS]),
    "frame_gauss": ("fields in the frame, Gaussian layer", "anova",
                    [{"alpha": a, "gamma": g} for a in GAUSS_ALPHAS for g in GAMMAS]),
}
ORDER = ["main", "onehot", "fingerprint", "descriptor", "invariant", "frame", "invariant_gauss",
         "frame_gauss"]
#: the aryl halide by the names of its arene and its halogen (the additive by name), the same two
#: with their product, and a control: fingerprints of the aryl halide with the additive by name
PARTS = ["parts_sum", "parts_product"]
METHODS["parts_sum"] = ("aryl halide by the names of its parts, added", "anova", [{}])
METHODS["parts_product"] = ("aryl halide by the names of its parts, with their product", "anova", [{}])
METHODS["fp_aryl"] = ("control: fingerprints of the aryl halide, additive by name", "anova", [{}])
#: the method the others are compared with, and the methods it is compared with
TESTED = "frame_gauss"          # the method the plan item tests: the frame with a Gaussian layer
REFERENCES = ["main", "onehot", "fingerprint", "descriptor", "invariant", "frame", "invariant_gauss"]

#: R^2 published for the same splits (Schwaller et al., Mach. Learn.: Sci. Technol. 2 (2021)
#: 015016, Table 1): the random-forest model on DFT descriptors of Ahneman et al., the one-hot
#: random forest, the multiple fingerprint features of Sandfort et al., and Yield-BERT
PUBLISHED = {
    "DFT descriptors, random forest (Ahneman 2018)": [0.92, 0.80, 0.77, 0.64, 0.54],
    "one-hot, random forest": [0.89, 0.69, 0.67, 0.49, 0.49],
    "multiple fingerprint features (Sandfort 2020)": [0.927, 0.85, 0.71, 0.64, 0.18],
    "Yield-BERT (Schwaller 2021)": [0.951, 0.84, 0.84, 0.75, 0.49],
}


# ----------------------------------------------------------------------------- the data


class Reactions:
    def __init__(self, data_dir: str = DATA):
        with open(os.path.join(data_dir, "reactions.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        with open(os.path.join(data_dir, "components.csv"), encoding="utf-8") as fh:
            comps = list(csv.DictReader(fh))
        with open(os.path.join(data_dir, "splits.csv"), encoding="utf-8") as fh:
            splits = list(csv.DictReader(fh))
        self.ids = [r["reaction"] for r in rows]
        self.y = np.array([float(r["yield_percent"]) for r in rows])
        self.labels = {role: [c["label"] for c in comps if c["role"] == role] for role in ROLES}
        self.smiles = {role: {c["label"]: c["smiles"] for c in comps if c["role"] == role} for role in ROLES}
        self.index = {role: np.array([self.labels[role].index(r[role]) for r in rows]) for role in ROLES}
        # the arene of every aryl halide ("4-Et-phenyl" of "4-Et-phenyl-Br"): a grouping of the reactions
        self.arenes = sorted({l.rsplit("-", 1)[0] for l in self.labels["aryl_halide"]})
        self.index["arene"] = np.array([self.arenes.index(self.labels["aryl_halide"][i].rsplit("-", 1)[0])
                                        for i in self.index["aryl_halide"]])
        if [s["reaction"] for s in splits] != self.ids:
            raise ValueError("splits.csv and reactions.csv list different reactions")
        self.published = {k: np.array([s[k] == "test" for s in splits])
                          for k in splits[0] if k != "reaction"}

    @property
    def n(self) -> int:
        return len(self.y)


def designs(data: Reactions, quick: bool = False) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """design -> list of (fold name, boolean test mask)."""
    random = [(k, m) for k, m in data.published.items() if k.startswith("random")]
    tests = [(k, m) for k, m in data.published.items() if k.startswith("additive_test")]
    loao = [(a, data.index["additive"] == i) for i, a in enumerate(data.labels["additive"])]
    loaho = [(a, data.index["aryl_halide"] == i) for i, a in enumerate(data.labels["aryl_halide"])]
    if quick:
        random, tests, loao, loaho = random[:2], tests[:2], loao[:4], loaho[:3]
    return {"random": random, "additive tests": tests, "one additive out": loao,
            "one aryl halide out": loaho}


INNER_GROUP = {"random": None, "additive tests": "additive", "one additive out": "additive",
               "one aryl halide out": "aryl_halide", "one arene out": "arene"}
#: the designs scored on the pooled predictions of all folds
POOLED = ("one additive out", "one aryl halide out", "one arene out")


def inner_folds(data: Reactions, train: np.ndarray, group: Optional[str], seed: int) -> List[np.ndarray]:
    """Test positions (within `train`) of a three-fold split of the training part."""
    rng = np.random.default_rng(seed)
    if group is None:
        perm = rng.permutation(len(train))
        return [np.sort(perm[k::INNER_FOLDS]) for k in range(INNER_FOLDS)]
    g = data.index[group][train]
    levels = rng.permutation(np.unique(g))
    return [np.where(np.isin(g, levels[k::INNER_FOLDS]))[0] for k in range(INNER_FOLDS)]


# ----------------------------------------------------------------------------- component kernels


def arene_designs(data: Reactions) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """Each arene out: the three halides of an arene are held out together."""
    return {"one arene out": [(a, data.index["arene"] == i) for i, a in enumerate(data.arenes)]}


def parts_kernel(data: Reactions, method: str) -> np.ndarray:
    """The aryl halides by the names of their parts: an identity kernel over the arene plus one over
    the halogen (``parts_sum``), and with the product of the two as well (``parts_product``)."""
    lab = data.labels["aryl_halide"]
    arene = np.array([l.rsplit("-", 1)[0] for l in lab])
    halogen = np.array([l.rsplit("-", 1)[1] for l in lab])
    Ka = (arene[:, None] == arene[None, :]).astype(float)
    Kh = (halogen[:, None] == halogen[None, :]).astype(float)
    return Ka + Kh if method == "parts_sum" else Ka + Kh + Ka * Kh


def scaled(K: np.ndarray) -> np.ndarray:
    return K / float(np.mean(np.diag(K)))


def field_table(role: str, alpha: float, data_dir: str = DATA):
    ds = load_dataset(os.path.join(data_dir, SDF[role]), ensemble_by="name")
    fs = FieldSet.create(physchem=PHYSCHEM, alpha=alpha, conformers="ensemble")
    mf = [build_fields(m, fs, name=n) for m, n in zip(ds.mols, ds.names)]
    return fs, AtomTable.from_molecules(mf), list(ds.names)


def reorder(K: np.ndarray, names: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    idx = [list(names).index(l) for l in labels]
    return K[np.ix_(idx, idx)]


def frame_kernel(data: Reactions, role: str, alpha: float, cosine: bool) -> np.ndarray:
    fs, table, names = field_table(role, alpha)
    K = CMFModel(fieldset=fs, method="krr", kernel_source="fields", normalize=False).training_kernel(table)
    if cosine:
        d = np.sqrt(np.diag(K))
        K = K / np.outer(d, d)
    else:
        K = scaled(K)
    return reorder(K, names, data.labels[role])


def gauss_layer(K: np.ndarray, gamma: float) -> np.ndarray:
    """exp(-gamma d^2) over the distances in the space of the kernel `K`, gamma in units of the
    median squared distance (the Gaussian layer of `CMFModel(kernel_layer="gauss")`). The median
    is taken over every member of the role; it uses structures only, never a yield."""
    d = np.diag(K)
    d2 = np.maximum(d[:, None] + d[None, :] - 2.0 * K, 0.0)
    return np.exp(-gamma * median_gamma_scale(K) * d2)


def raw_frame_kernel(data: Reactions, role: str, alpha: float) -> np.ndarray:
    fs, table, names = field_table(role, alpha)
    K = CMFModel(fieldset=fs, method="krr", kernel_source="fields", normalize=False).training_kernel(table)
    return reorder(K, names, data.labels[role])


def invariant_kernel(data: Reactions, role: str, width: float) -> np.ndarray:
    fs, table, names = field_table(role, 1.0)
    model = CMFModel(fieldset=fs, method="krr", kernel_source="autocorrelation", normalize=True,
                     ac_sigma=float(width), ac_d_max=AC_D_MAX)
    return reorder(model.training_kernel(table), names, data.labels[role])


def fingerprint_kernel(data: Reactions, role: str) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(data.smiles[role][l])) for l in data.labels[role]]
    return np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])


def descriptor_table(data: Reactions, role: str) -> np.ndarray:
    """A few descriptors of the whole molecule and the Gasteiger charges of the reacting bond."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    from make_buchwald_examples import isoxazole_bond, reacting_bond
    rows = []
    for l in data.labels[role]:
        mol = Chem.MolFromSmiles(data.smiles[role][l])
        mh = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mh)
        q = [float(a.GetDoubleProp("_GasteigerCharge")) for a in mh.GetAtoms()]
        bond = reacting_bond(mol) if role == "aryl_halide" else isoxazole_bond(mol)[:2]
        rows.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                     rdMolDescriptors.CalcNumRotatableBonds(mol), mol.GetNumHeavyAtoms(),
                     rdMolDescriptors.CalcNumAromaticRings(mol), rdMolDescriptors.CalcFractionCSP3(mol),
                     q[bond[0]], q[bond[1]]])
    X = np.array(rows, float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def descriptor_kernel(data: Reactions, role: str, factor: float) -> np.ndarray:
    X = descriptor_table(data, role)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    med = float(np.median(np.sqrt(d2[np.triu_indices(len(X), 1)])))
    return np.exp(-d2 / (2.0 * (factor * med) ** 2))


def component_kernels(data: Reactions, method: str, setting: dict) -> Dict[str, np.ndarray]:
    out = {role: np.eye(len(data.labels[role])) for role in ROLES}
    for role in BUILT:
        if method in PARTS:
            if role == "aryl_halide":
                out[role] = parts_kernel(data, method)
        elif method == "fp_aryl":
            if role == "aryl_halide":
                out[role] = fingerprint_kernel(data, role)
        elif method == "fingerprint":
            out[role] = fingerprint_kernel(data, role)
        elif method == "descriptor":
            out[role] = descriptor_kernel(data, role, setting["factor"])
        elif method == "invariant":
            out[role] = invariant_kernel(data, role, setting["width"])
        elif method == "frame":
            out[role] = frame_kernel(data, role, setting["alpha"], setting["cosine"])
        elif method == "invariant_gauss":
            out[role] = gauss_layer(invariant_kernel(data, role, setting["width"]), setting["gamma"])
        elif method == "frame_gauss":
            out[role] = gauss_layer(raw_frame_kernel(data, role, setting["alpha"]), setting["gamma"])
    return out


def reaction_kernel(data: Reactions, comps: Dict[str, np.ndarray], form: str) -> np.ndarray:
    K = np.zeros((data.n, data.n)) if form == "sum" else np.ones((data.n, data.n))
    for role in ROLES:
        i = data.index[role]
        Kc = comps[role][np.ix_(i, i)]
        if form == "sum":
            K += Kc
        else:
            K *= 1.0 + Kc
    return K


# ----------------------------------------------------------------------------- ridge over a grid


def ridge_path(K_tr: np.ndarray, y_tr: np.ndarray, K_te: np.ndarray, lams: Sequence[float]) -> np.ndarray:
    """Predictions on the test rows for every ridge in `lams`, the training mean as the bias."""
    mu = float(np.mean(y_tr))
    s, U = np.linalg.eigh(K_tr)
    s = np.clip(s, 0.0, None)
    Uy = U.T @ (y_tr - mu)
    KU = K_te @ U
    return np.array([mu + KU @ (Uy / (s + lam)) for lam in lams])


# ----------------------------------------------------------------------------- one method


def run_method(data: Reactions, method: str, folds: Dict[str, List[Tuple[str, np.ndarray]]],
               log=print) -> dict:
    label, form, settings = METHODS[method]
    t0 = time.time()
    # per design and fold: inner MSE and outer predictions for every (setting, ridge)
    inner = {d: np.zeros((len(f), len(settings), len(LAMS))) for d, f in folds.items()}
    outer = {d: [np.zeros((len(settings), len(LAMS), int(m.sum()))) for _, m in f] for d, f in folds.items()}
    for si, setting in enumerate(settings):
        K = reaction_kernel(data, component_kernels(data, method, setting), form)
        for d, flist in folds.items():
            for fi, (name, mask) in enumerate(flist):
                train, test = np.where(~mask)[0], np.where(mask)[0]
                err = np.zeros(len(LAMS))
                for k, pos in enumerate(inner_folds(data, train, INNER_GROUP[d], SEED + fi)):
                    keep = np.setdiff1d(np.arange(len(train)), pos)
                    a, b = train[keep], train[pos]
                    p = ridge_path(K[np.ix_(a, a)], data.y[a], K[np.ix_(b, a)], LAMS)
                    err += ((p - data.y[b]) ** 2).sum(axis=1)
                inner[d][fi, si] = err / len(train)
                outer[d][fi][si] = ridge_path(K[np.ix_(train, train)], data.y[train],
                                              K[np.ix_(test, train)], LAMS)
        log(f"  {label}: setting {si + 1}/{len(settings)} {setting} done, {time.time() - t0:.0f} s")
    res = {"method": method, "label": label, "form": form, "designs": {}, "seconds": 0.0}
    for d, flist in folds.items():
        pooled = np.full(data.n, np.nan)
        entries = []
        for fi, (name, mask) in enumerate(flist):
            si, li = np.unravel_index(int(np.argmin(inner[d][fi])), inner[d][fi].shape)
            p = outer[d][fi][si, li]
            test = np.where(mask)[0]
            pooled[test] = p
            entries.append({"fold": name, "setting": settings[si], "lam": LAMS[li],
                            "r2": r2_score(data.y[test], p), "rmse": rmse(data.y[test], p),
                            "predicted": p.tolist(), "test": test.tolist()})
        res["designs"][d] = {"folds": entries}
        if d in POOLED:
            res["designs"][d]["pooled"] = pooled.tolist()
            ok = np.isfinite(pooled)          # every reaction in a full run; a few folds with --quick
            res["designs"][d]["r2"] = r2_score(data.y[ok], pooled[ok])
            res["designs"][d]["rmse"] = rmse(data.y[ok], pooled[ok])
        else:
            r = [e["r2"] for e in entries]
            res["designs"][d]["r2"] = float(np.mean(r))
            res["designs"][d]["r2_sd"] = float(np.std(r))
            res["designs"][d]["rmse"] = float(np.mean([e["rmse"] for e in entries]))
        log(f"  {label}, {d}: R2 {res['designs'][d]['r2']:+.3f}")
    res["seconds"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------- comparisons


def bootstrap_difference(data: Reactions, group: str, a: np.ndarray, b: np.ndarray,
                         runs: int = RUNS, seed: int = SEED) -> dict:
    """R^2(a) - R^2(b) on pooled out-of-fold predictions, with a bootstrap over the held-out
    components of `group`."""
    g = data.index[group]
    levels = np.unique(g)
    members = [np.where(g == l)[0] for l in levels]
    rng = np.random.default_rng(seed)
    diff = r2_score(data.y, a) - r2_score(data.y, b)
    boot = []
    for _ in range(runs):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(levels), len(levels))])
        boot.append(r2_score(data.y[idx], a[idx]) - r2_score(data.y[idx], b[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(diff), "low": float(lo), "high": float(hi)}


# ----------------------------------------------------------------------------- the co-field


def most_chosen(res: dict, design: str) -> dict:
    seen: Dict[str, int] = {}
    for e in res["designs"][design]["folds"]:
        k = json.dumps(e["setting"], sort_keys=True)
        seen[k] = seen.get(k, 0) + 1
    return json.loads(max(seen, key=seen.get))


def main_effect_cofield(data: Reactions, setting: dict, log=print) -> dict:
    """Fit the frame model on every reaction and read the main effect of each built component."""
    comps = component_kernels(data, "frame", setting)
    K = reaction_kernel(data, comps, "anova")
    # the ridge: the one chosen most often by the leave-one-additive-out design
    lam = setting.get("lam", 1.0)
    mu = float(np.mean(data.y))
    coef = np.linalg.solve(K + lam * np.eye(data.n), data.y - mu)
    out = {"setting": setting, "lam": lam, "roles": {}}
    for role in BUILT:
        fs, table, names = field_table(role, setting["alpha"])
        model = CMFModel(fieldset=fs, method="krr", kernel_source="fields", normalize=False)
        Kraw = model.training_kernel(table)
        order = [list(names).index(l) for l in data.labels[role]]
        w = np.array([coef[data.index[role] == i].sum() for i in range(len(data.labels[role]))])
        # main effect of component a: sum_b w_b K_c(a, b), K_c as the model used it
        effect = comps[role] @ w
        mean_yield = np.array([data.y[data.index[role] == i].mean() for i in range(len(w))])
        d = np.sqrt(np.diag(Kraw))[order]
        per = w / d if setting["cosine"] else w / float(np.mean(np.diag(Kraw)))
        dual = np.zeros(len(names))
        dual[order] = per
        model.coef = dual
        out["roles"][role] = {"model": model, "table": table, "names": names,
                              "effect": effect.tolist(), "mean_yield": mean_yield.tolist(),
                              "r_effect_mean": float(np.corrcoef(effect, mean_yield)[0, 1])}
        log(f"  main effect of the {role.replace('_', ' ')}: correlation with its mean yield "
            f"{out['roles'][role]['r_effect_mean']:+.3f}")
    return out


def cofield_values(model: CMFModel, f: int, pts: np.ndarray) -> np.ndarray:
    centres, coeffs, widths = model.cofield_coefficients(f)
    v = evaluate_gaussians(np.asarray(pts, float).reshape(-1, 3), centres, coeffs, widths)
    return v * model.fieldset.specs[f].h * model.scale[f]


def named_places(data_dir: str = DATA) -> Dict[str, Dict[str, np.ndarray]]:
    """Places of the two frames, collected over the identity images of every molecule."""
    from rdkit import Chem
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    from make_buchwald_examples import isoxazole_bond, reacting_bond
    places: Dict[str, Dict[str, List[np.ndarray]]] = {"additive": {}, "aryl_halide": {}}

    def add(role, name, p):
        places[role].setdefault(name, []).append(np.asarray(p, float))

    for m in Chem.SDMolSupplier(os.path.join(data_dir, SDF["additive"]), removeHs=False):
        if m.GetProp("image") != "identity":
            continue
        pos = m.GetConformer().GetPositions()
        o, n, ring = isoxazole_bond(m)
        c3 = [a.GetIdx() for a in m.GetAtomWithIdx(n).GetNeighbors() if a.GetIdx() in ring and a.GetIdx() != o][0]
        c5 = [a.GetIdx() for a in m.GetAtomWithIdx(o).GetNeighbors() if a.GetIdx() in ring and a.GetIdx() != n][0]
        c4 = [i for i in ring if i not in (o, n, c3, c5)][0]
        for name, i in (("ring O", o), ("ring N", n), ("C3", c3), ("C4", c4), ("C5", c5)):
            add("additive", name, pos[i])
        for name, c in (("first atom of a 3-substituent", c3), ("first atom of a 5-substituent", c5),
                        ("first atom of a 4-substituent", c4)):
            for a in m.GetAtomWithIdx(c).GetNeighbors():
                if a.GetIdx() not in ring and a.GetAtomicNum() > 1:
                    add("additive", name, pos[a.GetIdx()])
    for m in Chem.SDMolSupplier(os.path.join(data_dir, SDF["aryl_halide"]), removeHs=False):
        if m.GetProp("image") != "identity":
            continue
        pos = m.GetConformer().GetPositions()
        ipso, x = reacting_bond(m)
        ring = [r for r in m.GetRingInfo().AtomRings() if ipso in r][0]
        add("aryl_halide", "halogen", pos[x])
        add("aryl_halide", "ipso carbon", pos[ipso])
        for a in m.GetAtomWithIdx(ipso).GetNeighbors():
            if a.GetIdx() in ring:
                add("aryl_halide", "ortho atoms", pos[a.GetIdx()])
        para = [i for i in ring if i != ipso and len(Chem.GetShortestPath(m, ipso, i)) == 4][0]
        add("aryl_halide", "para atom", pos[para])
        for a in m.GetAtomWithIdx(para).GetNeighbors():
            if a.GetIdx() not in ring and a.GetAtomicNum() > 1:
                add("aryl_halide", "first atom of the para substituent", pos[a.GetIdx()])
    return {r: {k: np.array(v) for k, v in d.items()} for r, d in places.items()}


def plot_cofields(cof: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem
    show = ["electrostatic", "steric", "hydrophobic", "hba"]
    refs = {"additive": "A10", "aryl_halide": "4-CF3-phenyl-Br"}
    fig, axes = plt.subplots(2, len(show), figsize=(3.6 * len(show), 6.6), squeeze=False)
    xs = np.arange(-7.0, 7.0, 0.1)
    for i, role in enumerate(BUILT):
        r = cof["roles"][role]
        model = r["model"]
        skeleton = None
        for m in Chem.SDMolSupplier(os.path.join(DATA, SDF[role]), removeHs=True):
            if m.GetProp("_Name") == refs[role] and m.GetProp("image") == "identity":
                skeleton = m.GetConformer().GetPositions()
        X, Y = np.meshgrid(xs, xs, indexing="xy")
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        for j, fname in enumerate(show):
            ax = axes[i, j]
            f = model.fieldset.names.index(fname)
            v = cofield_values(model, f, pts).reshape(X.shape)
            vmax = float(np.abs(v).max()) or 1.0
            im = ax.imshow(v, origin="lower", extent=(xs[0], xs[-1], xs[0], xs[-1]), cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            if skeleton is not None:
                ax.scatter(skeleton[:, 0], skeleton[:, 1], s=8, color="black", alpha=0.7)
            ax.set_title(f"{role.replace('_', ' ')}: {fname}", fontsize=9)
            ax.set_xlabel("x / A", fontsize=8)
            ax.set_ylabel("y / A", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, shrink=0.75).ax.tick_params(labelsize=6)
    fig.suptitle("Main-effect co-fields in the ring plane (red: more of the field there raises the "
                 "yield). Top: additive, N-O bond on the x axis, N at +x, ring centre on +y "
                 "(skeleton A10). Bottom: aryl halide, ipso C at 0, halogen at +x "
                 "(skeleton 4-CF3-phenyl-Br).", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(data: Reactions, results: Dict[str, dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(m, d) for m in ("onehot", TESTED) for d in ("one additive out", "one aryl halide out")
              if m in results]
    if not panels:                       # a partial run (one method of several) has nothing to draw
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 3.8), squeeze=False)
    for ax, (m, d) in zip(axes[0], panels):
        p = np.array(results[m]["designs"][d]["pooled"])
        ok = np.isfinite(p)
        p = np.where(ok, p, np.nan)
        ax.scatter(data.y, p, s=3, alpha=0.3)
        ax.plot([0, 100], [0, 100], color="gray", lw=0.8)
        ax.set_title(f"{results[m]['label']}\n{d}: R2 {r2_score(data.y[ok], p[ok]):+.2f}", fontsize=8)
        ax.set_xlabel("measured yield / %", fontsize=8)
        ax.set_ylabel("predicted yield / %", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- the table


def free_wilson_check(data: Reactions) -> dict:
    """How far the linear frame kernel of each built role is from full rank, and for the aryl
    halides how much of it lies in the span of an arene part plus a halogen part.

    A field is a sum of atom Gaussians, so in a shared frame the field of 4-CF3-phenyl bromide is
    nearly the field of the 4-CF3-phenyl part plus that of the bromine. On a factorial set of 5
    arenes and 3 halogens the 15 fields then span about 5 + 3 - 1 = 7 dimensions, and the linear
    overlap kernel can express an arene effect plus a halogen effect and nothing else: a
    Free-Wilson model in three dimensions."""
    out = {}
    for role in BUILT:
        K = frame_kernel(data, role, 2.0, False)
        e = np.sort(np.linalg.eigvalsh(K))[::-1]
        out[role] = {"n": len(e), "eigenvalues_over_largest": (e / e[0]).tolist(),
                     "above_1e-3": int((e / e[0] > 1e-3).sum())}
    K = frame_kernel(data, "aryl_halide", 2.0, False)
    lab = data.labels["aryl_halide"]
    arene = [l.rsplit("-", 1)[0] for l in lab]
    hal = [l.rsplit("-", 1)[1] for l in lab]
    X = np.hstack([np.array([[a == u for u in sorted(set(arene))] for a in arene], float),
                   np.array([[h == u for u in sorted(set(hal))] for h in hal], float)])
    P = X @ np.linalg.pinv(X)
    out["aryl_halide"]["share_in_arene_plus_halogen"] = float(np.linalg.norm(P @ K @ P) ** 2 /
                                                              np.linalg.norm(K) ** 2)
    out["aryl_halide"]["rank_arene_plus_halogen"] = int(np.linalg.matrix_rank(X))
    return out


def report(data: Reactions, results: Dict[str, dict], out_dir: str, log=print) -> dict:
    methods = [m for m in ORDER if m in results]
    summary: dict = {"n": data.n, "methods": {}, "differences": {}, "published": PUBLISHED}
    for m in methods:
        r = results[m]
        entry = {"label": r["label"], "seconds": r["seconds"]}
        for d, v in r["designs"].items():
            entry[d] = {"r2": v["r2"], "rmse": v["rmse"]}
            if "r2_sd" in v:
                entry[d]["r2_sd"] = v["r2_sd"]
            if d == "additive tests":
                entry[d]["per_test"] = [e["r2"] for e in v["folds"]]
            entry[d]["settings"] = [e["setting"] for e in v["folds"]]
            entry[d]["lams"] = [e["lam"] for e in v["folds"]]
        summary["methods"][m] = entry
    pairs = [(TESTED, ref) for ref in REFERENCES] + [("frame", "invariant")] + [
        ("invariant", ref) for ref in ("onehot", "fingerprint", "descriptor")]
    for a_name, b_name in pairs:
        if a_name not in results or b_name not in results:
            continue
        for d, group in (("one additive out", "additive"), ("one aryl halide out", "aryl_halide")):
            a = np.array(results[a_name]["designs"][d]["pooled"])
            b = np.array(results[b_name]["designs"][d]["pooled"])
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            summary["differences"][f"{d}: {a_name} - {b_name}"] = bootstrap_difference(data, group, a, b)
        # paired differences over the random and the additive-test splits
        for d in ("random", "additive tests"):
            fa = np.array([e["r2"] for e in results[a_name]["designs"][d]["folds"]])
            fb = np.array([e["r2"] for e in results[b_name]["designs"][d]["folds"]])
            diff = fa - fb
            summary["differences"][f"{d}: {a_name} - {b_name}"] = {
                "diff": float(diff.mean()), "low": float(diff.min()), "high": float(diff.max()),
                "per_fold": diff.tolist()}
    summary["free_wilson"] = free_wilson_check(data)
    parts_path = os.path.join(out_dir, "parts.json")
    if os.path.exists(parts_path):
        with open(parts_path, encoding="utf-8") as fh:
            summary["parts"] = json.load(fh)
    # the co-field of the linear frame model (the Gaussian layer has none), at the setting the
    # leave-one-additive-out design chose most
    if "frame" in results and "one additive out" in results["frame"]["designs"]:
        setting = most_chosen(results["frame"], "one additive out")
        lams = [e["lam"] for e in results["frame"]["designs"]["one additive out"]["folds"]
                if e["setting"] == setting]
        setting = dict(setting, lam=float(np.median(lams)))
        cof = main_effect_cofield(data, setting, log=log)
        places = named_places()
        summary["cofield"] = {"setting": setting, "roles": {}}
        for role in BUILT:
            r = cof["roles"][role]
            model = r["model"]
            vals = {}
            for fname in ("electrostatic", "steric", "hydrophobic", "hba"):
                f = model.fieldset.names.index(fname)
                vals[fname] = {k: float(np.mean(cofield_values(model, f, p))) for k, p in places[role].items()}
            summary["cofield"]["roles"][role] = {
                "effect": dict(zip(data.labels[role], r["effect"])),
                "mean_yield": dict(zip(data.labels[role], r["mean_yield"])),
                "r_effect_mean": r["r_effect_mean"], "places": vals,
                "place_counts": {k: int(len(p)) for k, p in places[role].items()}}
        plot_cofields(cof, os.path.join(out_dir, "cofields.png"))
    plot_scatter(data, results, os.path.join(out_dir, "cv_scatter.png"))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    write_table(summary, data, os.path.join(out_dir, "table.md"))
    return summary


def _setting_text(s: dict) -> str:
    if not s:
        return "none"
    g = f", gamma {s['gamma']:g}" if "gamma" in s else ""
    if "alpha" in s:
        kind = "" if "gamma" in s else (" cosine" if s["cosine"] else " raw")
        return f"alpha {s['alpha']:g}{kind}{g}"
    if "width" in s:
        return f"width {s['width']:g} A{g}"
    return f"width factor {s['factor']:g}"


def write_table(summary: dict, data: Reactions, path: str) -> None:
    L = ["# A reaction yield as the activity: the Buchwald-Hartwig screen with its components in frames", "",
         f"{data.n} palladium-catalysed C-N couplings (Ahneman et al., Science 2018): 15 aryl halides, "
         "4 ligands, 3 bases, 22 isoxazole additives. One kernel over reactions, the product over the "
         "four components of (1 + the kernel of that component); ligand and base by name, the aryl "
         "halide and the additive in the six ways of the rows below. Every setting and ridge is chosen "
         "inside the training part by a three-fold cross-validation of the same kind as the split. "
         "Score: R², the coefficient of determination on the test part.", ""]
    ms = summary["methods"]
    L += ["## R² by method and split", "",
          "| method | random 70/30, mean of 10 (sd) | additive tests 1-4, mean | test 1 | test 2 | test 3 | test 4 | one additive out, pooled | one aryl halide out, pooled |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, e in ms.items():
        rnd = e.get("random", {})
        tst = e.get("additive tests", {})
        per = tst.get("per_test", [])
        cells = [f"{rnd['r2']:+.3f} ({rnd['r2_sd']:.3f})" if rnd else "",
                 f"{tst['r2']:+.3f}" if tst else ""] + [f"{x:+.3f}" for x in per] + [""] * (4 - len(per))
        for d in ("one additive out", "one aryl halide out"):
            cells.append(f"{e[d]['r2']:+.3f}" if d in e else "")
        L.append(f"| {e['label']} | " + " | ".join(cells) + " |")
    L += ["", "RMSE in yield percent:", "",
          "| method | random 70/30 | additive tests | one additive out | one aryl halide out |",
          "|---|---:|---:|---:|---:|"]
    for m, e in ms.items():
        L.append(f"| {e['label']} | " + " | ".join(
            f"{e[d]['rmse']:.1f}" if d in e else "" for d in
            ("random", "additive tests", "one additive out", "one aryl halide out")) + " |")
    L += ["", "## Published on the same splits", "",
          "R² from Table 1 of Schwaller et al., Mach. Learn.: Sci. Technol. 2 (2021) 015016.", "",
          "| method | random 70/30 | test 1 | test 2 | test 3 | test 4 | tests 1-4, mean |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in PUBLISHED.items():
        L.append(f"| {k} | " + " | ".join(f"{x:.3g}" for x in v) + f" | {np.mean(v[1:]):.2f} |")
    if summary["differences"]:
        L += ["", "## Pairs of methods", "",
              "Difference in R². For the two leave-one-out designs: on the pooled out-of-fold "
              "predictions, with a 95% interval from 1000 bootstrap samples of the held-out "
              "components. For the published splits: the mean of the paired differences, with "
              "the smallest and the largest.", "",
              "| design | method | minus | difference in R² | interval or range |", "|---|---|---|---:|---|"]
        for k, v in summary["differences"].items():
            d, pair = k.split(": ")
            a, b = pair.split(" - ")
            L.append(f"| {d} | {ms[a]['label']} | {ms[b]['label']} | {v['diff']:+.3f} | "
                     f"[{v['low']:+.3f}, {v['high']:+.3f}] |")
    if "free_wilson" in summary:
        fw = summary["free_wilson"]
        L += ["", "## Why the linear frame kernel cannot fit the aryl halides one by one", "",
              "A field is a sum of atom Gaussians, so in a shared frame the field of a molecule is "
              "close to the sum of the fields of its parts. Eigenvalues of the linear frame kernel "
              "(attenuation 2, raw), divided by the largest:", ""]
        for role in BUILT:
            e = fw[role]["eigenvalues_over_largest"]
            L.append(f"* {role.replace('_', ' ')} ({fw[role]['n']} molecules): "
                     f"{fw[role]['above_1e-3']} above 1e-3; " + ", ".join(f"{x:.1e}" for x in e))
        a = fw["aryl_halide"]
        L += ["", f"The 15 aryl halides are every combination of 5 arenes and 3 halogens; "
              f"{100 * a['share_in_arene_plus_halogen']:.4f} % of their kernel lies in the span of an "
              f"arene part plus a halogen part, which has rank {a['rank_arene_plus_halogen']}. A linear "
              f"model on this kernel is a Free-Wilson model in three dimensions: it can give each arene "
              f"and each halogen an effect and cannot give a pair of them one. The Gaussian layer is "
              f"not linear in the atom contributions, which is why it lifts the limit."]
    if "parts" in summary:
        L += parts_section(summary["parts"], ms)
    L += ["", "## What the selection chose", ""]
    for m, e in ms.items():
        for d in ("random", "additive tests", "one additive out", "one aryl halide out"):
            if d not in e:
                continue
            seen: Dict[str, int] = {}
            for s, lam in zip(e[d]["settings"], e[d]["lams"]):
                k = f"{_setting_text(s)}, ridge {lam:.3g}"
                seen[k] = seen.get(k, 0) + 1
            L.append(f"* {e['label']}, {d}: " + "; ".join(f"{k} ({v})" for k, v in
                                                          sorted(seen.items(), key=lambda kv: -kv[1])))
    if "cofield" in summary:
        c = summary["cofield"]
        L += ["", f"## The main-effect co-fields of the linear frame model ({_setting_text(c['setting'])}, "
              f"ridge {c['setting']['lam']:.3g}, fitted on every reaction)", "",
              "The value of the co-field averaged over the named places of every molecule of that "
              "role, in the frame of the reacting bond. Positive: more of that field at that place "
              "raises the yield. The values are in yield percent per unit of field for the raw "
              "kernel and in arbitrary units for the cosine kernel; read the signs and the ratios.", ""]
        for role in BUILT:
            r = c["roles"][role]
            places = list(r["place_counts"])
            L += [f"**{role.replace('_', ' ')}** (main effect against the mean yield of the component: "
                  f"r = {r['r_effect_mean']:+.3f})", "",
                  "| field | " + " | ".join(f"{p} ({r['place_counts'][p]})" for p in places) + " |",
                  "|---|" + "---:|" * len(places)]
            for fname, vals in r["places"].items():
                L.append(f"| {fname} | " + " | ".join(f"{vals[p]:+.3g}" for p in places) + " |")
            L.append("")
            L += ["| component | mean yield, % | main effect in the model |", "|---|---:|---:|"]
            for k in sorted(r["mean_yield"], key=lambda k: -r["mean_yield"][k]):
                L.append(f"| {k} | {r['mean_yield'][k]:.1f} | {r['effect'][k]:+.2f} |")
            L.append("")
        L += ["![co-fields](cofields.png)", ""]
    L += ["![cross-validation](cv_scatter.png)", ""]
    total = sum(e["seconds"] for e in ms.values())
    L.append(f"Run time of the stored methods {total / 60:.1f} min.")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- the names of the parts

PARTS_DIR = os.path.join(OUT_DIR, "parts")
NEW = "one arene out"
HALIDE = "one aryl halide out"


def per_group_r2(data: Reactions, group: str, pooled: np.ndarray) -> Dict[str, float]:
    """R^2 of the pooled predictions inside every held-out group."""
    out = {}
    names = data.arenes if group == "arene" else data.labels[group]
    for i, name in enumerate(names):
        m = data.index[group] == i
        out[name] = r2_score(data.y[m], pooled[m])
    return out


def parts_stage(data: Reactions, out_dir: str = PARTS_DIR, main_dir: str = OUT_DIR, log=print) -> dict:
    """Every method on the design with each arene out, and the names of the parts also on the design
    with each aryl halide out, compared with the stored methods there."""
    os.makedirs(out_dir, exist_ok=True)
    folds_new = arene_designs(data)
    folds_halide = {HALIDE: designs(data)[HALIDE]}
    methods = ORDER + PARTS + ["fp_aryl"]
    results: Dict[str, dict] = {}
    for m in methods:
        path = os.path.join(out_dir, f"method_{m}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                results[m] = json.load(fh)
            continue
        folds = {}
        if m != "fp_aryl":
            folds.update(folds_new)
        if m in PARTS or m == "fp_aryl":
            folds.update(folds_halide)
        log(f"== {METHODS[m][0]} ({time.strftime('%Y-%m-%d %H:%M')})")
        results[m] = run_method(data, m, folds, log=log)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results[m], fh)
    stored: Dict[str, dict] = {}
    for m in ORDER:
        with open(os.path.join(main_dir, f"method_{m}.json"), encoding="utf-8") as fh:
            stored[m] = json.load(fh)
    labels = {m: METHODS[m][0] for m in methods}
    summary: dict = {"labels": labels, "new": {}, "halide": {}, "differences": {}}
    for m in methods:
        if NEW in results[m]["designs"]:
            p = np.array(results[m]["designs"][NEW]["pooled"])
            summary["new"][m] = {"r2": results[m]["designs"][NEW]["r2"],
                                 "rmse": results[m]["designs"][NEW]["rmse"],
                                 "per_arene": per_group_r2(data, "arene", p),
                                 "settings": [e["setting"] for e in results[m]["designs"][NEW]["folds"]]}
    for m in PARTS + ["fp_aryl"]:
        p = np.array(results[m]["designs"][HALIDE]["pooled"])
        summary["halide"][m] = {"r2": results[m]["designs"][HALIDE]["r2"],
                                "rmse": results[m]["designs"][HALIDE]["rmse"]}
    for m in ORDER:
        p = np.array(stored[m]["designs"][HALIDE]["pooled"])
        summary["halide"][m] = {"r2": stored[m]["designs"][HALIDE]["r2"],
                                "rmse": stored[m]["designs"][HALIDE]["rmse"]}
    # differences: every method minus the names of the parts, bootstrap over the held-out groups
    for ref in PARTS:
        pr_new = np.array(results[ref]["designs"][NEW]["pooled"])
        pr_hal = np.array(results[ref]["designs"][HALIDE]["pooled"])
        for m in ORDER:
            a = np.array(results[m]["designs"][NEW]["pooled"])
            summary["differences"][f"{NEW}: {m} - {ref}"] = bootstrap_difference(data, "arene", a, pr_new)
            b = np.array(stored[m]["designs"][HALIDE]["pooled"])
            summary["differences"][f"{HALIDE}: {m} - {ref}"] = bootstrap_difference(data, "aryl_halide", b, pr_hal)
        for m in ("fp_aryl",):
            a = np.array(results[m]["designs"][HALIDE]["pooled"])
            summary["differences"][f"{HALIDE}: {m} - {ref}"] = bootstrap_difference(data, "aryl_halide", a, pr_hal)
    # the control: fingerprints with the additive by name against fingerprints for both
    a = np.array(results["fp_aryl"]["designs"][HALIDE]["pooled"])
    b = np.array(stored["fingerprint"]["designs"][HALIDE]["pooled"])
    summary["differences"][f"{HALIDE}: fp_aryl - fingerprint"] = {"diff": float(r2_score(data.y, a) - r2_score(data.y, b))}
    with open(os.path.join(os.path.dirname(out_dir), "parts.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def parts_section(p: dict, ms: dict) -> List[str]:
    """Lines of the table for the stage of the names of the parts."""
    lab = p["labels"]

    def name(m):
        return lab.get(m) or ms[m]["label"]

    L = ["", "## The names of the parts as a reference for a new aryl halide", "",
         "The 15 aryl halides are every combination of 5 arenes and 3 halogens, so a held-out aryl "
         "halide has its arene and its halogen in the training part with other partners. The reference "
         "describes each aryl halide by the names of those two parts and the additive by name; it reads "
         "no structure. Two forms, both reported: the two names added, and with their product as well.", "",
         "R² of the pooled out-of-fold predictions:", "",
         "| method | each aryl halide out | each arene out |", "|---|---:|---:|"]
    order = ORDER + PARTS + ["fp_aryl"]
    for m in order:
        h = p["halide"].get(m)
        n = p["new"].get(m)
        L.append(f"| {name(m)} | {h['r2']:+.3f} |" if h else f"| {name(m)} | |")
        L[-1] += f" {n['r2']:+.3f} |" if n else " |"
    L += ["", "R² inside each held-out arene (all three halides together, three halides of 22 additives "
          "and 12 ligand and base pairs):", "",
          "| method | " + " | ".join(next(iter(p["new"].values()))["per_arene"]) + " |",
          "|---|" + "---:|" * len(next(iter(p["new"].values()))["per_arene"])]
    for m in order:
        if m in p["new"]:
            L.append(f"| {name(m)} | " + " | ".join(f"{v:+.3f}" for v in p["new"][m]["per_arene"].values()) + " |")
    L += ["", "Difference in R² against the names of the parts, with a 95% interval from 1000 bootstrap "
          "samples of the held-out aryl halides (15) or arenes (5):", "",
          "| design | method | minus | difference in R² | interval |", "|---|---|---|---:|---|"]
    for k, v in p["differences"].items():
        if "low" not in v:
            continue
        d, pair = k.split(": ")
        a, b = pair.split(" - ")
        L.append(f"| {d} | {name(a)} | {name(b)} | {v['diff']:+.3f} | [{v['low']:+.3f}, {v['high']:+.3f}] |")
    c = p["differences"].get(f"{HALIDE}: fp_aryl - fingerprint")
    if c:
        L += ["", f"Control: fingerprints for the aryl halide with the additive by name minus fingerprints "
              f"for both: {c['diff']:+.3f}, so the additive by name does not change what the reference "
              "is compared with."]
    return L


# ----------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", nargs="*", default=None, choices=ORDER,
                    help="methods to run again (default: every method without a stored result)")
    ap.add_argument("--report", action="store_true", help="only the table and figures from stored results")
    ap.add_argument("--quick", action="store_true", help="a few folds of every design, to a scratch directory")
    ap.add_argument("--parts", action="store_true",
                    help="the names of the parts and the design with each arene out, then the table")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or (os.path.join(OUT_DIR, "quick") if args.quick else OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    log_fh = open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        log_fh.write(s + "\n")
        log_fh.flush()

    data = Reactions()
    if args.parts:
        parts_stage(data, log=log)
        args.report = True
    folds = designs(data, quick=args.quick)
    results: Dict[str, dict] = {}
    for m in ORDER:
        path = os.path.join(out_dir, f"method_{m}.json")
        stored = os.path.exists(path)
        if args.report or (stored and (args.methods is None or m not in args.methods)):
            if stored:
                with open(path, encoding="utf-8") as fh:
                    results[m] = json.load(fh)
            continue
        if args.methods is not None and m not in args.methods:
            continue
        log(f"== {METHODS[m][0]} ({time.strftime('%Y-%m-%d %H:%M')})")
        results[m] = run_method(data, m, folds, log=log)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results[m], fh)
    report(data, results, out_dir, log=log)
    log(f"table written to {os.path.join(out_dir, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
