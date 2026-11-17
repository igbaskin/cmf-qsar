#!/usr/bin/env python
"""
Site-resolved models on particles (plan item NT7).

The data are in ``examples/particle_sites``: OH bound atop every surface site of 48 Pt and Au
particles of 38 to 147 atoms (eight shapes, each perfect and with 3 or 8 surface atoms removed),
computed with the UMA potential (oc20 head) by ``examples/particle_sites/make_particle_sites.py``,
the metal fixed, the O atom moving along the normal of its site only and the H atom freely. The
binding energy of a site is given relative to OH atop the (111) terrace of the same metal, in
eV, negative where the site binds more strongly. Sites with the same environment were computed
once; a sample here is one computed site.

Stage checks
------------
The binding energy against the generalized coordination number of the site, for each metal: the
slope, the intercept and the correlation of a linear regression. Calle-Vallejo and co-workers
found OH binding on Pt to weaken linearly with the generalized coordination number; this says
whether the computed set shows the same.

Stage model
-----------
A site profile (plan item M5) of every computed site, cutoff 6 A, channels every 0.5 A, with
kernel ridge regression; the ridge parameter is chosen in an inner 5-fold cross-validation whose
folds hold out whole particles. Compared on the same folds:

* the site profile with steric, electronegativity and one indicator field per metal;
* the same with the three core fields of item NT6 (coordination number, generalized coordination
  number, exposure);
* linear regressions, for each metal, on the generalized coordination number and on the
  coordination number of the site;
* a linear regression, for each metal, on the count reference of item M5 (the atoms in distance
  shells of 2, 3, 4.5 and 6 A around the site);
* a Gaussian kernel on the same counts with the two coordination numbers, for each metal, its
  width and ridge parameter chosen by an inner cross-validation.

Two splits: 5-fold cross-validation over the particles, repeated with 5 seeds, and folds that
hold out each of the eight shapes in both metals and all three variants. The inner folds that
choose the settings hold out what the outer folds hold out: particles in the first split,
shapes in the second, because the same shape in the other metal or another variant is a near
copy and leads the choice to too little regularisation for a new shape. For every held-out
particle the site each model predicts to bind most strongly is compared with the computed
energies: the excess of its computed binding over the strongest computed one (0 when the model
names the right site), and how often that excess is below 0.05 eV.

Stage references (added 17 November 2026, with tutorial 32)
------------------------------------------------------------
The site profile with the core fields led the Gaussian kernel on the counts. That reference
lacks two things the profile has: the exposure of the site, which is the third core field, and
the core values of the atoms around the site. The stage adds them as plain numbers:

* the counts and both coordination numbers (the reference of stage model, as a control);
* the same and the exposure of the site;
* the same and, for each distance shell of the counts, the sums of the coordination number, the
  generalized coordination number and the exposure over the atoms in the shell.

Each is a Gaussian kernel for each metal, with its width and ridge parameter chosen as in stage
model, on the same two splits; the two new ones on grids wider at both ends, because a first run
chose the edge of the stage's grids in a few blocks; the site profile with the core fields is fitted again beside them.
The difference of q2 carries a 95 % interval from a bootstrap over whole particles, on the
predictions averaged over the seeds.

Stage decomposition
-------------------
A whole-particle model: the target is the mean binding energy over all surface atoms of a
particle, the descriptor the periodic autocorrelation of the particle with the fields of the
model stage and the core fields. Fitted on all 48 particles, it is read site by site
(``cmfqsar.sites.site_contributions``, exact for a linear model), and the contributions are
compared with the computed binding energies of the same atoms: the share of the contributions
that falls on surface atoms, and the rank correlation over the surface atoms of every particle.

Usage
-----
    python benchmarks/run_particle_sites.py --stage checks          # seconds
    python benchmarks/run_particle_sites.py --stage model           # about 78 minutes
    python benchmarks/run_particle_sites.py --stage references      # about 70 minutes
    python benchmarks/run_particle_sites.py --stage decomposition   # about 2 minutes
    python benchmarks/run_particle_sites.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.model import CMFModel, _Learner  # noqa: E402
from cmfqsar.periodic import (core_columns, crystal_table,  # noqa: E402
                              inorganic_fieldset)
from cmfqsar.sites import shell_count_features, site_table  # noqa: E402

DATA = os.path.join(ROOT, "examples", "particle_sites")
RESULTS = os.path.join(HERE, "results", "particle_sites")
PHYSCHEM = ("steric", "prop:electronegativity")
METALS = ("Au", "Pt")
LAMBDAS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
#: the grids of the Gaussian kernel on the counts, wider than those of item NT6 because the
#: choices on this set reached both ends of those
COUNT_GAMMAS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
COUNT_LAMBDAS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
#: the grid of the whole-particle model, which chose the smallest value of a shorter grid
WHOLE_LAMBDAS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
SEEDS = (0, 1, 2, 3, 4)
#: the site profile: cutoff 6 A (SITE_SPEC of item M5), channels every 0.5 A
KERNEL = dict(kernel_source="periodic", ac_d_max=6.0, ac_sigma=0.5)
#: a site counts as found when its computed binding is within this of the strongest one
FOUND_EV = 0.05
#: an OH is bound atop its site when that atom is the nearest metal atom to the O, at no more
#: than BOUND_A and no less than CLASH_A (Pt-O and Au-O atop are about 2.0 to 2.1 A)
BOUND_A = 2.3
CLASH_A = 1.6


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #
def load(data: str = None) -> Dict:
    """The particles, every surface row, and the computed sites as samples."""
    from ase.io import read
    data = data or DATA
    particles = read(os.path.join(data, "particles.extxyz"), index=":")
    names = [p.info["name"] for p in particles]
    with open(os.path.join(data, "sites.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    done = {r["particle"] for r in rows}
    keep = [k for k, n in enumerate(names) if n in done]
    particles = [particles[k] for k in keep]
    names = [names[k] for k in keep]
    index = {n: k for k, n in enumerate(names)}
    per = [np.full(len(p), np.nan) for p in particles]
    every = [np.full(len(p), np.nan) for p in particles]
    unbound: Dict[int, int] = {}
    for r in rows:
        k, s, v = index[r["particle"]], int(r["site"]), float(r["binding_eV"])
        bound = (int(r["nearest"]) == int(r["computed_at"])
                 and CLASH_A <= float(r["o_metal_A"]) <= BOUND_A)
        if not bound:
            # the OH is not bound atop: the normal of an atom beside a vacancy points into the
            # hole, and the O collides with a neighbour or stays away from every atom
            if s == int(r["computed_at"]):
                cn = int(r["coordination"])
                unbound[cn] = unbound.get(cn, 0) + 1
            continue
        every[k][s] = v
        if s != int(r["computed_at"]):
            continue                         # an equivalent site: its value is a copy
        per[k][s] = v
    targets = AtomTargets.from_arrays(per, name="binding_eV")
    metal = np.array([particles[int(k)].info["metal"] for k in targets.molecule])
    shape = np.array([particles[int(k)].info["shape"] for k in targets.molecule])
    particle = np.array([names[int(k)] for k in targets.molecule])
    return {"particles": particles, "names": names, "targets": targets, "every": every,
            "metal": metal, "shape": shape, "particle": particle,
            "moved": int(sum(unbound.values())),
            "unbound_by_coordination": {str(c): n for c, n in sorted(unbound.items())},
            "surface_rows": len(rows)}


def site_columns(D: Dict) -> Dict[str, np.ndarray]:
    """The coordination number and the generalized coordination number of every sample."""
    t = D["targets"]
    cn, gcn = np.zeros(len(t)), np.zeros(len(t))
    cache = {}
    for r in range(len(t)):
        k, s = int(t.molecule[r]), int(t.atom[r])
        if k not in cache:
            cache[k] = core_columns(D["particles"][k], ["prop:coordination",
                                                        "prop:generalized_coordination"])
        cn[r] = cache[k]["prop:coordination"][s]
        gcn[r] = cache[k]["prop:generalized_coordination"][s]
    return {"cn": cn, "gcn": gcn}


# --------------------------------------------------------------------------- #
# Stage checks
# --------------------------------------------------------------------------- #
def stage_checks() -> None:
    D = load()
    C = site_columns(D)
    y = D["targets"].y
    out = {"particles": len(D["particles"]), "surface_rows": D["surface_rows"],
           "computed": int(len(y)), "moved": D["moved"],
           "unbound_by_coordination": D["unbound_by_coordination"], "metals": {}}
    for el in METALS:
        m = D["metal"] == el
        A = np.column_stack([np.ones(m.sum()), C["gcn"][m]])
        coef = np.linalg.lstsq(A, y[m], rcond=None)[0]
        r = float(np.corrcoef(C["gcn"][m], y[m])[0, 1])
        out["metals"][el] = {"sites": int(m.sum()), "slope": float(coef[1]),
                             "intercept": float(coef[0]), "r": r,
                             "range": [float(y[m].min()), float(y[m].max())],
                             "sd": float(y[m].std()),
                             "gcn_range": [float(C["gcn"][m].min()), float(C["gcn"][m].max())]}
    print(out, flush=True)
    jdump(out, os.path.join(RESULTS, "checks.json"))


# --------------------------------------------------------------------------- #
# Stage model
# --------------------------------------------------------------------------- #
class SiteKernels:
    """Kernel ridge regression on the model's own kernel of site records, with the site profiles
    computed once per field set."""

    def __init__(self, D: Dict, fieldsets: Dict):
        self.fs, self.tables, self.F, self.pos, self.seconds = {}, {}, {}, {}, {}
        self.chosen: Dict[str, List[float]] = {}
        for key, fs in fieldsets.items():
            t0 = time.perf_counter()
            crystals = list(crystal_table(D["particles"], fs, D["names"]).molecules)
            table = site_table(crystals, D["targets"])
            probe = CMFModel(fieldset=fs, method="krr", normalize=True, **KERNEL)
            self.F[key] = probe._ac_features(table)
            self.seconds[key] = time.perf_counter() - t0
            self.fs[key], self.tables[key] = fs, table
            self.pos[key] = {id(m): k for k, m in enumerate(table.molecules)}
            print(f"site profiles {key}: {self.F[key].shape}, {self.seconds[key]:.1f} s",
                  flush=True)

    def predict(self, key: str, y, tr, te, seed, groups) -> np.ndarray:
        from run_shape import _group_splits
        m = CMFModel(fieldset=self.fs[key], method="krr", normalize=True, **KERNEL)
        F, pos = self.F[key], self.pos[key]
        m._ac_features = lambda t: F[[pos[id(x)] for x in t.molecules]]
        table = self.tables[key]
        T_tr, T_te = table.subset(list(tr)), table.subset(list(te))
        K = m.training_kernel(T_tr)
        best, lam = np.inf, LAMBDAS[0]
        inner = _group_splits(np.asarray(groups)[tr], 5, seed)
        for value in LAMBDAS:
            p = np.full(len(tr), np.nan)
            for ite, itr in inner:
                learner = _Learner("krr", {"lam": value}).fit(K[np.ix_(itr, itr)], y[tr][itr])
                p[ite] = learner.predict(K[np.ix_(ite, itr)])
            err = float(np.mean((p - y[tr]) ** 2))
            if err < best:
                best, lam = err, value
        self.chosen.setdefault(key, []).append(lam)
        learner = _Learner("krr", {"lam": lam}).fit(K, y[tr])
        return learner.predict(m.test_kernel(T_te))


def _ranking(y, p, particle, te) -> List[Dict]:
    """For every particle of the test part: the excess of the computed binding of the site the
    model ranks strongest over the strongest computed binding."""
    out = []
    for name in sorted(set(particle[te])):
        idx = te[particle[te] == name]
        pick = idx[int(np.argmin(p[idx]))]
        out.append({"particle": str(name), "excess": float(y[pick] - y[idx].min()),
                    "sites": int(len(idx))})
    return out


def stage_model() -> None:
    warnings.filterwarnings("ignore")
    import run_core_fields
    from run_core_fields import _gaussian_counts
    from run_shape import _group_splits, _per_metal, _scores
    run_core_fields.CHOSEN.clear()
    D = load()
    t, y = D["targets"], D["targets"].y
    metal, shape, particle = D["metal"], D["shape"], D["particle"]
    C = site_columns(D)
    plain = inorganic_fieldset(list(METALS), physchem=PHYSCHEM, other=False)
    kern = SiteKernels(D, {"plain": plain,
                           "core": inorganic_fieldset(list(METALS), physchem=PHYSCHEM,
                                                      other=False, core=True)})
    crystals = list(crystal_table(D["particles"], plain, D["names"]).molecules)
    counts = shell_count_features(crystals, t, elements=list(METALS))
    counts = counts[:, counts.any(axis=0)]
    both = np.column_stack([C["cn"], C["gcn"], counts])
    # the inner folds hold out what the outer folds hold out: particles, or shapes
    inner = {"groups": particle}
    methods = {
        "site profile, steric, electronegativity and one indicator per metal":
            lambda tr, te, s: kern.predict("plain", y, tr, te, s, inner["groups"]),
        "the same and the three core fields of NT6":
            lambda tr, te, s: kern.predict("core", y, tr, te, s, inner["groups"]),
        "linear regression on the generalized coordination number, for each metal":
            lambda tr, te, s: _per_metal(metal, C["gcn"][:, None], y, tr, te),
        "linear regression on the coordination number, for each metal":
            lambda tr, te, s: _per_metal(metal, C["cn"][:, None], y, tr, te),
        "linear regression on the atoms in four distance shells, for each metal":
            lambda tr, te, s: _per_metal(metal, counts, y, tr, te),
        "Gaussian kernel on the shell counts and both coordination numbers, for each metal":
            lambda tr, te, s: _gaussian_counts(metal, both, y, tr, te, s, inner["groups"],
                                               COUNT_GAMMAS, COUNT_LAMBDAS)}
    shapes = sorted(set(shape))
    schemes = {"particles": lambda seed: _group_splits(particle, 5, seed),
               "shapes": lambda seed: [(np.flatnonzero(shape == g), np.flatnonzero(shape != g))
                                       for g in shapes]}
    out = {"samples": int(len(y)), "n_particles": int(t.n_measured_molecules),
           "moved": D["moved"], "seconds": kern.seconds, "sd": float(y.std()),
           "sd_by_metal": {el: float(y[metal == el].std()) for el in METALS}}
    for scheme, splits in schemes.items():
        res = {}
        seeds = SEEDS if scheme == "particles" else (0,)
        inner["groups"] = particle if scheme == "particles" else shape
        for mname, fn in methods.items():
            runs, ranks, per_metal = [], [], {el: [] for el in METALS}
            for seed in seeds:
                p = np.full(len(y), np.nan)
                for te, tr in splits(seed):
                    if not len(te):
                        continue
                    p[te] = fn(tr, te, seed)
                    if seed == 0:
                        ranks += _ranking(y, p, particle, te)
                runs.append(_scores(y, p, float(y.mean())))
                for el in METALS:
                    m = metal == el
                    per_metal[el].append(_scores(y[m], p[m], float(y[m].mean())))
            ex = np.array([r["excess"] for r in ranks])
            res[mname] = {
                "rmse": float(np.mean([r["rmse"] for r in runs])),
                "rmse_sd": float(np.std([r["rmse"] for r in runs])),
                "q2": float(np.mean([r["q2"] for r in runs])),
                "by_metal": {el: {"rmse": float(np.mean([r["rmse"] for r in v])),
                                  "q2": float(np.mean([r["q2"] for r in v]))}
                             for el, v in per_metal.items()},
                "ranking": {"particles": int(len(ex)), "mean_excess": float(ex.mean()),
                            "found": int(np.sum(ex < FOUND_EV)), "worst": float(ex.max())},
                "ranks": ranks}
            print(scheme, mname, {k: v for k, v in res[mname].items() if k != "ranks"},
                  flush=True)
        out[scheme] = res
    from collections import Counter
    out["chosen"] = {
        "site_profile_lambda": {k: {f"{v:g}": n for v, n in sorted(Counter(vals).items())}
                                for k, vals in kern.chosen.items()},
        "site_profile_grid": list(LAMBDAS),
        "count_kernel": {f"{g:g}/{l:g}": n for (g, l), n in sorted(Counter(
            (c["gamma"], c["lambda"]) for c in run_core_fields.CHOSEN).items())},
        "count_kernel_grid": {"gamma": list(COUNT_GAMMAS), "lambda": list(COUNT_LAMBDAS)}}
    print(out["chosen"], flush=True)
    jdump(out, os.path.join(RESULTS, "model.json"))


# --------------------------------------------------------------------------- #
# Stage references
# --------------------------------------------------------------------------- #
#: the core values of every atom that the references sum over a distance shell
CORE_NAMES = ("prop:coordination", "prop:generalized_coordination", "prop:exposure")
PROFILE = "site profile with the core fields"
REFERENCES = ("counts and both coordination numbers",
              "the same and the exposure of the site",
              "the same and the core values summed over each distance shell")
DRAWS = 2000
#: the grids of the two new references: those of stage model, extended at both ends, because a
#: first run chose the smallest width (0.0003) in 6 of 33 blocks and the largest ridge (1) in 2;
#: the control keeps the grids of stage model, so that it reproduces that stage
WIDE_GAMMAS = (0.00001, 0.00003, 0.0001) + COUNT_GAMMAS + (100.0,)
WIDE_LAMBDAS = (1e-11, 1e-10) + COUNT_LAMBDAS + (10.0, 100.0)


def site_numbers(D: Dict, counts: np.ndarray, shells) -> Dict[str, np.ndarray]:
    """The columns of the three references, one row per sample."""
    t = D["targets"]
    C = site_columns(D)
    core = [core_columns(p, CORE_NAMES) for p in D["particles"]]
    exposure = np.zeros(len(t))
    sums = np.zeros((len(t), len(CORE_NAMES) * len(shells)))
    for r in range(len(t)):
        k, s = int(t.molecule[r]), int(t.atom[r])
        pos = D["particles"][k].get_positions()
        d = np.linalg.norm(pos - pos[s], axis=1)
        exposure[r] = core[k]["prop:exposure"][s]
        low = 0.0
        for j, edge in enumerate(shells):
            m = (d > low + 1e-9) & (d <= edge)
            for f, name in enumerate(CORE_NAMES):
                sums[r, len(CORE_NAMES) * j + f] = core[k][name][m].sum()
            low = edge
    base = np.column_stack([C["cn"], C["gcn"], counts])
    return {REFERENCES[0]: base,
            REFERENCES[1]: np.column_stack([base, exposure]),
            REFERENCES[2]: np.column_stack([base, exposure, sums])}


def gaussian_per_metal(X, y, metal, tr, te, seed, groups, gammas=COUNT_GAMMAS,
                       lambdas=COUNT_LAMBDAS):
    """Kernel ridge regression with a Gaussian kernel on the standardised columns of X, zero
    between sites of different metals, around the training mean; width and ridge parameter from
    an inner 5-fold cross-validation over `groups`, on the grids `gammas` and `lambdas`. The same
    model and choice as
    run_core_fields._gaussian_counts, with one eigendecomposition per width and inner fold
    instead of one solve per ridge parameter. Returns the predictions and (width, ridge)."""
    from run_shape import _group_splits
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Z = (X - mu) / sd
    sq = (Z * Z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0) / Z.shape[1]
    same = (metal[:, None] == metal[None, :]).astype(float)
    inner = _group_splits(np.asarray(groups)[tr], 5, seed)
    best = None
    for gam in gammas:
        K = np.exp(-gam * d2) * same
        errs = np.zeros(len(lambdas))
        for ite, itr in inner:
            a, b = tr[itr], tr[ite]
            s, U = np.linalg.eigh(K[np.ix_(a, a)])
            r = U.T @ (y[a] - y[a].mean())
            Kb = K[np.ix_(b, a)] @ U
            for j, lam in enumerate(lambdas):
                errs[j] += float(np.sum((Kb @ (r / (s + lam)) + y[a].mean() - y[b]) ** 2))
        for j, lam in enumerate(lambdas):
            err = errs[j] / len(tr)
            if best is None or err < best[0]:
                best = (err, gam, lam)
    K = np.exp(-best[1] * d2) * same
    alpha = np.linalg.solve(K[np.ix_(tr, tr)] + best[2] * np.eye(len(tr)), y[tr] - y[tr].mean())
    return K[np.ix_(te, tr)] @ alpha + y[tr].mean(), (float(best[1]), float(best[2]))


def paired_over_particles(y, a, b, particle, draws: int = DRAWS, seed: int = 0) -> Dict:
    """q2 of `a` minus q2 of `b`, with a 95 % bootstrap interval that resamples whole particles."""
    def q2(i, p):
        return 1.0 - float(np.sum((p[i] - y[i]) ** 2)) / float(np.sum((y[i] - y[i].mean()) ** 2))
    groups = sorted(set(particle))
    idx = [np.flatnonzero(particle == g) for g in groups]
    rng = np.random.default_rng(seed)
    every = np.arange(len(y))
    diffs = []
    for _ in range(draws):
        i = np.concatenate([idx[k] for k in rng.integers(0, len(groups), len(groups))])
        diffs.append(q2(i, a) - q2(i, b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": q2(every, a) - q2(every, b), "lo": float(lo), "hi": float(hi)}


def stage_references() -> None:
    warnings.filterwarnings("ignore")
    from collections import Counter

    from run_shape import _group_splits, _scores
    from cmfqsar.sites import SHELLS
    D = load()
    t, y = D["targets"], D["targets"].y
    metal, shape, particle = D["metal"], D["shape"], D["particle"]
    plain = inorganic_fieldset(list(METALS), physchem=PHYSCHEM, other=False)
    crystals = list(crystal_table(D["particles"], plain, D["names"]).molecules)
    counts = shell_count_features(crystals, t, elements=list(METALS))
    counts = counts[:, counts.any(axis=0)]
    X = site_numbers(D, counts, SHELLS)
    kern = SiteKernels(D, {"core": inorganic_fieldset(list(METALS), physchem=PHYSCHEM,
                                                      other=False, core=True)})
    shapes = sorted(set(shape))
    schemes = {"particles": lambda seed: _group_splits(particle, 5, seed),
               "shapes": lambda seed: [(np.flatnonzero(shape == g), np.flatnonzero(shape != g))
                                       for g in shapes]}
    out = {"shells": [float(v) for v in SHELLS], "columns": {k: int(v.shape[1])
                                                             for k, v in X.items()},
           "draws": DRAWS, "grids": {"gamma": list(COUNT_GAMMAS), "lambda": list(COUNT_LAMBDAS)},
           "wide_grids": {"gamma": list(WIDE_GAMMAS), "lambda": list(WIDE_LAMBDAS)}}
    for scheme, splits in schemes.items():
        seeds = SEEDS if scheme == "particles" else (0,)
        groups = particle if scheme == "particles" else shape
        res, mean_pred = {}, {}
        for name in (PROFILE,) + REFERENCES:
            q2s, rmses, preds, chosen = [], [], [], []
            for seed in seeds:
                p = np.full(len(y), np.nan)
                for te, tr in splits(seed):
                    if not len(te):
                        continue
                    if name == PROFILE:
                        p[te] = kern.predict("core", y, tr, te, seed, groups)
                    else:
                        grids = ((COUNT_GAMMAS, COUNT_LAMBDAS) if name == REFERENCES[0]
                                 else (WIDE_GAMMAS, WIDE_LAMBDAS))
                        p[te], ch = gaussian_per_metal(X[name], y, metal, tr, te, seed, groups,
                                                       *grids)
                        chosen.append(ch)
                sc = _scores(y, p, float(y.mean()))
                q2s.append(sc["q2"])
                rmses.append(sc["rmse"])
                preds.append(p)
            mean_pred[name] = np.mean(preds, axis=0)
            res[name] = {"q2": float(np.mean(q2s)), "rmse": float(np.mean(rmses)),
                         "q2_by_seed": [float(v) for v in q2s],
                         "chosen": {f"{g:g}/{l:g}": n for (g, l), n in
                                    sorted(Counter(chosen).items())}}
            print(scheme, name, {k: v for k, v in res[name].items() if k != "chosen"},
                  flush=True)
        comps = []
        for name in REFERENCES:
            g = paired_over_particles(y, mean_pred[PROFILE], mean_pred[name], particle)
            comps.append({"model": PROFILE, "against": name, **g,
                          "by_seed": [a - b for a, b in zip(res[PROFILE]["q2_by_seed"],
                                                            res[name]["q2_by_seed"])]})
        out[scheme] = {"models": res, "comparisons": comps}
    jdump(out, os.path.join(RESULTS, "references.json"))


# --------------------------------------------------------------------------- #
# Stage decomposition
# --------------------------------------------------------------------------- #
def stage_decomposition() -> None:
    warnings.filterwarnings("ignore")
    from scipy.stats import spearmanr

    from cmfqsar.model import grid_search
    from cmfqsar.sites import site_contributions
    D = load()
    every = D["every"]
    names = D["names"]
    y = np.array([float(np.nanmean(v)) for v in every])
    fs = inorganic_fieldset(list(METALS), physchem=PHYSCHEM, other=False, core=True)
    table = crystal_table(D["particles"], fs, names)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_d_max=8.0, ac_sigma=0.5, pbc_taper=2.0)
    res = grid_search(table, y, model, [1.0], lams=WHOLE_LAMBDAS, folds=5, seed=0)
    model.fit(table, y, cv_folds=5, cv_seed=0)
    rows = []
    for k, crystal in enumerate(table.molecules):
        c = site_contributions(model, crystal)
        surf = np.isfinite(every[k])
        rho = spearmanr(c[surf], every[k][surf]).correlation if surf.sum() > 2 else np.nan
        cn = core_columns(D["particles"][k], ["prop:coordination"])["prop:coordination"]
        rho_cn = spearmanr(c[surf], cn[surf]).correlation if surf.sum() > 2 else np.nan
        rho_e = spearmanr(cn[surf], every[k][surf]).correlation if surf.sum() > 2 else np.nan
        rows.append({"particle": names[k], "atoms": int(len(c)), "surface": int(surf.sum()),
                     "surface_share": float(np.abs(c[surf]).sum() / max(np.abs(c).sum(),
                                                                         1e-300)),
                     "surface_atom_share": float(surf.mean()),
                     "spearman": float(rho) if np.isfinite(rho) else None,
                     "spearman_coordination": float(rho_cn) if np.isfinite(rho_cn) else None,
                     "spearman_coordination_binding": (float(rho_e) if np.isfinite(rho_e)
                                                       else None),
                     "sum_check": float(abs(c.sum() + model.bias
                                            - float(model.predict(table.subset([k]))[0])))})
    rhos = np.array([r["spearman"] for r in rows if r["spearman"] is not None])
    out = {"particles": len(rows), "target_range": [float(y.min()), float(y.max())],
           "target_sd": float(y.std()), "lam": res.best_params.get("lam"),
           "lam_grid": list(WHOLE_LAMBDAS),
           "q2": float(model.stats.q2), "rmse_cv": float(model.stats.rmse_cv),
           "median_spearman": float(np.median(rhos)), "positive_spearman": int(np.sum(rhos > 0)),
           "median_spearman_coordination": float(np.median(
               [r["spearman_coordination"] for r in rows
                if r["spearman_coordination"] is not None])),
           "median_spearman_coordination_binding": float(np.median(
               [r["spearman_coordination_binding"] for r in rows
                if r["spearman_coordination_binding"] is not None])),
           "median_surface_share": float(np.median([r["surface_share"] for r in rows])),
           "median_surface_atom_share": float(np.median([r["surface_atom_share"]
                                                         for r in rows])),
           "largest_sum_error": float(max(r["sum_check"] for r in rows)), "rows": rows}
    print({k: v for k, v in out.items() if k != "rows"}, flush=True)
    jdump(out, os.path.join(RESULTS, "decomposition.json"))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report() -> str:
    lines = ["# Site-resolved models on particles (plan item NT7)", ""]
    cp = os.path.join(RESULTS, "checks.json")
    if os.path.isfile(cp):
        C = jload(cp)
        lines += ["## The data", "",
                  f"{C['particles']} particles, {C['surface_rows']} surface atoms, "
                  f"{C['computed']} distinct sites where the OH is bound atop. "
                  f"{C['moved']} more distinct sites are left out because the OH is not bound "
                  f"atop them (its own atom is not the nearest to the O, or is not between "
                  f"{CLASH_A:g} and {BOUND_A:g} A away). By the number of neighbours of the site: "
                  + ", ".join(f"{n} with {c}" for c, n in
                              C["unbound_by_coordination"].items())
                  + ". These are mostly atoms beside a removed atom, whose normal points into "
                  "the hole.", "",
                  "| Metal | Sites | Binding energy, eV (lowest to highest) | Standard "
                  "deviation, eV | Slope on the generalized coordination number, eV | "
                  "Correlation |", "|---|---:|---:|---:|---:|---:|"]
        for el, m in C["metals"].items():
            lines.append(f"| {el} | {m['sites']} | {m['range'][0]:.2f} to {m['range'][1]:.2f} "
                         f"| {m['sd']:.3f} | {m['slope']:+.3f} | {m['r']:.3f} |")
        lines.append("")
    mp = os.path.join(RESULTS, "model.json")
    if os.path.isfile(mp):
        M = jload(mp)
        titles = {"particles": "Whole particles held out (5 folds, 5 seeds)",
                  "shapes": "Each of the eight shapes held out"}
        for scheme in ("particles", "shapes"):
            R = M[scheme]
            lines += [f"## {titles[scheme]}", "",
                      "| Model | RMSE, eV | q2 | q2 within Au | q2 within Pt | Strongest site "
                      "found | Mean excess of the chosen site, eV |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for m, r in R.items():
                rk = r["ranking"]
                lines.append(f"| {m} | {r['rmse']:.3f} | {r['q2']:.3f} | "
                             f"{r['by_metal']['Au']['q2']:.3f} | {r['by_metal']['Pt']['q2']:.3f}"
                             f" | {rk['found']} of {rk['particles']} | {rk['mean_excess']:.3f} |")
            lines.append("")
        lines += [f"A site is found when its computed binding is within {FOUND_EV:g} eV of the "
                  f"strongest site of its particle. The q2 within one metal is taken against the "
                  f"mean of that metal.", ""]
        ch = M.get("chosen")
        if ch:
            used = sorted({float(v) for d in ch["site_profile_lambda"].values() for v in d})
            lines += [f"Ridge parameters the inner cross-validation chose for the site profiles: "
                      f"{', '.join(f'{v:g}' for v in used)} (grid "
                      f"{', '.join(f'{v:g}' for v in ch['site_profile_grid'])}). Width and ridge "
                      f"parameter chosen for the Gaussian kernel on the counts: "
                      f"{', '.join(ch['count_kernel'])} (grids "
                      f"{', '.join(f'{v:g}' for v in ch['count_kernel_grid']['gamma'])} and "
                      f"{', '.join(f'{v:g}' for v in ch['count_kernel_grid']['lambda'])}).", ""]
    rp = os.path.join(RESULTS, "references.json")
    if os.path.isfile(rp):
        R = jload(rp)
        lines += ["## The site profile against numbers of the site", "",
                  "Gaussian kernels for each metal on numbers of the site, against the site "
                  "profile with the core fields, on the same splits. The difference of q2 is "
                  "taken on the predictions averaged over the seeds, with a 95 % interval from a "
                  f"bootstrap over whole particles ({R['draws']} draws).", "",
                  "| Model | Columns | q2, particles held out | q2, shapes held out | "
                  "Profile minus model, particles held out | Profile minus model, shapes held "
                  "out |", "|---|---:|---:|---:|---|---|"]
        for name in (PROFILE,) + REFERENCES:
            cells = []
            for scheme in ("particles", "shapes"):
                c = next((c for c in R[scheme]["comparisons"] if c["against"] == name), None)
                cells.append("" if c is None else
                             f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")
            cols = R["columns"].get(name, "")
            lines.append(f"| {name} | {cols} | {R['particles']['models'][name]['q2']:.3f} | "
                         f"{R['shapes']['models'][name]['q2']:.3f} | {cells[0]} | {cells[1]} |")
        lines += ["", "Width and ridge parameter chosen for each reference. The first uses the "
                  "grids of the model stage, "
                  f"{', '.join(f'{v:g}' for v in R['grids']['gamma'])} and "
                  f"{', '.join(f'{v:g}' for v in R['grids']['lambda'])}; the other two use "
                  f"{', '.join(f'{v:g}' for v in R['wide_grids']['gamma'])} and "
                  f"{', '.join(f'{v:g}' for v in R['wide_grids']['lambda'])}:", ""]
        for scheme in ("particles", "shapes"):
            for name in REFERENCES:
                ch = R[scheme]["models"][name]["chosen"]
                lines.append(f"* {scheme} held out, {name}: "
                             + ", ".join(f"{k} ({n})" for k, n in ch.items()))
        lines.append("")
    dp = os.path.join(RESULTS, "decomposition.json")
    if os.path.isfile(dp):
        X = jload(dp)
        lines += ["## A whole-particle model read site by site", "",
                  f"Target: the mean binding energy over the surface atoms of each of "
                  f"{X['particles']} particles ({X['target_range'][0]:.3f} to "
                  f"{X['target_range'][1]:.3f} eV, standard deviation {X['target_sd']:.3f}). "
                  f"Periodic autocorrelation with the core fields, q2 {X['q2']:.3f} in 5-fold "
                  f"cross-validation, ridge parameter {X['lam']:g} (grid "
                  f"{', '.join(f'{v:g}' for v in X.get('lam_grid', []))}).", "",
                  "| Quantity | Value |", "|---|---:|",
                  f"| Median share of the contributions on surface atoms | "
                  f"{X['median_surface_share']:.3f} |",
                  f"| Median share of the atoms that are surface atoms | "
                  f"{X['median_surface_atom_share']:.3f} |",
                  f"| Median rank correlation of contribution and binding over the surface "
                  f"atoms | {X['median_spearman']:+.3f} |",
                  f"| Particles where that correlation is positive | "
                  f"{X['positive_spearman']} of {X['particles']} |",
                  f"| Median rank correlation of contribution and coordination number over the "
                  f"surface atoms | {X['median_spearman_coordination']:+.3f} |",
                  f"| Median rank correlation of coordination number and binding over the "
                  f"surface atoms | {X['median_spearman_coordination_binding']:+.3f} |",
                  f"| Largest error of the sum of the contributions | "
                  f"{X['largest_sum_error']:.1e} |", ""]
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("checks", "model", "references", "decomposition"),
                    action="append",
                    help="stages to run (default: all)")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    if not args.report:
        for stage in args.stage or ("checks", "model", "references", "decomposition"):
            {"checks": stage_checks, "model": stage_model, "references": stage_references,
             "decomposition": stage_decomposition}[stage]()
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
