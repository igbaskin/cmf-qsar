#!/usr/bin/env python
"""
The co-field around a generic residue  (plan item PP5)

Plan items PP1 and PP2 put every measured position of 24 protein domains into one frame, the
frame of the residue (the alpha carbon at the origin, the nitrogen on the x axis, the carbonyl
carbon in the xy plane).  The field of the surroundings of a position, without its own side
chain, is then a sample like a molecule, and a kernel model of the change of folding stability
has a co-field: a map, in the frame of a generic residue, of where protein atoms of each field
type around the site make a substitution more or less destabilising.  PP2 found that this field
predicts on proteins the model has not seen (q2 +0.427 and +0.422, against +0.217 for the
substitution alone), so its co-field is a map of something and not of noise.

What is computed
----------------
The model of PP2's stage 1, "field of the surroundings": the cosine-normalised overlap kernel of
the five fields of the surroundings (envelope rho 0.1, attenuation 1), averaged over the five,
joined with a Gaussian kernel on the change of volume, hydropathy and charge.  The weight of the
field kernel and the ridge are chosen by cross-validation with whole domains held out; the model
is then fitted on draw 0 of PP2 (6,386 substitutions at 1,597 positions).  The co-field of field
f is

    C_f(r) = (w / 5) sum_j a_j X_jf(r) / |X_jf|,

where a_j are the dual coefficients, X_jf the field of the surroundings of sample j and |X_jf|
the square root of its self-overlap.  A positive value means that more field of that type there
raises ddG (makes a substitution less destabilising); ddG is the stability of the variant minus
that of the wild type, so a negative ddG is a destabilising substitution.

Three checks
------------
1. **Noise (plan item F2).**  The activities carry noise; propagated through the fit it gives
   every point of the co-field a standard deviation, with the out-of-fold error as the noise.  A
   point is kept where |C| exceeds twice it.
2. **Halves of the proteins.**  The 24 domains are split at random into two sets of twelve, a
   model is fitted on each, and the two co-fields are compared over the grid (Pearson r, and the
   share of kept points where they have the same sign); twenty splits.
3. **The frame alone (the null).**  The same, after the surroundings are given to another
   position of the same domain at random.  The substitutions keep their targets; only the link
   between a target and its surroundings is broken.  Whatever the frame puts in every context --
   the neighbouring backbone sits at nearly the same place around every residue -- is still there,
   so a map that came from the frame and not from the data would appear in the null as well.

What a site wants
-----------------
The same model fitted on classes of substitutions: those that add volume (more than 30 cubic
angstrom) and those that remove it; those that put a positive charge on a neutral residue and
those that put a negative one; those that add a neutral side-chain donor (S, T, Y, N, Q, W, H)
and those that remove one.  If the map is physics and not a picture of burial alone, the
electrostatic co-fields of the two charge classes have opposite signs (a negative environment
favours a positive charge), and the acceptor co-field of the donor classes does too.

Usage
-----
    python benchmarks/run_site_cofield.py            # about 30 minutes
    python benchmarks/run_site_cofield.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import run_mutants as M  # noqa: E402
import run_new_proteins as NP  # noqa: E402
import run_site_fields as SF  # noqa: E402
from cmfqsar import mutants as MU  # noqa: E402
from cmfqsar.cofields import _centres_and_widths, evaluate_gaussians  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet  # noqa: E402
from cmfqsar.kernels import self_kernels  # noqa: E402
from cmfqsar.model import krr_fit  # noqa: E402

RESULTS = os.path.join(ROOT, "benchmarks", "results", "site_cofield")
FIGURE = os.path.join(ROOT, "docs", "figures", "site_cofield.png")
DRAW = 0
HALF = 9.0                 # the grid covers -HALF..HALF angstrom on every axis of the frame
SPACING = 1.0
PLANE_SPACING = 0.25
SPLITS = 20
Z = 2.0
FIELD_NAMES = ("electrostatic", "steric", "hydrophobic", "acceptor", "donor")
DONORS = set("STYNQWH")
CLASSES = {
    "adds volume": lambda m, x: x[0] > 30.0,
    "removes volume": lambda m, x: x[0] < -30.0,
    "adds a positive charge": lambda m, x: m.new in "KR" and m.wild not in "KRDE",
    "adds a negative charge": lambda m, x: m.new in "DE" and m.wild not in "KRDE",
    "adds a donor": lambda m, x: m.new in DONORS and m.wild not in DONORS,
    "removes a donor": lambda m, x: m.wild in DONORS and m.new not in DONORS,
}
#: the pairs of classes, the field they are read in, and the sign physics expects for the
#: difference of their two co-fields (first minus second) in front of the residue, where the
#: new side chain sits.  A positive charge is worse than a negative one where the surroundings
#: are positive, so the electrostatic difference is negative there; a donor is worth adding
#: where the surroundings hold acceptors and is missed where it is removed from them, so the
#: acceptor difference is positive.  Volume has no stated sign: a crowded site punishes added
#: volume and a removed side chain leaves a cavity in it, both destabilising.
PAIRS = (("adds volume", "removes volume", "steric", None),
         ("adds a positive charge", "adds a negative charge", "electrostatic", -1),
         ("adds a donor", "removes a donor", "acceptor", 1))


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Geometry of the frame
# --------------------------------------------------------------------------- #
def frame_atoms() -> Dict[str, np.ndarray]:
    """The ideal backbone and the virtual beta carbon in the residue frame (angstrom)."""
    n = np.array([1.458, 0.0, 0.0])
    ang = np.radians(111.0)
    c = 1.525 * np.array([np.cos(ang), np.sin(ang), 0.0])
    u = SF.anchor_points()[0]
    u = u / np.linalg.norm(u)
    return {"CA": np.zeros(3), "N": n, "C": c, "CB": 1.53 * u}


def grid_points(half: float = HALF, spacing: float = SPACING) -> np.ndarray:
    ax = np.arange(-half, half + 1e-9, spacing)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
    return g.reshape(-1, 3)


def plane_basis() -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal directions of the plane through the alpha carbon that holds the virtual
    beta carbon and the nitrogen: e1 along the beta carbon, e2 towards the nitrogen."""
    at = frame_atoms()
    e1 = at["CB"] / np.linalg.norm(at["CB"])
    e2 = at["N"] - (at["N"] @ e1) * e1
    return e1, e2 / np.linalg.norm(e2)


def plane_points(half: float = HALF, spacing: float = PLANE_SPACING):
    e1, e2 = plane_basis()
    ax = np.arange(-half, half + 1e-9, spacing)
    s, t = np.meshgrid(ax, ax, indexing="ij")
    pts = s.reshape(-1, 1) * e1 + t.reshape(-1, 1) * e2
    return pts, ax


# --------------------------------------------------------------------------- #
# The basis: the normalised field of every position at every point
# --------------------------------------------------------------------------- #
def basis(contexts, keys, points: np.ndarray) -> np.ndarray:
    """(5, positions, points) float32: X_jf(r) / |X_jf| for every position j and field f."""
    fs = FieldSet.create(charge_model="file", alpha=M.CONTEXT_ALPHA)
    fs.set_alpha(float(M.CONTEXT_ALPHA))
    mols = [contexts[(float(M.SITE_RHO), d, p)] for d, p in keys]
    d = self_kernels(AtomTable.from_molecules(mols), fs.specs)
    X = np.zeros((len(fs.specs), len(mols), len(points)), np.float32)
    for j, m in enumerate(mols):
        for f, spec in enumerate(fs.specs):
            if d[f, j] <= 1e-300:
                continue
            c, w, wid = _centres_and_widths(m, spec, f)
            X[f, j] = evaluate_gaussians(points, c, w, wid) / np.sqrt(d[f, j])
    return X


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def position_index(s, keys) -> np.ndarray:
    idx = {k: i for i, k in enumerate(keys)}
    return np.array([idx[(d, int(p))] for d, p in zip(s.domain, s.position)], int)


def joint_kernel(s, Kp: np.ndarray, pos: np.ndarray, w: float) -> np.ndarray:
    X3 = MU.substitution_features(s.mutation)[0][:, :3]
    return M.mixed(M.gaussian_columns(X3), Kp[np.ix_(pos, pos)], w)


def cofields(a: np.ndarray, pos: np.ndarray, X: np.ndarray, w: float) -> np.ndarray:
    """(5, points): the co-field of every field type from the dual coefficients."""
    A = np.bincount(pos, weights=a, minlength=X.shape[1])
    return (w / X.shape[0]) * np.einsum("j,fjp->fp", A, X, optimize=True)


def dual_factor(K: np.ndarray, lam: float) -> np.ndarray:
    """G with Cov[a] = sigma^2 G G^T for the bordered system of `krr_fit` (with a bias)."""
    n = K.shape[0]
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = K + lam * np.eye(n)
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    rhs = np.vstack([np.eye(n), np.zeros((1, n))])
    return np.linalg.solve(A, rhs)[:n]


def cofield_sd(G: np.ndarray, pos: np.ndarray, X: np.ndarray, w: float, sigma: float,
               chunk: int = 1500) -> np.ndarray:
    """(5, points): the standard deviation of every co-field under noise sigma in the targets."""
    n_pos = X.shape[1]
    S = np.zeros((n_pos, len(pos)))
    S[pos, np.arange(len(pos))] = 1.0
    Gp = S @ G                                   # positions x samples
    out = np.zeros((X.shape[0], X.shape[2]))
    for f in range(X.shape[0]):
        for s0 in range(0, X.shape[2], chunk):
            P = X[f, :, s0:s0 + chunk].T.astype(float) @ Gp
            out[f, s0:s0 + chunk] = np.sqrt(np.einsum("ij,ij->i", P, P))
    return sigma * (w / X.shape[0]) * out


def choose(s, Kp, pos) -> Tuple[float, float, dict]:
    """The weight of the field kernel and the ridge, chosen with whole domains held out."""
    X3 = MU.substitution_features(s.mutation)[0][:, :3]
    Kc3 = M.gaussian_columns(X3)
    Kf = Kp[np.ix_(pos, pos)]
    fam = {w: M.mixed(Kc3, Kf, w) for w in M.MIXES}
    tg = s.targets("domain")
    res, picks = M.B.nested_choice(fam, tg, lams=M.LAMS, folds=NP.DOMAIN_FOLDS, seed=0,
                                   inner=M.INNER, name="field of the surroundings")
    keys = [(float(k), float(lam)) for k, lam in picks]
    w, lam = max(set(keys), key=keys.count)
    y = np.asarray(s.y, float)
    rmse = float(np.sqrt(np.mean((res.predicted - y) ** 2)))
    q2 = float(1.0 - np.sum((res.predicted - y) ** 2) / np.sum((y - y.mean()) ** 2))
    return w, lam, {"q2": round(q2, 4), "rmse": round(rmse, 4), "picks": [list(k) for k in keys]}


def fit_cofield(s, Kp, pos, X, w, lam, rows: Optional[np.ndarray] = None,
                perm: Optional[np.ndarray] = None) -> np.ndarray:
    """The co-field of a model fitted on the samples `rows` (all when None).  `perm` gives every
    position the surroundings of another (the null of check 3)."""
    rows = np.arange(len(s.y)) if rows is None else rows
    p = pos[rows] if perm is None else perm[pos[rows]]
    X3 = MU.substitution_features([s.mutation[i] for i in rows])[0][:, :3]
    # inside a class a column can be constant (every substitution adds the same charge), and a
    # constant column has no spread to standardise by
    X3 = X3[:, X3.std(axis=0) > 1e-12]
    K = M.mixed(M.gaussian_columns(X3), Kp[np.ix_(p, p)], w) if X3.shape[1] else \
        M.mixed(np.ones((len(rows), len(rows))), Kp[np.ix_(p, p)], w)
    a, _b = krr_fit(K, np.asarray(s.y, float)[rows], lam)
    return cofields(a, p, X, w)


def within_domain_permutation(keys, rng) -> np.ndarray:
    perm = np.arange(len(keys))
    doms = np.array([d for d, _p in keys])
    for d in sorted(set(doms)):
        idx = np.flatnonzero(doms == d)
        perm[idx] = rng.permutation(idx)
    return perm


def compare(c1: np.ndarray, c2: np.ndarray, cover: np.ndarray, keep: np.ndarray) -> Dict[str, list]:
    """Per field: Pearson r over the covered points, and the share of kept points with the same
    sign in both co-fields."""
    r, same = [], []
    for f in range(c1.shape[0]):
        m = cover[f]
        r.append(float(np.corrcoef(c1[f, m], c2[f, m])[0, 1]))
        k = keep[f]
        same.append(float(np.mean(np.sign(c1[f, k]) == np.sign(c2[f, k]))) if k.any()
                    else float("nan"))
    return {"r": r, "same_sign": same}


# --------------------------------------------------------------------------- #
# Reading the map
# --------------------------------------------------------------------------- #
def regions(points: np.ndarray, C: np.ndarray, keep: np.ndarray) -> Dict[str, dict]:
    """The kept co-field summed over four regions of the frame, per field: in front of the alpha
    carbon (along the beta carbon) or behind it, nearer than 5 angstrom or farther."""
    u = frame_atoms()["CB"]
    u = u / np.linalg.norm(u)
    along = points @ u
    dist = np.linalg.norm(points, axis=1)
    masks = {"in front, within 5 A": (along > 1.0) & (dist < 5.0),
             "in front, 5 to 9 A": (along > 1.0) & (dist >= 5.0) & (dist <= 9.0),
             "behind, within 5 A": (along < -1.0) & (dist < 5.0),
             "behind, 5 to 9 A": (along < -1.0) & (dist >= 5.0) & (dist <= 9.0)}
    out = {}
    for f, name in enumerate(FIELD_NAMES):
        out[name] = {}
        for rname, m in masks.items():
            k = m & keep[f]
            v = C[f, k]
            out[name][rname] = {"kept points": int(k.sum()),
                                "sum": round(float(v.sum()), 4),
                                "negative share": round(float(np.mean(v < 0)), 3) if len(v) else None}
    return out


def front_region(points: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """(5, points): the covered points in front of the alpha carbon (more than 1 angstrom along
    the beta carbon, within 9 angstrom), where a new side chain sits."""
    u = frame_atoms()["CB"]
    u = u / np.linalg.norm(u)
    m = (points @ u > 1.0) & (np.linalg.norm(points, axis=1) <= 9.0)
    return cover & m[None, :]


def front_means(C: np.ndarray, front: np.ndarray) -> np.ndarray:
    """The mean co-field of every field over its front region."""
    return np.array([float(C[f, front[f]].mean()) if front[f].any() else 0.0
                     for f in range(C.shape[0])])


def extremes(points: np.ndarray, C: np.ndarray, keep: np.ndarray) -> Dict[str, dict]:
    u = frame_atoms()["CB"]
    u = u / np.linalg.norm(u)
    out = {}
    for f, name in enumerate(FIELD_NAMES):
        v = np.where(keep[f], C[f], np.nan)
        if not np.isfinite(v).any():
            continue
        lo, hi = int(np.nanargmin(v)), int(np.nanargmax(v))
        out[name] = {k: {"value": round(float(C[f, i]), 5),
                         "distance from CA": round(float(np.linalg.norm(points[i])), 2),
                         "along the beta carbon": round(float(points[i] @ u), 2)}
                     for k, i in (("most negative", lo), ("most positive", hi))}
    return out


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(splits: int = SPLITS, results: str = RESULTS, figure: str = FIGURE) -> dict:
    os.makedirs(results, exist_ok=True)
    t0 = time.time()
    contexts = M.site_contexts(NP.DOMAINS)
    keys, Kp = NP.position_kernel(contexts)
    s = M.SiteSamples(contexts, NP.DOMAINS, NP.PER_POSITION, DRAW)
    pos = position_index(s, keys)
    y = np.asarray(s.y, float)
    log(f"{len(s)} substitutions at {len(keys)} positions after {time.time() - t0:.0f} s")
    pts = grid_points()
    ppts, pax = plane_points()
    allpts = np.vstack([pts, ppts])
    X = basis(contexts, keys, allpts)
    ng = len(pts)
    log(f"basis of {X.shape[1]} positions at {X.shape[2]} points after {time.time() - t0:.0f} s")
    # cover: points where the surroundings put field at all (the mean |field| over positions
    # above 1 % of its largest value), the only points where a co-field can be read
    mean_abs = np.abs(X).mean(axis=1)
    cover = mean_abs > 0.01 * mean_abs.max(axis=1, keepdims=True)
    w, lam, cv = choose(s, Kp, pos)
    log(f"chosen weight {w}, ridge {lam}; domains held out q2 {cv['q2']:+.3f}, "
        f"RMSE {cv['rmse']:.3f} after {time.time() - t0:.0f} s")
    K = joint_kernel(s, Kp, pos, w)
    a, b = krr_fit(K, y, lam)
    C = cofields(a, pos, X, w)
    G = dual_factor(K, lam)
    del K
    sd = cofield_sd(G, pos, X, w, cv["rmse"])
    del G
    keep = (np.abs(C) > Z * sd) & cover
    log(f"co-field and its error after {time.time() - t0:.0f} s")
    out = {"draw": DRAW, "n": len(s), "positions": len(keys), "weight": w, "ridge": lam,
           "cross_validation": cv, "grid": {"half": HALF, "spacing": SPACING, "points": ng},
           "z": Z, "fields": list(FIELD_NAMES),
           "covered_points": [int(cover[f, :ng].sum()) for f in range(len(FIELD_NAMES))],
           "kept_share": [round(float(keep[f, :ng].sum() / max(cover[f, :ng].sum(), 1)), 3)
                          for f in range(len(FIELD_NAMES))],
           "regions": regions(pts, C[:, :ng], keep[:, :ng]),
           "extremes": extremes(pts, C[:, :ng], keep[:, :ng])}
    # checks 2 and 3: halves of the proteins, with the data and with the frame alone
    rng = np.random.default_rng(0)
    doms = np.array(s.domain)
    names = sorted(set(s.domain))
    real, null = [], []
    for k in range(splits):
        order = rng.permutation(names)
        h1 = np.flatnonzero(np.isin(doms, order[:len(names) // 2]))
        h2 = np.flatnonzero(np.isin(doms, order[len(names) // 2:]))
        c1 = fit_cofield(s, Kp, pos, X, w, lam, h1)[:, :ng]
        c2 = fit_cofield(s, Kp, pos, X, w, lam, h2)[:, :ng]
        real.append(compare(c1, c2, cover[:, :ng], keep[:, :ng]))
        perm = within_domain_permutation(keys, rng)
        n1 = fit_cofield(s, Kp, pos, X, w, lam, h1, perm)[:, :ng]
        n2 = fit_cofield(s, Kp, pos, X, w, lam, h2, perm)[:, :ng]
        null.append(compare(n1, n2, cover[:, :ng], keep[:, :ng]))
        log(f"  split {k + 1}/{splits}: halves r {np.round(real[-1]['r'], 2)}, "
            f"null {np.round(null[-1]['r'], 2)}")
    out["halves"] = _summary(real)
    out["null"] = _summary(null)
    # what a site wants: the classes of substitutions
    feats = MU.substitution_features(s.mutation)[0]
    front = front_region(pts, cover[:, :ng])
    orders = [rng.permutation(names) for _k in range(10)]
    cls_maps, half_front, out["classes"] = {}, {}, {}
    for name, rule in CLASSES.items():
        rows = np.flatnonzero([bool(rule(m, x)) for m, x in zip(s.mutation, feats)])
        cls_maps[name] = fit_cofield(s, Kp, pos, X, w, lam, rows)
        hal, fr = [], []
        for order in orders:
            r1 = rows[np.isin(doms[rows], order[:len(names) // 2])]
            r2 = rows[np.isin(doms[rows], order[len(names) // 2:])]
            m1 = fit_cofield(s, Kp, pos, X, w, lam, r1)[:, :ng]
            m2 = fit_cofield(s, Kp, pos, X, w, lam, r2)[:, :ng]
            hal.append(compare(m1, m2, cover[:, :ng], keep[:, :ng]))
            fr += [front_means(m1, front), front_means(m2, front)]
        half_front[name] = np.array(fr)                  # (2 x splits, fields)
        out["classes"][name] = {"n": int(len(rows)), "positions": int(len(set(pos[rows]))),
                                "mean ddG": round(float(y[rows].mean()), 3),
                                "halves": _summary(hal)}
        log(f"  class {name}: {len(rows)} substitutions after {time.time() - t0:.0f} s")
    out["pairs"] = []
    for c1, c2, field, expect in PAIRS:
        f = FIELD_NAMES.index(field)
        m = cover[f, :ng]
        r = float(np.corrcoef(cls_maps[c1][f, :ng][m], cls_maps[c2][f, :ng][m])[0, 1])
        diff = float(front_means(cls_maps[c1][:, :ng], front)[f]
                     - front_means(cls_maps[c2][:, :ng], front)[f])
        halves = half_front[c1][:, f] - half_front[c2][:, f]
        out["pairs"].append({"first": c1, "second": c2, "field": field, "r": round(r, 3),
                             "front difference": round(diff, 6),
                             "halves with the same sign": round(float(np.mean(
                                 np.sign(halves) == np.sign(diff))), 3),
                             "expected sign": expect})
    figure_out(C[:, ng:], sd[:, ng:], cls_maps, pax, figure)
    np.savez_compressed(os.path.join(results, "site_cofield.npz"), points=pts,
                        cofield=C[:, :ng].astype(np.float32), sd=sd[:, :ng].astype(np.float32),
                        cover=cover[:, :ng])
    out["seconds"] = round(time.time() - t0, 1)
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(results, "site_cofield.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1)
    log(f"done after {out['seconds']:.0f} s")
    return out


def _summary(rows: List[dict]) -> dict:
    r = np.array([x["r"] for x in rows])
    same = np.array([x["same_sign"] for x in rows])
    return {"median r": [round(float(v), 3) for v in np.nanmedian(r, axis=0)],
            "lowest r": [round(float(v), 3) for v in np.nanmin(r, axis=0)],
            "median same sign": [round(float(v), 3) for v in np.nanmedian(same, axis=0)]}


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def figure_out(Cp: np.ndarray, sdp: np.ndarray, cls_maps: dict, ax_vals: np.ndarray,
               path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(ax_vals)
    ng = cls_maps[next(iter(cls_maps))].shape[1] - Cp.shape[1]
    e1, e2 = plane_basis()
    at = frame_atoms()
    panels = [("steric", Cp, sdp, "all substitutions, steric"),
              ("electrostatic", Cp, sdp, "all substitutions, electrostatic"),
              ("hydrophobic", Cp, sdp, "all substitutions, hydrophobic"),
              ("acceptor", Cp, sdp, "all substitutions, acceptor"),
              ("steric", cls_maps["adds volume"][:, ng:], None, "adds volume, steric"),
              ("steric", cls_maps["removes volume"][:, ng:], None, "removes volume, steric"),
              ("electrostatic", cls_maps["adds a positive charge"][:, ng:], None,
               "adds a positive charge, electrostatic"),
              ("electrostatic", cls_maps["adds a negative charge"][:, ng:], None,
               "adds a negative charge, electrostatic")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    ext = [ax_vals[0], ax_vals[-1], ax_vals[0], ax_vals[-1]]
    for ax, (field, Cm, sdm, title) in zip(axes.ravel(), panels):
        f = FIELD_NAMES.index(field)
        img = Cm[f].reshape(n, n).T                       # rows: e2, columns: e1
        lim = float(np.nanmax(np.abs(img))) or 1.0
        ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
        if sdm is not None:
            hidden = (np.abs(Cm[f]) <= Z * sdm[f]).reshape(n, n).T
            ax.imshow(np.where(hidden, 1.0, np.nan), origin="lower", extent=ext, cmap="Greys",
                      vmin=0.0, vmax=2.5, alpha=0.75)
        for name, xyz in at.items():
            ax.plot(xyz @ e1, xyz @ e2, "ko", ms=4)
            ax.annotate("CB*" if name == "CB" else name, (xyz @ e1, xyz @ e2),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("along the beta carbon, A")
        ax.set_ylabel("towards the nitrogen, A")
    fig.suptitle("Co-field of the surroundings in the frame of a generic residue: red raises "
                 "ddG (less destabilising), blue lowers it; grey hides what the noise could "
                 "produce (top row)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report(results: str = RESULTS) -> str:
    path = os.path.join(results, "site_cofield.json")
    if not os.path.exists(path):
        return "# PP5: no result yet\n"
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    F = d["fields"]
    L = ["# The co-field around a generic residue (plan item PP5)\n",
         f"`python benchmarks/run_site_cofield.py`, written {d.get('written', '?')}, "
         f"{d.get('seconds', 0) / 60:.0f} minutes.\n",
         f"The model of the field of the surroundings joined with the change of volume, "
         f"hydropathy and charge, fitted on {d['n']} substitutions at {d['positions']} positions "
         f"of 24 domains (draw {d['draw']} of plan item PP2). Field weight {d['weight']} and ridge "
         f"{d['ridge']:g}, chosen with whole domains held out (q2 "
         f"{d['cross_validation']['q2']:+.3f}, RMSE {d['cross_validation']['rmse']:.3f} kcal/mol). "
         f"Grid of {d['grid']['points']} points, -{d['grid']['half']:g} to {d['grid']['half']:g} A "
         f"on every axis of the residue frame, {d['grid']['spacing']:g} A apart. A point is "
         f"covered where the surroundings put field at all, and kept where the co-field exceeds "
         f"{d['z']:g} times its standard deviation under the noise of the targets.\n",
         "## The map and its checks\n",
         "| Field | Covered points | Kept share | Halves: median r | Halves: lowest r | "
         "Halves: same sign where kept | Frame alone: median r | Frame alone: same sign |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for f, name in enumerate(F):
        h, n = d["halves"], d["null"]
        L.append(f"| {name} | {d['covered_points'][f]} | {d['kept_share'][f]:.3f} | "
                 f"{h['median r'][f]:+.3f} | {h['lowest r'][f]:+.3f} | "
                 f"{h['median same sign'][f]:.3f} | {n['median r'][f]:+.3f} | "
                 f"{n['median same sign'][f]:.3f} |")
    L.append("")
    L.append("Halves: two models fitted on two disjoint sets of twelve domains, twenty splits. "
             "Frame alone: the same after every position is given the surroundings of another "
             "position of its domain.\n")
    L.append("## Where the kept co-field lies\n")
    L.append("The kept co-field summed in four regions of the frame (in front of the alpha "
             "carbon means along the beta carbon), and the share of kept points that are "
             "negative (where more field makes a substitution more destabilising):\n")
    regs = list(next(iter(d["regions"].values())))
    L.append("| Field | " + " | ".join(regs) + " |")
    L.append("|---|" + "---|" * len(regs))
    for name, rr in d["regions"].items():
        cells = []
        for r in regs:
            v = rr[r]
            cells.append(f"{v['sum']:+.3f} ({v['negative share']:.2f} negative, "
                         f"{v['kept points']} points)" if v["kept points"] else "nothing kept")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## What a site wants: classes of substitutions\n")
    L.append("| Class | Substitutions | Positions | Mean ddG, kcal/mol | " +
             " | ".join(f"Halves r, {n}" for n in F) + " |")
    L.append("|---|---:|---:|---:|" + "---:|" * len(F))
    for name, c in d["classes"].items():
        L.append(f"| {name} | {c['n']} | {c['positions']} | {c['mean ddG']:+.2f} | " +
                 " | ".join(f"{v:+.2f}" for v in c["halves"]["median r"]) + " |")
    L.append("")
    L.append("Pairs of classes: the correlation of their two co-fields over the covered points, "
             "and the first minus the second, averaged in front of the residue (more than 1 A "
             "along the beta carbon, within 9 A), where the new side chain sits. Halves: the "
             "share of the 20 half-models (ten splits of the domains) whose difference has the "
             "sign of the whole.\n")
    L.append("| First class | Second class | Field | r of the two co-fields | Difference in "
             "front | Halves with the same sign | Sign physics expects |")
    L.append("|---|---|---|---:|---:|---:|---|")
    for p in d["pairs"]:
        exp = {None: "none stated", -1: "negative", 1: "positive"}[p["expected sign"]]
        L.append(f"| {p['first']} | {p['second']} | {p['field']} | {p['r']:+.3f} | "
                 f"{p['front difference']:+.2e} | {p['halves with the same sign']:.2f} | {exp} |")
    L.append("")
    L.append("The figure is `docs/figures/site_cofield.png`.\n")
    return "\n".join(L)


def write_table(results: str = RESULTS) -> str:
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "table.md")
    text = report(results)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--splits", type=int, default=SPLITS,
                   help="how many random halvings of the domains")
    p.add_argument("--report", action="store_true", help="only rewrite the table")
    args = p.parse_args(argv)
    if not args.report:
        run(args.splits)
    print(write_table(RESULTS))


if __name__ == "__main__":
    main()
