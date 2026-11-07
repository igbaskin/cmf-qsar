"""Is the pair-interaction field where the pair is?  Second derivatives on a set with a known pair term.

``cmfqsar.nonlinearity`` takes second derivatives of a layered model's prediction with respect to
probe atoms: the pair-interaction field (one probe fixed on an atom, the other scanned over a
grid), the curvature field, and the interaction matrix of a molecule's atoms with its eigenmodes.
A map of "where the model reads two places together" can only be judged on a set where the
answer is known, and no measured set in the repository has one.  So this script makes one.

**The set.**  Para-disubstituted benzenes X-C6H4-Y from 14 substituents (105 molecules, X and Y
from H, CH3, CF3, F, Cl, Br, OH, OCH3, NH2, N(CH3)2, CN, NO2, COCH3, COOCH3), built with RDKit
ETKDG + MMFF and placed in the frame of the Hammett example: ring centroid at the origin, the X
ipso carbon on +x (so Y sits on -x), the ring in the xy plane.  Two activities from the
tabulated sigma_p of the substituents:

    additive     y = sigma(X) + sigma(Y)
    interacting  y = sigma(X) + sigma(Y) + 1.5 sigma(X) sigma(Y)

each with 0.02 units of noise.  The second has a pair term between the two sites and the first
has none, and that is the whole difference between them.

**The measurement.**  Linear kernel and Gaussian layer on each (five fields, alpha, lambda and
gamma by 10-fold cross-validation, unnormalised kernels as everywhere in these benchmarks).  For
the Gaussian layer, on every molecule with X and Y other than H and different (78):

1. the share of the atom interaction matrix (electrostatic field, off-diagonal, absolute) that
   sits between the X group and the Y group, against the share those pairs make of all pairs;
2. the share of the pair-interaction field (probe fixed on the first heavy atom of X) that lies
   within 2 A of the Y group, against the share of the grid volume that lies there;
3. whether the leading interaction mode has weight on both groups.

Those three say where the curvature of each model sits.  A Gaussian layer is curved whatever
the target, so they are not expected to separate the two activities by much, and they do not:
the signal of a pair term is the *size* of the second derivative, not its shape.  The decisive
measure therefore compares the fitted second derivative with the **designed** one.  Writing the
sensitivity of sigma to the charge of atom k as c_k, the interacting activity has the exact
cross derivative d2y / dq_k dq_l = 1.5 c_k c_l for k in X and l in Y, and none within a group;
the additive activity has none at all.  With c_k read off the additive model (its first
derivative per atom, which the module also gives), every cross pair of every molecule has a
designed value, and the fitted values are regressed on it through the origin: the slope should
be 1 for the interacting model and 0 for the additive one, with a correlation to match.  The
same is done for the pair field: within 2 A of the Y group its designed shape is 1.5 c_k w_k
times the additive model's first-derivative field there (w_k the charge of the fixed probe's
atom).  For one molecule, OCH3 / NO2, the pair field is also refitted five times on random
four-fifths of the set for its stability, and drawn.

Usage
-----
    python benchmarks/run_interactions.py
    python benchmarks/run_interactions.py --report
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from cmfqsar.cofields import Grid, make_grid  # noqa: E402
from cmfqsar.dataio import relative_path  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, MoleculeFields, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.nonlinearity import (atom_interactions, gradient_field, nonlinearity_indices,  # noqa: E402
                                  pair_interaction_field)
from make_hammett_examples import SUBSTITUENTS  # noqa: E402
from run_hammett import PHYSCHEM, draw_ring  # noqa: E402
from run_nonlinearity import XS, YS, fit, plane_grid, refit_like  # noqa: E402

OUT_DIR = os.path.join(HERE, "results", "interactions")
LABELS = ["H", "CH3", "CF3", "F", "Cl", "Br", "OH", "OCH3", "NH2", "N(CH3)2", "CN", "NO2",
          "COCH3", "COOCH3"]
KAPPA = 1.5
NOISE = 0.02
TARGETS = ["additive", "interacting"]
SHOWN = ("OCH3", "NO2")
NEAR = 2.0          # A: "near the Y group"


# ------------------------------------------------------------------------------------ the set

def _smiles(x: str, y: str) -> str:
    """X-C6H4-Y, para, with the two ipso carbons carrying atom maps 1 (X) and 2 (Y)."""
    cy = f"[c:2]({y})" if y else "[cH:2]"
    cx = f"[c:1]1{x}" if x else "[cH:1]1"
    return f"c1c{cy}cc{cx}"


def _group(mol, start: int, ring: set) -> List[int]:
    """The atoms of the substituent on ring atom `start`: everything reachable off the ring."""
    seen, todo = set(), [n.GetIdx() for n in mol.GetAtomWithIdx(start).GetNeighbors()
                         if n.GetIdx() not in ring]
    while todo:
        a = todo.pop()
        if a in seen:
            continue
        seen.add(a)
        todo += [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()
                 if n.GetIdx() not in ring and n.GetIdx() not in seen]
    return sorted(seen)


def _frame(coords: np.ndarray, ring: Sequence[int], ipso_x: int, x_group: Sequence[int]) -> np.ndarray:
    r = coords[list(ring)]
    centre = r.mean(axis=0)
    x = coords[ipso_x] - centre
    x /= np.linalg.norm(x)
    _, _, vt = np.linalg.svd(r - centre)
    z = vt[2] - x * np.dot(vt[2], x)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    out = (coords - centre) @ np.vstack([x, y, z]).T
    if len(x_group):
        c = out[list(x_group)].mean(axis=0)
        if c[1] < -1e-6:
            out[:, 1] *= -1
        if c[2] < -1e-6:
            out[:, 2] *= -1
    return out


def make_set(seed: int = 2026):
    """The 105 molecules, their groups and the two activities."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    frag = {s[0]: s[1] for s in SUBSTITUENTS}
    sigma = {s[0]: float(s[3]) for s in SUBSTITUENTS}
    rng = np.random.default_rng(seed)
    mols, info = [], []
    k = 0
    for i, x in enumerate(LABELS):
        for y in LABELS[i:]:
            m = Chem.AddHs(Chem.MolFromSmiles(_smiles(frag[x], frag[y])))
            ipso = {a.GetAtomMapNum(): a.GetIdx() for a in m.GetAtoms() if a.GetAtomMapNum()}
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            p = AllChem.ETKDGv3()
            p.randomSeed = seed + k
            k += 1
            if AllChem.EmbedMolecule(m, p) != 0:
                raise RuntimeError(f"could not embed {x}/{y}")
            AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
            Chem.SanitizeMol(m)
            ring = set(m.GetSubstructMatch(Chem.MolFromSmarts("c1ccccc1")))
            gx, gy = _group(m, ipso[1], ring), _group(m, ipso[2], ring)
            coords = _frame(m.GetConformer().GetPositions(), sorted(ring), ipso[1], gx)
            conf = m.GetConformer()
            for a, pos in enumerate(coords):
                conf.SetAtomPosition(a, pos.tolist())
            add = sigma[x] + sigma[y]
            inter = add + KAPPA * sigma[x] * sigma[y]
            m.SetProp("_Name", f"{x}/{y}")
            m.SetProp("x", x)
            m.SetProp("y", y)
            m.SetDoubleProp("additive", add + NOISE * rng.normal())
            m.SetDoubleProp("interacting", inter + NOISE * rng.normal())
            m.SetProp("x_atoms", ",".join(map(str, gx)))
            m.SetProp("y_atoms", ",".join(map(str, gy)))
            mols.append(m)
            info.append({"name": f"{x}/{y}", "x": x, "y": y, "ipso_x": ipso[1], "ipso_y": ipso[2],
                         "x_atoms": gx, "y_atoms": gy})
    return mols, info


def write_set(path: str, mols) -> None:
    from rdkit import Chem
    with Chem.SDWriter(path) as w:
        for m in mols:
            w.write(m)


# ------------------------------------------------------------------------------ the measures

def first_heavy(mol: MoleculeFields, gx: Sequence[int]) -> int:
    heavy = [a for a in gx if mol.symbols[a] != "H"]
    return heavy[0] if heavy else gx[0]


def eligible(info) -> List[int]:
    return [i for i, m in enumerate(info)
            if m["x"] != "H" and m["y"] != "H" and m["x"] != m["y"]]


def per_molecule(model: CMFModel, mf: List[MoleculeFields], info, which: Sequence[int], f: int,
                 spacing: float) -> List[dict]:
    """The atom interaction matrix and the pair field of one model on every chosen molecule."""
    out = []
    for i in which:
        mol, meta = mf[i], info[i]
        gx, gy = meta["x_atoms"], meta["y_atoms"]
        H, g = atom_interactions(model, mol, f)
        k = first_heavy(mol, gx)
        grid = make_grid(mol.coords, margin=2.5, spacing=spacing)
        w = float(mol.weights[k, f])
        vol = pair_interaction_field(model, mol, f, mol.coords[k], grid, weight=w)
        mask = _near_mask(grid, mol.coords, gy)
        G = gradient_field(model, mol, f, grid).ravel()[mask]
        out.append({"H": H, "g": g, "k": k, "w": w, "vol": vol.ravel(), "grid": grid,
                    "mask": mask, "G_near": G, "gx": gx, "gy": gy, "mol": mol})
    return out


def _near_mask(grid: Grid, coords: np.ndarray, group: Sequence[int]) -> np.ndarray:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(coords[list(group)]).query(grid.points(), k=1)
    return d <= NEAR


def stats(rows: List[dict], ref: Optional[List[dict]] = None) -> Dict[str, float]:
    """Cross-site share, near-Y share and the mode test, on the maps or on their difference."""
    cross, expect, near, volume, spans = [], [], [], [], []
    for n, r in enumerate(rows):
        H = r["H"] - (ref[n]["H"] if ref else 0.0)
        vol = r["vol"] - (ref[n]["vol"] if ref else 0.0)
        gx, gy = r["gx"], r["gy"]
        N = H.shape[0]
        A = np.abs(H - np.diag(np.diag(H)))
        tot = float(A.sum())
        cross.append(2.0 * float(A[np.ix_(gx, gy)].sum()) / tot if tot > 0 else np.nan)
        expect.append(2.0 * len(gx) * len(gy) / (N * (N - 1)))
        mask = _near_mask(r["grid"], r["mol"].coords, gy)
        v = np.abs(vol)
        tv = float(v.sum())
        near.append(float(v[mask].sum()) / tv if tv > 0 else np.nan)
        volume.append(float(mask.mean()))
        vals, vecs = np.linalg.eigh(0.5 * (H + H.T))
        lead = vecs[:, int(np.argmax(np.abs(vals)))]
        spans.append(min(float(np.sum(lead[gx] ** 2)), float(np.sum(lead[gy] ** 2))) >= 0.15)
    return {"cross_share": float(np.nanmean(cross)), "cross_expected": float(np.mean(expect)),
            "near_y": float(np.nanmean(near)), "near_y_volume": float(np.mean(volume)),
            "mode_spans_both": float(np.mean(spans))}


def designed(rows: List[dict], oracle: List[dict]) -> Dict[str, float]:
    """The fitted cross derivatives and pair fields against the designed ones.

    `oracle` holds the additive model's maps on the same molecules; its first derivatives per
    atom are the sensitivities c_k.  Designed: ``KAPPA c_k c_l`` for k in X and l in Y (zero
    within a group), and ``KAPPA c_k w_k G(r)`` for the pair field near Y, with G the additive
    model's first-derivative field and w_k the charge the fixed probe carries.  Slopes are least
    squares through the origin over the cross pairs (the near-Y points) of all molecules.
    """
    H_fit, H_des, h_fit, h_des, within, cross_des = [], [], [], [], [], []
    for r, o in zip(rows, oracle):
        gx, gy = r["gx"], r["gy"]
        c = o["g"]
        E = KAPPA * np.outer(c[gx], c[gy])
        H_fit.append(r["H"][np.ix_(gx, gy)].ravel())
        H_des.append(E.ravel())
        cross_des.append(np.sqrt(np.mean(E ** 2)))
        W = r["H"][np.ix_(gx, gx)]
        W = W[~np.eye(len(gx), dtype=bool)]
        within.append(np.sqrt(np.mean(W ** 2)) if W.size else np.nan)
        h_fit.append(r["vol"][r["mask"]])
        h_des.append(KAPPA * c[r["k"]] * r["w"] * o["G_near"])

    def slope_corr(a, b):
        a = np.concatenate(a)
        b = np.concatenate(b)
        s = float(a @ b / (b @ b)) if float(b @ b) > 0 else np.nan
        return s, float(np.corrcoef(a, b)[0, 1])

    s_h, r_h = slope_corr(H_fit, H_des)
    s_f, r_f = slope_corr(h_fit, h_des)
    return {"slope_atoms": s_h, "corr_atoms": r_h, "slope_field": s_f, "corr_field": r_f,
            "within_rms_over_designed": float(np.nanmean(within) / np.mean(cross_des))}


def stability(model: CMFModel, mf, y, mol, meta, f: int, grid: Grid, repeats: int = 5,
              seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(mf)
    k = first_heavy(mol, meta["x_atoms"])
    maps = []
    for _ in range(repeats):
        pick = np.sort(rng.choice(n, size=int(round(0.8 * n)), replace=False))
        m = refit_like(model, AtomTable.from_molecules([mf[i] for i in pick]), y[pick])
        maps.append(pair_interaction_field(m, mol, f, mol.coords[k], grid,
                                           weight=float(mol.weights[k, f])).ravel())
    c = np.corrcoef(np.stack(maps))
    off = ~np.eye(len(maps), dtype=bool)
    return float(np.mean(c[off]))


def plot(panels: Dict[str, np.ndarray], mol: MoleculeFields, probe_atom: int, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(panels)
    fig, axes = plt.subplots(1, len(names), figsize=(3.9 * len(names), 3.6), squeeze=False)
    ext = (XS[0], XS[-1], YS[0], YS[-1])
    heavy = [k for k in range(mol.n_atoms) if mol.symbols[k] != "H"]
    vmax = max(float(np.abs(v).max()) for v in panels.values()) or 1.0
    for ax, name in zip(axes[0], names):
        img = panels[name].T
        im = ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal")
        draw_ring(ax)
        ax.scatter(mol.coords[heavy, 0], mol.coords[heavy, 1], s=14, color="black", zorder=4)
        ax.scatter([mol.coords[probe_atom, 0]], [mol.coords[probe_atom, 1]], s=70,
                   facecolors="none", edgecolors="lime", linewidths=1.8, zorder=5)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("x / A", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("y / A", fontsize=8)
    fig.suptitle(f"{mol.name}: pair-interaction field of the electrostatic field in the ring "
                 f"plane; the fixed probe (green ring) is the first atom of X, Y sits on -x",
                 fontsize=9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def report(out_dir: str = OUT_DIR, path: Optional[str] = None) -> str:
    def rows(name):
        p = os.path.join(out_dir, name)
        return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []

    lines = ["# Is the pair-interaction field where the pair is", "",
             "`benchmarks/run_interactions.py`: 105 para-disubstituted benzenes X-C6H4-Y from 14 "
             "substituents, with an additive activity σ(X) + σ(Y) and an interacting one that adds "
             "1.5 σ(X) σ(Y); five fields, unnormalised kernels, α, λ and γ by 10-fold "
             "cross-validation. Stability: mean correlation of the OCH3/NO2 pair field over five "
             "refits on four-fifths of the set.", "",
             "| activity | n | q² linear | q² gauss | γ | index joint | stability |",
             "|---|---|---|---|---|---|---|"]
    for r in rows("fits.csv"):
        lines.append(f"| {r['target']} | {r['n']} | {float(r['q2_linear']):.2f} | "
                     f"{float(r['q2_gauss']):.2f} | {float(r['gamma']):g} | "
                     f"{float(r['index_joint']):.3f} | {float(r['stability']):.2f} |")
    lines += ["",
              "The measures below are taken on the 78 molecules with X and Y both other than H and "
              "different. Cross-site share: the part of the atom interaction matrix (electrostatic "
              "field, off-diagonal, absolute) between the X and the Y group, beside the share of all "
              "atom pairs those pairs are. Near Y: the part of the pair-interaction field (probe "
              "fixed on the first heavy atom of X) within 2 Å of the Y group, beside the share of "
              "the grid volume that lies there. Mode spans both: how often the leading eigenmode "
              "has at least 15 % of its weight on each group. `interacting − additive` is the "
              "difference between the two models' maps on the same molecule, which isolates the "
              "designed pair term from the curvature the Gaussian layer has on any target.", "",
              "| maps of | cross-site share | share of pairs | near Y | volume near Y | "
              "mode spans both |",
              "|---|---|---|---|---|---|"]
    for r in rows("measures.csv"):
        lines.append(f"| {r['maps']} | {float(r['cross_share']):.2f} | "
                     f"{float(r['cross_expected']):.2f} | {float(r['near_y']):.2f} | "
                     f"{float(r['near_y_volume']):.2f} | {100 * float(r['mode_spans_both']):.0f} % |")
    lines += ["",
              "The decisive comparison: the fitted second derivatives against the designed ones. "
              "Atoms: the cross block of the atom interaction matrix (k in X, l in Y) against "
              "1.5 c_k c_l, with c the additive model's first derivative per atom; the slope through "
              "the origin should be 1 for the interacting model and 0 for the additive one. Field: "
              "the pair field within 2 Å of the Y group against 1.5 c_k w_k times the additive "
              "model's first-derivative field. Within: the root mean square of the fitted "
              "within-X block, which should be zero, over that of the designed cross block.", "",
              "| model | slope, atoms | correlation, atoms | slope, field | correlation, field | "
              "within-group RMS / designed |",
              "|---|---|---|---|---|---|"]
    for r in rows("designed.csv"):
        lines.append(f"| {r['model']} | {float(r['slope_atoms']):.2f} | "
                     f"{float(r['corr_atoms']):.2f} | {float(r['slope_field']):.2f} | "
                     f"{float(r['corr_field']):.2f} | {float(r['within_rms_over_designed']):.2f} |")
    lines += ["", "The figure `pair_fields.png` cuts the ring plane for OCH3/NO2: the pair field "
              "of the additive and of the interacting model, and their difference, on one colour "
              "scale.", ""]
    text = "\n".join(lines)
    path = path or os.path.join(out_dir, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--spacing", type=float, default=0.6, help="grid of the near-Y share")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        report(a.out_dir)
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    mols, info = make_set()
    write_set(os.path.join(a.out_dir, "pairs.sdf"), mols)
    fs = FieldSet.create(physchem=PHYSCHEM)
    mf = [build_fields(m, fs, name=m.GetProp("_Name")) for m in mols]
    table = AtomTable.from_molecules(mf)
    f = fs.names.index("electrostatic")
    which = eligible(info)
    shown = next(i for i, m in enumerate(info) if {m["x"], m["y"]} == set(SHOWN))
    k_shown = first_heavy(mf[shown], info[shown]["x_atoms"])
    grid_plane = plane_grid(0.0)
    panels: Dict[str, np.ndarray] = {}
    fits, measures, maps = [], [], {}
    print(f"{len(mf)} molecules written to {relative_path(a.out_dir, ROOT)}/pairs.sdf")
    for target in TARGETS:
        y = np.array([m.GetDoubleProp(target) for m in mols])
        print(f"\n== {target}: sd(y) {np.std(y):.3f}")
        lin, rl = fit(table, y, "linear", a.folds)
        gau, rg = fit(table, y, "gauss", a.folds)
        print(f"  linear  alpha {rl.best_alpha:g} lambda {rl.best_params['lam']:g} q2 {rl.best_q2:.3f}")
        print(f"  gauss   alpha {rg.best_alpha:g} lambda {rg.best_params['lam']:g} "
              f"gamma {rg.best_params['gamma']:g} q2 {rg.best_q2:.3f}")
        idx = {i.field: i for i in nonlinearity_indices(gau)}
        maps[target] = per_molecule(gau, mf, info, which, f, a.spacing)
        st = stats(maps[target])
        print("  raw maps: " + ", ".join(f"{k} {v:.3f}" for k, v in st.items()))
        measures.append({"maps": f"{target} model", **{k: round(v, 4) for k, v in st.items()}})
        stab = stability(gau, mf, y, mf[shown], info[shown], f, grid_plane, repeats=a.repeats)
        print(f"  pair field of {mf[shown].name}: stability r = {stab:.2f}")
        panels[f"{target} model"] = pair_interaction_field(
            gau, mf[shown], f, mf[shown].coords[k_shown], grid_plane,
            weight=float(mf[shown].weights[k_shown, f]))[:, :, 0]
        fits.append({"target": target, "n": len(y), "q2_linear": round(rl.best_q2, 4),
                     "alpha_linear": rl.best_alpha, "lambda_linear": rl.best_params["lam"],
                     "q2_gauss": round(rg.best_q2, 4), "alpha_gauss": rg.best_alpha,
                     "lambda_gauss": rg.best_params["lam"], "gamma": rg.best_params["gamma"],
                     "index_joint": round(idx["joint"].spread, 4),
                     "index_electrostatic": round(idx["electrostatic"].spread, 4),
                     "stability": round(stab, 4)})
    st = stats(maps["interacting"], ref=maps["additive"])
    print("\n== interacting - additive: " + ", ".join(f"{k} {v:.3f}" for k, v in st.items()))
    measures.append({"maps": "interacting − additive", **{k: round(v, 4) for k, v in st.items()}})
    panels["interacting − additive"] = panels["interacting model"] - panels["additive model"]
    des = []
    for target in TARGETS:
        d = designed(maps[target], maps["additive"])
        print(f"== designed, {target} model: " + ", ".join(f"{k} {v:.3f}" for k, v in d.items()))
        des.append({"model": target, **{k: round(v, 4) for k, v in d.items()}})
    for name, rows_ in (("fits.csv", fits), ("measures.csv", measures), ("designed.csv", des)):
        with open(os.path.join(a.out_dir, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)
    plot(panels, mf[shown], k_shown, os.path.join(a.out_dir, "pair_fields.png"))
    report(a.out_dir)
    print(f"\nwritten to {relative_path(a.out_dir, ROOT)} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
