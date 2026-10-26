"""Place conformations inside the pocket instead of in the gas phase.

Every other way of building a 4D ensemble in this program has failed on an aligned series, and for
the same reason each time: conformations generated for an isolated molecule are not the poses a
receptor selects, so averaging over them, or choosing among them, only dilutes or re-finds the pose
the series already carried.  The remaining idea is to make the ensemble bound-like by construction,
which means putting every conformation into the site and letting the site decide where it sits.

This is a small rigid-body docking built from what the program already has: the fields of the
pocket from :mod:`cmfqsar.complementarity`, the exact overlap integrals between two sets of
Gaussians, and a soft clash term.  A pose is scored by

    score = complementarity of the fields with the pocket - clash penalty

and the six rigid-body parameters of each conformation are optimised from the pose it arrives in,
with optional random restarts.  The internal geometry is never touched, so the conformations stay
the ones that were generated; only their placement changes.

What this is not is a docking program.  There is no desolvation, no entropy, no torsional search
and no scoring function fitted to binding data, and a real program would do better.  It is enough
to answer the question this project is asking: whether an ensemble placed by the receptor behaves
differently from one placed by superposition on a pose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from .complementarity import COMPLEMENTARY_PAIRS, _widths
from .fields import FieldSet, MoleculeFields, build_fields

#: weight of the clash term relative to the field terms; a pose that clashes is not a pose
CLASH_WEIGHT = 2.0
#: weight of the burial term: without it nothing holds a pose inside the site, since the field
#: terms are satisfied at the mouth of the pocket and the clash term is lowest outside it
BURIAL_WEIGHT = 2.0
#: how far the centroid of a conformation may move before it is penalised, in angstrom
TETHER_RADIUS = 3.0
TETHER_WEIGHT = 1.0


@dataclass
class DockResult:
    """What happened to one conformation."""
    score_before: float
    score_after: float
    clash_before: float
    clash_after: float
    moved: float                      # how far the conformation moved, angstrom
    restarts_used: int = 0


def _clash(coords: np.ndarray, radii: np.ndarray, pocket_coords: np.ndarray,
           pocket_radii: np.ndarray, tolerance: float = 0.4) -> float:
    """Soft repulsion: the squared overlap of van der Waals spheres, summed over atom pairs."""
    d = cdist(coords, pocket_coords)
    lim = radii[:, None] + pocket_radii[None, :] - tolerance
    over = np.maximum(lim - d, 0.0)
    return float((over * over).sum())


class PocketScorer:
    """Scores a pose against one pocket, with everything that does not move precomputed."""

    def __init__(self, mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet,
                 clash_weight: float = CLASH_WEIGHT, clash_tolerance: float = 0.4,
                 heavy_only: bool = True, cutoff: float = 12.0,
                 burial_weight: float = BURIAL_WEIGHT, tether_radius: float = TETHER_RADIUS,
                 tether_weight: float = TETHER_WEIGHT):
        self.fieldset = fieldset
        self.names = [s.name for s in fieldset.specs]
        self.mol = mol
        self.clash_weight = float(clash_weight)
        self.clash_tolerance = float(clash_tolerance)
        self.burial_weight = float(burial_weight)
        self.tether_radius = float(tether_radius)
        self.tether_weight = float(tether_weight)
        self.start_centroid = mol.coords.mean(axis=0)
        self.terms: List[Tuple[int, int, float]] = []
        for lig, poc, sign, _label in COMPLEMENTARY_PAIRS:
            if lig in self.names and poc in self.names:
                self.terms.append((self.names.index(lig), self.names.index(poc), float(sign)))
        # only the pocket atoms the ligand can reach matter, and dropping the rest makes every
        # score evaluation several times cheaper without changing it measurably
        if cutoff and cutoff > 0 and pocket.n_atoms:
            near = cKDTree(pocket.coords).query_ball_point(mol.coords, r=float(cutoff))
            idx = np.unique(np.array([i for h in near for i in h], dtype=int)) if any(
                len(h) for h in near) else np.arange(pocket.n_atoms)
            if len(idx) == 0:
                idx = np.arange(pocket.n_atoms)
            pocket = _subset_fields(pocket, idx)
        # pocket side, fixed: coordinates, weights and Gaussian widths per field in use
        keep = [i for i in range(pocket.n_atoms) if not heavy_only or pocket.symbols[i] != "H"]
        self.p_coords = pocket.coords
        self.p_widths = {f: _widths(pocket, fieldset.specs[f])
                         for _, f, _ in [(0, g, 0) for g in {t[1] for t in self.terms}]}
        self.p_weights = {f: pocket.weights[:, f] for _, f, _ in
                          [(0, g, 0) for g in {t[1] for t in self.terms}]}
        self.p_clash_coords = pocket.coords[keep]
        self.p_clash_radii = pocket.radii[keep]
        self.m_widths = {f: _widths(mol, fieldset.specs[f]) for f, _, _ in self.terms}
        self.m_weights = {f: mol.weights[:, f] for f, _, _ in self.terms}
        self.m_heavy = [i for i in range(mol.n_atoms) if not heavy_only or mol.symbols[i] != "H"]
        self.m_radii = mol.radii[self.m_heavy]
        # normalisation: the self-overlap of each side, so the terms are cosines and comparable
        self.m_norm = {f: np.sqrt(max(_self_overlap(mol.coords, self.m_weights[f],
                                                    self.m_widths[f]), 1e-300))
                       for f, _, _ in self.terms}
        self.p_norm = {g: np.sqrt(max(_self_overlap(pocket.coords, self.p_weights[g],
                                                    self.p_widths[g]), 1e-300))
                       for _, g, _ in self.terms}
        self.steric = self.names.index("steric") if "steric" in self.names else None
        if self.steric is not None:
            f = self.steric
            self.m_widths[f] = _widths(mol, fieldset.specs[f])
            self.m_weights[f] = mol.weights[:, f]
            self.p_widths[f] = _widths(pocket, fieldset.specs[f])
            self.p_weights[f] = pocket.weights[:, f]
            self.m_norm[f] = np.sqrt(max(_self_overlap(mol.coords, self.m_weights[f],
                                                       self.m_widths[f]), 1e-300))
            self.p_norm[f] = np.sqrt(max(_self_overlap(pocket.coords, self.p_weights[f],
                                                       self.p_widths[f]), 1e-300))

    def score(self, coords: np.ndarray) -> float:
        total = 0.0
        for f, g, sign in self.terms:
            s = _cross_overlap(coords, self.m_weights[f], self.m_widths[f],
                               self.p_coords, self.p_weights[g], self.p_widths[g])
            total += sign * s / (self.m_norm[f] * self.p_norm[g])
        if self.steric is not None and self.burial_weight:
            f = self.steric
            b = _cross_overlap(coords, self.m_weights[f], self.m_widths[f],
                               self.p_coords, self.p_weights[f], self.p_widths[f])
            total += self.burial_weight * b / (self.m_norm[f] * self.p_norm[f])
        total -= self.clash_weight * _clash(coords[self.m_heavy], self.m_radii,
                                            self.p_clash_coords, self.p_clash_radii,
                                            self.clash_tolerance)
        if self.tether_weight:
            drift = float(np.linalg.norm(coords.mean(axis=0) - self.start_centroid))
            if drift > self.tether_radius:
                total -= self.tether_weight * (drift - self.tether_radius) ** 2
        return float(total)

    def clash(self, coords: np.ndarray) -> float:
        return _clash(coords[self.m_heavy], self.m_radii, self.p_clash_coords,
                      self.p_clash_radii, self.clash_tolerance)


def _subset_fields(mol: MoleculeFields, idx: np.ndarray) -> MoleculeFields:
    """The same field object restricted to some of its atoms; used to trim a pocket."""
    idx = np.asarray(idx, int)
    return MoleculeFields(name=mol.name, coords=mol.coords[idx], weights=mol.weights[idx],
                          radii=mol.radii[idx], symbols=[mol.symbols[i] for i in idx],
                          atom_types=[mol.atom_types[i] for i in idx],
                          bonds=np.zeros((0, 2), dtype=int), charges=mol.charges[idx])


def _self_overlap(coords: np.ndarray, weights: np.ndarray, widths: np.ndarray) -> float:
    return _cross_overlap(coords, weights, widths, coords, weights, widths)


def _cross_overlap(xa: np.ndarray, wa: np.ndarray, ba: np.ndarray,
                   xb: np.ndarray, wb: np.ndarray, bb: np.ndarray) -> float:
    ia, ib = np.flatnonzero(np.abs(wa) > 1e-12), np.flatnonzero(np.abs(wb) > 1e-12)
    if len(ia) == 0 or len(ib) == 0:
        return 0.0
    s = ba[ia][:, None] + bb[ib][None, :]
    d2 = cdist(xa[ia], xb[ib], "sqeuclidean")
    g = (2.0 * np.pi / s) ** 1.5 * np.exp(-(ba[ia][:, None] * bb[ib][None, :]) / (2.0 * s) * d2)
    return float(wa[ia] @ g @ wb[ib])


def _transform(coords: np.ndarray, p: np.ndarray, center: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(p[:3]).as_matrix()
    return (coords - center) @ R.T + center + p[3:]


def dock_coords(mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet,
                restarts: int = 0, seed: int = 0, max_iter: int = 200,
                max_shift: float = 2.0, max_rotation_deg: float = 30.0,
                clash_weight: float = CLASH_WEIGHT,
                tether_radius: float = TETHER_RADIUS) -> Tuple[np.ndarray, DockResult]:
    """Optimise the rigid-body placement of one conformation inside the pocket.

    The search starts from the pose the conformation arrives in and stays near it: restarts are
    bounded perturbations rather than a global search, because the series is already in the site
    and the question is where in the site each conformation belongs, not whether it binds.
    """
    scorer = PocketScorer(mol, pocket, fieldset, clash_weight=clash_weight,
                          tether_radius=tether_radius)
    center = mol.coords.mean(axis=0)

    def objective(p):
        return -scorer.score(_transform(mol.coords, p, center))

    best_p = np.zeros(6)
    before = -objective(best_p)
    best_val = -before
    rng = np.random.RandomState(seed)
    starts = [np.zeros(6)]
    for _ in range(max(int(restarts), 0)):
        axis = rng.normal(size=3)
        axis /= max(np.linalg.norm(axis), 1e-9)
        rv = axis * np.deg2rad(rng.uniform(0.0, max_rotation_deg))
        starts.append(np.concatenate([rv, rng.normal(scale=max_shift / 2.0, size=3)]))
    used = 0
    for k, p0 in enumerate(starts):
        res = minimize(objective, p0, method="Powell",
                       options={"maxiter": max_iter, "xtol": 1e-2, "ftol": 1e-4})
        if res.fun < best_val:
            best_val, best_p, used = float(res.fun), np.asarray(res.x, float), k
    coords = _transform(mol.coords, best_p, center)
    moved = float(np.sqrt(((coords - mol.coords) ** 2).sum(axis=1).mean()))
    return coords, DockResult(score_before=before, score_after=-best_val,
                              clash_before=scorer.clash(mol.coords),
                              clash_after=scorer.clash(coords), moved=moved, restarts_used=used)


def dock_conformers(mol: Chem.Mol, pocket: MoleculeFields, fieldset: FieldSet,
                    restarts: int = 0, seed: int = 0, max_shift: float = 2.0,
                    max_rotation_deg: float = 30.0, keep_best: int = 0,
                    clash_weight: float = CLASH_WEIGHT,
                    tether_radius: float = TETHER_RADIUS) -> Tuple[Chem.Mol, List[DockResult]]:
    """Dock every conformation of a molecule into the pocket, in place in a copy.

    With `keep_best` the conformations are ordered by their docked score and only that many are
    kept, which is how a docking program would return its best poses.
    """
    out = Chem.Mol(mol)
    single = fieldset.copy()
    single.conformers = "single"
    results: List[DockResult] = []
    coords_all = []
    for cid in [c.GetId() for c in mol.GetConformers()]:
        one = Chem.Mol(mol)
        keep = Chem.Conformer(mol.GetConformer(cid))
        one.RemoveAllConformers()
        one.AddConformer(keep, assignId=True)
        mf = build_fields(one, single, "")
        coords, res = dock_coords(mf, pocket, single, restarts=restarts, seed=seed,
                                  max_shift=max_shift, max_rotation_deg=max_rotation_deg,
                                  clash_weight=clash_weight, tether_radius=tether_radius)
        coords_all.append(coords)
        results.append(res)
    order = list(range(len(coords_all)))
    if keep_best and len(order) > keep_best:
        order = sorted(order, key=lambda i: -results[i].score_after)[:keep_best]
        order = sorted(order)
    out.RemoveAllConformers()
    for i in order:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for a, xyz in enumerate(coords_all[i]):
            conf.SetAtomPosition(a, [float(v) for v in xyz])
        out.AddConformer(conf, assignId=True)
    for prop in mol.GetPropNames():
        out.SetProp(prop, mol.GetProp(prop))
    if mol.HasProp("_Name"):
        out.SetProp("_Name", mol.GetProp("_Name"))
    return out, [results[i] for i in order]


def dock_dataset(mols: Sequence[Chem.Mol], pocket: MoleculeFields, fieldset: FieldSet,
                 restarts: int = 0, seed: int = 0, keep_best: int = 0, max_shift: float = 2.0,
                 max_rotation_deg: float = 30.0, tether_radius: float = TETHER_RADIUS,
                 progress=None, cancel=None) -> Tuple[List[Chem.Mol], List[DockResult]]:
    """Dock the conformations of a whole series; returns the molecules and every pose result."""
    out, all_res = [], []
    for i, m in enumerate(mols):
        if cancel is not None and cancel():
            break
        if progress is not None:
            progress(f"Docking {i + 1}/{len(mols)}", i / max(len(mols), 1))
        try:
            dm, res = dock_conformers(m, pocket, fieldset, restarts=restarts, seed=seed,
                                      keep_best=keep_best, max_shift=max_shift,
                                      max_rotation_deg=max_rotation_deg,
                                      tether_radius=tether_radius)
        except Exception:                        # noqa: BLE001 - a molecule that fails keeps its pose
            dm, res = Chem.Mol(m), []
        out.append(dm)
        all_res.extend(res)
    return out, all_res


def summarise(results: Sequence[DockResult]) -> dict:
    if not results:
        return {}
    return {"poses": float(len(results)),
            "score_before": float(np.mean([r.score_before for r in results])),
            "score_after": float(np.mean([r.score_after for r in results])),
            "clash_before": float(np.mean([r.clash_before for r in results])),
            "clash_after": float(np.mean([r.clash_after for r in results])),
            "moved": float(np.mean([r.moved for r in results]))}
