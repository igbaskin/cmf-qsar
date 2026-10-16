"""
3D structure generation and molecular alignment.

* embed_3d          - ETKDG conformer + MMFF optimisation for a SMILES-derived molecule
* mcs_align         - rigid alignment on the maximum common substructure with a reference
* field_align       - rigid-body refinement maximising the (normalised) CMF kernel
                      between a molecule and the reference (the alignment criterion
                      suggested in Baskin & Zhokhova 2013)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .fields import AtomTable, FieldSet, MoleculeFields, build_fields
from .kernels import atom_kernel_block


def embed_3d(mol: Chem.Mol, seed: int = 42, optimize: bool = True) -> Chem.Mol:
    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(m, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) != 0:
            raise RuntimeError("3D embedding failed")
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
        except Exception:
            pass
        # MMFF set-up re-perceives aromaticity with the MMFF model *in place*; restore RDKit's model
        Chem.SanitizeMol(m)
    return m


def transform_coords(coords: np.ndarray, rotvec: np.ndarray, trans: np.ndarray, center: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return (coords - center) @ R.T + center + trans


def set_coords(mol: Chem.Mol, coords: np.ndarray) -> None:
    conf = mol.GetConformer()
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, [float(v) for v in xyz])


def mcs_align(mol: Chem.Mol, ref: Chem.Mol, timeout: int = 10, ring_matches_ring_only: bool = True,
              complete_rings_only: bool = False, compare_elements: bool = True) -> Tuple[float, int]:
    """Rigidly superimpose `mol` onto `ref` using their maximum common substructure.

    Returns (rmsd, number of matched atoms).  `mol` is modified in place.
    """
    params = rdFMCS.MCSParameters()
    params.Timeout = timeout
    params.AtomCompareParameters.RingMatchesRingOnly = ring_matches_ring_only
    params.AtomCompareParameters.CompleteRingsOnly = complete_rings_only
    params.BondCompareParameters.RingMatchesRingOnly = ring_matches_ring_only
    params.BondCompareParameters.CompleteRingsOnly = complete_rings_only
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements if compare_elements else rdFMCS.AtomCompare.CompareAny
    params.BondTyper = rdFMCS.BondCompare.CompareOrder
    heavy_mol = Chem.RemoveHs(mol)
    heavy_ref = Chem.RemoveHs(ref)
    res = rdFMCS.FindMCS([heavy_ref, heavy_mol], params)
    if res.numAtoms < 3:
        raise RuntimeError("MCS too small for alignment (< 3 atoms)")
    patt = Chem.MolFromSmarts(res.smartsString)
    m_ref = ref.GetSubstructMatch(patt)
    m_mol = mol.GetSubstructMatch(patt)
    if not m_ref or not m_mol:
        raise RuntimeError("MCS substructure match failed")
    amap = list(zip(m_mol, m_ref))
    rmsd = rdMolAlign.AlignMol(mol, ref, atomMap=amap)
    return float(rmsd), int(res.numAtoms)


def _kernel_score(coords: np.ndarray, probe: MoleculeFields, ref_table: AtomTable, specs, ref_self: np.ndarray,
                  probe_self: np.ndarray, h: np.ndarray) -> float:
    blk = atom_kernel_block(coords, probe.weights, probe.radii, ref_table, specs)   # (F, N, 1)
    k = blk.sum(axis=1)[:, 0]
    denom = np.sqrt(np.maximum(ref_self * probe_self, 1e-300))
    kn = np.where(ref_self * probe_self > 1e-300, k / denom, 0.0)
    return float(np.sum(h * kn))


def field_align(mol: Chem.Mol, ref: Chem.Mol, fieldset: FieldSet, restarts: int = 0, seed: int = 0,
                max_iter: int = 300) -> Tuple[float, float]:
    """Refine the pose of `mol` by maximising the normalised CMF kernel with `ref`.

    Returns (score_before, score_after); `mol` is modified in place.
    """
    probe = build_fields(mol, fieldset)
    reff = build_fields(ref, fieldset)
    ref_table = AtomTable.from_molecules([reff])
    specs = fieldset.specs
    h = fieldset.mixing_vector()
    ref_self = atom_kernel_block(reff.coords, reff.weights, reff.radii, ref_table, specs).sum(axis=1)[:, 0]
    probe_table = AtomTable.from_molecules([probe])
    probe_self = atom_kernel_block(probe.coords, probe.weights, probe.radii, probe_table, specs).sum(axis=1)[:, 0]
    center = probe.coords.mean(axis=0)

    def objective(p):
        c = transform_coords(probe.coords, p[:3], p[3:], center)
        return -_kernel_score(c, probe, ref_table, specs, ref_self, probe_self, h)

    best_p = np.zeros(6)
    before = -objective(best_p)
    best_val = objective(best_p)
    rng = np.random.RandomState(seed)
    starts = [np.zeros(6)]
    for _ in range(restarts):
        rv = Rotation.random(random_state=rng).as_rotvec()
        starts.append(np.concatenate([rv, rng.normal(scale=1.0, size=3)]))
    for p0 in starts:
        res = minimize(objective, p0, method="Powell", options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-6})
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
    new_coords = transform_coords(probe.coords, best_p[:3], best_p[3:], center)
    set_coords(mol, new_coords)
    return float(before), float(-best_val)


def align_dataset(mols: Sequence[Chem.Mol], ref_index: int, use_mcs: bool = True,
                  field_refine: bool = False, fieldset: Optional[FieldSet] = None,
                  progress=None) -> List[str]:
    """Align every molecule to mols[ref_index] in place. Returns a log line per molecule."""
    ref = mols[ref_index]
    log = []
    for i, m in enumerate(mols):
        if i == ref_index:
            log.append("reference")
            continue
        msg = []
        try:
            if use_mcs:
                rmsd, n = mcs_align(m, ref)
                msg.append(f"MCS {n} atoms, RMSD {rmsd:.2f}")
            if field_refine and fieldset is not None:
                b, a = field_align(m, ref, fieldset)
                msg.append(f"field similarity {b:.3f} -> {a:.3f}")
        except Exception as exc:  # noqa: BLE001
            msg.append(f"FAILED: {exc}")
        log.append("; ".join(msg))
        if progress:
            progress(i + 1, len(mols))
    return log


# --------------------------------------------------------------------------- #
# Unalign: randomise the pose of every molecule
# --------------------------------------------------------------------------- #

def random_rigid_transform(coords: np.ndarray, rng: np.random.RandomState,
                           max_rotation: Optional[float] = None,
                           max_translation: float = 2.0) -> np.ndarray:
    """Randomly rotate `coords` about their centroid and translate them.

    Parameters
    ----------
    max_rotation : maximum rotation angle in degrees.  ``None`` draws a uniformly
        distributed orientation (the alignment is then destroyed completely);
        a number draws a random axis and an angle uniform in [0, max_rotation].
    max_translation : the displacement is a random direction times a length drawn
        uniformly from [0, max_translation] (A).

    The transformation is rigid, so all interatomic distances are preserved.
    """
    coords = np.asarray(coords, dtype=float)
    c = coords.mean(axis=0)
    if max_rotation is None:
        R = Rotation.random(random_state=rng).as_matrix()
    else:
        axis = rng.normal(size=3)
        axis /= max(np.linalg.norm(axis), 1e-12)
        R = Rotation.from_rotvec(axis * np.deg2rad(rng.uniform(0.0, float(max_rotation)))).as_matrix()
    d = rng.normal(size=3)
    d /= max(np.linalg.norm(d), 1e-12)
    t = d * rng.uniform(0.0, float(max_translation))
    return (coords - c) @ R.T + c + t


def unalign(mols: Sequence[MoleculeFields], max_rotation: Optional[float] = None,
            max_translation: float = 2.0, seed: int = 0, progress=None) -> List[float]:
    """Destroy the alignment of a set of molecules by giving each a random pose.

    Every molecule is rotated about its own centroid and translated, so the internal
    geometry (and therefore the self-kernel of every molecule) is unchanged while the
    mutual arrangement is randomised.  This is the control experiment for any alignment
    method: scramble the poses, realign, and measure how much of the original arrangement
    comes back.

    The molecules are modified in place; the displacement (RMSD, A) of each is returned.
    """
    rng = np.random.RandomState(seed)
    out = []
    for i, mol in enumerate(mols):
        before = np.array(mol.coords, dtype=float)
        mol.coords = random_rigid_transform(before, rng, max_rotation, max_translation)
        out.append(float(np.sqrt(((mol.coords - before) ** 2).sum(1).mean())))
        if progress:
            progress(i + 1, len(mols))
    return out
