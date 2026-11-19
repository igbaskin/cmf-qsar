"""The frame a host's cavity gives (plan item SC5).

A series of crown ethers, cryptands or other macrocyclic hosts has a physical frame that few drug
series have: the cavity.  Two hosts of a series often share no scaffold that a substructure
superposition could use (18-crown-6 and [2.2.2]cryptand have no common ring), but both hold a
cation at the centre of a set of donor atoms.  The frame of this module is put on that centre:

* **the donors** are the oxygen, nitrogen and sulfur atoms that are members of a ring of at
  least :data:`MIN_MACROCYCLE` atoms (the macrocycle); a donor on a side arm or on a fused
  benzene ring is not one of them.  A caller may give the donors instead;
* **the origin** is the centroid of the donors;
* **the z axis** is the unique axis of the donor set: the principal axis whose moment is
  farthest from the other two.  For the ring of a crown ether, flat or folded, that is the
  normal of the ring; for an elongated cage such as a cryptand it is the axis through the two
  bridgehead nitrogens;
* **the x axis** is the direction, in the plane normal to z, of the largest extent of the heavy
  atoms, so a benzo or cyclohexano ring fused to the macrocycle lies along x;
* **the signs** of x and z are those that make the third moment of the heavy atoms along them
  positive, so the side of the ring that carries the substituents is the same in every host;
  y completes a right-handed frame.

A ring that is symmetric about z, such as 18-crown-6 itself, leaves the turn about z free, and
:func:`cavity_frame_quality` reports how close the two in-plane moments are.  For such a host the
fields are nearly the same in every turn, which is why the frame is still usable.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

#: the smallest ring counted as a macrocycle; 12-crown-4 has a ring of 12 atoms
MIN_MACROCYCLE = 9
#: the elements that count as donors
DONOR_ELEMENTS = ("O", "N", "S")


class CavityError(ValueError):
    """A molecule that has no macrocycle, or too few donors in it to fix a frame."""


def macrocycle_atoms(mol: Chem.Mol, min_size: int = MIN_MACROCYCLE) -> List[int]:
    """The atoms that are members of a ring of at least `min_size` atoms.

    RDKit's smallest set of smallest rings does not always contain the large ring of a cage
    (in a cryptand every ring passes through both bridgeheads), so the rings are taken from the
    symmetrised set, which holds every ring of a bicyclic cage."""
    rings = Chem.GetSymmSSSR(mol)
    out = set()
    for ring in rings:
        if len(ring) >= int(min_size):
            out.update(int(i) for i in ring)
    return sorted(out)


def donor_atoms(mol: Chem.Mol, min_size: int = MIN_MACROCYCLE) -> List[int]:
    """The O, N and S atoms of the macrocycle."""
    ring = set(macrocycle_atoms(mol, min_size))
    return [a.GetIdx() for a in mol.GetAtoms()
            if a.GetIdx() in ring and a.GetSymbol() in DONOR_ELEMENTS]


def _moments(p: np.ndarray) -> (np.ndarray, np.ndarray):
    """Eigenvalues (ascending) and eigenvectors (rows) of the covariance of points `p`
    about their centroid."""
    c = p - p.mean(axis=0)
    w, v = np.linalg.eigh(c.T @ c / max(len(p), 1))
    return w, v.T


def cavity_axes(coords: np.ndarray, donors: Sequence[int],
                heavy: Optional[Sequence[int]] = None) -> (np.ndarray, np.ndarray):
    """(origin, rotation) of the cavity frame for `coords`; frame coordinates are
    ``(coords - origin) @ rotation.T``.  `heavy` are the atoms that fix the turn about z and the
    signs (all atoms when not given)."""
    xyz = np.asarray(coords, float)
    donors = np.asarray(donors, int)
    if len(donors) < 3:
        raise CavityError(f"{len(donors)} donor atoms do not fix a plane; a cavity frame needs "
                          "at least three")
    heavy = np.arange(len(xyz)) if heavy is None else np.asarray(heavy, int)
    origin = xyz[donors].mean(axis=0)
    w, v = _moments(xyz[donors])
    # the unique axis: the moment farthest from the mean of the other two
    gaps = [abs(w[k] - np.mean(np.delete(w, k))) for k in range(3)]
    z = v[int(np.argmax(gaps))]
    h = xyz[heavy] - origin
    if float(((h @ z) ** 3).sum()) < 0:
        z = -z
    # x: the largest in-plane extent of the heavy atoms
    flat = h - np.outer(h @ z, z)
    w2, v2 = np.linalg.eigh(flat.T @ flat)
    x = v2[:, 2] - float(v2[:, 2] @ z) * z
    nx = float(np.linalg.norm(x))
    if nx < 1e-8:                                 # every heavy atom on the axis
        x = np.cross(z, [1.0, 0.0, 0.0])
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(z, [0.0, 1.0, 0.0])
        nx = float(np.linalg.norm(x))
    x = x / nx
    if float(((h @ x) ** 3).sum()) < 0:
        x = -x
    y = np.cross(z, x)
    return origin, np.stack([x, y, z])


def cavity_frame(mol: Chem.Mol, donors: Optional[Sequence[int]] = None,
                 conf_id: int = -1) -> np.ndarray:
    """The coordinates of `mol` in its cavity frame (see the module text)."""
    donors = donor_atoms(mol) if donors is None else list(donors)
    if not donors:
        raise CavityError("the molecule has no O, N or S atom in a ring of at least "
                          f"{MIN_MACROCYCLE} atoms, so it has no cavity to put a frame on")
    xyz = np.asarray(mol.GetConformer(conf_id).GetPositions(), float)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    origin, rot = cavity_axes(xyz, donors, heavy)
    return (xyz - origin) @ rot.T


def cavity_frame_quality(mol: Chem.Mol, donors: Optional[Sequence[int]] = None,
                         conf_id: int = -1) -> Dict[str, float]:
    """How well the donors fix the frame: their number, the mean distance from the origin (the
    cavity radius), how flat they are, and how close the two moments that set the turn about z
    are (a ratio near 1 leaves that turn free)."""
    donors = donor_atoms(mol) if donors is None else list(donors)
    xyz = np.asarray(mol.GetConformer(conf_id).GetPositions(), float)
    p = xyz[np.asarray(donors, int)]
    w, _v = _moments(p)
    r = np.linalg.norm(p - p.mean(axis=0), axis=1)
    gaps = [abs(w[k] - np.mean(np.delete(w, k))) for k in range(3)]
    k = int(np.argmax(gaps))
    rest = np.sort(np.delete(w, k))
    return {"donors": float(len(donors)), "radius": float(r.mean()),
            "radius_spread": float(r.std()),
            "axis_moment": float(w[k]), "in_plane_ratio": float(rest[0] / max(rest[1], 1e-12))}


def with_cavity_frame(mol: Chem.Mol, donors: Optional[Sequence[int]] = None,
                      conf_id: int = -1) -> Chem.Mol:
    """A copy of `mol` whose conformer is moved into its cavity frame."""
    out = Chem.Mol(mol)
    xyz = cavity_frame(mol, donors, conf_id)
    conf = out.GetConformer(conf_id)
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in p]))
    return out


# --------------------------------------------------------------------------- #
# A host with its ion: the conformation the binding constant belongs to
# --------------------------------------------------------------------------- #
#: element symbol and charge of the ions a complex can be built with
IONS = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Ag": 1, "Mg": 2, "Ca": 2, "Sr": 2,
        "Ba": 2}


def _xtb_optimise(symbols: Sequence[str], xyz: np.ndarray, charge: int, workdir: str,
                  solvent: Optional[str], exe: str, threads: int = 1) -> (np.ndarray, float):
    """Optimise one structure with GFN2-xTB; returns (coordinates, energy in hartree)."""
    import os
    import subprocess
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "in.xyz")
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n\n")
        for s, p in zip(symbols, xyz):
            fh.write(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    cmd = [exe, "in.xyz", "--opt", "--gfn", "2", "--chrg", str(int(charge))]
    if solvent:
        cmd += ["--alpb", solvent]
    env = dict(os.environ, OMP_NUM_THREADS=str(int(threads)), MKL_NUM_THREADS=str(int(threads)),
               OMP_STACKSIZE="1G")
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace", timeout=3600)
    out = os.path.join(workdir, "xtbopt.xyz")
    if proc.returncode != 0 or not os.path.exists(out):
        raise CavityError(f"xtb failed in {workdir}: {proc.stderr[-300:] or proc.stdout[-300:]}")
    with open(out) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0])
    energy = float(lines[1].split("energy:")[1].split()[0])
    coords = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + n]])
    return coords, energy


def _bonds_intact(mol: Chem.Mol, xyz: np.ndarray, stretch: float = 1.25) -> bool:
    """Every bond of `mol` is at most `stretch` times its length in the starting conformer,
    and no two atoms that are not bonded came closer than 0.9 A to each other."""
    start = np.asarray(mol.GetConformer().GetPositions(), float)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if np.linalg.norm(xyz[i] - xyz[j]) > stretch * np.linalg.norm(start[i] - start[j]):
            return False
    from scipy.spatial.distance import pdist, squareform
    d = squareform(pdist(xyz))
    np.fill_diagonal(d, np.inf)
    for b in mol.GetBonds():
        d[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = d[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = \
            np.inf
    return bool(d.min() > 0.9)


def host_complex(smiles: str, ion: str = "K", conformers: int = 8, seed: int = 0,
                 solvent: Optional[str] = "methanol", workdir: Optional[str] = None,
                 xtb: Optional[str] = None, threads: int = 1) -> Dict[str, object]:
    """The host of `smiles` holding `ion`, optimised with GFN2-xTB in implicit `solvent`.

    A conformer generator does not give the conformation a binding constant belongs to: an
    empty crown ether folds its ring and turns its oxygens outwards.  So `conformers` starts are
    made with RDKit's macrocycle torsions, the ion is put at the centroid of each start's
    donors, and every complex is optimised.  The lowest energy is kept among the results whose
    host bonds are intact (:func:`_bonds_intact`) and whose ion is still within 1.5 A of the
    donor centroid; the second condition rejects an ion that left the cavity for the outside of
    the ring.

    Returns the complex as an RDKit molecule (the host, then the ion as the last atom), the
    host alone in the same conformation, the energy in hartree, and how many starts were kept.
    """
    import tempfile
    from rdkit.Chem import AllChem
    from .xtb import find_xtb
    if ion not in IONS:
        raise CavityError(f"no ion {ion!r}; use one of {', '.join(IONS)}")
    exe = find_xtb(xtb)
    host = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useMacrocycleTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(host, int(conformers), params))
    if not cids:
        raise CavityError(f"RDKit could not embed {smiles}")
    AllChem.MMFFOptimizeMoleculeConfs(host)
    donors = donor_atoms(host)
    if len(donors) < 3:
        raise CavityError(f"{smiles} has {len(donors)} donors in a macrocycle")
    symbols = [a.GetSymbol() for a in host.GetAtoms()] + [ion]
    tmp = tempfile.mkdtemp(prefix="cavity_") if workdir is None else workdir
    best, kept = None, 0
    for k, cid in enumerate(cids):
        xyz = np.asarray(host.GetConformer(cid).GetPositions(), float)
        start = np.vstack([xyz, xyz[donors].mean(axis=0)])
        one = Chem.Mol(host)
        one.RemoveAllConformers()
        conf = Chem.Conformer(host.GetConformer(cid))
        one.AddConformer(conf, assignId=True)
        try:
            got, energy = _xtb_optimise(symbols, start, IONS[ion],
                                        f"{tmp}/{ion}_{k}", solvent, exe, threads)
        except CavityError:
            continue
        hx, ix = got[:-1], got[-1]
        if not _bonds_intact(one, hx):
            continue
        if np.linalg.norm(ix - hx[donors].mean(axis=0)) > 1.5:
            continue
        kept += 1
        if best is None or energy < best[0]:
            best = (energy, got, one)
    if best is None:
        raise CavityError(f"no start of {smiles} with {ion} kept its bonds and its ion")
    energy, got, one = best
    rw = Chem.RWMol(one)
    atom = Chem.Atom(ion)
    atom.SetFormalCharge(IONS[ion])
    idx = rw.AddAtom(atom)
    conf = rw.GetConformer()
    for i, p in enumerate(got):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in p]))
    complex_ = rw.GetMol()
    host_only = Chem.Mol(one)
    hc = host_only.GetConformer()
    for i, p in enumerate(got[:-1]):
        hc.SetAtomPosition(i, Point3D(*[float(v) for v in p]))
    return {"complex": complex_, "host": host_only, "energy": energy, "kept": kept,
            "starts": len(cids), "ion_index": idx}


def cavity_align_dataset(mols: Sequence[Chem.Mol]) -> List[str]:
    """Move every molecule into its own cavity frame, in place; one log line per molecule.

    Nothing is aligned onto a reference: the frame is a property of each host alone, which is
    what lets two hosts with no common scaffold share it."""
    log = []
    for m in mols:
        try:
            xyz = cavity_frame(m)
            q = cavity_frame_quality(m)
            conf = m.GetConformer()
            for i, p in enumerate(xyz):
                conf.SetAtomPosition(i, Point3D(*[float(v) for v in p]))
            log.append(f"{int(q['donors'])} donors, cavity radius {q['radius']:.2f} A, "
                       f"in-plane moment ratio {q['in_plane_ratio']:.2f}")
        except CavityError as exc:
            log.append(f"FAILED: {exc}")
    return log
