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
#: Shannon's ionic radii for six-fold coordination, angstrom (Acta Cryst. A32 (1976) 751)
ION_RADII = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Ag": 1.15,
             "Mg": 0.72, "Ca": 1.00, "Sr": 1.18, "Ba": 1.35}
#: the radius of a donor oxygen in the same scale, and how much longer than the sum of the two
#: radii a contact may be and still count as a contact, angstrom
DONOR_RADIUS = 1.40
CONTACT_SLACK = 0.60
#: the number of donors an ion has to touch to count as bound
MIN_CONTACTS = 3
#: the force constant, in hartree per square bohr, of the restraints that pull every donor of
#: the ring to its contact distance from the ion before the free optimisation
WRAP_FORCE = 0.05


def contact_distance(ion: str) -> float:
    """The largest ion-donor distance counted as a contact, angstrom."""
    return ION_RADII[ion] + DONOR_RADIUS + CONTACT_SLACK


def _clear_point(point: np.ndarray, xyz: np.ndarray, need: float = 1.8) -> np.ndarray:
    """`point`, or the nearest point of a 0.5 A grid within 3 A of it that no atom of `xyz` is
    closer to than `need`.  The centroid of the donors of a large folded ring can lie on an
    atom, and an ion started there throws that atom off in the first optimisation step."""
    xyz = np.asarray(xyz, float)
    if float(np.linalg.norm(xyz - point, axis=1).min()) >= need:
        return point
    g = np.arange(-3.0, 3.01, 0.5)
    shifts = np.array([(x, y, z) for x in g for y in g for z in g])
    shifts = shifts[np.argsort(np.linalg.norm(shifts, axis=1), kind="stable")]
    for s in shifts:
        if float(np.linalg.norm(xyz - (point + s), axis=1).min()) >= need:
            return point + s
    return point


def ion_starts(xyz: np.ndarray, donors: Sequence[int], ion: str) -> List[np.ndarray]:
    """Where an ion is put before a complex is optimised.

    The centroid of the donors when the ring is wide enough for the ion.  When the mean
    distance of the donors from their centroid is shorter than the sum of the two radii, the
    ion does not fit in the plane, and it is put above and below the ring at the height that
    gives it that contact distance: two starts in place of one.  An ion started in the plane
    of a ring that is too small for it is pushed to the edge of the ring by the optimiser and
    ends on two donors only.  A start that lies on an atom is moved off it
    (:func:`_clear_point`).
    """
    xyz = np.asarray(xyz, float)
    p = xyz[list(donors)]
    centre = p.mean(axis=0)
    radius = float(np.linalg.norm(p - centre, axis=1).mean())
    reach = ION_RADII[ion] + DONOR_RADIUS
    height = float(np.sqrt(max(reach * reach - radius * radius, 0.0)))
    if height < 0.5:
        return [_clear_point(centre, xyz)]
    normal = np.linalg.svd(p - centre)[2][2]
    return [_clear_point(centre + height * normal, xyz),
            _clear_point(centre - height * normal, xyz)]


def donor_misfit(xyz: np.ndarray, donors: Sequence[int], ion_xyz: np.ndarray, ion: str) -> float:
    """The root-mean-square difference, in angstrom, between the ion-donor distances and the
    sum of the two radii: zero when every donor of the ring is at its contact distance."""
    d = np.linalg.norm(np.asarray(xyz, float)[list(donors)] - np.asarray(ion_xyz, float), axis=1)
    return float(np.sqrt(np.mean((d - (ION_RADII[ion] + DONOR_RADIUS)) ** 2)))


def ion_contacts(xyz: np.ndarray, donors: Sequence[int], ion_xyz: np.ndarray, ion: str) -> int:
    """How many of `donors` are in contact with an ion at `ion_xyz`."""
    d = np.linalg.norm(np.asarray(xyz, float)[list(donors)] - np.asarray(ion_xyz, float), axis=1)
    return int(np.sum(d <= contact_distance(ion)))


def _xtb_optimise(symbols: Sequence[str], xyz: np.ndarray, charge: int, workdir: str,
                  solvent: Optional[str], exe: str, threads: int = 1,
                  gfn: int = 1, restraints: Sequence = ()) -> (np.ndarray, float):
    """Optimise one structure with GFN`gfn`-xTB; returns (coordinates, energy in hartree).

    `restraints` are (atom, atom, distance in angstrom) triples, atoms counted from 0, held
    with a harmonic force of :data:`WRAP_FORCE`; the energy returned then includes them.
    """
    import os
    import subprocess
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "in.xyz")
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n\n")
        for s, p in zip(symbols, xyz):
            fh.write(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    # "loose" stops at 4e-3 Eh/bohr on the gradient: enough for the shape of a cavity, and a
    # long alkyl chain otherwise takes hundreds of steps that do not move the ring
    cmd = [exe, "in.xyz", "--opt", "loose", "--gfn", str(int(gfn)), "--chrg", str(int(charge))]
    if restraints:
        with open(os.path.join(workdir, "restrain.inp"), "w") as fh:
            fh.write(f"$constrain\n  force constant={WRAP_FORCE}\n")
            for i, j, dist in restraints:
                fh.write(f"  distance: {int(i) + 1}, {int(j) + 1}, {float(dist):.4f}\n")
            fh.write("$end\n")
        cmd += ["--input", "restrain.inp"]
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
                 xtb: Optional[str] = None, threads: int = 1,
                 gfn: int = 1, release: bool = False) -> Dict[str, object]:
    """The host of `smiles` wrapped around `ion`, with GFN`gfn`-xTB in implicit `solvent`.

    A conformer generator does not give the conformation a binding constant belongs to: an
    empty crown ether folds its ring and turns its oxygens outwards.  So `conformers` starts are
    made with RDKit's macrocycle torsions, the ion is put where :func:`ion_starts` says, and
    each start is optimised while every donor of the ring is pulled to its contact distance
    from the ion (the sum of :data:`ION_RADII` and :data:`DONOR_RADIUS`) by a harmonic
    restraint of :data:`WRAP_FORCE`.  The start that is returned is the one whose donors come
    closest to that distance (:func:`donor_misfit`) among those whose bonds are intact
    (:func:`_bonds_intact`).

    The result is therefore **the host as far around the ion as it can go**, not a minimum of
    the energy, and that is a choice forced by the method.  Without the restraint none of
    GFN1-xTB, GFN2-xTB and GFN-FF keeps the potassium ion of 15-crown-5 on all five of its
    oxygens (three with GFN1 and GFN-FF, four with GFN2), and the contacts are too short: 2.50,
    2.50 and 2.59 A with GFN1, where the sum of the radii is 2.78 A.  A method that draws the
    contact too short prefers a few short contacts to many of the right length.  `release` adds the unrestrained optimisation
    after the restrained one and returns its result, the lowest energy among the starts whose
    ion still touches at least :data:`MIN_CONTACTS` donors (:func:`ion_contacts`).

    What the restraint cannot do is the information: a ring that is too small leaves the ion
    above it, a ring that is too large cannot bring every donor in, and the misfit that
    remains is returned.

    GFN1 is the default and not GFN2, because GFN2 draws potassium nearly as small as sodium:
    on 18-crown-6, dibenzo-18-crown-6 and [2.2.2]cryptand the mean potassium-oxygen contact of
    the unrestrained complex is 2.80, 2.79 and 2.78 A with GFN1 and 2.68, 2.60 and 2.71 A with
    GFN2, and the sum of the two radii is 2.78 A.

    Returns the complex as an RDKit molecule (the host, then the ion as the last atom), the
    host alone in the same conformation, the energy in hartree (with the restraint energy when
    the complex is not released), the misfit, the number of contacts, and how many starts were
    kept of how many.
    """
    import tempfile
    from rdkit.Chem import AllChem
    from .xtb import find_xtb
    if ion not in IONS:
        raise CavityError(f"no ion {ion!r}; use one of {', '.join(IONS)}")
    exe = find_xtb(xtb)
    host = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    # RDKit multiplies the seed by the number of the conformer, so a seed of 0 gives the same
    # conformer every time; the seed asked for is therefore shifted by one
    params.randomSeed = max(1, int(seed) + 1)
    params.useMacrocycleTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(host, int(conformers), params))
    if not cids:
        raise CavityError(f"RDKit could not embed {smiles}")
    AllChem.MMFFOptimizeMoleculeConfs(host, maxIters=2000)
    donors = donor_atoms(host)
    if len(donors) < 3:
        raise CavityError(f"{smiles} has {len(donors)} donors in a macrocycle")
    symbols = [a.GetSymbol() for a in host.GetAtoms()] + [ion]
    tmp = tempfile.mkdtemp(prefix="cavity_") if workdir is None else workdir
    reach = ION_RADII[ion] + DONOR_RADIUS
    pulls = [(dn, len(symbols) - 1, reach) for dn in donors]
    best, kept, tried = None, 0, 0
    starts = [(cid, place) for cid in cids
              for place in ion_starts(host.GetConformer(cid).GetPositions(), donors, ion)]
    for k, (cid, place) in enumerate(starts):
        xyz = np.asarray(host.GetConformer(cid).GetPositions(), float)
        one = Chem.Mol(host)
        one.RemoveAllConformers()
        one.AddConformer(Chem.Conformer(host.GetConformer(cid)), assignId=True)
        tried += 1
        try:
            got, energy = _xtb_optimise(symbols, np.vstack([xyz, place]), IONS[ion],
                                        f"{tmp}/{ion}_{k}_wrap", solvent, exe, threads, gfn,
                                        restraints=pulls)
            if release:
                got, energy = _xtb_optimise(symbols, got, IONS[ion], f"{tmp}/{ion}_{k}",
                                            solvent, exe, threads, gfn)
        except CavityError:
            continue
        hx, ix = got[:-1], got[-1]
        if not _bonds_intact(one, hx):
            continue
        touching = ion_contacts(hx, donors, ix, ion)
        misfit = donor_misfit(hx, donors, ix, ion)
        if release:
            bound = touching >= min(MIN_CONTACTS, len(donors))
            kept += int(bound)
            # a bound start first, then more contacts among the unbound, then the energy
            rank = (0, 0, energy) if bound else (1, -touching, energy)
        else:
            kept += 1
            rank = (misfit, energy)
        if best is None or rank < best[0]:
            best = (rank, energy, got, one, misfit, touching)
    if best is None:
        raise CavityError(f"no start of {smiles} with {ion} kept its bonds")
    _rank, energy, got, one, misfit, touching = best
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
            "starts": tried, "ion_index": idx, "misfit": misfit, "contacts": touching,
            "released": bool(release)}


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
