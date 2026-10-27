"""Local alignment: an environment posed on its centre and first shell (plan item LP3).

:mod:`cmfqsar.localenv` (plan item LP2) makes the sample an atom and its surroundings, and
measures it with the rotation-invariant autocorrelation, which needs no frame.  That kernel has
the same limitation as the invariant kernel for whole molecules: with no frame there is no
co-field, so it cannot say **where** around the atom, and in which field, a feature raises or
lowers the property.

For one site the frame is not a problem.  An aromatic carbon with its two ring neighbours, or a
carboxyl carbon with its two oxygens, fixes a frame to within a mirror image, so the environments
of that site in different molecules can be put in one frame by a rule, with nothing fitted and
nothing to converge.  This module is that rule, and with it the overlap kernel and the co-field of
the rest of the program apply to an environment exactly as they apply to a molecule.

**The rule.**  The centre is the origin already (:func:`cmfqsar.localenv.environment` puts it
there and makes it the first atom).  What is left is the rotation, and it is fixed by two of the
centre's bonded neighbours: the first gives the x axis, the second the xy plane.  Which two is
decided by the atom type, so that the same site of two molecules is posed the same way, and by
nothing else: a frame that depended on which of two ring bonds is a hundredth of an angstrom
longer would not be a rule.  Neighbours of the same type are therefore interchangeable, and when
that leaves the choice open, as it does for the two ring neighbours of an aromatic carbon or the
three hydrogens of a methyl group, every choice is kept and
the environment becomes the **mean of those poses**, which is a mean-map kernel over the frames
the site allows (:func:`cmfqsar.fields.combine_members`).  That is the same device the symmetrised
Hammett set uses for the four images of a monosubstituted benzene: a frame ambiguity is averaged
over rather than resolved by a coin toss.

**The ring site.**  :func:`ring_frames` is the frame for the site the plan names, an aromatic
carbon of a six-membered carbocycle: the carbon at the origin, the bond that leaves the ring on
+x, the ring in the xy plane, and the two mirror images kept.  In that frame the positions of the
ortho, meta and para substituents of the ring are fixed numbers (:data:`RING_POSITIONS`), so a
co-field can be read at them and compared with what substituent effects on an aromatic carbon are
known to be.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .atomtargets import AtomTargets
from .fields import MoleculeFields, combine_members
from .localenv import DEFAULT_RHO, ENVELOPE_FLOOR, environment

#: poses kept at most when the key cannot separate the neighbours
MAX_IMAGES = 6
#: benzene geometry, in angstrom: the circumradius of the ring and a carbon-substituent bond
RING_RADIUS = 1.397
SUBSTITUENT_BOND = 1.40
#: how far from the centre a bonded neighbour can be
BOND_MAX = 2.1


def _neighbours(env: MoleculeFields, centre: int = 0) -> List[int]:
    """The atoms bonded to the centre, from the bonds the environment carries."""
    bonds = np.asarray(env.bonds, int).reshape(-1, 2)
    out = []
    for a, b in bonds:
        if a == centre:
            out.append(int(b))
        elif b == centre:
            out.append(int(a))
    if not out:                      # an environment built without bonds: fall back to distance
        d = np.linalg.norm(np.asarray(env.coords, float) - env.coords[centre], axis=1)
        out = [i for i in np.argsort(d) if i != centre and d[i] <= BOND_MAX]
    return sorted(set(out))


def _group(env: MoleculeFields, i: int) -> str:
    """What separates two neighbours: the atom type, and nothing else.

    A bond length is not a reliable way to tell two neighbours apart -- the two ring bonds of an
    aromatic carbon differ by hundredths of an angstrom, and by which of them is longer in one
    conformation -- so a frame that used it would not be a rule.  Neighbours of the same type are
    therefore treated as interchangeable, and every choice between them becomes a pose to average
    over.
    """
    return str(env.atom_types[i])


def _order(env: MoleculeFields, i: int, centre: int = 0) -> Tuple:
    """A reproducible order inside a group of interchangeable neighbours."""
    d = float(np.linalg.norm(np.asarray(env.coords, float)[i] - env.coords[centre]))
    return (_group(env, i), round(d, 3), int(i))


def frame_from(coords: np.ndarray, i: int, j: int, centre: int = 0) -> Optional[np.ndarray]:
    """The rotation whose rows are the x, y and z axes fixed by atoms `i` and `j`.

    x is along the centre-to-`i` direction, y is what is left of the centre-to-`j` direction after
    x is taken out, and z completes a right-handed set.  Returns None when the three atoms are
    collinear, where no frame exists.
    """
    r = np.asarray(coords, float)
    x = r[i] - r[centre]
    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        return None
    x = x / nx
    v = r[j] - r[centre]
    y = v - float(v @ x) * x
    ny = float(np.linalg.norm(y))
    if ny < 1e-6:
        return None
    y = y / ny
    return np.stack([x, y, np.cross(x, y)])


def local_frames(env: MoleculeFields, max_images: int = MAX_IMAGES,
                 centre: int = 0) -> List[np.ndarray]:
    """Every pose of this environment the key allows, at most `max_images` of them.

    One rotation is returned for a site the key separates, two for an aromatic carbon with two
    equivalent ring neighbours, three for a methyl carbon, and the identity for a centre with
    nothing bonded to it, where there is no frame to speak of.
    """
    nb = _neighbours(env, centre)
    if not nb:
        return [np.eye(3)]
    nb.sort(key=lambda i: _order(env, i, centre))
    first = [i for i in nb if _group(env, i) == _group(env, nb[0])]
    out: List[np.ndarray] = []
    for i in first:
        rest = [k for k in nb if k != i]
        if not rest:
            # one neighbour only: the nearest other atom has to fix the xy plane
            d = np.linalg.norm(np.asarray(env.coords, float) - env.coords[centre], axis=1)
            rest = [int(k) for k in np.argsort(d) if k not in (centre, i)][:1]
        if not rest:
            continue
        rest.sort(key=lambda k: _order(env, k, centre))
        for j in [k for k in rest if _group(env, k) == _group(env, rest[0])]:
            R = frame_from(env.coords, i, j, centre)
            if R is not None:
                out.append(R)
            if len(out) >= int(max_images):
                return out
    return out or [np.eye(3)]


def ring_frames(env: MoleculeFields, centre: int = 0) -> List[np.ndarray]:
    """The two poses of an aromatic ring carbon: the bond leaving the ring on +x, ring in xy.

    This is the site of plan item LP3 and the local analogue of the frame the Hammett set uses for
    a whole molecule.  x points away from the ring, so a substituent on the centre sits on +x at
    about 1.4 A; the ring is in the xy plane with its two neighbours at about +-120 degrees; z is
    the ring normal.  The two poses are mirror images in y, which is the ambiguity of which ring
    neighbour is called the first one, and averaging over them is what makes the pose a rule
    rather than a choice.

    Returns an empty list when the centre is not an aromatic carbon with exactly two aromatic
    neighbours, which is what a fused-ring carbon and a non-aromatic centre are.  A five-membered
    aromatic ring passes as well, and the frame is as good there, but :data:`RING_POSITIONS` is
    the geometry of a six-membered ring and does not apply to it.
    """
    nb = _neighbours(env, centre)
    ar = [i for i in nb if str(env.atom_types[i]).startswith(("C.ar", "N.ar"))]
    if not str(env.atom_types[centre]).startswith(("C.ar", "N.ar")) or len(ar) != 2:
        return []
    r = np.asarray(env.coords, float)
    u = [(r[i] - r[centre]) / max(float(np.linalg.norm(r[i] - r[centre])), 1e-12) for i in ar]
    inward = u[0] + u[1]
    n = float(np.linalg.norm(inward))
    if n < 1e-6:
        return []
    x = -inward / n                                  # away from the ring
    side = u[0] - u[1]
    y = side - float(side @ x) * x
    ny = float(np.linalg.norm(y))
    if ny < 1e-6:
        return []
    y = y / ny
    return [np.stack([x, y, np.cross(x, y)]), np.stack([x, -y, np.cross(x, -y)])]


def ring_positions(radius: float = RING_RADIUS,
                   bond: float = SUBSTITUENT_BOND) -> Dict[str, np.ndarray]:
    """Where a substituent sits, in the frame of :func:`ring_frames`, by ring position.

    The centre is a carbon of a regular six-membered ring of circumradius `radius`, so the ring
    centroid is at (-radius, 0, 0) and the other five carbons follow from the hexagon.  Each
    position is the place the first atom of a substituent on that carbon occupies, `bond` angstrom
    out from it along the direction away from the centroid.  The y > 0 member of each pair is
    given; the two poses of :func:`ring_frames` are mirror images in y, so the co-field of a set
    posed that way is symmetric in y and one of each pair is enough.
    """
    out = {}
    centroid = np.array([-radius, 0.0, 0.0])
    for name, theta in (("ipso", 0.0), ("ortho", 60.0), ("meta", 120.0), ("para", 180.0)):
        t = np.radians(theta)
        u = np.array([np.cos(t), np.sin(t), 0.0])
        out[name] = centroid + (radius + bond) * u
    return out


#: the four substituent positions of a benzene ring in the frame of :func:`ring_frames`
RING_POSITIONS = ring_positions()


def pose(env: MoleculeFields, R: np.ndarray, name: str = "") -> MoleculeFields:
    """The same environment with its coordinates turned into the frame `R`."""
    out = MoleculeFields(**{k: v for k, v in vars(env).items()})
    out.coords = np.asarray(env.coords, float) @ np.asarray(R, float).T
    out.name = name or env.name
    return out


def posed(env: MoleculeFields, max_images: int = MAX_IMAGES,
          frames: Optional[Sequence[np.ndarray]] = None, name: str = "") -> MoleculeFields:
    """One field: the environment in every pose its site allows, averaged.

    With one pose the result is that pose.  With several it is their mean field, which is a
    mean-map kernel over the frames the site allows and removes the arbitrary part of the rule
    instead of hiding it.
    """
    rots = list(frames) if frames is not None else local_frames(env, max_images)
    if not rots:
        raise ValueError(f"{env.name or 'this environment'} has no frame: its centre has fewer "
                         f"than two neighbours to fix one")
    poses = [pose(env, R, name=f"{env.name}#{k}") for k, R in enumerate(rots)]
    if len(poses) == 1:
        poses[0].name = name or env.name
        return poses[0]
    out = combine_members(poses, np.ones(len(poses)), kind="conformers",
                          name=name or env.name)
    return out


def posed_ensemble(members: Sequence[MoleculeFields], atom: int, rho: float = DEFAULT_RHO,
                   floor: float = ENVELOPE_FLOOR, weights: Optional[Sequence[float]] = None,
                   max_images: int = MAX_IMAGES,
                   frames_of: Optional[Callable[[MoleculeFields], List[np.ndarray]]] = None,
                   name: str = "") -> MoleculeFields:
    """One field: an atom's environment in every conformation, each posed in its own frame.

    The frame is a property of the site, and the site moves with the conformation, so the pose
    has to be taken per conformation and the averaging has to come after it.  Posing the mean
    field instead would put one rotation on geometries that do not share it.

    The result is the mean over conformations and over the poses each site allows, which is one
    ensemble field: the two averages compose, because a mean of means over a shared atom list is
    a mean (:func:`cmfqsar.fields.combine_members`).
    """
    from .localenv import kept_atoms

    if not members:
        raise ValueError("a posed ensemble needs at least one conformation")
    n = members[0].n_atoms
    if any(m.n_atoms != n for m in members):
        raise ValueError("the conformations of one molecule must have the same atoms")
    union = sorted(set().union(*[set(kept_atoms(m, atom, rho, floor).tolist()) for m in members]))
    envs = []
    for m in members:
        env = environment(m, atom, rho, floor, keep=union)
        frames = frames_of(env) if frames_of is not None else None
        envs.append(posed(env, max_images, frames))
    sizes = {e.n_atoms for e in envs}
    if len(sizes) != 1:
        raise ValueError("the conformations gave different numbers of poses, so their fields "
                         "cannot be averaged: " + ", ".join(str(s) for s in sorted(sizes)))
    if len(envs) == 1:
        envs[0].name = name or envs[0].name
        return envs[0]
    w = np.ones(len(envs)) if weights is None else np.asarray(weights, float)
    return combine_members(envs, w, kind="conformers",
                           name=name or f"{members[0].name or 'molecule'}:{int(atom)}")


def posed_environments(mols: Sequence[MoleculeFields], targets: AtomTargets,
                       rho: float = DEFAULT_RHO, floor: float = ENVELOPE_FLOOR,
                       max_images: int = MAX_IMAGES,
                       frames_of: Optional[Callable[[MoleculeFields], List[np.ndarray]]] = None,
                       progress=None) -> List[MoleculeFields]:
    """One posed environment per measured atom, in the row order of `targets`.

    `frames_of` replaces the general rule by a site rule -- :func:`ring_frames` for an aromatic
    carbon -- and is given the environment, whose centre is its first atom.
    """
    out: List[MoleculeFields] = []
    for r in range(len(targets)):
        k, i = int(targets.molecule[r]), int(targets.atom[r])
        env = environment(mols[k], i, rho, floor)
        frames = frames_of(env) if frames_of is not None else None
        out.append(posed(env, max_images, frames))
        if progress is not None and r % 200 == 0:
            progress(f"Posing environments {r + 1}/{len(targets)}", r / max(len(targets), 1))
    return out


def images_of(env: MoleculeFields, max_images: int = MAX_IMAGES,
              frames_of: Optional[Callable[[MoleculeFields], List[np.ndarray]]] = None) -> int:
    """How many poses the site of this environment allows: 1 when the key settles it."""
    rots = frames_of(env) if frames_of is not None else local_frames(env, max_images)
    return len(rots)


def well_posed(mols: Sequence[MoleculeFields], targets: AtomTargets, rho: float = DEFAULT_RHO,
               floor: float = ENVELOPE_FLOOR, max_images: int = MAX_IMAGES,
               frames_of: Optional[Callable[[MoleculeFields], List[np.ndarray]]] = None
               ) -> Dict[int, int]:
    """How many of the samples have how many poses: {poses: samples}.

    A site that gives one pose is settled by the key; two is the usual mirror pair; zero means the
    site rule refuses this centre, and such a sample has to be left out rather than posed anyway.
    """
    out: Dict[int, int] = {}
    for r in range(len(targets)):
        k, i = int(targets.molecule[r]), int(targets.atom[r])
        env = environment(mols[k], i, rho, floor)
        n = images_of(env, max_images, frames_of)
        out[n] = out.get(n, 0) + 1
    return out


def overlap_kernel(posed_envs: Sequence[MoleculeFields], fieldset, y=None, progress=None):
    """The overlap kernel of a set of posed environments, and the model that holds it.

    The environments are in one frame, so the ordinary CMF overlap kernel applies to them exactly
    as it applies to molecules: this builds the :class:`~cmfqsar.fields.AtomTable` and returns the
    fitted-kernel model together with the joint training kernel, which
    :func:`cmfqsar.localenv.cross_validate` then runs over folds that hold out whole molecules.
    """
    from .fields import AtomTable
    from .model import CMFModel

    table = AtomTable.from_molecules(list(posed_envs))
    model = CMFModel(fieldset=fieldset, method="krr", fit_bias=True)
    K = model.training_kernel(table, y, progress=progress)
    return model, table, K
