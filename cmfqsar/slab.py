"""
A metal slab as the receptor (plan item EC6).

:mod:`cmfqsar.protein` reads a protein and :mod:`cmfqsar.complementarity` scores a molecule
against it, residue by residue.  An electrode is the same problem with a different receptor: a
slab, periodic in the two directions of the surface, with no residues and no side chains.  This
module supplies the three pieces that item EC6 asks for.

**The receptor.**  :func:`slab_receptor` turns a slab into a
:class:`cmfqsar.fields.MoleculeFields`, which is what everything downstream already takes: the
atoms of its top layers, together with their periodic images out to a cutoff, so that a molecule
near the edge of the cell sees the surface continue rather than end.  A slab is read from a POSCAR
or CIF with :func:`read_slab`, or built with :func:`bcc110_slab`.

**What a clean metal surface actually offers, which is less than a protein does.**  The five
physicochemical fields of this program describe an organic molecule.  On a one-element metal slab
three of them are identically zero and the reason is not a gap in the parameters:

* the **electrostatic** field is zero because every atom of a one-element cell has the same
  electronegativity, so every charge model this program has gives it zero;
* the **hydrophobic** field and both **hydrogen-bond** fields are zero because the published
  increments are fitted to organic fragments and a metal carries none (plan item EC4 made that
  explicit rather than leaving a default in place);
* the **steric** field is real, from the van der Waals parameters item EC4 added.

Of the four complementary pairs of :mod:`cmfqsar.complementarity`, therefore, *all four* are
identically zero against a clean metal slab, and what is left is burial — the steric overlap.  So
a complementarity score of a molecule against a metal surface, built from the fields the program
has, is a shape score and nothing more.  That is stated here rather than discovered later.

**What a metal offers instead: its image charge.**  A conductor responds to a molecule's own
charges, and that response is the physics the fields miss.  :func:`image_charge_energy` computes
it in closed form: the interaction of the charge distribution with its mirror image through the
surface plane, ``-(1/2) sum_ij q_i q_j / |r_i - r_j*|``, which for one charge at height h is the
textbook ``-q^2 / 4h``.  It needs no parameters beyond the plane, and it is the term that makes a
polar molecule prefer a metal.

One property of it is worth stating because it decides what a pose search can do: the image
energy of a charge distribution above an infinite plane is **invariant to translation in the plane
and to rotation about the normal**.  It can set a molecule's height and its tilt and it cannot say
where on the lattice the molecule sits.  Only the shape term can do that.

**The pose.**  :func:`dock_on_surface` optimises the four degrees of freedom a surface leaves —
height, tilt, rotation about the normal, translation in the plane — instead of the six of a free
rigid body, with the tilt bounded so that a molecule that arrives lying flat is not turned over.
Item EC2 found that the face a planar molecule presents is a mirror no rule and no reference
decides; here it is a choice the caller makes with `both_faces`.

**The sites.**  :func:`surface_sites` lists the top, bridge and hollow sites of the top layer and
:func:`site_profile` reports how much of a molecule sits over each kind, which is the surface's
answer to a residue profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .atomtypes import vdw_params
from .fields import FieldSet, MoleculeFields

#: Coulomb constant in eV angstrom per squared elementary charge, so that an image energy of
#: charges in units of e at distances in angstrom comes out in eV
COULOMB_EV = 14.399645

#: how close to the plane a charge may be taken before the image energy is held constant.  The
#: image energy diverges as a charge approaches a conductor, which is an artefact of the point
#: charge and not physics; a molecule in contact sits about 2 angstrom above the top layer.
MIN_IMAGE_HEIGHT = 0.5

#: the fields a clean one-element metal slab carries a non-zero weight for
SLAB_NONZERO_FIELDS = ("steric",)


# --------------------------------------------------------------------------- #
# The slab
# --------------------------------------------------------------------------- #

def bcc110_slab(element: str = "Fe", nx: int = 6, ny: int = 6, layers: int = 4,
                a: float = 2.78, vacuum: float = 15.0):
    """A body-centred cubic (110) slab, in the frame this program uses for a surface.

    The defaults are the slab the Fe(110) reference poses of ``examples/corrosion_fe110`` were
    computed on: ``a = 2.78`` angstrom, the minimum of the UMA oc20 head for bcc iron rather than
    the experimental 2.87, a 6 x 6 orthogonal surface cell and four layers.  The slab is shifted
    so that its **top layer sits at z = 0**, which is the frame those poses are given in.

    Needs ASE.
    """
    from ase.build import bcc110
    atoms = bcc110(str(element), size=(int(nx), int(ny), int(layers)), a=float(a),
                   vacuum=float(vacuum) / 2.0, orthogonal=True)
    z = atoms.positions[:, 2]
    atoms.positions[:, 2] -= z.max()
    atoms.info.update(nx=int(nx), ny=int(ny), layers=int(layers), a=float(a),
                      vacuum=float(vacuum))
    return atoms


def read_slab(path: str, top_at_zero: bool = True):
    """A slab read from any file ASE understands (POSCAR, CIF, extended XYZ).

    With `top_at_zero` the structure is shifted along z so that its highest atom is at z = 0,
    which is the surface frame of this program: the normal along z and the top layer at the
    origin.
    """
    from ase.io import read
    atoms = read(path)
    if top_at_zero and len(atoms):
        atoms.positions[:, 2] -= atoms.positions[:, 2].max()
    return atoms


def layer_heights(atoms, tolerance: float = 0.5) -> List[float]:
    """The distinct layer heights of a slab, highest first.

    Two atoms are in the same layer when their heights differ by less than `tolerance`, which is
    far below a layer spacing (1.97 angstrom for bcc Fe(110) at a = 2.78) and far above the
    numerical spread of a flat layer.
    """
    z = np.sort(np.asarray(atoms.positions, float)[:, 2])[::-1]
    out: List[float] = []
    for value in z:
        if not out or abs(value - out[-1]) > float(tolerance):
            out.append(float(value))
    return out


def top_layer_indices(atoms, top_layers: int = 2, tolerance: float = 0.5) -> np.ndarray:
    """The atoms of the top `top_layers` layers of a slab."""
    heights = layer_heights(atoms, tolerance)[:max(int(top_layers), 1)]
    if not heights:
        return np.zeros(0, dtype=int)
    cut = heights[-1] - float(tolerance)
    return np.flatnonzero(np.asarray(atoms.positions, float)[:, 2] >= cut)


def slab_images(atoms, top_layers: int = 2, cutoff: float = 10.0,
                tolerance: float = 0.5) -> Tuple[np.ndarray, List[str]]:
    """The top layers of a slab repeated over the surface, as plain coordinates and symbols.

    Returns ``(coords, symbols)``.  Only the two surface directions are repeated, far enough
    that every point of the primary surface cell has the surface continuing for at least
    `cutoff` angstrom around it; a slab is not periodic along its normal, and its images along
    the normal are vacuum.

    This is the expansion :func:`slab_receptor` turns into fields and the one the application
    draws, so the picture shows exactly the atoms the receptor is made of.
    """
    pos = np.asarray(atoms.positions, float)
    cell = np.asarray(atoms.get_cell(), float)
    idx = top_layer_indices(atoms, top_layers, tolerance)
    symbols = [atoms[i].symbol for i in idx]
    base = pos[idx]
    a1, a2 = cell[0], cell[1]
    span1 = max(float(np.linalg.norm(a1[:2])), 1e-6)
    span2 = max(float(np.linalg.norm(a2[:2])), 1e-6)
    n1 = int(np.ceil(float(cutoff) / span1))
    n2 = int(np.ceil(float(cutoff) / span2))
    coords, syms = [], []
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            coords.append(base + i * a1 + j * a2)
            syms.extend(symbols)
    return (np.vstack(coords) if coords else np.zeros((0, 3))), syms


def slab_receptor(atoms, fieldset: FieldSet, top_layers: int = 2, cutoff: float = 10.0,
                  tolerance: float = 0.5, name: str = "slab") -> MoleculeFields:
    """The top layers of a slab, with their periodic images, as fields.

    `cutoff` is how far beyond the primary surface cell the images are carried, in angstrom: a
    molecule anywhere in the cell then sees the surface continue for at least that far in every
    direction.  Only the two surface directions are repeated; a slab is not periodic along its
    normal and its images along z are vacuum.

    The weights are not perceived from a molecule, because there is no molecule: every field of
    the field set is zero except the steric one, which carries the van der Waals well depth of
    the element, and an element indicator field ``cif:<symbol>``, which carries one.  The
    docstring of this module says why the others are zero.
    """
    coords, syms = slab_images(atoms, top_layers=top_layers, cutoff=cutoff, tolerance=tolerance)
    names = [s.name for s in fieldset.specs]
    weights = np.zeros((len(coords), len(names)))
    radii = np.array([vdw_params(s)[0] for s in syms]) if syms else np.zeros(0)
    if "steric" in names:
        weights[:, names.index("steric")] = [vdw_params(s)[1] for s in syms]
    for k, nm in enumerate(names):
        if nm.startswith("cif:") and nm[4:] in set(syms):
            weights[:, k] = [1.0 if s == nm[4:] else 0.0 for s in syms]
    return MoleculeFields(name=name, coords=coords, weights=weights, radii=radii,
                          symbols=list(syms), atom_types=list(syms),
                          bonds=np.zeros((0, 2), dtype=int), charges=np.zeros(len(coords)),
                          charge_source="slab")


# --------------------------------------------------------------------------- #
# The image charge
# --------------------------------------------------------------------------- #

def image_charge_energy(coords: np.ndarray, charges: np.ndarray, plane_z: float = 0.0,
                        min_height: float = MIN_IMAGE_HEIGHT) -> float:
    """The interaction of a charge distribution with its image in a conducting plane, in eV.

    ``E = -(1/2) sum_i sum_j q_i q_j / |r_i - r_j*|`` with ``r*`` the mirror of ``r`` through the
    plane.  The one-half is the linear-response factor of an induced charge, and the diagonal term
    is a charge with its own image: for a single charge at height h it gives the textbook
    ``-q^2 / 4h``.  The result is negative (attractive) for any distribution, which is why a
    conductor attracts a neutral polar molecule as well as an ion.

    Heights are measured from `plane_z`, and a charge closer to it than `min_height` is held at
    that height: the point-charge expression diverges at contact, which is an artefact.
    """
    xyz = np.asarray(coords, float).reshape(-1, 3)
    q = np.asarray(charges, float).reshape(-1)
    if len(xyz) == 0 or not np.any(np.abs(q) > 1e-12):
        return 0.0
    h = np.maximum(xyz[:, 2] - float(plane_z), float(min_height))
    mirror = np.column_stack([xyz[:, 0], xyz[:, 1], float(plane_z) - h])
    real = np.column_stack([xyz[:, 0], xyz[:, 1], float(plane_z) + h])
    d = cdist(real, mirror)
    d = np.maximum(d, 2.0 * float(min_height))
    return float(-0.5 * COULOMB_EV * np.sum(np.outer(q, q) / d))


# --------------------------------------------------------------------------- #
# The pose
# --------------------------------------------------------------------------- #

def surface_transform(coords: np.ndarray, p: Sequence[float],
                      centre: Optional[np.ndarray] = None) -> np.ndarray:
    """Move a molecule by the four freedoms a surface leaves.

    ``p = (dx, dy, dz, psi, tilt, azimuth)``: a turn of `psi` about the surface normal, a tilt of
    `tilt` about the in-plane axis at angle `azimuth`, and a translation.  The tilt and the turn
    are applied about `centre`, which defaults to the centroid, so that a bounded tilt moves the
    molecule's own orientation and not its position.
    """
    xyz = np.asarray(coords, float)
    centre = xyz.mean(axis=0) if centre is None else np.asarray(centre, float)
    dx, dy, dz, psi, tilt, azimuth = [float(v) for v in p]
    axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    c, s = np.cos(tilt), np.sin(tilt)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    r_tilt = np.eye(3) + s * k + (1.0 - c) * (k @ k)
    cz, sz = np.cos(psi), np.sin(psi)
    r_psi = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    r = r_psi @ r_tilt
    return (xyz - centre) @ r.T + centre + np.array([dx, dy, dz])


@dataclass
class SurfaceDock:
    """What happened to one pose on the surface."""
    score_before: float
    score_after: float
    burial_before: float
    burial_after: float
    image_before: float
    image_after: float
    clash_after: float
    height_before: float
    height_after: float
    moved: float
    tilt_deg: float
    turn_deg: float
    flipped: bool = False
    starts_used: int = 0


class SurfaceScorer:
    """Scores a pose against a slab: burial, the image charge, and a clash penalty.

    `image_weight` is how many units of burial one electron volt of image attraction is worth.
    It defaults to zero, so that the pose a search finds is decided by shape alone: the image term
    is invariant to translation in the plane and to rotation about the normal, so it cannot place
    a molecule laterally, and letting it set the height while the shape sets the position mixes
    two things that are better measured apart.  The term is always reported.
    """

    def __init__(self, mol: MoleculeFields, slab: MoleculeFields, fieldset: FieldSet,
                 image_weight: float = 0.0, clash_weight: float = 1.0,
                 clash_tolerance: float = 0.4, plane_z: float = 0.0, cutoff: float = 12.0):
        names = [s.name for s in fieldset.specs]
        if "steric" not in names:
            raise ValueError("a surface score needs the steric field: it is the only field a "
                             "clean metal slab carries")
        self.f = names.index("steric")
        self.spec = fieldset.specs[self.f]
        self.mol = mol
        self.plane_z = float(plane_z)
        self.image_weight = float(image_weight)
        self.clash_weight = float(clash_weight)
        self.clash_tolerance = float(clash_tolerance)
        self.charges = np.asarray(mol.charges, float) if mol.charges is not None \
            else np.zeros(mol.n_atoms)
        if cutoff and cutoff > 0 and len(slab.coords):
            near = cKDTree(slab.coords).query_ball_point(mol.coords, r=float(cutoff))
            keep = np.unique(np.array([i for h in near for i in h], dtype=int)) \
                if any(len(h) for h in near) else np.arange(len(slab.coords))
            if len(keep) == 0:
                keep = np.arange(len(slab.coords))
        else:
            keep = np.arange(len(slab.coords))
        self.s_coords = slab.coords[keep]
        self.s_radii = slab.radii[keep]
        self.s_weights = slab.weights[keep, self.f]
        self.s_widths = self._widths(slab.radii[keep])
        self.m_widths = self._widths(mol.radii)
        self.m_weights = mol.weights[:, self.f]
        self.heavy = [i for i in range(mol.n_atoms) if mol.symbols[i] != "H"]
        self.m_radii = mol.radii[self.heavy]
        self.m_norm = np.sqrt(max(_overlap(mol.coords, self.m_weights, self.m_widths,
                                           mol.coords, self.m_weights, self.m_widths), 1e-300))
        self.s_norm = np.sqrt(max(_overlap(self.s_coords, self.s_weights, self.s_widths,
                                           self.s_coords, self.s_weights, self.s_widths),
                                  1e-300))

    def _widths(self, radii: np.ndarray) -> np.ndarray:
        if self.spec.kind == "gauss":
            return np.full(len(radii), float(self.spec.alpha))
        return float(self.spec.alpha) / np.asarray(radii, float) ** 2

    def burial(self, coords: np.ndarray) -> float:
        """The steric overlap of the molecule with the surface, as a cosine in [0, 1]."""
        value = _overlap(coords, self.m_weights, self.m_widths,
                         self.s_coords, self.s_weights, self.s_widths)
        return float(value / (self.m_norm * self.s_norm))

    def image(self, coords: np.ndarray) -> float:
        return image_charge_energy(coords, self.charges, self.plane_z)

    def clash(self, coords: np.ndarray) -> float:
        if not len(self.s_coords) or not self.heavy:
            return 0.0
        d = cdist(np.asarray(coords, float)[self.heavy], self.s_coords)
        lim = self.m_radii[:, None] + self.s_radii[None, :] - self.clash_tolerance
        over = np.maximum(lim - d, 0.0)
        return float((over * over).sum())

    def score(self, coords: np.ndarray) -> float:
        total = self.burial(coords) - self.clash_weight * self.clash(coords)
        if self.image_weight:
            total += self.image_weight * (-self.image(coords))
        return float(total)


def _overlap(xa, wa, ba, xb, wb, bb) -> float:
    """Integral over space of two sums of Gaussians, in closed form."""
    xa = np.asarray(xa, float)
    xb = np.asarray(xb, float)
    if len(xa) == 0 or len(xb) == 0:
        return 0.0
    d2 = cdist(xa, xb, "sqeuclidean")
    bs = np.asarray(ba, float)[:, None] + np.asarray(bb, float)[None, :]
    e = np.exp(-(np.asarray(ba, float)[:, None] * np.asarray(bb, float)[None, :]) / (2.0 * bs)
               * d2) * (2.0 * np.pi / bs) ** 1.5
    return float((np.asarray(wa, float)[:, None] * e * np.asarray(wb, float)[None, :]).sum())


def dock_on_surface(mol: MoleculeFields, slab: MoleculeFields, fieldset: FieldSet,
                    starts: int = 8, seed: int = 0, max_iter: int = 300,
                    max_shift: float = 3.0, max_tilt_deg: float = 25.0,
                    min_height: float = 1.2, max_height: float = 6.0,
                    image_weight: float = 0.0, clash_weight: float = 1.0,
                    plane_z: float = 0.0, both_faces: bool = False
                    ) -> Tuple[np.ndarray, SurfaceDock]:
    """Place one molecule on the surface, using the four freedoms a surface leaves.

    The search is over translation in the plane, height, rotation about the normal and a bounded
    tilt.  Rotation about the normal is the one freedom with no good starting guess -- a flat
    molecule may sit at any angle to the lattice -- so the starts are spread uniformly over it.
    The height is bounded between `min_height` and `max_height`, measured as the lowest heavy
    atom above the plane, which keeps the optimiser from either burying the molecule in the metal
    or letting it drift into the vacuum where every term is flat.

    `both_faces` tries the molecule the other way up as well and keeps the better score.  Item
    EC2 measured that the face is a mirror no rule decides, so whether to search it is the
    caller's decision and is reported in the result.
    """
    scorer = SurfaceScorer(mol, slab, fieldset, image_weight=image_weight,
                           clash_weight=clash_weight, plane_z=plane_z)
    centre = mol.coords.mean(axis=0)
    heavy = [i for i in range(mol.n_atoms) if mol.symbols[i] != "H"]
    floor = float(mol.coords[heavy, 2].min()) if heavy else float(mol.coords[:, 2].min())

    def place(p: Sequence[float], flip: bool) -> np.ndarray:
        xyz = mol.coords if not flip else _flip(mol.coords, centre)
        return surface_transform(xyz, p, centre)

    def objective(p, flip):
        coords = place(p, flip)
        low = coords[heavy, 2].min() if heavy else coords[:, 2].min()
        penalty = 0.0
        if low < min_height:
            penalty += 10.0 * (min_height - low) ** 2
        if low > max_height:
            penalty += 10.0 * (low - max_height) ** 2
        return -(scorer.score(coords) - penalty)

    tilt_max = np.deg2rad(float(max_tilt_deg))
    bounds = [(-max_shift, max_shift), (-max_shift, max_shift),
              (min_height - floor - 1.0, max_height - floor + 1.0),
              (-np.pi, np.pi), (0.0, tilt_max), (-np.pi, np.pi)]
    rng = np.random.RandomState(int(seed))
    p0s = [np.array([0.0, 0.0, 2.0 - floor, 0.0, 0.0, 0.0])]
    for k in range(max(int(starts) - 1, 0)):
        p0s.append(np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
                             2.0 - floor + rng.uniform(-0.5, 1.0),
                             -np.pi + 2.0 * np.pi * (k + 0.5) / max(int(starts) - 1, 1),
                             rng.uniform(0.0, tilt_max * 0.5), rng.uniform(-np.pi, np.pi)]))
    best = None
    for flip in ((False, True) if both_faces else (False,)):
        for k, p0 in enumerate(p0s):
            res = minimize(objective, np.clip(p0, [b[0] for b in bounds],
                                              [b[1] for b in bounds]),
                           args=(flip,), method="Powell", bounds=bounds,
                           options={"maxiter": max_iter, "xtol": 1e-2, "ftol": 1e-4})
            if best is None or res.fun < best[0]:
                best = (float(res.fun), np.asarray(res.x, float), flip, k)
    _value, p, flip, used = best
    coords = place(p, flip)
    start = place(np.zeros(6), False)
    low_before = float(start[heavy, 2].min()) if heavy else float(start[:, 2].min())
    low_after = float(coords[heavy, 2].min()) if heavy else float(coords[:, 2].min())
    return coords, SurfaceDock(
        score_before=scorer.score(mol.coords), score_after=scorer.score(coords),
        burial_before=scorer.burial(mol.coords), burial_after=scorer.burial(coords),
        image_before=scorer.image(mol.coords), image_after=scorer.image(coords),
        clash_after=scorer.clash(coords),
        height_before=low_before, height_after=low_after,
        moved=float(np.sqrt(((coords - mol.coords) ** 2).sum(axis=1).mean())),
        tilt_deg=float(np.rad2deg(p[4])), turn_deg=float(np.rad2deg(p[3])),
        flipped=bool(flip), starts_used=int(used))


def _flip(coords: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """The molecule turned over: a half turn about the x axis through `centre`.

    A rotation and not a reflection, so the molecule stays itself and only the face it presents
    to the surface changes.
    """
    xyz = np.asarray(coords, float) - np.asarray(centre, float)
    xyz = xyz @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).T
    return xyz + np.asarray(centre, float)


# --------------------------------------------------------------------------- #
# The score and the sites
# --------------------------------------------------------------------------- #

@dataclass
class SurfaceScore:
    """What a molecule's placement on a surface is worth, term by term."""
    name: str
    burial: float = float("nan")
    image_eV: float = float("nan")
    clash: float = 0.0
    height: float = float("nan")
    sites: Dict[str, float] = _field(default_factory=dict)

    def describe(self) -> str:
        s = (f"{self.name}: burial {self.burial:.3f}, image {self.image_eV:+.3f} eV, "
             f"lowest heavy atom {self.height:.2f} A")
        if self.clash:
            s += f", clash {self.clash:.2f}"
        if self.sites:
            s += "; over " + ", ".join(f"{k} {v:.2f}" for k, v in self.sites.items())
        return s


def surface_sites(atoms, tolerance: float = 0.5, height: float = 0.0):
    """The top, bridge and hollow sites of the top layer, as labels and positions.

    A top site is above a surface atom, a bridge site above the midpoint of two neighbours, and a
    hollow site above the centre of a triangle of them.  The neighbours and the triangles come
    from a Delaunay triangulation of the top layer in the plane, so the three kinds are found for
    any surface and not only for the one lattice this module can build.
    """
    from scipy.spatial import Delaunay
    idx = top_layer_indices(atoms, 1, tolerance)
    xy = np.asarray(atoms.positions, float)[idx][:, :2]
    labels: List[str] = []
    points: List[np.ndarray] = []
    for p in xy:
        labels.append("top")
        points.append(np.array([p[0], p[1], height]))
    if len(xy) >= 4:
        tri = Delaunay(xy)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                edges.add((min(a, b), max(a, b)))
            centre = xy[simplex].mean(axis=0)
            labels.append("hollow")
            points.append(np.array([centre[0], centre[1], height]))
        for a, b in sorted(edges):
            mid = 0.5 * (xy[a] + xy[b])
            labels.append("bridge")
            points.append(np.array([mid[0], mid[1], height]))
    return labels, (np.vstack(points) if points else np.zeros((0, 3)))


def site_profile(coords: np.ndarray, radii: np.ndarray, labels: Sequence[str],
                 points: np.ndarray, alpha: float = 1.0) -> Dict[str, float]:
    """How much of a molecule sits over each kind of surface site.

    The molecule's steric field is evaluated at every site position and the values are summed by
    kind and divided by the number of sites of that kind, so the three numbers are comparable
    whatever the lattice offers.  This is the surface's answer to a residue profile: it says
    whether a molecule sits over the metal atoms or between them.
    """
    xyz = np.asarray(coords, float)
    if len(xyz) == 0 or len(points) == 0:
        return {}
    widths = float(alpha) / np.asarray(radii, float) ** 2
    d2 = cdist(np.asarray(points, float), xyz, "sqeuclidean")
    value = np.exp(-0.5 * widths[None, :] * d2).sum(axis=1)
    out: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for label, v in zip(labels, value):
        out[label] = out.get(label, 0.0) + float(v)
        counts[label] = counts.get(label, 0) + 1
    return {k: out[k] / max(counts[k], 1) for k in sorted(out)}


def surface_score(mol: MoleculeFields, slab: MoleculeFields, fieldset: FieldSet,
                  plane_z: float = 0.0, sites=None, alpha: float = 1.0) -> SurfaceScore:
    """Score one placed molecule against the surface, term by term.

    `sites` is the pair that :func:`surface_sites` returns; when it is given, the score carries
    the site profile as well.
    """
    scorer = SurfaceScorer(mol, slab, fieldset, plane_z=plane_z)
    heavy = [i for i in range(mol.n_atoms) if mol.symbols[i] != "H"]
    height = float(mol.coords[heavy, 2].min()) if heavy else float(mol.coords[:, 2].min())
    profile = {}
    if sites is not None:
        labels, points = sites
        profile = site_profile(mol.coords, mol.radii, labels, points, alpha=alpha)
    return SurfaceScore(name=mol.name, burial=scorer.burial(mol.coords),
                        image_eV=scorer.image(mol.coords), clash=scorer.clash(mol.coords),
                        height=height - float(plane_z), sites=profile)


__all__ = ["COULOMB_EV", "MIN_IMAGE_HEIGHT", "SLAB_NONZERO_FIELDS", "SurfaceDock",
           "SurfaceScore", "SurfaceScorer", "bcc110_slab", "dock_on_surface",
           "image_charge_energy", "layer_heights", "read_slab", "site_profile",
           "slab_images", "slab_receptor", "surface_score", "surface_sites",
           "surface_transform", "top_layer_indices"]
