"""
A surface behind the molecules, in the application (plan item EC10).

The electrochemistry items built three things the window could not show:

* **the slab** (item EC6).  :mod:`cmfqsar.slab` turns the top layers of a metal slab, with their
  periodic images, into the same object a protein pocket becomes, and
  :mod:`cmfqsar.complementarity` scores a molecule against it.  The viewer already draws a
  pocket behind a ligand; a slab is drawn the same way, with one difference that matters: a slab
  has no edge, so what is drawn is the expansion around the molecules and not one cell.
* **the surface frame** (item EC2).  :mod:`cmfqsar.planealign` poses a molecule on the plane
  ``z = 0`` with the surface normal as z.  The frame is a convention, and a convention that is
  not drawn is one the reader has to take on trust, so the axes are drawn and the two numbers
  the frame is made of -- the height and the tilt -- are stated.
* **the vector co-field** (item EC5).  The co-field of the electric-field kind is a direction at
  a place rather than a value, so it is drawn as arrows; that part lives in
  :meth:`cmfqsar.gui.viewer3d.Viewer3D.add_arrows` and in the main window.

This module is the first two: reading a surface, choosing which of its atoms to draw, and saying
in one line what is on screen.  Everything here is a plain function or a small dataclass, so the
tests and ``benchmarks/run_surface_view.py`` drive it without a window.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: how far beyond the drawn region the surface is expanded before the atoms are cut to it, so
#: that the cut never leaves a straight edge where the expansion happened to stop
EXPANSION_PAD = 4.0
#: the slab of the Fe(110) reference poses of ``examples/corrosion_fe110``: a = 2.78 angstrom
#: (the minimum of the UMA oc20 head for bcc iron), a 6 x 6 orthogonal surface cell, four layers
FE110 = dict(element="Fe", nx=6, ny=6, layers=4, a=2.78, vacuum=15.0)


@dataclass
class Surface:
    """A slab in the surface frame: the normal along z, the top atomic layer at z = 0."""
    name: str
    atoms: object                       # an ASE Atoms object
    source: str = ""                    # the file it was read from, or how it was built
    layers: List[float] = None          # the distinct layer heights, highest first

    def __post_init__(self):
        if self.layers is None:
            from ..slab import layer_heights
            self.layers = [float(h) for h in layer_heights(self.atoms)]

    @property
    def n_atoms(self) -> int:
        return int(len(self.atoms))

    @property
    def elements(self) -> List[str]:
        return sorted({a.symbol for a in self.atoms})

    @property
    def cell2d(self) -> np.ndarray:
        """The two surface lattice vectors, as a 2 x 3 array."""
        return np.asarray(self.atoms.get_cell(), float)[:2]

    @property
    def spacing(self) -> float:
        """The distance between the first two layers, in angstrom; NaN for a single layer."""
        if len(self.layers) < 2:
            return float("nan")
        return float(self.layers[0] - self.layers[1])

    def describe(self) -> str:
        a, b = (float(np.linalg.norm(v)) for v in self.cell2d)
        s = (f"{self.name}: {self.n_atoms} atoms of {'/'.join(self.elements)} in "
             f"{len(self.layers)} layers, surface cell {a:.2f} x {b:.2f} A, top layer at z = 0")
        if np.isfinite(self.spacing):
            s += f", layer spacing {self.spacing:.2f} A"
        return s


# --------------------------------------------------------------------------- #
# Reading a surface
# --------------------------------------------------------------------------- #
def load_surface(path: str, top_at_zero: bool = True) -> Surface:
    """Read a slab from any file ASE understands (POSCAR, CIF, extended XYZ) into the frame.

    With `top_at_zero` the structure is shifted along z so that its highest atom sits at z = 0,
    which is the frame the reference poses of ``examples/corrosion_fe110`` are given in and the
    frame :mod:`cmfqsar.planealign` poses a molecule in.
    """
    from ..slab import read_slab
    atoms = read_slab(path, top_at_zero=top_at_zero)
    return Surface(name=os.path.basename(path), atoms=atoms, source=path)


def build_bcc110(element: str = "Fe", nx: int = 6, ny: int = 6, layers: int = 4,
                 a: float = 2.78, vacuum: float = 15.0) -> Surface:
    """A body-centred cubic (110) slab, built rather than read; the Fe(110) defaults are FE110.

    ASE needs an even number of repeats along the second surface direction for an orthogonal
    cell, so `ny` is rounded up to the next even number and the name says what was built.
    """
    from ..slab import bcc110_slab
    ny = int(ny) + (int(ny) % 2)
    atoms = bcc110_slab(element=element, nx=int(nx), ny=ny, layers=int(layers), a=float(a),
                        vacuum=float(vacuum))
    name = f"{element}(110) {int(nx)}x{ny}x{int(layers)}, a = {float(a):g} A"
    return Surface(name=name, atoms=atoms, source="built by cmfqsar.slab.bcc110_slab")


# --------------------------------------------------------------------------- #
# Which atoms to draw
# --------------------------------------------------------------------------- #
def drawn_atoms(surface: Surface, points: np.ndarray, radius: float = 8.0,
                top_layers: int = 2, max_atoms: int = 20_000
                ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """The surface atoms worth drawing around `points`, as ``(coords, symbols, radii)``.

    The top `top_layers` layers are repeated over the surface lattice and then cut to the atoms
    whose **in-plane** distance to the nearest of `points` is at most `radius`.  The cut is in
    the plane and not in space, so the surface stays a surface under a molecule that sits high
    above it instead of thinning out to nothing.

    `radius` is the only control a user needs: a small radius shows the patch under the molecule,
    a large one shows that the surface has no edge.  Beyond `max_atoms` the drawing would be slow
    with no gain, so the radius is reduced until the count fits and the caller is told by the
    number of atoms it gets back.
    """
    from scipy.spatial import cKDTree

    from ..atomtypes import vdw_params
    from ..slab import slab_images
    pts = np.asarray(points, float).reshape(-1, 3)
    r = float(radius)
    if not len(pts) or r <= 0:
        return np.zeros((0, 3)), [], np.zeros(0)
    while True:
        span = float(np.ptp(pts[:, :2], axis=0).max()) if len(pts) > 1 else 0.0
        coords, symbols = slab_images(surface.atoms, top_layers=top_layers,
                                      cutoff=r + span + EXPANSION_PAD)
        if not len(coords):
            return np.zeros((0, 3)), [], np.zeros(0)
        d, _ = cKDTree(pts[:, :2]).query(coords[:, :2], k=1)
        keep = np.flatnonzero(d <= r)
        if len(keep) <= int(max_atoms) or r <= 2.0:
            break
        r *= 0.7
    out = coords[keep]
    syms = [symbols[int(i)] for i in keep]
    radii = np.array([vdw_params(s)[0] for s in syms]) if syms else np.zeros(0)
    return out, syms, radii


def site_points(surface: Surface, points: np.ndarray, radius: float = 8.0,
                height: float = 0.0) -> Tuple[List[str], np.ndarray]:
    """The top, bridge and hollow sites of the top layer near `points`.

    :func:`cmfqsar.slab.surface_sites` finds them in the primary surface cell; a molecule is
    rarely in that cell, so the sites are translated over the surface lattice to the molecules
    and then cut to `radius` in the plane, exactly as :func:`drawn_atoms` cuts the atoms.
    """
    from scipy.spatial import cKDTree

    from ..slab import surface_sites
    labels, base = surface_sites(surface.atoms, height=float(height))
    pts = np.asarray(points, float).reshape(-1, 3)
    if not len(base) or not len(pts):
        return [], np.zeros((0, 3))
    a1, a2 = surface.cell2d[0], surface.cell2d[1]
    span = float(np.ptp(pts[:, :2], axis=0).max()) if len(pts) > 1 else 0.0
    reach = float(radius) + span + EXPANSION_PAD
    n1 = int(np.ceil(reach / max(float(np.linalg.norm(a1[:2])), 1e-6)))
    n2 = int(np.ceil(reach / max(float(np.linalg.norm(a2[:2])), 1e-6)))
    all_pts, all_labels = [], []
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            shift = i * a1 + j * a2
            shifted = base + shift
            shifted[:, 2] = float(height)
            all_pts.append(shifted)
            all_labels.extend(labels)
    coords = np.vstack(all_pts)
    d, _ = cKDTree(pts[:, :2]).query(coords[:, :2], k=1)
    keep = np.flatnonzero(d <= float(radius))
    return [all_labels[int(i)] for i in keep], coords[keep]


# --------------------------------------------------------------------------- #
# What the panel says about a pose
# --------------------------------------------------------------------------- #
def pose_line(coords: np.ndarray, symbols: Sequence[str], name: str = "",
              plane_z: float = 0.0) -> str:
    """One line for the panel: the height and the tilt of a molecule in the surface frame."""
    from ..planealign import pose_geometry
    g = pose_geometry(coords, symbols, plane_z=plane_z)
    head = f"{name}: " if name else ""
    return head + g.describe()


def pose_table(coords: Sequence[np.ndarray], symbols: Sequence[Sequence[str]],
               names: Sequence[str], plane_z: float = 0.0
               ) -> Tuple[List[str], List[List[str]]]:
    """The pose of every molecule of a set, as the column titles and one row each."""
    from ..planealign import pose_geometry
    cols = ["Molecule", "Lowest heavy atom (A)", "Centroid height (A)", "Tilt (degrees)",
            "Thickness (A)", "Flat"]
    rows = []
    for xyz, sym, nm in zip(coords, symbols, names):
        g = pose_geometry(xyz, sym, plane_z=plane_z)
        rows.append([str(nm), f"{g.height:.2f}", f"{g.centre_height:.2f}", f"{g.tilt:.1f}",
                     f"{g.thickness:.2f}", "yes" if g.flat else "no"])
    return cols, rows


__all__ = ["EXPANSION_PAD", "FE110", "Surface", "build_bcc110", "drawn_atoms", "load_surface",
           "pose_line", "pose_table", "site_points"]
