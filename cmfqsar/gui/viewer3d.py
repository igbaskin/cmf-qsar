"""
OpenGL 3D viewer (pyqtgraph.opengl) for molecules, co-field isosurfaces,
molecular-field meshes and atom-contribution spheres.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import MeshData
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QWidget, QVBoxLayout

from ..fields import MoleculeFields

CPK = {
    "C": (0.45, 0.45, 0.45), "H": (0.90, 0.90, 0.90), "N": (0.20, 0.30, 0.95), "O": (0.95, 0.15, 0.15),
    "S": (0.95, 0.85, 0.15), "F": (0.40, 0.90, 0.40), "Cl": (0.15, 0.80, 0.15), "Br": (0.65, 0.20, 0.10),
    "I": (0.55, 0.10, 0.65), "P": (1.00, 0.55, 0.10), "Si": (0.70, 0.60, 0.80), "B": (1.0, 0.7, 0.7),
    "Se": (0.9, 0.6, 0.1), "X": (0.6, 0.2, 0.8),
}
ATOM_RADIUS = {"H": 0.25, "C": 0.36, "N": 0.34, "O": 0.33, "S": 0.40, "P": 0.42, "F": 0.32,
               "Cl": 0.40, "Br": 0.44, "I": 0.48}

# positive / negative colours for co-field isosurfaces of each field type
COFIELD_COLORS = {
    "electrostatic": ((0.90, 0.15, 0.15), (0.15, 0.30, 0.95)),
    "steric": ((0.15, 0.80, 0.20), (0.95, 0.85, 0.10)),
    "hydrophobic": ((1.00, 0.55, 0.05), (0.45, 0.80, 1.00)),
    "hba": ((0.90, 0.20, 0.90), (0.55, 0.05, 0.10)),
    "hbd": ((0.10, 0.85, 0.90), (0.55, 0.20, 0.80)),
    "cif": ((0.90, 0.15, 0.15), (0.15, 0.30, 0.95)),
}


def cofield_colors(field_name: str):
    key = "cif" if field_name.startswith("cif:") else field_name
    return COFIELD_COLORS.get(key, ((0.9, 0.2, 0.2), (0.2, 0.3, 0.9)))


def color_name(c) -> str:
    """The name of one of the fixed co-field colours, for a legend that has to be read.

    A legend saying "positive = rgb(0.9,0.2,0.2)" tells the reader nothing; the colours of the
    co-fields are fixed, so each one has a name.
    """
    r, g, b = c[:3]
    names = {(0.90, 0.15, 0.15): "red", (0.15, 0.30, 0.95): "blue", (0.15, 0.80, 0.20): "green",
             (0.95, 0.85, 0.10): "yellow", (1.00, 0.55, 0.05): "orange", (0.45, 0.80, 1.00): "light blue",
             (0.90, 0.20, 0.90): "magenta", (0.55, 0.05, 0.10): "dark red", (0.10, 0.85, 0.90): "cyan",
             (0.55, 0.20, 0.80): "purple"}
    return names.get((r, g, b), f"rgb({r:.1f},{g:.1f},{b:.1f})")


#: colours already asked for, element by element, so that a crystal of a thousand atoms does not
#: look its colours up a thousand times
_ELEMENT_COLORS: Dict[str, Tuple[float, float, float]] = {}


def element_color(symbol: str, default=(0.6, 0.6, 0.6)) -> Tuple[float, float, float]:
    """The colour of an element: the table above where it has one, else the Jmol colour.

    The organic elements keep the colours the molecular views have always used, so that a
    molecule looks the same beside a crystal.  Everything else -- the metals, the alkali ions,
    the rare earths an inorganic structure is made of -- takes the standard Jmol colour from
    ASE's table, which covers every element.  An element neither table knows is grey.
    """
    s = (symbol or "").capitalize()
    if s in CPK:
        return CPK[s]
    if s in _ELEMENT_COLORS:
        return _ELEMENT_COLORS[s]
    try:
        from ase.data import atomic_numbers
        from ase.data.colors import jmol_colors
        col = tuple(float(v) for v in jmol_colors[atomic_numbers[s]])
    except Exception:                              # noqa: BLE001 - no ASE, or an unknown symbol
        col = tuple(default)
    _ELEMENT_COLORS[s] = col
    return col


#: radius of a drawn atom, as a fraction of its covalent radius
ATOM_RADIUS_SCALE = 0.45


def signed_spheres(values: Sequence[float], vmax: Optional[float] = None,
                   min_radius: float = 0.12, max_radius: float = 0.85,
                   positive=(0.95, 0.20, 0.20), negative=(0.20, 0.40, 0.95)):
    """Radii and colours for one sphere per signed value: size by size, colour by sign.

    Returns ``(radii, colours, vmax)``.  The radius grows as the square root of the magnitude,
    so that the *area* a sphere shows on screen is about proportional to the value; `vmax` is the
    value the largest sphere stands for and is returned so that a legend can state it.  This is
    the convention the atom contributions of a molecule are drawn with, used here for the
    per-site contributions of a cell.
    """
    v = np.asarray(values, float)
    m = float(vmax) if vmax else float(np.abs(v[np.isfinite(v)]).max() if len(v) else 0.0)
    if not np.isfinite(m) or m <= 0:
        return np.zeros(len(v)), [positive] * len(v), 0.0
    radii = min_radius + (max_radius - min_radius) * np.sqrt(np.clip(np.abs(v) / m, 0.0, 1.0))
    cols = [positive if x > 0 else negative for x in v]
    return radii, cols, m


def _rotation_to(d: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping the z axis onto unit vector d."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def molecule_mesh(mol: MoleculeFields, style: str = "ballstick", carbon_color=None, alpha: float = 1.0):
    """Build (vertices, faces, colours) of a ball-and-stick / stick / spacefill molecule."""
    verts, faces, colors = [], [], []
    off = 0
    sphere = MeshData.sphere(rows=8, cols=14, radius=1.0)
    sv, sf = sphere.vertexes(), sphere.faces()
    cyl = MeshData.cylinder(rows=1, cols=12, radius=[1.0, 1.0], length=1.0)
    cv, cf = cyl.vertexes(), cyl.faces()

    def col(sym):
        if sym == "C" and carbon_color is not None:
            return carbon_color
        return CPK.get(sym, (0.6, 0.6, 0.6))

    bond_r = 0.10 if style == "ballstick" else 0.16
    if style != "sticks":
        for xyz, sym, rvdw in zip(mol.coords, mol.symbols, mol.radii):
            r = rvdw * 0.95 if style == "spacefill" else ATOM_RADIUS.get(sym, 0.38)
            verts.append(sv * r + xyz)
            faces.append(sf + off)
            colors.append(np.tile(np.array(col(sym) + (alpha,)), (len(sv), 1)))
            off += len(sv)
    if style != "spacefill":
        for a, b in mol.bonds:
            p1, p2 = mol.coords[a], mol.coords[b]
            d = p2 - p1
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            R = _rotation_to(d / L)
            mid = 0.5 * (p1 + p2)
            for start, end, sym in ((p1, mid, mol.symbols[a]), (mid, p2, mol.symbols[b])):
                seg = cv.copy()
                seg[:, :2] *= bond_r
                seg[:, 2] *= L / 2
                seg = seg @ R.T + start
                verts.append(seg)
                faces.append(cf + off)
                colors.append(np.tile(np.array(col(sym) + (alpha,)), (len(cv), 1)))
                off += len(cv)
            if style == "sticks":
                for p, sym in ((p1, mol.symbols[a]), (p2, mol.symbols[b])):
                    verts.append(sv * bond_r + p)
                    faces.append(sf + off)
                    colors.append(np.tile(np.array(col(sym) + (alpha,)), (len(sv), 1)))
                    off += len(sv)
    if not verts:
        return None
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(colors)


class Viewer3D(QWidget):
    """Widget wrapping a GLViewWidget with named groups of items."""
    atom_picked = Signal(int, int)   # (molecule index, atom index) - reserved

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = 30
        self.view.setBackgroundColor((20, 22, 28))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self._groups: Dict[str, List[gl.GLGraphicsItem.GLGraphicsItem]] = {}
        self._axes: Optional[gl.GLAxisItem] = None
        self.dark = True

    # ------------------------------------------------------------- groups
    def clear_group(self, name: str) -> None:
        for it in self._groups.pop(name, []):
            try:
                self.view.removeItem(it)
            except Exception:
                pass

    def clear_all(self) -> None:
        for name in list(self._groups):
            self.clear_group(name)

    def _add(self, name: str, item) -> None:
        self.view.addItem(item)
        self._groups.setdefault(name, []).append(item)

    def has_group(self, name: str) -> bool:
        return bool(self._groups.get(name))

    # ---------------------------------------------------------- molecules
    def add_molecule(self, mol: MoleculeFields, group: str, style: str = "ballstick",
                     carbon_color=None, alpha: float = 1.0) -> None:
        if style == "lines":
            self._add_molecule_lines(mol, group, carbon_color, alpha)
            return
        mesh = molecule_mesh(mol, style=style, carbon_color=carbon_color, alpha=alpha)
        if mesh is None:
            return
        v, f, c = mesh
        item = gl.GLMeshItem(vertexes=v, faces=f, vertexColors=c, smooth=True, shader="shaded",
                             drawEdges=False, glOptions="translucent" if alpha < 1 else "opaque")
        self._add(group, item)

    def _add_molecule_lines(self, mol: MoleculeFields, group: str, carbon_color=None, alpha: float = 1.0,
                            width: float = 2.0) -> None:
        """Wire-frame ('lines') representation: every bond as two half-segments coloured by atom."""
        def col(sym):
            if sym == "C" and carbon_color is not None:
                return carbon_color
            return CPK.get(sym, (0.6, 0.6, 0.6))

        pos, cols = [], []
        bonded = set()
        for a, b in mol.bonds:
            p1, p2 = mol.coords[a], mol.coords[b]
            mid = 0.5 * (p1 + p2)
            pos += [p1, mid, mid, p2]
            ca, cb = col(mol.symbols[a]) + (alpha,), col(mol.symbols[b]) + (alpha,)
            cols += [ca, ca, cb, cb]
            bonded.update((int(a), int(b)))
        # isolated atoms (ions, single-atom fragments): small crosses
        for i in range(mol.n_atoms):
            if i in bonded:
                continue
            c = col(mol.symbols[i]) + (alpha,)
            for d in np.eye(3) * 0.25:
                pos += [mol.coords[i] - d, mol.coords[i] + d]
                cols += [c, c]
        if not pos:
            return
        item = gl.GLLinePlotItem(pos=np.array(pos, dtype=float), color=np.array(cols, dtype=float),
                                 width=width, mode="lines", antialias=True,
                                 glOptions="translucent" if alpha < 1 else "opaque")
        self._add(group, item)

    def add_structure(self, coords: np.ndarray, bonds: np.ndarray, symbols: Sequence[str],
                      group: str, style: str = "lines", carbon_color=(0.55, 0.60, 0.68),
                      alpha: float = 1.0, width: float = 1.6, radius: float = 0.10) -> None:
        """Draw an arbitrary bonded structure, such as a protein binding site.

        Unlike :meth:`add_molecule` this takes plain arrays, because a crystal structure is not a
        sanitised molecule.  'lines' is a wire frame and is fast enough for thousands of atoms;
        'sticks' builds cylinders and should be kept for small sites.
        """
        coords = np.asarray(coords, float).reshape(-1, 3)
        bonds = np.asarray(bonds, int).reshape(-1, 2)

        def col(sym):
            s = (sym or "C").capitalize()
            if s == "C":
                return carbon_color
            return CPK.get(s, CPK.get(s.upper(), (0.6, 0.6, 0.6)))

        if style == "sticks" and len(bonds):
            verts, faces, cols = [], [], []
            off = 0
            cyl = MeshData.cylinder(rows=1, cols=8, radius=[radius, radius], length=1.0)
            cv, cf = cyl.vertexes(), cyl.faces()
            for a, b in bonds:
                p1, p2 = coords[a], coords[b]
                for start, end, sym in ((p1, 0.5 * (p1 + p2), symbols[a]),
                                        (0.5 * (p1 + p2), p2, symbols[b])):
                    d = end - start
                    L = float(np.linalg.norm(d))
                    if L < 1e-6:
                        continue
                    R = _rotation_to(d / L)
                    v = (cv * np.array([1.0, 1.0, L])) @ R.T + start
                    verts.append(v)
                    faces.append(cf + off)
                    cols.append(np.tile(np.array(col(sym) + (alpha,)), (len(cv), 1)))
                    off += len(cv)
            if verts:
                item = gl.GLMeshItem(vertexes=np.concatenate(verts), faces=np.concatenate(faces),
                                     vertexColors=np.concatenate(cols), smooth=True, shader="shaded",
                                     glOptions="translucent" if alpha < 1 else "opaque")
                self._add(group, item)
            return
        pos, cols = [], []
        bonded = set()
        for a, b in bonds:
            p1, p2 = coords[a], coords[b]
            mid = 0.5 * (p1 + p2)
            pos += [p1, mid, mid, p2]
            ca, cb = col(symbols[a]) + (alpha,), col(symbols[b]) + (alpha,)
            cols += [ca, ca, cb, cb]
            bonded.update((int(a), int(b)))
        for i in range(len(coords)):             # ions and other isolated atoms as small crosses
            if i in bonded:
                continue
            c = col(symbols[i]) + (alpha,)
            for d in np.eye(3) * 0.25:
                pos += [coords[i] - d, coords[i] + d]
                cols += [c, c]
        if not pos:
            return
        item = gl.GLLinePlotItem(pos=np.array(pos, float), color=np.array(cols, float),
                                 width=width, mode="lines", antialias=True,
                                 glOptions="translucent" if alpha < 1 else "opaque")
        self._add(group, item)

    # ------------------------------------------------------------- crystals
    def add_cell(self, cell: np.ndarray, group: str, origin=(0.0, 0.0, 0.0),
                 color=(0.70, 0.74, 0.82), alpha: float = 0.85, width: float = 1.6,
                 axes: bool = True) -> None:
        """The unit cell as the twelve edges of its box.

        With `axes`, the three edges that leave the origin are drawn in red, green and blue for
        the first, second and third lattice vector, which is the usual convention and says at a
        glance which way the cell is turned.  The box is the frame everything periodic is drawn
        in: the atoms inside it are one cell, and the images outside it are the same cell
        translated.
        """
        A = np.asarray(cell, float).reshape(3, 3)
        o = np.asarray(origin, float).reshape(3)
        corner = {(i, j, k): o + i * A[0] + j * A[1] + k * A[2]
                  for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        axis_colors = ((0.90, 0.25, 0.25), (0.30, 0.85, 0.30), (0.35, 0.50, 0.95))
        pos, cols = [], []
        for a in corner:
            for d in range(3):
                b = list(a)
                if b[d]:
                    continue
                b[d] = 1
                edge_from_origin = not any(a)
                c = (axis_colors[d] if (axes and edge_from_origin) else tuple(color[:3]))
                pos += [corner[a], corner[tuple(b)]]
                cols += [c + (alpha,), c + (alpha,)]
        item = gl.GLLinePlotItem(pos=np.array(pos, float), color=np.array(cols, float),
                                 width=width, mode="lines", antialias=True,
                                 glOptions="translucent")
        self._add(group, item)

    def add_atoms(self, coords: np.ndarray, symbols: Sequence[str], group: str,
                  radii: Optional[Sequence[float]] = None, scale: float = ATOM_RADIUS_SCALE,
                  alpha: float = 1.0) -> None:
        """One sphere per atom, coloured by element and sized by its covalent radius.

        A crystal has no bond table, so this draws the atoms alone.  `radii` are the covalent
        radii a structure carries; without them every atom gets the same size.
        """
        coords = np.asarray(coords, float).reshape(-1, 3)
        if not len(coords):
            return
        r = (np.asarray(radii, float) * scale if radii is not None
             else np.full(len(coords), 0.35))
        self.add_spheres(coords, r, [element_color(s) for s in symbols], group, alpha=alpha)

    def add_points(self, coords: np.ndarray, symbols: Sequence[str], group: str,
                   size: float = 6.0, alpha: float = 0.45) -> None:
        """One dot per atom, coloured by element: the cheap drawing for the periodic images.

        A cell with its images out to the kernel's cutoff is thousands of atoms, which is too
        many for spheres and exactly what a scatter item is for.  The images are drawn
        translucent so that the cell itself stays the thing one looks at.
        """
        coords = np.asarray(coords, float).reshape(-1, 3)
        if not len(coords):
            return
        cols = np.array([element_color(s) + (alpha,) for s in symbols], float)
        item = gl.GLScatterPlotItem(pos=coords, color=cols, size=float(size), pxMode=True)
        item.setGLOptions("translucent")
        self._add(group, item)

    def add_scalar_surface(self, verts: np.ndarray, faces: np.ndarray, values: np.ndarray,
                           group: str, alpha: float = 0.9, vmax: Optional[float] = None,
                           negative=(0.95, 0.25, 0.20), positive=(0.25, 0.85, 0.35),
                           neutral=(0.85, 0.85, 0.85)) -> None:
        """A surface coloured per vertex by a signed value, red through grey to green.

        Used for the complementarity of a molecule with the pocket, where the sign carries the
        meaning: green where the two fields complement each other, red where they conflict.
        """
        v = np.asarray(verts, float)
        f = np.asarray(faces, int)
        x = np.asarray(values, float)
        if len(v) == 0 or len(f) == 0:
            return
        m = float(vmax) if vmax else float(np.percentile(np.abs(x[np.isfinite(x)]), 95) or 1.0)
        t = np.clip(np.nan_to_num(x) / (m if m > 1e-12 else 1.0), -1.0, 1.0)
        cols = np.zeros((len(v), 4))
        up, dn, nu = np.array(positive), np.array(negative), np.array(neutral)
        pos = t >= 0
        cols[pos, :3] = nu + (up - nu) * t[pos][:, None]
        cols[~pos, :3] = nu + (dn - nu) * (-t[~pos])[:, None]
        cols[:, 3] = alpha
        item = gl.GLMeshItem(vertexes=v, faces=f, vertexColors=cols, smooth=True, shader="shaded",
                             glOptions="translucent" if alpha < 1 else "opaque")
        self._add(group, item)

    def add_labels(self, positions: np.ndarray, texts: Sequence[str], group: str,
                   color=(0.95, 0.95, 0.95), alpha: float = 0.95, size: int = 9,
                   offset: float = 0.0) -> None:
        """Draw short texts at the given points: atom codes, residue names and the like.

        One item per label, so this is meant for the tens of labels of a molecule or a binding
        site rather than for a whole structure.
        """
        from PySide6.QtGui import QColor, QFont
        pos = np.asarray(positions, float).reshape(-1, 3)
        if offset:
            pos = pos + offset
        qcol = QColor.fromRgbF(*(tuple(color[:3]) + (alpha,)))
        font = QFont()
        font.setPointSize(int(size))
        for p, t in zip(pos, texts):
            if not t:
                continue
            item = gl.GLTextItem(pos=p, text=str(t), color=qcol, font=font)
            item.setGLOptions("translucent")
            self._add(group, item)

    def add_spheres(self, centers: np.ndarray, radii: Sequence[float], colors, group: str,
                    alpha: float = 0.55) -> None:
        sphere = MeshData.sphere(rows=10, cols=18, radius=1.0)
        sv, sf = sphere.vertexes(), sphere.faces()
        verts, faces, cols = [], [], []
        off = 0
        for xyz, r, c in zip(centers, radii, colors):
            if r <= 1e-4:
                continue
            verts.append(sv * r + xyz)
            faces.append(sf + off)
            cols.append(np.tile(np.array(tuple(c[:3]) + (alpha,)), (len(sv), 1)))
            off += len(sv)
        if not verts:
            return
        item = gl.GLMeshItem(vertexes=np.concatenate(verts), faces=np.concatenate(faces),
                             vertexColors=np.concatenate(cols), smooth=True, shader="shaded",
                             glOptions="translucent")
        self._add(group, item)

    # -------------------------------------------------------------- arrows
    def add_arrows(self, points: np.ndarray, vectors: np.ndarray, group: str,
                   scale: float = 1.0, color=(0.95, 0.85, 0.20), alpha: float = 0.95,
                   shaft_radius: float = 0.07, head_radius: float = 0.20,
                   head_fraction: float = 0.35, min_length: float = 0.15) -> None:
        """One arrow per point: a cylinder with a cone on its end, pointing along the vector.

        This is how a field that is a **direction** is drawn.  The co-field of a scalar field is
        a value at a place and is drawn as an isosurface; the co-field of the electric-field kind
        (plan item EC5) is a vector at a place, and only an arrow can show it.  `scale` is
        angstrom per unit of the vector, so the drawing's length scale is a number the caller
        states in the legend; an arrow shorter than `min_length` angstrom after scaling is left
        out, because a dot in the picture carries no direction.
        """
        pts = np.asarray(points, float).reshape(-1, 3)
        vec = np.asarray(vectors, float).reshape(-1, 3)
        if not len(pts):
            return
        cyl = MeshData.cylinder(rows=1, cols=10, radius=[1.0, 1.0], length=1.0)
        cv, cf = cyl.vertexes(), cyl.faces()
        cone = MeshData.cylinder(rows=1, cols=10, radius=[1.0, 0.0], length=1.0)
        nv, nf = cone.vertexes(), cone.faces()
        verts, faces, cols = [], [], []
        off = 0
        rgba = np.array(tuple(color[:3]) + (alpha,))
        for p, d in zip(pts, vec):
            L = float(np.linalg.norm(d)) * float(scale)
            if L < float(min_length):
                continue
            R = _rotation_to(d / max(float(np.linalg.norm(d)), 1e-12))
            head = L * float(head_fraction)
            stem = max(L - head, 1e-6)
            s = cv.copy()
            s[:, :2] *= float(shaft_radius)
            s[:, 2] *= stem
            verts.append(s @ R.T + p)
            faces.append(cf + off)
            cols.append(np.tile(rgba, (len(cv), 1)))
            off += len(cv)
            h = nv.copy()
            h[:, :2] *= float(head_radius)
            h[:, 2] *= head
            verts.append(h @ R.T + p + (R @ np.array([0.0, 0.0, stem])))
            faces.append(nf + off)
            cols.append(np.tile(rgba, (len(nv), 1)))
            off += len(nv)
        if not verts:
            return
        item = gl.GLMeshItem(vertexes=np.concatenate(verts), faces=np.concatenate(faces),
                             vertexColors=np.concatenate(cols), smooth=True, shader="shaded",
                             glOptions="translucent" if alpha < 1 else "opaque")
        self._add(group, item)

    def add_frame(self, origin, group: str, length: float = 4.0,
                  labels: Sequence[str] = ("x", "y", "z"), alpha: float = 0.95) -> None:
        """The three axes of a frame as arrows at `origin`, red, green and blue, with labels.

        A surface frame is a convention -- the normal is z and the top atomic layer is z = 0 --
        and a convention that is not drawn is a convention a reader has to take on trust.  The
        colours are the same as the lattice vectors of a cell carry, so the two views agree.
        """
        o = np.asarray(origin, float).reshape(3)
        colors = ((0.90, 0.25, 0.25), (0.30, 0.85, 0.30), (0.35, 0.50, 0.95))
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1.0
            self.add_arrows(o[None, :], d[None, :] * float(length), group, scale=1.0,
                            color=colors[k], alpha=alpha, shaft_radius=0.05, head_radius=0.16,
                            head_fraction=0.22)
        if labels:
            pos = np.array([o + np.eye(3)[k] * float(length) * 1.12 for k in range(3)])
            self.add_labels(pos, list(labels)[:3], group, color=(0.95, 0.95, 0.95), size=9)

    # -------------------------------------------------------- isosurfaces
    def add_isosurface(self, verts: np.ndarray, faces: np.ndarray, color, group: str,
                       alpha: float = 0.45, wireframe: bool = False, normals: Optional[np.ndarray] = None) -> None:
        if wireframe:
            item = gl.GLMeshItem(vertexes=verts, faces=faces, drawFaces=False, drawEdges=True,
                                 edgeColor=tuple(color[:3]) + (min(1.0, alpha + 0.3),), glOptions="translucent")
        else:
            md = MeshData(vertexes=verts, faces=faces)
            item = gl.GLMeshItem(meshdata=md, color=tuple(color[:3]) + (alpha,), smooth=True,
                                 shader="shaded", drawEdges=False, glOptions="translucent")
        self._add(group, item)

    # --------------------------------------------------------------- misc
    def set_axes(self, on: bool, size: float = 5.0) -> None:
        if self._axes is not None:
            self.view.removeItem(self._axes)
            self._axes = None
        if on:
            self._axes = gl.GLAxisItem()
            self._axes.setSize(size, size, size)
            self.view.addItem(self._axes)

    def set_dark(self, dark: bool) -> None:
        self.dark = dark
        self.view.setBackgroundColor((20, 22, 28) if dark else (245, 245, 248))

    def center_on(self, coords: np.ndarray, distance: Optional[float] = None) -> None:
        if coords is None or len(coords) == 0:
            return
        c = coords.mean(axis=0)
        self.view.opts["center"] = QVector3D(float(c[0]), float(c[1]), float(c[2]))
        if distance is None:
            span = np.ptp(coords, axis=0).max() if len(coords) > 1 else 10.0
            distance = max(12.0, 2.2 * span)
        self.view.setCameraPosition(distance=distance)
        self.view.update()

    def reset_view(self) -> None:
        self.view.setCameraPosition(elevation=20, azimuth=-60)
        self.view.update()

    def screenshot(self, path: str) -> None:
        img = self.view.grabFramebuffer()
        img.save(path)
