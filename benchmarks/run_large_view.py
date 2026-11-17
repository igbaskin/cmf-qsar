#!/usr/bin/env python
"""
What a structure of ten thousand atoms costs to show, and whether the faster drawing shows the
same picture (plan item NT8).

The 3D view drew N atoms as one mesh of N spheres.  This script measures that drawing against
the one plan item NT8 put in its place, where one sphere (or one stick) is held by the graphics
card and every atom is a copy of it, drawn in one call ("instanced drawing",
:mod:`cmfqsar.gui.instanced`).  A view is not judged by a q2, so the numbers are times, memory
and pixels.

Stage normals (needs no screen)
-------------------------------
pyqtgraph shades a mesh from one normal per vertex, and it computes those normals in a Python
loop over the vertices, once, before the first frame.  For a mesh of many spheres that loop is
the whole cost of showing them.  This stage times the loop against the array computation of
:func:`cmfqsar.gui.instanced.vertex_normals` on the meshes of 256 to 4,000 spheres and on an
isosurface, and reports the largest angle between the two sets of normals.

Stage draw (needs an OpenGL context)
------------------------------------
* **Particles.**  Pt icosahedra of 923, 10,179 and 49,049 atoms, drawn as spheres the old way
  (one mesh, kept here as a copy of the old code) and the new way.  For each: the time to build
  the item, the first frame, the median of 20 later frames while the view turns, and the size
  of the data sent to the graphics card.  Each time is the fastest of three repeats, because
  other jobs on the machine can only make a repeat slower.  The old way is not timed on the
  largest particle: the loop that makes its first frame slow grows with the number of vertices,
  so it would take about five times as long as on the particle of 10,179 atoms.
* **Sticks.**  A graphene sheet of 9,800 atoms with its bonds drawn as sticks, the old way and
  the new way, and as lines, which is what the crystal window fell back to above
  ``bonds.MANY_BONDS`` sticks.
* **The same picture.**  300 spheres of different sizes and colours and 60 sticks, drawn the
  old way, the new way, and by the fallback the new item uses on a card without instanced
  drawing.  The pictures are compared pixel by pixel.
* **The windows.**  The crystal window showing the 10,179-atom particle, and the largest cell of
  ``examples/hops/disordered`` with its periodic images out to the kernel's cutoff drawn as dots
  and as balls: the time from the change of the control to the first frame.  And two views in
  one application, which is what the program is when the crystal window is open beside the
  main window: does the second view draw?  It is run in a child process twice, with the OpenGL
  contexts of the views shared, as the application now sets them, and without.

A machine without a screen (``QT_QPA_PLATFORM=offscreen``, as on a runner) has no OpenGL
context: there the draw stage says so and keeps the stored measurement, so the table stays the
one measured on the machine it names.

Usage
-----
    python benchmarks/run_large_view.py                    # both stages, then the table
    python benchmarks/run_large_view.py --stage normals
    python benchmarks/run_large_view.py --report
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RESULTS = os.path.join(HERE, "results", "large_view")
EXAMPLES = os.path.join(ROOT, "examples")
ELECTROLYTES = os.path.join(EXAMPLES, "hops", "disordered")
#: shells of the Pt icosahedra the drawing is timed on: 923, 10,179 and 49,049 atoms
SHELLS = (7, 15, 25)
#: the largest particle the old drawing is timed on
OLD_LIMIT = 12000
#: sizes (cubic fcc cells per edge) of the sphere meshes whose normals are timed
NORMAL_SIZES = (4, 8, 10)
#: the graphene sheet of the stick test, in cells of two atoms per edge
SHEET = 70
#: frames timed while the view turns
FRAMES = 20
#: every time is the fastest of this many repeats: the machine may run other jobs, which can
#: only make a repeat slower
REPEATS = 3
#: view size, in pixels
VIEW = (900, 700)
#: the radius the periodic images are drawn to in the window test: the cutoff of the periodic
#: overlap kernel of the standard inorganic field set, as in benchmarks/run_cell_view.py
IMAGE_RADIUS = 14.9


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def jdump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=float)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def busy_percent() -> float:
    """How busy the processor is with all programs, in per cent over two seconds, or nan.

    The times of this script are wall-clock times on a machine that may run other jobs; the
    load is stored beside them so that a slow row can be read for what it is.
    """
    try:
        import psutil
        return float(psutil.cpu_percent(interval=2.0))
    except ImportError:
        return float("nan")


def screen_problem() -> Optional[str]:
    """Why this process cannot open an OpenGL context, or None when it can try."""
    if os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen"):
        return "QT_QPA_PLATFORM is offscreen, which has no OpenGL"
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY")
                                                 or os.environ.get("WAYLAND_DISPLAY")):
        return "no display"
    return None


def _app(share: bool = True):
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication
    if QApplication.instance() is None and share:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication.instance() or QApplication([])


def icosahedron(shells: int):
    from ase.cluster import Icosahedron
    return Icosahedron("Pt", shells, latticeconstant=3.92)


def fcc_block(n: int):
    from ase.build import bulk
    return bulk("Pt", "fcc", a=3.92, cubic=True).repeat((n, n, n))


def graphene_sheet(n: int = SHEET):
    """A flat sheet of n x n two-atom cells, open at its edges, and its C-C bonds."""
    from ase.build import graphene
    from ase.neighborlist import neighbor_list
    at = graphene(formula="C2", a=2.46, size=(n, n, 1), vacuum=5.0)
    at.pbc = False
    i, j = neighbor_list("ij", at, 1.6)
    keep = i < j
    return at, np.stack([i[keep], j[keep]], axis=1)


# ------------------------------------------------------- the old drawing, kept as the reference
def old_spheres_item(centers, radii, colors, alpha=1.0):
    """The spheres as ``Viewer3D.add_spheres`` drew them before plan item NT8: one mesh."""
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import MeshData
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
    return gl.GLMeshItem(vertexes=np.concatenate(verts), faces=np.concatenate(faces),
                         vertexColors=np.concatenate(cols), smooth=True, shader="shaded",
                         glOptions="opaque")


def old_sticks_item(coords, bonds, colors, radius=0.12, alpha=1.0):
    """The sticks as ``Viewer3D.add_structure`` drew them before plan item NT8: one mesh."""
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import MeshData
    from cmfqsar.gui.viewer3d import _rotation_to
    verts, faces, cols = [], [], []
    off = 0
    cyl = MeshData.cylinder(rows=1, cols=8, radius=[radius, radius], length=1.0)
    cv, cf = cyl.vertexes(), cyl.faces()
    for a, b in bonds:
        p1, p2 = coords[a], coords[b]
        for start, end, c in ((p1, 0.5 * (p1 + p2), colors[a]), (0.5 * (p1 + p2), p2, colors[b])):
            d = end - start
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            R = _rotation_to(d / L)
            verts.append((cv * np.array([1.0, 1.0, L])) @ R.T + start)
            faces.append(cf + off)
            cols.append(np.tile(np.array(tuple(c[:3]) + (alpha,)), (len(cv), 1)))
            off += len(cv)
    return gl.GLMeshItem(vertexes=np.concatenate(verts), faces=np.concatenate(faces),
                         vertexColors=np.concatenate(cols), smooth=True, shader="shaded",
                         glOptions="opaque")


# --------------------------------------------------------------------------- #
# Stage normals
# --------------------------------------------------------------------------- #
def _angle_max(a, b) -> float:
    cos = np.clip(np.sum(np.asarray(a, float) * np.asarray(b, float), axis=1), -1.0, 1.0)
    ok = (np.linalg.norm(a, axis=1) > 0) & (np.linalg.norm(b, axis=1) > 0)
    return float(np.degrees(np.arccos(cos[ok])).max()) if ok.any() else 0.0


def _normals_row(name, verts, faces, log) -> dict:
    from pyqtgraph.opengl import MeshData
    from cmfqsar.gui.instanced import vertex_normals
    t_loop = t_fast = float("inf")
    for _ in range(REPEATS):
        md = MeshData(vertexes=np.asarray(verts, np.float32), faces=np.asarray(faces))
        t = time.perf_counter()
        loop = md.vertexNormals()
        t_loop = min(t_loop, time.perf_counter() - t)
        t = time.perf_counter()
        fast = vertex_normals(verts, faces)
        t_fast = min(t_fast, time.perf_counter() - t)
        del md
        gc.collect()
    row = {"mesh": name, "vertices": int(len(verts)), "faces": int(len(faces)),
           "loop_s": t_loop, "arrays_s": t_fast, "angle_max_deg": _angle_max(loop, fast)}
    log(f"  {name}: {len(verts)} vertices, loop {t_loop:.2f} s, arrays {t_fast:.3f} s, "
        f"largest angle {row['angle_max_deg']:.2e} degrees")
    return row


def normals_stage(log=print) -> dict:
    from cmfqsar.gui import instanced as ins
    from cmfqsar.cofields import Grid, isosurface
    rows = []
    load = busy_percent()
    log(f"normals: the processor is {load:.0f} % busy")
    sv, sn, sf = ins.unit_sphere(10, 18)
    for n in NORMAL_SIZES:
        at = fcc_block(n)
        inst = ins.sphere_instances(at.positions, np.full(len(at), 0.61), (0.8, 0.8, 0.8))
        V, _, F, _ = ins.expand_instances(sv, sn, sf, inst)
        rows.append(_normals_row(f"{len(at)} spheres", V, F, log))
    # an isosurface: the sum of Gaussians on the atoms of the 923-atom icosahedron
    at = icosahedron(SHELLS[0])
    pos = at.positions - at.positions.mean(axis=0)
    spacing = 0.35
    lo = pos.min(axis=0) - 3.0
    shape = tuple(int(x) for x in np.ceil((pos.max(axis=0) + 3.0 - lo) / spacing) + 1)
    axes = [lo[k] + spacing * np.arange(shape[k]) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    dens = np.zeros(len(pts))
    for i, nb in enumerate(tree.query_ball_point(pts, 3.0)):
        if nb:
            d2 = np.sum((pos[nb] - pts[i]) ** 2, axis=1)
            dens[i] = np.exp(-d2 / 0.8).sum()
    grid = Grid(origin=lo, spacing=spacing, shape=shape)
    surf = isosurface(dens.reshape(shape), 0.5, grid)
    rows.append(_normals_row(f"isosurface of a {len(at)}-atom particle", surf[0], surf[1], log))
    return {"rows": rows, "busy_percent": load}


# --------------------------------------------------------------------------- #
# Stage draw
# --------------------------------------------------------------------------- #
class _View:
    """One Viewer3D on screen, with a way to time its frames and read its pixels."""

    def __init__(self):
        from cmfqsar.gui.viewer3d import Viewer3D
        self.app = _app()
        self.w = Viewer3D()
        self.w.resize(*VIEW)
        self.w.show()
        self.app.processEvents()

    def frame(self) -> float:
        from OpenGL import GL
        v = self.w.view
        v.makeCurrent()
        t = time.perf_counter()
        v.paintGL()
        GL.glFinish()
        return time.perf_counter() - t

    def frames(self, n: int = FRAMES) -> float:
        out = []
        for _ in range(n):
            self.w.view.opts["azimuth"] += 7
            out.append(self.frame())
        return float(np.median(out))

    def pixels(self) -> np.ndarray:
        self.w.view.update()
        self.app.processEvents()
        img = self.w.view.grabFramebuffer()
        a = np.array(img.convertToFormat(img.Format.Format_RGBA8888).constBits())
        return a.reshape(img.height(), img.width(), 4)[..., :3].astype(int)

    def renderer(self) -> dict:
        from OpenGL import GL
        self.w.view.makeCurrent()
        return {"renderer": GL.glGetString(GL.GL_RENDERER).decode(errors="replace"),
                "version": GL.glGetString(GL.GL_VERSION).decode(errors="replace")}

    def clear(self):
        self.w.clear_all()
        self.app.processEvents()
        gc.collect()


def card_mb(item) -> float:
    """The data an item sends to the graphics card, in MB, counted from its arrays.

    A mesh sends per vertex a position (3 numbers), a normal (3) and a colour (4), and three
    indices per triangle, all of 4 bytes.  The instanced item sends the base shape once and 16
    numbers per copy.  Lines send a position and a colour per end.  This is exact and does not
    depend on the machine, unlike the memory of the process, which other allocations move.
    """
    from cmfqsar.gui.instanced import InstancedMeshItem
    if isinstance(item, InstancedMeshItem):
        n = (item.base_verts.size + item.base_normals.size + item.base_faces.size
             + item.instances.size)
    elif "meshdata" in getattr(item, "opts", {}):
        md = item.opts["meshdata"]
        n = len(md.vertexes()) * 10 + md.faces().size
    else:
        n = len(item.pos) * 7
    return 4.0 * n / 1e6


def _timed(view, make, center, how, what, n_atoms, log) -> dict:
    """Build an item with `make`, add it, and time the first and the later frames.

    The whole of it is done ``REPEATS`` times from a new item, and each time is the fastest.
    """
    build = first = later = float("inf")
    for _ in range(REPEATS):
        view.clear()
        t = time.perf_counter()
        items = make()
        build = min(build, time.perf_counter() - t)
        for it in items:
            view.w._add("x", it)
        view.w.center_on(center)
        first = min(first, view.frame())
        later = min(later, view.frames())
    mode = ",".join(sorted({getattr(it, "mode", "") or "mesh" for it in items}))
    row = {"what": what, "how": how, "atoms": int(n_atoms), "build_s": build,
           "first_frame_s": first, "frame_ms": 1000 * later,
           "card_mb": sum(card_mb(it) for it in items), "mode": mode}
    log(f"  {what}, {how}: build {build:.2f} s, first frame {first:.2f} s, later frames "
        f"{1000 * later:.1f} ms, {row['card_mb']:.1f} MB to the card ({mode})")
    view.clear()
    return row


def _particles(view, log) -> List[dict]:
    from cmfqsar.gui import instanced as ins
    from cmfqsar.gui.viewer3d import element_color
    rows = []
    for k in SHELLS:
        at = icosahedron(k)
        pos = at.positions
        radii = np.full(len(at), 1.36 * 0.45)
        cols = [element_color("Pt")] * len(at)
        what = f"Pt icosahedron of {len(at)} atoms"
        if len(at) <= OLD_LIMIT:
            rows.append(_timed(view, lambda: [old_spheres_item(pos, radii, cols)], pos,
                               "one mesh (before NT8)", what, len(at), log))
        rows.append(_timed(view, lambda: [ins.spheres_item(pos, radii, cols, alpha=1.0)], pos,
                           "copies of one sphere", what, len(at), log))
    return rows


def _sticks(view, log) -> List[dict]:
    from cmfqsar.gui import instanced as ins
    import pyqtgraph.opengl as gl
    at, bonds = graphene_sheet()
    pos = at.positions
    col = (0.45, 0.45, 0.45)
    cols = [col] * len(at)
    what = f"graphene sheet, {len(at)} atoms, {len(bonds)} bonds"

    def lines():
        p = np.empty((2 * len(bonds), 3))
        p[0::2], p[1::2] = pos[bonds[:, 0]], pos[bonds[:, 1]]
        return [gl.GLLinePlotItem(pos=p, color=np.tile(col + (1.0,), (len(p), 1)), width=2.0,
                                  mode="lines", antialias=True, glOptions="opaque")]

    def new():
        p1, p2 = pos[bonds[:, 0]], pos[bonds[:, 1]]
        mid = 0.5 * (p1 + p2)
        return [ins.sticks_item(np.vstack([p1, mid]), np.vstack([mid, p2]), 0.12, col)]

    return [_timed(view, lambda: [old_sticks_item(pos, bonds, cols)], pos,
                   "one mesh (before NT8)", what, len(at), log),
            _timed(view, new, pos, "copies of one stick", what, len(at), log),
            _timed(view, lines, pos, "lines", what, len(at), log)]


def _same_picture(view, log) -> dict:
    from cmfqsar.gui import instanced as ins
    rng = np.random.default_rng(0)
    centers = rng.uniform(-8, 8, (300, 3))
    radii = rng.uniform(0.3, 0.9, 300)
    cols = rng.uniform(0.2, 1.0, (300, 3))
    starts = rng.uniform(-8, 8, (60, 3))
    ends = starts + rng.normal(0, 1.5, (60, 3))
    sc = np.vstack([starts, ends])
    bonds = np.stack([np.arange(60), np.arange(60, 120)], axis=1)
    stick_cols = np.vstack([cols[:60], cols[:60]])
    pics, modes = {}, {}
    for how in ("old", "instanced", "fallback"):
        view.clear()
        if how == "old":
            items = [old_spheres_item(centers, radii, cols),
                     old_sticks_item(sc, bonds, stick_cols, radius=0.15)]
        else:
            # the old sticks are two half sticks per bond, each in its own atom's colour
            mid = 0.5 * (starts + ends)
            items = [ins.spheres_item(centers, radii, cols, alpha=1.0, detail=(10, 18)),
                     ins.sticks_item(np.vstack([starts, mid]), np.vstack([mid, ends]), 0.15,
                                     np.vstack([cols[:60], cols[:60]]), alpha=1.0)]
            if how == "fallback":
                for it in items:
                    it.mode = "merged"
        for it in items:
            view.w._add("x", it)
        view.w.center_on(centers)
        pics[how] = view.pixels()
        modes[how] = sorted({getattr(it, "mode", "") or "mesh" for it in items})
    view.clear()
    bg = pics["old"][0, 0]
    rows = []
    for x, y in (("old", "instanced"), ("old", "fallback"), ("instanced", "fallback")):
        d = np.abs(pics[x] - pics[y]).max(axis=2)
        cx = np.abs(pics[x] - bg).max(axis=2) > 0
        cy = np.abs(pics[y] - bg).max(axis=2) > 0
        big = d > 8
        # a differing pixel is isolated when none of its eight neighbours differs
        pad = np.pad(big, 1)
        around = sum(pad[1 + i:pad.shape[0] - 1 + i, 1 + j:pad.shape[1] - 1 + j]
                     for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
        rows.append({"pair": f"{x} / {y}", "drawn": int((cx | cy).sum()),
                     "differ_8": int(big.sum()), "isolated": int((big & (around == 0)).sum()),
                     "largest": int(d.max()), "coverage": int((cx != cy).sum())})
        log(f"  {x} against {y}: {rows[-1]}")
    return {"rows": rows, "modes": modes}


def _windows(log) -> List[dict]:
    from ase.io import read
    from cmfqsar.gui.crystal_window import CrystalWindow
    app = _app()
    out = []

    def timed_window(atoms, name, setup, what):
        w = CrystalWindow(background=False)
        w.resize(1200, 800)
        w.atoms, w.names, w.y = [atoms], [name], np.array([np.nan])
        w.populate()
        w.show()
        app.processEvents()
        setup(w)
        t = time.perf_counter()
        w.table.setCurrentCell(0, 0)
        w.center_current()
        w.update_display()
        v = w.viewer.view
        from OpenGL import GL
        v.makeCurrent()
        v.paintGL()
        GL.glFinish()
        first = time.perf_counter() - t
        times = []
        for _ in range(FRAMES):
            v.opts["azimuth"] += 7
            t = time.perf_counter()
            v.makeCurrent()
            v.paintGL()
            GL.glFinish()
            times.append(time.perf_counter() - t)
        legend = w.legend.text()
        img = v.grabFramebuffer()
        a = np.array(img.convertToFormat(img.Format.Format_RGBA8888).constBits()).reshape(
            img.height(), img.width(), 4)[..., :3].astype(int)
        drawn = int((np.abs(a - a[0, 0]).max(axis=2) > 0).sum())
        row = {"what": what, "atoms": len(atoms), "first_frame_s": first,
               "frame_ms": 1000 * float(np.median(times)),
               "drawn_pixels": drawn, "legend": legend}
        log(f"  {what}: to the first frame {first:.2f} s, later frames {row['frame_ms']:.1f} ms, "
            f"{drawn} pixels drawn")
        w.close()
        w.deleteLater()
        app.processEvents()
        gc.collect()
        return row

    at = icosahedron(SHELLS[1])
    at.center(vacuum=10.0)
    at.pbc = False

    def balls(w):
        w.cmb_atoms.setCurrentText("balls")
        w.chk_images.setChecked(False)
    out.append(timed_window(at, "Pt10179", balls,
                            f"crystal window, a Pt particle of {len(at)} atoms as balls"))
    cells = [(f, read(os.path.join(ELECTROLYTES, f))) for f in sorted(os.listdir(ELECTROLYTES))
             if f.endswith(".cif")]
    name, cell = max(cells, key=lambda c: len(c[1]))
    for style in ("dots", "balls"):
        def images(w, style=style):
            w.cmb_atoms.setCurrentText("balls")
            w.chk_images.setChecked(True)
            w.spin_images.setValue(IMAGE_RADIUS)
            w.cmb_images.setCurrentText(style)
        out.append(timed_window(cell, name, images,
                                f"crystal window, {name} ({len(cell)} atoms) with its images to "
                                f"{IMAGE_RADIUS:g} A as {style}"))
    return out


def two_viewers_child(share: bool) -> None:
    """Run in a child process: two views in one application, each drawing spheres and a mesh."""
    app = _app(share=share)
    from cmfqsar.gui.viewer3d import Viewer3D
    rng = np.random.default_rng(1)
    pts = rng.uniform(-5, 5, (50, 3))
    drawn = []
    for _ in range(2):
        w = Viewer3D()
        w.resize(400, 300)
        w.show()
        app.processEvents()
        w.add_spheres(pts, np.full(50, 0.6), [(0.9, 0.3, 0.3)] * 50, "a", alpha=1.0)
        w.add_isosurface(np.array([[0.0, 0, 0], [3, 0, 0], [0, 3, 0]]), np.array([[0, 1, 2]]),
                         (0.2, 0.9, 0.2), "b", alpha=1.0)
        w.center_on(pts)
        img = w.view.grabFramebuffer()
        a = np.array(img.convertToFormat(img.Format.Format_RGBA8888).constBits()).reshape(
            img.height(), img.width(), 4)[..., :3].astype(int)
        drawn.append(int((np.abs(a - a[0, 0]).max(axis=2) > 0).sum()))
    print(json.dumps({"share": share, "drawn": drawn}), flush=True)


def _two_viewers(log) -> List[dict]:
    out = []
    for share in (True, False):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--two-viewers",
                            "share" if share else "separate"], capture_output=True, text=True,
                           cwd=ROOT, timeout=600)
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        rec = json.loads(line[-1]) if line else {"share": share, "drawn": [], "error":
                                                 r.stderr[-400:]}
        errors = (r.stdout + r.stderr).count("Error while drawing item")
        rec["drawing_errors"] = errors
        out.append(rec)
        log(f"  two views, contexts {'shared' if share else 'separate'}: pixels drawn "
            f"{rec['drawn']}, drawing errors {errors}")
    return out


def draw_stage(log=print) -> Optional[dict]:
    why = screen_problem()
    if why:
        log(f"draw: no OpenGL context here ({why}); the stored measurement is kept")
        return None
    view = _View()
    info = view.renderer()
    log(f"draw: {info['renderer']}, OpenGL {info['version']}")
    rec = {"renderer": info, "view": list(VIEW), "busy_percent": busy_percent()}
    log(f"draw: the processor is {rec['busy_percent']:.0f} % busy")
    log("particles")
    rec["particles"] = _particles(view, log)
    log("sticks")
    rec["sticks"] = _sticks(view, log)
    log("the same picture")
    rec["same"] = _same_picture(view, log)
    view.w.close()
    log("windows")
    rec["windows"] = _windows(log)
    rec["two_viewers"] = _two_viewers(log)
    return rec


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _s(x: float) -> str:
    return f"{x:.2f} s" if x >= 0.995 else f"{1000 * x:.0f} ms"


def report(results_dir: str = RESULTS) -> str:
    L = ["# A structure of ten thousand atoms on screen: what it costs, and whether it is the "
         "same picture\n"]
    prov_path = os.path.join(results_dir, "run.json")
    if os.path.exists(prov_path):
        prov = jload(prov_path)
        L.append("`" + " ".join(["python benchmarks/run_large_view.py"] + list(prov["argv"]))
                 + f"`, written {prov['written']}.\n")
    path = os.path.join(results_dir, "normals.json")
    if os.path.exists(path):
        d = jload(path)
        L.append("## The normals of a mesh: pyqtgraph's loop against array operations\n")
        L.append("A mesh is shaded from one normal per vertex. pyqtgraph computes them in a "
                 "Python loop over the vertices before the first frame; "
                 "`instanced.vertex_normals` computes the same sum with array operations. The "
                 "spheres are 10 x 18 meshes, as the view draws up to 2,000 atoms. The "
                 f"processor was {d.get('busy_percent', float('nan')):.0f} % busy with all "
                 f"programs when the stage began; each time is the fastest of {REPEATS} "
                 "repeats.\n")
        L.append("| Mesh | Vertices | pyqtgraph's loop | Array operations | Largest angle "
                 "between the two normals |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for r in d["rows"]:
            L.append(f"| {r['mesh']} | {r['vertices']:,} | {_s(r['loop_s'])} | "
                     f"{_s(r['arrays_s'])} | {r['angle_max_deg']:.1e} degrees |")
        L.append("")
    path = os.path.join(results_dir, "draw.json")
    if os.path.exists(path):
        d = jload(path)
        L.append(f"Everything below was measured on **{d['renderer']['renderer']}** (OpenGL "
                 f"{d['renderer']['version']}) in a view of {d['view'][0]} x {d['view'][1]} "
                 f"pixels, with the processor {d.get('busy_percent', float('nan')):.0f} % busy "
                 "with all programs when the stage began. A machine without a screen keeps "
                 "this measurement.\n")
        L.append("## Atoms as spheres: one mesh against copies of one sphere\n")
        L.append("The time to build the item in Python, the first frame (which includes "
                 "computing the normals and sending the data to the graphics card), the median "
                 f"of {FRAMES} later frames while the view turns, and the size of the data sent "
                 "to the graphics card, counted from the arrays (4 bytes a number). Each time "
                 f"is the fastest of {REPEATS} repeats, each from a new item. 'One mesh' is the "
                 "drawing before plan item NT8, kept in the script as a copy of the old code.\n")
        L.append("| Structure | Drawing | Building the item | First frame | Later frames | "
                 "Data sent to the graphics card |")
        L.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for r in d["particles"] + d["sticks"]:
            L.append(f"| {r['what']} | {r['how']} | {_s(r['build_s'])} | "
                     f"{_s(r['first_frame_s'])} | {r['frame_ms']:.1f} ms | "
                     f"{r['card_mb']:.1f} MB |")
        L.append("")
        L.append("## Is it the same picture\n")
        L.append("300 spheres of different sizes and colours and 60 sticks, drawn the old way, "
                 "as copies, and by the fallback the new item uses on a card without instanced "
                 "drawing (it builds the merged mesh itself). A pixel is drawn when it differs "
                 "from the background in either picture.\n")
        L.append("| Pictures compared | Pixels drawn | Pixels that differ by more than 8 of 255 "
                 "levels | Largest difference (of 255) | Pixels drawn in one picture only |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for r in d["same"]["rows"]:
            L.append(f"| {r['pair']} | {r['drawn']:,} | {r['differ_8']:,} | {r['largest']} | "
                     f"{r['coverage']:,} |")
        L.append("")
        rows = d["same"]["rows"]
        worst = max(rows, key=lambda r: r["differ_8"])
        cover = max(r["coverage"] for r in rows)
        isolated = sum(r.get("isolated", 0) for r in rows)
        L.append(f"At most {worst['differ_8']} of the {worst['drawn']:,} drawn pixels differ by "
                 "more than 8 levels, and "
                 + ("no pixel is drawn in one picture and not in the other: every sphere and "
                    "stick covers the same pixels." if cover == 0 else
                    f"{cover} pixels are drawn in one picture only.")
                 + f" Of the {sum(r['differ_8'] for r in rows)} differing pixels of the three "
                 f"comparisons, {isolated} have no differing neighbour. A single pixel that "
                 "differs while its neighbours agree is where two surfaces are at nearly the "
                 "same depth: the rounding of the vertex positions (on the graphics card for "
                 "the copies, in Python for a mesh) then decides which one is in front.\n")
        L.append("## In the windows\n")
        L.append("The time from the change of a control to the first frame of the crystal "
                 "window, which includes everything the window computes for the view.\n")
        L.append("| What is shown | Atoms in the cell | To the first frame | Later frames | "
                 "Pixels drawn |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for r in d["windows"]:
            L.append(f"| {r['what']} | {r['atoms']:,} | {_s(r['first_frame_s'])} | "
                     f"{r['frame_ms']:.1f} ms | {r['drawn_pixels']:,} |")
        L.append("")
        L.append("Two views in one application, each drawing 50 spheres and a triangle, as the "
                 "main window and the crystal window do when both are open:\n")
        L.append("| OpenGL contexts of the views | Pixels drawn by the first view | By the second "
                 "| Drawing errors |")
        L.append("| --- | ---: | ---: | ---: |")
        for r in d["two_viewers"]:
            dr = r.get("drawn") or ["-", "-"]
            L.append(f"| {'shared (the application now)' if r['share'] else 'separate (before)'}"
                     f" | {dr[0]:,} | {dr[1]:,} | {r['drawing_errors']} |"
                     if dr[0] != "-" else
                     f"| {'shared' if r['share'] else 'separate'} | - | - | "
                     f"{r['drawing_errors']} |")
        L.append("")
    return "\n".join(L)


def write_table(results_dir: str = RESULTS) -> str:
    path = os.path.join(results_dir, "table.md")
    text = report(results_dir)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


STAGES = ("normals", "draw")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--force", action="store_true", help="redo a stage that is already stored")
    ap.add_argument("--report", action="store_true", help="rebuild the table and stop")
    ap.add_argument("--two-viewers", choices=("share", "separate"), default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    if args.two_viewers:
        two_viewers_child(args.two_viewers == "share")
        return
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        print(write_table(RESULTS))
        return
    t0 = time.time()
    lines: List[str] = []

    def log(s=""):
        print(s, flush=True)
        lines.append(str(s))

    jobs = {"normals": lambda: normals_stage(log=log), "draw": lambda: draw_stage(log=log)}
    if args.in_process:
        (name,) = args.stage
        rec = jobs[name]()
        if rec is not None:
            jdump(rec, result(f"{name}.json"))
        return
    for name in STAGES:
        if name not in args.stage:
            continue
        path = result(f"{name}.json")
        if os.path.exists(path) and not args.force:
            log(f"{name}: already done")
            continue
        if name == "normals":
            # in this process: on Windows a child process without a window ran this stage
            # about eight times slower (42 s against 5 s for the loop over 4,000 spheres), as
            # the system gives such a process less of the processor
            rec = jobs[name]()
            jdump(rec, path)
            continue
        # The draw stage runs in a process of its own, which opens windows and so is not slowed
        # as above.  pyqtgraph's normal loop builds a Python list per vertex, and its time grew
        # from 12 s to 124 s for the same particle when the normals stage had run before it in
        # the same process; a fresh process gives the stage the same start every time.
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--stage", name,
                            "--in-process"], capture_output=True, text=True, cwd=ROOT)
        for line in r.stdout.splitlines():
            log(line)
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"stage {name} failed with exit {r.returncode}")
    with open(result("log.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    jdump({"argv": list(argv) if argv is not None else sys.argv[1:],
           "minutes": round((time.time() - t0) / 60, 2),
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}, result("run.json"))
    print(write_table(RESULTS))
    print(f"Elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
