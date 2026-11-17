"""
Drawing many copies of one shape at once, for structures of ten thousand atoms (plan item NT8).

The viewer used to draw N atoms as one mesh of N spheres.  Two things made that slow for a large
structure, and only the second one was visible in the code:

* the mesh holds every vertex of every sphere, so it grows with N times the size of one sphere;
* pyqtgraph computes the normal of each vertex (the direction its shading is taken from) in a
  Python loop over the vertices, once, on the first frame.  Four thousand atoms took half a
  minute before anything appeared.

Two repairs follow.  :func:`vertex_normals` computes the same normals with array operations, and
:func:`mesh_item` hands them to pyqtgraph, so every mesh the viewer draws (isosurfaces, arrows,
molecules) skips the loop.  :class:`InstancedMeshItem` goes further for the shapes that repeat:
one sphere, or one cylinder, is sent to the graphics card once with its normals, and every atom
or bond is a row of 16 numbers -- where the copy stands, the three columns of the matrix that
scales and turns it, and its colour.  The card draws all copies in one call
(``glDrawElementsInstanced``, the OpenGL name for this "instanced drawing").

A card or driver without instanced drawing (it needs OpenGL 3.3) gets the same picture another
way: the item builds the merged mesh itself, with :func:`expand_instances`, and draws that.  The
two ways give the same vertices, which the tests check.
"""
from __future__ import annotations

import ctypes
import importlib
from typing import Optional, Sequence, Tuple

import numpy as np
from OpenGL import GL
from pyqtgraph.opengl import GLMeshItem, MeshData, shaders
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QT_LIB, QtGui

QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")

#: numbers per copy in the instance buffer: offset (3), the matrix by columns (9), colour (4)
INSTANCE_WIDTH = 16

#: sphere resolution (rows, columns) by the number of copies: a small molecule keeps the smooth
#: sphere it always had, and a particle of ten thousand atoms gets a coarser one that still looks
#: round at the size an atom has on screen when ten thousand are in view
SPHERE_DETAIL = ((2000, (10, 18)), (20000, (8, 14)), (None, (6, 10)))


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The normal of each vertex: the sum of the normals of the faces around it, made unit.

    This is what ``pyqtgraph``'s ``MeshData.vertexNormals`` returns -- the face normals are not
    made unit before the sum, so a large face counts more -- but computed with array operations
    instead of a Python loop over the vertices.  A vertex that belongs to no face, or whose
    faces cancel, gets the zero vector, as there.
    """
    v = np.asarray(verts, float).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    out = np.zeros_like(v)
    if len(f):
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for k in range(3):
            np.add.at(out, f[:, k], fn)
    norm = np.linalg.norm(out, axis=1)
    ok = norm > 0
    out[ok] /= norm[ok, None]
    return out.astype(np.float32)


def mesh_item(verts: np.ndarray, faces: np.ndarray, colors=None, color=None,
              smooth: bool = True, **kwargs) -> GLMeshItem:
    """A ``GLMeshItem`` whose vertex normals are computed here, not by pyqtgraph's loop.

    `colors` are per-vertex RGBA rows; `color` is one RGBA for the whole mesh.  The other
    keyword arguments go to ``GLMeshItem`` unchanged.
    """
    v = np.asarray(verts, np.float32).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    md = MeshData(vertexes=v, faces=f,
                  vertexColors=None if colors is None else np.asarray(colors, np.float32))
    if smooth:
        # MeshData keeps its normals in this attribute and computes them only when it is None
        md._vertexNormals = vertex_normals(v, f)
    if color is not None:
        kwargs["color"] = tuple(color)
    return GLMeshItem(meshdata=md, smooth=smooth, **kwargs)


# ------------------------------------------------------------------------------- base shapes
def unit_sphere(rows: int = 10, cols: int = 18) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, normals, faces) of a sphere of radius 1 at the origin.

    The normals are taken from the faces (:func:`vertex_normals`), as pyqtgraph takes them.
    That is not only for speed: their sign follows the order of the corners of each face, and
    pyqtgraph's spheres are wound so that it points *into* the sphere.  The lighting of the
    'shaded' program is set up for that sign, so a sphere with normals pointing out is lit from
    behind and looks dark.  Taking them the same way keeps the old picture.
    """
    md = MeshData.sphere(rows=rows, cols=cols, radius=1.0)
    v = md.vertexes().astype(np.float32)
    f = md.faces().astype(np.uint32)
    return v, vertex_normals(v, f), f


def unit_cylinder(cols: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, normals, faces) of an open cylinder of radius 1 from z = 0 to z = 1.

    The normals are taken from the faces, as for :func:`unit_sphere`.  The ends are open, as
    the sticks of the viewer have always been: a sphere or the next stick covers them.
    """
    md = MeshData.cylinder(rows=1, cols=cols, radius=[1.0, 1.0], length=1.0)
    v = md.vertexes().astype(np.float32)
    f = md.faces().astype(np.uint32)
    return v, vertex_normals(v, f), f


def sphere_detail(n: int) -> Tuple[int, int]:
    """The (rows, columns) of the sphere used for `n` copies; see ``SPHERE_DETAIL``."""
    for limit, detail in SPHERE_DETAIL:
        if limit is None or n <= limit:
            return detail
    return SPHERE_DETAIL[-1][1]


# ------------------------------------------------------------------------------ the copies
def _rgba(colors, n: int, alpha: Optional[float]) -> np.ndarray:
    c = np.asarray(colors, float)
    if c.ndim == 1:
        c = np.tile(c, (n, 1))
    c = c.reshape(n, -1)
    if c.shape[1] == 3:
        c = np.hstack([c, np.ones((n, 1))])
    if alpha is not None:
        c = c.copy()
        c[:, 3] = alpha
    return c


def sphere_instances(centers: np.ndarray, radii: Sequence[float], colors,
                     alpha: Optional[float] = None, min_radius: float = 1e-4) -> np.ndarray:
    """One row of ``INSTANCE_WIDTH`` numbers per sphere: centre, radius times the unit matrix,
    colour.  A sphere of radius `min_radius` or less is left out, as the viewer always did."""
    c = np.asarray(centers, float).reshape(-1, 3)
    r = np.broadcast_to(np.asarray(radii, float), (len(c),))
    rgba = _rgba(colors, len(c), alpha)
    keep = r > min_radius
    out = np.zeros((int(keep.sum()), INSTANCE_WIDTH), np.float32)
    out[:, 0:3] = c[keep]
    out[:, 3] = out[:, 7] = out[:, 11] = r[keep]
    out[:, 12:16] = rgba[keep]
    return out


def _turn_z_onto(d: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) that turn the z axis onto each unit vector of `d`."""
    n = len(d)
    R = np.tile(np.eye(3), (n, 1, 1))
    c = d[:, 2]
    v = np.stack([-d[:, 1], d[:, 0], np.zeros(n)], axis=1)          # z cross d
    general = (c > -1 + 1e-9) & (c < 1 - 1e-9)
    if general.any():
        vg, cg = v[general], c[general]
        K = np.zeros((len(vg), 3, 3))
        K[:, 0, 1], K[:, 0, 2] = -vg[:, 2], vg[:, 1]
        K[:, 1, 0], K[:, 1, 2] = vg[:, 2], -vg[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -vg[:, 1], vg[:, 0]
        R[general] = np.eye(3) + K + (K @ K) / (1.0 + cg)[:, None, None]
    R[c <= -1 + 1e-9] = np.diag([1.0, -1.0, -1.0])
    return R


def cylinder_instances(starts: np.ndarray, ends: np.ndarray, radius, colors,
                       alpha: Optional[float] = None, min_length: float = 1e-6) -> np.ndarray:
    """One row per cylinder from `starts` to `ends`: start, the turned and scaled matrix, colour.

    The matrix maps the unit cylinder (radius 1, z from 0 to 1) onto the stick: its first two
    columns are the radius times two directions across the stick, its third the vector from the
    start to the end.  A stick shorter than `min_length` is left out.
    """
    p = np.asarray(starts, float).reshape(-1, 3)
    q = np.asarray(ends, float).reshape(-1, 3)
    d = q - p
    L = np.linalg.norm(d, axis=1)
    keep = L > min_length
    rad = np.broadcast_to(np.asarray(radius, float), (len(p),))[keep]
    rgba = _rgba(colors, len(p), alpha)[keep]
    R = _turn_z_onto(d[keep] / L[keep, None])
    M = R * np.stack([rad, rad, L[keep]], axis=1)[:, None, :]       # scale the columns
    out = np.zeros((int(keep.sum()), INSTANCE_WIDTH), np.float32)
    out[:, 0:3] = p[keep]
    out[:, 3:12] = M.transpose(0, 2, 1).reshape(-1, 9)               # column by column
    out[:, 12:16] = rgba
    return out


def instance_matrices(instances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(offsets, matrices, colours) of instance rows; matrices are (n, 3, 3)."""
    a = np.asarray(instances, float).reshape(-1, INSTANCE_WIDTH)
    M = a[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1)             # columns back to a matrix
    return a[:, 0:3], M, a[:, 12:16]


def expand_instances(verts: np.ndarray, normals: np.ndarray, faces: np.ndarray,
                     instances: np.ndarray):
    """The merged mesh the instances stand for: (vertices, normals, faces, colours).

    This is what the graphics card computes for each copy, done on the processor: the vertex is
    ``M v + offset`` and the normal is ``M (n / s)`` made unit, where `s` holds the squared
    lengths of the columns of ``M``.  For a matrix whose columns are at right angles (a sphere,
    a stick) that is the inverse transpose, which is the matrix that keeps a normal at right
    angles to its surface.  It is the fallback of :class:`InstancedMeshItem` and the reference
    the vertex shader is tested against.
    """
    off, M, col = instance_matrices(instances)
    v = np.asarray(verts, float)
    n = np.asarray(normals, float)
    f = np.asarray(faces, np.int64)
    k = len(off)
    if not k:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.uint32), np.zeros((0, 4), np.float32))
    V = np.einsum("kij,vj->kvi", M, v) + off[:, None, :]
    s = np.einsum("kij,kij->kj", M, M)                               # squared column lengths
    N = np.einsum("kij,kvj->kvi", M, n[None, :, :] / s[:, None, :])
    N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-12)
    F = f[None, :, :] + (np.arange(k) * len(v))[:, None, None]
    C = np.repeat(col[:, None, :], len(v), axis=1)
    return (V.reshape(-1, 3).astype(np.float32), N.reshape(-1, 3).astype(np.float32),
            F.reshape(-1, 3).astype(np.uint32), C.reshape(-1, 4).astype(np.float32))


# ------------------------------------------------------------------------------ the item
_VERTEX = """
uniform mat4 u_mvp;
uniform mat3 u_normal;
attribute vec4 a_position;
attribute vec3 a_normal;
attribute vec3 i_offset;
attribute vec3 i_col0;
attribute vec3 i_col1;
attribute vec3 i_col2;
attribute vec4 i_color;
varying vec4 v_color;
varying vec3 v_normal;
void main() {
    mat3 m = mat3(i_col0, i_col1, i_col2);
    vec3 s = vec3(dot(i_col0, i_col0), dot(i_col1, i_col1), dot(i_col2, i_col2));
    v_normal = normalize(u_normal * (m * (a_normal / s)));
    v_color = i_color;
    gl_Position = u_mvp * vec4(m * a_position.xyz + i_offset, 1.0);
}
"""

# the lighting of pyqtgraph's 'shaded' program, so an instanced sphere looks like a meshed one
_FRAGMENT = """
#ifdef GL_ES
precision mediump float;
#endif
varying vec4 v_color;
varying vec3 v_normal;
void main() {
    float p = dot(v_normal, normalize(vec3(1.0, -1.0, -1.0)));
    p = p < 0. ? 0. : p * 0.8;
    gl_FragColor = vec4(v_color.rgb * (0.2 + p), v_color.a);
}
"""

_PROGRAM = None


def _program():
    global _PROGRAM
    if _PROGRAM is None:
        _PROGRAM = shaders.ShaderProgram("cmf-instanced", [shaders.VertexShader(_VERTEX),
                                                           shaders.FragmentShader(_FRAGMENT)])
    return _PROGRAM


def _buffer(kind):
    return QtOpenGL.QOpenGLBuffer(kind)


class InstancedMeshItem(GLGraphicsItem):
    """One base mesh drawn once per instance row, in one call to the graphics card.

    `verts`, `normals` and `faces` are the base shape (:func:`unit_sphere`,
    :func:`unit_cylinder`); `instances` are rows from :func:`sphere_instances` or
    :func:`cylinder_instances`.  ``mode`` says how the last frame was drawn: "instanced",
    "merged" (the fallback) or "" before the first frame.
    """

    def __init__(self, verts, normals, faces, instances, glOptions="opaque", parentItem=None):
        super().__init__()
        self.setGLOptions(glOptions)
        self.setParentItem(parentItem)
        self.base_verts = np.ascontiguousarray(verts, np.float32)
        self.base_normals = np.ascontiguousarray(normals, np.float32)
        self.base_faces = np.ascontiguousarray(faces, np.uint32)
        self.instances = np.ascontiguousarray(instances, np.float32).reshape(-1, INSTANCE_WIDTH)
        self.mode = ""
        self._uploaded = False
        self._merged = None
        Type = QtOpenGL.QOpenGLBuffer.Type
        self._vbo_v, self._vbo_n = _buffer(Type.VertexBuffer), _buffer(Type.VertexBuffer)
        self._vbo_i, self._ibo = _buffer(Type.VertexBuffer), _buffer(Type.IndexBuffer)
        self._merged_bufs = None

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    @staticmethod
    def _upload(buf, arr):
        if not buf.isCreated():
            buf.create()
        buf.bind()
        buf.allocate(arr, arr.nbytes)
        buf.release()

    def _attrib(self, program, name, buf, size, stride=0, offset=0, divisor=0, locs=None):
        loc = GL.glGetAttribLocation(program, name)
        if loc == -1:
            return
        buf.bind()
        GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(offset) if offset else None)
        buf.release()
        GL.glEnableVertexAttribArray(loc)
        if divisor:
            GL.glVertexAttribDivisor(loc, divisor)
        locs.append((loc, divisor))

    def paint(self):
        if not self.n_instances or not len(self.base_faces):
            return
        self.setupGLState()
        if self.mode != "merged":
            try:
                self._paint_instanced()
                self.mode = "instanced"
                return
            except Exception:                       # noqa: BLE001 - no instancing: fall back
                self.mode = "merged"
        self._paint_merged()

    def _matrices(self):
        mvp = np.array(self.mvpMatrix().data(), dtype=np.float32)
        nrm = np.array(self.modelViewMatrix().normalMatrix().data(), dtype=np.float32)
        return mvp, nrm

    def _paint_instanced(self):
        if not self._uploaded:
            self._upload(self._vbo_v, self.base_verts)
            self._upload(self._vbo_n, self.base_normals)
            self._upload(self._vbo_i, self.instances)
            self._upload(self._ibo, self.base_faces)
            self._uploaded = True
        es2 = QtGui.QOpenGLContext.currentContext().hasExtension(b"GL_ARB_ES2_compatibility")
        shader = _program()
        program = shader.program(es2_compat=es2)
        mvp, nrm = self._matrices()
        locs = []
        stride = INSTANCE_WIDTH * 4
        try:
            self._attrib(program, "a_position", self._vbo_v, 3, locs=locs)
            self._attrib(program, "a_normal", self._vbo_n, 3, locs=locs)
            for name, size, offset in (("i_offset", 3, 0), ("i_col0", 3, 12), ("i_col1", 3, 24),
                                       ("i_col2", 3, 36), ("i_color", 4, 48)):
                self._attrib(program, name, self._vbo_i, size, stride, offset, 1, locs)
            with shader:
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, "u_mvp"), 1, False, mvp)
                GL.glUniformMatrix3fv(GL.glGetUniformLocation(program, "u_normal"), 1, False,
                                      nrm)
                self._ibo.bind()
                GL.glDrawElementsInstanced(GL.GL_TRIANGLES, self.base_faces.size,
                                           GL.GL_UNSIGNED_INT, None, self.n_instances)
                self._ibo.release()
        finally:
            # a divisor left on an attribute location would turn the next item's colours into
            # one colour per copy: every location is put back as it was
            for loc, divisor in locs:
                if divisor:
                    GL.glVertexAttribDivisor(loc, 0)
                GL.glDisableVertexAttribArray(loc)

    def _paint_merged(self):
        if self._merged is None:
            self._merged = expand_instances(self.base_verts, self.base_normals,
                                            self.base_faces, self.instances)
            Type = QtOpenGL.QOpenGLBuffer.Type
            self._merged_bufs = [_buffer(Type.VertexBuffer), _buffer(Type.VertexBuffer),
                                 _buffer(Type.VertexBuffer), _buffer(Type.IndexBuffer)]
            V, N, F, C = self._merged
            for buf, arr in zip(self._merged_bufs, (V, N, C, F)):
                self._upload(buf, np.ascontiguousarray(arr))
        es2 = QtGui.QOpenGLContext.currentContext().hasExtension(b"GL_ARB_ES2_compatibility")
        shader = shaders.getShaderProgram("shaded")
        program = shader.program(es2_compat=es2)
        mvp, nrm = self._matrices()
        bv, bn, bc, bf = self._merged_bufs
        locs = []
        try:
            self._attrib(program, "a_position", bv, 3, locs=locs)
            self._attrib(program, "a_normal", bn, 3, locs=locs)
            self._attrib(program, "a_color", bc, 4, locs=locs)
            with shader:
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, "u_mvp"), 1, False, mvp)
                GL.glUniformMatrix3fv(GL.glGetUniformLocation(program, "u_normal"), 1, False,
                                      nrm)
                bf.bind()
                GL.glDrawElements(GL.GL_TRIANGLES, self._merged[2].size, GL.GL_UNSIGNED_INT,
                                  None)
                bf.release()
        finally:
            for loc, _ in locs:
                GL.glDisableVertexAttribArray(loc)


def spheres_item(centers, radii, colors, alpha: Optional[float] = None,
                 glOptions: str = "opaque", detail: Optional[Tuple[int, int]] = None):
    """An :class:`InstancedMeshItem` of spheres, or None when no sphere is left to draw."""
    inst = sphere_instances(centers, radii, colors, alpha)
    if not len(inst):
        return None
    v, n, f = unit_sphere(*(detail or sphere_detail(len(inst))))
    return InstancedMeshItem(v, n, f, inst, glOptions=glOptions)


def sticks_item(starts, ends, radius, colors, alpha: Optional[float] = None,
                glOptions: str = "opaque", cols: int = 8):
    """An :class:`InstancedMeshItem` of open cylinders, or None when none is left to draw."""
    inst = cylinder_instances(starts, ends, radius, colors, alpha)
    if not len(inst):
        return None
    v, n, f = unit_cylinder(cols)
    return InstancedMeshItem(v, n, f, inst, glOptions=glOptions)


__all__ = ["INSTANCE_WIDTH", "InstancedMeshItem", "SPHERE_DETAIL", "cylinder_instances",
           "expand_instances", "instance_matrices", "mesh_item", "sphere_detail",
           "sphere_instances", "spheres_item", "sticks_item", "unit_cylinder", "unit_sphere",
           "vertex_normals"]
