"""
Fields on a grid, read from DFT output, and two kernels on them (plan item M9).

Everywhere else in the program a field is a sum of Gaussians on the atoms, and its kernels are
closed-form integrals. A DFT calculation gives the fields themselves, as values on a grid: the
electron density, the electron localisation function (ELF, a number between 0 and 1 that is high
where electron pairs are localised) and the electrostatic potential. This module takes those grids
as the fields, without fitting anything to them.

**The representation.** A :class:`GridField` is a 3D array of values, the three edge vectors of
the box the grid divides evenly, the origin, whether the box repeats (a crystal cell) or not (a
molecule in a box), and the atoms. Point ``(i, j, k)`` of an ``n1 x n2 x n3`` grid lies at
``origin + (i/n1) a + (j/n2) b + (k/n3) c``, so the volume of one grid cell is the box volume
divided by the number of points, in both cases.

**Readers.** :func:`read_cube` reads a Gaussian cube (the format of ``orca_plot`` and of the
co-field writer in :mod:`cmfqsar.cofields`); :func:`read_vasp_grid` reads the volumetric files of
VASP: ``CHGCAR``, ``CHG``, ``AECCAR0/2`` and ``PARCHG`` (density), ``ELFCAR`` (ELF) and ``LOCPOT``
(potential), compressed with gzip or not. Units on return: density in electrons per cubic
angstrom (VASP stores the density times the cell volume, and the reader divides it out), the
potential as the file has it (eV for VASP, hartree per electron for a cube), ELF without unit.

**Two kernels, as the plan names them.**

* *A shared frame*: :func:`grid_overlap_kernel` is the overlap integral of two fields sampled on
  the same points, ``K_AB = sum_g F_A(g) F_B(g) dV``. It is the exact form of the overlap kernel
  the rest of the program computes in closed form, and it needs the structures in one frame: a
  common Cartesian box for aligned molecules (:func:`on_common_box`), a common fractional grid
  for crystals on one lattice (:func:`resample_fractional`).
* *No frame*: :func:`radial_correlations` is the radial autocorrelation of a field, and the radial
  cross-correlation of two fields, ``A_fg(r) = integral of C_fg(t) over the sphere |t| = r``, with
  ``C_fg(t) = integral of F_f(x) F_g(x + t) dx``. It is computed with the fast Fourier transform,
  and the sphere is smoothed with a Gaussian window of width ``sigma`` around each radius. It
  does not change when the structure is moved or turned, and for a crystal it is taken per unit
  volume, so a supercell gives the same profile. It is the grid analogue of the field
  autocorrelation of :mod:`cmfqsar.invariant` and of
  :func:`cmfqsar.periodic.periodic_autocorrelation`,
  laid out in the same blocks (the pairs of :func:`cmfqsar.invariant.field_pairs`), so the kernel
  of those modules (:func:`cmfqsar.invariant.autocorrelation_kernel`) applies unchanged.

**The fitted comparison.** :func:`fit_atom_gaussians` fits a grid field with Gaussians of a few
shared widths on every atom (and its periodic images), and returns the fitted field on the same
grid. Feeding the fitted grid to the same kernel shows what the grid holds that atom-centred
Gaussians cannot carry, as :mod:`cmfqsar.cdft` did for the Fukui functions.

**Rule 3 and this module.** The plan keeps the kernel in closed form and allows grids for
measurement and display only. This item is the one exception it allows, and only if a grid kernel
beats the fitted atom-centred one by more than the interval. `benchmarks/run_gridfields.py` makes
that measurement; the module is not connected to :class:`cmfqsar.model.CMFModel`.
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cdft import BOHR, ELEMENTS
from .cdft import read_cube as _read_cube_raw

#: the kinds of field this module knows, with the unit each reader returns
KINDS = {"density": "e/A^3", "elf": "1", "potential": "as in the file", "other": "as in the file"}

#: how a VASP file name tells its kind
_VASP_KIND = (("CHGCAR", "density"), ("CHG", "density"), ("AECCAR", "density"),
              ("PARCHG", "density"), ("ELFCAR", "elf"), ("LOCPOT", "potential"))

#: width of the fine histogram of distances inside :func:`radial_correlations`, in angstrom
_HIST_STEP = 0.01


@dataclass
class GridField:
    """One field on a regular grid, with the box it divides and the atoms it belongs to."""
    values: np.ndarray                  # (n1, n2, n3)
    cell: np.ndarray                    # (3, 3) rows: the box edges the grid divides evenly
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    periodic: bool = True
    symbols: List[str] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    kind: str = "other"
    name: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, float)
        self.cell = np.asarray(self.cell, float).reshape(3, 3)
        self.origin = np.asarray(self.origin, float).reshape(3)
        self.positions = np.asarray(self.positions, float).reshape(-1, 3)
        if self.values.ndim != 3:
            raise ValueError(f"a grid field needs a 3D array, not shape {self.values.shape}")
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; known: {sorted(KINDS)}")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(n) for n in self.values.shape)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def dV(self) -> float:
        """The volume of one grid cell, in cubic angstrom."""
        return self.volume / float(self.values.size)

    @property
    def steps(self) -> np.ndarray:
        """(3, 3) the vector from one grid point to the next along each axis."""
        return self.cell / np.asarray(self.shape, float)[:, None]

    def points(self) -> np.ndarray:
        """(N, 3) Cartesian coordinates of every grid point, in the order of ``values.ravel()``."""
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij"), -1)
        return self.origin + idx.reshape(-1, 3) @ self.steps

    def integral(self) -> float:
        """The integral of the field over the box."""
        return float(self.values.sum() * self.dV)

    def with_values(self, values: np.ndarray, kind: Optional[str] = None) -> "GridField":
        """The same grid and atoms with other values (for example a windowed or fitted field)."""
        values = np.asarray(values, float)
        if values.shape != self.values.shape:
            raise ValueError(f"values of shape {values.shape} for a grid of shape {self.shape}")
        return GridField(values, self.cell, self.origin, self.periodic, list(self.symbols),
                         self.positions, kind or self.kind, self.name)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")


def read_cube(path: str, kind: str = "density", name: str = "") -> GridField:
    """A Gaussian cube as a :class:`GridField`, lengths in angstrom.

    `kind` says what the cube holds, because the format does not: ``"density"`` converts electrons
    per cubic bohr to electrons per cubic angstrom, any other kind keeps the values as written
    (a potential from ``orca_plot`` is in hartree per electron). A cube does not repeat, so the
    field is not periodic; its box is the grid's own, one step longer than the last point along
    each axis.
    """
    cube = _read_cube_raw(path)                  # divides by BOHR^3, which only a density wants
    values = cube.values if kind == "density" else cube.values * BOHR ** 3
    shape = np.asarray(cube.values.shape, float)
    cell = np.asarray(cube.vectors, float) * shape[:, None]
    symbols = [ELEMENTS.get(int(z), str(int(z))) for z in cube.numbers]
    return GridField(values, cell, cube.origin, False, symbols, cube.positions, kind,
                     name or os.path.basename(path))


def vasp_kind(path: str) -> str:
    """The kind of a VASP volumetric file, from its name (``CHGCAR.gz`` is a density)."""
    base = os.path.basename(path).upper()
    base = base[:-3] if base.endswith(".GZ") else base
    for prefix, kind in _VASP_KIND:
        if base.startswith(prefix):
            return kind
    raise ValueError(f"{path}: cannot tell the kind from the name; pass kind= explicitly")


def read_vasp_grid(path: str, kind: Optional[str] = None, block: int = 0,
                   name: str = "") -> GridField:
    """A VASP volumetric file (CHGCAR, CHG, AECCAR, PARCHG, ELFCAR, LOCPOT) as a GridField.

    The file holds a POSCAR, a blank line, the grid size and then the values, with the first
    index running fastest. A density is stored times the cell volume; this reader divides it out,
    so the result is in electrons per cubic angstrom and integrates to the electron count. A
    spin-polarised run stores a second grid of the same size after the first (the magnetisation
    density for a CHGCAR, the ELF of the second spin for an ELFCAR); `block` = 1 returns it. The
    file may be compressed with gzip. The kind is taken from the file name unless given.
    """
    kind = kind or vasp_kind(path)
    with _open_text(path) as fh:
        text = fh.read()
    lines = text.splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(v) for v in lines[2 + k].split()[:3]] for k in range(3)])
    if scale < 0:                                  # a negative scale is the target volume
        cell *= (-scale / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
    else:
        cell *= scale
    row = 5
    first = lines[5].split()
    if first and not first[0].lstrip("+-").isdigit():  # VASP 5: a line of species names
        species = first
        row = 6
    else:
        species = []
    counts = [int(v) for v in lines[row].split()]
    if not species:
        species = lines[0].split()[:len(counts)]
    row += 1
    if lines[row].strip()[:1].upper() == "S":       # selective dynamics
        row += 1
    cartesian = lines[row].strip()[:1].upper() in "CK"
    row += 1
    n_atoms = sum(counts)
    coords = np.array([[float(v) for v in lines[row + i].split()[:3]] for i in range(n_atoms)])
    positions = coords * (scale if scale > 0 else 1.0) if cartesian else coords @ cell
    symbols = [s for s, c in zip(species, counts) for _ in range(c)]
    row += n_atoms
    while row < len(lines) and not lines[row].split():
        row += 1
    dims_line = lines[row]
    shape = tuple(int(v) for v in dims_line.split()[:3])
    n = int(np.prod(shape))
    rest = "\n".join(lines[row + 1:])
    values = None
    for b in range(block + 1):
        tokens = rest.split(maxsplit=n)
        if len(tokens) < n:
            raise ValueError(f"{path}: grid block {b} holds fewer than {n} values")
        if b == block:
            values = np.array(tokens[:n], float)
            break
        tail = tokens[n] if len(tokens) > n else ""
        # skip whatever follows the block (augmentation occupancies, magnetic moments) up to the
        # next line that repeats the grid size
        found = None
        for i, line in enumerate(tail.splitlines()):
            if tuple(line.split()[:3]) == tuple(dims_line.split()[:3]) and len(line.split()) == 3:
                found = i
                break
        if found is None:
            raise ValueError(f"{path}: the file holds no grid block {block}")
        rest = "\n".join(tail.splitlines()[found + 1:])
    grid = values.reshape(shape[::-1]).transpose(2, 1, 0)       # first index runs fastest
    volume = abs(np.linalg.det(cell))
    if kind == "density":
        grid = grid / volume
    return GridField(grid, cell, np.zeros(3), True, symbols, positions, kind,
                     name or os.path.basename(path))


def write_vasp_grid(grid: GridField, path: str) -> None:
    """Write a periodic GridField in the CHGCAR layout (the density times the volume).

    Meant for tests and for exchange with VASP tools such as a Bader program; the inverse of
    :func:`read_vasp_grid` for one block.
    """
    if not grid.periodic:
        raise ValueError("only a periodic grid can be written as a VASP file")
    species = list(dict.fromkeys(grid.symbols))
    order = [i for s in species for i, t in enumerate(grid.symbols) if t == s]
    frac = np.linalg.solve(grid.cell.T, (grid.positions[order] - grid.origin).T).T
    out = [grid.name or "grid", "1.0"]
    out += ["  %.10f %.10f %.10f" % tuple(v) for v in grid.cell]
    out += ["  " + " ".join(species), "  " + " ".join(str(grid.symbols.count(s)) for s in species)]
    out += ["Direct"] + ["  %.8f %.8f %.8f" % tuple(p) for p in frac]
    out += ["", " %d %d %d" % grid.shape]
    vals = grid.values * (grid.volume if grid.kind == "density" else 1.0)
    flat = vals.transpose(2, 1, 0).ravel()
    for i in range(0, flat.size, 5):
        out.append(" ".join("%.11E" % v for v in flat[i:i + 5]))
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as fh:
        fh.write("\n".join(out) + "\n")


# --------------------------------------------------------------------------- #
# Putting grids on common points
# --------------------------------------------------------------------------- #

def fourier_resample(values: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """A periodic grid on another number of points, by cutting or padding its Fourier series.

    Exact for a field that has no frequency above the smaller grid's limit; going to fewer points
    removes the high frequencies first, so the coarse grid is not aliased. The mean is kept.
    """
    v = np.asarray(values, float)
    shape = tuple(int(n) for n in shape)
    if v.shape == shape:
        return v.copy()
    F = np.fft.fftshift(np.fft.fftn(v))
    out = np.zeros(shape, complex)
    src, dst = [], []
    for n_old, n_new in zip(v.shape, shape):
        m = min(n_old, n_new)
        c_old, c_new = n_old // 2, n_new // 2
        lo = -(m // 2)
        src.append(slice(c_old + lo, c_old + lo + m))
        dst.append(slice(c_new + lo, c_new + lo + m))
    out[tuple(dst)] = F[tuple(src)]
    res = np.fft.ifftn(np.fft.ifftshift(out)).real
    return res * (np.prod(shape) / float(v.size))


def resample_fractional(grid: GridField, shape: Sequence[int]) -> GridField:
    """A periodic field on an ``n1 x n2 x n3`` grid of its own cell (see :func:`fourier_resample`).

    Two crystals given the same lattice and resampled to the same shape share their grid points,
    which is the shared frame of :func:`grid_overlap_kernel`.
    """
    if not grid.periodic:
        raise ValueError("resample_fractional is for periodic grids; use on_common_box")
    return GridField(fourier_resample(grid.values, shape), grid.cell, grid.origin, True,
                     list(grid.symbols), grid.positions, grid.kind, grid.name)


def common_box(grids: Sequence[GridField], pad: float = 0.0,
               spacing: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """(origin, shape) of an axis-aligned box that holds every grid's box, plus `pad` angstrom."""
    lo, hi = [], []
    for g in grids:
        corners = g.origin + np.array([[i, j, k] for i in (0, 1) for j in (0, 1)
                                       for k in (0, 1)], float) @ g.cell
        lo.append(corners.min(0))
        hi.append(corners.max(0))
    lo = np.min(lo, axis=0) - pad
    hi = np.max(hi, axis=0) + pad
    shape = np.maximum(np.ceil((hi - lo) / spacing).astype(int), 2)
    return lo, shape


def on_common_box(grid: GridField, origin: np.ndarray, shape: Sequence[int],
                  spacing: float = 0.25) -> GridField:
    """A field interpolated onto an axis-aligned box of the given origin, shape and spacing.

    Linear interpolation; points outside the field's own box get zero, which is right for a
    molecule in vacuum and wrong for a crystal, so a periodic grid is refused.
    """
    from scipy.ndimage import map_coordinates

    if grid.periodic:
        raise ValueError("on_common_box is for grids that do not repeat; use resample_fractional")
    shape = tuple(int(n) for n in shape)
    axes = [origin[k] + np.arange(shape[k]) * spacing for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    idx = np.linalg.solve(grid.steps.T, (pts - grid.origin).T)
    vals = map_coordinates(grid.values, idx, order=1, mode="constant", cval=0.0)
    cell = np.diag(np.asarray(shape, float) * spacing)
    return GridField(vals.reshape(shape), cell, np.asarray(origin, float), False,
                     list(grid.symbols), grid.positions, grid.kind, grid.name)


# --------------------------------------------------------------------------- #
# The kernels
# --------------------------------------------------------------------------- #

def grid_overlap_kernel(A: Sequence[GridField], B: Optional[Sequence[GridField]] = None,
                        normalize: bool = True) -> np.ndarray:
    """``K[i, j] = sum_g F_Bi(g) F_Aj(g) dV`` for fields that share their grid points.

    Rows are the fields of `B` (the training fields `A` when `B` is not given), columns those of
    `A`, as in the rest of the program. With `normalize` the kernel is cosine-normalised, which
    removes the volume of a grid cell and makes the diagonal one.
    """
    B = A if B is None else B
    ref = A[0]
    for g in list(A) + list(B):
        if g.shape != ref.shape or not np.allclose(g.cell, ref.cell, rtol=1e-6, atol=1e-6):
            raise ValueError("grid_overlap_kernel needs every field on the same grid points; "
                             f"{g.name or 'a field'} has shape {g.shape} and another box")
    XA = np.stack([g.values.ravel() for g in A])
    XB = XA if B is A else np.stack([g.values.ravel() for g in B])
    K = XB @ XA.T * ref.dV
    if not normalize:
        return K
    da = np.sqrt(np.maximum(np.einsum("ij,ij->i", XA, XA) * ref.dV, 1e-300))
    db = da if B is A else np.sqrt(np.maximum(np.einsum("ij,ij->i", XB, XB) * ref.dV, 1e-300))
    return K / np.outer(db, da)


@dataclass
class RadialSpec:
    """The distance channels of :func:`radial_correlations`: centres 0, step, ..., r_max."""
    r_max: float = 8.0
    step: float = 0.5
    sigma: float = 0.5

    @property
    def centres(self) -> np.ndarray:
        return np.arange(0.0, self.r_max + 1e-9, self.step)

    @property
    def n_bins(self) -> int:
        return int(self.centres.size)


def _correlation_grids(fields: Sequence[np.ndarray], pairs, pad: Sequence[int]):
    """The cross-correlation C_fg(t) of every pair, on the (padded) grid of displacements."""
    if any(p > 0 for p in pad):
        fields = [np.pad(f, [(0, p) for p in pad]) for f in fields]
    F = [np.fft.rfftn(f, axes=(0, 1, 2)) for f in fields]
    shape = fields[0].shape
    return [np.fft.irfftn(np.conj(F[f]) * F[g], s=shape, axes=(0, 1, 2)) for f, g in pairs], shape


def _displacement_lengths(shape, steps, periodic: bool, reach: float):
    """The length of every displacement vector the correlation grid holds, and its grid index.

    For a periodic grid every lattice image of a displacement is a displacement of its own, so
    each grid index appears once per image within `reach`. For a padded grid the index is read
    as a signed displacement and appears once.
    """
    idx = [np.arange(n) for n in shape]
    if periodic:
        frac = np.stack(np.meshgrid(*idx, indexing="ij"), -1).reshape(-1, 3)
        base = frac @ steps                              # displacements inside one cell
        cell = steps * np.asarray(shape, float)[:, None]
        heights = 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)    # plane spacings
        m = [int(np.ceil(reach / h)) + 1 for h in heights]
        flat = np.arange(base.shape[0])
        lengths, which = [], []
        for i in range(-m[0], m[0] + 1):
            for j in range(-m[1], m[1] + 1):
                for k in range(-m[2], m[2] + 1):
                    shifted = base + np.array([i, j, k], float) @ cell
                    d = np.linalg.norm(shifted, axis=1)
                    keep = d <= reach
                    if keep.any():
                        lengths.append(d[keep])
                        which.append(flat[keep])
        return np.concatenate(lengths), np.concatenate(which)
    signed = [np.where(a < n / 2.0, a, a - n) for a, n in zip(idx, shape)]
    disp = np.stack(np.meshgrid(*signed, indexing="ij"), -1).reshape(-1, 3) @ steps
    d = np.linalg.norm(disp, axis=1)
    keep = np.nonzero(d <= reach)[0]
    return d[keep], keep


@dataclass
class RadialHistogram:
    """The fine histogram of the correlations of one structure, before the Gaussian windows.

    ``hist[p, b]`` sums ``C_fg(t) dV`` over the displacements ``t`` whose length falls in bin
    ``b`` of width :data:`_HIST_STEP`, for pair ``p``; :func:`radial_profiles` turns it into the
    channels of any :class:`RadialSpec` whose windows end inside ``reach``. Computing it once and
    reading several specs from it is what makes a choice of width cheap.
    """
    hist: np.ndarray                    # (pairs, bins)
    pairs: List[Tuple[int, int]]
    reach: float
    volume: Optional[float]             # the cell volume of a crystal, None for a molecule


def radial_histogram(grids: Sequence[GridField], reach: float = 10.0,
                     pairs: Optional[Sequence[Tuple[int, int]]] = None,
                     centre: Optional[Sequence[bool]] = None) -> RadialHistogram:
    """The correlations of several fields of one structure, binned by displacement length.

    ``C_fg(t) = sum_x F_f(x) F_g(x + t) dV`` is computed for every pair with the fast Fourier
    transform. For a periodic structure ``x`` runs over one cell and every lattice image of a
    displacement counts; a molecule is padded with zeros so that no displacement wraps round.
    `grids` are the fields of one structure on one grid; `pairs` defaults to
    :func:`cmfqsar.invariant.field_pairs`; `centre` subtracts the mean of a field first, which a
    potential needs (its mean is set by convention, not by the structure).
    """
    from .invariant import field_pairs

    ref = grids[0]
    for g in grids[1:]:
        if g.shape != ref.shape or not np.allclose(g.cell, ref.cell, atol=1e-6):
            raise ValueError("radial_histogram needs the fields of one structure on one grid")
    pairs = list(pairs) if pairs is not None else field_pairs(len(grids))
    centre = list(centre) if centre is not None else [False] * len(grids)
    vals = [g.values - (g.values.mean() if c else 0.0) for g, c in zip(grids, centre)]
    steps = ref.steps
    if ref.periodic:
        pad = [0, 0, 0]
    else:
        lens = np.linalg.norm(steps, axis=1)
        pad = [int(np.ceil(reach / max(h, 1e-12))) + 1 for h in lens]
    corr, shape = _correlation_grids(vals, pairs, pad)
    d, which = _displacement_lengths(shape, steps, ref.periodic, reach)
    dV = ref.dV
    n_hist = int(np.ceil(reach / _HIST_STEP)) + 2
    hbin = np.minimum((d / _HIST_STEP).astype(int), n_hist - 1)
    hist = np.empty((len(pairs), n_hist))
    for p in range(len(pairs)):
        c = corr[p].ravel()[which] * dV * dV            # C(t) dV, then the displacement's dV
        hist[p] = np.bincount(hbin, weights=c, minlength=n_hist)
    return RadialHistogram(hist, pairs, float(reach), ref.volume if ref.periodic else None)


def radial_profiles(h: RadialHistogram, spec: RadialSpec) -> np.ndarray:
    """The channels of `spec` read from a histogram, flattened ``(pairs * n_bins,)``.

    ``A_fg(r_k) = sum_t C_fg(t) g(|t| - r_k) dV`` with ``g`` a normalised Gaussian of width
    ``spec.sigma``. A cross pair is counted both ways round, as
    :func:`cmfqsar.invariant.autocorrelation` counts it, and a crystal's channels are divided by
    its cell volume, so a supercell gives the same profile.
    """
    if spec.r_max + 4.0 * spec.sigma > h.reach + 1e-9:
        raise ValueError(f"the windows of r_max {spec.r_max} and sigma {spec.sigma} reach past "
                         f"the histogram's {h.reach} A")
    n_hist = h.hist.shape[1]
    mids = (np.arange(n_hist) + 0.5) * _HIST_STEP
    centres = spec.centres
    norm = spec.sigma * np.sqrt(2.0 * np.pi)
    G = np.exp(-0.5 * ((mids[None, :] - centres[:, None]) / spec.sigma) ** 2) / norm
    out = h.hist @ G.T
    for p, (f, g) in enumerate(h.pairs):
        if f != g:
            out[p] *= 2.0
    if h.volume is not None:
        out /= h.volume
    return out.ravel()


def radial_correlations(grids: Sequence[GridField], spec: RadialSpec = RadialSpec(),
                        pairs: Optional[Sequence[Tuple[int, int]]] = None,
                        centre: Optional[Sequence[bool]] = None) -> np.ndarray:
    """The radial auto- and cross-correlations of several fields of one structure, flattened.

    :func:`radial_histogram` followed by :func:`radial_profiles`; see those for the definitions.
    It does not change when the structure is moved or turned. Returns
    ``(len(pairs) * spec.n_bins,)`` in the block order of :func:`cmfqsar.invariant.field_pairs`.
    """
    h = radial_histogram(grids, spec.r_max + 4.0 * spec.sigma, pairs, centre)
    return radial_profiles(h, spec)


def radial_block_sizes(spec: RadialSpec, n_fields: int) -> List[int]:
    """Column counts of the blocks of :func:`radial_correlations`, one per pair of fields."""
    n_pairs = n_fields * (n_fields + 1) // 2
    return [spec.n_bins] * n_pairs


def radial_channel_labels(names: Sequence[str], spec: RadialSpec) -> List[str]:
    """``density x elf @ 1.5 A`` and so on, in the column order of :func:`radial_correlations`."""
    from .invariant import field_pairs

    return [f"{names[f]} x {names[g]} @ {r:g} A" for f, g in field_pairs(len(names))
            for r in spec.centres]


# --------------------------------------------------------------------------- #
# The atom-centred fit of a grid field
# --------------------------------------------------------------------------- #

def _gaussian_columns(grid: GridField, widths: Sequence[float], reach: float) -> np.ndarray:
    """(N points, atoms * widths): each atom's Gaussians, summed over images for a crystal."""
    pts = grid.points()
    widths = np.asarray(widths, float)
    cols = np.zeros((pts.shape[0], len(grid.positions) * widths.size))
    if grid.periodic:
        heights = 1.0 / np.linalg.norm(np.linalg.inv(grid.cell).T, axis=1)
        m = [int(np.ceil(reach / h)) + 1 for h in heights]
        shifts = np.array([[i, j, k] for i in range(-m[0], m[0] + 1)
                           for j in range(-m[1], m[1] + 1)
                           for k in range(-m[2], m[2] + 1)], float) @ grid.cell
    else:
        shifts = np.zeros((1, 3))
    for a, pos in enumerate(grid.positions):
        for s in shifts:
            d2 = np.sum((pts - pos - s) ** 2, axis=1)
            near = d2 <= reach * reach
            if not near.any():
                continue
            for t, beta in enumerate(widths):
                cols[near, a * widths.size + t] += np.exp(-0.5 * beta * d2[near])
    return cols


def fit_atom_gaussians(grid: GridField, widths: Sequence[float] = (8.0, 2.0, 0.5),
                       ridge: float = 1e-6, centre: bool = False,
                       reach: Optional[float] = None) -> Tuple[GridField, np.ndarray, float]:
    """The best fit of a grid field by Gaussians of shared widths on every atom.

    ``F(r) ~ sum_a sum_t c_at exp(-beta_t |r - R_a|^2 / 2)``, with the periodic images of every
    atom summed in for a crystal; `widths` are the ``beta_t`` in inverse square angstrom, and a
    small ridge (relative to the mean diagonal of the normal matrix) keeps two nearly equal
    columns from cancelling. `centre` fits the field minus its mean, as for a potential.

    Returns the fitted field on the same grid, the coefficients ``(atoms, widths)``, and the
    relative error ``|F - fit| / |F|`` in the norm the overlap kernel uses.
    """
    y = grid.values.ravel() - (grid.values.mean() if centre else 0.0)
    reach = reach if reach is not None else float(np.sqrt(2.0 * 12.0 / min(widths)))
    X = _gaussian_columns(grid, widths, reach)
    A = X.T @ X
    lam = ridge * float(np.mean(np.diag(A)))
    coef = np.linalg.solve(A + lam * np.eye(A.shape[0]), X.T @ y)
    fit = X @ coef
    err = float(np.linalg.norm(y - fit) / max(np.linalg.norm(y), 1e-300))
    return (grid.with_values(fit.reshape(grid.shape)),
            coef.reshape(len(grid.positions), len(widths)), err)


def summary(grid: GridField) -> Dict[str, object]:
    """A few numbers that say what a grid is: shape, box volume, range and integral."""
    return {"name": grid.name, "kind": grid.kind, "shape": list(grid.shape),
            "periodic": bool(grid.periodic), "volume": grid.volume,
            "min": float(grid.values.min()), "max": float(grid.values.max()),
            "integral": grid.integral(), "atoms": len(grid.positions)}
