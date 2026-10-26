"""
Conceptual-DFT fields on a grid, and what atom-centred Gaussians can and cannot make of them.

The program already carries the *condensed* conceptual-DFT quantities -- a Hirshfeld f+ per atom,
a dual descriptor per atom -- and makes a field of them by putting one Gaussian of the field's own
width on each atom.  That discards the shape: a lone pair is a lobe to one side of the oxygen,
a pi system sits above and below the plane, and a single isotropic Gaussian centred on the nucleus
can say neither.  The quantity those numbers are condensed *from* is a function of position,
f(r) = rho_{N+1}(r) - rho_N(r), and ORCA will write it on a grid.

This module reads those grids (:func:`read_cube`, :func:`cdft_fields`, :func:`run_orca_plot`) and
tries to bring them into CMF's algebra the way the probe fields were brought in -- by fitting
several Gaussians per atom,

    F(r)  ~  sum_l sum_t  c_{l,t} exp(-beta_t |r - R_l|^2 / 2),

with the widths `beta_t` shared across atoms, which is what makes the overlap of two such fields
a closed-form double sum over atom pairs and term pairs, and so keeps the co-field, the
training-set decomposition and the atom contributions.

**The fit reproduces the field badly and the kernel well, and that difference is what this
module shows.**  Taking the two measurements in the order they were made:

*The field is not reproduced.*  On acrolein, fitted
against the valence part of the map (the raw maps peak *on the nuclei* -- every one of f+, f-,
the dual descriptor and the Parr functions has its maximum within 0.2 A of the oxygen, because a
density difference is dominated by core relaxation -- so the target is multiplied by
:func:`valence_window`):

===========================  ==========  ======  ==========
centres                      f+          dual    Parr P+
===========================  ==========  ======  ==========
nuclei only (8)              77%         89%     72%
+ pi centres at 0.8 A (16)   53%         69%     50%
+ pi centres and bonds (19)  47%         62%     41%
===========================  ==========  ======  ==========

the numbers being the relative error in the norm the kernel sees.  The fitter is not what fails:
on a field that *is* a sum of Gaussians on these widths it recovers the coefficients to within a
few per cent (``tests/test_cdft.py``).  The target is what cannot be represented.  A Fukui
function is essentially a frontier-orbital density, and acrolein's is pi*: three quarters of its
squared magnitude lies more than 0.5 A off the molecular plane, with a node *in* the plane, and an
isotropic Gaussian centred on a nucleus cannot make a node.  Putting centres above and below the
plane recovers a third of the error -- which confirms the diagnosis -- at the cost of doubling
the centres, and so quadrupling the kernel, and of giving up the one-to-one correspondence between
a centre and an atom that makes the atom contributions mean anything.

*The kernel is.*  A kernel is an inner product between molecules, and an error that falls the same
way for every molecule cancels in it.  On the twenty benzothiazoles of the corrosion series
(``benchmarks/run_cdft_kernels.py``), with the field kept exactly on a common grid and the kernel
done by quadrature, against the same field fitted and the kernel done in closed form:

=========================  =======  =======  =========  ==========
f+(r)                      alone    + the 5  r to grid  fit error
=========================  =======  =======  =========  ==========
grid quadrature            +0.638   +0.669   --         --
fitted Gaussians           +0.636   +0.658   +0.933     81%
condensed (one per atom)   +0.553   +0.684   +0.724     --
=========================  =======  =======  =========  ==========

and for the dual descriptor +0.522 / +0.556 / +0.448 alone, the fitted kernel again matching the
exact one (r = +0.899) while losing 91% of the field.  So the angular shape that the fit cannot
carry does not reach the kernel: as a *predictor* the fitted field is the grid field, at a
fraction of the cost and with CMF's algebra intact.

What the fit does carry that the condensed field does not is the *radial* shape -- four widths per
atom rather than one -- and that is worth something: alone, the fitted field beats the condensed
one by 0.08 in q2 on f+ and by 0.11 on the dual descriptor, and its kernel is much closer to the
exact one (+0.93 against +0.72).

Worth remembering before building on any of it: on that series none of the three helps.  The
five ordinary fields alone give +0.689, and the best quantum combination is +0.684.

Two practical lessons survive for whoever tries again:

* **Use a narrow width basis.**  With the probe widths (2.0 down to 0.08 A^-2) the coefficients
  reach +/-100 and cancel, on a field whose integral is 1; with 20 down to 0.5 A^-2 the largest
  is 0.5 and the fitted field integrates to +0.95 against a true +1.00.  Broad Gaussians on atoms
  a bond apart are very nearly the same function.
* **What survives the fit is the condensation.**  Once the coefficients are well conditioned, what
  the fit reproduces is each atom's share of the integral (:meth:`FieldFit.condensed`) -- which is
  what the Hirshfeld condensation in :mod:`cmfqsar.orca` already gives, more cheaply and without
  a grid.  The shape, which was the whole reason to go to the grid, is what is lost.

Nothing here is connected to the field machinery.  It computes the conceptual-DFT fields, fits
them, and reports how well -- by the measure that decides it, which is the kernel and not the
field.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

BOHR = 0.529177210903          # A per bohr

#: the density at which the window is half closed, in e/bohr^3; 0.2 sits between a valence
#: shell (1e-3 to 1e-2) and a first-row core (several e/bohr^3)
RHO_CORE = 0.2
#: how sharply it closes
WINDOW_POWER = 2.0

ELEMENTS = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S",
            17: "Cl", 35: "Br", 53: "I"}


@dataclass
class Cube:
    """A Gaussian cube, in angstrom: values on a regular grid plus the atoms it was made for."""
    values: np.ndarray                      # (n1, n2, n3)
    origin: np.ndarray                      # (3,)
    vectors: np.ndarray                     # (3, 3) step along each axis
    numbers: np.ndarray                     # (A,) atomic numbers
    positions: np.ndarray                   # (A, 3)
    path: str = ""

    @property
    def dV(self) -> float:
        return float(abs(np.linalg.det(self.vectors)))

    @property
    def symbols(self) -> List[str]:
        return [ELEMENTS.get(int(z), str(int(z))) for z in self.numbers]

    def points(self) -> np.ndarray:
        """(N, 3) coordinates of every grid point, in the cube's order."""
        axes = [self.origin[k] + np.arange(self.values.shape[k]) * self.vectors[k][k]
                for k in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def integral(self) -> float:
        return float(self.values.sum() * self.dV)

    def same_grid_as(self, other: "Cube") -> bool:
        return (self.values.shape == other.values.shape
                and np.allclose(self.origin, other.origin)
                and np.allclose(self.vectors, other.vectors))


def read_cube(path: str) -> Cube:
    """Read a Gaussian cube written by ``orca_plot``; lengths come back in angstrom.

    Cube files are in bohr, and a negative atom count marks the (rare) variant that carries a
    trailing list of orbital indices -- both are handled here rather than by the caller.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    head = lines[2].split()
    n_atoms = int(head[0])
    origin = np.array([float(v) for v in head[1:4]]) * BOHR
    shape, vectors = [], []
    for k in range(3):
        parts = lines[3 + k].split()
        shape.append(int(parts[0]))
        vectors.append([float(v) * BOHR for v in parts[1:4]])
    numbers, positions = [], []
    for i in range(abs(n_atoms)):
        parts = lines[6 + i].split()
        numbers.append(int(float(parts[0])))
        positions.append([float(v) * BOHR for v in parts[2:5]])
    rest = lines[6 + abs(n_atoms):]
    if n_atoms < 0:                      # a line of orbital indices precedes the data
        rest = rest[1:]
    values = np.fromstring(" ".join(rest), sep=" ")
    want = int(np.prod(shape))
    if values.size < want:
        raise ValueError(f"{path}: {values.size} values for a {shape} grid")
    values = values[:want].reshape(shape) / BOHR ** 3      # e/bohr^3 -> e/A^3
    return Cube(values, origin, np.array(vectors), np.array(numbers), np.array(positions), path)


def valence_window(rho: np.ndarray, rho_core: float = RHO_CORE,
                   power: float = WINDOW_POWER) -> np.ndarray:
    """1 where the density is low, 0 in the cores, smooth in between.

    ``w = 1 / (1 + (rho / rho_core)^power)`` with `rho` in e/bohr^3.  Smooth matters: a hard
    cut-off puts a step into the field, and a step is the one thing a sum of Gaussians cannot
    follow, so the fit would spend its terms on the edge instead of on the chemistry.
    """
    r = np.asarray(rho, float) * BOHR ** 3 / rho_core      # back to e/bohr^3, then relative
    return 1.0 / (1.0 + np.abs(r) ** power)


@dataclass
class FieldFit:
    """The fitted expansion of one conceptual-DFT field of one molecule."""
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (A, n_terms)
    widths: np.ndarray = field(default_factory=lambda: np.zeros(0))        # (n_terms,)
    rel_error: float = float("nan")      # ||F - fit|| / ||F|| over the fitted region
    r2: float = float("nan")
    name: str = ""

    def condensed(self) -> np.ndarray:
        """What each atom's share of the fitted field integrates to.

        The fit's own answer to the question the Hirshfeld condensation answers, and the natural
        thing to compare it with: sum_t c_t (2 pi / beta_t)^{3/2}, one number per atom.
        """
        return self.coeffs @ (2.0 * np.pi / self.widths) ** 1.5

    def evaluate(self, points: np.ndarray, positions: np.ndarray,
                 chunk: int = 200000) -> np.ndarray:
        """The fitted field at arbitrary points -- for checking it against the grid it came from."""
        points = np.asarray(points, float)
        out = np.empty(len(points))
        for lo in range(0, len(points), chunk):
            P = points[lo:lo + chunk]
            D2 = ((P[:, None, :] - positions[None, :, :]) ** 2).sum(-1)      # (n, A)
            B = np.exp(-0.5 * self.widths[None, None, :] * D2[:, :, None])   # (n, A, T)
            out[lo:lo + chunk] = np.einsum("nat,at->n", B, self.coeffs)
        return out


def fit_field(values: np.ndarray, cube: Cube, positions: Optional[np.ndarray] = None,
              window: Optional[np.ndarray] = None, n_terms: int = 4, ridge: float = 1e-6,
              name: str = "", chunk: int = 200000,
              widths: Optional[np.ndarray] = None) -> FieldFit:
    """Fit `values` on the grid of `cube` with `n_terms` Gaussians per atom.

    `window` multiplies the target before fitting -- pass :func:`valence_window` of the N-electron
    density to fit the valence part, which is what these maps mean.  The widths are the ones the
    probe fields share, so the resulting coefficients can be used directly in the probe kernel.

    The normal equations are accumulated in chunks: the design matrix of a 48-atom molecule on an
    80^3 grid is 100 million entries, and the matrix that matters, A'A, is 192 by 192.
    """
    from .probes import basis_widths
    pos = cube.positions if positions is None else np.asarray(positions, float)
    widths = basis_widths(n_terms) if widths is None else np.asarray(widths, float)
    n_terms = len(widths)
    target = np.asarray(values, float).reshape(-1)
    if window is not None:
        target = target * np.asarray(window, float).reshape(-1)
    points = cube.points()
    n_col = len(pos) * n_terms
    AtA = np.zeros((n_col, n_col))
    Aty = np.zeros(n_col)
    yty = 0.0
    for lo in range(0, len(points), chunk):
        P, y = points[lo:lo + chunk], target[lo:lo + chunk]
        D2 = ((P[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        B = np.exp(-0.5 * widths[None, None, :] * D2[:, :, None]).reshape(len(P), n_col)
        AtA += B.T @ B
        Aty += B.T @ y
        yty += float(y @ y)
    scale = max(np.trace(AtA) / max(n_col, 1), 1e-30)
    c = np.linalg.solve(AtA + ridge * scale * np.eye(n_col), Aty)
    # ||F - Ac||^2 = y'y - 2 c'A'y + c'A'A c, without ever forming the residual vector
    resid2 = max(yty - 2.0 * float(c @ Aty) + float(c @ AtA @ c), 0.0)
    rel = float(np.sqrt(resid2 / yty)) if yty > 0 else float("nan")
    mean = target.mean()
    ss_tot = yty - len(target) * mean ** 2
    r2 = float(1.0 - resid2 / ss_tot) if ss_tot > 0 else float("nan")
    return FieldFit(c.reshape(len(pos), n_terms), widths, rel, r2, name)


def overlap_kernel(coeffs: Sequence[np.ndarray], positions: Sequence[np.ndarray],
                   widths: np.ndarray) -> np.ndarray:
    """The closed-form overlap integrals between fitted fields: K[i, j] = int F_i F_j dr.

    The same formula the probe branch of :mod:`cmfqsar.kernels` evaluates, written here over
    explicit coefficient matrices so that a fitted field can be scored without being connected to
    a FieldSet.  Two Gaussians of widths b1, b2 a distance D apart overlap by
    ``(2 pi / (b1 + b2))^{3/2} exp(-b1 b2 D^2 / (2 (b1 + b2)))``; the rest is bookkeeping over
    pairs of centres and pairs of terms.
    """
    widths = np.asarray(widths, float)
    n = len(coeffs)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            D2 = ((np.asarray(positions[i])[:, None, :]
                   - np.asarray(positions[j])[None, :, :]) ** 2).sum(-1)
            total = 0.0
            for k1, b1 in enumerate(widths):
                for k2, b2 in enumerate(widths):
                    bs = b1 + b2
                    E = np.exp(-(b1 * b2) / (2.0 * bs) * D2) * (2.0 * np.pi / bs) ** 1.5
                    total += float(coeffs[i][:, k1] @ E @ coeffs[j][:, k2])
            K[i, j] = K[j, i] = total
    return K


# ----------------------------------------------------------------------------------------------
# the conceptual-DFT fields themselves, from the cubes of the three states
# ----------------------------------------------------------------------------------------------

#: name -> (how it is made from the three densities and the two spin densities)
CDFT_FIELDS = ("rho", "fukui_plus", "fukui_minus", "fukui_zero", "dual", "parr_plus", "parr_minus")


def cdft_fields(rho_n: Cube, rho_np1: Optional[Cube] = None, rho_nm1: Optional[Cube] = None,
                spin_np1: Optional[Cube] = None,
                spin_nm1: Optional[Cube] = None) -> dict:
    """The conceptual-DFT fields as arrays on the common grid of the cubes given.

    Finite differences at fixed geometry, which is what the condensed indices of
    :mod:`cmfqsar.orca` are made of as well -- the same quantities before they are condensed.
    """
    for other in (rho_np1, rho_nm1, spin_np1, spin_nm1):
        if other is not None and not rho_n.same_grid_as(other):
            raise ValueError("the cubes are on different grids; plot them at one geometry")
    out = {"rho": rho_n.values}
    if rho_np1 is not None:
        out["fukui_plus"] = rho_np1.values - rho_n.values
    if rho_nm1 is not None:
        out["fukui_minus"] = rho_n.values - rho_nm1.values
    if "fukui_plus" in out and "fukui_minus" in out:
        out["fukui_zero"] = 0.5 * (out["fukui_plus"] + out["fukui_minus"])
        out["dual"] = out["fukui_plus"] - out["fukui_minus"]
    if spin_np1 is not None:
        out["parr_plus"] = spin_np1.values
    if spin_nm1 is not None:
        out["parr_minus"] = spin_nm1.values
    return out


def fit_cdft(cubes: dict, rho_n: Cube, names: Sequence[str] = CDFT_FIELDS, n_terms: int = 4,
             rho_core: float = RHO_CORE, ridge: float = 1e-6) -> dict:
    """Fit every named conceptual-DFT field of one molecule; returns {name: FieldFit}."""
    window = valence_window(rho_n.values, rho_core)
    out = {}
    for name in names:
        if name not in cubes:
            continue
        out[name] = fit_field(cubes[name], rho_n, window=window, n_terms=n_terms,
                              ridge=ridge, name=name)
    return out


# ----------------------------------------------------------------------------------------------
# running orca_plot
# ----------------------------------------------------------------------------------------------

#: orca_plot menu: plot type, then the density kind, then accept its name, then the resolution
_PLOT_KEYS = "1\n{kind}\ny\n4\n{intervals}\n11\n12\n"
PLOT_DENSITY, PLOT_SPIN = 2, 3


def run_orca_plot(gbw: str, kind: int = PLOT_DENSITY, intervals: int = 80,
                  orca_plot: Optional[str] = None, timeout: Optional[float] = 1800.0,
                  reuse: bool = True) -> str:
    """Write a cube beside `gbw` with ``orca_plot``, and return its path.

    ORCA 6 has no ``%plot`` input block -- it is rejected outright -- so the cube can only come
    from the separate program, which is menu-driven and is therefore driven by its menu.  The box
    is the one orca_plot chooses from the geometry, which is what makes the three states of a
    molecule share a grid exactly.
    """
    import subprocess
    exe = orca_plot or _default_orca_plot()
    base = gbw[:-4] if gbw.lower().endswith(".gbw") else gbw
    out = f"{base}." + {PLOT_DENSITY: "eldens", PLOT_SPIN: "spindens"}[kind] + ".cube"
    if reuse and os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(gbw):
        return out          # a cube costs minutes on a big molecule; it is worth keeping
    keys = _PLOT_KEYS.format(kind=kind, intervals=int(intervals))
    subprocess.run([exe, os.path.basename(gbw), "-i"], cwd=os.path.dirname(os.path.abspath(gbw)),
                   input=keys, text=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if not os.path.isfile(out):
        raise RuntimeError(f"orca_plot wrote no cube for {gbw}")
    return out


def _default_orca_plot() -> str:
    from .orca import find_orca
    d = os.path.dirname(os.path.abspath(find_orca()))
    for name in ("orca_plot.exe", "orca_plot"):
        cand = os.path.join(d, name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError("orca_plot not found beside the ORCA executable")
