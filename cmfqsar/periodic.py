"""Continuous molecular fields of crystals: the periodic field autocorrelation.

The autocorrelation kernel of ``invariant.py`` compares two molecules by the distance
distributions of their field weights, which is the real-space form of the orientation-averaged
product of their typed structure factors (the Debye scattering equation).  A crystal is the same
object with three differences, and this module is those differences:

* **The pair sum runs over a neighbour list.**  A molecule has finitely many pairs; a crystal has
  one atom per cell times every neighbour out to a cutoff radius, periodic images included, and
  the sum is divided by the number of atoms in the cell so that the descriptor is intensive: a
  supercell describes the same material, and gives the same channels.

* **The uniform background is removed.**  In an infinite solid the number of pairs at distance r
  grows as r^2, the same for every structure, and dominates everything else at long distance.  What
  the Faber-Ziman partial structure factors keep is the deviation from a uniform distribution of
  each field over the cell volume, and that is what is subtracted here, pair of field types by
  pair of field types.  The l = m self term stays, as it does in the molecular descriptor: it
  carries the composition, the mean of w_f w_g over the atoms.

* **The cutoff is tapered.**  A hard cutoff produces oscillations; the last ``taper`` angstrom
  before ``r_max`` are weighted by a cosine that falls smoothly to zero, so that a pair leaving the
  sphere leaves the descriptor continuously.

Everything else is shared: the Gaussian along the distance axis, the pair-of-field-types
channels, the per-block scaling and cosine normalisation of the kernel, the channel labels, and
the learners.  The descriptor is invariant to rotation, translation, the choice of origin and of
unit cell by construction, and cannot see what any pair-distance distribution cannot see:
enantiomorphs and homometric structures look the same, and so does anything beyond ``r_max``.

Fields for inorganic solids
---------------------------
The classical hydrophobic and hydrogen-bond fields mean nothing in an oxide.  What carries over
is the *electrostatic* field with oxidation states as weights, the *steric* field with the atomic
volume (the cube of the covalent radius), the *electronegativity*, the atomic *mass* or its inverse
square root (``prop:mass``, ``prop:inverse_sqrt_mass``) as property fields, and above all the
*continuous indicator fields* of the 2015 paper, one per element: their pair products are
the partial structure factors of each element pair, which is what a diffraction experiment
measures.  :func:`inorganic_fieldset` builds such a set and :func:`crystal_fields` evaluates the
weights for one structure.  Any per-atom array a structure file carries (Bader charges, magnetic
moments, ...) can be a field as ``prop:<name>``.

ASE is needed for the neighbour list and to read structure files; pymatgen is needed for the
oxidation-state guess and the electronegativities and nothing else.  Both are imported lazily, so
that the rest of the program does not depend on them.
"""
from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, FieldSet, FieldSpec
from .invariant import block_sizes, channel_labels, field_pairs

INORGANIC_FIELDS: Tuple[str, ...] = ("electrostatic", "steric", "prop:electronegativity")
OTHER_ELEMENT = "other"          # the indicator field that collects the elements not listed


# --------------------------------------------------------------------------- #
# The descriptor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeriodicSpec:
    """The distance grid, cutoff, taper and background treatment of the periodic autocorrelation.

    The grid is laid out like :class:`invariant.AutocorrelationSpec`: channels at 0, step, 2 step,
    ... up to ``r_max``, each a Gaussian of width ``sigma`` along the distance axis.  ``chirality``
    exists only so that the block and label helpers of ``invariant.py`` accept this spec; the
    two-point descriptor of a crystal has no signed block.
    """
    r_max: float = 10.0
    step: float = 0.5
    sigma: float = 0.5
    include_self: bool = True        # the l = m terms: the composition channel at d = 0
    background: bool = True          # subtract the uniform-density term of every field pair
    taper: float = 2.0               # width of the cosine taper below r_max (0: hard cutoff)
    chirality: bool = False          # always False; see the class docstring

    @property
    def d_max(self) -> float:
        return self.r_max

    @property
    def centres(self) -> np.ndarray:
        n = max(int(round(self.r_max / self.step)) + 1, 1)
        return np.arange(n, dtype=float) * self.step

    @property
    def n_bins(self) -> int:
        return len(self.centres)

    def window(self, r: np.ndarray) -> np.ndarray:
        """1 inside the sphere, a half-cosine over the last ``taper`` angstrom, 0 beyond."""
        r = np.asarray(r, float)
        w = np.where(r < self.r_max, 1.0, 0.0)
        if self.taper > 0:
            r0 = self.r_max - self.taper
            inside = (r > r0) & (r < self.r_max)
            w = np.where(inside, 0.5 * (1.0 + np.cos(np.pi * (r - r0) / self.taper)), w)
        return w

    def shell_integral(self, n_points: int = 4000) -> np.ndarray:
        """``∫ 4π r² exp(-(r - d_k)² / 2σ²) w(r) dr`` for every channel d_k.

        The pair count a uniform distribution of unit density puts into each channel.  It depends
        on the grid, the width and the window only, so it is computed once per spec.
        """
        r = np.linspace(0.0, self.r_max, int(n_points))
        g = np.exp(-((r[:, None] - self.centres[None, :]) ** 2) / (2.0 * self.sigma ** 2))
        f = 4.0 * np.pi * r[:, None] ** 2 * g * self.window(r)[:, None]
        return np.trapezoid(f, r, axis=0)


@dataclass
class CrystalFields:
    """Atom-level data of one periodic structure: what :class:`fields.MoleculeFields` is to a
    molecule.  It carries what an :class:`fields.AtomTable` and the periodic kernel need, and
    the unit cell."""
    name: str
    coords: np.ndarray            # (N, 3) Cartesian positions of the atoms of one cell
    weights: np.ndarray           # (N, F) field weights
    radii: np.ndarray             # (N,) covalent radii
    symbols: List[str]
    cell: np.ndarray              # (3, 3) lattice vectors as rows
    pbc: Tuple[bool, bool, bool] = (True, True, True)
    charges: Optional[np.ndarray] = None
    charge_source: str = ""
    atom_types: List[str] = field(default_factory=list)
    bonds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    molblock: str = ""
    n_conformers: int = 1
    conformer_weights: Optional[np.ndarray] = None
    atoms_per_conformer: int = 0
    descriptors: Optional[np.ndarray] = None
    probe_coeffs: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def is_ensemble(self) -> bool:
        return False

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    def to_atoms(self):
        from ase import Atoms
        return Atoms(symbols=list(self.symbols), positions=np.asarray(self.coords, float),
                     cell=np.asarray(self.cell, float), pbc=self.pbc)


@functools.lru_cache(maxsize=32)
def _shell_integral(spec: PeriodicSpec) -> np.ndarray:
    """The shell integral of a spec, computed once: it does not depend on the structure."""
    out = spec.shell_integral()
    out.flags.writeable = False
    return out


def _neighbour_pairs(crystal: CrystalFields, r_max: float):
    """Ordered atom pairs (i, j, d) with 0 < d < r_max, periodic images included."""
    from ase.neighborlist import neighbor_list
    atoms = crystal.to_atoms()
    i, j, d = neighbor_list("ijd", atoms, float(r_max))
    return np.asarray(i, int), np.asarray(j, int), np.asarray(d, float)


def periodic_autocorrelation(crystal: CrystalFields, spec: PeriodicSpec = PeriodicSpec(),
                             pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """Periodic field autocorrelation of one structure, flattened: (n_pairs * n_bins,).

    For a pair of field types (f, g) and a channel at distance d_k,

        Phi_fg(d_k) = (1/N) [ sum_{l in cell} sum_{m: r_lm < r_max} w_lf w_mg
                                     exp(-(r_lm - d_k)^2 / 2 sigma^2) w(r_lm)
                               - S_f S_g / V * I_k ],

    with N the atoms in the cell, S_f the sum of the weights of field f over the cell, V its
    volume and I_k the shell integral of the spec.  The sum runs over ordered pairs, so a pair of
    different field types is counted in both orders, as the molecular descriptor counts it; the
    background of such a pair is accordingly 2 S_f S_g / V * I_k.
    """
    W = np.asarray(crystal.weights, float)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    centres = spec.centres
    if n_atoms == 0:
        return np.zeros(len(pairs) * len(centres))
    i, j, d = _neighbour_pairs(crystal, spec.r_max)
    inv = 2.0 * spec.sigma ** 2
    if len(d):
        B = np.exp(-((d[:, None] - centres[None, :]) ** 2) / inv) * spec.window(d)[:, None]
    else:
        B = np.zeros((0, len(centres)))
    self_row = np.exp(-(centres ** 2) / inv) if spec.include_self else np.zeros(len(centres))
    shell = _shell_integral(spec) if spec.background else None
    S = W.sum(axis=0)
    V = crystal.volume
    # T[f, k, g] = sum over ordered pairs of w_if B_pk w_jg, for every pair of field types at once
    T = np.zeros((n_fields, len(centres), n_fields))
    if len(d):
        Wi, Wj = W[i], W[j]
        for k in range(len(centres)):                      # one BLAS product per channel
            T[:, k, :] = (Wi * B[:, k:k + 1]).T @ Wj
    G = W.T @ W                                            # the self terms, sum_l w_lf w_lg
    fi = np.array([f for f, _ in pairs], int)
    gi = np.array([g for _, g in pairs], int)
    diag = fi == gi
    mult = np.where(diag, 1.0, 2.0)                        # a pair of different types, both ways
    rows = T[fi, :, gi] + np.where(diag[:, None], 0.0, T[gi, :, fi])
    rows = rows + (mult * G[fi, gi])[:, None] * self_row[None, :]
    if shell is not None and V > 0:
        rows = rows - (mult * S[fi] * S[gi] / V)[:, None] * shell[None, :]
    return (rows / n_atoms).ravel()


def periodic_feature_matrix(crystals: Sequence[CrystalFields],
                            spec: PeriodicSpec = PeriodicSpec(),
                            n_fields: Optional[int] = None, progress=None) -> np.ndarray:
    """Periodic autocorrelation features of a set of structures, one row each."""
    if not len(crystals):
        return np.zeros((0, 0))
    n_fields = n_fields if n_fields is not None else crystals[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    width = sum(block_sizes(spec, n_fields))
    out = np.zeros((len(crystals), width))
    for k, c in enumerate(crystals):
        if c.weights.shape[1] != n_fields:
            raise ValueError(
                f"{c.name or f'structure {k}'} carries {c.weights.shape[1]} fields where the set "
                f"carries {n_fields}: build the fields of every structure from one field set.")
        out[k] = periodic_autocorrelation(c, spec, pairs)
        if progress is not None and k % 25 == 0:
            progress(f"Periodic autocorrelation {k + 1}/{len(crystals)}", k / max(len(crystals), 1))
    return out


def periodic_block_sizes(spec: PeriodicSpec, n_fields: int) -> List[int]:
    return block_sizes(spec, n_fields)


def periodic_channel_labels(fieldset: FieldSet, spec: PeriodicSpec) -> List[str]:
    """One label per feature: which pair of field types, at which distance."""
    return channel_labels(fieldset, spec)


# --------------------------------------------------------------------------- #
# Fields for solids
# --------------------------------------------------------------------------- #
def inorganic_fieldset(elements: Sequence[str] = (), physchem: Sequence[str] = INORGANIC_FIELDS,
                       charge_model: str = "oxidation", other: bool = True,
                       alpha: float = 1.0) -> FieldSet:
    """A field set for solids: the physico-chemical fields named, one indicator field per element
    listed (``cif:<symbol>``) and, with ``other``, one more for every element not listed.

    ``charge_model`` says where the electrostatic weights come from: ``"oxidation"`` guesses
    oxidation states from the composition (pymatgen), falling back to the electronegativity proxy
    when no assignment exists; ``"electronegativity"`` uses that proxy throughout: the deviation
    of each atom's Pauling electronegativity from the mean over the cell, with the sign of a
    charge (an electronegative atom is negative); ``"file"`` reads ``oxi_states`` or ``charges``
    from the structure's per-atom arrays and refuses when there are none.
    """
    specs = []
    for name in physchem:
        kind = "steric" if name == "steric" else "gauss"
        specs.append(FieldSpec(name=name, kind=kind, alpha=alpha))
    for el in elements:
        specs.append(FieldSpec(name=f"cif:{el}", kind="gauss", alpha=alpha))
    if other and elements:
        specs.append(FieldSpec(name=f"cif:{OTHER_ELEMENT}", kind="gauss", alpha=alpha))
    return FieldSet(specs=specs, charge_model=charge_model, cif_scheme="element")


def elements_in(structures: Iterable, min_structures: int = 5) -> List[str]:
    """The element symbols present in at least ``min_structures`` of the structures, sorted."""
    counts: Dict[str, int] = {}
    for s in structures:
        for el in set(_symbols_of(s)):
            counts[el] = counts.get(el, 0) + 1
    return sorted(el for el, n in counts.items() if n >= min_structures)


def _symbols_of(s) -> List[str]:
    if isinstance(s, CrystalFields):
        return list(s.symbols)
    return list(s.get_chemical_symbols())


_EN_CACHE: Dict[str, float] = {}
_OXI_CACHE: Dict[str, Optional[Dict[str, float]]] = {}


def electronegativity(symbol: str) -> float:
    """Pauling electronegativity (pymatgen's table); NaN where none is defined (noble gases)."""
    if symbol not in _EN_CACHE:
        from pymatgen.core import Element
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = float(Element(symbol).X)
        _EN_CACHE[symbol] = x
    return _EN_CACHE[symbol]


def oxidation_states(symbols: Sequence[str]) -> Optional[Dict[str, float]]:
    """A charge-balanced assignment of common oxidation states for the composition, or None.

    pymatgen's ``oxi_state_guesses`` on the reduced composition; the first guess is taken,
    which is the one with the most common states.  Elemental solids get zero.
    """
    from collections import Counter
    counts = Counter(symbols)
    key = " ".join(f"{el}{n}" for el, n in sorted(counts.items()))
    if key in _OXI_CACHE:
        return _OXI_CACHE[key]
    result: Optional[Dict[str, float]] = None
    if len(counts) == 1:
        result = {next(iter(counts)): 0.0}
    else:
        try:
            from pymatgen.core import Composition
            guesses = Composition(key).oxi_state_guesses(max_sites=-1)
            if guesses:
                result = {el: float(v) for el, v in guesses[0].items()}
        except Exception:
            result = None
    _OXI_CACHE[key] = result
    return result


def _electronegativity_charges(symbols: Sequence[str]) -> np.ndarray:
    x = np.array([electronegativity(s) for s in symbols], float)
    x = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
    return x.mean() - x


def crystal_charges(atoms, model: str = "oxidation") -> Tuple[np.ndarray, str]:
    """Per-atom charges for the electrostatic field of a solid, and where they came from."""
    symbols = list(atoms.get_chemical_symbols())
    if model == "file":
        for key in ("oxi_states", "charges"):
            if key in atoms.arrays:
                return np.asarray(atoms.arrays[key], float), key
        q = atoms.get_initial_charges()
        if np.any(q != 0):
            return np.asarray(q, float), "initial_charges"
        raise ValueError(f"structure {atoms.get_chemical_formula()} carries no per-atom "
                         f"'oxi_states' or 'charges' array and no initial charges")
    if model == "oxidation":
        ox = oxidation_states(symbols)
        if ox is not None:
            return np.array([ox[s] for s in symbols], float), "oxidation"
        return _electronegativity_charges(symbols), "electronegativity"
    if model == "electronegativity":
        return _electronegativity_charges(symbols), "electronegativity"
    raise ValueError(f"unknown charge model for solids: {model!r}")


def crystal_fields(atoms, fieldset: FieldSet, name: str = "") -> CrystalFields:
    """Evaluate the field weights of one periodic structure (an ASE ``Atoms``)."""
    from ase.data import atomic_masses, atomic_numbers, covalent_radii
    symbols = list(atoms.get_chemical_symbols())
    n = len(symbols)
    radii = np.array([covalent_radii[atomic_numbers[s]] for s in symbols], float)
    listed = {sp.name[4:] for sp in fieldset.specs if sp.is_cif and sp.name[4:] != OTHER_ELEMENT}
    charges, source = None, ""
    cols = []
    for sp in fieldset.specs:
        if sp.name == "electrostatic":
            if charges is None:
                charges, source = crystal_charges(atoms, fieldset.charge_model)
            cols.append(charges)
        elif sp.name == "steric":
            cols.append(radii ** 3)
        elif sp.name == "prop:electronegativity":
            x = np.array([electronegativity(s) for s in symbols], float)
            cols.append(np.where(np.isfinite(x), x, 0.0))
        elif sp.name == "prop:mass":
            cols.append(np.array([atomic_masses[atomic_numbers[s]] for s in symbols], float))
        elif sp.name == "prop:inverse_sqrt_mass":
            # a vibrational frequency goes as 1/sqrt(m): the field of the light atoms
            cols.append(np.array([atomic_masses[atomic_numbers[s]] ** -0.5 for s in symbols]))
        elif sp.name.startswith("prop:"):
            key = sp.name[5:]
            if key not in atoms.arrays:
                if fieldset.prop_missing == "zero":
                    cols.append(np.zeros(n))
                    continue
                raise ValueError(f"structure {name or atoms.get_chemical_formula()} carries no "
                                 f"per-atom array {key!r} for the field {sp.name}")
            cols.append(np.asarray(atoms.arrays[key], float).reshape(n))
        elif sp.is_cif:
            el = sp.name[4:]
            if el == OTHER_ELEMENT:
                cols.append(np.array([0.0 if s in listed else 1.0 for s in symbols]))
            else:
                cols.append(np.array([1.0 if s == el else 0.0 for s in symbols]))
        else:
            raise ValueError(f"the field {sp.name!r} is not defined for a periodic solid; use "
                             f"electrostatic, steric, prop:<array> or cif:<element>")
    W = np.column_stack(cols) if cols else np.zeros((n, 0))
    return CrystalFields(name=name or atoms.get_chemical_formula(),
                         coords=np.asarray(atoms.get_positions(), float), weights=W, radii=radii,
                         symbols=symbols, cell=np.asarray(atoms.get_cell()[:], float),
                         pbc=tuple(bool(p) for p in atoms.get_pbc()),
                         charges=charges, charge_source=source, atom_types=list(symbols))


def crystal_table(structures: Sequence, fieldset: FieldSet,
                  names: Optional[Sequence[str]] = None) -> AtomTable:
    """An :class:`fields.AtomTable` of periodic structures, which the model takes like any other."""
    names = list(names) if names is not None else [""] * len(structures)
    return AtomTable.from_molecules([crystal_fields(a, fieldset, nm)
                                     for a, nm in zip(structures, names)])


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_structures(path: str, index: str = ":") -> list:
    """Every structure of an ASE-readable file (CIF, POSCAR, extended XYZ, ASE JSON, ...)."""
    from ase.io import read
    out = read(path, index=index)
    return list(out) if isinstance(out, (list, tuple)) else [out]


def structures_from_matbench(table: dict, target: Optional[str] = None):
    """The structures and target of a Matbench task file (``{"index", "columns", "data"}``,
    structures as pymatgen dictionaries), as ASE ``Atoms`` and a float array.  Partial
    occupancies are not handled; the Matbench tasks have none."""
    from ase import Atoms
    cols = list(table.get("columns", []))
    if target is None:
        target = next(c for c in cols if c != "structure")
    si, ti = cols.index("structure"), cols.index(target)
    atoms, y, names = [], [], []
    for idx, row in zip(table.get("index", range(len(table["data"]))), table["data"]):
        s = row[si]
        cell = np.asarray(s["lattice"]["matrix"], float)
        symbols, xyz = [], []
        for site in s["sites"]:
            symbols.append(max(site["species"], key=lambda sp: sp.get("occu", 1.0))["element"])
            xyz.append(site["xyz"])
        atoms.append(Atoms(symbols=symbols, positions=np.asarray(xyz, float), cell=cell, pbc=True))
        y.append(float(row[ti]))
        names.append(str(idx))
    return atoms, np.array(y), names


# --------------------------------------------------------------------------- #
# Reading a fitted model: the co-pattern
# --------------------------------------------------------------------------- #
def channel_weights(model) -> np.ndarray:
    """The weight of every channel in a fitted linear model on an invariant kernel.

    The kernel is the dot product of the scaled, unit-normalised channel vectors, so the
    prediction is ``b + sum_c w_c phi_c(x)`` with ``phi`` the scaled and normalised channels of
    ``x`` and ``w = sum_j a_j phi_j`` over the training structures.  That weight vector is to the
    autocorrelation kernels what the co-field is to the overlap kernel: which pair of field types
    at which distance raises the property, and which lowers it.  Defined for the linear layer
    only; a Gaussian or arc-cosine layer has no such vector.
    """
    F = getattr(model, "ac_train_", None)
    if F is None or model.coef is None:
        raise RuntimeError("the model is not fitted on an invariant kernel")
    if not model.is_linear:
        raise ValueError("channel weights are defined for the linear kernel layer only")
    Z = np.asarray(F, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    return Z.T @ np.asarray(model.coef, float)


def top_channels(model, labels: Sequence[str], n: int = 10) -> List[Tuple[str, float, float]]:
    """The ``n`` channels that move the prediction most, as (label, weight, spread).

    ``spread`` is the standard deviation of the scaled, normalised channel over the training
    structures: a weight only matters where the channel varies, so the ranking is by
    ``|weight| * spread`` and both are reported.
    """
    w = channel_weights(model)
    Z = np.asarray(model.ac_train_, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    sd = Z.std(axis=0)
    order = np.argsort(-np.abs(w) * sd)[:n]
    return [(labels[c], float(w[c]), float(sd[c])) for c in order]
