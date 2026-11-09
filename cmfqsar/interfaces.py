"""Reading a fitted model from another program (plan item MS7).

A multiscale scheme needs the model as a component, not as a report. Three interfaces are what
the other programs ask for, and this module holds them.

**An ASE calculator.** :class:`CMFCalculator` answers an :class:`ase.Atoms` object with the
model's prediction as its energy and, where the kernel has an analytic gradient (plan item MS3),
the negative gradient as its forces. Anything that drives atoms through the ASE interface -- a
relaxation, a molecular-dynamics run, a nudged elastic band -- can then drive them with a fitted
model. The prediction is an activity and not an energy, so what the calculator means by "energy"
is said plainly below and controlled by one argument.

**A barrier per migration path.** :class:`HopBarriers` is the interface the hop model of plan item
BT7 is read through: a periodic structure goes in, and out comes one predicted barrier per
symmetry-distinct migration path of the mobile ion, with the path's two end sites and its length,
in a table a kinetic Monte Carlo or a percolation analysis reads. The model is the fast level of
the hierarchy and this is the form the coarse level consumes.

**A field on a grid.** :func:`write_cofield_cube` writes the co-field of a fitted model as a
Gaussian cube file, which is what a visualiser or a grid-based code reads. For a model on one
lattice (plan item M4) the cube is the cell itself, with the lattice vectors in its header, so
the file is a periodic field and not a box cut out of one.

The map of plan item MS2 needs no new interface: :func:`cmfqsar.sitemap.write_maps` already writes
one row per place. It is re-exported here so that one import gives every way out of a model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .sitemap import SiteMap, site_map, site_maps, write_maps  # noqa: F401  (re-exported)

#: what the calculator can answer
PROPERTIES = ("energy", "free_energy", "forces")


# --------------------------------------------------------------------------- #
# An ASE calculator
# --------------------------------------------------------------------------- #
def _ase_calculator_base():
    """The ASE base class, imported late so that the module works without ASE installed."""
    try:
        from ase.calculators.calculator import Calculator
    except Exception as exc:                                   # pragma: no cover - no ASE
        raise ImportError(
            "an ASE calculator needs the ase package: pip install ase. The rest of this module "
            "works without it.") from exc
    return Calculator


def _fields_of(model, atoms, name: str = "structure"):
    """The field record of one ASE structure, built the way the model's training set was.

    A periodic model takes the cell. A molecular model takes the atoms as a molecule, and an ASE
    structure carries no bonds, so they are perceived from the coordinates by the same code that
    reads an XYZ file (:func:`cmfqsar.complexes.molecule_from_coordinates` and RDKit's
    ``rdDetermineBonds``). The charge is ``atoms.info["charge"]`` when the structure states one
    and zero otherwise, because the coordinates do not say.
    """
    periodic = bool(getattr(model, "pbc_overlap", False)) or model.kernel_source == "periodic"
    if periodic:
        from .periodic import crystal_fields
        return crystal_fields(atoms, model.fieldset, name)
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    from .complexes import molecule_from_coordinates
    from .fields import build_fields
    info = dict(getattr(atoms, "info", {}) or {})
    charge = int(round(float(info.get("charge", 0))))
    symbols = [str(s) for s in atoms.get_chemical_symbols()]
    mol = molecule_from_coordinates(symbols, np.asarray(atoms.get_positions(), float))
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge, useHueckel=False)
        Chem.SanitizeMol(mol)
    except Exception as exc:                 # noqa: BLE001 - say which structure, and why
        raise ValueError(
            f"the bonds of {name or 'this structure'} could not be worked out from its "
            f"coordinates at charge {charge}, so the fields of a molecular model cannot be "
            "built from it. State the charge in atoms.info['charge'], or build the molecule "
            "with RDKit and use the model directly.") from exc
    mol.SetProp("_Name", name)
    return build_fields(mol, model.fieldset, name)


class CMFCalculator:
    """A fitted model as an ASE calculator: the prediction as the energy (plan item MS7).

    What the calculator calls the energy is ``sign`` times the model's prediction, in the units of
    the activity and not in eV. With the default ``sign=+1`` an ASE optimiser, which minimises,
    moves the structure towards a *lower* prediction; ``sign=-1`` makes it move towards a higher
    one, which is the direction a design loop wants. Nothing here converts units, because a
    predicted activity has no energy scale: a relaxation under this calculator is a relaxation of
    the prediction, and the step sizes of the optimiser have to be chosen for it.

    Forces are ``-sign`` times the gradient of the prediction with respect to the atom positions,
    from :func:`cmfqsar.gradients.prediction_gradient`. They exist for the kernels item MS3
    differentiates -- the overlap integral and the rotation-invariant autocorrelation -- and the
    others raise with the reason rather than returning zeros. A model with no gradient is still
    usable for energies alone, which is what a single-point screen needs.

    The gradient knows nothing about valence, as item MS3 measured: following it alone pulls bonds
    out of shape. This calculator is a term to add to a force field, not a force field.
    """

    #: ASE reads this to know what the calculator can answer
    implemented_properties = list(PROPERTIES)

    def __init__(self, model, sign: float = 1.0, name: str = "structure",
                 forces: bool = True, label: Optional[str] = None):
        if getattr(model, "coef", None) is None:
            raise ValueError("this model is not fitted: there is nothing for a calculator to "
                             "answer with")
        self.model = model
        self.sign = float(sign)
        self.name = str(name)
        self.want_forces = bool(forces)
        self.label = label
        self.results: Dict[str, object] = {}
        self.atoms = None
        self._base = _ase_calculator_base()                    # fails early without ASE
        # what this calculator can answer for this model, decided once
        self.has_forces = bool(self.want_forces and self._gradient_reason() is None)
        if not self.has_forces:
            self.implemented_properties = ["energy", "free_energy"]

    # ------------------------------------------------------------------ what it can do
    def _gradient_reason(self) -> Optional[str]:
        """Why this model has no forces, or None when it has them."""
        from .gradients import _check_model
        try:
            _check_model(self.model)
        except Exception as exc:
            return str(exc)
        return None

    def __repr__(self) -> str:
        how = "energies and forces" if self.has_forces else "energies only"
        return (f"CMFCalculator({self.model.kernel_source or 'fields'} kernel, {how}, "
                f"sign {self.sign:+g})")

    # ------------------------------------------------------------------ the ASE protocol
    def get_potential_energy(self, atoms=None, force_consistent: bool = False) -> float:
        a = self._atoms(atoms)
        from .fields import AtomTable
        rec = _fields_of(self.model, a, self.name)
        value = float(np.asarray(self.model.predict(AtomTable.from_molecules([rec])),
                                 float).ravel()[0])
        out = self.sign * value
        self.results = {"energy": out, "free_energy": out, "prediction": value}
        return out

    def get_potential_energies(self, atoms=None):
        raise NotImplementedError(
            "the prediction belongs to the structure as a whole, so there is no energy per atom. "
            "cmfqsar.sitemap.site_map splits it over the places instead, which is a share of the "
            "prediction and not an energy.")

    def get_forces(self, atoms=None) -> np.ndarray:
        a = self._atoms(atoms)
        reason = self._gradient_reason()
        if reason is not None:
            from ase.calculators.calculator import PropertyNotImplementedError
            raise PropertyNotImplementedError(
                "this model has no analytic gradient, so the calculator gives energies only: "
                + reason)
        from .gradients import prediction_gradient
        rec = _fields_of(self.model, a, self.name)
        g = prediction_gradient(self.model, rec, wrt="coords")
        out = -self.sign * np.asarray(g.coords, float)
        value = float(g.value)
        self.results = {"energy": self.sign * value, "free_energy": self.sign * value,
                        "prediction": value, "forces": out}
        return out

    def get_stress(self, atoms=None):
        raise NotImplementedError(
            "the derivative with respect to the cell is not implemented: the periodic kernels are "
            "the ones plan item MS3 left without a gradient.")

    def get_property(self, name: str, atoms=None, allow_calculation: bool = True):
        if name in ("energy", "free_energy"):
            return self.get_potential_energy(atoms)
        if name == "forces":
            return self.get_forces(atoms)
        from ase.calculators.calculator import PropertyNotImplementedError
        raise PropertyNotImplementedError(f"this calculator does not answer '{name}'")

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        """The entry ASE's own machinery calls; it fills `results`."""
        self.get_potential_energy(atoms)
        if "forces" in tuple(properties) and self.has_forces:
            self.get_forces(atoms)
        return self.results

    def check_state(self, atoms, tol: float = 1e-15):
        """Nothing is cached between structures, so every call is a fresh calculation."""
        return ["positions"]

    def reset(self):
        self.results = {}

    def _atoms(self, atoms):
        a = atoms if atoms is not None else self.atoms
        if a is None:
            raise ValueError("no structure was given: pass atoms, or attach the calculator to "
                             "one with atoms.calc = this")
        self.atoms = a
        return a


# --------------------------------------------------------------------------- #
# A barrier per migration path
# --------------------------------------------------------------------------- #
@dataclass
class BarrierTable:
    """One predicted barrier per symmetry-distinct migration path of one structure.

    `rows` holds a dictionary per path: the two end sites, the length in angstrom, the midpoint,
    how many paths of the structure that one stands for, and the predicted barrier in the units
    the model was fitted on. `hopset` is the path network itself, which the percolation step and
    a kinetic Monte Carlo both need, and `barriers` maps every path of the structure -- not only
    the distinct ones -- to its barrier, which is what those steps read.
    """

    name: str
    rows: List[Dict[str, object]]
    barriers: Dict[int, float]
    hopset: object = None
    units: str = ""
    #: which structure of `hopset` these paths belong to
    structure: int = 0

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def values(self) -> np.ndarray:
        return np.array([float(r["barrier"]) for r in self.rows])

    def write_table(self, path: str) -> str:
        """Write the table as CSV, one row per path. Returns the path."""
        import csv
        folder = os.path.dirname(os.path.abspath(path))
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        cols = ("structure", "path", "from_site", "to_site", "length", "x", "y", "z",
                "stands_for", "barrier")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols))
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in cols})
        return path

    def percolation(self, levels: Sequence[float] = ()) -> Dict[str, float]:
        """The percolation thresholds of this structure under the predicted barriers.

        The lowest barrier at which the paths form a connected network in one, two and three
        dimensions, which is the number a conductivity estimate starts from
        (:func:`cmfqsar.hops.structure_percolation`).
        """
        from .hops import saddle_from_barrier, structure_percolation
        if self.hopset is None:
            raise ValueError("this table carries no path network, so there is nothing to "
                             "percolate; build it with HopBarriers.table")
        return structure_percolation(self.hopset, int(self.structure),
                                     saddle_from_barrier(self.hopset, self.barriers))


class HopBarriers:
    """A fitted hop model read as a barrier per migration path (plan items BT7 and MS7).

    The model is a :class:`cmfqsar.hops.HopModel`, fitted on the environments of migration paths
    in their own frames. This interface takes the structure instead of the environments: it
    enumerates the symmetry-distinct paths of the mobile ion, builds each environment the way the
    training set was built, and predicts.

    The settings have to be the ones the model was fitted with -- the envelope width `rho`, the
    reach `r_max`, the mobile element and the path length limits -- because an environment built
    differently is a different descriptor. They are arguments and not guesses, and a model file
    that records them should be read for them.
    """

    def __init__(self, model, fieldset=None, rho: Optional[float] = None,
                 r_max: Optional[float] = None, mobile: str = "Li",
                 d_min: Optional[float] = None, d_max: Optional[float] = None,
                 units: str = "eV", annotate_r_max: Optional[float] = None):
        from .hops import DEFAULT_RHO, HOP_MAX, HOP_MIN
        if getattr(model, "coef", None) is None:
            raise ValueError("this hop model is not fitted: there is nothing to predict with")
        self.model = model
        self.fieldset = fieldset if fieldset is not None else model.fieldset
        self.rho = float(DEFAULT_RHO if rho is None else rho)
        self.r_max = None if r_max is None else float(r_max)
        self.mobile = str(mobile)
        self.d_min = float(HOP_MIN if d_min is None else d_min)
        self.d_max = float(HOP_MAX if d_max is None else d_max)
        self.units = str(units)
        #: the reach used when deciding which paths are equivalent, which is a question about
        #: the structure and not about the descriptor; the library's own default (no cut) unless
        #: a caller has a reason to change it
        self.annotate_r_max = None if annotate_r_max is None else float(annotate_r_max)

    def __repr__(self) -> str:
        return (f"HopBarriers({self.mobile} paths of {self.d_min:g} to {self.d_max:g} A, "
                f"rho {self.rho:g}, {len(self.model.train)} training paths)")

    def hopset(self, atoms, name: str = ""):
        """The path network of one structure, with its paths reduced by symmetry.

        :meth:`cmfqsar.hops.HopSet.annotate` is what finds the bottleneck of every path and which
        paths are equivalent, and it is run here, because a path that stands for others has to
        carry their barrier in the network a simulation reads.
        """
        from .hops import HopSet
        label = name or str(dict(getattr(atoms, "info", {}) or {}).get("name", "")) or None
        hs = HopSet.build([atoms], names=[label] if label else None, mobile=self.mobile,
                          d_min=self.d_min, d_max=self.d_max)
        if not hs.hops:
            raise ValueError(
                f"{label or 'this structure'} has no {self.mobile} path between {self.d_min:g} "
                f"and {self.d_max:g} A: either it carries no {self.mobile}, or its sites are "
                "further apart than the limits allow")
        hs.annotate(hs.crystals(self.fieldset), rho=self.rho, r_max=self.annotate_r_max)
        return hs

    def table(self, atoms=None, name: str = "", progress=None, hopset=None,
              structure: int = 0, paths: Optional[Sequence[int]] = None) -> BarrierTable:
        """The predicted barrier of every symmetry-distinct path of one structure.

        `hopset` is for a caller that already has the path network: it is taken as it is,
        `structure` says which of its structures to report, and only the environments and the
        prediction are done here. Nothing in the network is changed. Without it the network is
        built from `atoms`.

        `paths` asks for named paths of the network instead of the distinct ones, which is what a
        caller who wants one particular jump needs. Each is then predicted from its own
        environment, and nothing is filled in from a representative.
        """
        from .hops import framed_environments
        if hopset is not None:
            hs = hopset
            k = int(structure)
            if not 0 <= k < len(hs.structures):
                raise IndexError(f"this network holds {len(hs.structures)} structures, so "
                                 f"structure {k} does not exist")
        else:
            if atoms is None:
                raise ValueError("pass a structure, or a path network as hopset=")
            hs = self.hopset(atoms, name)
            k = 0
        mine = hs.of_structure(k)
        if paths is not None:
            asked = [int(i) for i in paths]
            for i in asked:
                if not 0 <= i < len(hs.hops):
                    raise IndexError(f"this network holds {len(hs.hops)} paths, so path {i} "
                                     f"does not exist")
                if int(hs.hops[i].structure) != k:
                    raise ValueError(
                        f"path {i} belongs to {hs.names[int(hs.hops[i].structure)]} and not to "
                        f"{hs.names[k]}: a table is one structure's")
            return self._table(hs, k, asked, progress, filled=False)
        distinct = [i for i in hs.representatives() if int(hs.hops[i].structure) == k]
        if not distinct:
            raise ValueError(
                f"{hs.names[k]} has no path of its own in this network: it holds "
                f"{len(mine)} paths and none of them stands for the others")
        return self._table(hs, k, distinct, progress, filled=True)

    def _table(self, hs, k: int, asked: Sequence[int], progress, filled: bool) -> BarrierTable:
        """The barriers of `asked`, with the equivalent paths filled in when `filled`."""
        from .hops import framed_environments
        crystals = hs.crystals(self.fieldset)
        envs = framed_environments(hs, crystals, rows=list(asked), rho=self.rho,
                                   r_max=self.r_max, progress=progress)
        predicted = np.asarray(self.model.predict(envs), float).ravel()
        barriers = {int(i): float(v) for i, v in zip(asked, predicted)}
        stands_for = {int(i): 1 for i in asked}
        if filled:
            for i in hs.of_structure(k):
                rep = int(hs.hops[i].representative)
                if rep in barriers and rep != int(i):
                    barriers[int(i)] = barriers[rep]
                    stands_for[rep] = stands_for.get(rep, 1) + 1
        rows = []
        for i, value in zip(asked, predicted):
            h = hs.hops[int(i)]
            mid = np.asarray(h.midpoint, float).ravel()
            rows.append({"structure": hs.names[k], "path": int(i),
                         "from_site": int(h.a), "to_site": int(h.b),
                         "length": float(h.length),
                         "x": f"{mid[0]:.4f}", "y": f"{mid[1]:.4f}", "z": f"{mid[2]:.4f}",
                         "stands_for": int(stands_for.get(int(i), 1)),
                         "barrier": float(value)})
        return BarrierTable(name=hs.names[k], rows=rows, barriers=barriers, hopset=hs,
                            units=self.units, structure=k)

    def percolation(self, atoms=None, name: str = "", hopset=None,
                    structure: int = 0) -> Dict[str, float]:
        """The percolation thresholds of one structure under the predicted barriers."""
        return self.table(atoms, name, hopset=hopset, structure=structure).percolation()


# --------------------------------------------------------------------------- #
# A field on a grid
# --------------------------------------------------------------------------- #
def write_cofield_cube(model, path: str, field: int = 0, spacing: float = 0.4,
                       margin: float = 3.0, structure=None, name: str = "") -> str:
    """Write the co-field of a fitted model as a Gaussian cube file. Returns the path.

    For a model on the overlap kernel the cube covers the training set's own frame, padded by
    `margin`; for a model on one lattice (``pbc_overlap=True``) it covers one cell of
    `structure`, with the lattice vectors as the cube's axes, so the file is the periodic field
    itself. `spacing` is the grid step in angstrom.

    The rotation-invariant kernels have no co-field at all -- their channels know no place -- and
    are refused rather than drawn, as the manual says of them.
    """
    from .cofields import cofield_volume, export_cube, make_grid
    model._require_linear("a co-field")
    if not (model.uses_fields or bool(getattr(model, "pbc_overlap", False))):
        raise ValueError(
            f"the '{model.kernel_source}' kernel has no co-field: its channels are invariant to "
            "where the structure sits, so there is no field in space to write. The map of plan "
            "item MS2 (cmfqsar.sitemap) is what such a model offers instead.")
    if bool(getattr(model, "pbc_overlap", False)):
        if structure is None:
            raise ValueError(
                "a periodic co-field is drawn inside one cell, so this needs the structure whose "
                "cell to use: pass structure=an ASE Atoms object or a CrystalFields record on the "
                "model's own lattice")
        return _periodic_cube(model, path, field, spacing, structure, name)
    grid = make_grid(np.asarray(model.train.coords, float), spacing=spacing, margin=margin)
    volume = cofield_volume(model, field, grid)
    export_cube(path, grid, volume, model.train.molecules[0])
    return path


def _periodic_cube(model, path: str, field: int, spacing: float, structure, name: str) -> str:
    """One cell of a periodic co-field, with the lattice vectors in the cube's header."""
    from .cofields import cofield_at_points
    from .periodic import crystal_fields
    rec = structure
    if hasattr(structure, "get_cell"):
        rec = crystal_fields(structure, model.fieldset, name or "cell")
    cell = np.asarray(rec.cell, float).reshape(3, 3)
    lengths = np.linalg.norm(cell, axis=1)
    n = np.maximum(np.ceil(lengths / float(spacing)).astype(int), 1)
    axes = cell / n[:, None]                      # one cube axis per lattice vector, scaled
    fractions = [np.arange(k) / float(k) for k in n]
    fa, fb, fc = np.meshgrid(*fractions, indexing="ij")
    points = (fa.ravel()[:, None] * cell[0] + fb.ravel()[:, None] * cell[1]
              + fc.ravel()[:, None] * cell[2])
    values = np.asarray(cofield_at_points(model, int(field), points), float)
    _write_cube(path, np.asarray(rec.coords, float), list(rec.symbols),
                values.reshape(tuple(int(k) for k in n)), axes,
                origin=np.zeros(3),
                comment=f"CMF co-field of field {int(field)} in one cell of "
                        f"{name or rec.name or 'the structure'}")
    return path


#: the atomic numbers a cube file needs, by element symbol
def _atomic_numbers(symbols: Sequence[str]) -> List[int]:
    from ase.data import atomic_numbers
    return [int(atomic_numbers.get(str(s), 0)) for s in symbols]


def _write_cube(path: str, coords: np.ndarray, symbols: Sequence[str], values: np.ndarray,
                axes: np.ndarray, origin: np.ndarray, comment: str = "") -> str:
    """A Gaussian cube file with general (not necessarily orthogonal) axes, in bohr."""
    bohr = 1.0 / 0.529177210903
    numbers = _atomic_numbers(symbols)
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write((comment or "CMF co-field") + "\n")
        fh.write("the value at every grid point, in activity units\n")
        o = np.asarray(origin, float) * bohr
        fh.write(f"{len(numbers):5d} {o[0]:12.6f} {o[1]:12.6f} {o[2]:12.6f}\n")
        for k in range(3):
            v = np.asarray(axes[k], float) * bohr
            fh.write(f"{values.shape[k]:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
        for z, p in zip(numbers, np.asarray(coords, float)):
            q = p * bohr
            fh.write(f"{z:5d} {float(z):12.6f} {q[0]:12.6f} {q[1]:12.6f} {q[2]:12.6f}\n")
        flat = values.reshape(values.shape[0], values.shape[1], values.shape[2])
        for i in range(flat.shape[0]):
            for j in range(flat.shape[1]):
                column = flat[i, j]
                for start in range(0, len(column), 6):
                    fh.write("".join(f"{v:13.5e}" for v in column[start:start + 6]) + "\n")
    return path
