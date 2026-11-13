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

**A parameter set for a cell model.** :class:`CellParameter` and :class:`CellParameterSet` are the
interface a continuum battery model is fed through (plan item MS9). A continuum cell model is a set
of partial differential equations with time and transport in it, which this program is not; what
this program can give it is a value for one of its parameters, with a standard deviation in the
unit the cell model reads. The functions in that section do the conversions: a migration barrier to
a solid diffusion coefficient through an Arrhenius law, carrying the deviation through the
exponential (:func:`diffusivity_from_barrier`); a predicted conductivity to a factor on the
published one (:func:`conductivity_scale_from_log10`), and to another temperature with a
predicted activation energy (:func:`conductivity_scale_at_temperature`); a predicted average
voltage to a shift of
the published open-circuit voltage (:func:`ocp_shift`); an insertion ladder to a
voltage-against-stoichiometry curve (:func:`ladder_to_ocp_curve`); and an effective conductivity of
a porous electrode to a Bruggeman exponent (:func:`bruggeman_from_conductivity_ratio`). The sampler
that runs the cell model once per drawn parameter set, and the report that says which parameter's
deviation moves the result most, are in :mod:`cmfqsar.cellmodel`, because they need the cell model
itself installed and this module needs nothing.

The map of plan item MS2 needs no new interface: :func:`cmfqsar.sitemap.write_maps` already writes
one row per place. It is re-exported here so that one import gives every way out of a model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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


# --------------------------------------------------------------------------- #
# A parameter set for a cell model
# --------------------------------------------------------------------------- #
#: Boltzmann's constant in electronvolt per kelvin
BOLTZMANN_EV = 8.617333262e-5
#: how a drawn parameter is spread: around its value, around its logarithm, or as a set of
#: values that were drawn already
DISTRIBUTIONS = ("normal", "lognormal", "sample")
#: how the cell model's own value is changed: replaced, multiplied, or added to
KINDS = ("value", "scale", "shift")


@dataclass
class CellParameter:
    """One parameter of a cell model, as this program can give it: a value and a deviation.

    `name` is the cell model's own name for the parameter, with the unit it reads, so that the
    record can be handed over without a translation table. `kind` says what the number does to
    the published value: ``"value"`` replaces it, ``"scale"`` multiplies it, and ``"shift"`` is
    added to it. A function of concentration or of state of charge cannot be replaced by one
    number, so a prediction for such a parameter is a scale or a shift.

    `sd` is the standard deviation in the same unit, and `distribution` says how to read it:
    ``"normal"`` spreads around `value`, ``"lognormal"`` spreads the natural logarithm (so `sd`
    is then the deviation of ``ln value``, which is what an Arrhenius law produces), and
    ``"sample"`` means the values were drawn already and are in `sample`, which keeps the shape
    of the distribution instead of assuming one.

    The remaining fields are what the receiving side has to know to use the number responsibly:
    `source` names the model and the data it was fitted on, `calibration` is the root mean square
    of the standardised residual of that model on held-out activities (plan item MS1; one means
    the deviation means what it says), `predicted` is false for a parameter that this program does
    not predict and that carries a stated spread instead, and `in_domain` is false when the
    structure the prediction was made for lies outside what the model was fitted on.
    """

    name: str
    value: float
    sd: float = 0.0
    kind: str = "value"
    distribution: str = "normal"
    sample: Optional[np.ndarray] = None
    published: Optional[float] = None
    units: str = ""
    source: str = ""
    calibration: Optional[float] = None
    predicted: bool = True
    in_domain: bool = True
    note: str = ""
    #: limits the drawn values are kept inside, for a parameter that cannot be negative or that
    #: the cell model refuses outside a range
    low: Optional[float] = None
    high: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"a parameter is a {' or a '.join(KINDS)}, not a {self.kind!r}")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"the spread is {' or '.join(DISTRIBUTIONS)}, not "
                             f"{self.distribution!r}")
        if self.distribution == "sample":
            if self.sample is None or len(np.asarray(self.sample, float)) < 2:
                raise ValueError(f"{self.name} says its values were drawn already, so at least "
                                 "two of them have to be given as sample=")
            self.sample = np.asarray(self.sample, float).ravel()
        if self.distribution == "lognormal" and self.value <= 0:
            raise ValueError(f"{self.name} spreads its logarithm, so its value has to be above "
                             f"zero, not {self.value!r}")
        if self.sd < 0:
            raise ValueError(f"{self.name} has a negative deviation ({self.sd!r})")

    @property
    def calibrated(self) -> bool:
        """Does the deviation mean what it says?  The rule of plan item MS1: the root mean square
        of the standardised residual between 0.8 and 1.25."""
        c = self.calibration
        return c is not None and np.isfinite(c) and 0.8 <= float(c) <= 1.25

    def draw(self, n: int, rng=None) -> np.ndarray:
        """`n` values of this parameter, drawn from its deviation."""
        rng = np.random.default_rng() if rng is None else rng
        n = int(n)
        if self.distribution == "sample":
            out = np.asarray(rng.choice(self.sample, size=n, replace=True), float)
        elif self.distribution == "lognormal":
            out = float(self.value) * np.exp(rng.normal(0.0, float(self.sd), n))
        else:
            out = rng.normal(float(self.value), float(self.sd), n)
        if self.low is not None:
            out = np.maximum(out, float(self.low))
        if self.high is not None:
            out = np.minimum(out, float(self.high))
        return out

    def interval(self, level: float = 95.0, draws: int = 20000, seed: int = 0) -> List[float]:
        """The central interval of this parameter at `level` per cent."""
        half = (100.0 - float(level)) / 2.0
        v = self.draw(draws, np.random.default_rng(seed))
        return [float(np.percentile(v, half)), float(np.percentile(v, 100.0 - half))]

    def apply(self, published, value: float):
        """The published entry of the cell model, changed by `value` the way `kind` says.

        A published entry that is a function of something (of the concentration, of the state of
        charge) is wrapped, so that a scale or a shift works on a function as it works on a
        number. A function cannot be replaced by a number, and that is refused.
        """
        value = float(value)
        if callable(published):
            if self.kind == "value":
                raise TypeError(
                    f"{self.name} is a function in the cell model, so one number cannot replace "
                    "it; predict a scale or a shift of it instead")
            if self.kind == "scale":
                return lambda *a, _f=published, _v=value: _v * _f(*a)
            return lambda *a, _f=published, _v=value: _f(*a) + _v
        if self.kind == "value":
            return value
        if self.kind == "scale":
            return value * float(published)
        return float(published) + value

    def as_dict(self) -> Dict[str, object]:
        """The record as plain data, for the file the receiving side reads."""
        out = {"name": self.name, "kind": self.kind, "value": float(self.value),
               "sd": float(self.sd), "distribution": self.distribution, "units": self.units,
               "published": None if self.published is None else float(self.published),
               "source": self.source, "predicted": bool(self.predicted),
               "in_domain": bool(self.in_domain),
               "calibration": None if self.calibration is None else float(self.calibration),
               "calibrated": bool(self.calibrated), "note": self.note,
               "interval_95": self.interval()}
        if self.distribution == "sample":
            out["sample_size"] = int(len(self.sample))
            out["sample"] = [float(v) for v in self.sample]
        if self.low is not None:
            out["low"] = float(self.low)
        if self.high is not None:
            out["high"] = float(self.high)
        return out

    @classmethod
    def from_dict(cls, record: Dict[str, object]) -> "CellParameter":
        """A record read back from the file :meth:`as_dict` wrote.

        The fields the file carries for the reader's benefit alone -- the interval, whether the
        deviation is calibrated, how many values the sample holds -- are ignored, because they
        follow from the rest.
        """
        keep = ("name", "value", "sd", "kind", "distribution", "sample", "published", "units",
                "source", "calibration", "predicted", "in_domain", "note", "low", "high")
        return cls(**{k: record[k] for k in keep if k in record})

    def describe(self) -> str:
        lo, hi = self.interval()
        what = {"value": "replaces", "scale": "multiplies", "shift": "is added to"}[self.kind]
        body = (f"{self.name}: {self.value:.4g} {self.units} ({what} the published value), "
                f"95 % interval {lo:.4g} to {hi:.4g}")
        if self.calibration is not None and np.isfinite(self.calibration):
            body += (f", deviation " +
                     ("calibrated" if self.calibrated else "not calibrated") +
                     f" (z rms {self.calibration:.2f})")
        if not self.predicted:
            body += "; not predicted here, a stated spread"
        if not self.in_domain:
            body += "; outside what the model was fitted on"
        return body


@dataclass
class CellParameterSet:
    """A parameter set for a cell model: the published set, and what this program changes in it.

    The set is written out as it stands, with one record per parameter, so that the receiving
    side can read the deviations without running this program. Nothing here knows about any
    particular cell model: the names are strings and the values are numbers.
    """

    base: str
    parameters: List[CellParameter] = field(default_factory=list)
    temperature: float = 298.15
    note: str = ""

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    def __getitem__(self, name: str) -> CellParameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"this set holds no parameter called {name!r}; it holds "
                       f"{', '.join(self.names) or 'none'}")

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def add(self, parameter: CellParameter) -> "CellParameterSet":
        if parameter.name in self.names:
            raise ValueError(f"{parameter.name} is in this set already")
        self.parameters.append(parameter)
        return self

    def as_dict(self) -> Dict[str, object]:
        return {"base": self.base, "temperature": float(self.temperature), "note": self.note,
                "parameters": [p.as_dict() for p in self.parameters]}

    def write(self, path: str) -> str:
        """Write the set as JSON, one record per parameter. Returns the path."""
        import json
        folder = os.path.dirname(os.path.abspath(path))
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=1)
        return path

    @classmethod
    def from_dict(cls, record: Dict[str, object]) -> "CellParameterSet":
        """A set read back from the file :meth:`write` wrote."""
        return cls(base=str(record["base"]),
                   parameters=[CellParameter.from_dict(p) for p in record["parameters"]],
                   temperature=float(record.get("temperature", 298.15)),
                   note=str(record.get("note", "")))

    @classmethod
    def read(cls, path: str) -> "CellParameterSet":
        """Read a set from the JSON file :meth:`write` wrote, parameter for parameter."""
        import json
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def write_table(self, path: str) -> str:
        """Write the set as CSV, one row per parameter. Returns the path."""
        import csv
        folder = os.path.dirname(os.path.abspath(path))
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        cols = ("name", "kind", "value", "sd", "distribution", "units", "published",
                "calibration", "calibrated", "predicted", "in_domain", "source", "note")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols))
            w.writeheader()
            for p in self.parameters:
                r = p.as_dict()
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
        return path

    def describe(self) -> str:
        head = (f"a parameter set for {self.base} at {self.temperature:g} K, "
                f"{len(self.parameters)} parameters")
        return "\n".join([head] + [f"  {p.describe()}" for p in self.parameters])


def arrhenius_diffusivity(barrier: float, hop_length: float, temperature: float = 298.15,
                          attempt_frequency: float = 1e13, dimension: int = 3) -> float:
    """A diffusion coefficient in m^2/s from a migration barrier in electronvolt.

    The random walk of one ion over sites a distance `hop_length` apart (in angstrom), each jump
    attempted `attempt_frequency` times a second and succeeding with the Boltzmann factor of the
    barrier, diffuses with ``D = a^2 nu exp(-E / kT) / (2 d)``, with `d` the number of directions
    the ion can move in. The attempt frequency is not predicted by this program: it is an
    argument, and 10^13 per second is the usual order for an ion in a solid.
    """
    a = float(hop_length) * 1e-10
    kt = BOLTZMANN_EV * float(temperature)
    d = max(int(dimension), 1)
    return a * a * float(attempt_frequency) * float(np.exp(-float(barrier) / kt)) / (2.0 * d)


def diffusivity_from_barrier(barrier, hop_length: float, name: str, temperature: float = 298.15,
                             attempt_frequency: float = 1e13, dimension: int = 3,
                             sd: Optional[float] = None, published: Optional[float] = None,
                             source: str = "", calibration: Optional[float] = None,
                             in_domain: bool = True, note: str = "",
                             high: Optional[float] = None) -> CellParameter:
    """A solid diffusion coefficient as a cell-model parameter, from a predicted barrier.

    `barrier` is either one barrier in electronvolt, and then `sd` is its standard deviation, or
    a set of barriers that were drawn already from a model's deviation, and then the diffusion
    coefficients of those draws are carried over as they are. The second form is the better one,
    because the exponential of a normal deviation is not symmetric and the drawn barriers already
    hold the shape the deviation has.

    The deviation of the logarithm of the diffusion coefficient is the deviation of the barrier
    divided by ``kT``, which at room temperature is 0.0257 eV: a barrier known to 0.1 eV gives a
    diffusion coefficient known to a factor of 50.

    `high` is a limit the drawn values are kept under. A deviation of several electronvolt
    reaches diffusion coefficients no solid has, and a cell model spends a long time on them, so
    a caller that draws from such a deviation should say where physics stops.
    """
    kt = BOLTZMANN_EV * float(temperature)
    values = np.asarray(barrier, float).ravel()
    if values.size > 1:
        d = np.array([arrhenius_diffusivity(b, hop_length, temperature, attempt_frequency,
                                            dimension) for b in values])
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) < 2:
            raise ValueError(f"{name}: the drawn barriers give no usable diffusion coefficient")
        return CellParameter(
            name=name, value=float(np.median(d)), sd=float(np.std(np.log(d))), kind="value",
            distribution="sample", sample=d, published=published, units="m2.s-1", source=source,
            calibration=calibration, in_domain=in_domain, note=note, low=0.0, high=high)
    one = float(values.ravel()[0])
    value = arrhenius_diffusivity(one, hop_length, temperature, attempt_frequency, dimension)
    return CellParameter(name=name, value=value,
                         sd=float(0.0 if sd is None else float(sd) / kt), kind="value",
                         distribution="lognormal", published=published, units="m2.s-1",
                         source=source, calibration=calibration, in_domain=in_domain, note=note,
                         low=0.0, high=high)


def conductivity_scale_from_log10(predicted: float, reference: float, name: str,
                                  sd: float = 0.0, published: Optional[float] = None,
                                  source: str = "", calibration: Optional[float] = None,
                                  in_domain: bool = True, note: str = "") -> CellParameter:
    """A factor on the published conductivity, from a prediction on a base-ten logarithmic scale.

    A cell model holds the conductivity of the electrolyte as a function of concentration and
    temperature, which one number cannot replace. What a model of one condition can say is how
    far above or below the published function its own value lies, and that is a factor on it.
    `predicted` and `reference` are both logarithms to base ten of a conductivity in the same
    unit, so only their difference matters. `sd` is the deviation of the prediction on that same
    logarithmic scale.
    """
    return CellParameter(name=name, value=float(10.0 ** (float(predicted) - float(reference))),
                         sd=float(np.log(10.0) * float(sd)), kind="scale",
                         distribution="lognormal", published=published, units="factor",
                         source=source, calibration=calibration, in_domain=in_domain, note=note,
                         low=0.0)


#: the gas constant in joule per mole and kelvin
GAS_CONSTANT = 8.314462618


def conductivity_scale_at_temperature(predicted: float, sd: float, activation: float,
                                      sd_activation: float, reference: float,
                                      temperature: float, name: str,
                                      prediction_temperature: float = 298.15,
                                      draws: int = 2000, seed: int = 0,
                                      published: Optional[float] = None, source: str = "",
                                      calibration: Optional[float] = None,
                                      in_domain: bool = True, note: str = "") -> CellParameter:
    """A factor on the published conductivity at another temperature, from a predicted
    conductivity and a predicted activation energy.

    The conductivity predicted at `prediction_temperature` is carried to `temperature` by an
    Arrhenius law, ``log10 k(T) = log10 k(T0) - E / (R ln 10) (1/T - 1/T0)``, with `activation`
    the activation energy in kJ/mol. `reference` is the base-ten logarithm of the cell model's
    own conductivity *at* `temperature`, in the same unit as `predicted`, so the factor relates
    the prediction to what the cell model would use there. A published function with no
    temperature dependence simply has the same reference at every temperature.

    The two predictions come from two models, so their errors are drawn independently: `draws`
    pairs, each giving one factor. The record carries the drawn factors, which keeps the shape
    the exponential gives them.
    """
    rng = np.random.default_rng(int(seed))
    logs = rng.normal(float(predicted), float(sd), int(draws))
    energies = rng.normal(float(activation), float(sd_activation), int(draws))
    shift = (1.0 / float(temperature) - 1.0 / float(prediction_temperature))
    logs_t = logs - energies * 1000.0 / (GAS_CONSTANT * np.log(10.0)) * shift
    factors = 10.0 ** (logs_t - float(reference))
    centre = 10.0 ** (float(predicted) - float(activation) * 1000.0
                      / (GAS_CONSTANT * np.log(10.0)) * shift - float(reference))
    return CellParameter(name=name, value=float(centre), sd=float(np.std(np.log(factors))),
                         kind="scale", distribution="sample", sample=factors,
                         published=published, units="factor", source=source,
                         calibration=calibration, in_domain=in_domain, note=note, low=0.0)


def ocp_shift(predicted: float, published: float, name: str, sd: float = 0.0,
              source: str = "", calibration: Optional[float] = None, in_domain: bool = True,
              note: str = "") -> CellParameter:
    """A shift in volt of the published open-circuit voltage, from a predicted average voltage.

    The ladder of plan item BT3 predicts the average voltage of an insertion electrode, not the
    shape of its curve against the state of charge. The part of that prediction a cell model can
    take is therefore a shift of its own curve: the predicted average minus the published one.
    :func:`ladder_to_ocp_curve` is the other conversion, for a model that gives the whole curve.
    """
    return CellParameter(name=name, value=float(predicted) - float(published), sd=float(sd),
                         kind="shift", distribution="normal", published=float(published),
                         units="V", source=source, calibration=calibration, in_domain=in_domain,
                         note=note)


def bruggeman_from_conductivity_ratio(ratio: float, porosity: float, name: str, sd: float = 0.0,
                                      published: Optional[float] = None, source: str = "",
                                      calibration: Optional[float] = None,
                                      in_domain: bool = True, note: str = "") -> CellParameter:
    """A Bruggeman exponent from the effective conductivity of a porous electrode.

    A cell model writes the conductivity of the electrolyte inside a porous electrode as the bulk
    value times the porosity raised to an exponent: ``sigma_eff = sigma eps^b``. A model that
    predicts the effective conductivity of a two-phase structure therefore gives the exponent as
    ``b = ln(sigma_eff / sigma) / ln(eps)``. `ratio` is ``sigma_eff / sigma``, between zero and
    one, and `sd` is its standard deviation, which is carried over by the derivative of the
    logarithm.
    """
    r, eps = float(ratio), float(porosity)
    if not 0.0 < r < 1.0:
        raise ValueError(f"{name}: the effective conductivity is a fraction of the bulk one, so "
                         f"the ratio has to lie between zero and one, not {r!r}")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"{name}: a porosity lies between zero and one, not {eps!r}")
    log_eps = float(np.log(eps))
    return CellParameter(name=name, value=float(np.log(r) / log_eps),
                         sd=float(abs(float(sd) / (r * log_eps))), kind="value",
                         distribution="normal", published=published, units="exponent",
                         source=source, calibration=calibration, in_domain=in_domain, note=note,
                         low=0.0)


def ladder_to_ocp_curve(stoichiometry: Sequence[float], voltages: Sequence[float]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """An insertion ladder as a voltage-against-stoichiometry curve a cell model can read.

    A ladder is a list of two-phase steps: between the stoichiometries ``x[i]`` and ``x[i+1]`` the
    voltage is flat at ``voltages[i]``, because the two phases coexist. The curve is therefore a
    staircase, and it is returned as such: two points per step, at the beginning and at the end of
    its plateau, in the order of increasing `stoichiometry`. A cell model reads it as an
    interpolant between those points.

    `stoichiometry` holds ``len(voltages) + 1`` values. A real curve is not a staircase, because
    each phase takes up some working ion in solid solution before the next one appears; the
    staircase is what a ladder of computed phases gives, and the difference is a limit of the
    conversion, not of the cell model.
    """
    x = np.asarray(stoichiometry, float).ravel()
    v = np.asarray(voltages, float).ravel()
    if len(x) != len(v) + 1:
        raise ValueError(f"a ladder of {len(v)} steps needs {len(v) + 1} stoichiometries, "
                         f"not {len(x)}")
    if len(v) < 1:
        raise ValueError("a ladder needs at least one step")
    order = np.argsort(x)
    if not np.all(order == np.arange(len(x))):
        x, v = x[order], v[np.argsort(x[:-1])]
    xs, vs = [], []
    for i in range(len(v)):
        xs += [float(x[i]), float(x[i + 1])]
        vs += [float(v[i]), float(v[i])]
    return np.array(xs), np.array(vs)


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
