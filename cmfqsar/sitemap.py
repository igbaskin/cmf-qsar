"""A spatially resolved prediction: one number per atom, site or region (plan item MS2).

A fitted model answers with one number per structure. A coarser model needs a map instead: a
barrier per site, a reactivity per surface site, a growth rate per patch. Two different maps
answer that question, and this module writes both in one form.

**The split of one prediction.** The kernels of this program are sums over pairs of atoms, and a
linear layer is linear, so a prediction splits into one number per atom, exactly and with nothing
assigned by a rule of thumb: the numbers add up to the prediction minus its fitted constant. That
is decomposition rule 3 of the development plan, which :meth:`cmfqsar.model.CMFModel.
atom_contributions` gives for molecules and :func:`cmfqsar.sites.site_contributions` for the sites
of a periodic cell. This module puts the two behind one entry, carries the constant as a place of
its own so that the column adds up to the prediction itself, and groups atoms into regions when a
region and not an atom is what the coarser model steps on.

**A prediction per site.** A site model (plan item M5) takes one site of a cell as its sample, so
asking it about every site of a cell is a prediction and not a split. Those numbers are in the
units of the property and are not shares of anything, so they do not add up to a total.

Which of the two a map holds is recorded in its ``quantity`` field, because the two must not be
confused. A contribution is a share of one structure's answer. A prediction is an answer of its
own.

What a map does not do is turn the model into a site-level model it was not fitted as. A split
says where the answer came from; it does not promise that the same number would be measured at
that place on its own.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: what a map's ``quantity`` field can say
CONTRIBUTION = "contribution"
PREDICTION = "prediction"

#: the label the fitted constant is written under
CONSTANT_LABEL = "the fitted constant"

#: the kernels whose prediction splits over places, and the function that splits each one
_SPLIT_BY_SOURCE = {
    "": "the overlap integral over atom pairs",
    "fields": "the overlap integral over atom pairs",
    "autocorrelation": "the distance channels, which are sums over atom pairs",
    "periodic": "the periodic distance channels, which are sums over atom pairs",
}

#: the kernels that have no split yet, and why each one needs its own
_NO_SPLIT = {
    "surface": "its channels carry a height as well as a distance, so a place is a column of the "
               "slab and not an atom",
    "reciprocal": "its channels are structure factors, which are sums over the cell as a whole "
                  "in reciprocal space",
    "threebody": "its channels are sums over triples of atoms, so a share belongs to a triangle "
                 "and not to one atom",
}


# --------------------------------------------------------------------------- #
# The map itself
# --------------------------------------------------------------------------- #
@dataclass
class SiteMap:
    """One number per place of one structure, with where each place is.

    ``values`` holds the numbers and ``labels`` names the places, both in the same order.
    ``coords`` is where each place is, in angstrom, or None when a place has no position.
    ``quantity`` says what the numbers are: :data:`CONTRIBUTION`, shares of one prediction that
    add up to it, or :data:`PREDICTION`, one answer per place in the units of the property.

    For a contribution map ``bias`` is the model's fitted constant, which belongs to no place,
    and ``prediction`` is the structure's own prediction, so that ``total`` equals it exactly.
    """

    name: str
    labels: List[str]
    values: np.ndarray
    coords: Optional[np.ndarray] = None
    quantity: str = CONTRIBUTION
    bias: float = 0.0
    prediction: float = float("nan")
    kind: str = "atom"                       # "atom", "site" or "region"
    source: str = ""                         # which kernel the numbers came from
    units: str = ""                          # the units of the activity, when they are known
    members: Optional[List[List[int]]] = None  # which atoms each place covers
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.values = np.asarray(self.values, float).ravel()
        if len(self.labels) != len(self.values):
            raise ValueError(f"{len(self.labels)} labels for {len(self.values)} values")
        if self.coords is not None:
            self.coords = np.asarray(self.coords, float).reshape(len(self.values), 3)
        if self.quantity not in (CONTRIBUTION, PREDICTION):
            raise ValueError(f"quantity is '{self.quantity}', not "
                             f"'{CONTRIBUTION}' or '{PREDICTION}'")

    def __len__(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        """The sum of the places and the fitted constant: the prediction, for a split."""
        return float(np.sum(self.values)) + float(self.bias)

    @property
    def residual(self) -> float:
        """How far the sum is from the prediction. Zero to round-off for a split."""
        return self.total - float(self.prediction)

    def check(self, tol: float = 1e-8) -> float:
        """Raise unless the places and the constant add up to the prediction.

        This is the first half of the gate of plan item MS2, and it is cheap enough to run every
        time a map is written. It returns the disagreement it accepted.
        """
        if self.quantity != CONTRIBUTION:
            raise ValueError(
                f"a map of {self.quantity}s has no total to check: its numbers are one answer "
                "per place and not shares of one answer")
        if not np.isfinite(self.prediction):
            raise ValueError("this map carries no prediction to compare its sum with")
        gap = abs(self.residual)
        if gap > tol:
            raise ValueError(
                f"the {len(self)} places of {self.name or 'this structure'} and the fitted "
                f"constant add up to {self.total:.8g}, and the prediction is "
                f"{float(self.prediction):.8g}: they differ by {gap:.3g}, which is more than the "
                f"{tol:g} allowed. The split is exact for a linear layer, so a difference this "
                "large means the map and the prediction were made from different structures or "
                "different field sets.")
        return float(gap)

    def top(self, n: int = 10) -> List[Tuple[str, float]]:
        """The `n` places with the largest numbers, by size and not by sign."""
        order = np.argsort(-np.abs(self.values))[:int(n)]
        return [(self.labels[int(k)], float(self.values[int(k)])) for k in order]

    def share(self) -> np.ndarray:
        """What part of the total size of the map each place carries, between 0 and 1.

        The sizes are used and not the signed numbers, because a split can hold values of both
        signs whose sum is small while each one is large.
        """
        total = float(np.sum(np.abs(self.values)))
        if total <= 0.0:
            return np.zeros(len(self))
        return np.abs(self.values) / total

    # ---------------------------------------------------------------- output
    def rows(self, include_constant: bool = True) -> List[Dict[str, object]]:
        """The map as one dictionary per row, which is what the writers below put in a file."""
        out: List[Dict[str, object]] = []
        for i in range(len(self)):
            xyz = ("", "", "")
            if self.coords is not None:
                xyz = tuple(f"{float(v):.4f}" for v in self.coords[i])
            atoms = ""
            if self.members is not None:
                atoms = " ".join(str(int(a)) for a in self.members[i])
            elif self.kind in ("atom", "site"):
                atoms = str(i)
            out.append({"structure": self.name, "place": i, "label": self.labels[i],
                        "atoms": atoms, "x": xyz[0], "y": xyz[1], "z": xyz[2],
                        "quantity": self.quantity, "value": float(self.values[i])})
        if include_constant and self.quantity == CONTRIBUTION and self.bias:
            out.append({"structure": self.name, "place": -1, "label": CONSTANT_LABEL,
                        "atoms": "", "x": "", "y": "", "z": "",
                        "quantity": self.quantity, "value": float(self.bias)})
        return out

    def to_dict(self) -> Dict[str, object]:
        """The whole map as plain data, for a JSON file or a stored result."""
        return {"name": self.name, "labels": list(self.labels),
                "values": [float(v) for v in self.values],
                "coords": (None if self.coords is None
                           else [[float(c) for c in p] for p in self.coords]),
                "quantity": self.quantity, "bias": float(self.bias),
                "prediction": float(self.prediction), "kind": self.kind,
                "source": self.source, "units": self.units,
                "members": (None if self.members is None
                            else [[int(a) for a in m] for m in self.members]),
                "notes": list(self.notes)}

    def write_table(self, path: str, include_constant: bool = True) -> str:
        """Write the map as a CSV table a coarse model reads. Returns the path."""
        return write_maps([self], path, include_constant=include_constant)

    def write_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_dict(), fh, indent=1)
            fh.write("\n")
        return path


#: the columns of the table the writers produce, in order
COLUMNS = ("structure", "place", "label", "atoms", "x", "y", "z", "quantity", "value")


def write_maps(maps: Sequence[SiteMap], path: str, include_constant: bool = True) -> str:
    """Write several maps as one CSV table, one row per place. Returns the path.

    The columns are ``structure, place, label, atoms, x, y, z, quantity, value``. ``place`` is
    the index of the atom, site or region the row holds, or -1 for the fitted constant; the atoms
    a region covers are listed in ``atoms``. A reader that steps on places keeps the rows whose
    ``quantity`` is ``contribution`` or ``prediction``; a reader that wants the structure's own
    answer back adds the ``value`` column of one structure, constant included.
    """
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        for m in maps:
            for row in m.rows(include_constant=include_constant):
                w.writerow(row)
    return path


# --------------------------------------------------------------------------- #
# What a model can and cannot split
# --------------------------------------------------------------------------- #
def splits_prediction(model) -> bool:
    """Can a prediction of this model be split over places?"""
    return (str(getattr(model, "kernel_source", "") or "") in _SPLIT_BY_SOURCE
            and not getattr(model, "pbc_overlap", False)
            and bool(getattr(model, "is_linear", True)))


def _require_split(model) -> str:
    """The name of the split this model's kernel has, or a refusal that says why it has none."""
    source = str(getattr(model, "kernel_source", "") or "")
    if getattr(model, "pbc_overlap", False):
        raise ValueError(
            "the periodic overlap kernel compares two cells point by point over their images, "
            "so a place of one cell overlaps places of every image of the other: that split is "
            "not implemented. Fit the periodic autocorrelation kernel "
            "(kernel_source='periodic') to map a cell.")
    if source in _NO_SPLIT:
        raise ValueError(
            f"the '{source}' kernel has no split over places, because {_NO_SPLIT[source]}. A "
            "split for it has to be derived for its own channels rather than borrowed from "
            "another kernel.")
    if source not in _SPLIT_BY_SOURCE:
        raise ValueError(f"unknown kernel source '{source}': there is no split for it")
    if not bool(getattr(model, "is_linear", True)):
        raise ValueError(
            f"a map needs the linear layer: with the '{model.kernel_layer}' layer the prediction "
            "is not a sum over the channels, so it does not split over places. The Shapley "
            "decomposition (cmfqsar.shapley) works for any kernel and answers a weaker question.")
    return _SPLIT_BY_SOURCE[source]


# --------------------------------------------------------------------------- #
# The split of one structure
# --------------------------------------------------------------------------- #
def _place_labels(structure) -> List[str]:
    symbols = list(getattr(structure, "symbols", []) or [])
    n = int(getattr(structure, "n_atoms", len(symbols)))
    return [f"{i} {symbols[i] if i < len(symbols) else '?'}" for i in range(n)]


def atom_map(model, mol, name: Optional[str] = None, units: str = "") -> SiteMap:
    """The split of one molecule's prediction over its atoms (plan item MS2).

    The numbers are in the units of the activity and, with the fitted constant, add up to the
    prediction. Defined for the overlap kernel and for the rotation-invariant kernel.
    """
    how = _require_split(model)
    if getattr(mol, "is_ensemble", False):
        raise ValueError(
            "this record is an ensemble of "
            f"{int(getattr(mol, 'n_conformers', 0))} members, whose atoms are those of every "
            "member one after another: a place of the map would be an atom of one member and "
            "not of the molecule. Map a single structure, or sum the members yourself.")
    from .fields import AtomTable
    v = np.asarray(model.atom_contributions(mol), float).ravel()
    pred = float(np.asarray(model.predict(AtomTable.from_molecules([mol])), float).ravel()[0])
    return SiteMap(name=name or getattr(mol, "name", "") or "structure",
                   labels=_place_labels(mol), values=v,
                   coords=np.asarray(mol.coords, float), quantity=CONTRIBUTION,
                   bias=float(getattr(model, "bias", 0.0) or 0.0), prediction=pred,
                   kind="atom", source=how, units=units)


def cell_map(model, crystal, spec=None, name: Optional[str] = None, units: str = "") -> SiteMap:
    """The split of one cell's prediction over its sites (plan items M5 and MS2).

    The cell's channels are the mean of its site profiles, so the prediction minus the fitted
    constant is the sum of one number per site, exactly.
    """
    how = _require_split(model)
    if getattr(crystal, "centre", None) is not None:
        raise ValueError(
            f"{crystal.name or 'this record'} is one site of a cell (its centre is site "
            f"{int(crystal.centre)}), not the cell: its channels are that site's own profile, so "
            "there is nothing to split over the places. Map the cell this site came from, or ask "
            "a site model for its prediction with site_predictions.")
    from .fields import AtomTable
    from .sites import site_contributions
    v = np.asarray(site_contributions(model, crystal, spec), float).ravel()
    pred = float(np.asarray(model.predict(AtomTable.from_molecules([crystal])),
                            float).ravel()[0])
    return SiteMap(name=name or getattr(crystal, "name", "") or "cell",
                   labels=_place_labels(crystal), values=v,
                   coords=np.asarray(crystal.coords, float), quantity=CONTRIBUTION,
                   bias=float(getattr(model, "bias", 0.0) or 0.0), prediction=pred,
                   kind="site", source=how, units=units)


def site_map(model, structure, spec=None, regions=None, name: Optional[str] = None,
             units: str = "", check: bool = True, tol: float = 1e-8) -> SiteMap:
    """A spatially resolved prediction of one structure: the entry point of plan item MS2.

    A periodic cell is split over its sites and a molecule over its atoms, so the caller does not
    have to know which. `regions` groups the places afterwards, as :func:`region_map` describes.
    With `check` the split is verified against the prediction before it is returned.
    """
    periodic = getattr(structure, "cell", None) is not None
    out = (cell_map(model, structure, spec=spec, name=name, units=units) if periodic
           else atom_map(model, structure, name=name, units=units))
    if check:
        out.check(tol=tol)
    if regions is not None:
        out = region_map(out, regions)
    return out


def site_maps(model, structures: Sequence, spec=None, regions=None, units: str = "",
              check: bool = True, progress=None) -> List[SiteMap]:
    """One map per structure, in the order given."""
    out = []
    for i, s in enumerate(structures):
        out.append(site_map(model, s, spec=spec, regions=regions, units=units, check=check))
        if progress is not None:
            progress(f"Maps {i + 1}/{len(structures)}", i / max(len(structures), 1))
    return out


# --------------------------------------------------------------------------- #
# Regions: a place larger than an atom
# --------------------------------------------------------------------------- #
def region_map(smap: SiteMap, regions: Dict[str, Sequence[int]],
               rest: Optional[str] = "the rest of the structure") -> SiteMap:
    """Group the places of a map into regions, summing their numbers.

    `regions` maps a name to the places that belong to it. A place may belong to one region at
    most, because a split is shared out and not copied. Places left over are collected into one
    region named by `rest`, so the map still adds up to the prediction; `rest=None` drops them
    instead, and then the total no longer matches and :meth:`SiteMap.check` will say so.

    The position of a region is the mean position of its places, which is where a coarse model
    puts the value it reads.
    """
    if smap.members is not None:
        raise ValueError("this map is already grouped into regions; group the atom map instead")
    n = len(smap)
    seen = np.zeros(n, dtype=bool)
    names: List[str] = []
    members: List[List[int]] = []
    for key, idx in regions.items():
        here = [int(i) for i in idx]
        for i in here:
            if not 0 <= i < n:
                raise IndexError(f"region '{key}' names place {i} of a map with {n} places")
            if seen[i]:
                raise ValueError(
                    f"place {i} ({smap.labels[i]}) is in region '{key}' and in another one: a "
                    "split is shared out over the places, so a place belongs to one region at "
                    "most")
            seen[i] = True
        names.append(str(key))
        members.append(here)
    left = [int(i) for i in np.nonzero(~seen)[0]]
    if left and rest is not None:
        names.append(str(rest))
        members.append(left)
    values = np.array([float(np.sum(smap.values[m])) if m else 0.0 for m in members])
    coords = None
    if smap.coords is not None:
        coords = np.array([smap.coords[m].mean(axis=0) if m else np.zeros(3) for m in members])
    notes = list(smap.notes)
    if left and rest is None:
        notes.append(f"{len(left)} of the {n} places were left out of the regions, so the map no "
                     f"longer adds up to the prediction")
    return SiteMap(name=smap.name, labels=names, values=values, coords=coords,
                   quantity=smap.quantity, bias=smap.bias, prediction=smap.prediction,
                   kind="region", source=smap.source, units=smap.units, members=members,
                   notes=notes)


def by_element(structure) -> Dict[str, List[int]]:
    """Regions that collect the places of each element, in alphabetical order."""
    out: Dict[str, List[int]] = {}
    for i, s in enumerate(getattr(structure, "symbols", []) or []):
        out.setdefault(str(s), []).append(int(i))
    return {k: out[k] for k in sorted(out)}


def shells_around(structure, centre: int, edges: Sequence[float] = (2.4, 3.5, 5.0)
                  ) -> Dict[str, List[int]]:
    """Regions that collect the places by distance from one place of the structure.

    The first region is the centre itself, then one region per edge, named by the distance range
    it covers in angstrom. Distances of a periodic cell are measured through the images, so a
    neighbour across the cell boundary falls in the shell it really belongs to.
    """
    centre = int(centre)
    coords = np.asarray(structure.coords, float)
    n = len(coords)
    if not 0 <= centre < n:
        raise IndexError(f"this structure has {n} places, so place {centre} does not exist")
    if getattr(structure, "cell", None) is not None:
        d = np.asarray(structure.to_atoms().get_distances(centre, range(n), mic=True), float)
    else:
        d = np.linalg.norm(coords - coords[centre], axis=1)
    sym = list(getattr(structure, "symbols", []) or [])
    label = f"{centre} {sym[centre] if centre < len(sym) else '?'}"
    out: Dict[str, List[int]] = {f"the centre, {label}": [centre]}
    low = 0.0
    for high in list(edges) + [float("inf")]:
        here = [int(i) for i in np.nonzero((d > low) & (d <= high))[0] if i != centre]
        name = (f"{low:g} to {high:g} A from the centre" if np.isfinite(high)
                else f"more than {low:g} A from the centre")
        out[name] = here
        low = float(high)
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------- #
# A prediction per site, for a model fitted on sites
# --------------------------------------------------------------------------- #
def is_site_model(model) -> bool:
    """Was this model fitted on sites of a cell (plan item M5) rather than on whole cells?"""
    mols = list(getattr(getattr(model, "train", None), "molecules", []) or [])
    return bool(mols) and all(getattr(m, "centre", None) is not None for m in mols)


def site_predictions(model, crystal, sites: Optional[Sequence[int]] = None,
                     name: Optional[str] = None, units: str = "") -> SiteMap:
    """What a site model predicts at every site of one cell (plan items M5 and MS2).

    This is a prediction per place and not a split, so the numbers are in the units of the
    property and do not add up to a total. `sites` restricts the map to some of the sites; the
    cell is featurised once either way, because the periodic split gives all of its sites at
    once.
    """
    from .atomtargets import AtomTargets
    from .sites import site_table
    if not is_site_model(model):
        raise ValueError(
            "this model was fitted on whole structures, so it has no answer for a single site. "
            "Use site_map, which splits its prediction over the places instead.")
    n = int(crystal.n_atoms)
    want = list(range(n)) if sites is None else [int(s) for s in sites]
    for s in want:
        if not 0 <= s < n:
            raise IndexError(f"{crystal.name or 'this cell'} has {n} sites, so site {s} does "
                             f"not exist")
    row = np.full(n, np.nan)
    row[want] = 0.0
    index = AtomTargets.from_arrays([row], name="site")
    table = site_table([crystal], index)
    v = np.asarray(model.predict(table), float).ravel()
    order = [int(index.atom[r]) for r in range(len(index))]
    by_site = dict(zip(order, v))
    sym = list(getattr(crystal, "symbols", []) or [])
    coords = np.asarray(crystal.coords, float)
    return SiteMap(name=name or getattr(crystal, "name", "") or "cell",
                   labels=[f"{s} {sym[s] if s < len(sym) else '?'}" for s in want],
                   values=np.array([by_site[s] for s in want]),
                   coords=coords[want], quantity=PREDICTION,
                   bias=0.0, prediction=float("nan"), kind="site",
                   source="a model fitted on sites of a cell", units=units)
