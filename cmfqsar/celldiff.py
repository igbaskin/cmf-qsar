"""
The difference between two cells of one ladder as a field in the cell (plan item BT6).

An intercalation voltage is not a property of a structure but of a **pair**: the energy of the
lithiated cell minus the energy of the delithiated one, per electron.  The fields of this program
are linear in the atoms, so the same difference can be taken of the fields themselves,

    dX_f(r) = X_f(Li_y Host)(r) - X_f(Li_x Host)(r),

and the result is an ordinary field in the cell: the atoms of the lithiated cell with their
weights and the atoms of the delithiated one with their weights negated.  Where the two cells
agree -- most of the host -- the two contributions cancel, and what is left is the inserted ions
and the relaxation the insertion caused.  That is the object the voltage is a property of, and it
is what :func:`difference_fields` builds.

Because a field difference is a field, every kernel in the program takes it unchanged: it is
bilinear in the weights, so nothing has to be re-derived for signed ones.  Two kernels are
available and the set decides which:

* **the periodic autocorrelation** (:mod:`cmfqsar.periodic`), which needs no common frame between
  *pairs* and so works across host frameworks.  This is what a collection of different hosts
  allows, and what ``benchmarks/run_battery.py --stage difference`` measures.
* **the periodic overlap kernel** of plan item M4, which compares fields point by point and so
  needs every pair on one parent lattice.  On a ladder of one host it is available and gives a
  co-field in the cell; across hosts it is not.

**What the two cells of a pair need, and why most pairs do not have it.** A difference is only
defined when the two cells stand on one lattice and hold the same host atoms.  Two cells relaxed
separately do not: their lattice constants differ by the volume change of the insertion, which is
the quantity a battery paper reports.  :func:`difference_fields` therefore rebuilds the
delithiated cell in the lattice of the lithiated one, keeping its fractional coordinates, and
refuses a pair whose cell lengths differ by more than ``tol`` or whose host atom counts do not
match.  The refusal is not a failure of the method: it says the two cells are too far apart for a
difference to mean anything, and :func:`usable_pairs` reports how much of a set survives it.

The scale of a difference field is not a free parameter but it does not matter either: the field
kernels of this program are cosine-normalised, so a difference field of two inserted ions and one
of one inserted ion are compared by their shape and not by their size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import FieldSet
from .periodic import CrystalFields, crystal_fields

#: how far the cell lengths of the two cells of a pair may differ, as a fraction
LATTICE_TOL = 0.05


def lattice_mismatch(a, b) -> float:
    """How far two cells are from one lattice: the largest relative difference of their lengths.

    The lengths are sorted before they are compared, so the same lattice written with its vectors
    in another order counts as the same lattice.  The angles are not compared, because a pair that
    passes the length test and fails an angle test has not been seen in practice and the number
    reported here should mean one thing.
    """
    la = np.sort(np.linalg.norm(np.asarray(a.get_cell()[:], float), axis=1))
    lb = np.sort(np.linalg.norm(np.asarray(b.get_cell()[:], float), axis=1))
    if la.shape != lb.shape or not np.all(la > 0):
        return float("inf")
    return float(np.max(np.abs(lb / la - 1.0)))


def host_counts(atoms, ion: str) -> dict:
    """The atoms of the cell by element, the working ion left out."""
    from collections import Counter
    return dict(Counter(s for s in atoms.get_chemical_symbols() if s != ion))


def on_lattice(atoms, cell):
    """The same structure in another lattice: its fractional coordinates, the given cell.

    This is how the two cells of a pair are brought together.  It moves every atom by the
    difference of the two lattices, which for a pair that passes :func:`lattice_mismatch` is a
    fraction of an angstrom, and it is the only honest way to subtract two fields: a field
    difference taken between two different cells would measure the lattices and not the
    insertion.
    """
    out = atoms.copy()
    scaled = atoms.get_scaled_positions()
    out.set_cell(np.asarray(cell, float))
    out.set_scaled_positions(scaled)
    return out


def can_difference(charged, discharged, ion: str = "Li",
                   tol: float = LATTICE_TOL) -> Tuple[bool, str]:
    """Can these two cells carry a difference field, and if not, which test failed?"""
    gap = lattice_mismatch(charged, discharged)
    if not np.isfinite(gap):
        return False, "the two cells have no comparable lattice"
    if gap > tol:
        return False, f"the cell lengths differ by {gap:.1%}, more than {tol:.0%}"
    if host_counts(charged, ion) != host_counts(discharged, ion):
        return False, ("the two cells hold different host atoms "
                       f"({host_counts(charged, ion)} against {host_counts(discharged, ion)})")
    n_c = sum(1 for s in charged.get_chemical_symbols() if s == ion)
    n_d = sum(1 for s in discharged.get_chemical_symbols() if s == ion)
    if n_d <= n_c:
        return False, f"the discharged cell holds no more of the working ion ({n_d} against {n_c})"
    return True, ""


def difference_fields(charged, discharged, fieldset: FieldSet, name: str = "",
                      ion: str = "Li", tol: float = LATTICE_TOL) -> CrystalFields:
    """The field difference of one pair, as a :class:`cmfqsar.periodic.CrystalFields`.

    The atoms of the discharged cell keep their weights and the atoms of the charged cell, moved
    into the discharged cell's lattice, carry theirs negated.  Radii, symbols and the cell come
    with them, so every kernel and every feature matrix of the program takes the result as it
    takes any other structure.  The charges of the two cells are computed in their own cells
    first, which matters: the formal charge of the metal is not the same before and after the
    insertion (plan item BT1), and that change is part of what the difference field carries.
    """
    ok, why = can_difference(charged, discharged, ion, tol)
    if not ok:
        raise ValueError(f"these two cells cannot carry a difference field: {why}")
    moved = on_lattice(charged, discharged.get_cell()[:])
    a = crystal_fields(discharged, fieldset, name or "discharged")
    b = crystal_fields(moved, fieldset, "charged")
    return CrystalFields(
        name=name or f"{discharged.get_chemical_formula()} - {charged.get_chemical_formula()}",
        coords=np.vstack([a.coords, b.coords]),
        weights=np.vstack([a.weights, -b.weights]),
        radii=np.concatenate([a.radii, b.radii]),
        symbols=list(a.symbols) + list(b.symbols),
        cell=np.asarray(discharged.get_cell()[:], float),
        pbc=a.pbc, atom_types=list(a.atom_types) + list(b.atom_types),
        charges=(None if a.charges is None or b.charges is None else
                 np.concatenate([a.charges, -b.charges])),
        charge_source=a.charge_source)


def inserted_positions(charged, discharged, ion: str = "Li",
                       tol: float = 0.5) -> np.ndarray:
    """Where the working ions went in: the sites of the discharged cell with no partner.

    Every ion of the charged cell is matched to the nearest ion of the discharged cell within
    ``tol`` angstrom, the minimum image taken; the discharged sites left over are the inserted
    ones.  It is a reading of the pair, not part of the kernel, and it is what a figure of the
    difference field is centred on.
    """
    from ase.geometry import get_distances
    cell = np.asarray(discharged.get_cell()[:], float)
    sites_d = np.array([p for p, s in zip(discharged.get_positions(),
                                          discharged.get_chemical_symbols()) if s == ion])
    sites_c = np.array([p for p, s in zip(on_lattice(charged, cell).get_positions(),
                                          charged.get_chemical_symbols()) if s == ion])
    if not len(sites_d):
        return np.zeros((0, 3))
    if not len(sites_c):
        return sites_d
    _, dist = get_distances(sites_c, sites_d, cell=cell, pbc=True)
    taken = set()
    for row in range(dist.shape[0]):
        order = np.argsort(dist[row])
        for col in order:
            if col not in taken and dist[row, col] <= tol:
                taken.add(int(col))
                break
    return np.array([p for k, p in enumerate(sites_d) if k not in taken])


@dataclass
class PairDifference:
    """One pair that can carry a difference field, with what the construction found."""
    name: str
    fields: CrystalFields
    target: float
    group: str
    lattice_mismatch: float
    n_inserted: int
    inserted: np.ndarray = None          # (n, 3) the sites the ions went into
    discharged: object = None            # the lithiated cell itself, for readings of the pair
    info: dict = None


def usable_pairs(pairs: Sequence, fieldset: FieldSet, tol: float = LATTICE_TOL,
                 ion: str = "Li", target=lambda p: p.voltage,
                 group=lambda p: p.framework) -> Tuple[List[PairDifference], List[dict]]:
    """Every pair of a set that can carry a difference field, and why the others cannot.

    ``pairs`` are the insertion pairs of :mod:`cmfqsar.battery`, or anything with ``charged`` and
    ``discharged`` attributes; ``target`` and ``group`` say what is being predicted and what holds
    the folds together.  The second list is the refusals, one record each, so a benchmark can
    report how much of its set the frame costs instead of quietly dropping it.
    """
    kept, refused = [], []
    for p in pairs:
        ok, why = can_difference(p.charged, p.discharged, ion, tol)
        if not ok:
            refused.append({"name": getattr(p, "name", ""), "group": group(p), "reason": why})
            continue
        fields = difference_fields(p.charged, p.discharged, fieldset,
                                   name=getattr(p, "name", ""), ion=ion, tol=tol)
        inserted = inserted_positions(p.charged, p.discharged, ion)
        kept.append(PairDifference(
            name=getattr(p, "name", "") or fields.name, fields=fields, target=float(target(p)),
            group=str(group(p)), lattice_mismatch=lattice_mismatch(p.charged, p.discharged),
            n_inserted=len(inserted), inserted=inserted, discharged=p.discharged,
            info={"x_charged": getattr(p, "x_charged", None),
                  "x_discharged": getattr(p, "x_discharged", None)}))
    return kept, refused


def shell_of_the_inserted_ion(pair: PairDifference, anion: str = "O",
                              cutoff: float = 3.0) -> dict:
    """What surrounds the sites the ions went into: how many anions, and how far.

    This is the reading the item asks for, as far as a set of different hosts allows one: the
    coordination of the site the working ion takes, measured on the discharged cell itself rather
    than read off a model.  A co-field in the cell needs every pair on one lattice, which a
    collection of different hosts does not give.
    """
    from ase.geometry import get_distances
    if pair.inserted is None or not len(pair.inserted) or pair.discharged is None:
        return {"n_sites": 0, "coordination": float("nan"), "distance": float("nan")}
    cell = np.asarray(pair.discharged.get_cell()[:], float)
    anions = np.array([p for p, s in zip(pair.discharged.get_positions(),
                                         pair.discharged.get_chemical_symbols()) if s == anion])
    if not len(anions):
        return {"n_sites": int(len(pair.inserted)), "coordination": float("nan"),
                "distance": float("nan")}
    _, dist = get_distances(pair.inserted, anions, cell=cell, pbc=True)
    counts, near = [], []
    for row in range(dist.shape[0]):
        d = dist[row]
        inside = d[(d > 1e-6) & (d <= cutoff)]
        counts.append(float(len(inside)))
        if len(inside):
            near.append(float(np.mean(inside)))
    return {"n_sites": int(len(pair.inserted)),
            "coordination": float(np.mean(counts)) if counts else float("nan"),
            "distance": float(np.mean(near)) if near else float("nan")}
