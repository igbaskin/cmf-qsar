"""
Solvent exposure as a weight on the fields: the claim of the surface methods, without a surface.

A CMF kernel is an overlap integral over all of space, so it weights an atom buried in the middle
of a molecule exactly as it weights one on the outside -- although a binding partner can never
touch the first.  That observation is what the surface methods are built on (CoMSA and its
relatives map properties onto the molecular surface and compare them there), and it arrived in
this program from another direction: the conceptual-DFT study found density differences dominated
by the buried core, which had to be windowed out before the maps meant anything
(:mod:`cmfqsar.cdft`).

Taking a surface literally would cost the thing the program is built around.  Every molecule has
its *own* surface, so there is no shared integration domain -- worse than the grid kernel, which at
least had one -- and the usual repair, projecting both surfaces onto a learned map, cannot be
inverted back into a field in space, so the co-field, the atom contributions and the Shapley
decomposition are all lost with it.

The same physical claim is available at the cost of a scalar per atom.  Multiply each atomic
weight by how exposed that atom is,

    w'_f,l  =  w_f,l * e_l,        e_l = SASA_l / 4 pi (R_l + R_probe)^2   in [0, 1],

and a buried atom contributes little to any field while an exposed one contributes fully.  The
field is still a sum of atom-centred Gaussians, so the kernel is still the closed-form overlap and
everything that reads it keeps working; the co-field of an exposure-weighted field is the co-field
of a field that happens to vanish inside the molecule.  There is a shorter way to say this: a
window that is itself a sum of atom-centred Gaussians leaves the triple-product integral analytic,
and in practice such a window *is* a per-atom weight.

What this does and does not touch: the steric field becomes the bulk a partner can feel rather than
the bulk that exists, which is the intent; indicator and per-atom property fields are scaled the
same way; a probe field's expansion coefficients are scaled per atom too; per-molecule descriptors
are left alone, since they are not fields.  Nothing here is a default -- it is an option to be
measured against the plain fields first, which is what ``benchmarks/run_exposure.py`` does.
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
from rdkit import Chem

#: the water probe of the usual accessible-surface definition, angstrom
PROBE_RADIUS = 1.4


def atom_exposure(mol: Chem.Mol, probe_radius: float = PROBE_RADIUS, floor: float = 0.0,
                  power: float = 1.0, united: bool = True) -> np.ndarray:
    """Relative solvent accessibility of every atom, in [0, 1].

    The accessible area divided by the area the same atoms would have alone, so the number is
    dimensionless and comparable between a carbon and an iodine.  `floor` keeps a wholly buried
    atom from vanishing (a molecule most of whose atoms are buried would otherwise have almost no
    field at all), and `power` sharpens or softens the contrast: 0.5 is gentler, 2 harsher.

    With `united`, a heavy atom and its hydrogens count as one group and share the group's
    exposure.  This is not a detail: measured atom by atom, the carbon of a methyl group is 93%
    occluded **by its own hydrogens**, so ethanol comes out two thirds buried and the weighting
    would be shielding rather than burial.  A partner meeting a methyl group meets the group.
    """
    from rdkit.Chem import rdFreeSASA

    m = Chem.Mol(mol)
    radii = rdFreeSASA.classifyAtoms(m)
    rdFreeSASA.CalcSASA(m, radii)
    sasa = np.array([float(a.GetProp("SASA")) if a.HasProp("SASA") else 0.0
                     for a in m.GetAtoms()], dtype=float)
    r = np.asarray(radii, dtype=float)
    whole = 4.0 * np.pi * np.maximum(r + float(probe_radius), 1e-9) ** 2
    if united:
        # the group's own area over the group's own isolated area, shared by its members
        area, whole_group = sasa.copy(), whole.copy()
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 1:
                continue
            group = [a.GetIdx()] + [n.GetIdx() for n in a.GetNeighbors()
                                    if n.GetAtomicNum() == 1]
            if len(group) > 1:
                total, whole_total = sasa[group].sum(), whole[group].sum()
                for i in group:
                    area[i], whole_group[i] = total, whole_total
        sasa, whole = area, whole_group
    e = np.clip(sasa / whole, 0.0, 1.0)
    if power != 1.0:
        e = e ** float(power)
    floor = float(np.clip(floor, 0.0, 1.0))
    return floor + (1.0 - floor) * e


def weight_by_exposure(fields, exposure: Sequence[float]):
    """A copy of one molecule's fields with every atomic weight scaled by its atom's exposure.

    Returns a new :class:`cmfqsar.fields.MoleculeFields`; the original is untouched, so the plain
    and the weighted representation of a set can be held side by side and compared on the same
    molecules.
    """
    e = np.asarray(exposure, dtype=float)
    if e.shape[0] != fields.weights.shape[0]:
        raise ValueError(f"{len(e)} exposures for {fields.weights.shape[0]} atoms")
    out = copy.copy(fields)
    out.weights = fields.weights * e[:, None]
    probe = getattr(fields, "probe_coeffs", None)
    if probe:
        out.probe_coeffs = {k: v * e[:, None] for k, v in probe.items()}
    return out


def exposed_fields(mol: Chem.Mol, fieldset, name: str = "", probe_radius: float = PROBE_RADIUS,
                   floor: float = 0.0, power: float = 1.0, united: bool = True, **kwargs):
    """``build_fields`` followed by the exposure weighting, for one molecule."""
    from .fields import build_fields

    base = build_fields(mol, fieldset, name, **kwargs)
    return weight_by_exposure(base, atom_exposure(mol, probe_radius, floor, power, united))


def exposed_set(mols: Sequence[Chem.Mol], fieldset, names: Optional[Sequence[str]] = None,
                probe_radius: float = PROBE_RADIUS, floor: float = 0.0, power: float = 1.0,
                united: bool = True):
    """The same for a whole set, in the order given."""
    names = list(names) if names is not None else [f"mol{i}" for i in range(len(mols))]
    return [exposed_fields(m, fieldset, n, probe_radius, floor, power, united)
            for m, n in zip(mols, names)]


def buried_fraction(mol: Chem.Mol, cutoff: float = 0.05, heavy_only: bool = True,
                    probe_radius: float = PROBE_RADIUS, united: bool = True) -> float:
    """The fraction of atoms the weighting almost removes: how much of a molecule is inside.

    Reported beside a measurement because it says whether the weighting had anything to do, and
    the answer for a ligand is usually *nothing*: measured by group, a drug-sized molecule has a
    median exposure near 0.4 and no buried group at all, because thirty heavy atoms are all
    surface.  There is no interior to discount until a molecule is much larger than that.
    """
    e = atom_exposure(mol, probe_radius, united=united)
    keep = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()]) if heavy_only \
        else np.ones(mol.GetNumAtoms(), dtype=bool)
    if not keep.any():
        return float("nan")
    return float(np.mean(e[keep] < cutoff))
