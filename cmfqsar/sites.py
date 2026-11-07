"""Site-resolved models of a periodic structure (plan item M5).

The periodic autocorrelation of :mod:`cmfqsar.periodic` is a sum over ordered pairs of atoms of
one cell, so it splits exactly into one contribution per **site**: the term of a pair belongs to
the atom that stands first in it.  That one observation gives two things the whole-cell descriptor
cannot.

**A sample that is a site.**  The row of the split that belongs to site *l* is the typed radial
profile of the cell as seen *from that site*: for every pair of field types (f, g) and every
distance channel d_k,

    s_l(f, g, d_k) = w_lf * sum_m w_mg G_k(r_lm) + w_lg * sum_m w_mf G_k(r_lm)   (f != g)

with the sum over the atoms of the cell and their periodic images inside the cutoff, the self
term where the spec keeps it, and the uniform background of the site removed.  The pair term is
local: only the neighbours of site *l* inside ``r_max`` enter it.  The uniform background is not, because it is
the mean field of the whole cell, so two cells of different composition give different
backgrounds at the same site; a profile that depends on nothing beyond the cutoff is taken with
``background=False``.  The profile needs no alignment,
because it is built from distances alone, and it is the periodic counterpart of the atom-centred
environment of :mod:`cmfqsar.localenv` (plan item LP2), with two differences: the images of the
cell are summed in, so nothing has to be cut out of the solid, and the profile is one-centred
(pairs from the site to its neighbours) instead of all-pairs inside an envelope.

A site record is an ordinary structure as far as the rest of the program is concerned:
:func:`site_fields` returns a :class:`~cmfqsar.periodic.CrystalFields` whose ``centre`` is the
site, and :func:`cmfqsar.periodic.periodic_autocorrelation` of such a record is that site's row.
So :class:`~cmfqsar.model.CMFModel` with ``kernel_source="periodic"`` fits a set of sites exactly
as it fits a set of cells, and the targets of that shape are the ones
:mod:`cmfqsar.atomtargets` already holds (plan item LP1), down to the folds, which hold out whole
**structures**: two sites of one cell, one in the training set and one in the test set, would let
the model see the cell it is being tested on.

**A local reading of a whole-cell model.**  For a cell-level model on the same kernel the split is
exact in the other direction: the cell descriptor is the mean of its site rows, the linear layer
is linear, so the prediction minus the bias is the sum of one number per site
(:func:`site_contributions`).  That is the local co-pattern the autocorrelation kernel could not
give: not only which pair of field types at which distance raises the property, but *which site*
of the cell carries it.  It is the same decomposition rule as the atom contributions of the
overlap kernel (rule 3 of the plan), and it is exact to machine precision rather than attributed.

**The reference under rule 1.**  The simplest thing that could predict a per-site property is a
count: which element stands at the site, and how many atoms of each element lie in each distance
shell around it.  :func:`shell_count_features` builds exactly that from the same cells, the same
sites and the same images, so that the only difference between the two numbers is the descriptor.

**What is not here.**  The height channels of a layer (plan item M1, ``z_max > 0``) have no site
split in this version: :func:`site_autocorrelation` refuses a spec that carries them, because the
descriptor of a site would then need the two height axes of the pair and the averaging over the
two faces, which is a second code path and not yet measured.  A site model of a slab therefore
uses the plain distance profile, which does not tell a neighbour above the surface from one below
it at the same distance.
"""
from __future__ import annotations

import weakref
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .atomtargets import AtomTargets
from .fields import AtomTable, FieldSet
from .invariant import autocorrelation_kernel, block_sizes, field_pairs
from .periodic import (CrystalFields, PeriodicSpec, _background_profile, _neighbour_pairs,
                       _normal_separations, _pbc_of, crystal_fields, periodic_measure)

#: the distance grid of a site profile: shorter than the one of a whole cell, because a site
#: feels its neighbours and not the far side of a supercell.  A starting point and no more; the
#: cutoff decides how far the model looks and has to be tuned for the property.
SITE_SPEC = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5)
#: shell edges of the count reference, in angstrom: the first coordination shell, the second,
#: and two wider ones
SHELLS: Tuple[float, ...] = (2.0, 3.0, 4.5, 6.0)


# --------------------------------------------------------------------------- #
# The site profile
# --------------------------------------------------------------------------- #
#: one cell's split, kept so that a table of the sites of one cell is not split once per site:
#: (weak reference to the cell, the spec, the rows).  One entry is enough, because a feature
#: matrix takes the sites of a cell together, and only the records of :func:`site_fields` read
#: it, so a cell a caller holds and changes is never answered from here.
_LAST: Optional[Tuple["weakref.ref", PeriodicSpec, np.ndarray]] = None


def site_autocorrelation(crystal: CrystalFields, spec: PeriodicSpec = SITE_SPEC,
                         pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """The periodic autocorrelation of one cell split over its sites: ``(n_atoms, width)``.

    Row *l* is the profile of site *l* described above, in the layout of
    :func:`cmfqsar.periodic.periodic_autocorrelation`, and the rows are **not** divided by the
    number of atoms, so that a site profile does not depend on how large a cell the site was
    written in.  The mean of the rows is the descriptor of the cell:
    ``site_autocorrelation(c, spec).mean(0) == periodic_autocorrelation(c, spec)`` for an ordered
    cell, and the sum over the rows divided by the expected atom count for a disordered one.
    """
    if spec.has_heights:
        raise ValueError(
            "the height channels of a layer (z_max > 0) have no site split in this version: "
            "take the site profile with z_max = 0, or the whole-cell descriptor with the "
            "heights")
    W = np.asarray(crystal.weights, float)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    global _LAST
    # only a site record shares its split with others (the records of one cell); a call on the
    # cell itself always recomputes, so nothing is remembered about an object a caller may change
    owner = getattr(crystal, "_parent", None)
    key = (spec, tuple(pairs))
    if owner is not None and _LAST is not None:
        ref, kept, rows = _LAST
        if ref() is owner and kept == key:
            return rows
    centres = spec.centres
    n_bins = len(centres)
    if n_atoms == 0:
        return np.zeros((0, len(pairs) * n_bins))
    inv = 2.0 * spec.sigma ** 2
    i, j, d = _neighbour_pairs(crystal, spec.r_max)
    if len(d):
        B = np.exp(-((d[:, None] - centres[None, :]) ** 2) / inv) * spec.window(d)[:, None]
    else:
        B = np.zeros((0, n_bins))
    self_row = np.exp(-(centres ** 2) / inv) if spec.include_self else np.zeros(n_bins)
    # V[l, g, k] = sum over the pairs that start at site l of w_mg G_k(r_lm): the neighbours of
    # the site, by field type and distance channel.  One scatter-add per channel; the pair list
    # is long and the channels are few.
    V = np.zeros((n_atoms, n_fields, n_bins))
    if len(d):
        Wj = W[j]
        for k in range(n_bins):
            Z = Wj * B[:, k:k + 1]
            for g in range(n_fields):
                V[:, g, k] = np.bincount(i, weights=Z[:, g], minlength=n_atoms)
    # the self term and, on a disordered site, the covariance that a product of means misses
    if crystal.second_moments is not None:
        E = np.asarray(crystal.second_moments, float)                    # (N, F, F)
        cov = E - W[:, :, None] * W[:, None, :]
    else:
        E = W[:, :, None] * W[:, None, :]
        cov = None
    own_rows = None
    if cov is not None and len(d):
        # a site and its own periodic image are occupied together or not at all (plan item M2)
        own = i == j
        if own.any():
            own_rows = np.zeros((n_atoms, n_bins))
            for k in range(n_bins):
                own_rows[:, k] = np.bincount(i[own], weights=B[own, k], minlength=n_atoms)
    BG = _site_background(crystal, spec, W) if spec.background else None
    fi = np.array([f for f, _ in pairs], int)
    gi = np.array([g for _, g in pairs], int)
    diag = fi == gi
    mult = np.where(diag, 1.0, 2.0)
    out = np.empty((n_atoms, len(pairs), n_bins))
    for q, (f, g) in enumerate(pairs):
        rows = W[:, f:f + 1] * V[:, g, :]
        if f != g:
            rows = rows + W[:, g:g + 1] * V[:, f, :]
        rows = rows + (mult[q] * E[:, f, g])[:, None] * self_row[None, :]
        if own_rows is not None:
            rows = rows + mult[q] * cov[:, f, g][:, None] * own_rows
        if BG is not None:
            J0, S = BG
            bg = W[:, f:f + 1] * S[:, g, :]
            if f != g:
                bg = bg + W[:, g:g + 1] * S[:, f, :]
            if cov is not None:
                bg = bg + (mult[q] * cov[:, f, g])[:, None] * J0[None, :]
            rows = rows - bg
        out[:, q, :] = rows
    flat = out.reshape(n_atoms, len(pairs) * n_bins)
    if owner is not None:
        _LAST = (weakref.ref(owner), key, flat)
    return flat


def _site_background(crystal: CrystalFields, spec: PeriodicSpec,
                     W: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(J0, S)`` of the uniform background of every site: ``S[l, g, k]`` is what a uniform
    spread of field *g* over the periodic directions puts at distance channel *k* from site *l*,
    divided by the cell measure, and ``J0[k]`` the same profile at zero separation, which the
    covariance of a disordered site needs.  None where there is no background to remove.
    """
    pbc = _pbc_of(crystal)
    measure = periodic_measure(crystal.cell, pbc)
    if measure is None or measure <= 0:
        return None
    n_periodic = int(sum(pbc))
    h, J = _background_profile(spec, n_periodic)
    n_atoms, n_fields = W.shape
    if n_periodic == 3:
        # a uniform density has no direction: every site sees the whole cell's field sum
        S = (W.sum(axis=0)[None, :, None] / measure) * J[0][None, None, :]
        return J[0] / measure, np.broadcast_to(S, (n_atoms, n_fields, len(J[0]))).copy()
    # fewer periodic directions: how much of the uniform spread a site sees depends on how far
    # the other atoms stand from it across the open directions
    H = _normal_separations(crystal.coords, crystal.cell, pbc)
    n_bins = J.shape[1]
    S = np.zeros((n_atoms, n_fields, n_bins))
    flat = H.ravel()
    for k in range(n_bins):
        Jk = np.interp(flat, h, J[:, k]).reshape(H.shape)
        S[:, :, k] = (Jk @ W) / measure
    return J[0] / measure, S


def site_width(spec: PeriodicSpec = SITE_SPEC, n_fields: int = 1) -> int:
    """How many channels one site profile carries."""
    return int(sum(block_sizes(spec, n_fields)))


# --------------------------------------------------------------------------- #
# A site as a sample
# --------------------------------------------------------------------------- #
def site_fields(crystal: CrystalFields, site: int, name: str = "") -> CrystalFields:
    """The cell as one sample centred on `site`: a record whose ``centre`` is that site.

    The arrays are shared with `crystal`, not copied; a site record is a view of the cell with a
    centre, and :func:`cmfqsar.periodic.periodic_autocorrelation` of it is that site's profile.
    """
    site = int(site)
    if not 0 <= site < crystal.n_atoms:
        raise IndexError(f"{crystal.name or 'this cell'} has {crystal.n_atoms} sites, so site "
                         f"{site} does not exist")
    out = CrystalFields(
        name=name or f"{crystal.name}#{site}", coords=crystal.coords, weights=crystal.weights,
        radii=crystal.radii, symbols=crystal.symbols, cell=crystal.cell, pbc=crystal.pbc,
        charges=crystal.charges, charge_source=crystal.charge_source,
        atom_types=list(crystal.atom_types), occupancy=crystal.occupancy,
        site_species=crystal.site_species, second_moments=crystal.second_moments,
        centre=site)
    # the cell every site of it is split from, so that one split serves all of its sites
    out._parent = crystal
    return out


def site_structures(crystals: Sequence[CrystalFields], targets: AtomTargets,
                    names: Optional[Sequence[str]] = None) -> List[CrystalFields]:
    """One site record per measured sample of `targets`, in the order of its rows."""
    out = []
    for r in range(len(targets)):
        k, s = int(targets.molecule[r]), int(targets.atom[r])
        c = crystals[k]
        label = names[r] if names is not None and r < len(names) else ""
        out.append(site_fields(c, s, label))
    return out


def site_table(crystals: Sequence[CrystalFields], targets: AtomTargets) -> AtomTable:
    """An :class:`~cmfqsar.fields.AtomTable` whose structures are the measured sites, which
    :class:`~cmfqsar.model.CMFModel` with ``kernel_source="periodic"`` takes like any other."""
    return AtomTable.from_molecules(site_structures(crystals, targets))


def site_names(crystals: Sequence[CrystalFields], targets: AtomTargets) -> List[str]:
    """``name#index element`` of every sample: what a table of predictions is read by."""
    out = []
    for r in range(len(targets)):
        k, s = int(targets.molecule[r]), int(targets.atom[r])
        c = crystals[k]
        sym = c.symbols[s] if s < len(c.symbols) else "?"
        out.append(f"{c.name or f'structure {k}'}#{s} {sym}")
    return out


def site_features(crystals: Sequence[CrystalFields], targets: AtomTargets,
                  spec: PeriodicSpec = SITE_SPEC, progress=None) -> np.ndarray:
    """The site profiles of every measured sample, one row each.

    The split of one cell gives every one of its sites at once, so each cell is featurised once
    however many of its sites were measured.
    """
    if not len(targets):
        return np.zeros((0, 0))
    n_fields = crystals[0].weights.shape[1]
    width = site_width(spec, n_fields)
    out = np.zeros((len(targets), width))
    wanted = sorted(set(int(m) for m in targets.molecule))
    pairs = field_pairs(n_fields)
    for n, k in enumerate(wanted):
        c = crystals[k]
        if c.weights.shape[1] != n_fields:
            raise ValueError(
                f"{c.name or f'structure {k}'} carries {c.weights.shape[1]} fields where the set "
                f"carries {n_fields}: build the fields of every structure from one field set.")
        rows = site_autocorrelation(c, spec, pairs)
        for r in targets.of_molecule(k):
            out[r] = rows[int(targets.atom[r])]
        if progress is not None and n % 25 == 0:
            progress(f"Site profiles {n + 1}/{len(wanted)}", n / max(len(wanted), 1))
    return out


def site_kernel(train: np.ndarray, test: Optional[np.ndarray] = None,
                n_bins: int = 1, blocks: Optional[Sequence[int]] = None,
                normalize: bool = True) -> np.ndarray:
    """The cosine-normalised linear kernel of the site profiles: the same kernel the whole-cell
    autocorrelation uses, so that only the descriptor differs between the two.

    `normalize=False` keeps the length of each row, which the cosine kernel divides out. It is
    here for the measurement of plan item M15 and changes no default.
    """
    return autocorrelation_kernel(train, test, n_bins=n_bins, blocks=blocks,
                                  normalize=normalize)


def site_targets(values: Sequence[Sequence[float]], name: str = "") -> AtomTargets:
    """The target index of a per-site property: one value per site of every cell, not a number
    where the site was not measured."""
    return AtomTargets.from_arrays(values, name=name)


def one_site_targets(sites: Sequence[int], values: Sequence[float], n_sites: Sequence[int],
                     name: str = "") -> AtomTargets:
    """The target index of a set where one site of every cell is measured: the site of each cell,
    its value, and how many sites that cell has."""
    if not (len(sites) == len(values) == len(n_sites)):
        raise ValueError(f"{len(sites)} sites, {len(values)} values and {len(n_sites)} cell sizes")
    per = []
    for s, v, n in zip(sites, values, n_sites):
        row = np.full(int(n), np.nan)
        row[int(s)] = float(v)
        per.append(row)
    return AtomTargets.from_arrays(per, name=name)


# --------------------------------------------------------------------------- #
# The reference under rule 1: counts in shells around the site
# --------------------------------------------------------------------------- #
def elements_of(crystals: Sequence[CrystalFields], min_sites: int = 1) -> List[str]:
    """The elements that appear in the set, in alphabetical order."""
    seen: Dict[str, int] = {}
    for c in crystals:
        for s in c.symbols:
            seen[s] = seen.get(s, 0) + 1
    return sorted(s for s, n in seen.items() if n >= min_sites)


def shell_counts(crystal: CrystalFields, site: int, elements: Sequence[str],
                 shells: Sequence[float] = SHELLS) -> np.ndarray:
    """``(1 + n_shells, n_elements)`` flattened: the element of the site itself, then how many
    atoms of every element lie in each distance shell around it, images included."""
    idx = {e: k for k, e in enumerate(elements)}
    out = np.zeros((1 + len(shells), len(elements)))
    s = crystal.symbols[int(site)]
    if s in idx:
        out[0, idx[s]] = 1.0
    i, j, d = _neighbour_pairs(crystal, float(max(shells)))
    here = i == int(site)
    edges = np.asarray(shells, float)
    for m, dist in zip(j[here], d[here]):
        sym = crystal.symbols[int(m)]
        if sym not in idx:
            continue
        shell = int(np.searchsorted(edges, dist, side="left"))
        if shell < len(shells):
            out[1 + shell, idx[sym]] += 1.0
    return out.ravel()


def shell_count_features(crystals: Sequence[CrystalFields], targets: AtomTargets,
                         elements: Optional[Sequence[str]] = None,
                         shells: Sequence[float] = SHELLS) -> np.ndarray:
    """The count reference of every measured sample, one row each."""
    elements = list(elements) if elements is not None else elements_of(crystals)
    out = np.zeros((len(targets), (1 + len(shells)) * len(elements)))
    for r in range(len(targets)):
        k, s = int(targets.molecule[r]), int(targets.atom[r])
        out[r] = shell_counts(crystals[k], s, elements, shells)
    return out


def count_labels(elements: Sequence[str], shells: Sequence[float] = SHELLS) -> List[str]:
    """One label per column of the count reference."""
    out = [f"the site is {e}" for e in elements]
    low = 0.0
    for edge in shells:
        for e in elements:
            out.append(f"{e} between {low:.1f} and {edge:.1f} A")
        low = edge
    return out


# --------------------------------------------------------------------------- #
# Reading a whole-cell model site by site
# --------------------------------------------------------------------------- #
def cell_norm(crystal: CrystalFields) -> float:
    """What the cell descriptor divides its site rows by: the number of atoms, or the expected
    number on a disordered cell (plan item M2)."""
    if crystal.occupancy is not None:
        return max(float(np.sum(crystal.occupancy)), 1e-12)
    return max(float(crystal.n_atoms), 1e-12)


def site_contributions(model, crystal: CrystalFields,
                       spec: Optional[PeriodicSpec] = None) -> np.ndarray:
    """One number per site of `crystal`: what that site adds to the prediction of a fitted
    whole-cell model on the periodic autocorrelation.

    The sum of the numbers is the prediction minus the bias, exactly.  The kernel is the dot
    product of the scaled, unit-normalised channels, the cell's channels are the mean of its
    site profiles, and a linear layer is linear, so the decomposition needs nothing attributed:
    it is rule 3 of the plan for the autocorrelation kernel.  Defined for the linear layer only.
    """
    from .periodic import channel_weights
    w = channel_weights(model)                      # the co-pattern of the fitted model
    spec = spec if spec is not None else model._ac_spec()
    rows = site_autocorrelation(crystal, spec)
    scale = np.asarray(model.ac_column_scale, float)
    n = cell_norm(crystal)
    cell = (rows.sum(axis=0) / n) * scale
    norm = float(np.sqrt(max(float(cell @ cell), 1e-300)))
    return (rows * scale) @ w / (norm * n)


def site_contribution_table(model, crystal: CrystalFields, n: int = 10,
                            spec: Optional[PeriodicSpec] = None) -> List[Tuple[str, float]]:
    """The `n` sites that move the prediction most, as (``index element``, contribution)."""
    v = site_contributions(model, crystal, spec)
    order = np.argsort(-np.abs(v))[:n]
    return [(f"{int(k)} {crystal.symbols[int(k)]}", float(v[k])) for k in order]


def site_top_channels(model, crystal: CrystalFields, site: int, labels: Sequence[str],
                      n: int = 10, spec: Optional[PeriodicSpec] = None) -> List[Tuple[str, float]]:
    """The `n` channels of one site's profile that move the prediction most, as (label, amount).

    The amounts sum to the contribution of that site, so this says which pair of field types at
    which distance *from this site* raises the property and which lowers it.
    """
    from .periodic import channel_weights
    w = channel_weights(model)
    spec = spec if spec is not None else model._ac_spec()
    rows = site_autocorrelation(crystal, spec)
    scale = np.asarray(model.ac_column_scale, float)
    per_atom = cell_norm(crystal)
    cell = (rows.sum(axis=0) / per_atom) * scale
    norm = float(np.sqrt(max(float(cell @ cell), 1e-300)))
    v = rows[int(site)] * scale * w / (norm * per_atom)
    order = np.argsort(-np.abs(v))[:n]
    return [(labels[c], float(v[c])) for c in order]
