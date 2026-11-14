"""A kernel that knows the resolution of its description (plan item MS4).

Two things live here, and they rest on one fact: a CMF field is a sum of Gaussian functions, and
the width of each Gaussian is the resolution at which the field describes its atom.

**Several widths in one model.** The field width (``alpha``) is chosen per model, so a model sees
the structure at one resolution: fine with a narrow Gaussian, coarse with a wide one. A sum of
the kernels at several widths sees both at once, and it is still a valid kernel. The weights of
the sum are fitted by the evidence of the Gaussian-process reading of the model
(:func:`cmfqsar.mixing.evidence_mixing`), the same rule that fits the weights of the field types,
so no width has to be chosen by hand. :func:`width_kernels` builds the stack of kernels and
:func:`fit_width_mixture` the weights.

**Beads.** A coarse-grained model replaces a group of atoms by one bead. In CMF terms a bead is
one Gaussian that stands for the Gaussians of its atoms. Its centre is the centre of the atoms,
its weight in each field is the sum of their weights, and its width matches the second moment
of the group: the variance of the bead's Gaussian along each axis is the variance of one atom's
Gaussian plus the variance of the atom positions about the centre. So the field of a bead has the
same total weight and the same spread as the field of its atoms; what it loses is the shape
inside the group, for example the dipole of a carbonyl group, whose charges cancel in the sum.

The kernel of two bead descriptions needs Gaussians of different widths, one per bead, and the
program has that integral already: it is the radius-scaled overlap of the steric field
(``b = alpha / R^2``). A bead of per-axis spread ``s^2`` is therefore given the radius
``R = sqrt(1 + alpha s^2)`` in an ordinary field and ``R = sqrt(<R_atom^2> + alpha s^2)`` in the
steric field, and the existing closed-form code computes its kernel. A bead of one atom has
``s = 0`` and gives back the atom's own Gaussian exactly.

:func:`bead_groups` maps atoms to beads by a stated rule (``"united"``: each heavy atom with its
hydrogens; ``"four"``: about four heavy atoms per bead, grown along bonds in the manner of the
MARTINI force field; ``"atoms"``: every atom its own bead, the control); :func:`beads_of`
builds the beads of one molecule and :func:`bead_kernels` their kernel matrices.
"""

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .fields import AtomTable, FieldSet, FieldSpec, MoleculeFields
from .kernels import kernel_matrices, normalize_kernels, self_kernels

RULES = ("atoms", "united", "four")
#: heavy atoms per bead in the "four" rule
FOUR = 4


# ------------------------------------------------------------------------- several widths

def _specs_at(fieldset: FieldSet, alpha: float) -> List[FieldSpec]:
    return [dataclasses.replace(s, alpha=float(alpha)) for s in fieldset.specs]


def cosine_field_kernels(P: AtomTable, specs: Sequence[FieldSpec],
                         Q: Optional[AtomTable] = None) -> np.ndarray:
    """Per-field kernels (F, nP, nQ), each divided by the square root of its two self-kernels."""
    Q = P if Q is None else Q
    K = kernel_matrices(P, Q, specs)
    dP = self_kernels(P, specs)
    dQ = dP if Q is P else self_kernels(Q, specs)
    return normalize_kernels(K, dP, dQ)


def width_kernels(P: AtomTable, fieldset: FieldSet, alphas: Sequence[float],
                  Q: Optional[AtomTable] = None) -> np.ndarray:
    """The joint kernel at every width: an array (W, nP, nQ).

    At each width the per-field kernels are cosine-normalised and summed over the fields the
    field set enables, which is the joint kernel with equal field weights.
    """
    on = [s.enabled for s in fieldset.specs]
    out = []
    for a in alphas:
        Kf = cosine_field_kernels(P, _specs_at(fieldset, a), Q)
        out.append(Kf[np.asarray(on, bool)].sum(axis=0))
    return np.asarray(out)


def fit_width_mixture(stack: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
    """Weights of the widths that maximise the evidence, one per width, for :func:`mixed_kernel`.

    The prior of :func:`cmfqsar.mixing.evidence_mixing` is centred at equal weights, so a width
    gets a large share only when the evidence asks for it. Extra keywords go to that function.
    """
    from .mixing import evidence_mixing
    return evidence_mixing(np.asarray(stack, float), np.asarray(y, float), **kw)


def mixed_kernel(stack: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """The weighted sum of the kernels of a stack."""
    from .mixing import joint_kernel
    stack = np.asarray(stack, float)
    return joint_kernel(stack, weights, np.ones(stack.shape[0]))


# ------------------------------------------------------------------------------------- beads

def bead_groups(mf: MoleculeFields, rule: str = "four") -> List[List[int]]:
    """The atoms of each bead, by a stated rule, from the atoms and bonds of one molecule.

    ``"atoms"`` makes every atom a bead. ``"united"`` makes each heavy atom one bead with the
    hydrogens bonded to it. ``"four"`` grows beads of up to four bonded heavy atoms: it starts
    from the unassigned heavy atom with the fewest unassigned heavy neighbours (the end of a
    chain first), adds unassigned heavy neighbours breadth first, and gives every hydrogen to
    the bead of its heavy atom. The order is fixed, so one molecule always gives the same beads.
    """
    if rule not in RULES:
        raise ValueError(f"the rule {rule!r} is not one of {RULES}")
    if mf.is_ensemble:
        raise ValueError("beads are built for one structure; this one is an ensemble")
    n = mf.n_atoms
    if rule == "atoms":
        return [[i] for i in range(n)]
    heavy = [i for i in range(n) if mf.symbols[i] != "H"]
    nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in np.asarray(mf.bonds, int).reshape(-1, 2):
        nbrs[int(a)].append(int(b))
        nbrs[int(b)].append(int(a))
    owner: Dict[int, int] = {}
    groups: List[List[int]] = []
    if rule == "united":
        for i in heavy:
            owner[i] = len(groups)
            groups.append([i])
    else:
        free = set(heavy)

        def free_heavy(i):
            return sorted(j for j in nbrs[i] if j in free)

        while free:
            start = min(free, key=lambda i: (len(free_heavy(i)), i))
            group = [start]
            free.discard(start)
            queue = [start]
            while queue and len(group) < FOUR:
                cur = queue.pop(0)
                for j in free_heavy(cur):
                    if len(group) >= FOUR:
                        break
                    group.append(j)
                    free.discard(j)
                    queue.append(j)
            for i in group:
                owner[i] = len(groups)
            groups.append(sorted(group))
    for i in range(n):
        if i in owner:
            continue
        heavy_nbrs = [j for j in nbrs[i] if j in owner]
        if heavy_nbrs:
            groups[owner[min(heavy_nbrs)]].append(i)
        else:
            owner[i] = len(groups)
            groups.append([i])
    return [sorted(g) for g in groups]


@dataclass
class Beads:
    """The beads of one molecule: centres, summed weights and the spread of their atoms."""
    name: str
    coords: np.ndarray            # (B, 3) the centre of each bead
    weights: np.ndarray           # (B, F) the summed weight of its atoms in each field
    spread: np.ndarray            # (B,) per-axis variance of its atom positions about the centre
    radius2: np.ndarray           # (B,) mean of the squared radii of its atoms
    groups: List[List[int]]

    @property
    def n_beads(self) -> int:
        return len(self.coords)


def beads_of(mf: MoleculeFields, rule: str = "four") -> Beads:
    """The beads of one molecule under a rule (see :func:`bead_groups`)."""
    groups = bead_groups(mf, rule)
    coords, weights, spread, radius2 = [], [], [], []
    for g in groups:
        x = np.asarray(mf.coords, float)[g]
        c = x.mean(axis=0)
        coords.append(c)
        weights.append(np.asarray(mf.weights, float)[g].sum(axis=0))
        spread.append(float(np.mean(np.sum((x - c) ** 2, axis=1))) / 3.0)
        radius2.append(float(np.mean(np.asarray(mf.radii, float)[g] ** 2)))
    return Beads(name=mf.name, coords=np.asarray(coords), weights=np.asarray(weights),
                 spread=np.asarray(spread), radius2=np.asarray(radius2), groups=groups)


def _as_fields(b: Beads, radii: np.ndarray) -> MoleculeFields:
    k = b.n_beads
    return MoleculeFields(name=b.name, coords=b.coords, weights=b.weights, radii=radii,
                          symbols=["X"] * k, atom_types=["bead"] * k,
                          bonds=np.zeros((0, 2), int), charges=np.zeros(k))


def bead_kernels(P: Sequence[Beads], fieldset: FieldSet, alpha: float,
                 Q: Optional[Sequence[Beads]] = None, normalise: bool = True) -> np.ndarray:
    """Per-field kernels (F, nP, nQ) of two lists of bead descriptions at one width.

    Every bead is a Gaussian of its own width, so every field is computed with the
    radius-scaled overlap: ordinary fields with ``R^2 = 1 + alpha s^2``, the steric field with
    ``R^2 = <R_atom^2> + alpha s^2``. With `normalise` each field is cosine-normalised.
    """
    same = Q is None
    Q = P if same else Q
    specs = _specs_at(fieldset, alpha)
    kinds = [s.kind for s in specs]
    bad = [s.name for s in specs if s.kind not in ("gauss", "steric")
           or getattr(s, "is_probe", False)]
    if bad:
        raise ValueError(f"beads are defined for ordinary and steric fields; not for {bad}")
    a = float(alpha)

    def tables(items, steric):
        return AtomTable.from_molecules([
            _as_fields(b, np.sqrt((b.radius2 if steric else 1.0) + a * b.spread)) for b in items])

    # the kernel code reads the weights of the spec at position f from column f, so the full
    # list is passed each time and the rows of the other kind are dropped afterwards
    as_radius = [dataclasses.replace(s, kind="steric") for s in specs]
    out = np.zeros((len(specs), len(P), len(Q)))
    for steric in (False, True):
        idx = [f for f, k in enumerate(kinds) if (k == "steric") == steric]
        if not idx:
            continue
        tp = tables(P, steric)
        tq = tp if same else tables(Q, steric)
        K = kernel_matrices(tp, tq, as_radius)
        if normalise:
            dp = self_kernels(tp, as_radius)
            dq = dp if same else self_kernels(tq, as_radius)
            K = normalize_kernels(K, dp, dq)
        out[idx] = K[idx]
    return out


def bead_joint_kernel(P: Sequence[Beads], fieldset: FieldSet, alpha: float,
                      Q: Optional[Sequence[Beads]] = None) -> np.ndarray:
    """The joint bead kernel with equal field weights, as :func:`width_kernels` builds it."""
    on = np.asarray([s.enabled for s in fieldset.specs], bool)
    return bead_kernels(P, fieldset, alpha, Q)[on].sum(axis=0)


# --------------------------------------------------------------------- agreement of kernels

def kernel_correlation(K: np.ndarray, L: np.ndarray) -> float:
    """Pearson correlation of the off-diagonal entries of two kernels (the reading of EC5)."""
    iu = np.triu_indices_from(np.asarray(K), 1)
    a, b = np.asarray(K, float)[iu], np.asarray(L, float)[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def centred_alignment(K: np.ndarray, L: np.ndarray) -> float:
    """Centred kernel alignment: the cosine of the two kernels after removing the mean row."""
    n = np.asarray(K).shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ np.asarray(K, float) @ H
    Lc = H @ np.asarray(L, float) @ H
    denom = np.linalg.norm(Kc) * np.linalg.norm(Lc)
    return float((Kc * Lc).sum() / denom) if denom > 1e-300 else float("nan")
