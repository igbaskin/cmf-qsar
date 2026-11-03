"""
Non-field descriptors: one number per molecule, beside the fields rather than instead of them.

Everything else in this program describes a molecule as a function of space.  Some things are not
functions of space and never will be -- a measured logP, a molecular weight, a melting point, a
medium or a temperature a series was recorded at, a count of rotatable bonds, an energy from a
quantum calculation.  A 3D-QSAR model that cannot see them has to pretend the activity depends on
geometry alone, which for an adsorption potency or a permeability is simply false.

A descriptor block is therefore added to the kernel as a block of its own::

    K(A, B) = sum_f h_f c_f K_f(A, B)  +  h_d c_d  z(A) . z(B)

where ``z`` holds the descriptors of a molecule, standardised with the **training set's** mean and
standard deviation so that a descriptor in kilojoules does not dominate one in log units, and
``h_d`` is a mixing coefficient like any field's.  The block is a linear kernel, so three things
follow and all three are worth having:

* the model stays linear in its inputs, so the decomposition rules survive -- every prediction
  splits into a field part and a per-descriptor part, and :func:`contributions` reports the latter
  as one number per descriptor;
* a molecule's descriptor vector enters exactly as a field does, so cross-validation, the nested
  loop, the mixing search and the saved model need no special case;
* the block is *scaled*, not cosine-normalised: dividing by the two norms would send a molecule
  that happens to sit at the training mean in every descriptor to a zero row and a zero diagonal,
  i.e. out of the model altogether, which for a perfectly ordinary molecule is absurd.  Instead
  the block is multiplied by a constant that makes its mean diagonal 1 over the training set, so
  it enters on the same basis as a scaled field and the relative magnitudes survive.

What the block is *not* is a way to insert the activity itself into the inputs: a descriptor
correlated with the activity by construction will dominate and the geometry will look worthless.
The program reports the descriptor contributions beside the field ones so that this is visible
rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: what to do with a molecule that lacks a descriptor the model needs
MISSING_POLICIES = ("error", "mean")


#: descriptors the program can compute itself, so that a file carrying no numeric tags still has
#: something to offer.  They are named with a ``calc:`` prefix, which keeps them distinct from a
#: property of the same name that a file might carry and makes a command line unambiguous.
COMPUTED: Dict[str, Tuple[str, str]] = {
    "mw": ("molecular weight", "MolWt"),
    "clogp": ("Wildman-Crippen logP", "MolLogP"),
    "tpsa": ("topological polar surface area", "TPSA"),
    "rotatable": ("rotatable bonds", "NumRotatableBonds"),
}


#: descriptors a periodic structure can offer, under the same ``calc:`` prefix (plan item BT2).
#: They are the quantities an electrode series is compared on and that no field carries, because
#: they are properties of the composition and the cell rather than functions of space.  The
#: working ion is lithium unless the name says otherwise: ``calc:capacity@Na`` is the capacity for
#: sodium.
CRYSTAL: Dict[str, str] = {
    "capacity": "theoretical specific capacity for the working ion, mAh/g",
    "ion_count": "working-ion atoms per host formula unit: the x of Li_x Host",
    "ion_fraction": "working-ion atoms as a fraction of all atoms in the cell",
    "volume_per_formula": "cell volume per formula unit, cubic angstrom",
    "volume_per_atom": "cell volume per atom, cubic angstrom",
    "density": "density, g/cm3",
    "ion_coordination": "oxygen neighbours of a working-ion site, mean over the sites",
    "ion_oxygen_distance": "distance from a working-ion site to its oxygen neighbours, angstrom, "
                           "mean over the sites",
}

#: the working ion of the crystal descriptors when the name does not say which
WORKING_ION = "Li"


def computed_names() -> List[str]:
    """The computed descriptors, as they are written in a field set: ``calc:mw`` and so on."""
    return [f"calc:{k}" for k in COMPUTED]


def crystal_names() -> List[str]:
    """The crystal descriptors, as they are written in a field set: ``calc:capacity`` and so on."""
    return [f"calc:{k}" for k in CRYSTAL]


def is_structure(obj) -> bool:
    """Is this an ASE structure rather than an RDKit molecule?"""
    return hasattr(obj, "get_chemical_symbols") and hasattr(obj, "get_positions")


def _formula_units(symbols: Sequence[str]) -> int:
    """How many formula units the cell holds: the common divisor of the element counts."""
    from collections import Counter
    from math import gcd
    counts = list(Counter(symbols).values())
    z = 0
    for c in counts:
        z = gcd(z, int(c))
    return max(z, 1)


def _host_units(symbols: Sequence[str], ion: str) -> float:
    """How many host formula units the cell holds, the working ion left out of the count.

    This is the denominator of the *x* in Li_x Host, and it is not the formula unit of the whole
    cell: a cell of Li2Co4O8 holds two formula units of itself and four of the host CoO2, so its
    lithium content is 0.5 and not 1.
    """
    from collections import Counter
    from math import gcd
    counts = Counter(s for s in symbols if s != ion)
    if not counts:
        return float(len(list(symbols)))
    g = 0
    for c in counts.values():
        g = gcd(g, int(c))
    g = max(g, 1)
    return float(min(n / (int(n) // g) for n in counts.values()))


def _ion_shell(atoms, ion: str, anion: str = "O"):
    """The oxygen neighbours of every working-ion site: their count and their distances.

    The cutoff is one and a half times the sum of the two covalent radii, which holds the first
    shell of an oxide (2.9 angstrom for lithium and oxygen, 3.5 for sodium) and leaves the second
    one out.  The neighbour list is the periodic one, so a shell across the cell boundary counts.
    """
    import numpy as np
    from ase.data import atomic_numbers, covalent_radii
    from ase.neighborlist import neighbor_list
    symbols = list(atoms.get_chemical_symbols())
    ions = [i for i, s in enumerate(symbols) if s == ion]
    if not ions or anion not in symbols:
        return np.zeros(0), np.zeros(0)
    cutoff = 1.5 * (covalent_radii[atomic_numbers[ion]] + covalent_radii[atomic_numbers[anion]])
    i_idx, j_idx, dist = neighbor_list("ijd", atoms, cutoff)
    keep = np.array([symbols[i] == ion and symbols[j] == anion for i, j in zip(i_idx, j_idx)],
                    dtype=bool)
    if not keep.any():
        return np.zeros(len(ions)), np.zeros(0)
    i_keep, d_keep = i_idx[keep], dist[keep]
    counts = np.array([float(np.count_nonzero(i_keep == i)) for i in ions], float)
    return counts, d_keep


def compute_crystal(atoms, name: str) -> float:
    """One crystal descriptor of one periodic structure; NaN where the cell cannot give it.

    ``name`` is the short name, with an optional working ion after an ``@``: ``capacity@Na``.
    """
    import numpy as np
    key, _, ion = name.partition("@")
    ion = ion or WORKING_ION
    if key not in CRYSTAL:
        import difflib
        near = difflib.get_close_matches(key, list(CRYSTAL), n=4)
        raise KeyError(f"unknown crystal descriptor {key!r}"
                       + (f"; did you mean {', '.join(near)}?" if near else
                          f"; the crystal descriptors are {', '.join(CRYSTAL)}"))
    symbols = list(atoms.get_chemical_symbols())
    n_ion = float(sum(1 for s in symbols if s == ion))
    mass = float(np.sum(atoms.get_masses()))                 # g/mol of the whole cell
    try:
        volume = float(atoms.get_volume())
    except Exception:                                        # noqa: BLE001 - an open structure
        volume = float("nan")
    z = _formula_units(symbols)
    if key == "capacity":
        # n F / M, with F / 3.6 = 26801.5 mAh per mole of charge; one electron per working ion
        return float("nan") if mass <= 0 else 26801.5 * n_ion / mass
    if key == "ion_count":
        # per host formula unit, so that Li2Co4O8 reads 0.5 and not 1: this is the x of Li_x Host
        host = _host_units(symbols, ion)
        return n_ion / host if host > 0 else float("nan")
    if key == "ion_fraction":
        return n_ion / len(symbols) if symbols else float("nan")
    if key == "volume_per_formula":
        return volume / z
    if key == "volume_per_atom":
        return volume / len(symbols) if symbols else float("nan")
    if key == "density":
        # 1 amu per cubic angstrom is 1.66054 g/cm3
        return float("nan") if not np.isfinite(volume) or volume <= 0 else 1.66054 * mass / volume
    counts, dist = _ion_shell(atoms, ion)
    if key == "ion_coordination":
        return float(np.mean(counts)) if len(counts) else float("nan")
    if key == "ion_oxygen_distance":
        return float(np.mean(dist)) if len(dist) else float("nan")
    raise AssertionError(key)                                # unreachable: CRYSTAL is checked


def rdkit_descriptors() -> Dict[str, object]:
    """Every descriptor RDKit exposes, by its own name: about 200 of them, as a lookup table."""
    from rdkit.Chem import Descriptors
    return {name: fn for name, fn in Descriptors._descList}


def _resolve(key: str):
    """The function behind a `calc:` name: one of the four short names, or any RDKit descriptor."""
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    short = {"mw": Descriptors.MolWt, "clogp": Crippen.MolLogP, "tpsa": Descriptors.TPSA,
             "rotatable": Lipinski.NumRotatableBonds}
    if key in short:
        return short[key]
    table = rdkit_descriptors()
    if key in table:
        return table[key]
    lowered = {k.lower(): v for k, v in table.items()}
    if key.lower() in lowered:                       # RDKit's names are CamelCase; accept any case
        return lowered[key.lower()]
    import difflib
    near = difflib.get_close_matches(key, list(table) + list(short), n=4)
    raise KeyError(f"unknown computed descriptor {key!r}"
                   + (f"; did you mean {', '.join(near)}?" if near else
                      f"; the short names are {', '.join(COMPUTED)} and any of the "
                      f"{len(table)} RDKit descriptors may be named directly"))


def compute(mol, name: str) -> float:
    """One computed descriptor of one molecule; NaN if RDKit cannot produce it.

    The four short names are shortcuts; beyond them any of RDKit's own descriptors may be asked
    for by its RDKit name -- ``calc:NumHAcceptors``, ``calc:FractionCSP3``, ``calc:BalabanJ`` --
    which is why there is no list here to keep up to date.
    """
    key = name[5:] if name.startswith("calc:") else name
    if is_structure(mol):                             # a periodic structure, not a molecule (BT2)
        try:
            return float(compute_crystal(mol, key))
        except KeyError:
            raise
        except Exception:                             # noqa: BLE001 - one descriptor, not the set
            return float("nan")
    fn = _resolve(key)
    try:
        return float(fn(mol))
    except Exception:                                 # noqa: BLE001 - one descriptor, not the set
        return float("nan")


def available(mols: Sequence) -> List[str]:
    """The molecule properties that are numeric on at least one molecule, in file order."""
    out: List[str] = []
    for mol in mols:
        if mol is None:
            continue
        for name in mol.GetPropNames():
            if name in out:
                continue
            try:
                float(mol.GetProp(name))
            except (ValueError, TypeError):
                continue
            out.append(name)
    return out


def read_values(mol, names: Sequence[str]) -> np.ndarray:
    """The descriptor vector of one molecule or one periodic structure; anything missing is NaN.

    A molecule's own properties are read from the file's tags, a structure's from
    ``atoms.info``, which is where a loader puts the numbers that came with the structure
    (plan item BT2).
    """
    out = np.full(len(names), np.nan)
    if mol is None:
        return out
    structure = is_structure(mol)
    for k, name in enumerate(names):
        if name.startswith("calc:"):
            out[k] = compute(mol, name)
            continue
        if structure:
            value = mol.info.get(name)
            if value is None:
                continue
            try:
                out[k] = float(value)
            except (ValueError, TypeError):
                continue
            continue
        if not mol.HasProp(name):
            continue
        try:
            out[k] = float(mol.GetProp(name))
        except (ValueError, TypeError):
            continue
    return out


@dataclass
class DescriptorScaler:
    """Mean and spread of every descriptor over the training set, and nothing else.

    The statistics belong to the training molecules and are stored with the model: standardising a
    test molecule by its own set would score two test sets on two different scales, which is the
    same mistake as rescaling the field kernels at prediction time.
    """
    names: Tuple[str, ...] = ()
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    missing: str = "error"

    @classmethod
    def fit(cls, values: np.ndarray, names: Sequence[str], missing: str = "error") -> "DescriptorScaler":
        V = np.atleast_2d(np.asarray(values, float))
        if missing not in MISSING_POLICIES:
            raise ValueError(f"missing policy must be one of {MISSING_POLICIES}")
        if missing == "error":
            bad = [names[k] for k in range(V.shape[1]) if not np.isfinite(V[:, k]).all()]
            if bad:
                raise ValueError(
                    f"descriptor(s) {', '.join(bad)} are missing or unreadable on some training "
                    f"molecules; give every molecule a value or use the 'mean' policy, which "
                    f"substitutes the training mean and therefore says nothing about those "
                    f"molecules")
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(V, axis=0)
            sd = np.nanstd(V, axis=0, ddof=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        # a constant descriptor carries no information; its spread of zero would divide by zero,
        # so it is left at zero and contributes nothing rather than infinity
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, np.inf)
        return cls(names=tuple(names), mean=mean, sd=sd, missing=missing)

    @property
    def constant(self) -> List[str]:
        """Descriptors that do not vary over the training set, and so contribute nothing."""
        return [n for n, s in zip(self.names, self.sd) if not np.isfinite(s)]

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Standardised descriptors; a missing value becomes the training mean, i.e. zero."""
        V = np.atleast_2d(np.asarray(values, float))
        Z = (V - self.mean) / self.sd
        return np.where(np.isfinite(Z), Z, 0.0)

    def describe(self) -> str:
        if not self.names:
            return "no descriptors"
        parts = [f"{n} (mean {m:.4g}, sd {'constant' if not np.isfinite(s) else format(s, '.4g')})"
                 for n, m, s in zip(self.names, self.mean, self.sd)]
        return "Descriptors: " + "; ".join(parts)


def descriptor_kernel(Z_rows: np.ndarray, Z_cols: np.ndarray,
                      normalize: bool = False) -> np.ndarray:
    """The block itself: a dot product of standardised vectors, cosine-normalised only on request.

    Normalisation is off by default and the model leaves it off.  For a field, dividing by the
    norms removes molecular size and keeps the shape; for a descriptor vector it removes the one
    thing the descriptors are there to carry, and it sends any molecule sitting at the training
    mean -- an unremarkable molecule, not a degenerate one -- to a zero row and a zero diagonal.
    The magnitude is handled by scaling the whole block instead, exactly as *Scale fields* does.
    """
    A = np.atleast_2d(np.asarray(Z_rows, float))
    B = np.atleast_2d(np.asarray(Z_cols, float))
    K = A @ B.T
    if not normalize:
        return K
    na = np.sqrt(np.maximum(np.sum(A * A, axis=1), 0.0))
    nb = np.sqrt(np.maximum(np.sum(B * B, axis=1), 0.0))
    denom = np.outer(na, nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-12, K / denom, 0.0)


def descriptor_self(Z: np.ndarray, normalize: bool = False) -> np.ndarray:
    """K_d(x, x) for every row, which the one-class scorers and the GP variance need."""
    A = np.atleast_2d(np.asarray(Z, float))
    d = np.sum(A * A, axis=1)
    if not normalize:
        return d
    return np.where(d > 1e-12, 1.0, 0.0)


def contributions(Z_mol: np.ndarray, Z_train: np.ndarray, coef: np.ndarray, h: float = 1.0,
                  scale: float = 1.0, normalize: bool = False) -> np.ndarray:
    """What each descriptor contributed to one prediction.

    The block is a dot product, so the prediction's descriptor part splits exactly::

        h c z(x) . sum_j a_j z(j)  =  sum_d  [ h c z_d(x) (sum_j a_j z_dj) ]

    one term per descriptor, and they add up to the block's whole contribution.  With cosine
    normalisation the two norms divide in, which is done here as well so that the terms still sum
    to the number the model actually used.
    """
    z = np.asarray(Z_mol, float).ravel()
    Z = np.atleast_2d(np.asarray(Z_train, float))
    a = np.asarray(coef, float).ravel()
    w = Z * a[:, None]                                   # per training molecule, per descriptor
    if normalize:
        nz = np.sqrt(max(float(z @ z), 0.0))
        nj = np.sqrt(np.maximum(np.sum(Z * Z, axis=1), 0.0))
        if nz <= 1e-12:
            return np.zeros_like(z)
        w = np.where(nj[:, None] > 1e-12, w / np.maximum(nj[:, None], 1e-12), 0.0)
        z = z / nz
    return float(h) * float(scale) * z * w.sum(axis=0)


# --------------------------------------------------------------------------- #
# A whole-molecule fingerprint as a kernel
# --------------------------------------------------------------------------- #


def fingerprint_kernel(mols: Sequence, radius: int = 2, bits: int = 2048) -> np.ndarray:
    """The Tanimoto coefficient of Morgan fingerprints, which is a positive semi-definite kernel.

    The default is 2048-bit ECFP4 (radius 2), the fingerprint the screening literature's fusion
    rules were established on.  It describes a molecule by its substructures and knows nothing
    of its shape or its pose, which is what makes it the second view plan item F6 measures the
    field against: what the field sees that the fingerprint does not is the part of the
    description that is three-dimensional.  :mod:`cmfqsar.bioisostere` has the same similarity
    for substituents; this one is over whole molecules.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(bits))
    fps = [gen.GetFingerprint(m) for m in mols]
    n = len(fps)
    S = np.eye(n)
    for i in range(n):
        S[i, :] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    return 0.5 * (S + S.T)
