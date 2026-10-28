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


def computed_names() -> List[str]:
    """The computed descriptors, as they are written in a field set: ``calc:mw`` and so on."""
    return [f"calc:{k}" for k in COMPUTED]


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
    """The descriptor vector of one molecule; anything missing or unreadable becomes NaN."""
    out = np.full(len(names), np.nan)
    if mol is None:
        return out
    for k, name in enumerate(names):
        if name.startswith("calc:"):
            out[k] = compute(mol, name)
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
