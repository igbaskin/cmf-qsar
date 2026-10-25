"""A pharmacophore read out of a fitted co-field.

A pharmacophore is a handful of typed points with tolerance spheres; a co-field is a continuous
function saying, at every point in space, how much field of one type raises or lowers the
predicted activity.  The two are the same statement at different resolutions, and the fields of
this program carry exactly the classical feature types -- hydrogen-bond donor and acceptor,
hydrophobic, steric, electrostatic -- so the discretisation is a matter of finding the lobes:

* a connected region where the co-field of field *f* exceeds a fraction of its largest value is
  one feature; its centre is the field-weighted centroid of the region, and its tolerance radius
  the radius of the sphere of the same volume, so a broad lobe gives a loose feature and a tight
  one a strict feature;
* the sign says what the feature asks for.  Positive electrostatic co-field means a positive
  partial charge there raises the activity, negative means a negative one does; positive steric
  co-field means bulk is wanted, and **negative steric co-field is an exclusion volume** -- the
  model's own statement that atoms there lower the activity;
* each feature carries the integral of the co-field over its region, which is its contribution to
  the prediction for a unit of field density, and is what the features are ranked by.

With a protein loaded, the pocket supplies exclusion volumes of the harder kind: positions
actually occupied by the receptor.  Those come from :func:`pocket_exclusions` and are marked as
such, since a model fitted on ligands alone cannot know them.

The result is an ordinary pharmacophore hypothesis, derived from a fitted model rather than
guessed, which can be used to align molecules of unrelated scaffolds, to filter a library before
the field model scores it, or simply to read what the model has learnt.  What it is not is a
replacement for the co-field: discretising throws away what the continuous description was for.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .cofields import Grid, cofield_volume, make_grid

#: field name and sign -> (feature type, what it asks for, whether it is something to satisfy)
#: Both signs of the electrostatic co-field are features to satisfy -- one asks for positive
#: charge, the other for negative -- while a negative lobe of any other field is a warning.
FEATURE_TYPES = {
    ("electrostatic", +1): ("positive", "positive partial charge raises the activity", True),
    ("electrostatic", -1): ("negative", "negative partial charge raises the activity", True),
    ("hydrophobic", +1): ("hydrophobic", "hydrophobic bulk raises the activity", True),
    ("hydrophobic", -1): ("polar", "hydrophobic bulk lowers the activity", False),
    ("hba", +1): ("acceptor", "a hydrogen-bond acceptor raises the activity", True),
    ("hba", -1): ("acceptor_bad", "a hydrogen-bond acceptor lowers the activity", False),
    ("hbd", +1): ("donor", "a hydrogen-bond donor raises the activity", True),
    ("hbd", -1): ("donor_bad", "a hydrogen-bond donor lowers the activity", False),
    ("steric", +1): ("bulk", "atoms here raise the activity", True),
    ("steric", -1): ("exclusion", "atoms here lower the activity", False),
}

#: element written for every feature type in the PDB export, so that a viewer colours them apart
PDB_ELEMENT = {"positive": "N", "negative": "O", "hydrophobic": "C", "polar": "S",
               "acceptor": "O", "acceptor_bad": "S", "donor": "N", "donor_bad": "S",
               "bulk": "C", "exclusion": "Fe", "pocket": "Fe"}


@dataclass
class Feature:
    """One pharmacophore feature: where, how big, what it asks for, and how much it is worth."""
    kind: str                      # "donor", "acceptor", "hydrophobic", "exclusion", ...
    centre: np.ndarray
    radius: float                  # tolerance radius in angstrom
    field: str                     # the field it was read from ("" for pocket exclusions)
    sign: int                      # +1 favourable, -1 disfavoured
    peak: float = 0.0              # largest co-field value inside the region, with its own sign
    importance: float = 0.0        # how much the feature is worth, + to satisfy, - to avoid
    meaning: str = ""
    favourable: bool = True        # something to satisfy, or a region to keep away from

    def describe(self) -> str:
        return (f"{self.kind:12s} at ({self.centre[0]:7.2f},{self.centre[1]:7.2f},"
                f"{self.centre[2]:7.2f})  radius {self.radius:4.2f} A  "
                f"weight {self.importance:+9.3f}  {self.meaning}")


def _regions(volume: np.ndarray, level: float):
    """Connected components of `volume >= level`, as index arrays; a flood fill without scipy."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(volume >= level)
        return [np.flatnonzero(lab.ravel() == k + 1) for k in range(n)]
    except Exception:                                   # noqa: BLE001 - scipy is optional here
        mask = volume >= level
        seen = np.zeros(mask.shape, bool)
        out = []
        for start in zip(*np.nonzero(mask)):
            if seen[start]:
                continue
            stack, comp = [start], []
            seen[start] = True
            while stack:
                p = stack.pop()
                comp.append(np.ravel_multi_index(p, mask.shape))
                for d in range(3):
                    for step in (-1, 1):
                        q = list(p)
                        q[d] += step
                        q = tuple(q)
                        if all(0 <= q[k] < mask.shape[k] for k in range(3)) and mask[q] \
                                and not seen[q]:
                            seen[q] = True
                            stack.append(q)
            out.append(np.array(comp))
        return out


def features_from_volume(volume: np.ndarray, grid: Grid, field: str, sign: int,
                         level: float = 0.5, min_radius: float = 0.6) -> List[Feature]:
    """The lobes of one co-field, of one sign, as features."""
    w = sign * np.asarray(volume, float)
    peak = float(w.max()) if w.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return []
    cell = grid.spacing ** 3
    points = grid.points()
    kind, meaning, good = FEATURE_TYPES.get(
        (field, sign), (f"{field}{'+' if sign > 0 else '-'}", "", sign > 0))
    out = []
    for idx in _regions(w, level * peak):
        vals = w.ravel()[idx]
        total = float(vals.sum())
        if total <= 0:
            continue
        centre = (points[idx] * vals[:, None]).sum(axis=0) / total
        radius = float((3.0 * len(idx) * cell / (4.0 * np.pi)) ** (1.0 / 3.0))
        if radius < min_radius:
            continue
        # the weight says how much the feature is worth and whether it is to be satisfied or
        # avoided, which is not the same as the sign of the co-field: a negative electrostatic
        # lobe is a feature asking for negative charge, not a warning
        out.append(Feature(kind=kind, centre=centre, radius=radius, field=field, sign=sign,
                           peak=sign * float(vals.max()),
                           importance=(1.0 if good else -1.0) * total * cell,
                           meaning=meaning, favourable=good))
    return out


def extract(model, grid: Optional[Grid] = None, level: float = 0.5, max_features: int = 12,
            min_radius: float = 0.6, fields: Optional[Sequence[str]] = None,
            spacing: float = 0.6, margin: float = 3.0, progress=None,
            min_z: float = 0.0, sigma: Optional[float] = None) -> List[Feature]:
    """Read a pharmacophore out of the co-fields of a fitted model.

    `level` is the fraction of each co-field's largest value that bounds a lobe: 0.5 gives the
    half-height regions, a larger value tighter and fewer features.  Features are ranked by the
    integral of the co-field over them and the strongest `max_features` are kept, exclusion
    volumes counted separately so that they never crowd out the favourable ones.

    With `min_z` above zero the co-field is first masked where it stands less than that many
    standard deviations from zero (:func:`cmfqsar.cofields.cofield_with_error`), so that a lobe
    the activities' noise could have produced never becomes a feature.  Two is the usual value.
    A feature that survives is still a feature of this training set, but it is at least not a
    feature of its measurement error.
    """
    model._require_linear("pharmacophore features")
    model._require_fields("pharmacophore features")
    if model.train is None:
        raise RuntimeError("the model has no training molecules to span a grid")
    grid = grid or make_grid(model.train.coords, margin=margin, spacing=spacing)
    wanted = None if fields is None else {str(f) for f in fields}
    favourable: List[Feature] = []
    exclusions: List[Feature] = []
    specs = model.fieldset.specs
    for f, spec in enumerate(specs):
        if not spec.enabled or (wanted is not None and spec.name not in wanted):
            continue
        if progress is not None:
            progress(f"Co-field of {spec.label}", f / max(len(specs), 1))
        if min_z > 0:
            from .cofields import cofield_with_error
            ce = cofield_with_error(model, f, grid, sigma=sigma, include_scale=True, z=min_z)
            vol = ce.masked
        else:
            vol = cofield_volume(model, f, grid, include_scale=True)
        for sign in (+1, -1):
            for feat in features_from_volume(vol, grid, spec.name, sign, level=level,
                                             min_radius=min_radius):
                (exclusions if feat.kind == "exclusion" else favourable).append(feat)
    favourable.sort(key=lambda ft: -abs(ft.importance))
    exclusions.sort(key=lambda ft: -abs(ft.importance))
    return favourable[:max_features] + exclusions[:max_features]


def pocket_exclusions(protein, grid: Grid, max_spheres: int = 18, min_separation: float = 2.5,
                      radius: float = 1.6) -> List[Feature]:
    """Exclusion volumes from the receptor itself: where the protein already is.

    Heavy atoms inside the grid box are thinned to one sphere every `min_separation` angstrom,
    nearest the centre of the box first, so that a handful of spheres marks the walls instead of
    a copy of the pocket.
    """
    coords = np.asarray(protein.coords, float)
    heavy = np.array([str(e).upper() != "H" for e in protein.elements], bool) \
        if getattr(protein, "elements", None) is not None else np.ones(len(coords), bool)
    lo, hi = grid.extent
    inside = np.all((coords >= lo) & (coords <= hi), axis=1) & heavy
    pts = coords[inside]
    if not len(pts):
        return []
    centre = 0.5 * (np.asarray(lo) + np.asarray(hi))
    order = np.argsort(np.linalg.norm(pts - centre, axis=1))
    kept: List[np.ndarray] = []
    for i in order:
        p = pts[i]
        if all(np.linalg.norm(p - q) >= min_separation for q in kept):
            kept.append(p)
        if len(kept) >= max_spheres:
            break
    return [Feature(kind="pocket", centre=p, radius=radius, field="", sign=-1,
                    meaning="occupied by the receptor", favourable=False) for p in kept]


#: colours used for the features in the 3D view, by type
FEATURE_COLOUR = {"positive": (0.25, 0.45, 0.95), "negative": (0.95, 0.25, 0.25),
                  "hydrophobic": (0.95, 0.80, 0.25), "polar": (0.55, 0.55, 0.60),
                  "acceptor": (0.95, 0.35, 0.55), "acceptor_bad": (0.50, 0.35, 0.40),
                  "donor": (0.35, 0.85, 0.85), "donor_bad": (0.35, 0.55, 0.55),
                  "bulk": (0.45, 0.85, 0.45), "exclusion": (0.45, 0.45, 0.45),
                  "pocket": (0.30, 0.30, 0.30)}


def colour_of(feature: "Feature"):
    return FEATURE_COLOUR.get(feature.kind, (0.7, 0.7, 0.7))


# --------------------------------------------------------------------------- #
# Reporting and export
# --------------------------------------------------------------------------- #

def summary(features: Sequence[Feature]) -> str:
    if not features:
        return "no features above the threshold"
    kinds = {}
    for f in features:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    return f"{len(features)} features: " + ", ".join(f"{n} x {k}" for k, n in sorted(kinds.items()))


def write_csv(path: str, features: Sequence[Feature]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "x", "y", "z", "radius", "field", "sign", "peak", "weight", "meaning"])
        for f in features:
            w.writerow([f.kind, f"{f.centre[0]:.3f}", f"{f.centre[1]:.3f}", f"{f.centre[2]:.3f}",
                        f"{f.radius:.3f}", f.field, f.sign, f"{f.peak:.5f}",
                        f"{f.importance:.5f}", f.meaning])


def write_pdb(path: str, features: Sequence[Feature]) -> None:
    """Features as pseudo-atoms, so that any viewer can draw them.

    The residue name is the feature type, the occupancy its weight and the B-factor its tolerance
    radius; the element is chosen per type (N donor, O acceptor, C hydrophobic, Fe exclusion) only
    so that a viewer colours them apart.
    """
    with open(path, "w") as fh:
        fh.write("REMARK   pharmacophore extracted from a CMF co-field (cmfqsar.pharmacophore)\n")
        fh.write("REMARK   occupancy = weight, B-factor = tolerance radius in angstrom\n")
        for i, f in enumerate(features, start=1):
            el = PDB_ELEMENT.get(f.kind, "C")
            res = f.kind[:3].upper()
            fh.write(f"HETATM{i:5d} {el:<4s}{res:>3s} P{i:4d}    "
                     f"{f.centre[0]:8.3f}{f.centre[1]:8.3f}{f.centre[2]:8.3f}"
                     f"{min(abs(f.importance), 99.99):6.2f}{f.radius:6.2f}          {el:>2s}\n")
        fh.write("END\n")
