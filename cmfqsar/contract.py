"""What has to match between fitting and predicting, checked rather than assumed.

A saved model predicts a new molecule by the overlap of its fields with the training fields, so
the prediction is only meaningful when the new molecule was prepared the way the training
molecules were.  The model file records how they were prepared (:attr:`CMFModel.provenance`,
written by :meth:`CMFModel.state`), and this module compares a set of molecules about to be
predicted with that record.  It returns findings of three levels:

* ``refuse`` -- the prediction would be silently wrong: the model was fitted on partial charges
  read from the file and a molecule carries none, so the program would fall back to Gasteiger
  charges and call the result the same electrostatic field.
* ``warn`` -- the prediction is probably wrong and the user should look: the molecule has no
  explicit hydrogens where the training set had them (or the reverse), its atoms lie outside the
  frame of the training set, or a per-atom property field is empty for it.
* ``note`` -- worth knowing: the model was written by another version of the program, or by a
  version that recorded no provenance at all.

The command-line tool stops on a refusal unless told ``--allow-mismatch``; the application asks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .fields import MoleculeFields
from .model import CMFModel

#: how far outside the training set's bounding box a molecule's centroid may sit before the
#: frame is doubted (A); the fields fall off over a few angstroms, so beyond this the overlap
#: with every training molecule is small and the prediction is the bias
FRAME_MARGIN = 4.0

#: how near an atom must come to some training atom to count as occupying the training volume
#: (A).  The box of a training set is mostly empty -- a congeneric series fills a shape, not a
#: box -- so the cloud is what a new molecule has to lie in, and 1.5 A is about one bond
OCCUPANCY_RADIUS = 1.5

#: the occupancy of a molecule is doubted below the training set's own fifth percentile, less
#: this much.  Taking the threshold from the set rather than from a fixed number is what the
#: applicability domain does, and for the same reason: a tight series gets a tight rule
OCCUPANCY_SLACK = 0.05


@dataclass
class Finding:
    level: str          # "refuse" | "warn" | "note"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.message}"


def _names(mf: Sequence[MoleculeFields], idx: Sequence[int], limit: int = 5) -> str:
    names = [mf[i].name or f"#{i + 1}" for i in idx]
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _occupancy(model: CMFModel, coords: np.ndarray) -> float:
    """The share of `coords` that come within OCCUPANCY_RADIUS of a training atom."""
    from scipy.spatial import cKDTree

    tree = getattr(model, "_contract_tree", None)
    if tree is None:
        tree = cKDTree(np.asarray(model.train.coords, float))
        try:
            model._contract_tree = tree
        except Exception:                      # noqa: BLE001 - a frozen model is fine, just slower
            pass
    d, _ = tree.query(np.asarray(coords, float))
    return float(np.mean(d <= OCCUPANCY_RADIUS))


def _occupancy_threshold(model: CMFModel, k: int = 24) -> "float | None":
    """How much overlap with itself the training set requires of a molecule, from its spread.

    Each training atom is measured against the nearest atom of a *different* training molecule,
    so that a molecule is never judged by its own atoms; the fifth percentile of the resulting
    per-molecule shares, less a slack, is the threshold.  A set whose molecules overlap closely
    gets a strict rule and a loose one gets a loose rule, which is the same principle the
    applicability domain's threshold follows.
    """
    from scipy.spatial import cKDTree

    cached = getattr(model, "_contract_occupancy", None)
    if cached is not None:
        return cached
    train = model.train
    coords = np.asarray(train.coords, float)
    offsets = np.asarray(getattr(train, "offsets", []), int)
    if len(coords) < 4 or len(offsets) < 3:
        return None
    owner = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    kk = int(min(max(2, k), len(coords)))
    d, j = cKDTree(coords).query(coords, k=kk)
    shares = []
    for a in range(len(offsets) - 1):
        sel = np.arange(offsets[a], offsets[a + 1])
        if not len(sel):
            continue
        near = np.full(len(sel), np.inf)
        for col in range(kk):
            other = owner[j[sel, col]] != a
            near = np.where(np.isinf(near) & other, d[sel, col], near)
        shares.append(float(np.mean(near <= OCCUPANCY_RADIUS)))
    if not shares:
        return None
    thr = max(0.5, float(np.percentile(shares, 5)) - OCCUPANCY_SLACK)
    try:
        model._contract_occupancy = thr
    except Exception:                          # noqa: BLE001
        pass
    return thr


def check_molecules(model: CMFModel, mf: Sequence[MoleculeFields]) -> List[Finding]:
    """Compare molecules about to be predicted with how the model's training set was prepared."""
    out: List[Finding] = []
    prov = dict(getattr(model, "provenance", None) or {})
    if not prov:
        out.append(Finding("note", "no-provenance",
                           "the model file records nothing about how its training set was "
                           "prepared (written before version 1.0.0 recorded it); the checks "
                           "below use the training molecules stored in it instead"))
        prov = CMFModel.describe_training(model)
    else:
        from . import __version__
        if prov.get("cmfqsar") and prov["cmfqsar"] != __version__:
            out.append(Finding("note", "version",
                               f"the model was written by cmfqsar {prov['cmfqsar']}; this is "
                               f"{__version__}"))
    if not mf:
        return out

    # 1. the chemistry behind the electrostatic field: charges from the file, or computed
    if model.fieldset.charge_model == "file" and prov.get("charge_source") == "file":
        bad = [i for i, m in enumerate(mf) if getattr(m, "charge_source", "file") != "file"]
        if bad:
            out.append(Finding(
                "refuse", "charges",
                f"the model's electrostatic field was fitted on partial charges read from the "
                f"training file, and {len(bad)} of {len(mf)} molecules carry none "
                f"({_names(mf, bad)}): the program would fall back to Gasteiger charges and "
                f"call them the same field. Supply the charges (a MOL2 charge column or a "
                f"partial_charge atom property), or rebuild the model with computed charges."))

    # 2. explicit hydrogens: the donor field sits on them, the steric field counts them
    frac = prov.get("explicit_hydrogens")
    if frac is not None:
        with_h = [i for i, m in enumerate(mf) if "H" in m.symbols]
        without = [i for i in range(len(mf)) if i not in set(with_h)]
        if frac >= 0.5 and without:
            out.append(Finding(
                "warn", "hydrogens",
                f"the training molecules carry explicit hydrogens and {len(without)} of "
                f"{len(mf)} molecules do not ({_names(mf, without)}); the donor and steric "
                f"fields differ between the two conventions"))
        elif frac < 0.5 and with_h:
            out.append(Finding(
                "warn", "hydrogens",
                f"the training molecules carry no explicit hydrogens and {len(with_h)} of "
                f"{len(mf)} molecules do ({_names(mf, with_h)}); the donor and steric fields "
                f"differ between the two conventions"))

    # 3. the frame: a molecule outside the training set's box overlaps nothing
    lo, hi = prov.get("box_min"), prov.get("box_max")
    outside: List[int] = []
    if lo is not None and hi is not None and model.uses_fields:
        lo = np.asarray(lo, float) - FRAME_MARGIN
        hi = np.asarray(hi, float) + FRAME_MARGIN
        outside[:] = [i for i, m in enumerate(mf)
                   if np.any(m.coords.mean(axis=0) < lo) or np.any(m.coords.mean(axis=0) > hi)]
        if outside:
            out.append(Finding(
                "warn", "frame",
                f"{len(outside)} of {len(mf)} molecules sit outside the frame of the training "
                f"set ({_names(mf, outside)}): their centroid lies more than {FRAME_MARGIN:g} A "
                f"beyond the box of the training atoms, so they overlap no training field and "
                f"are predicted towards the mean. Align them to the training set first."))


    # 3b. the orientation: a molecule in the right place can still be turned the wrong way, and
    # then it overlaps the training fields as an unrelated molecule does -- the prediction falls
    # back towards the bias while the box test above sees nothing, because a rotation about the
    # molecule's own centre moves no centroid.  This is the one case the model file's record could
    # not see when it was written
    if model.uses_fields and model.train is not None and len(getattr(model.train, "coords", [])):
        thr = _occupancy_threshold(model)
        if thr is not None:
            turned, shares = [], []
            for i, m in enumerate(mf):
                if len(m.coords) < 3 or i in set(outside):
                    continue
                share = _occupancy(model, m.coords)
                if share < thr:
                    turned.append(i)
                    shares.append(share)
            if turned:
                out.append(Finding(
                    "warn", "orientation",
                    f"{len(turned)} of {len(mf)} molecules lie in the frame of the training set "
                    f"but not in its shape ({_names(mf, turned)}): "
                    f"{100 * float(np.mean(shares)):.0f} % of their atoms come within "
                    f"{OCCUPANCY_RADIUS:g} A of a training atom, against the "
                    f"{100 * thr:.0f} % this training set asks of its own molecules. A molecule "
                    f"turned the wrong way looks exactly like this -- its centroid is where it "
                    f"should be and its atoms are not -- and the model cannot tell it from a "
                    f"molecule it has nothing to say about."))

    # 4. a per-atom property field that is empty for a molecule: prop_missing = "zero" is silent
    for k, spec in enumerate(model.fieldset.specs):
        if not spec.is_prop or not spec.enabled:
            continue
        empty = [i for i, m in enumerate(mf)
                 if m.weights.shape[1] > k and not np.any(m.weights[:, k])]
        if empty:
            out.append(Finding(
                "warn", "property",
                f"the field '{spec.name}' is zero on {len(empty)} of {len(mf)} molecules "
                f"({_names(mf, empty)}): they carry no value of that per-atom property"))
    return out


#: how far a cell's lattice vectors may differ from the training mean before the frame of the
#: periodic overlap kernel is doubted, as a fraction.  The kernel compares two cells point by
#: point, so it is the crystal form of the frame check above
LATTICE_MARGIN = 0.10


def check_crystals(model: CMFModel, crystals: Sequence) -> List[Finding]:
    """Compare periodic structures about to be predicted with the training cells of the model.

    A crystal is prepared differently from a molecule, so the questions are different ones: the
    structure must carry a lattice, its elements must be ones the fields can see, the
    electrostatic weights must come from where they came from in training, and -- for the
    periodic overlap kernel, which compares two cells point by point -- the cells must stand on
    about one lattice.  The levels mean what they mean for molecules: ``refuse`` is a prediction
    that would be silently wrong, ``warn`` is one to look at, ``note`` is worth knowing.
    """
    from .periodic import lattice_spread

    out: List[Finding] = []
    prov = dict(getattr(model, "provenance", None) or {})
    if not prov:
        out.append(Finding("note", "no-provenance",
                           "the model file records nothing about how its training cells were "
                           "prepared; the checks below use the training structures stored in it"))
    else:
        from . import __version__
        if prov.get("cmfqsar") and prov["cmfqsar"] != __version__:
            out.append(Finding("note", "version",
                               f"the model was written by cmfqsar {prov['cmfqsar']}; this is "
                               f"{__version__}"))
    if not crystals:
        return out
    train = list(getattr(model.train, "molecules", []) or [])

    # 1. a lattice: without one there is no periodic field and no image sum
    flat = [i for i, c in enumerate(crystals)
            if getattr(c, "cell", None) is None
            or abs(float(np.linalg.det(np.asarray(c.cell, float).reshape(3, 3)))) < 1e-9]
    if flat:
        out.append(Finding(
            "refuse", "no-cell",
            f"{len(flat)} of {len(crystals)} structures carry no usable unit cell "
            f"({_names(crystals, flat)}): a periodic kernel sums over the images of the cell, "
            f"so there is nothing to predict from. Read them from a CIF, a POSCAR or an "
            f"extended-XYZ file that holds the lattice."))

    # 2. the elements: one the field set does not name is invisible, or lands in 'other'
    listed = {sp.name[4:] for sp in model.fieldset.specs if sp.is_cif and sp.enabled}
    has_other = "other" in listed
    seen = {s for c in train for s in c.symbols}
    if listed - {"other"}:
        unlisted = sorted({s for c in crystals for s in c.symbols} - listed)
        if unlisted and has_other:
            out.append(Finding(
                "warn", "elements",
                f"{len(unlisted)} elements of these structures have no indicator field of their "
                f"own ({', '.join(unlisted)}): they share the field 'cif:other', so the model "
                f"cannot tell them apart. Only the physico-chemical fields distinguish them."))
        elif unlisted:
            out.append(Finding(
                "refuse", "elements",
                f"{len(unlisted)} elements of these structures have no field at all "
                f"({', '.join(unlisted)}): the field set names indicator fields and none of "
                f"them, and no 'cif:other' field collects the rest, so those atoms carry no "
                f"indicator weight. Refit with these elements listed."))
    if seen:
        unseen = sorted({s for c in crystals for s in c.symbols} - seen)
        if unseen:
            out.append(Finding(
                "warn", "new-elements",
                f"the training cells contain no {', '.join(unseen)}; a prediction for a "
                f"structure made of elements the model has never seen is an extrapolation"))

    # 3. the electrostatic weights: the same source as in training, or the field is another field
    if any(sp.name == "electrostatic" and sp.enabled for sp in model.fieldset.specs):
        want = {getattr(c, "charge_source", "") for c in train if getattr(c, "charge_source", "")}
        got = {getattr(c, "charge_source", "") for c in crystals}
        new = sorted(s for s in got - want if s)
        if want and new:
            out.append(Finding(
                "warn", "charges",
                f"the electrostatic field of the training cells came from "
                f"{', '.join(sorted(want))} and these structures give it from "
                f"{', '.join(new)}: the two are not the same field. Supply the same kind of "
                f"charges, or refit."))

    # 4. one lattice, for the kernel that needs one
    if getattr(model, "pbc_overlap", False) and train:
        from .fields import AtomTable
        spread = lattice_spread(AtomTable.from_molecules(train))
        mean_len = np.mean([np.linalg.norm(np.asarray(c.cell, float).reshape(3, 3), axis=1)
                            for c in train], axis=0)
        tol = max(float(spread.get("length_spread", 0.0)), LATTICE_MARGIN)
        off = []
        for i, c in enumerate(crystals):
            if getattr(c, "cell", None) is None:
                continue
            lens = np.linalg.norm(np.asarray(c.cell, float).reshape(3, 3), axis=1)
            if np.any(np.abs(lens - mean_len) / np.maximum(mean_len, 1e-12) > tol):
                off.append(i)
        if off:
            out.append(Finding(
                "warn", "lattice",
                f"{len(off)} of {len(crystals)} cells lie further from the training lattice than "
                f"the training set's own spread allows ({_names(crystals, off)}): their lattice "
                f"vectors differ from the mean training cell by more than {100 * tol:.0f} %. The "
                f"periodic overlap kernel compares two cells point by point, so a cell on "
                f"another lattice is compared in a frame that does not exist, which is what a "
                f"bad alignment is for molecules."))

    # 5. partial occupancy: the exact mean field of a disordered site is not an ordered site
    dis_train = sum(1 for c in train if getattr(c, "site_species", None) is not None)
    dis_new = [i for i, c in enumerate(crystals) if getattr(c, "site_species", None) is not None]
    if dis_new and not dis_train:
        out.append(Finding(
            "warn", "occupancy",
            f"{len(dis_new)} of {len(crystals)} structures carry partly occupied sites "
            f"({_names(crystals, dis_new)}) and no training cell does: their fields are the "
            f"occupancy-weighted mean over the species of each site, which is a different "
            f"quantity from the fields of an ordered cell"))
    elif dis_train and not dis_new:
        out.append(Finding(
            "note", "occupancy",
            f"{dis_train} of the {len(train)} training cells carry partly occupied sites and "
            f"none of these structures does"))
    return out


def summary(model: CMFModel) -> str:
    """One line on how the model's training set was prepared, from its provenance."""
    p = dict(getattr(model, "provenance", None) or {})
    if not p:
        return "model file without provenance (written before version 1.0.0 recorded it)"
    if bool(getattr(model, "pbc_overlap", False)) or model.kernel_source == "periodic":
        # a crystal has no hydrogens to count and no increment scheme; what it has is a lattice,
        # a set of elements and a source for the electrostatic weights
        cells = list(getattr(model.train, "molecules", []) or [])
        elements = sorted({s for c in cells for s in c.symbols})
        atoms = np.mean([c.n_atoms for c in cells]) if cells else float("nan")
        kernel = ("periodic overlap kernel" if getattr(model, "pbc_overlap", False)
                  else "periodic field autocorrelation")
        return (f"cmfqsar {p.get('cmfqsar', '?')}, {p.get('created', '?')[:10]}: "
                f"{p.get('n_train', '?')} training cells, {len(p.get('fields', []))} fields, "
                f"{kernel}, electrostatic weights from '{p.get('charge_model', '?')}', "
                f"{atoms:.0f} atoms per cell on average, elements "
                f"{', '.join(elements) if elements else 'unknown'}")
    charges = {"file": "charges from the file", "gasteiger": "Gasteiger charges",
               "mmff": "MMFF charges"}.get(p.get("charge_source"), "charges of unknown origin")
    h = p.get("explicit_hydrogens")
    hyd = ("explicit hydrogens" if h is not None and h >= 0.5 else
           "no explicit hydrogens" if h is not None else "hydrogens unknown")
    return (f"cmfqsar {p.get('cmfqsar', '?')}, {p.get('created', '?')[:10]}: "
            f"{p.get('n_train', '?')} training molecules, {len(p.get('fields', []))} fields, "
            f"{p.get('param_scheme', '?')} increments, {charges}, {hyd}")
