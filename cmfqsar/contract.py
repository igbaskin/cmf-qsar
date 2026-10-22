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
    if lo is not None and hi is not None and model.uses_fields:
        lo = np.asarray(lo, float) - FRAME_MARGIN
        hi = np.asarray(hi, float) + FRAME_MARGIN
        outside = [i for i, m in enumerate(mf)
                   if np.any(m.coords.mean(axis=0) < lo) or np.any(m.coords.mean(axis=0) > hi)]
        if outside:
            out.append(Finding(
                "warn", "frame",
                f"{len(outside)} of {len(mf)} molecules sit outside the frame of the training "
                f"set ({_names(mf, outside)}): their centroid lies more than {FRAME_MARGIN:g} A "
                f"beyond the box of the training atoms, so they overlap no training field and "
                f"are predicted towards the mean. Align them to the training set first."))

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


def summary(model: CMFModel) -> str:
    """One line on how the model's training set was prepared, from its provenance."""
    p = dict(getattr(model, "provenance", None) or {})
    if not p:
        return "model file without provenance (written before version 1.0.0 recorded it)"
    charges = {"file": "charges from the file", "gasteiger": "Gasteiger charges",
               "mmff": "MMFF charges"}.get(p.get("charge_source"), "charges of unknown origin")
    h = p.get("explicit_hydrogens")
    hyd = ("explicit hydrogens" if h is not None and h >= 0.5 else
           "no explicit hydrogens" if h is not None else "hydrogens unknown")
    return (f"cmfqsar {p.get('cmfqsar', '?')}, {p.get('created', '?')[:10]}: "
            f"{p.get('n_train', '?')} training molecules, {len(p.get('fields', []))} fields, "
            f"{p.get('param_scheme', '?')} increments, {charges}, {hyd}")
