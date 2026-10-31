"""Scoring a candidate only where the co-field has weight (plan item MC5).

Every other comparison in this program is made over the whole of both molecules.  That is right
for a congeneric series, where the molecules differ in a substituent and agree everywhere else,
and it is wrong for the question a chemist asks when a scaffold has to be replaced: *does this
other skeleton put the same field where the model rewards field?*  What the candidate does
elsewhere is not the question, and an overlap taken over the whole molecule answers mostly with
how similar the skeletons are.

Item E3 measured where that leaves scaffold hopping: the topomer frame is defined against a
scaffold that is held fixed, so replacing the scaffold leaves no frame, and the design loop has
nothing to pose the candidate with.  This module is the construction E3 lacked.  It needs no
frame of the candidate's own beyond a pose in the model's frame, and it asks nothing of the
candidate away from the region the model cares about.

How it works
------------
A **mask** is a soft region of space: a sum of Gaussians on the atoms whose contribution to the
prediction is largest (:func:`cmfqsar.cofields.top_atom_mask`).  Multiplying a co-field by it
gives the co-field inside that region and nothing outside, and because the product of two
Gaussians is a Gaussian the masked co-field is again a sum of Gaussians
(:func:`cmfqsar.kernels.gaussian_product`).  So rule 3 holds: every integral below is closed
form, with no grid and no sampling.

Three numbers come out of it, for every candidate:

* **inside** -- the part of the model's prediction that comes from the masked region,
  ``sum_f h_f c_f INT C_f(r) m(r) X_f(r) dr``, in the units of the measured activity.  This is
  the score a replacement scaffold should be ranked by.
* **outside** -- the rest of the prediction, ``prediction - bias - inside``.  It is the second
  score the item asks for: field the candidate places where the model rewards none.  A candidate
  whose promise is all outside is a candidate the model is extrapolating about.
* **cosine** -- the same comparison with the size of the molecule divided out.  The mask defines
  an inner product ``<f, g>_m = INT f g m``, and this is the cosine of the co-field and the
  candidate's field in it.  A raw overlap grows with the number of atoms, so a ranking by
  ``inside`` alone prefers large molecules; the cosine does not.

What it does not do
-------------------
It does not pose the candidate.  The molecules handed to it must already stand in the model's
frame -- docked, posed by field similarity, or built by a rule -- exactly as for
:func:`cmfqsar.screen.screen`.  A mask makes the *comparison* local; it does not make the
placement free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cofields import CofieldMask, _centres_and_widths, masked_cofield
from .fields import AtomTable, MoleculeFields, build_fields
from .kernels import expansion_overlap, gaussian_product
from .model import CMFModel


@dataclass
class MaskedScore:
    """What one candidate collects inside the mask, outside it, and how much of its size that is."""
    name: str
    inside: float
    outside: float
    prediction: float
    cosine: float
    n_atoms: int = 0
    conformer: int = 0

    def __str__(self) -> str:
        return (f"{self.name:<28s} inside {self.inside:+8.3f}  outside {self.outside:+8.3f}  "
                f"cosine {self.cosine:+.3f}  prediction {self.prediction:+8.3f}")


class MaskedScorer:
    """A model, a mask, and everything about them that does not depend on the candidate.

    The masked co-field is the same for every candidate, and it is the expensive part -- one
    product per field over the training atoms -- so it is built once here and reused.  Scoring a
    candidate is then two overlaps per field.
    """

    def __init__(self, model: CMFModel, mask: CofieldMask, fields: Optional[Sequence[int]] = None,
                 tol: float = 1e-9):
        model._require_linear("a masked score")
        self.model = model
        self.mask = mask
        self.tol = float(tol)
        n_fields = len(model.fieldset.specs)
        self.fields = list(range(n_fields)) if fields is None else [int(f) for f in fields]
        self.weight = model.fieldset.mixing_vector() * model.scale
        self.cofield = {}            # f -> the co-field's own Gaussians
        self.masked = {}             # f -> the same, multiplied by the mask
        self.self_masked = {}        # f -> <C_f, C_f>_m
        for f in self.fields:
            self.cofield[f] = model.cofield_coefficients(f)
            self.masked[f] = masked_cofield(model, f, mask, tol=self.tol)
            mc, mw, mb = self.masked[f]
            cc, cw, cb = self.cofield[f]
            self.self_masked[f] = expansion_overlap(mc, mw, mb, cc, cw, cb)

    def score(self, mol: MoleculeFields, name: Optional[str] = None,
              conformer: int = 0) -> MaskedScore:
        """The three numbers for one molecule, which must already stand in the model's frame."""
        inside = whole = cc_m = xx_m = 0.0
        for f in self.fields:
            w = float(self.weight[f])
            if w == 0.0:
                continue
            spec = self.model.fieldset.specs[f]
            xc, xw, xb = _centres_and_widths(mol, spec, f)
            mc, mw, mb = self.masked[f]
            cc, cw, cb = self.cofield[f]
            inside += w * expansion_overlap(mc, mw, mb, xc, xw, xb)
            whole += w * expansion_overlap(cc, cw, cb, xc, xw, xb)
            cc_m += w * self.self_masked[f]
            # <X_f, X_f>_m, the candidate's own size inside the mask
            px, pw, pb = gaussian_product(xc, xw, xb, *self.mask.expansion(), tol=self.tol)
            xx_m += w * expansion_overlap(px, pw, pb, xc, xw, xb)
        denom = np.sqrt(max(cc_m, 0.0) * max(xx_m, 0.0))
        cosine = float(inside / denom) if denom > 1e-300 else 0.0
        pred = whole + float(self.model.bias)
        return MaskedScore(name=name if name is not None else mol.name, inside=float(inside),
                           outside=float(whole - inside), prediction=float(pred), cosine=cosine,
                           n_atoms=int(mol.n_atoms), conformer=int(conformer))

    def prediction_check(self, mol: MoleculeFields) -> Tuple[float, float]:
        """(the primal prediction from these overlaps, the model's own) -- a test of rule 3.

        The two must agree: the unmasked overlap summed over the fields *is* the prediction, so a
        masked score that is built from the same integrals cannot be measuring something else.
        """
        s = self.score(mol)
        got = float(self.model.predict(AtomTable.from_molecules([mol]))[0])
        return s.prediction, got


def score_candidates(model: CMFModel, mols: Sequence[MoleculeFields], mask: CofieldMask,
                     fields: Optional[Sequence[int]] = None) -> List[MaskedScore]:
    """Every molecule of `mols` scored against the masked co-field, best first by the cosine."""
    scorer = MaskedScorer(model, mask, fields=fields)
    out = [scorer.score(m) for m in mols]
    out.sort(key=lambda s: -s.cosine)
    return out


def screen_conformers(model: CMFModel, mols, mask: CofieldMask, names: Optional[Sequence] = None,
                      fields: Optional[Sequence[int]] = None,
                      rank_by: str = "cosine") -> List[MaskedScore]:
    """The same over a conformer library: every conformation scored, the best one kept.

    `mols` are RDKit molecules that may carry several conformations each, already placed in the
    model's frame.  A candidate is as good as its best conformation, which is the question a
    library asks, and the conformer that wins is reported so that it can be looked at.
    """
    scorer = MaskedScorer(model, mask, fields=fields)
    best = {}
    for i, mol in enumerate(mols):
        name = (str(names[i]) if names is not None
                else (mol.GetProp("_Name") if mol.HasProp("_Name") else f"candidate_{i + 1}"))
        n_conf = mol.GetNumConformers()
        for c in range(max(n_conf, 1)):
            single = _one_conformer(mol, c) if n_conf > 1 else mol
            try:
                mf = build_fields(single, model.fieldset, name)
            except Exception:                 # noqa: BLE001 - a molecule the typing cannot take
                continue
            s = scorer.score(mf, name=name, conformer=c)
            key = name
            if key not in best or getattr(s, rank_by) > getattr(best[key], rank_by):
                best[key] = s
    out = list(best.values())
    out.sort(key=lambda s: -getattr(s, rank_by))
    return out


def _one_conformer(mol, c: int):
    from rdkit import Chem

    one = Chem.Mol(mol)
    one.RemoveAllConformers()
    one.AddConformer(mol.GetConformer(c), assignId=True)
    return one
