"""Property-matched decoy selection, following DUD-E, from the molecules shipped here.

A background of ligands of unrelated targets picked at random is easy: they differ from the
actives in size, shape and charge, so a screening model separates them before any field is
involved (measured: every one-class description reached AUC 0.97 to 1.00 on such a set).  The
method that makes a background hard is the one used by DUD-E: for every active, choose molecules
that *match its physical properties* -- molecular weight, lipophilicity, hydrogen-bond donors and
acceptors, rotatable bonds, net charge -- while being *topologically unlike* it, so that nothing
but the binding hypothesis can distinguish them.

Here the candidate pool is every other example series shipped with the program, which is small by
DUD-E standards but needs no download and no licence.  Decoys remain presumed inactive, not
measured inactive: they are ligands of other targets, which is the usual and acknowledged
weakness of any decoy set.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

#: the properties a decoy has to match, and how each is read off a molecule
PROPERTIES = (
    ("molecular weight", Descriptors.MolWt),
    ("logP", Crippen.MolLogP),
    ("H-bond donors", rdMolDescriptors.CalcNumHBD),
    ("H-bond acceptors", rdMolDescriptors.CalcNumHBA),
    ("rotatable bonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("net charge", lambda m: float(Chem.GetFormalCharge(m))),
)


def profile(mols: Sequence[Chem.Mol]) -> np.ndarray:
    """The property matrix of a set of molecules, one row each."""
    out = np.zeros((len(mols), len(PROPERTIES)))
    for i, m in enumerate(mols):
        for j, (_, fn) in enumerate(PROPERTIES):
            try:
                out[i, j] = float(fn(m))
            except Exception:                              # noqa: BLE001 - odd valences
                out[i, j] = np.nan
    return np.nan_to_num(out, nan=0.0)


def _fingerprints(mols: Sequence[Chem.Mol]):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(m) for m in mols]


def select(actives: Sequence[Chem.Mol], pool: Sequence[Chem.Mol], n: int,
           max_similarity: float = 0.35, per_active: int = 50,
           seed: int = 0) -> Tuple[List[int], dict]:
    """Indices into `pool` of `n` decoys matched to `actives`, with a report.

    Every active in turn takes the unused pool molecule that is closest to it in the scaled
    property space, among those whose ECFP4 Tanimoto similarity to *every* active is below
    `max_similarity`.  Going active by active rather than matching the pooled distribution keeps
    the background spread over the actives instead of concentrating around their centre.
    """
    if not len(pool):
        return [], {"reason": "empty pool"}
    fp_act, fp_pool = _fingerprints(actives), _fingerprints(pool)
    worst = np.zeros(len(pool))
    for j, fp in enumerate(fp_pool):
        worst[j] = max(DataStructs.BulkTanimotoSimilarity(fp, list(fp_act)))
    allowed = np.flatnonzero(worst < max_similarity)
    report = {"pool": len(pool), "dissimilar": int(len(allowed)),
              "max_similarity": max_similarity}
    if len(allowed) == 0:
        return [], report
    P_act, P_pool = profile(actives), profile(pool)
    scale = np.maximum(np.subtract(*np.percentile(np.vstack([P_act, P_pool]), [75, 25], axis=0)),
                       1e-6)
    taken: List[int] = []
    used = np.zeros(len(allowed), bool)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(actives))
    cost = np.zeros((len(actives), len(allowed)))
    for a in range(len(actives)):
        cost[a] = (np.abs(P_pool[allowed] - P_act[a]) / scale).mean(axis=1)
    rounds = max(int(per_active), int(np.ceil(n / max(len(actives), 1))) + 1)
    for _ in range(rounds):
        progress = False
        for a in order:
            if len(taken) >= n or used.all():
                break
            c = np.where(used, np.inf, cost[a])
            k = int(np.argmin(c))
            if not np.isfinite(c[k]):
                continue
            used[k] = True
            taken.append(int(allowed[k]))
            progress = True
        if len(taken) >= n or not progress:
            break
    P_taken = P_pool[taken] if taken else np.zeros((0, len(PROPERTIES)))
    report["selected"] = len(taken)
    report["property_gap"] = {
        name: (float(P_act[:, j].mean()), float(P_taken[:, j].mean()) if len(taken) else float("nan"))
        for j, (name, _) in enumerate(PROPERTIES)}
    report["max_similarity_taken"] = float(worst[taken].max()) if taken else float("nan")
    return taken, report
