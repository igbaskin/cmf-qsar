"""
Preparing a background (decoy) set for field-based virtual screening.

A decoy has to be placed in the same frame as the training actives before its fields can be
compared with theirs, and that placement decides its score: a molecule posed badly scores low for
the wrong reason and overstates the model's quality.  The safest choice is therefore to give every
decoy its *best* pose, i.e. to align it so that its field similarity to the training actives is as
large as it can be.  The evaluation is then conservative: whatever enrichment survives is not an
artefact of careless placement.

The criterion maximised here is a smooth maximum (log-sum-exp) of the cosine-normalised CMF kernel
between the decoy and each training molecule,

    S(x) = (1/beta) log sum_j exp(beta * Kn(x, M_j)),   Kn in [0, 1]

which tends to the similarity to the best-matching active as beta grows.  The search is over the
six rigid-body parameters, with analytic gradients and several random starting orientations,
because a molecule of a different chemotype has no reason to lie in the basin of its best pose.

Note what this does *not* do: it does not make a decoy a good ligand, and it cannot repair the fact
that a score computed after such an alignment measures the alignment and the model together.  See
the README for the choice of background sets (DUD-E, DEKOIS, LIT-PCBA, or the other example series).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .fields import AtomTable, FieldSet, MoleculeFields, build_fields
from .kernels import atom_kernel_block, kernel_gradient, so3_right_jacobian


@dataclass
class DecoyAlignResult:
    name: str
    coords: np.ndarray
    similarity_before: float
    similarity_after: float
    best_match: str = ""          # training molecule the decoy ends up most similar to
    n_eval: int = 0
    conformer: int = 0            # which conformation was chosen, 0 being the one from the file
    n_conformers: int = 1


class DecoyAligner:
    """Place molecules in the frame of a reference set by maximising their field similarity to it.

    Parameters
    ----------
    reference : AtomTable of the aligned molecules that define the frame (the training actives).
    fieldset  : the fields to compare with; cosine normalisation is applied internally, so the
        result does not depend on molecular size.
    beta : sharpness of the smooth maximum.  Large values align a decoy to its single best-matching
        reference molecule, small values to the ensemble as a whole.
    """

    def __init__(self, reference: AtomTable, fieldset: FieldSet, beta: float = 20.0):
        self.ref = reference
        self.fieldset = fieldset
        self.specs = fieldset.specs
        self.beta = float(beta)
        self.h = fieldset.mixing_vector()
        self.h = self.h / max(self.h.sum(), 1e-12)          # similarity in [0, 1]
        d = np.zeros((len(self.specs), reference.n_mol))
        for j in range(reference.n_mol):
            sub = reference.subset([j])
            d[:, j] = atom_kernel_block(sub.coords, sub.weights, sub.radii, sub, self.specs).sum(axis=1)[:, 0]
        self.ref_self = np.sqrt(np.maximum(d, 1e-300))

    # ------------------------------------------------------------------ similarity
    def _self_norm(self, mol: MoleculeFields) -> np.ndarray:
        one = AtomTable.from_molecules([mol])
        d = atom_kernel_block(mol.coords, mol.weights, mol.radii, one, self.specs).sum(axis=1)[:, 0]
        return 1.0 / np.sqrt(np.maximum(d, 1e-300))         # pose-invariant

    def similarities(self, mol: MoleculeFields, coords: np.ndarray, sp: Optional[np.ndarray] = None):
        """Normalised similarity of `mol` in the given pose to every reference molecule."""
        if sp is None:
            sp = self._self_norm(mol)
        kraw = atom_kernel_block(coords, mol.weights, mol.radii, self.ref, self.specs).sum(axis=1)
        kn = kraw * sp[:, None] / np.maximum(self.ref_self, 1e-300)
        return self.h @ kn, kn

    def objective(self, mol, X0, c, sp, p):
        """Negative smooth maximum of the similarity, and its gradient in the rigid-body parameters."""
        V = X0 - c
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        X = V @ R.T + c + p[3:]
        s, kn = self.similarities(mol, X, sp)
        m = float(np.max(s))
        e = np.exp(self.beta * (s - m))
        total = float(e.sum())
        value = m + np.log(max(total, 1e-300)) / self.beta    # smooth maximum (log-sum-exp)
        w = e / max(total, 1e-300)                            # softmax weights over reference molecules
        u = (self.h[:, None] * w[None, :]) * (sp[:, None] / np.maximum(self.ref_self, 1e-300))
        g = kernel_gradient(X, mol.weights, mol.radii, self.ref, self.specs, u)
        h_ = g @ R
        grad = np.empty(6)
        grad[:3] = so3_right_jacobian(p[:3]).T @ np.cross(V, h_).sum(axis=0)
        grad[3:] = g.sum(axis=0)
        return -value, -grad

    # ------------------------------------------------------------------ alignment
    def align(self, mol: MoleculeFields, restarts: int = 6, seed: int = 0, max_iter: int = 200,
              apply: bool = True) -> DecoyAlignResult:
        """Give `mol` the pose of highest field similarity to the reference set."""
        X0 = np.array(mol.coords, dtype=float)
        centre = self.ref.coords.mean(axis=0)
        X0 = X0 - X0.mean(axis=0) + centre                   # start from a centroid superposition
        c = X0.mean(axis=0)
        sp = self._self_norm(mol)
        counter = {"n": 0}

        def fun(p):
            counter["n"] += 1
            return self.objective(mol, X0, c, sp, p)

        rng = np.random.RandomState(seed)
        starts = [np.zeros(6)]
        for _ in range(max(0, restarts)):
            starts.append(np.concatenate([Rotation.random(random_state=rng).as_rotvec(),
                                          rng.normal(scale=1.0, size=3)]))
        best = None
        for p0 in starts:
            res = minimize(fun, p0, jac=True, method="L-BFGS-B",
                           options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        R = Rotation.from_rotvec(best.x[:3]).as_matrix()
        Xn = (X0 - c) @ R.T + c + best.x[3:]
        s0, _ = self.similarities(mol, X0, sp)
        sn, _ = self.similarities(mol, Xn, sp)
        j = int(np.argmax(sn))
        out = DecoyAlignResult(name=mol.name, coords=Xn,
                               similarity_before=float(np.max(s0)), similarity_after=float(np.max(sn)),
                               best_match=self.ref.names[j] if j < len(self.ref.names) else "",
                               n_eval=counter["n"])
        if apply:
            mol.coords = Xn
        return out


def align_decoys(mols: Sequence[MoleculeFields], reference: AtomTable, fieldset: FieldSet,
                 restarts: int = 6, beta: float = 20.0, seed: int = 0,
                 progress: Optional[Callable[[int, int], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None) -> List[DecoyAlignResult]:
    """Align every molecule of `mols` into the frame of `reference`, in place."""
    aligner = DecoyAligner(reference, fieldset, beta=beta)
    out: List[DecoyAlignResult] = []
    for i, mol in enumerate(mols):
        if cancel is not None and cancel():
            break
        out.append(aligner.align(mol, restarts=restarts, seed=seed + i))
        if progress is not None:
            progress(i + 1, len(mols))
    return out


def align_decoys_flexible(mols: Sequence["Chem.Mol"], names: Sequence[str],
                          reference: AtomTable, fieldset: FieldSet, n_conformers: int = 8,
                          restarts: int = 2, beta: float = 20.0, seed: int = 0,
                          prune_rmsd: float = 0.5,
                          progress: Optional[Callable[[int, int], None]] = None,
                          cancel: Optional[Callable[[], bool]] = None):
    """Pose every decoy from several conformations and keep the one that scores highest.

    The rigid aligner above can only turn and move the conformation a decoy arrives with, so a
    molecule that would resemble the actives in some other conformation never has the opportunity,
    and its low score may be an accident of the file it came from.  Here the conformations are
    generated (the input pose is kept as the first, so this can never do worse than the rigid
    alignment), each is given its best rigid pose, and the best of those is the decoy's pose.

    The cost is the product: `n_conformers` times `1 + restarts` optimisations per decoy, which
    is why the restarts default lower here than in :func:`align_decoys`.

    Returns (fields, results): the chosen :class:`MoleculeFields` of every decoy, already in the
    frame of the reference, and one :class:`DecoyAlignResult` each, saying which conformation was
    chosen.
    """
    from .conformers import generate_conformers, single_conformer
    aligner = DecoyAligner(reference, fieldset, beta=beta)
    fields: List[MoleculeFields] = []
    out: List[DecoyAlignResult] = []
    for i, (mol, name) in enumerate(zip(mols, names)):
        if cancel is not None and cancel():
            break
        candidates = [mol]
        if n_conformers > 1:
            try:
                multi = generate_conformers(mol, n=int(n_conformers), seed=seed + i + 1,
                                            prune_rmsd=prune_rmsd, align=False, keep_input=True)
                candidates = [single_conformer(multi, k)
                              for k in range(multi.GetNumConformers())]
            except Exception:                       # noqa: BLE001 - keep the pose from the file
                candidates = [mol]
        best_f, best_r = None, None
        for k, cand in enumerate(candidates):
            f = build_fields(cand, fieldset, str(name))
            r = aligner.align(f, restarts=restarts, seed=seed + 17 * i + k)
            r.conformer, r.n_conformers = k, len(candidates)
            if best_r is None or r.similarity_after > best_r.similarity_after:
                best_f, best_r = f, r
        fields.append(best_f)
        out.append(best_r)
        if progress is not None:
            progress(i + 1, len(mols))
    return fields, out
