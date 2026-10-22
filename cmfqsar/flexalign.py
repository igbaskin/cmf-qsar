"""
Flexible alignment by confidence: the rigid-body search of 2019, with the torsions.

:mod:`cmfqsar.varalign` places a molecule by minimising the posterior variance of its predicted
activity -- the proposal of Baskin & Zhokhova, Future Med. Chem. (2019) -- but it moves the
molecule as a rigid body, so a flexible ligand can only be placed as well as its one conformation
allows.  This module adds the rotatable bonds to the same search, which is that proposal in full:
the pose and the conformation are chosen together, by confidence.

    minimise   sigma^2(X(p))  +  penalty(strain)  +  penalty(displacement),
    p = (rotation vector, translation, torsion angles)

**Why there is a strain term.**  Confidence alone will fold a molecule into whatever shape the
training set finds familiar, and nothing in the variance knows that a torsion has a barrier.  The
prior is the force field: the conformation may rise freely to `energy_slack` kcal/mol above the
one it started in, and is penalised quadratically beyond that.  One-sided on purpose -- a reward
for going *below* the starting energy would turn the aligner into a conformer optimiser, which is
not what it is for.

**The gradient is analytic throughout.**  The kernel gradient with respect to atom positions is
already in :func:`cmfqsar.kernels.kernel_gradient` (probe fields included, as of this work); the
chain rule to a torsion is the generator of a rotation about the bond axis,

    d x_i / d theta_j  =  a_j x (x_i - p_j)      for the atoms that bond j moves,

evaluated in the current configuration, which is exact when the bonds are applied from the root
outwards: a rotation about bond j carries every bond inside its moving set with it.  The force
field supplies Cartesian gradients for the strain term, and they go through the same Jacobian.

**What the measurement said before this was written.**  Under the C1 protocol
(:mod:`cmfqsar.poses`), the rigid confidence search from a scrambled start reached only 1.35 A
median over sixteen FEP sets and predicted at R2pred -0.219, *worse than leaving the poses
scrambled*; from a substructure frame it reached 0.11 A and the crystal-pose R2pred exactly.
Adding torsions enlarges the space in which the objective was shown to find confident but wrong
answers, so the defaults here are the cautious ones: start from a chemically motivated pose, keep
the displacement guard, keep the strain prior.  ``benchmarks/run_poses.py`` judges it the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .kernels import kernel_gradient, so3_right_jacobian
from .varalign import VarianceAligner, VarianceAlignResult

#: bonds that may turn: RDKit's strict definition, which already excludes amides and the like
ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])"
                               "([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@"
                               "[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]"
                               "-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C"
                               "([CH3])([CH3])[CH3])]")


@dataclass
class Torsion:
    """One rotatable bond: the axis, and the atoms that turn with it."""
    pivot: int                  # the axis atom that stays
    tip: int                    # the axis atom on the moving side
    moving: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def __len__(self) -> int:
        return int(len(self.moving))


def _moving_side(mol: Chem.Mol, i: int, j: int) -> Optional[np.ndarray]:
    """The atoms reachable from `j` without crossing the bond i-j; None if the bond is in a ring."""
    bond = mol.GetBondBetweenAtoms(i, j)
    if bond is None or bond.IsInRing():
        return None
    seen, stack = {j}, [j]
    while stack:
        a = stack.pop()
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            k = nb.GetIdx()
            if k == i and a == j:
                continue
            if k not in seen:
                seen.add(k)
                stack.append(k)
    if i in seen:                       # a ring by another path
        return None
    return np.array(sorted(seen - {j}) + [j], dtype=int)


def rotatable_torsions(mol: Chem.Mol, max_torsions: int = 12) -> List[Torsion]:
    """The rotatable bonds of a molecule, ordered from the root outwards.

    The smaller side of each bond is the one that moves, so the molecule's bulk stays where the
    rigid-body parameters put it.  Ordering by the size of the moving set, largest first, is the
    root-outward order the chain rule needs: a rotation then carries every later bond with it.
    """
    out: List[Torsion] = []
    for match in mol.GetSubstructMatches(ROTATABLE):
        i, j = int(match[0]), int(match[1])
        side_j = _moving_side(mol, i, j)
        side_i = _moving_side(mol, j, i)
        if side_j is None or side_i is None:
            continue
        if len(side_j) <= len(side_i):
            out.append(Torsion(pivot=i, tip=j, moving=side_j))
        else:
            out.append(Torsion(pivot=j, tip=i, moving=side_i))
    out.sort(key=len, reverse=True)
    return out[:int(max_torsions)]


def _rotate(X: np.ndarray, axis: np.ndarray, origin: np.ndarray, theta: float,
            rows: np.ndarray) -> None:
    """Turn `rows` of X about the axis through `origin` by `theta`, in place."""
    a = axis / max(float(np.linalg.norm(axis)), 1e-12)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    X[rows] = (X[rows] - origin) @ R.T + origin


def apply_torsions(X0: np.ndarray, torsions: Sequence[Torsion],
                   thetas: Sequence[float]) -> np.ndarray:
    """X0 with every torsion turned by its angle, applied from the root outwards."""
    X = np.array(X0, dtype=float)
    for t, th in zip(torsions, thetas):
        if abs(float(th)) < 1e-15 or not len(t):
            continue
        _rotate(X, X[t.tip] - X[t.pivot], X[t.pivot], float(th), t.moving)
    return X


def torsion_gradient(X: np.ndarray, torsions: Sequence[Torsion],
                     g: np.ndarray) -> np.ndarray:
    """dF/dtheta from dF/dX, by the chain rule over the bond axes at the current geometry."""
    out = np.zeros(len(torsions))
    for n, t in enumerate(torsions):
        if not len(t):
            continue
        a = X[t.tip] - X[t.pivot]
        a = a / max(float(np.linalg.norm(a)), 1e-12)
        v = np.cross(a, X[t.moving] - X[t.pivot])          # d x_i / d theta
        out[n] = float(np.einsum("ik,ik->", g[t.moving], v))
    return out


class _Strain:
    """The force field as a prior on the conformation: energy and Cartesian gradient."""

    def __init__(self, mol: Chem.Mol, slack: float = 10.0, weight: float = 1.0):
        from rdkit.Chem import AllChem

        self.ok = False
        self.slack, self.weight = float(slack), float(weight)
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                self.ff = AllChem.MMFFGetMoleculeForceField(Chem.Mol(mol), props)
                self.ok = self.ff is not None
        except Exception:                    # noqa: BLE001  -- a molecule MMFF cannot type
            self.ok = False
        self.e0 = self.energy(np.asarray(mol.GetConformer().GetPositions(), float)) if self.ok \
            else 0.0

    def energy(self, X: np.ndarray) -> float:
        if not self.ok:
            return 0.0
        self.ff.Initialize()
        pos = [float(v) for v in np.asarray(X, float).reshape(-1)]
        return float(self.ff.CalcEnergy(pos))

    def penalty(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """``w (E - E0 - slack)^2`` above the slack, zero below it, with its gradient."""
        if not self.ok or self.weight <= 0:
            return 0.0, np.zeros_like(X)
        pos = [float(v) for v in np.asarray(X, float).reshape(-1)]
        self.ff.Initialize()
        e = float(self.ff.CalcEnergy(pos))
        over = e - self.e0 - self.slack
        if over <= 0.0:
            return 0.0, np.zeros_like(X)
        grad = np.asarray(self.ff.CalcGrad(pos), float).reshape(X.shape)
        return self.weight * over ** 2, self.weight * 2.0 * over * grad


class FlexibleAligner(VarianceAligner):
    """The confidence search over rigid motion *and* the rotatable bonds.

    `mol_for` must return the RDKit molecule behind a :class:`~cmfqsar.fields.MoleculeFields`,
    since the torsions and the force field come from the bond graph, which the field
    representation does not carry.
    """

    def __init__(self, model, max_rmsd: Optional[float] = 2.0, penalty: float = 50.0,
                 energy_slack: float = 10.0, energy_weight: float = 1.0,
                 max_torsions: int = 12):
        super().__init__(model, max_rmsd=max_rmsd, penalty=penalty)
        self.energy_slack = float(energy_slack)
        self.energy_weight = float(energy_weight)
        self.max_torsions = int(max_torsions)

    # ------------------------------------------------------------------ objective
    def objective_flex(self, mol, rd: Chem.Mol, X0: np.ndarray, c: np.ndarray,
                       torsions: Sequence[Torsion], strain: _Strain, sp: np.ndarray,
                       kself: float, p: np.ndarray):
        n_t = len(torsions)
        Y = apply_torsions(X0, torsions, p[6:6 + n_t]) if n_t else X0
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        V = Y - c
        X = V @ R.T + c + p[3:6]
        k, scale = self._kvec(mol, X, sp)
        Ak = self.A @ k
        val = kself - float(k @ Ak)
        u = -2.0 * self.coef[:, None] * scale * Ak[None, :]
        g = kernel_gradient(X, mol.weights, mol.radii, self.model.train, self.specs, u,
                            ci=mol.probe_coeffs)
        if self.max_rmsd is not None:
            d = X - X0
            msd = float((d * d).sum() / len(X0))
            lim = self.max_rmsd ** 2
            if msd > lim:
                val += self.penalty * (msd - lim) ** 2
                g = g + self.penalty * 4.0 * (msd - lim) * d / len(X0)
        grad = np.empty(6 + n_t)
        h = g @ R
        grad[:3] = so3_right_jacobian(p[:3]).T @ np.cross(V, h).sum(axis=0)
        grad[3:6] = g.sum(axis=0)
        if n_t:
            # the strain is a function of the conformation only, so it enters through the
            # torsions and never through the rigid parameters
            e_val, e_grad = strain.penalty(Y)
            val += e_val
            grad[6:] = torsion_gradient(Y, torsions, h + e_grad)
        return val, grad

    def align_flexible(self, mol, rd: Chem.Mol, restarts: int = 2, seed: int = 0,
                       max_iter: int = 300, apply: bool = True,
                       torsion_start: float = 0.4) -> VarianceAlignResult:
        """Place `mol` by confidence, turning its rotatable bonds as well as moving it."""
        X0 = np.array(mol.coords, dtype=float)
        if rd.GetNumAtoms() != len(X0):
            raise ValueError("the molecule and its fields have different atom counts")
        torsions = rotatable_torsions(rd, self.max_torsions)
        strain = _Strain(rd, self.energy_slack, self.energy_weight)
        c = X0.mean(axis=0)
        sp, kself = self._self_terms(mol)
        n_t = len(torsions)
        counter = {"n": 0}

        def fun(p):
            counter["n"] += 1
            return self.objective_flex(mol, rd, X0, c, torsions, strain, sp, kself, p)

        starts = [np.zeros(6 + n_t)]
        rng = np.random.RandomState(seed)
        for _ in range(max(0, restarts)):
            starts.append(np.concatenate([
                Rotation.random(random_state=rng).as_rotvec() * 0.35,
                rng.normal(scale=0.8, size=3),
                rng.normal(scale=torsion_start, size=n_t)]))
        best = None
        for p0 in starts:
            res = minimize(fun, p0, jac=True, method="L-BFGS-B",
                           options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-9})
            if best is None or res.fun < best.fun:
                best = res
        Y = apply_torsions(X0, torsions, best.x[6:]) if n_t else X0
        R = Rotation.from_rotvec(best.x[:3]).as_matrix()
        Xn = (Y - c) @ R.T + c + best.x[3:6]
        k0, _ = self._kvec(mol, X0, sp)
        kn, _ = self._kvec(mol, Xn, sp)
        out = VarianceAlignResult(
            name=mol.name, coords=Xn,
            variance_before=float(kself - k0 @ self.A @ k0),
            variance_after=float(kself - kn @ self.A @ kn),
            prediction_before=float(k0 @ self.model.coef + self.model.bias),
            prediction_after=float(kn @ self.model.coef + self.model.bias),
            rmsd=float(np.sqrt(((Xn - X0) ** 2).sum(1).mean())), n_eval=counter["n"],
            limited=self.max_rmsd is not None
            and float(np.sqrt(((Xn - X0) ** 2).sum(1).mean())) > 0.995 * self.max_rmsd)
        if apply:
            mol.coords = Xn
        return out


def flexible_align(mols: Sequence, rdmols: Sequence[Chem.Mol], model,
                   max_rmsd: Optional[float] = 2.0, energy_slack: float = 10.0,
                   energy_weight: float = 1.0, restarts: int = 2, seed: int = 0,
                   max_torsions: int = 12, progress=None, cancel=None
                   ) -> List[VarianceAlignResult]:
    """Align every molecule in place by confidence over rigid motion and the rotatable bonds."""
    aligner = FlexibleAligner(model, max_rmsd=max_rmsd, energy_slack=energy_slack,
                              energy_weight=energy_weight, max_torsions=max_torsions)
    out: List[VarianceAlignResult] = []
    for i, (mf, rd) in enumerate(zip(mols, rdmols)):
        if cancel is not None and cancel():
            break
        out.append(aligner.align_flexible(mf, rd, restarts=restarts, seed=seed + i))
        if progress is not None:
            progress(i + 1, len(mols))
    return out
