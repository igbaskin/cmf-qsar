"""Conformational ensembles: the fourth dimension of the CMF approach.

A 3D-QSAR model represents every molecule by one conformation, and the result depends on which
one was chosen.  4D-QSAR (Hopfinger et al., J. Am. Chem. Soc. 119 (1997) 10509) replaces that
choice with an ensemble and averages over it, so that the model no longer rests on a single
guessed pose.

In the CMF framework the averaging is exact and costs no new mathematics.  A field is a sum of
atom-centred Gaussians and the kernel is bilinear in the fields, so the mean field of an ensemble

    X_f(r) = sum_k w_k X_f^(k)(r)

is itself a field, and the kernel of two ensembles is the weighted mean of the kernels of their
conformation pairs,

    K(A, B) = sum_k sum_l w_k^A w_l^B K(X_A^(k), X_B^(l)),

which is the mean-map kernel of the two conformational distributions and is positive definite by
construction.  The model stays linear in that mean field, so co-fields, the decomposition rules,
the Shapley machinery and the complementarity scores all continue to hold; contributions simply
arrive per atom per conformation and are summed back onto the atoms with
:meth:`cmfqsar.fields.MoleculeFields.per_atom`.

This module builds the ensembles.  The field set decides whether they are used: ``conformers`` is
``"single"`` by default, which is the behaviour of the papers and of every earlier version.

What to watch out for: the kernel cost grows with the square of the number of conformations kept,
and the alignment tools act on the first conformation, so conformations are generated after a
series has been aligned and are then superimposed on that first one.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from .fields import FieldSet, conformer_weights

ENERGY_PROP = "conformer_energies"


def set_conformer_energies(mol: Chem.Mol, energies: Sequence[float],
                           prop: str = ENERGY_PROP) -> None:
    """Store per-conformer energies in kcal/mol on the molecule, for Boltzmann weighting."""
    mol.SetProp(prop, ",".join(f"{float(e):.4f}" for e in energies))


def get_conformer_energies(mol: Chem.Mol, prop: str = ENERGY_PROP) -> Optional[np.ndarray]:
    if not mol.HasProp(prop):
        return None
    try:
        e = np.array([float(x) for x in mol.GetProp(prop).replace(";", ",").split(",")], float)
    except ValueError:
        return None
    return e if len(e) == mol.GetNumConformers() else None


def generate_conformers(mol: Chem.Mol, n: int = 10, seed: int = 0, prune_rmsd: float = 0.5,
                        optimise: bool = True, align: bool = True, max_iters: int = 200,
                        keep_input: bool = True) -> Chem.Mol:
    """Return a copy of `mol` carrying up to `n` conformations, with their energies stored.

    The input conformation is kept as the first one by default, so a series that has been
    aligned stays where it was and the ensemble is built around that pose: the other
    conformations are superimposed on it by heavy-atom root-mean-square fit.  Energies come from
    MMFF94s where it can be set up and from UFF otherwise, which is enough for Boltzmann weights.
    """
    work = Chem.Mol(mol)
    first = None
    if keep_input and work.GetNumConformers():
        first = Chem.Conformer(work.GetConformer())
    params = AllChem.ETKDGv3()
    # a seed of exactly zero makes RDKit return the same geometry for every conformation, which
    # would silently collapse the ensemble, so zero is mapped to a fixed non-zero seed and the
    # result stays reproducible
    params.randomSeed = int(seed) if int(seed) != 0 else 1000003
    params.useSmallRingTorsions = True
    # pruning inside the embedding is brutal, turning six requested conformations into one on an
    # ordinary flexible molecule, so it is done afterwards on optimised geometries instead
    params.pruneRmsThresh = -1.0
    work.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(work, numConfs=int(max(n, 1)) * 2, params=params)
    if work.GetNumConformers() == 0:            # embedding can fail on strained inputs
        work = Chem.Mol(mol)
        if work.GetNumConformers() == 0:
            raise RuntimeError(f"no conformation could be generated for {_name(mol)}")
        return work
    energies = _optimise(work, max_iters) if optimise else np.zeros(work.GetNumConformers())
    if prune_rmsd and prune_rmsd > 0 and work.GetNumConformers() > 1:
        work, energies = _prune_by_rmsd(work, energies, float(prune_rmsd), int(max(n, 1)))
    elif work.GetNumConformers() > n:
        keep = sorted(np.argsort(energies)[:int(max(n, 1))])
        work, energies = _reorder(work, keep, energies)
    if first is not None:
        # put the input conformation back as conformer 0 and drop the worst one if we are over n
        work.AddConformer(first, assignId=True)
        energies = np.append(energies, _energy_of(work, work.GetNumConformers() - 1))
        order = [work.GetNumConformers() - 1] + list(range(work.GetNumConformers() - 1))
        work, energies = _reorder(work, order, energies)
        if work.GetNumConformers() > n:
            keep = [0] + list(np.argsort(energies[1:])[: n - 1] + 1)
            work, energies = _reorder(work, sorted(keep), energies)
    if align and work.GetNumConformers() > 1:
        align_conformers(work)
    set_conformer_energies(work, energies)
    for prop in mol.GetPropNames():
        if prop != ENERGY_PROP:
            work.SetProp(prop, mol.GetProp(prop))
    if mol.HasProp("_Name"):
        work.SetProp("_Name", mol.GetProp("_Name"))
    return work


def _name(mol: Chem.Mol) -> str:
    return mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule"


def _optimise(mol: Chem.Mol, max_iters: int) -> np.ndarray:
    try:
        props = AllChem.MMFFGetMoleculeProperties(Chem.Mol(mol), mmffVariant="MMFF94s")
    except Exception:                            # noqa: BLE001
        props = None
    try:
        if props is not None:
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, mmffVariant="MMFF94s")
        else:
            res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        return np.array([e for _, e in res], float)
    except Exception:                            # noqa: BLE001 - geometry stays, energies do not
        return np.zeros(mol.GetNumConformers())


def _prune_by_rmsd(mol: Chem.Mol, energies: np.ndarray, threshold: float,
                   keep_max: int) -> Tuple[Chem.Mol, np.ndarray]:
    """Keep the lowest conformations that differ from each other by more than `threshold`.

    The distance is the best-fit root-mean-square deviation over heavy atoms, computed after
    superposition, which is the only comparison that means anything for freshly embedded
    geometries in arbitrary orientations.
    """
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1] or None
    ids = [c.GetId() for c in mol.GetConformers()]
    kept: List[int] = []
    for pos in np.argsort(energies):
        ok = True
        for other in kept:
            try:
                rms = AllChem.GetConformerRMS(mol, ids[int(pos)], ids[other], atomIds=heavy,
                                              prealigned=False)
            except Exception:                    # noqa: BLE001 - keep it rather than lose it
                rms = threshold + 1.0
            if rms < threshold:
                ok = False
                break
        if ok:
            kept.append(int(pos))
        if len(kept) >= keep_max:
            break
    return _reorder(mol, sorted(kept), energies)


def _energy_of(mol: Chem.Mol, conf_id: int) -> float:
    try:
        props = AllChem.MMFFGetMoleculeProperties(Chem.Mol(mol), mmffVariant="MMFF94s")
        if props is None:
            return 0.0
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(mol.GetConformers()[conf_id].GetId()))
        return float(ff.CalcEnergy())
    except Exception:                            # noqa: BLE001
        return 0.0


def _reorder(mol: Chem.Mol, order: Sequence[int], energies: np.ndarray) -> Tuple[Chem.Mol, np.ndarray]:
    confs = [Chem.Conformer(c) for c in mol.GetConformers()]
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    for i in order:
        out.AddConformer(confs[i], assignId=True)
    return out, energies[list(order)]


def align_conformers(mol: Chem.Mol, atom_ids: Optional[Sequence[int]] = None) -> List[float]:
    """Superimpose every conformation of a molecule on its first one; returns the RMSD values."""
    if atom_ids is None:
        atom_ids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1] or None
    rms: List[float] = []
    rdMolAlign.AlignMolConformers(mol, atomIds=atom_ids, RMSlist=rms)
    return [float(r) for r in rms]


def generate_for_dataset(mols: Sequence[Chem.Mol], n: int = 10, seed: int = 0,
                         prune_rmsd: float = 0.5, optimise: bool = True,
                         progress=None, cancel=None) -> Tuple[List[Chem.Mol], List[str]]:
    """Conformations for a whole series; returns the molecules and a line of log per failure."""
    out, failures = [], []
    for i, m in enumerate(mols):
        if cancel is not None and cancel():
            break
        if progress is not None:
            progress(f"Conformations {i + 1}/{len(mols)}", i / max(len(mols), 1))
        try:
            out.append(generate_conformers(m, n=n, seed=seed, prune_rmsd=prune_rmsd,
                                           optimise=optimise))
        except Exception as exc:                 # noqa: BLE001 - reported, molecule kept as it was
            failures.append(f"{_name(m)}: {exc}")
            out.append(Chem.Mol(m))
    return out, failures


def group_conformers(mols: Sequence[Chem.Mol], names: Sequence[str],
                     by: str = "name") -> Tuple[List[Chem.Mol], List[str], List[int]]:
    """Collapse the records of a multi-conformer file into one molecule per compound.

    ``by="name"`` groups records that share a title, which is how most conformer generators write
    their output; any other value is read as the name of a property to group by.  The records of
    a group must have the same atom order, which is the case when they come from one generator.
    Returns the grouped molecules, their names, and the number of conformations in each.
    """
    order: List[str] = []
    groups: Dict[str, List[int]] = {}
    for i, (m, n) in enumerate(zip(mols, names)):
        if m is None:
            continue
        key = n if by == "name" else (m.GetProp(by) if m.HasProp(by) else n)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    out, out_names, counts = [], [], []
    for key in order:
        idx = groups[key]
        base = Chem.Mol(mols[idx[0]])
        n_atoms = base.GetNumAtoms()
        for j in idx[1:]:
            other = mols[j]
            if other.GetNumAtoms() != n_atoms or other.GetNumConformers() == 0:
                continue
            for c in other.GetConformers():
                base.AddConformer(Chem.Conformer(c), assignId=True)
        out.append(base)
        out_names.append(key)
        counts.append(base.GetNumConformers())
    return out, out_names, counts


def ensemble_summary(mols: Sequence[Chem.Mol], fieldset: FieldSet) -> Dict[str, float]:
    """How many conformations a series carries and how concentrated their weights are.

    The effective number of conformations is the exponential of the entropy of the weights: it is
    the number of conformations that actually count, which with Boltzmann weights and a dominant
    minimum can be far below the number present.
    """
    k = [m.GetNumConformers() for m in mols if m is not None]
    if not k:
        return {}
    eff = []
    for m in mols:
        if m is None:
            continue
        w = conformer_weights(m, fieldset)
        w = w[w > 0]
        eff.append(float(np.exp(-(w * np.log(w)).sum())) if len(w) else 1.0)
    return {"molecules": float(len(k)), "conformations_total": float(sum(k)),
            "conformations_mean": float(np.mean(k)), "conformations_max": float(max(k)),
            "effective_mean": float(np.mean(eff)),
            "kernel_cost_factor": float(np.mean(k) ** 2)}
