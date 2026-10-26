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

What to be careful about: the kernel cost grows with the square of the number of conformations kept,
and the conformations have to be put somewhere.  That last point is not a detail.  The CMF kernel
compares molecules with each other, so what matters is not where a conformation sits relative to
its own molecule but where it sits relative to the rest of the series.  Superimposing the
conformations of a molecule on that molecule's own first pose - all that
``rdMolAlign.AlignMolConformers`` can do - leaves them in arbitrary orientations with respect to
every other molecule, and in a series that its authors aligned by hand that blurs the very
arrangement the model depends on.  :data:`CONFORMER_ALIGNMENTS` lists what is offered instead, all
of it relative to one common reference molecule of the series:

``self``
    superimpose the conformations on the molecule's own first pose over all heavy atoms (what the
    first version of this module did, kept as the default so that nothing changes silently).
``core``
    superimpose them on the molecule's own first pose over the atoms of the maximum common
    substructure with the reference only, so the scaffold stays exactly where the published
    alignment put it and only the flexible part moves.
``reference``
    superimpose each conformation on the reference molecule itself through that same maximum
    common substructure, which forces every conformation of the series into one frame.
``field``
    as ``reference``, followed by the rigid-body refinement of :func:`cmfqsar.align.field_align`,
    which maximises the normalised CMF kernel with the reference - the alignment criterion of
    Baskin & Zhokhova (2013) applied to each conformation.

In every case the molecule's first conformation is left exactly where it was, so a series that
arrives aligned keeps its alignment and only the generated conformations are placed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from .fields import FieldSet, conformer_weights

ENERGY_PROP = "conformer_energies"

#: how the generated conformations are placed relative to the rest of the series
CONFORMER_ALIGNMENTS = ("self", "core", "reference", "field")


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
                        keep_input: bool = True, align_to: str = "self",
                        reference: Optional[Chem.Mol] = None,
                        fieldset: Optional[FieldSet] = None,
                        atom_map: Optional[Sequence[Tuple[int, int]]] = None,
                        window_rmsd: float = 0.0) -> Chem.Mol:
    """Return a copy of `mol` carrying up to `n` conformations, with their energies stored.

    The input conformation is kept as the first one by default, so a series that has been
    aligned stays where it was and the ensemble is built around that pose.  `align_to` decides
    where the other conformations go - see :data:`CONFORMER_ALIGNMENTS` and
    :func:`align_conformers_to_reference`; ``"self"``, the default, superimposes them on the
    molecule's own first pose, which is what the earlier versions did.  Energies come from
    MMFF94s where it can be set up and from UFF otherwise, which is enough for Boltzmann weights.

    `window_rmsd`, when positive, keeps only the conformations that lie within that heavy-atom
    root-mean-square distance of the input pose, measured where they are rather than after a
    best fit, so the ensemble covers the local flexibility around the pose the series was aligned
    to instead of the whole conformational surface.  A wider pool is embedded when a window is
    asked for, since most of it will be discarded.
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
    # pruning inside the embedding is very aggressive, turning six requested conformations into one
    # on an ordinary flexible molecule, so it is done afterwards on optimised geometries instead
    params.pruneRmsThresh = -1.0
    work.RemoveAllConformers()
    window = float(window_rmsd) if window_rmsd and window_rmsd > 0 else 0.0
    factor, pool = (2, int(max(n, 1))) if not window else (6, int(max(n, 1)) * 4)
    AllChem.EmbedMultipleConfs(work, numConfs=int(max(n, 1)) * factor, params=params)
    if work.GetNumConformers() == 0:            # embedding can fail on strained inputs
        work = Chem.Mol(mol)
        if work.GetNumConformers() == 0:
            raise RuntimeError(f"no conformation could be generated for {_name(mol)}")
        return work
    energies = _optimise(work, max_iters) if optimise else np.zeros(work.GetNumConformers())
    if prune_rmsd and prune_rmsd > 0 and work.GetNumConformers() > 1:
        work, energies = _prune_by_rmsd(work, energies, float(prune_rmsd), pool)
    elif work.GetNumConformers() > pool:
        keep = sorted(np.argsort(energies)[:pool])
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
        align_conformers_to_reference(work, reference, mode=align_to, fieldset=fieldset,
                                      atom_map=atom_map)
    if window and work.GetNumConformers() > 1:
        # after placement, because what matters is how far a conformation lies from the pose
        # the series was aligned to, not how similar its shape is
        work, energies = _window_filter(work, energies, window, int(max(n, 1)))
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


def conformer_displacements(mol: Chem.Mol, reference_conf: int = 0) -> np.ndarray:
    """Heavy-atom RMSD of every conformation from `reference_conf`, measured in place.

    No superposition: two conformations that have the same shape but sit in different places are
    far apart here, which is the distance that matters once a series has been aligned.
    """
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy or mol.GetNumConformers() == 0:
        return np.zeros(mol.GetNumConformers())
    ref = np.array(mol.GetConformer(mol.GetConformers()[reference_conf].GetId()).GetPositions())[heavy]
    out = []
    for c in mol.GetConformers():
        d = np.array(c.GetPositions())[heavy] - ref
        out.append(float(np.sqrt((d * d).sum(axis=1).mean())))
    return np.array(out)


def _window_filter(mol: Chem.Mol, energies: np.ndarray, window: float,
                   keep_max: int) -> Tuple[Chem.Mol, np.ndarray]:
    """Keep the conformations within `window` of the first one, the first one always included."""
    d = conformer_displacements(mol)
    inside = [i for i in range(mol.GetNumConformers()) if i == 0 or d[i] <= window]
    if len(inside) > keep_max:                   # the lowest in energy among those inside
        rest = sorted(inside[1:], key=lambda i: energies[i])[: max(keep_max - 1, 0)]
        inside = [0] + sorted(rest)
    return _reorder(mol, sorted(inside), energies)


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


def single_conformer(mol: Chem.Mol, index: int = 0) -> Chem.Mol:
    """A copy of `mol` carrying only its conformation number `index`, as conformer 0."""
    keep = Chem.Conformer(mol.GetConformers()[int(index)])
    keep.SetId(0)
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    out.AddConformer(keep, assignId=True)
    return out


def _put_coords(src: Chem.Mol, dst: Chem.Mol, conf_id: int) -> None:
    """Copy the coordinates of `src`'s only conformation into conformation `conf_id` of `dst`."""
    pos = src.GetConformer().GetPositions()
    conf = dst.GetConformer(int(conf_id))
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])


def align_conformers_to_reference(mol: Chem.Mol, ref: Optional[Chem.Mol] = None,
                                  mode: str = "self", fieldset: Optional[FieldSet] = None,
                                  atom_map: Optional[Sequence[Tuple[int, int]]] = None,
                                  timeout: int = 10, field_restarts: int = 0,
                                  field_max_iter: int = 150) -> str:
    """Place the generated conformations of `mol` in the frame of the series, in place.

    `mode` is one of :data:`CONFORMER_ALIGNMENTS` and is explained in the module docstring.  Every
    mode but ``self`` needs a reference molecule of the series, in the pose the series is aligned
    to, and ``field`` needs a field set as well.  Conformation 0 is never moved.  `atom_map` is
    the maximum common substructure correspondence with the reference, computed here when it is
    not supplied; it depends on the molecule and not on its conformations, so it is worth passing
    in when the same molecule is aligned more than once.  Returns one line of log.
    """
    if mode not in CONFORMER_ALIGNMENTS:
        raise ValueError(f"unknown conformer alignment {mode!r}; expected one of "
                         + ", ".join(CONFORMER_ALIGNMENTS))
    if mol.GetNumConformers() < 2:
        return "one conformation, nothing to align"
    if mode == "self" or ref is None:
        rms = align_conformers(mol)
        return f"on the input pose, RMSD {np.mean(rms):.2f} A" if rms else "on the input pose"
    from .align import field_align, mcs_atom_map          # imported late: scipy is heavy
    if atom_map is None:
        try:
            atom_map = mcs_atom_map(mol, ref, timeout=timeout)
        except Exception as exc:                 # noqa: BLE001 - fall back, never lose the molecule
            align_conformers(mol)
            return f"MCS failed ({exc}); superimposed on the input pose"
    if len(atom_map) < 3:
        align_conformers(mol)
        return "MCS below 3 atoms; superimposed on the input pose"
    ids = [c.GetId() for c in mol.GetConformers()]
    if mode == "core":
        anchor = single_conformer(mol, 0)
        core = [(int(i), int(i)) for i, _ in atom_map]
        rms = [float(rdMolAlign.AlignMol(mol, anchor, prbCid=int(cid), refCid=0, atomMap=core))
               for cid in ids[1:]]
        return f"core of {len(core)} atoms on the input pose, RMSD {np.mean(rms):.2f} A"
    ref_id = int(ref.GetConformers()[0].GetId())
    amap = [(int(a), int(b)) for a, b in atom_map]
    rms = [float(rdMolAlign.AlignMol(mol, ref, prbCid=int(cid), refCid=ref_id, atomMap=amap))
           for cid in ids[1:]]
    msg = f"MCS {len(amap)} atoms on the reference, RMSD {np.mean(rms):.2f} A"
    if mode == "field":
        if fieldset is None:
            return msg + "; no field set, no field refinement"
        fs = fieldset.copy()
        fs.conformers = "single"                 # one conformation at a time, whatever the model does
        ref1 = ref if ref.GetNumConformers() == 1 else single_conformer(ref, 0)
        gains = []
        for k, cid in enumerate(ids[1:], start=1):
            probe = single_conformer(mol, k)
            try:
                before, after = field_align(probe, ref1, fs, restarts=field_restarts,
                                            max_iter=field_max_iter)
            except Exception:                    # noqa: BLE001 - keep the MCS pose
                continue
            if after > before:
                _put_coords(probe, mol, cid)
                gains.append(after - before)
        if gains:
            msg += f"; field similarity +{np.mean(gains):.3f} on {len(gains)} conformations"
    return msg


def dataset_reference(mols: Sequence[Chem.Mol], index: Optional[int] = None) -> Chem.Mol:
    """The molecule the conformations of a series are aligned to, as a one-conformation copy.

    `index` picks it; without one the largest molecule is taken, its maximum common substructure
    with the others being the most likely to hold the scaffold of the series.  The copy is made
    before any conformations are generated, so the reference keeps the pose the series arrived in.
    """
    good = [i for i, m in enumerate(mols) if m is not None and m.GetNumConformers()]
    if not good:
        raise ValueError("no molecule with a conformation to align the conformations to")
    if index is None:
        index = max(good, key=lambda i: mols[i].GetNumHeavyAtoms())
    elif int(index) not in good:
        raise ValueError(f"molecule {index} has no conformation to align to")
    return single_conformer(mols[int(index)], 0)


def generate_for_dataset(mols: Sequence[Chem.Mol], n: int = 10, seed: int = 0,
                         prune_rmsd: float = 0.5, optimise: bool = True,
                         align_to: str = "self", reference_index: Optional[int] = None,
                         fieldset: Optional[FieldSet] = None, window_rmsd: float = 0.0,
                         progress=None, cancel=None) -> Tuple[List[Chem.Mol], List[str]]:
    """Conformations for a whole series; returns the molecules and a line of log per failure.

    `align_to` other than ``"self"`` places the conformations of every molecule relative to one
    common reference molecule of the series instead of relative to the molecule itself, which is
    what the kernel actually compares; the reference is taken - in the pose it arrives in, before
    anything is generated - from `reference_index`, or the largest molecule if that is ``None``.
    """
    out, failures = [], []
    reference = None
    if align_to != "self":
        try:
            reference = dataset_reference(mols, reference_index)
        except Exception as exc:                 # noqa: BLE001 - degrade to the old behaviour
            failures.append(f"alignment reference: {exc}; conformations aligned on their own pose")
            align_to = "self"
    for i, m in enumerate(mols):
        if cancel is not None and cancel():
            break
        if progress is not None:
            progress(f"Conformations {i + 1}/{len(mols)}", i / max(len(mols), 1))
        try:
            out.append(generate_conformers(m, n=n, seed=seed, prune_rmsd=prune_rmsd,
                                           optimise=optimise, align_to=align_to,
                                           reference=reference, fieldset=fieldset,
                                           window_rmsd=window_rmsd))
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
    the number of conformations that actually matter, which with Boltzmann weights and a dominant
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
