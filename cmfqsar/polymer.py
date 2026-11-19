"""
Polymers as oligomers: a repeat unit written as SMILES becomes a capped chain in 3D, and the field
autocorrelation of one repeat unit inside that chain becomes the descriptor of the polymer.

A homopolymer has no single structure.  Polymer data sets write the repeat unit as a SMILES string
with two attachment points, ``[*]CC([*])c1ccccc1`` for polystyrene, and a property such as the
glass transition temperature belongs to the long chain, not to that fragment.  Every field
description in this program needs atoms at fixed places, so a polymer needs a stand-in.  This
module builds the usual one: an oligomer of a few repeat units with the two open ends capped,
in an extended conformation.

**Parsing.**  :func:`parse_repeat_unit` accepts a repeat unit with two attachment points (written
``*`` or ``[*]``) or, for a ladder polymer, four.  Each attachment point is bonded to one atom,
all carry bonds of one order, and anything else is refused with a message that says why.  A salt
written as two fragments is refused rather than guessed.

**Ladder polymers.**  A ladder polymer, such as the polymers of intrinsic microporosity (PIM-1),
joins its repeat units by two bonds each, so that a ring forms at every junction.  Its four
attachment points form two ends of two points each.  With atom map numbers, ``[*:1]`` and
``[*:2]`` are the head and ``[*:3]`` and ``[*:4]`` the tail, and 3 is joined to 1 and 4 to 2 of
the next unit.  Without them, the two ends are the split of the four points into two pairs whose
atoms are closest in the bond graph (counted in bonds; a split that ties with another is refused),
the head is the end with the first point written, and within each end the points are taken in the
order written: the first of the tail is joined to the first of the head.  The size of the ring
that forms does not depend on this order; only which of two regioisomers is built does, and for a
symmetric end, such as the two oxygens of a catechol, both are the same.

**Joining.**  :func:`oligomer` joins ``n`` copies head to tail with RDKit's ``molzip``, which
keeps the configuration of every stereocentre when an attachment point is replaced by a bond.
Each open end gets a hydrogen (a ``CH2=`` or ``CH#`` group when the open bond is double or
triple); a ladder has two open points at each end.  Every atom carries the integer property
``cmf_unit``: the number of its repeat unit, counted from 0, or -1 for a cap.  Stereochemistry
that the repeat unit does not specify is either fixed once and copied into every unit
(``stereo="isotactic"``, the default, which makes the build reproducible) or left to the
embedding (``stereo="random"``).  Polymer tables almost never say which tacticity was measured,
so neither choice is the measured one.

**Embedding.**  :func:`embed_oligomer` makes a few ETKDG conformations (four by default) from
random starting coordinates and keeps the most extended one -- the largest distance between
the first and the last backbone atom -- before an MMFF94 optimisation (UFF where MMFF has no
parameters) that holds that distance fixed.  An extended chain is the conformation a single
chain has in a crystal and the one that is closest to the same for every polymer; a random coil
of three units, or a chain that an optimisation in vacuum has folded, would put a different fold
into the descriptor of every polymer.  A ladder polymer cannot fold, and :func:`build` optimises
it without holding the distance.

**The descriptor of one repeat unit.**  :func:`autocorrelation_rows` computes the field
autocorrelation of :func:`cmfqsar.invariant.autocorrelation`, optionally counting only the atom
pairs whose *first* atom lies in a chosen set, here the central repeat unit.  With every atom in
the set the result equals the plain autocorrelation (the tests compare them).  With the central
unit it is the field environment of one repeat unit inside a chain: its pairs with its own atoms
and with the atoms of its neighbours, and nothing that starts at the caps.  The function also
replaces a dense array of (atoms x atoms x distance bins) by a sparse one, so an oligomer of 500
atoms at a width of 0.01 A costs megabytes, not gigabytes; one neighbour list serves every width
asked for.

**The periodic chain (plan item PL3).**  The central unit of a trimer sees one neighbour on each
side; an infinite chain sees as many as the distance cutoff reaches.  An extended trimer is not
a chain that repeats, because its three units have different conformations, so
:func:`chain_screw` builds one: copies of the middle unit, related by the screw (a rotation and a
translation) that carries the junction before the middle unit onto the junction after it.  Every
junction of that chain then has the bond, the bond angles and the torsions of the trimer's
junction, and the chain differs from the trimer only by the copies beyond the nearest
neighbours and the absence of caps.  The copies can meet again a few units further on, which
the screw's ``contact`` reports.  :func:`chain_autocorrelation_rows` gives the field of one unit
of that chain, sparse, so it works at the narrow widths where the dense periodic sum would need
gigabytes.  When the screw is a translation, :func:`periodic_chain` makes the cell of
:mod:`cmfqsar.periodic` (plan item M1), periodic along the translation only, and
:func:`cmfqsar.periodic.periodic_autocorrelation` gives the same field, with or without the
uniform background per repeat length.

**Copolymers (plan item PL4).**  :func:`sequence_oligomer` and :func:`build_sequence` join
different repeat units in a given order.  A copolymer has units in several local environments
(a unit with a neighbour on each side), and :func:`environment_weights` counts how often each
occurs in a set of chains: the copolymer is then the weighted set of the trimers of its
environments, and its descriptor the weighted mean of their central-unit rows.  The module does
not choose the chains; the caller gives them, as the architecture and the composition define
them.

**What the module does not do.**  It does not describe chain length, its distribution, packing,
crystallinity or crosslinking: none of these is in a field of a single oligomer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

#: the integer atom property that names the repeat unit of an atom (-1 for a cap)
UNIT_PROP = "cmf_unit"
#: the cap of an open end, by the order of the bond that the attachment point carried
_CAPS = {Chem.BondType.SINGLE: "[*:{}][H]", Chem.BondType.DOUBLE: "[*:{}]=[CH2]",
         Chem.BondType.TRIPLE: "[*:{}]#[CH]", Chem.BondType.AROMATIC: "[*:{}][H]"}
#: a Gaussian is cut off this many widths away from its centre; exp(-32) is 1e-14
_TAIL = 8.0


@dataclass
class RepeatUnit:
    """A repeat unit with its two attachment points, or four for a ladder polymer.

    `head` and `tail` are the indices of the two attachment atoms (atomic number 0) in `mol`,
    the head being the one written first; `bond` is the order of the bonds they carry.  A ladder
    polymer has a second point at each end, `head2` and `tail2`: `tail` is joined to `head` of the
    next unit and `tail2` to `head2`.  They are None for a linear polymer.
    """
    smiles: str
    mol: Chem.Mol
    head: int
    tail: int
    bond: Chem.BondType
    head2: Optional[int] = None
    tail2: Optional[int] = None

    @property
    def ladder(self) -> bool:
        return self.head2 is not None

    @property
    def heavy_atoms(self) -> int:
        return sum(1 for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1)


def _ladder_ends(mol: Chem.Mol, stars: List[int], psmiles: str) -> Tuple[int, int, int, int]:
    """(head, head2, tail, tail2) of four attachment points; see *Ladder polymers* above."""
    maps = [mol.GetAtomWithIdx(s).GetAtomMapNum() for s in stars]
    if sorted(maps) == [1, 2, 3, 4]:
        by = {m: s for m, s in zip(maps, stars)}
        return by[1], by[2], by[3], by[4]
    nb = [mol.GetAtomWithIdx(s).GetNeighbors()[0].GetIdx() for s in stars]
    D = Chem.GetDistanceMatrix(mol)
    splits = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    score = [D[nb[a], nb[b]] + D[nb[c], nb[d]] for (a, b), (c, d) in splits]
    best = int(np.argmin(score))
    if sum(1 for s in score if s == score[best]) > 1:
        raise ValueError(f"{psmiles!r}: the four attachment points cannot be split into two ends "
                         f"without a choice; number them [*:1] and [*:2] (head) and [*:3] and "
                         f"[*:4] (tail)")
    (a, b), (c, d) = splits[best]               # a = 0, the first point written, is in the head
    return stars[a], stars[b], stars[c], stars[d]


def parse_repeat_unit(psmiles: str) -> RepeatUnit:
    """Read a repeat unit written as SMILES with two attachment points (``*`` or ``[*]``), or
    four for a ladder polymer."""
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        raise ValueError(f"{psmiles!r} is not valid SMILES")
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) not in (2, 4):
        raise ValueError(f"{psmiles!r} has {len(stars)} attachment points; a repeat unit has two "
                         f"(a linear polymer) or four (a ladder polymer)")
    for s in stars:
        if mol.GetAtomWithIdx(s).GetDegree() != 1:
            raise ValueError(f"{psmiles!r}: an attachment point must be bonded to exactly one atom")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"{psmiles!r} has more than one fragment (a salt or a mixture); write the "
                         f"repeat unit as one connected piece")
    bonds = [mol.GetAtomWithIdx(s).GetBonds()[0].GetBondType() for s in stars]
    if len(set(bonds)) != 1:
        raise ValueError(f"{psmiles!r}: the attachment points carry bonds of different order "
                         f"({', '.join(str(b) for b in bonds)}), so the units cannot be joined "
                         f"head to tail")
    if bonds[0] not in _CAPS:
        raise ValueError(f"{psmiles!r}: an attachment bond of order {bonds[0]} is not supported")
    if len(stars) == 4:
        h, h2, t, t2 = _ladder_ends(mol, stars, psmiles)
        return RepeatUnit(psmiles, mol, h, t, bonds[0], h2, t2)
    return RepeatUnit(psmiles, mol, stars[0], stars[1], bonds[0])


def _fix_stereo(mol: Chem.Mol) -> Chem.Mol:
    """The first stereoisomer of `mol` with every unspecified centre and double bond fixed."""
    from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
                                                   StereoEnumerationOptions)
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, tryEmbedding=False,
                                    maxIsomers=1)
    for iso in EnumerateStereoisomers(mol, options=opts):
        return iso
    return mol


def oligomer(unit, n: int = 3, stereo: str = "isotactic", add_hydrogens: bool = True) -> Chem.Mol:
    """`n` repeat units joined head to tail, with both ends capped.

    `unit` is a :class:`RepeatUnit` or a SMILES string.  Every atom gets the integer property
    ``cmf_unit`` (see :data:`UNIT_PROP`): its unit, counted from 0 at the head end, or -1 for a cap
    atom; a hydrogen added here takes the unit of the atom it is bonded to.  ``stereo`` is
    ``"isotactic"`` (every centre the repeat unit leaves open is fixed once, the same way in every
    unit) or ``"random"`` (left open, so that the embedding chooses).
    """
    if isinstance(unit, str):
        unit = parse_repeat_unit(unit)
    if n < 1:
        raise ValueError("an oligomer needs at least one repeat unit")
    if stereo not in ("isotactic", "random"):
        raise ValueError(f"stereo must be 'isotactic' or 'random', not {stereo!r}")
    base = Chem.Mol(unit.mol)
    if stereo == "isotactic":
        base = _fix_stereo(base)
    # the points of one end: one for a linear polymer, two for a ladder
    e = 2 if unit.ladder else 1
    pieces = []
    for k in range(n):
        m = Chem.RWMol(base)
        for a in m.GetAtoms():
            a.SetIntProp(UNIT_PROP, k)
        # label the attachment points so that molzip joins tail k to head k + 1
        m.GetAtomWithIdx(unit.head).SetAtomMapNum(e * k + 1)
        m.GetAtomWithIdx(unit.tail).SetAtomMapNum(e * k + e + 1)
        if unit.ladder:
            m.GetAtomWithIdx(unit.head2).SetAtomMapNum(e * k + 2)
            m.GetAtomWithIdx(unit.tail2).SetAtomMapNum(e * k + e + 2)
        pieces.append(m.GetMol())
    caps = []
    for label in list(range(1, e + 1)) + list(range(e * n + 1, e * n + e + 1)):
        cap = Chem.MolFromSmiles(_CAPS[unit.bond].format(label), sanitize=False)
        for a in cap.GetAtoms():
            a.SetIntProp(UNIT_PROP, -1)
        caps.append(cap)
    combined = caps[0]
    for p in caps[1:e] + pieces + caps[e:]:
        combined = Chem.CombineMols(combined, p)
    out = Chem.molzip(combined)
    Chem.SanitizeMol(out)
    for a in out.GetAtoms():
        a.SetAtomMapNum(0)
    if add_hydrogens:
        out = Chem.AddHs(out)
        for a in out.GetAtoms():
            if not a.HasProp(UNIT_PROP):
                a.SetIntProp(UNIT_PROP, a.GetNeighbors()[0].GetIntProp(UNIT_PROP))
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    return out


def unit_labels(mol: Chem.Mol) -> np.ndarray:
    """The ``cmf_unit`` label of every atom, read from the atoms or from the SD property list."""
    if all(a.HasProp(UNIT_PROP) for a in mol.GetAtoms()):
        return np.array([a.GetIntProp(UNIT_PROP) for a in mol.GetAtoms()], dtype=int)
    key = f"atom.iprop.{UNIT_PROP}"
    if mol.HasProp(key):
        vals = [int(v) for v in mol.GetProp(key).split()]
        if len(vals) == mol.GetNumAtoms():
            return np.array(vals, dtype=int)
    raise ValueError(f"{mol.GetProp('_Name') if mol.HasProp('_Name') else 'the molecule'} carries "
                     f"no '{UNIT_PROP}' label on its atoms; build it with cmfqsar.polymer.oligomer")


def central_unit(mol: Chem.Mol) -> np.ndarray:
    """A boolean mask of the atoms of the middle repeat unit of an oligomer with an odd count."""
    labels = unit_labels(mol)
    n = int(labels.max()) + 1
    if n % 2 == 0:
        raise ValueError(f"an oligomer of {n} units has no middle unit; build an odd number")
    return labels == n // 2


def _end_atoms(mol: Chem.Mol) -> Tuple[int, int]:
    """The two atoms of the chain bonded to the caps: the first and the last backbone atom.

    A ladder has two capped atoms at each end; the first of the first unit and the first of the
    last unit are taken, so that the two never lie at the same end of the chain.
    """
    labels = unit_labels(mol)
    ends = []
    for a in mol.GetAtoms():
        if labels[a.GetIdx()] == -1 and a.GetAtomicNum() > 0:
            for nb in a.GetNeighbors():
                if labels[nb.GetIdx()] >= 0:
                    ends.append(nb.GetIdx())
    if len(ends) < 2:
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        return heavy[0], heavy[-1]
    first = [i for i in ends if labels[i] == 0]
    last = [i for i in ends if labels[i] == labels.max()]
    if labels.max() > 0 and first and last:
        # in the order of `ends`, so that a linear chain gets exactly (ends[0], ends[-1]) as before
        a, b = sorted((min(first), min(last)), key=ends.index)
        return a, b
    return ends[0], ends[-1]


def embed_oligomer(mol: Chem.Mol, conformers: int = 4, seed: int = 0,
                   optimise: bool = True, max_iters: int = 2000, timeout: int = 30,
                   hold_span: bool = True) -> Tuple[Chem.Mol, Dict[str, float]]:
    """The most extended of `conformers` ETKDG conformations, optimised with MMFF94.

    The embedding starts from random coordinates, which RDKit recommends for large molecules: on
    trimers of a few hundred atoms the default start often fails and costs more time than it
    saves.  `timeout` (seconds) bounds each embedding.  When no conformation can be made with the
    stereocentres enforced -- a phosphazene trimer has nine tetrahedral phosphorus and sulfur
    centres, and a bridged ring system can be given a set of configurations that does not exist
    in 3D -- one conformation is made without enforcing them, with four times the time, and the
    record says so (``stereo_enforced`` 0).

    An optimisation in vacuum folds a chain onto itself: on a polythiophene trimer with dodecyl
    side chains MMFF94 halved the end-to-end distance, from 19.8 to 9.6 A.  With `hold_span` (the
    default) the distance between the first and the last backbone atom is held at its embedded
    value by a stiff harmonic restraint, so that bond lengths, angles and side chains relax while
    the chain stays as extended as it was embedded.

    Returns the molecule with one conformer and a record: the end-to-end distance before and
    after the optimisation, the force field used ("MMFF94", "UFF" or "none") and whether the
    optimisation converged.  A copy is given to the force field and only the coordinates are
    copied back, because MMFF typing changes the aromaticity flags of the molecule it is given.
    """
    m = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    params.useRandomCoords = True
    params.timeout = int(timeout)
    enforced = 1.0
    # a failed embedding comes back as the id -1, not as an empty list, once a timeout is set
    ids = [i for i in AllChem.EmbedMultipleConfs(m, numConfs=int(conformers), params=params)
           if i >= 0]
    if not ids:
        params.enforceChirality = False
        params.timeout = 4 * int(timeout)
        enforced = 0.0
        m.RemoveAllConformers()
        ids = [i for i in AllChem.EmbedMultipleConfs(m, numConfs=1, params=params) if i >= 0]
    if not ids:
        raise ValueError("ETKDG could not embed the oligomer")
    a, b = _end_atoms(m)
    span = {cid: float(np.linalg.norm(np.asarray(m.GetConformer(cid).GetAtomPosition(a))
                                      - np.asarray(m.GetConformer(cid).GetAtomPosition(b))))
            for cid in ids}
    best = max(ids, key=lambda c: span[c])
    X = np.array(m.GetConformer(best).GetPositions())
    out = Chem.Mol(m)
    out.RemoveAllConformers()
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, p in enumerate(X):
        conf.SetAtomPosition(i, p.tolist())
    out.AddConformer(conf, assignId=True)
    info = {"span_embedded": span[best], "force_field": "none", "converged": 0.0,
            "stereo_enforced": enforced}
    if optimise:
        work = Chem.Mol(out)
        props = AllChem.MMFFGetMoleculeProperties(work)
        d0 = span[best]
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(work, props)
            info["force_field"] = "MMFF94"
            if ff is not None and hold_span:
                ff.MMFFAddDistanceConstraint(int(a), int(b), False, d0, d0, 100.0)
        else:
            ff = AllChem.UFFGetMoleculeForceField(work) if AllChem.UFFHasAllMoleculeParams(work) \
                else None
            info["force_field"] = "UFF" if ff is not None else "none"
            if ff is not None and hold_span:
                ff.UFFAddDistanceConstraint(int(a), int(b), False, d0, d0, 100.0)
        if ff is not None:
            info["converged"] = float(ff.Minimize(maxIts=int(max_iters)) == 0)
            Y = np.array(ff.Positions()).reshape(-1, 3)
            c = out.GetConformer()
            for i, p in enumerate(Y):
                c.SetAtomPosition(i, p.tolist())
    c = out.GetConformer()
    info["span"] = float(np.linalg.norm(np.asarray(c.GetAtomPosition(a))
                                        - np.asarray(c.GetAtomPosition(b))))
    return out, info


def build(psmiles: str, n: int = 3, stereo: str = "isotactic", conformers: int = 4,
          seed: int = 0, optimise: bool = True) -> Tuple[Chem.Mol, Dict[str, float]]:
    """Repeat unit to embedded oligomer in one call; the record also gives the atom counts.

    A ladder polymer is optimised without holding its end-to-end distance.  Its units are joined
    by rings, so the chain cannot fold by turning about single bonds, and on large fused systems
    the embedding is often bent: holding that span kept a bend that a free optimisation removes
    (one PIM trimer went from 17.6 to 36.3 A, and its aromatic rings from 0.2 to 0.03 A out of
    plane).
    """
    unit = parse_repeat_unit(psmiles)
    mol = oligomer(unit, n=n, stereo=stereo)
    hold = not unit.ladder

    def embed(m):
        return embed_oligomer(m, conformers=conformers, seed=seed, optimise=optimise,
                              hold_span=hold)
    try:
        out, info = embed(mol)
        info["stereo_open"] = 0.0
    except ValueError:
        if stereo != "isotactic":
            raise
        # a configuration fixed centre by centre can be one that no 3D structure has (a bridged
        # ring system); with the centres left open the embedding chooses ones that exist
        mol = oligomer(unit, n=n, stereo="random")
        out, info = embed(mol)
        info["stereo_open"] = 1.0
    labels = unit_labels(out)
    info.update({"units": float(n), "atoms": float(out.GetNumAtoms()),
                 "unit_heavy_atoms": float(unit.heavy_atoms),
                 "central_atoms": float(np.sum(labels == n // 2)) if n % 2 else float("nan")})
    Chem.CreateAtomIntPropertyList(out, UNIT_PROP)
    return out, info


def _build_one(task):
    """Worker of :func:`build_many`: (index, SMILES, options) to (index, MOL block, unit labels,
    record, error)."""
    k, smi, opts = task
    try:
        mol, info = build(smi, **opts)
        return k, Chem.MolToMolBlock(mol), unit_labels(mol).tolist(), info, ""
    except Exception as exc:                     # noqa: BLE001 - reported per repeat unit
        return k, "", [], {}, f"{type(exc).__name__}: {exc}"


def build_many(smiles: Sequence[str], n: int = 3, stereo: str = "isotactic",
               conformers: int = 4, seed: int = 0, jobs: int = 1, progress=None):
    """:func:`build` for a list of repeat units, in `jobs` processes.

    Returns one (molecule or None, record, error message) per input, in the input order.  The
    molecules come back through MOL blocks, so the ``cmf_unit`` labels travel as the SD property
    list that :func:`unit_labels` reads.
    """
    opts = {"n": n, "stereo": stereo, "conformers": conformers, "seed": seed}
    tasks = [(k, s, opts) for k, s in enumerate(smiles)]
    results = [None] * len(tasks)
    if jobs > 1 and len(tasks) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(int(jobs)) as pool:
            for done, r in enumerate(pool.imap_unordered(_build_one, tasks, chunksize=4)):
                results[r[0]] = r
                if progress is not None:
                    progress(done + 1, len(tasks))
    else:
        for done, t in enumerate(tasks):
            results[t[0]] = _build_one(t)
            if progress is not None:
                progress(done + 1, len(tasks))
    out = []
    for _, block, labels, info, err in results:
        mol = Chem.MolFromMolBlock(block, removeHs=False) if block else None
        if block and mol is None:
            err = "the built oligomer could not be read back"
        if mol is not None:
            for a, v in zip(mol.GetAtoms(), labels):
                a.SetIntProp(UNIT_PROP, v)
            Chem.CreateAtomIntPropertyList(mol, UNIT_PROP)
        out.append((mol, info, err))
    return out


def _pair_columns(W: np.ndarray, i: np.ndarray, j: np.ndarray, c: np.ndarray,
                  pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """(n_pairs_of_atoms, n_field_pairs): what each atom pair adds to each field-pair channel.

    For the atom pair (i, j) and the field pair (f, g) the ordered sum of
    :func:`cmfqsar.invariant.autocorrelation` gives W_if W_jg + W_jf W_ig, doubled when f != g.
    Restricted to ordered pairs whose first atom is in the centre set (c = 1), the term starting
    at i keeps the factor c_i and the term starting at j the factor c_j.
    """
    Wi, Wj = W[i], W[j]
    ci, cj = c[i], c[j]
    out = np.empty((len(i), len(pairs)))
    for k, (f, g) in enumerate(pairs):
        s = ci * Wi[:, f] * Wj[:, g] + cj * Wj[:, f] * Wi[:, g]
        if f != g:
            s = s + ci * Wi[:, g] * Wj[:, f] + cj * Wj[:, g] * Wi[:, f]
        out[:, k] = s
    return out


def autocorrelation_rows(mol, specs: Sequence, centre: Optional[np.ndarray] = None,
                         chunk: int = 200_000,
                         atom_pairs: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """The field autocorrelation of one molecule for each spec of `specs`, from one neighbour list.

    `mol` is a :class:`cmfqsar.fields.MoleculeFields`.  With `centre` None the rows equal those of
    :func:`cmfqsar.invariant.autocorrelation` (to the size of a Gaussian 8 widths from its centre,
    1e-14 relative).  With a boolean mask only the ordered atom pairs whose first atom is in the
    mask are counted: the pairs inside the set twice (both ways round), the pairs from the set to
    the rest once.  Over a partition of the atoms these rows add up to the plain autocorrelation,
    so the row of the central unit is the share of one repeat unit, as a periodic cell's is.

    `atom_pairs`, when given, is the list of atom pairs (i < j) to count in place of all pairs within
    reach; :func:`chain_autocorrelation_rows` passes only the pairs that start in its middle copy.
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    from .invariant import field_pairs

    if getattr(mol, "probe_coeffs", None):
        raise ValueError("autocorrelation_rows works on atom weights; a probe field is not supported")
    for s in specs:
        if s.chirality:
            raise ValueError("autocorrelation_rows has no signed (chirality) channels")
    X = np.asarray(mol.coords, float)
    W = np.asarray(mol.weights, float)
    n_atoms, n_fields = W.shape
    pairs = field_pairs(n_fields)
    c = np.ones(n_atoms) if centre is None else np.asarray(centre, bool).astype(float)
    if len(c) != n_atoms:
        raise ValueError(f"the centre mask has {len(c)} entries for {n_atoms} atoms")
    outs = [np.zeros((len(pairs), s.n_bins)) for s in specs]
    if n_atoms == 0:
        return [o.ravel() for o in outs]
    # the self terms: an atom with itself, at distance 0
    self_cols = _pair_columns(W, np.arange(n_atoms), np.arange(n_atoms), c, pairs) * 0.5
    self_sum = self_cols.sum(axis=0)
    for s, o in zip(specs, outs):
        if s.include_self:
            o += self_sum[:, None] * np.exp(-(s.centres ** 2) / (2.0 * s.sigma ** 2))[None, :]
    reach = max(s.d_max + _TAIL * s.sigma for s in specs)
    ij = (cKDTree(X).query_pairs(reach, output_type="ndarray") if atom_pairs is None
          else np.asarray(atom_pairs, int).reshape(-1, 2))
    if centre is not None and len(ij):
        keep = (c[ij[:, 0]] > 0) | (c[ij[:, 1]] > 0)
        ij = ij[keep]
    for start in range(0, len(ij), chunk):
        i, j = ij[start:start + chunk, 0], ij[start:start + chunk, 1]
        d = np.linalg.norm(X[i] - X[j], axis=1)
        U = _pair_columns(W, i, j, c, pairs)
        for s, o in zip(specs, outs):
            near = d <= s.d_max + _TAIL * s.sigma
            if not near.any():
                continue
            dd, UU = d[near], U[near]
            nb = s.n_bins
            width = int(math.ceil(2.0 * _TAIL * s.sigma / s.step)) + 2
            k0 = np.floor((dd - _TAIL * s.sigma) / s.step).astype(int)
            K = k0[:, None] + np.arange(width)[None, :]
            ok = (K >= 0) & (K < nb)
            Kc = np.clip(K, 0, nb - 1)
            vals = np.exp(-((dd[:, None] - s.centres[Kc]) ** 2) / (2.0 * s.sigma ** 2)) * ok
            rows = np.broadcast_to(np.arange(len(dd))[:, None], K.shape)
            G = csr_matrix((vals[ok], (rows[ok], K[ok])), shape=(len(dd), nb))
            o += (G.T @ UU).T
    return [o.ravel() for o in outs]


# --------------------------------------------------------------------------- the periodic chain

#: two atoms of a chain closer than this (A), without a bond between them, sit on top of each
#: other; :func:`cmfqsar.orca.geometry_faults` uses the same limit
CHAIN_CONTACT = 1.3
#: a screw that turns by less than this (degrees) per unit is taken as a translation, and the
#: chain then has the periodic cell of :func:`periodic_chain`
TRANSLATION_TURN = 5.0


@dataclass
class ChainScrew:
    """The operation that carries one unit of a chain onto the next: x -> R x + t.

    `R` is a rotation and `t` a translation; a chain of identical units is the middle unit and its
    images under the powers of this operation.  The record fields describe it: ``turn`` (degrees
    per unit) and ``rise`` (A per unit along the screw axis); ``frame_residual``, the root mean
    square misfit of the atoms that define it; ``rmsd_next``, how far the next unit of the
    trimer is from the image of the middle unit (0 when the trimer itself repeats); ``junction``,
    the bond length between two units; ``contact``, the closest approach of two atoms of the
    chain that are not bonded (below :data:`CHAIN_CONTACT` the chain overlaps itself).
    """
    R: np.ndarray
    t: np.ndarray
    turn: float
    rise: float
    frame_residual: float
    rmsd_next: float
    junction: float
    contact: float

    def apply(self, X: np.ndarray, k: int = 1) -> np.ndarray:
        """The positions `X` moved by the operation `k` times (k < 0 for the inverse)."""
        Y = np.asarray(X, float)
        for _ in range(abs(int(k))):
            Y = Y @ self.R.T + self.t if k > 0 else (Y - self.t) @ self.R
        return Y

    @property
    def is_translation(self) -> bool:
        return self.turn < TRANSLATION_TURN

    @property
    def overlaps(self) -> bool:
        return self.contact < CHAIN_CONTACT

    def record(self) -> Dict[str, float]:
        return {"turn": self.turn, "rise": self.rise, "frame_residual": self.frame_residual,
                "rmsd_next": self.rmsd_next, "junction": self.junction, "contact": self.contact}


def _unit_atoms(labels: np.ndarray, k: int) -> np.ndarray:
    return np.where(labels == k)[0]


def _chain_bonds(mol: Chem.Mol, labels: np.ndarray, mid: int):
    """The bonds of one unit as pairs of positions in the unit, and the junction bonds as
    (position in this unit, position in the next unit)."""
    u = _unit_atoms(labels, mid)
    nxt = _unit_atoms(labels, mid + 1)
    pos = {int(a): p for p, a in enumerate(u)}
    pos_next = {int(a): p for p, a in enumerate(nxt)}
    inner, junction = [], []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in pos and a2 in pos:
            inner.append((pos[a1], pos[a2]))
        elif a1 in pos and a2 in pos_next:
            junction.append((pos[a1], pos_next[a2]))
        elif a2 in pos and a1 in pos_next:
            junction.append((pos[a2], pos_next[a1]))
    return inner, junction


def _kabsch(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The rotation R and translation t with B ~ A R^T + t, in the least-squares sense."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - ca @ R.T


def _screw_contact(X: np.ndarray, screw_R, screw_t, junction, rise: float, reach: float) -> float:
    best = np.inf
    Y = X.copy()
    extent = 2.0 * float(np.max(np.linalg.norm(X - X.mean(axis=0), axis=1)))
    K = int(min(200, math.ceil((reach + extent) / max(rise, 0.1)) + 1))
    for k in range(1, K + 1):
        Y = Y @ screw_R.T + screw_t
        D = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
        if k == 1:
            for a, b in junction:
                D[a, b] = np.inf
        best = min(best, float(D.min()))
    return best


def chain_screw(mol: Chem.Mol, reach: float = 12.0) -> ChainScrew:
    """The screw of the chain that the middle unit of an oligomer stands for (plan item PL3).

    `mol` is an oligomer of an odd number of units, at least three, with one conformer and the
    ``cmf_unit`` labels (as :func:`build` returns it).  The atoms of all units come in the same
    order, so the units correspond atom by atom (checked here).  The screw carries the junction
    before the middle unit onto the junction after it: the atom of the previous unit bonded to the
    middle unit, the middle unit's head atom and its heavy atoms up to two bonds away go onto the
    middle unit's tail atom, the next unit's head atom and the same heavy atoms of the next unit.
    The chain made of copies of the middle unit then has, at every junction, the bond, the two
    bond angles and the torsions of the trimer's junction after the middle unit.

    An extended trimer is not a chain that repeats: its three units have different conformations.
    A pure translation of the middle unit put atoms of a chain 0.5 A apart on four test polymers,
    and the best rigid fit of one unit onto the next does so in 57 % of a sample of 432 PL2
    trimers.  The screw of
    the junction keeps the chemistry of a junction exactly, but its copies can still meet again
    a few units further on; ``contact`` says so.  `reach` (A) is how far along the chain the
    contacts are looked for.
    """
    labels = unit_labels(mol)
    n = int(labels.max()) + 1
    if n < 3 or n % 2 == 0:
        raise ValueError(f"a chain screw needs an odd number of units, at least three; this "
                         f"oligomer has {n}")
    mid = n // 2
    units = [_unit_atoms(labels, k) for k in (mid - 1, mid, mid + 1)]
    z = [np.array([mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in u]) for u in units]
    if not (len(z[0]) == len(z[1]) == len(z[2]) and (z[0] == z[1]).all()
            and (z[1] == z[2]).all()):
        raise ValueError("the units of this oligomer do not have their atoms in the same order")
    _, j_next = _chain_bonds(mol, labels, mid)
    _, j_prev = _chain_bonds(mol, labels, mid - 1)
    if not j_next or sorted(j_next) != sorted(j_prev):
        raise ValueError("the junctions before and after the middle unit could not be matched")
    X = np.asarray(mol.GetConformer().GetPositions(), float)
    pos = {int(a): p for p, a in enumerate(units[1])}
    src, dst = [], []
    for tp, hp in j_next:
        src += [X[units[0][tp]], X[units[1][hp]]]
        dst += [X[units[1][tp]], X[units[2][hp]]]
        shell = {int(units[1][hp])}
        for _ in range(2):
            shell |= {nb.GetIdx() for a in list(shell)
                      for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                      if nb.GetIdx() in pos and nb.GetAtomicNum() > 1}
        for a in sorted(shell - {int(units[1][hp])}):
            src.append(X[a])
            dst.append(X[units[2][pos[a]]])
    src, dst = np.array(src), np.array(dst)
    R, t = _kabsch(src, dst)
    residual = float(np.sqrt(np.mean(np.sum((src @ R.T + t - dst) ** 2, axis=1))))
    turn = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    w, v = np.linalg.eig(R)
    axis = np.real(v[:, int(np.argmin(np.abs(w - 1.0)))])
    rise = abs(float(t @ axis))
    heavy = z[1] > 1
    A = X[units[1]]
    moved = A @ R.T + t
    rmsd = float(np.sqrt(np.mean(np.sum((moved[heavy] - X[units[2]][heavy]) ** 2, axis=1))))
    junction = float(np.mean([np.linalg.norm(X[units[1][a]] - X[units[2][b]]) for a, b in j_next]))
    contact = _screw_contact(A, R, t, j_next, rise, reach)
    return ChainScrew(R=R, t=t, turn=turn, rise=rise, frame_residual=residual, rmsd_next=rmsd,
                      junction=junction, contact=contact)


def periodic_chain(mol: Chem.Mol, mf, screw: Optional[ChainScrew] = None, name: str = "",
                   side: float = 100.0):
    """The middle unit of an oligomer as a cell periodic along one lattice vector (plan item M1).

    Only a chain whose screw is a translation (:attr:`ChainScrew.is_translation`) has such a
    cell; the translation is the third lattice vector, the only periodic one, and two vectors of
    `side` angstrom normal to it complete the cell, which no pair crosses.  `mf` is the
    :class:`cmfqsar.fields.MoleculeFields` of the whole oligomer, so that the weights of the
    middle unit (the Gasteiger charges above all) are those it has inside a chain.  Returns a
    :class:`cmfqsar.periodic.CrystalFields` for :func:`cmfqsar.periodic.periodic_autocorrelation`,
    which adds the uniform background per repeat length that M1 defines.
    """
    from .periodic import CrystalFields
    screw = chain_screw(mol) if screw is None else screw
    if not screw.is_translation:
        raise ValueError(f"the chain turns by {screw.turn:.1f} degrees per unit, so it has no "
                         f"periodic cell; use chain_autocorrelation_rows")
    labels = unit_labels(mol)
    u = _unit_atoms(labels, int(labels.max()) // 2)
    tvec = np.asarray(screw.t, float)
    e = tvec / max(float(np.linalg.norm(tvec)), 1e-12)
    trial = np.array([1.0, 0.0, 0.0]) if abs(e[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(e, trial)
    a /= np.linalg.norm(a)
    b = np.cross(e, a)
    radii = getattr(mf, "radii", None)
    radii = np.ones(len(u)) if radii is None else np.asarray(radii, float)[u]
    return CrystalFields(name=name, coords=np.asarray(mf.coords, float)[u],
                         weights=np.asarray(mf.weights, float)[u], radii=radii,
                         symbols=[mol.GetAtomWithIdx(int(i)).GetSymbol() for i in u],
                         cell=np.array([side * a, side * b, tvec]), pbc=(False, False, True))


def chain_autocorrelation_rows(mol: Chem.Mol, mf, specs: Sequence,
                               screw: Optional[ChainScrew] = None,
                               copies: Optional[int] = None) -> List[np.ndarray]:
    """The field autocorrelation of one unit of the chain, without a background, for each spec of
    `specs` (:class:`cmfqsar.invariant.AutocorrelationSpec`).

    The middle unit is copied by the powers of the screw as far as the longest Gaussian reaches,
    and :func:`autocorrelation_rows` counts the pairs that start in the copy at the origin.  The
    rows follow the convention of the central-unit rows of a trimer, so the two compare channel
    by channel.  For a chain whose screw is a translation they equal
    :func:`cmfqsar.periodic.periodic_autocorrelation` on :func:`periodic_chain` with the
    background and the taper off, times the number of atoms of the unit.  `copies` limits the
    chain to that many copies on each side (1: the nearest neighbours only, as in a trimer).
    """
    from types import SimpleNamespace
    screw = chain_screw(mol) if screw is None else screw
    labels = unit_labels(mol)
    u = _unit_atoms(labels, int(labels.max()) // 2)
    X = np.asarray(mf.coords, float)[u]
    W = np.asarray(mf.weights, float)[u]
    reach = max(s.d_max + _TAIL * s.sigma for s in specs)
    extent = 2.0 * float(np.max(np.linalg.norm(X - X.mean(axis=0), axis=1)))
    K = int(min(200, math.ceil((reach + extent) / max(screw.rise, 0.1)) + 1))
    if copies is not None:
        K = min(K, int(copies))
    images = [X] + [screw.apply(X, k) for k in range(1, K + 1)] + \
             [screw.apply(X, -k) for k in range(1, K + 1)]
    Xall = np.concatenate(images)
    Wall = np.tile(W, (len(images), 1))
    centre = np.zeros(len(Xall), bool)
    centre[:len(u)] = True
    # only the pairs that start in the middle copy count: search from it alone
    from scipy.spatial import cKDTree
    near = cKDTree(X).query_ball_tree(cKDTree(Xall), reach)
    i = np.repeat(np.arange(len(u)), [len(v) for v in near])
    j = np.concatenate([np.asarray(v, int) for v in near]) if len(i) else np.zeros(0, int)
    keep = (j != i) & ((j >= len(u)) | (i < j))
    ij = np.stack([np.minimum(i, j)[keep], np.maximum(i, j)[keep]], axis=1)
    return autocorrelation_rows(SimpleNamespace(coords=Xall, weights=Wall), specs, centre=centre,
                                atom_pairs=ij)


# --------------------------------------------------------------------------- copolymers

def sequence_oligomer(units: Sequence, stereo: str = "isotactic",
                      add_hydrogens: bool = True) -> Chem.Mol:
    """An oligomer of the repeat units `units` in the order given (plan item PL4).

    Each entry is a :class:`RepeatUnit` or a SMILES string; unit k + 1 is joined to the tail of
    unit k, as :func:`oligomer` joins copies of one unit, and the labels, caps and stereo options
    are those of :func:`oligomer`.  A sequence of one unit repeated n times gives exactly
    ``oligomer(unit, n)`` (tested).  All units must have the same number of attachment points and
    the same bond order at them.
    """
    units = [parse_repeat_unit(u) if isinstance(u, str) else u for u in units]
    if not units:
        raise ValueError("a sequence needs at least one repeat unit")
    if stereo not in ("isotactic", "random"):
        raise ValueError(f"stereo must be 'isotactic' or 'random', not {stereo!r}")
    if len({u.ladder for u in units}) > 1 or len({u.bond for u in units}) > 1:
        raise ValueError("the units of a sequence must all be linear or all ladders, with one "
                         "bond order at their attachment points")
    first = units[0]
    e = 2 if first.ladder else 1
    n = len(units)
    bases: Dict[str, Chem.Mol] = {}
    pieces = []
    for k, unit in enumerate(units):
        if unit.smiles not in bases:
            base = Chem.Mol(unit.mol)
            bases[unit.smiles] = _fix_stereo(base) if stereo == "isotactic" else base
        m = Chem.RWMol(bases[unit.smiles])
        for a in m.GetAtoms():
            a.SetIntProp(UNIT_PROP, k)
        m.GetAtomWithIdx(unit.head).SetAtomMapNum(e * k + 1)
        m.GetAtomWithIdx(unit.tail).SetAtomMapNum(e * k + e + 1)
        if unit.ladder:
            m.GetAtomWithIdx(unit.head2).SetAtomMapNum(e * k + 2)
            m.GetAtomWithIdx(unit.tail2).SetAtomMapNum(e * k + e + 2)
        pieces.append(m.GetMol())
    caps = []
    for label in list(range(1, e + 1)) + list(range(e * n + 1, e * n + e + 1)):
        cap = Chem.MolFromSmiles(_CAPS[first.bond].format(label), sanitize=False)
        for a in cap.GetAtoms():
            a.SetIntProp(UNIT_PROP, -1)
        caps.append(cap)
    combined = caps[0]
    for p in caps[1:e] + pieces + caps[e:]:
        combined = Chem.CombineMols(combined, p)
    out = Chem.molzip(combined)
    Chem.SanitizeMol(out)
    for a in out.GetAtoms():
        a.SetAtomMapNum(0)
    if add_hydrogens:
        out = Chem.AddHs(out)
        for a in out.GetAtoms():
            if not a.HasProp(UNIT_PROP):
                a.SetIntProp(UNIT_PROP, a.GetNeighbors()[0].GetIntProp(UNIT_PROP))
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    return out


def build_sequence(units: Sequence[str], stereo: str = "isotactic", conformers: int = 4,
                   seed: int = 0, optimise: bool = True) -> Tuple[Chem.Mol, Dict[str, float]]:
    """:func:`build` for a sequence of repeat units: the oligomer of :func:`sequence_oligomer`,
    embedded and optimised as :func:`build` does it."""
    parsed = [parse_repeat_unit(u) for u in units]
    hold = not parsed[0].ladder

    def embed(m):
        return embed_oligomer(m, conformers=conformers, seed=seed, optimise=optimise,
                              hold_span=hold)
    try:
        out, info = embed(sequence_oligomer(parsed, stereo=stereo))
        info["stereo_open"] = 0.0
    except ValueError:
        if stereo != "isotactic":
            raise
        out, info = embed(sequence_oligomer(parsed, stereo="random"))
        info["stereo_open"] = 1.0
    n = len(parsed)
    labels = unit_labels(out)
    info.update({"units": float(n), "atoms": float(out.GetNumAtoms()),
                 "central_atoms": float(np.sum(labels == n // 2)) if n % 2 else float("nan")})
    Chem.CreateAtomIntPropertyList(out, UNIT_PROP)
    return out, info


def environment_weights(sequences: Iterable[Sequence], weights: Optional[Sequence[float]] = None,
                        both_directions: bool = True) -> Dict[Tuple, float]:
    """How often each local environment occurs inside a set of chains (plan item PL4).

    A chain is a sequence of unit names, such as ``"AABABBAB"``.  The environment of an inner
    unit is the triple (unit before, unit, unit after); the two end units, which have one
    neighbour, are left out.  With `both_directions` every chain is counted also read backwards,
    because a chain has no direction, while the builder joins each unit's tail to the next
    unit's head.  The result is normalised to sum to 1: the weights of the trimers whose central
    units, averaged, describe a copolymer as an ensemble of environments.
    """
    seqs = [tuple(s) for s in sequences]
    w = np.ones(len(seqs)) if weights is None else np.asarray(weights, float)
    out: Dict[Tuple, float] = {}
    for s, ws in zip(seqs, w):
        if len(s) < 3:
            raise ValueError(f"a chain of {len(s)} units has no inner unit")
        readings = [s, s[::-1]] if both_directions else [s]
        share = float(ws) / (len(readings) * (len(s) - 2))
        for r in readings:
            for i in range(1, len(r) - 1):
                key = (r[i - 1], r[i], r[i + 1])
                out[key] = out.get(key, 0.0) + share
    total = sum(out.values())
    return {k: v / total for k, v in sorted(out.items())}
