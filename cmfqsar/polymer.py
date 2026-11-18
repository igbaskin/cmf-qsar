"""
Polymers as oligomers: a repeat unit written as SMILES becomes a capped chain in 3D, and the field
autocorrelation of one repeat unit inside that chain becomes the descriptor of the polymer.

A homopolymer has no single structure.  Polymer data sets write the repeat unit as a SMILES string
with two attachment points, ``[*]CC([*])c1ccccc1`` for polystyrene, and a property such as the
glass transition temperature belongs to the long chain, not to that fragment.  Every field
description in this program needs atoms at fixed places, so a polymer needs a stand-in.  This
module builds the usual one: an oligomer of a few repeat units with the two open ends capped,
in an extended conformation.

**Parsing.**  :func:`parse_repeat_unit` accepts a repeat unit with exactly two attachment points
(written ``*`` or ``[*]``), each bonded to one atom, with the same bond order at both ends, and
refuses anything else with a message that says why.  A ladder polymer (four attachment points) or
a salt written as two fragments is refused rather than guessed.

**Joining.**  :func:`oligomer` joins ``n`` copies head to tail with RDKit's ``molzip``, which
keeps the configuration of every stereocentre when an attachment point is replaced by a bond.
The two open ends get a hydrogen (a ``CH2=`` or ``CH#`` group when the open bond is double or
triple).  Every atom carries the integer property ``cmf_unit``: the number of its repeat unit,
counted from 0, or -1 for a cap.  Stereochemistry that the repeat unit does not specify is either
fixed once and copied into every unit (``stereo="isotactic"``, the default, which makes the build
reproducible) or left to the embedding (``stereo="random"``).  Polymer tables almost never say
which tacticity was measured, so neither choice is the measured one.

**Embedding.**  :func:`embed_oligomer` makes a few ETKDG conformations (four by default) from
random starting coordinates and keeps the most extended one -- the largest distance between
the first and the last backbone atom -- before an MMFF94 optimisation (UFF where MMFF has no
parameters) that holds that distance fixed.  An extended chain is the conformation a single
chain has in a crystal and the one that is closest to the same for every polymer; a random coil
of three units, or a chain that an optimisation in vacuum has folded, would put a different fold
into the descriptor of every polymer.

**The descriptor of one repeat unit.**  :func:`autocorrelation_rows` computes the field
autocorrelation of :func:`cmfqsar.invariant.autocorrelation`, optionally counting only the atom
pairs whose *first* atom lies in a chosen set, here the central repeat unit.  With every atom in
the set the result equals the plain autocorrelation (the tests compare them).  With the central
unit it is the field environment of one repeat unit inside a chain: its pairs with its own atoms
and with the atoms of its neighbours, and nothing that starts at the caps.  This is what the
periodic chain of :mod:`cmfqsar.periodic` computes per repeat length, without the need to build a
periodic cell with the right repeat distance.  The function also replaces a dense array of
(atoms x atoms x distance bins) by a sparse one, so an oligomer of 500 atoms at a width of
0.01 A costs megabytes, not gigabytes; one neighbour list serves every width asked for.

**What the module does not do.**  It does not describe chain length, its distribution, packing,
crystallinity or crosslinking: none of these is in a field of a single oligomer.  A copolymer
needs a sequence, which this module does not choose.
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
    """A repeat unit with its two attachment points.

    `head` and `tail` are the indices of the two attachment atoms (atomic number 0) in `mol`,
    the head being the one written first; `bond` is the order of the bonds they carry.
    """
    smiles: str
    mol: Chem.Mol
    head: int
    tail: int
    bond: Chem.BondType

    @property
    def heavy_atoms(self) -> int:
        return sum(1 for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1)


def parse_repeat_unit(psmiles: str) -> RepeatUnit:
    """Read a repeat unit written as SMILES with two attachment points (``*`` or ``[*]``)."""
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        raise ValueError(f"{psmiles!r} is not valid SMILES")
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        raise ValueError(f"{psmiles!r} has {len(stars)} attachment points; a repeat unit of a "
                         f"linear polymer has exactly two")
    for s in stars:
        if mol.GetAtomWithIdx(s).GetDegree() != 1:
            raise ValueError(f"{psmiles!r}: an attachment point must be bonded to exactly one atom")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"{psmiles!r} has more than one fragment (a salt or a mixture); write the "
                         f"repeat unit as one connected piece")
    bonds = [mol.GetAtomWithIdx(s).GetBonds()[0].GetBondType() for s in stars]
    if bonds[0] != bonds[1]:
        raise ValueError(f"{psmiles!r}: the two attachment points carry bonds of different order "
                         f"({bonds[0]} and {bonds[1]}), so the units cannot be joined head to tail")
    if bonds[0] not in _CAPS:
        raise ValueError(f"{psmiles!r}: an attachment bond of order {bonds[0]} is not supported")
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
    pieces = []
    for k in range(n):
        m = Chem.RWMol(base)
        for a in m.GetAtoms():
            a.SetIntProp(UNIT_PROP, k)
        # label the attachment points so that molzip joins tail k to head k + 1
        m.GetAtomWithIdx(unit.head).SetAtomMapNum(k + 1)
        m.GetAtomWithIdx(unit.tail).SetAtomMapNum(k + 2)
        pieces.append(m.GetMol())
    caps = []
    for label in (1, n + 1):
        cap = Chem.MolFromSmiles(_CAPS[unit.bond].format(label), sanitize=False)
        for a in cap.GetAtoms():
            a.SetIntProp(UNIT_PROP, -1)
        caps.append(cap)
    combined = caps[0]
    for p in pieces + [caps[1]]:
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
    """The two heavy atoms of the chain bonded to the caps: the first and the last backbone atom."""
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
    """Repeat unit to embedded oligomer in one call; the record also gives the atom counts."""
    unit = parse_repeat_unit(psmiles)
    mol = oligomer(unit, n=n, stereo=stereo)
    try:
        out, info = embed_oligomer(mol, conformers=conformers, seed=seed, optimise=optimise)
        info["stereo_open"] = 0.0
    except ValueError:
        if stereo != "isotactic":
            raise
        # a configuration fixed centre by centre can be one that no 3D structure has (a bridged
        # ring system); with the centres left open the embedding chooses ones that exist
        mol = oligomer(unit, n=n, stereo="random")
        out, info = embed_oligomer(mol, conformers=conformers, seed=seed, optimise=optimise)
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
                         chunk: int = 200_000) -> List[np.ndarray]:
    """The field autocorrelation of one molecule for each spec of `specs`, from one neighbour list.

    `mol` is a :class:`cmfqsar.fields.MoleculeFields`.  With `centre` None the rows equal those of
    :func:`cmfqsar.invariant.autocorrelation` (to the size of a Gaussian 8 widths from its centre,
    1e-14 relative).  With a boolean mask only the ordered atom pairs whose first atom is in the
    mask are counted: the pairs inside the set twice (both ways round), the pairs from the set to
    the rest once.  Over a partition of the atoms these rows add up to the plain autocorrelation,
    so the row of the central unit is the share of one repeat unit, as a periodic cell's is.
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
    ij = cKDTree(X).query_pairs(reach, output_type="ndarray")
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
