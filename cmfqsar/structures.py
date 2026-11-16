"""Structures that are not molecules: crystals, layers, chains and particles (plan item NT1).

The rest of the program was built around one object, an RDKit molecule with one conformation.
A crystal, a layer, a nanotube or a metal cluster is not such an object: it has no bond table,
and it may have a cell and directions along which it repeats.  This module reads such structures
as ASE ``Atoms`` and puts them in the same :class:`cmfqsar.dataio.Dataset` the molecules use,
with a name, a target value and a training or test label for each.

Four things are read here.

* **The files.**  CIF, the POSCAR and CONTCAR files of VASP, extended XYZ, ASE trajectories, and
  ASE databases in both of their forms (SQLite ``.db`` and ``.json``).  ASE reads all of them,
  but its general reader drops the key-value pairs of a database row, which is where a database
  keeps the name and the property of each structure; :func:`read_database` keeps them.
* **The boundary conditions.**  The number of directions along which a structure repeats: three
  for a crystal, two for a layer (a sheet or a slab), one for a chain (a tube or a wire), none
  for a particle or a molecule.  A structure with no periodic direction needs no cell at all;
  :func:`boundary` checks that every periodic direction has a lattice vector.
* **The name.**  Taken from the first of the info fields in :data:`NAME_KEYS` that a structure
  carries, else from the file name.  Names are made unique, because the target values of a CSV
  file are matched on them.
* **The target and the split.**  A target is a numeric info field (:func:`numeric_info_keys`),
  or a column of a CSV table matched on the names; an info field called ``subset``, ``set`` or
  ``split`` gives the published training and test sets, as an SD tag does for molecules.

What this module does not do is decide which kernel a structure gets.  The periodic
autocorrelation of :mod:`cmfqsar.periodic` takes all four kinds: with no periodic direction it
has no background term, and a cluster with no cell gives the channels of the same cluster alone
in a large periodic box with the background switched off (``benchmarks/run_structures.py``).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: file name endings read as structures rather than as molecules, lower-cased.  ``.xyz`` and
#: ``.json`` are not here: an XYZ file is a structure only when it states a lattice, and a JSON
#: file only when it is an ASE database (:func:`is_structure_file`).
STRUCTURE_ENDINGS: Tuple[str, ...] = (".cif", ".vasp", ".poscar", ".contcar", ".extxyz", ".db",
                                      ".traj", ".cell", ".res", ".xsf", ".pwo")
#: the info fields a name is taken from, in this order
NAME_KEYS: Tuple[str, ...] = ("name", "title", "label", "material_id", "cod_id", "id", "cod")
#: info fields that hold an identifier or a bookkeeping number and never a target
_NOT_A_TARGET = re.compile(
    r"(^|_)(id|ids|cod|csid|icsd|index|idx|number|num|seed|spacegroup|sg|space_group|nsites|"
    r"natoms|unique_id|ctime|mtime|user|calculator|subset|set|split|frame|step)$", re.I)
#: the info fields read as the published split
SPLIT_KEYS: Tuple[str, ...] = ("subset", "set", "split")
#: what a number of periodic directions is called
BOUNDARY_NAMES = {3: "crystal", 2: "layer", 1: "chain", 0: "open"}


# --------------------------------------------------------------------------- #
# Which files are structures
# --------------------------------------------------------------------------- #
def _head(path: str, size: int = 4096) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(size)
    except OSError:
        return ""


def is_structure_file(path: str) -> bool:
    """True when a file holds structures for this module rather than molecules.

    The endings of :data:`STRUCTURE_ENDINGS` and the names POSCAR and CONTCAR, which VASP writes
    without an ending, always are.  An XYZ file is one when its first comment line states a
    lattice (``Lattice="..."``, the extended XYZ convention); a plain XYZ file holds molecules
    and stays with the molecule reader.  A JSON file is one when it is an ASE database.
    """
    low = os.path.basename(str(path)).lower()
    if low.endswith(STRUCTURE_ENDINGS) or low.split(".")[0] in ("poscar", "contcar"):
        return True
    if low.endswith(".xyz"):
        lines = _head(path).splitlines()
        return len(lines) > 1 and "lattice=" in lines[1].lower()
    if low.endswith(".json"):
        text = _head(path)
        return text.lstrip().startswith("{") and '"numbers"' in text and '"positions"' in text
    return False


def is_database(path: str) -> bool:
    """True for an ASE database: an SQLite ``.db`` file or an ASE ``.json`` database."""
    low = str(path).lower()
    return low.endswith(".db") or (low.endswith(".json") and is_structure_file(path))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_database(path: str, selection: Optional[str] = None) -> list:
    """The structures of an ASE database, each with its row's key-value pairs in ``info``.

    ASE's general reader (``ase.io.read``) returns the atoms of each row and drops its key-value
    pairs, which is where a database keeps the name and the properties of a structure.  Here
    they are copied into ``atoms.info``, with the row number as ``id`` when the row has no name.
    ``selection`` is an ASE query string such as ``"natoms<20,gap>1"``.
    """
    import ase.db

    db = ase.db.connect(path)
    out = []
    for row in (db.select(selection) if selection else db.select()):
        atoms = row.toatoms()
        atoms.info.update({str(k): v for k, v in row.key_value_pairs.items()})
        atoms.info.setdefault("id", int(row.id))
        out.append(atoms)
    return out


def read_structure_file(path: str, index: str = ":") -> list:
    """Every structure of one file, as a list of ASE ``Atoms``.

    A database is read by :func:`read_database`; every other file by
    :func:`cmfqsar.periodic.read_structures`, which reads a CIF with partial occupancies through
    the disorder reader.  ``index`` picks structures as ASE does (``":"`` for all, ``"0"`` for
    the first, ``"-1"`` for the last).
    """
    if is_database(path):
        got = read_database(path)
        if index not in (":", None, ""):
            from ase.io.formats import string2index
            pick = string2index(index)
            got = got[pick] if isinstance(pick, slice) else [got[pick]]
    else:
        from .periodic import read_structures
        got = read_structures(path, index=index)
    for a in got:
        _results_to_info(a)
    return got


def _results_to_info(atoms) -> None:
    """Copy the single numbers a file stored as calculator results into ``atoms.info``.

    An extended XYZ file or a database row that holds an ``energy`` or a ``free_energy`` gives
    it to ASE as the result of a calculator, not as an info field, so a target stored that way
    would not be found.  A value already in ``info`` is kept.
    """
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", None) or {}
    for k, v in results.items():
        if np.ndim(v) == 0 and _number(v) is not None:
            atoms.info.setdefault(str(k), float(v))


def boundary(atoms) -> int:
    """The number of directions along which a structure repeats: 3, 2, 1 or 0.

    A periodic direction needs a lattice vector of non-zero length, and a structure that says
    it repeats along a direction it has no vector for is refused with the reason.  A structure
    with no periodic direction (a particle, a cluster, a molecule) needs no cell.
    """
    pbc = [bool(p) for p in atoms.get_pbc()]
    lengths = np.linalg.norm(np.asarray(atoms.get_cell(), float), axis=1)
    for k, (p, ln) in enumerate(zip(pbc, lengths)):
        if p and ln < 1e-8:
            raise ValueError(
                f"structure {structure_name(atoms, '?')!r} repeats along lattice vector "
                f"{'abc'[k]} but that vector has zero length; give the cell or turn that "
                f"direction off")
    return int(sum(pbc))


def boundary_name(atoms) -> str:
    """``crystal``, ``layer``, ``chain`` or ``open`` (no periodic direction)."""
    return BOUNDARY_NAMES[boundary(atoms)]


def structure_name(atoms, fallback: str) -> str:
    """The first non-empty info field of :data:`NAME_KEYS`, else ``fallback``."""
    for key in NAME_KEYS:
        v = atoms.info.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return fallback


def unique_names(names: Sequence[str]) -> List[str]:
    """The names with ``_2``, ``_3``, ... added to the second and later copies of a name."""
    seen: Dict[str, int] = {}
    out = []
    for nm in names:
        seen[nm] = seen.get(nm, 0) + 1
        out.append(nm if seen[nm] == 1 else f"{nm}_{seen[nm]}")
    return out


def _gather(atoms):
    """A layer or a chain left split across its cell boundary, gathered about the middle.

    A structure with no periodic direction is returned as it is: it has nothing to wrap across,
    and it may have no cell to take fractional coordinates in.
    """
    from .periodic import centre_layer
    pbc = [bool(p) for p in atoms.get_pbc()]
    if all(pbc) or not any(pbc):
        return atoms
    cell = np.asarray(atoms.get_cell(), float)
    if abs(np.linalg.det(cell)) < 1e-12:      # a missing vector along an open direction
        return atoms
    return centre_layer(atoms)


def read_structure_set(path: str, pbc: Optional[str] = None,
                       index: str = ":") -> Tuple[list, List[str]]:
    """A set of structures with a name for each, from one file or a folder of files.

    A folder is read file by file (:func:`cmfqsar.periodic.structure_files`); a file with one
    structure gives it the file's name, and a file with several gives each its name from the
    info fields, else ``<file>_<k>``.  A single file names each structure from its info fields,
    else after the file (one structure) or ``<formula>_<k>`` (several).

    ``pbc`` sets which lattice directions repeat for every structure (``abc``, ``ab``, ``a``, or
    ``none`` for open boundaries); without it the files decide.  A layer or a chain is gathered
    about the middle of its cell, because a file may leave it split across the boundary along
    the open direction.  The names are made unique.
    """
    from .periodic import pbc_flags, structure_files
    if os.path.isdir(path):
        files = structure_files(path)
        if not files:
            raise ValueError(f"{path} holds no structure file; the endings read are "
                             f"{', '.join(STRUCTURE_ENDINGS)}, POSCAR and CONTCAR")
        atoms, names = [], []
        for f in files:
            got = read_structure_file(f, index=index)
            stem = os.path.splitext(os.path.basename(f))[0]
            for k, a in enumerate(got):
                atoms.append(a)
                names.append(stem if len(got) == 1 else structure_name(a, f"{stem}_{k}"))
    else:
        atoms = read_structure_file(path, index=index)
        stem = os.path.splitext(os.path.basename(path))[0]
        names = [structure_name(a, stem if len(atoms) == 1 else
                                f"{a.get_chemical_formula()}_{k}")
                 for k, a in enumerate(atoms)]
    if pbc:
        flags = (False, False, False) if str(pbc).lower() == "none" else pbc_flags(pbc)
        for a in atoms:
            a.set_pbc(flags)
    for a in atoms:
        boundary(a)
    atoms = [_gather(a) for a in atoms]
    names = unique_names(names)
    for a, nm in zip(atoms, names):
        a.info["name"] = nm
    return atoms, names


# --------------------------------------------------------------------------- #
# Targets and the split
# --------------------------------------------------------------------------- #
def _number(v) -> Optional[float]:
    if isinstance(v, (bool, np.bool_)):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def numeric_info_keys(structures: Sequence, min_fraction: float = 0.5,
                      targets_only: bool = False) -> List[str]:
    """Info fields that hold a number for at least ``min_fraction`` of the structures.

    With ``targets_only`` the fields that hold identifiers or bookkeeping numbers (an ``id``, a
    COD number, a space group, a step) are left out, which is what a list of candidate targets
    should show.
    """
    counts: Dict[str, int] = {}
    order: List[str] = []
    for a in structures:
        for k, v in a.info.items():
            if np.ndim(v) != 0 or _number(v) is None:
                continue
            if k not in counts:
                order.append(k)
            counts[k] = counts.get(k, 0) + 1
    n = max(1, len(structures))
    keys = [k for k in order if counts[k] / n >= min_fraction]
    if targets_only:
        keys = [k for k in keys if not _NOT_A_TARGET.search(k)]
    return keys


def guess_target_key(structures: Sequence) -> Optional[str]:
    """The info field that is most likely the target, or None when that is not clear.

    A field called ``target``, ``y``, ``value``, ``property`` or ``activity`` is taken; otherwise
    the one candidate of :func:`numeric_info_keys` when there is exactly one.  With two or more
    candidates nothing is guessed, and the caller must name the field.
    """
    keys = numeric_info_keys(structures, targets_only=True)
    low = {k.lower(): k for k in keys}
    for k in ("target", "y", "value", "property", "activity"):
        if k in low:
            return low[k]
    return keys[0] if len(keys) == 1 else None


def info_values(structures: Sequence, key: str) -> np.ndarray:
    """``info[key]`` of each structure as a float, NaN where it is missing or not a number."""
    out = np.full(len(structures), np.nan)
    for i, a in enumerate(structures):
        x = _number(a.info.get(key))
        if x is not None:
            out[i] = x
    return out


def split_labels(structures: Sequence) -> Optional[List[Optional[str]]]:
    """``train``, ``test`` or ``decoy`` for each structure, from a ``subset`` info field.

    The values are read as for molecules (``1``/``train``, ``2``/``test``); a structure with no
    such field gets None.  None for the whole set when no structure has one.
    """
    from .dataio import _subset_label
    key = next((k for k in SPLIT_KEYS if any(k in a.info for a in structures)), None)
    if key is None:
        return None
    labels = [_subset_label(str(a.info[key])) if key in a.info else None for a in structures]
    return labels if any(lb is not None for lb in labels) else None


def target_csv_column(path: str) -> Optional[str]:
    """The column of a CSV table to read the target from when none is named.

    A file with no header, or with one numeric column, needs no choice (None: the second column
    is read).  Otherwise the numeric columns that hold an identifier (a COD number, an id) are
    set aside, and the one column left is taken; with two or more left the choice is refused,
    with the list, because taking the first numeric column of a prepared table quietly reads a
    database number as the property.
    """
    from .dataio import activity_columns
    cols = activity_columns(path)
    if len(cols) <= 1:
        return None
    left = [c for c in cols if not _NOT_A_TARGET.search(c)]
    if len(left) == 1:
        return left[0]
    raise ValueError(f"{os.path.basename(path)} has several numeric columns "
                     f"({', '.join(left or cols)}); name the one that holds the target")


def load_structures(path: str, target: Optional[str] = None, target_file: Optional[str] = None,
                    target_column: Optional[str] = None, pbc: Optional[str] = None,
                    subset: str = "train", use_subset_column: bool = True, index: str = ":"):
    """A :class:`cmfqsar.dataio.Dataset` of structures read from a file or a folder.

    The target is the info field ``target`` names, or the column ``target_column`` of the CSV
    file ``target_file`` matched on the names, or, when neither is given, the field
    :func:`guess_target_key` picks; a structure with no value gets NaN.  An info field ``subset``
    gives the published split when ``use_subset_column`` is set; every other structure is in
    ``subset``.
    """
    from .dataio import Dataset, read_activities_csv
    atoms, names = read_structure_set(path, pbc=pbc, index=index)
    keys = numeric_info_keys(atoms)
    ds = Dataset(names=list(names), structures=list(atoms), activity=np.full(len(atoms), np.nan),
                 subset=np.array([subset] * len(atoms), dtype=object), properties=keys,
                 path=path)
    if target_file:
        target_column = target_column or target or target_csv_column(target_file)
        table = read_activities_csv(target_file, column=target_column)
        ds.activity = np.array([float(table[n]) if n in table else np.nan for n in names])
        ds.activity_prop = os.path.basename(target_file) + (f" [{target_column}]"
                                                            if target_column else "")
    else:
        key = target or guess_target_key(atoms)
        if key:
            ds.set_activity_from_property(key)
    if use_subset_column:
        labels = split_labels(atoms)
        if labels is not None:
            ds.subset = np.array([lb or subset for lb in labels], dtype=object)
    return ds


def summary(ds) -> str:
    """One line on a structure set: how many, of what kind, how many atoms, how many targets."""
    kinds: Dict[str, int] = {}
    for a in ds.structures:
        kinds[boundary_name(a)] = kinds.get(boundary_name(a), 0) + 1
    sizes = [len(a) for a in ds.structures]
    n_y = int(np.isfinite(np.asarray(ds.activity, float)).sum()) if len(ds.activity) else 0
    words = {"crystal": "crystals", "layer": "layers", "chain": "chains",
             "open": "with open boundaries"}
    parts = [f"{n} {words[k]}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])]
    return (f"{len(ds)} structures ({', '.join(parts)}), "
            f"{min(sizes) if sizes else 0} to {max(sizes) if sizes else 0} atoms each, "
            f"{n_y} with a target" + (f" ({ds.activity_prop})" if ds.activity_prop else ""))
