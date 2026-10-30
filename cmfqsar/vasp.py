"""Per-atom arrays from a VASP run, read into an ASE structure (plan item M3).

Every materials study so far had to script the same step by hand: take the Bader charges of a
run, subtract the valence electrons, and put the result on the atoms as an array the periodic
field builder can read.  This module does that once.  It reads the three files a run leaves
behind and returns an ASE ``Atoms`` whose ``arrays`` carry what the fields need:

* ``CONTCAR`` (or ``POSCAR``): the structure, with the species in the order VASP wrote them.
* ``ACF.dat`` of the Bader program (Henkelman group): the electrons each atom holds, in the
  column ``CHARGE``.  A *net* charge is the valence electrons of the pseudopotential minus that
  number, so the Bader file alone is not enough.
* ``OUTCAR``: the valence electrons per species (``ZVAL``), the ions per species, and, for a
  spin-polarised run, the magnetic moment per atom (the last ``magnetization (x)`` block).

The result carries ``bader_charges`` (net Bader charges, also set as the atoms' initial charges,
both of which :func:`cmfqsar.periodic.crystal_charges` reads with the charge model ``"file"``),
``bader_electrons`` (the raw column), and ``magnetic_moment`` when the run was spin-polarised.
Any of them can then be a field as ``prop:<name>``, and the electrostatic field of :func:`cmfqsar.periodic.inorganic_fieldset` with ``charge_model="file"``
takes the net Bader charges in place of the oxidation-state guess.

A file that is missing is reported by name; a file whose atom count does not match the structure
is an error naming both counts, because the alternative is to put a charge on the wrong atom.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np


def read_acf(path: str) -> np.ndarray:
    """The ``CHARGE`` column of a Bader ``ACF.dat``: electrons per atom, in atom order."""
    out: List[float] = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                out.append(float(parts[4]))
    if not out:
        raise ValueError(f"{path}: no atom rows found; is this a Bader ACF.dat file?")
    return np.asarray(out, float)


def read_outcar_species(path: str) -> Dict[str, Sequence]:
    """What the OUTCAR says about the species: symbols, ions per type and valence electrons.

    Returns ``{"symbols": [...], "counts": [...], "zval": [...]}`` with one entry per species in
    the order VASP uses.  The symbols come from the ``VRHFIN`` lines of the pseudopotentials, the
    counts from ``ions per type`` and the valence electrons from the ``ZVAL`` line that lists
    them all at once (the per-pseudopotential ``ZVAL`` lines are used when that line is absent).
    """
    symbols: List[str] = []
    zval_each: List[float] = []
    zval_line: Optional[List[float]] = None
    counts: Optional[List[int]] = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            m = re.search(r"VRHFIN\s*=\s*([A-Za-z]{1,2})", line)
            if m:
                symbols.append(m.group(1))
                continue
            if "ions per type" in line:
                counts = [int(x) for x in line.split("=")[1].split()]
                continue
            s = line.strip()
            if s.startswith("ZVAL") and "mass" not in s:
                zval_line = [float(x) for x in s.split("=")[1].split()]
                continue
            m = re.search(r"ZVAL\s*=\s*([-\d.]+)", line)
            if m and "mass and valenz" in line:
                zval_each.append(float(m.group(1)))
    zval = zval_line if zval_line is not None else zval_each
    if not symbols or counts is None or not zval:
        raise ValueError(f"{path}: could not read the species (VRHFIN), the ions per type or "
                         f"the valence electrons (ZVAL) from this OUTCAR")
    if not (len(symbols) == len(counts) == len(zval)):
        raise ValueError(f"{path}: {len(symbols)} species, {len(counts)} counts and {len(zval)} "
                         f"ZVAL values do not agree")
    return {"symbols": symbols, "counts": counts, "zval": zval}


def read_outcar_magmoms(path: str) -> Optional[np.ndarray]:
    """The per-atom magnetic moments of the last ``magnetization (x)`` block, or None."""
    text = open(path, "r", errors="ignore").read()
    blocks = [m.start() for m in re.finditer(r"magnetization \(x\)", text)]
    if not blocks:
        return None
    tail = text[blocks[-1]:]
    out: List[float] = []
    started = False
    for line in tail.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            out.append(float(parts[-1]))
            started = True
        elif started and (not parts or parts[0].startswith("-") or parts[0] == "tot"):
            break
    return np.asarray(out, float) if out else None


def read_outcar_energy(path: str) -> Optional[float]:
    """The last ``energy  without entropy`` of an OUTCAR, in eV, or None."""
    e = re.findall(r"energy\s+without\s+entropy\s*=\s*([-\d.]+)", open(path, errors="ignore").read())
    return float(e[-1]) if e else None


def net_bader_charges(electrons: np.ndarray, symbols: Sequence[str],
                      zval: Dict[str, float]) -> np.ndarray:
    """Valence electrons of each atom's species minus the electrons Bader gave it."""
    missing = sorted({s for s in symbols if s not in zval})
    if missing:
        raise ValueError("no valence-electron count (ZVAL) for " + ", ".join(missing))
    z = np.array([float(zval[s]) for s in symbols])
    e = np.asarray(electrons, float)
    if len(e) != len(z):
        raise ValueError(f"{len(e)} Bader charges for {len(z)} atoms")
    return z - e


def structure_with_arrays(run_dir: str, structure: str = "CONTCAR",
                          zval: Optional[Dict[str, float]] = None,
                          require_bader: bool = True):
    """An ASE ``Atoms`` of a VASP run with its per-atom arrays attached.

    `run_dir` holds the run; `structure` names the file the geometry is read from (``CONTCAR``
    by default, ``POSCAR`` for the input geometry).  `zval` overrides the valence electrons per
    species, which are otherwise read from the OUTCAR.  With `require_bader` a run without an
    ``ACF.dat`` is an error; without it the structure is returned with whatever arrays exist.

    The arrays set are ``bader_charges`` (net Bader charges, also the initial charges of the
    atoms, so that they survive a write to extended XYZ), ``bader_electrons``, and
    ``magnetic_moment`` when the OUTCAR holds a magnetisation block; ``atoms.info["vasp_energy"]`` holds the final
    energy.
    """
    from ase.io import read

    geom = os.path.join(run_dir, structure)
    if not os.path.exists(geom):
        alt = os.path.join(run_dir, "POSCAR" if structure == "CONTCAR" else "CONTCAR")
        if not os.path.exists(alt):
            raise FileNotFoundError(f"{run_dir}: no {structure} (nor its alternative)")
        geom = alt
    atoms = read(geom, format="vasp")
    symbols = list(atoms.get_chemical_symbols())
    outcar = os.path.join(run_dir, "OUTCAR")
    have_outcar = os.path.exists(outcar)
    if zval is None and have_outcar:
        sp = read_outcar_species(outcar)
        # the OUTCAR's order of species is the POTCAR order, which is the order of the geometry
        zval = {s: float(z) for s, z in zip(sp["symbols"], sp["zval"])}
        if sum(sp["counts"]) != len(atoms):
            raise ValueError(f"{run_dir}: the OUTCAR has {sum(sp['counts'])} ions and the "
                             f"{os.path.basename(geom)} has {len(atoms)} atoms")
    acf = os.path.join(run_dir, "ACF.dat")
    if os.path.exists(acf):
        electrons = read_acf(acf)
        if len(electrons) != len(atoms):
            raise ValueError(f"{run_dir}: ACF.dat has {len(electrons)} atoms and the "
                             f"{os.path.basename(geom)} has {len(atoms)}")
        atoms.set_array("bader_electrons", electrons)
        if zval is None:
            raise ValueError(f"{run_dir}: the net Bader charges need the valence electrons per "
                             f"species, from the OUTCAR or the zval argument")
        q = net_bader_charges(electrons, symbols, zval)
        atoms.set_array("bader_charges", q)
        atoms.set_initial_charges(q)
    elif require_bader:
        raise FileNotFoundError(f"{run_dir}: no ACF.dat; run the Bader program on the CHGCAR "
                                f"first, or pass require_bader=False")
    if have_outcar:
        mm = read_outcar_magmoms(outcar)
        if mm is not None and len(mm) == len(atoms):
            atoms.set_array("magnetic_moment", mm)
        e = read_outcar_energy(outcar)
        if e is not None:
            atoms.info["vasp_energy"] = e
    atoms.info["source"] = os.path.abspath(run_dir)
    return atoms


def write_structure(atoms, path: str) -> None:
    """Write the structure with its arrays as extended XYZ, which keeps every per-atom array.

    The ``bader_charges`` array and the initial charges both come back from such a file, and
    :func:`cmfqsar.periodic.crystal_charges` reads either under the charge model ``"file"``.
    """
    from ase.io import write
    out = atoms.copy()
    if "momenta" in out.arrays and not np.any(out.arrays["momenta"]):
        del out.arrays["momenta"]
    write(path, out, format="extxyz")


def describe_arrays(atoms) -> str:
    """One line per per-atom array: its name, mean and range."""
    lines = []
    for key in sorted(atoms.arrays):
        if key in ("numbers", "positions"):
            continue
        v = np.asarray(atoms.arrays[key], float).reshape(len(atoms), -1)
        if v.shape[1] != 1:
            continue
        v = v[:, 0]
        lines.append(f"{key}: mean {v.mean():+.3f}, from {v.min():+.3f} to {v.max():+.3f}")
    return "\n".join(lines) if lines else "no per-atom arrays"
