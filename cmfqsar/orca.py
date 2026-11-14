"""
Quantum-chemical properties from ORCA: per-atom quantities that become fields, per-molecule ones
that become descriptors.

The program's own fields are built from what it can work out itself -- Gasteiger charges, Crippen
increments, van der Waals parameters -- and that is a limit.  A quantum calculation raises it: a
Hirshfeld charge, a Fukui index or a local softness says something about an atom that no increment
table knows, and the frontier orbital energies say something about the molecule that no field can
carry.  This module runs ORCA over a set of molecules and brings both back: the per-atom values as
atom property lists, which become ``prop:`` fields, and the per-molecule values as molecule
properties, which become descriptors.

What it computes, with the default settings: Mulliken, Loewdin, Hirshfeld and CHELPG charges, the
HOMO and LUMO populations of every atom, and -- when the N+1 and N-1 states are run as well -- the
condensed Fukui indices f+, f- and f0, the dual descriptor, local softnesses and the local
electrophilicity.  Per molecule: the total energy, the frontier orbital energies and their gap,
the dipole, and from the ionised states the ionisation potential, electron affinity, chemical
hardness, softness, chemical potential and electrophilicity index.

Three practical things are built in rather than left to the caller.  Every job is **cached** by a
key over the molecule, its charge, the level of theory and the geometry, so an interrupted run
resumes instead of starting again -- which matters when a molecule takes minutes and a set takes
hours.  A molecule that fails is **reported and skipped**, never fatal to the set.  And the
geometry is **not optimised** unless asked: a superimposed series that is optimised molecule by
molecule is no longer superimposed, and the fields built from it would be comparing poses that
have nothing to do with each other.

The command-line front end is ``examples/orca_atom_properties.py``; the application runs the same
code from *Tools -> Quantum-chemical properties with ORCA*.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

DEFAULT_LEVEL = "B3LYP D3BJ def2-SVP RIJCOSX def2/J TightSCF"
FAST_LEVEL = "r2SCAN-3c TightSCF"
#: a larger def2 basis with the matching effective core potential, which def2 defines for every
#: element from rubidium on.  Naming ``def2-ECP`` changes nothing for a light molecule and makes
#: the intent explicit for a transition-metal complex (plan item EC1).
HEAVY_LEVEL = "B3LYP D3BJ def2-TZVP def2-ECP RIJCOSX def2/J TightSCF"
#: elements for which a def2 basis uses an effective core potential: rubidium (37) and above.
#: The original def2 sets cover hydrogen to radon but leave out the lanthanides, lanthanum to
#: lutetium (57 to 71), which were published separately; a calculation on one of those needs a
#: basis named by hand.
ECP_FROM_Z = 37
DEFAULT_ORCA = r"C:\ORCA_6.1.0\orca.exe"
HARTREE_EV = 27.211386245988

#: (dielectric constant, refractive index) of solvents ORCA's own CPCM list does not name.  A
#: caller may also pass the pair directly.
SOLVENT_CONSTANTS = {
    "dme": (7.20, 1.3796),            # 1,2-dimethoxyethane (monoglyme)
    "diglyme": (7.23, 1.4097),
    "propylene carbonate": (64.0, 1.4209),
    "ethylene carbonate": (89.78, 1.4158),
    "dimethyl carbonate": (3.087, 1.3687),
}

# per-atom property names written to the SDF (atom double property lists)
ATOM_PROPS_N = ["q_mulliken", "q_loewdin", "q_hirshfeld", "q_chelpg", "homo_pop", "lumo_pop"]
#: Mulliken spin populations, written only when a state is open-shell (plan item EC1).  The name
#: of the field a model builds from them is ``prop:spin``.
ATOM_PROPS_SPIN = ["spin", "spin_anion", "spin_cation"]
ATOM_PROPS_FUKUI = ["fukui_plus", "fukui_minus", "fukui_zero", "dual",
                    "softness_plus", "softness_minus", "local_electrophilicity",
                    "fukui_plus_mulliken", "fukui_minus_mulliken", "fukui_zero_mulliken",
                    "dual_mulliken"]
MOL_PROPS_N = ["E_total", "E_homo", "E_lumo", "gap", "dipole", "charge", "multiplicity"]
MOL_PROPS_SPIN = ["spin_total", "spin_total_anion", "spin_total_cation"]
MOL_PROPS_FUKUI = ["IP", "EA", "hardness", "softness", "electrophilicity", "chemical_potential"]


class OrcaCancelled(RuntimeError):
    """Raised when the caller asked for the run to stop; not a failure of the calculation."""


def _terminate(proc) -> None:
    """Stop an ORCA process and the children it started: ask it to terminate, then kill it."""
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:                                 # noqa: BLE001 - it may already be gone
        try:
            proc.kill()
        except Exception:                             # noqa: BLE001
            pass


class OrcaError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# ORCA input
# ----------------------------------------------------------------------------------------------

def find_orca(explicit: Optional[str] = None) -> str:
    """The ORCA executable: the argument, ``$ORCA_EXE``, the standard location, or the PATH."""
    for cand in (explicit, os.environ.get("ORCA_EXE"), DEFAULT_ORCA):
        if cand and os.path.isfile(cand):
            return cand
    found = shutil.which("orca")
    if found:
        return found
    raise FileNotFoundError("orca.exe not found; pass --orca or set ORCA_EXE")


def orca_has_mpi(exe: str) -> bool:
    """Whether this ORCA build can run ``%pal`` jobs (the serial build lacks the MPI helpers)."""
    d = os.path.dirname(os.path.abspath(exe))
    return any(os.path.isfile(os.path.join(d, f)) for f in ("orca_startup_mpi.exe", "orca_startup_mpi"))


def electron_count(mol: Chem.Mol, charge: int) -> int:
    return sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge


def multiplicity_for(mol: Chem.Mol, charge: int) -> int:
    """Lowest spin state: singlet for an even number of electrons, doublet for an odd one."""
    return 1 if electron_count(mol, charge) % 2 == 0 else 2


def state_from_properties(mol: Chem.Mol) -> Tuple[int, int]:
    """(charge, multiplicity) of a molecule, from its SD properties when it carries them.

    A catalyst is Fe(II) or Fe(I) and a triplet or a quintet, and nothing in a connection table
    says which; a radical electrolyte is a doublet.  So ``charge`` and ``multiplicity`` are read
    from the SD properties of the record when they are given, and only otherwise taken from the
    formal charges and the lowest spin state (plan item EC1).  A multiplicity whose parity does
    not match the electron count is refused rather than silently corrected: it is a mistake in
    the data file, and a wrong spin state gives a plausible number for the wrong state.
    """
    charge = None
    for name in ("charge", "Charge", "total_charge"):
        if mol.HasProp(name):
            charge = int(round(float(mol.GetProp(name))))
            break
    if charge is None:
        charge = Chem.GetFormalCharge(mol)
    mult = None
    for name in ("multiplicity", "Multiplicity", "spin_multiplicity"):
        if mol.HasProp(name):
            mult = int(round(float(mol.GetProp(name))))
            break
    if mult is None:
        return charge, multiplicity_for(mol, charge)
    if mult < 1:
        raise OrcaError(f"multiplicity {mult} is not a spin state")
    n_el = electron_count(mol, charge)
    if (n_el - (mult - 1)) % 2 != 0:
        raise OrcaError(f"multiplicity {mult} does not fit {n_el} electrons at charge {charge}")
    return charge, mult


def ion_multiplicity(mult: int) -> int:
    """The spin state of a state with one electron more or fewer, given the parent's.

    Adding or removing one electron flips the parity of the electron count, so it flips the
    parity of the multiplicity.  The rule here is to pair the new electron where it can be
    paired, that is to take ``mult - 1`` when that is still a spin state and ``mult + 1``
    otherwise.  For a closed-shell parent this gives the doublet the Fukui functions have always
    used; for a quintet parent it gives a quartet, which is the state a caller who wrote
    ``multiplicity 5`` almost always means.
    """
    return mult - 1 if mult > 1 else mult + 1


def needs_ecp(mol: Chem.Mol) -> bool:
    """Whether this molecule holds an element for which a def2 basis uses a core potential."""
    return any(a.GetAtomicNum() >= ECP_FROM_Z for a in mol.GetAtoms())


def solvent_lines(solvent) -> Tuple[List[str], List[str]]:
    """(keywords, input blocks) for a continuum solvent; ``([], [])`` for the gas phase.

    `solvent` is the name of a solvent ORCA's CPCM list knows, a name in
    :data:`SOLVENT_CONSTANTS`, or a (dielectric constant, refractive index) pair.  A name ORCA
    knows becomes the keyword ``CPCM(name)``; the other two become an explicit ``%cpcm`` block,
    because a solvent ORCA cannot name is still usable when its two constants are given.
    """
    if solvent is None:
        return [], []
    if isinstance(solvent, (tuple, list)):
        eps, refrac = float(solvent[0]), float(solvent[1])
    else:
        key = str(solvent).strip().lower()
        if key not in SOLVENT_CONSTANTS:
            return [f"CPCM({str(solvent).strip()})"], []
        eps, refrac = SOLVENT_CONSTANTS[key]
    return ["CPCM"], ["%cpcm", f"  epsilon {eps}", f"  refrac {refrac}", "end"]


def make_input(level: str, charge: int, mult: int, xyz_name: str, nprocs: int = 1,
               opt: bool = False, populations: bool = True, chelpg: bool = True,
               maxcore: int = 2000, solvent=None) -> str:
    """Text of an ORCA input.

    `populations` adds the Hirshfeld analysis and the per-MO Mulliken atom populations; `chelpg`
    the CHELPG fit (needed for the N-electron state only).  An open-shell state is run
    unrestricted; a geometry optimisation prints nothing but what it needs.  `solvent` puts the
    molecule in a continuum solvent, as :func:`solvent_lines` describes (plan item EC1).
    """
    keywords = [level]
    if opt:
        keywords.append("Opt")
    if mult > 1:
        keywords.append("UKS")
    if chelpg and not opt:
        keywords.append("CHELPG")
    solv_keywords, solv_block = solvent_lines(solvent)
    keywords += solv_keywords
    lines = ["! " + " ".join(keywords)]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.append(f"%maxcore {maxcore}")
    lines.append("%scf MaxIter 300 end")
    lines += solv_block
    if populations and not opt:
        lines += ["%output", "  Print[P_Hirshfeld] 1", "  Print[P_AtPopMO_M] 1", "end"]
    lines.append(f"* xyzfile {charge} {mult} {xyz_name}")
    return "\n".join(lines) + "\n"


def xyz_block(symbols: Sequence[str], coords: np.ndarray, comment: str = "") -> str:
    out = [str(len(symbols)), comment]
    for s, (x, y, z) in zip(symbols, coords):
        out.append(f"{s:2s} {x:14.6f} {y:14.6f} {z:14.6f}")
    return "\n".join(out) + "\n"


def read_xyz(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def cache_key(smiles: str, charge: int, level: str, opt: bool, coords: Optional[np.ndarray] = None,
              solvent=None, mult: Optional[int] = None) -> str:
    """Identity of a computation: the molecule, its state, the level and the geometry.

    The SMILES is canonicalised so that the same compound written two ways shares a key; the
    coordinates (to 1e-3 A) are part of it because a different starting geometry is a different
    result even when nothing else changed.

    `solvent` and `mult` are folded in only when they are given.  That keeps the key of a
    gas-phase run at its lowest spin state exactly what it was before plan item EC1 added the
    two, so the ORCA results already cached for the other benchmarks of this repository are not
    thrown away.  A caller passes `mult` when it is not the lowest spin state, which is the only
    case in which two runs of the same molecule in the same charge state differ.
    """
    m = Chem.MolFromSmiles(smiles)
    can = Chem.MolToSmiles(m) if m is not None else smiles
    h = hashlib.sha1()
    h.update(f"{can}|{charge}|{' '.join(level.split())}|{int(bool(opt))}".encode())
    if coords is not None:
        h.update(np.round(np.asarray(coords, float), 3).tobytes())
    if solvent is not None:
        h.update(f"|solvent={solvent}".encode())
    if mult is not None:
        h.update(f"|mult={int(mult)}".encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# ORCA output
# ----------------------------------------------------------------------------------------------

_CHARGE_LINE = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s*:?\s+(-?\d+\.\d+)")
#: the same line of an unrestricted run, which carries the spin population after the charge
_SPIN_LINE = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s*:?\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_ORBITAL_LINE = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
_COORD_LINE = re.compile(r"^\s*([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_FLOAT = re.compile(r"^-?\d+\.\d+$")


def _last_block(lines: List[str], header: str) -> Optional[int]:
    """Index of the line after the last line containing `header`."""
    for i in range(len(lines) - 1, -1, -1):
        if header in lines[i]:
            return i + 1
    return None


def _parse_charges(lines: List[str], header: str) -> Optional[np.ndarray]:
    """Atomic charges from the last block headed `header`: lines ``idx  Sym [:]  q [spin]``."""
    start = _last_block(lines, header)
    if start is None:
        return None
    vals: Dict[int, float] = {}
    for ln in lines[start:start + 20000]:
        m = _CHARGE_LINE.match(ln)
        if m:
            vals[int(m.group(1))] = float(m.group(3))
        elif vals:
            break
    if not vals or sorted(vals) != list(range(len(vals))):
        return None
    return np.array([vals[i] for i in range(len(vals))])


def _parse_spin(lines: List[str], header: str) -> Optional[np.ndarray]:
    """Atomic spin populations from the last block headed `header`.

    An unrestricted run prints two numbers on each atom's line, the charge and the spin
    population, under the heading ``... CHARGES AND SPIN POPULATIONS``; a closed-shell run prints
    one number and no spin.  So the second column is what is read here, and its absence is the
    answer "this state has no spin density" rather than a failure (plan item EC1).
    """
    start = _last_block(lines, header)
    if start is None:
        return None
    vals: Dict[int, float] = {}
    for ln in lines[start:start + 20000]:
        m = _SPIN_LINE.match(ln)
        if m:
            vals[int(m.group(1))] = float(m.group(4))
        elif _CHARGE_LINE.match(ln):
            return None                 # a charges-only block: this state is closed-shell
        elif vals:
            break
    if not vals or sorted(vals) != list(range(len(vals))):
        return None
    return np.array([vals[i] for i in range(len(vals))])


def _parse_orbitals(lines: List[str]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """(occupation, energy/Eh) of the orbitals in the last ORBITAL ENERGIES block, alpha and beta.

    For a closed-shell run everything is in the first list.  ORCA prints only the first ten
    virtuals, which is all the LUMO needs.
    """
    start = _last_block(lines, "ORBITAL ENERGIES")
    alpha: List[Tuple[float, float]] = []
    beta: List[Tuple[float, float]] = []
    if start is None:
        return alpha, beta
    cur = alpha
    for ln in lines[start:start + 50000]:
        m = _ORBITAL_LINE.match(ln)
        if m:
            cur.append((float(m.group(2)), float(m.group(3))))
        elif "SPIN DOWN" in ln:
            cur = beta
        elif ("SPIN UP" in ln or not ln.strip() or ln.startswith("---") or "NO   OCC" in ln
              or ln.lstrip().startswith("*Only")):
            continue
        else:
            break
    return alpha, beta


def _parse_mo_populations(lines: List[str]) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float]]:
    """The last MULLIKEN ATOM POPULATIONS PER MO block (alpha spin if unrestricted).

    Returns ({mo: {atom: fraction}}, {mo: occupation}); ORCA prints percentages in chunks of six
    orbitals, and may leave out an atom whose share is below its printing threshold, so the rows
    are keyed by atom index rather than counted.
    """
    start = _last_block(lines, "MULLIKEN ATOM POPULATIONS PER MO")
    pops: Dict[int, Dict[int, float]] = {}
    occs: Dict[int, float] = {}
    if start is None:
        return pops, occs
    mos: List[int] = []
    pending: List[List[float]] = []     # energies then occupations of the current chunk
    for ln in lines[start:start + 200000]:
        s = ln.strip()
        if not s or s.startswith("THRESHOLD") or s.startswith("SPIN UP"):
            continue
        if s.startswith("SPIN DOWN") or s.startswith("*****") or re.fullmatch(r"-{5,}", s):
            if s.startswith("SPIN DOWN") or pops:
                break
            continue
        toks = s.split()
        if all(t.isdigit() for t in toks):
            mos = [int(t) for t in toks]
            pending = []
            for mo in mos:
                pops.setdefault(mo, {})
            continue
        if set(s) <= set("- "):
            continue
        if all(_FLOAT.match(t) for t in toks) and len(toks) == len(mos):
            pending.append([float(t) for t in toks])
            if len(pending) == 2:
                for mo, occ in zip(mos, pending[1]):
                    occs[mo] = occ
            continue
        if len(toks) >= 3 and toks[0].isdigit() and toks[1].isalpha() and mos:
            atom = int(toks[0])
            for mo, t in zip(mos, toks[2:]):
                pops[mo][atom] = float(t) / 100.0
            continue
        break
    return pops, occs


def _parse_coords(lines: List[str]) -> Tuple[List[str], Optional[np.ndarray]]:
    start = _last_block(lines, "CARTESIAN COORDINATES (ANGSTROEM)")
    if start is None:
        return [], None
    syms, xyz = [], []
    for ln in lines[start:start + 100000]:
        m = _COORD_LINE.match(ln)
        if m:
            syms.append(m.group(1))
            xyz.append([float(m.group(k)) for k in (2, 3, 4)])
        elif syms:
            break
    return syms, (np.array(xyz) if xyz else None)


_FREQUENCY_LINE = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1")
_IR_LINE = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def _parse_frequencies(lines: List[str]) -> Optional[np.ndarray]:
    """The ``VIBRATIONAL FREQUENCIES`` block: one value per mode, in cm^-1 (plan item SP2)."""
    start = _last_block(lines, "VIBRATIONAL FREQUENCIES")
    if start is None:
        return None
    vals: Dict[int, float] = {}
    for ln in lines[start:start + 20000]:
        m = _FREQUENCY_LINE.match(ln)
        if m:
            vals[int(m.group(1))] = float(m.group(2))
        elif vals:
            break
    if not vals or sorted(vals) != list(range(len(vals))):
        return None
    return np.array([vals[i] for i in range(len(vals))])


def _parse_normal_modes(lines: List[str]) -> Optional[np.ndarray]:
    """The ``NORMAL MODES`` block as a (3N, modes) array.

    ORCA prints the modes in blocks of six columns: a line of mode numbers, then 3N lines of a
    row index followed by one value per column.  The values are the mass-weighted Cartesian
    displacements, normalised but not orthogonal, as the block's own header says.
    """
    start = _last_block(lines, "NORMAL MODES")
    if start is None:
        return None
    cols: Dict[int, List[float]] = {}
    header: Optional[List[int]] = None
    for ln in lines[start:start + 400000]:
        parts = ln.split()
        if not parts:
            if cols:
                break
            continue
        if all(p.isdigit() for p in parts) and len(parts) <= 6:
            header = [int(p) for p in parts]
            continue
        if (header is not None and parts[0].isdigit() and len(parts) == len(header) + 1
                and all(_FLOAT.match(p) for p in parts[1:])):
            for h, v in zip(header, parts[1:]):
                cols.setdefault(h, []).append(float(v))
            continue
        if cols:
            break
    if not cols or sorted(cols) != list(range(len(cols))):
        return None
    n_rows = {len(v) for v in cols.values()}
    if len(n_rows) != 1:
        return None
    return np.array([cols[k] for k in range(len(cols))], float).T


def _parse_ir_intensities(lines: List[str], n_modes: int) -> Optional[np.ndarray]:
    """The ``IR SPECTRUM`` block: the intensity in km/mol of every mode, zero where none is
    printed (the translations and rotations)."""
    start = _last_block(lines, "IR SPECTRUM")
    if start is None:
        return None
    out = np.zeros(int(n_modes))
    seen = False
    for ln in lines[start:start + 20000]:
        m = _IR_LINE.match(ln)
        if m:
            k = int(m.group(1))
            if 0 <= k < n_modes:
                out[k] = float(m.group(4))
                seen = True
        elif seen:
            break
    return out if seen else None


@dataclass
class OrcaResult:
    terminated: bool = False
    converged: bool = False                 # geometry optimisation only
    energy: float = float("nan")            # Eh
    dipole: float = float("nan")            # Debye
    mulliken: Optional[np.ndarray] = None
    loewdin: Optional[np.ndarray] = None
    hirshfeld: Optional[np.ndarray] = None
    chelpg: Optional[np.ndarray] = None
    #: Mulliken spin populations; None for a closed-shell state (plan item EC1)
    spin: Optional[np.ndarray] = None
    homo: float = float("nan")              # Eh
    lumo: float = float("nan")              # Eh
    homo_index: int = -1
    lumo_index: int = -1
    mo_populations: Dict[int, Dict[int, float]] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    coords: Optional[np.ndarray] = None
    #: harmonic frequencies in cm^-1 as ORCA lists them, the zero ones of translation and
    #: rotation included; None when the job computed none (plan item SP2)
    frequencies: Optional[np.ndarray] = None
    #: the normal modes, one column per frequency and 3N rows of mass-weighted Cartesian
    #: displacement, in ORCA's normalisation
    modes: Optional[np.ndarray] = None
    #: infrared intensities in km/mol, one per frequency, zero where ORCA printed none
    ir_intensities: Optional[np.ndarray] = None

    @property
    def n_atoms(self) -> int:
        return len(self.mulliken) if self.mulliken is not None else len(self.symbols)

    def mo_population(self, mo: int, n_atoms: int) -> np.ndarray:
        """Fraction of orbital `mo` on every atom (0 where ORCA printed nothing)."""
        d = self.mo_populations.get(mo, {})
        return np.array([d.get(i, 0.0) for i in range(n_atoms)])


def parse_orca_output(text: str) -> OrcaResult:
    """Everything the script needs from an ORCA output, each from the last occurrence of its block."""
    lines = text.splitlines()
    r = OrcaResult()
    r.terminated = "ORCA TERMINATED NORMALLY" in text
    r.converged = "THE OPTIMIZATION HAS CONVERGED" in text
    energies = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    if energies:
        r.energy = float(energies[-1])
    dip = re.findall(r"Magnitude \(Debye\)\s*:\s*(-?\d+\.\d+)", text)
    if dip:
        r.dipole = float(dip[-1])
    r.mulliken = _parse_charges(lines, "MULLIKEN ATOMIC CHARGES")
    r.loewdin = _parse_charges(lines, "LOEWDIN ATOMIC CHARGES")
    r.hirshfeld = _parse_charges(lines, "HIRSHFELD ANALYSIS")
    r.chelpg = _parse_charges(lines, "CHELPG Charges")
    r.spin = _parse_spin(lines, "MULLIKEN ATOMIC CHARGES")
    alpha, beta = _parse_orbitals(lines)
    occ = ([(i, e) for i, (o, e) in enumerate(alpha) if o > 0.5]
           + [(i, e) for i, (o, e) in enumerate(beta) if o > 0.5])
    virt = ([(i, e) for i, (o, e) in enumerate(alpha) if o <= 0.5]
            + [(i, e) for i, (o, e) in enumerate(beta) if o <= 0.5])
    if occ:
        r.homo_index, r.homo = max(occ, key=lambda t: t[1])
    if virt:
        r.lumo_index, r.lumo = min(virt, key=lambda t: t[1])
    r.mo_populations, _ = _parse_mo_populations(lines)
    r.symbols, r.coords = _parse_coords(lines)
    r.frequencies = _parse_frequencies(lines)
    if r.frequencies is not None:
        modes = _parse_normal_modes(lines)
        if modes is not None and modes.shape[1] == len(r.frequencies):
            r.modes = modes
        r.ir_intensities = _parse_ir_intensities(lines, len(r.frequencies))
    return r


# ----------------------------------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------------------------------

def derive_properties(n: OrcaResult, np1: Optional[OrcaResult] = None, nm1: Optional[OrcaResult] = None,
                      charge: int = 0, mult: int = 1) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Per-atom and global descriptors from the N-electron result and, if given, the N+1 / N-1 ones.

    Returns (atom properties, molecule properties); the Fukui-derived entries are present only when
    both ionic states are given.  Energies in eV, softness in 1/eV, dipole in Debye, charges in e.
    """
    if n.hirshfeld is None or n.mulliken is None:
        raise OrcaError("the N-electron output lacks the Mulliken or Hirshfeld charges")
    na = len(n.hirshfeld)
    atom: Dict[str, np.ndarray] = {
        "q_mulliken": n.mulliken,
        "q_loewdin": n.loewdin if n.loewdin is not None else np.full(na, np.nan),
        "q_hirshfeld": n.hirshfeld,
        "q_chelpg": n.chelpg if n.chelpg is not None else np.full(na, np.nan),
        "homo_pop": n.mo_population(n.homo_index, na),
        "lumo_pop": n.mo_population(n.lumo_index, na),
    }
    mol: Dict[str, float] = {
        "E_total": n.energy,
        "E_homo": n.homo * HARTREE_EV,
        "E_lumo": n.lumo * HARTREE_EV,
        "gap": (n.lumo - n.homo) * HARTREE_EV,
        "dipole": n.dipole,
        "charge": float(charge),
        "multiplicity": float(mult),
    }
    # The spin population is where the unpaired electron sits. For a closed-shell parent it is
    # zero everywhere and is not written; for the radical ion of a redox series it is the field
    # that says which atoms carry the added or removed electron (plan item EC1).
    if n.spin is not None and len(n.spin) == na:
        atom["spin"] = n.spin
        mol["spin_total"] = float(np.sum(n.spin))
    if np1 is None or nm1 is None:
        return atom, mol
    for other, what in ((np1, "N+1"), (nm1, "N-1")):
        if other.hirshfeld is None or other.mulliken is None or len(other.hirshfeld) != na:
            raise OrcaError(f"the {what} output lacks usable charges")
    f_plus = n.hirshfeld - np1.hirshfeld
    f_minus = nm1.hirshfeld - n.hirshfeld
    f_plus_m = n.mulliken - np1.mulliken
    f_minus_m = nm1.mulliken - n.mulliken
    ip = (nm1.energy - n.energy) * HARTREE_EV
    ea = (n.energy - np1.energy) * HARTREE_EV
    hardness = (ip - ea) / 2.0
    softness = 1.0 / (2.0 * hardness) if hardness != 0 else float("nan")
    mu = -(ip + ea) / 2.0
    omega = mu ** 2 / (2.0 * hardness) if hardness != 0 else float("nan")
    atom.update({
        "fukui_plus": f_plus, "fukui_minus": f_minus, "fukui_zero": (f_plus + f_minus) / 2.0,
        "dual": f_plus - f_minus,
        "softness_plus": softness * f_plus, "softness_minus": softness * f_minus,
        "local_electrophilicity": omega * f_plus,
        "fukui_plus_mulliken": f_plus_m, "fukui_minus_mulliken": f_minus_m,
        "fukui_zero_mulliken": (f_plus_m + f_minus_m) / 2.0, "dual_mulliken": f_plus_m - f_minus_m,
    })
    mol.update({"IP": ip, "EA": ea, "hardness": hardness, "softness": softness,
                "electrophilicity": omega, "chemical_potential": mu})
    # Both ionic states of a closed-shell molecule are radicals, so each has a spin density, and
    # which atoms carry it is what a redox series is about.
    for other, which in ((np1, "anion"), (nm1, "cation")):
        if other.spin is not None and len(other.spin) == na:
            atom[f"spin_{which}"] = other.spin
            mol[f"spin_total_{which}"] = float(np.sum(other.spin))
    return atom, mol


# ----------------------------------------------------------------------------------------------
# the SDF
# ----------------------------------------------------------------------------------------------

def annotate(mol: Chem.Mol, atom_props: Dict[str, np.ndarray], mol_props: Dict[str, float],
             coords: Optional[np.ndarray] = None, extra: Optional[Dict[str, str]] = None) -> Chem.Mol:
    """A copy of `mol` carrying the properties (and `coords`, when the geometry was optimised).

    Atom properties go through RDKit's property lists, which is the only way they reach the SDF
    and are read back as atom properties (``atom.dprop.<name>``); the molecule's own properties
    (activity, name, ...) are kept.
    """
    m = Chem.Mol(mol)
    if coords is not None:
        if m.GetNumConformers() == 0 or len(coords) != m.GetNumAtoms():
            raise OrcaError("coordinate count does not match the molecule")
        conf = m.GetConformer()
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    n = m.GetNumAtoms()
    for name, vals in atom_props.items():
        if len(vals) != n:
            raise OrcaError(f"{name}: {len(vals)} values for {n} atoms")
        for a, v in zip(m.GetAtoms(), vals):
            a.SetDoubleProp(name, float(v))
        Chem.CreateAtomDoublePropertyList(m, name)
    for name, v in mol_props.items():
        if name in ("charge", "multiplicity"):
            m.SetProp(name, str(int(round(v))))
        else:
            m.SetProp(name, f"{v:.6f}")
    for k, v in (extra or {}).items():
        m.SetProp(k, str(v))
    return m


def verify_sdf(path: str, expect: Sequence[str]) -> List[str]:
    """Load the file as the program does and return the per-atom properties it offers."""
    from cmfqsar.dataio import load_dataset
    ds = load_dataset(path)
    missing = [p for p in expect if p not in ds.atom_properties]
    if missing:
        raise OrcaError(f"{path}: atom properties not seen by the loader: {missing}")
    return ds.atom_properties


# ----------------------------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------------------------

@dataclass
class Config:
    orca: str = DEFAULT_ORCA
    level: str = DEFAULT_LEVEL
    opt: bool = False
    fukui: bool = True
    #: a continuum solvent for every job: the name of one ORCA knows, a name in
    #: :data:`SOLVENT_CONSTANTS`, or a (dielectric constant, refractive index) pair.  None is the
    #: gas phase, which is what every benchmark before plan item EC1 used.
    solvent: Optional[object] = None
    nprocs: int = 1
    maxcore: int = 2000
    timeout: Optional[float] = None
    workdir: str = "orca_work"
    #: asked between polls of a running job; when it returns True the run stops and the
    #: molecules finished so far are kept
    cancel: Optional[object] = None


def safe_name(name: str, limit: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_.")
    return (s or "mol")[:limit]


_SCRATCH_FILE = re.compile(r"\.(tmp(\.\d+)?|bas\d+|hostnames)$")


def _output_ok(path: str, need_converged: bool) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, errors="replace") as fh:
        text = fh.read()
    return ("ORCA TERMINATED NORMALLY" in text
            and (not need_converged or "THE OPTIMIZATION HAS CONVERGED" in text))


def run_job(jobdir: str, job: str, inp: str, cfg: Config, need_converged: bool = False) -> OrcaResult:
    """Run (or reuse) the ORCA job `job` in `jobdir`; returns the parsed output or raises OrcaError."""
    inp_path = os.path.join(jobdir, job + ".inp")
    out_path = os.path.join(jobdir, job + ".out")
    same = False
    if os.path.isfile(inp_path):
        with open(inp_path) as fh:
            same = fh.read() == inp
        if not same and os.path.isfile(out_path):
            os.remove(out_path)
    if not same:
        with open(inp_path, "w", newline="\n") as fh:
            fh.write(inp)
    if not _output_ok(out_path, need_converged):
        with open(out_path, "w") as fh:
            # Popen rather than run, so that a cancelled job stops ORCA itself: a quantum
            # calculation can take minutes, and a Cancel that only stops the bookkeeping while
            # the process keeps the cores busy is not a cancel at all.
            proc = subprocess.Popen([cfg.orca, job + ".inp"], cwd=jobdir, stdout=fh,
                                    stderr=subprocess.STDOUT)
            deadline = None if not cfg.timeout else time.time() + float(cfg.timeout)
            while True:
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cfg.cancel is not None and cfg.cancel():
                    _terminate(proc)
                    raise OrcaCancelled(f"{job}: cancelled")
                if deadline is not None and time.time() > deadline:
                    _terminate(proc)
                    raise OrcaError(f"{job}: ORCA exceeded {cfg.timeout:.0f} s")
        for f in os.listdir(jobdir):      # ORCA's scratch files; the .gbw, densities and output stay
            if f.startswith(job + ".") and _SCRATCH_FILE.search(f):
                try:
                    os.remove(os.path.join(jobdir, f))
                except OSError:
                    pass
    with open(out_path, errors="replace") as fh:
        text = fh.read()
    res = parse_orca_output(text)
    if not res.terminated:
        tail = " | ".join(ln.strip() for ln in text.splitlines()[-12:] if ln.strip())[:400]
        raise OrcaError(f"{job}: ORCA did not terminate normally: {tail}")
    if need_converged and not res.converged:
        raise OrcaError(f"{job}: the optimisation did not converge")
    return res


def prepare_molecule(mol: Chem.Mol) -> Chem.Mol:
    """All hydrogens explicit, with coordinates; what ORCA is given and what the SDF will hold."""
    if mol.GetNumConformers() == 0:
        from cmfqsar.align import embed_3d
        return embed_3d(Chem.RemoveHs(mol))
    if any(a.GetTotalNumHs() > 0 for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)
    return mol


@dataclass
class MoleculeOutcome:
    index: int
    name: str
    mol: Optional[Chem.Mol] = None          # annotated, None on failure
    error: str = ""
    seconds: float = 0.0
    reused: bool = False


def run_molecule(mol: Chem.Mol, name: str, index: int, cfg: Config) -> MoleculeOutcome:
    """The whole chain for one molecule: geometry, ORCA jobs, parsing, the annotated copy."""
    t0 = time.time()
    outcome = MoleculeOutcome(index=index, name=name)
    status_path = None
    key = None
    try:
        mol = prepare_molecule(mol)
        charge, mult = state_from_properties(mol)
        lowest = multiplicity_for(mol, charge)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        coords0 = mol.GetConformer().GetPositions()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        key = cache_key(smiles, charge, cfg.level, cfg.opt, coords0, solvent=cfg.solvent,
                        mult=None if mult == lowest else mult)
        jobdir = os.path.join(cfg.workdir, f"{index:03d}_{safe_name(name)}")
        status_path = os.path.join(jobdir, "status.json")
        status = {}
        if os.path.isfile(status_path):
            with open(status_path) as fh:
                status = json.load(fh)
        if status.get("key") != key and os.path.isdir(jobdir):
            # the directory belongs to another computation (or another molecule); start again
            shutil.rmtree(jobdir)
            status = {}
        os.makedirs(jobdir, exist_ok=True)
        with open(os.path.join(jobdir, "geom.xyz"), "w", newline="\n") as fh:
            fh.write(xyz_block(symbols, coords0, f"{name} charge={charge} mult={mult} (input geometry)"))
        outcome.reused = status.get("status") == "done"

        xyz = "geom.xyz"
        coords = coords0
        if cfg.opt:
            inp = make_input(cfg.level, charge, mult, "geom.xyz", cfg.nprocs, opt=True,
                             populations=False, chelpg=False, maxcore=cfg.maxcore,
                             solvent=cfg.solvent)
            run_job(jobdir, "opt", inp, cfg, need_converged=True)
            osyms, coords = read_xyz(os.path.join(jobdir, "opt.xyz"))
            if [s.upper() for s in osyms] != [s.upper() for s in symbols]:
                raise OrcaError("the optimised geometry lists different atoms")
            xyz = "opt.xyz"
        res_n = run_job(jobdir, "sp_n", make_input(cfg.level, charge, mult, xyz, cfg.nprocs,
                                                   maxcore=cfg.maxcore, solvent=cfg.solvent), cfg)
        if res_n.n_atoms != len(symbols):
            raise OrcaError(f"ORCA reports {res_n.n_atoms} atoms, the molecule has {len(symbols)}")
        res_p = res_m = None
        if cfg.fukui:
            # The ionic states take their spin from the parent's: for a closed-shell molecule
            # that is the doublet the Fukui functions have always used, and for an open-shell
            # parent it is the state one electron up or down from the one the caller asked for.
            ion_mult = ion_multiplicity(mult)
            res_p = run_job(jobdir, "sp_np1",
                            make_input(cfg.level, charge - 1, ion_mult, xyz,
                                       cfg.nprocs, chelpg=False, maxcore=cfg.maxcore,
                                       solvent=cfg.solvent), cfg)
            res_m = run_job(jobdir, "sp_nm1",
                            make_input(cfg.level, charge + 1, ion_mult, xyz,
                                       cfg.nprocs, chelpg=False, maxcore=cfg.maxcore,
                                       solvent=cfg.solvent), cfg)
        atom_props, mol_props = derive_properties(res_n, res_p, res_m, charge=charge, mult=mult)
        extra = {"qm_level": cfg.level,
                 "qm_geometry": "orca-opt" if cfg.opt else "input",
                 "qm_solvent": "gas" if cfg.solvent is None else str(cfg.solvent),
                 "qm_dir": os.path.basename(jobdir)}
        outcome.mol = annotate(mol, atom_props, mol_props, coords=coords if cfg.opt else None, extra=extra)
        outcome.seconds = time.time() - t0
        with open(os.path.join(jobdir, "results.json"), "w") as fh:
            json.dump({"name": name, "smiles": smiles, "symbols": symbols,
                       "atom": {k: [float(x) for x in v] for k, v in atom_props.items()},
                       "molecule": mol_props}, fh, indent=1)
        seconds = status.get("seconds", outcome.seconds) if outcome.reused else outcome.seconds
        with open(status_path, "w") as fh:
            json.dump({"key": key, "status": "done", "name": name, "smiles": smiles, "charge": charge,
                       "multiplicity": mult, "level": cfg.level, "opt": cfg.opt,
                       "solvent": "gas" if cfg.solvent is None else str(cfg.solvent),
                       "fukui": cfg.fukui, "seconds": seconds},
                      fh, indent=1)
    except Exception as exc:        # a failed molecule is reported, not fatal
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.seconds = time.time() - t0
        if status_path:
            try:
                with open(status_path, "w") as fh:
                    json.dump({"key": key, "status": "failed", "name": name, "error": outcome.error}, fh, indent=1)
            except Exception:
                pass
    return outcome


def process_dataset(path: str, out: str, cfg: Config, limit: Optional[int] = None, jobs: int = 1,
                    log=print) -> List[MoleculeOutcome]:
    """Run every molecule of `path` and write the annotated SDF `out`; returns the outcomes."""
    from cmfqsar.dataio import read_molecules
    mols, names, _ = read_molecules(path)
    if limit:
        mols, names = mols[:limit], names[:limit]
    os.makedirs(cfg.workdir, exist_ok=True)
    log(f"{len(mols)} molecules from {path}; level '{cfg.level}', "
        f"{'optimised' if cfg.opt else 'input'} geometry, Fukui {'on' if cfg.fukui else 'off'}, "
        f"nprocs {cfg.nprocs}, {jobs} molecule(s) at a time -> {cfg.workdir}")

    def work(k):
        oc = run_molecule(mols[k], names[k], k + 1, cfg)
        if oc.error:
            log(f"[{k + 1}/{len(mols)}] {oc.name}: FAILED after {oc.seconds:.0f} s -- {oc.error}")
        else:
            log(f"[{k + 1}/{len(mols)}] {oc.name}: {'reused' if oc.reused else 'done'} in {oc.seconds:.0f} s")
        return oc

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            outcomes = list(ex.map(work, range(len(mols))))
    else:
        outcomes = [work(k) for k in range(len(mols))]

    good = [oc for oc in outcomes if oc.mol is not None]
    if good:
        w = Chem.SDWriter(out)
        for oc in good:
            w.write(oc.mol)
        w.close()
        expect = ATOM_PROPS_N + (ATOM_PROPS_FUKUI if cfg.fukui else [])
        seen = verify_sdf(out, expect)
        log(f"{len(good)} molecules written to {out}; per-atom properties: {', '.join(seen)}")
    failed = [oc for oc in outcomes if oc.mol is None]
    if failed:
        log(f"{len(failed)} molecule(s) failed and were left out: " + ", ".join(oc.name for oc in failed))
    fresh = [oc.seconds for oc in good if not oc.reused]
    if fresh:
        log(f"wall time per molecule: mean {np.mean(fresh):.0f} s, min {min(fresh):.0f} s, "
            f"max {max(fresh):.0f} s ({len(fresh)} computed; {jobs} at a time)")
    return outcomes


def annotate_molecules(mols: Sequence[Chem.Mol], names: Sequence[str], cfg: Config,
                       jobs: int = 1, indices: Optional[Sequence[int]] = None,
                       progress=None, cancel=None) -> List[MoleculeOutcome]:
    """Run ORCA over molecules already in memory and return the annotated copies.

    This is what the application calls: :func:`process_dataset` reads and writes files, which a
    window cannot do while the user watches.  `indices` restricts the run to part of the set --
    the selected molecules, or the ones still missing their properties -- and `progress` is
    called with a message and a fraction so the window can show where the run is.

    Cancelling stops the ORCA process that is running and returns what is finished; because every
    job is cached, starting again continues from there rather than from the beginning.
    """
    idx = list(range(len(mols))) if indices is None else [int(i) for i in indices]
    cfg = replace(cfg, cancel=cancel) if cancel is not None else cfg
    os.makedirs(cfg.workdir, exist_ok=True)
    done = [0]
    lock = threading.Lock()

    def work(k: int) -> MoleculeOutcome:
        if cancel is not None and cancel():
            return MoleculeOutcome(index=k + 1, name=names[k], error="cancelled")
        oc = run_molecule(mols[k], names[k], k + 1, cfg)
        with lock:
            done[0] += 1
            if progress is not None:
                state = "failed" if oc.error else ("reused" if oc.reused else "done")
                progress(f"ORCA {done[0]}/{len(idx)}: {names[k]} {state}", done[0] / max(len(idx), 1))
        return oc

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(work, idx))
    return [work(k) for k in idx]


def missing_properties(mols: Sequence[Chem.Mol], fukui: bool = True) -> List[int]:
    """Which molecules do not carry the per-atom properties a run would give them."""
    want = ATOM_PROPS_N[0]
    out = []
    for i, m in enumerate(mols):
        if m is None:
            continue
        if not m.HasProp(f"atom.dprop.{want}"):
            out.append(i)
    return out


def estimate_seconds(n_molecules: int, fukui: bool = True, jobs: int = 1,
                     per_job: float = 150.0) -> float:
    """A rough wall time for a run, so that a dialog can say so before it starts.

    `per_job` is the time one single-point calculation takes on this machine; the Fukui indices
    need the N+1 and N-1 states as well, which triples it.
    """
    per_molecule = per_job * (3 if fukui else 1)
    return per_molecule * max(1, n_molecules) / max(1, jobs)


def measured_seconds_per_job(workdir: str, fukui: bool = True) -> Optional[float]:
    """The mean time one ORCA job took in `workdir`, from the status files a previous run left.

    A dialog that has to warn about the cost is more convincing with a number measured on this
    machine than with a guess, and after one run the number is available.
    """
    times = []
    try:
        entries = sorted(os.listdir(workdir))
    except OSError:
        return None
    for entry in entries:
        path = os.path.join(workdir, entry, "status.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            continue
        if st.get("status") != "done" or not st.get("seconds"):
            continue
        jobs = (3 if st.get("fukui", fukui) else 1) + (1 if st.get("opt") else 0)
        times.append(float(st["seconds"]) / jobs)
    return float(np.mean(times)) if times else None


# ----------------------------------------------------------------------------------------------
# A vibrational band as a local target (plan item SP2)
# ----------------------------------------------------------------------------------------------

#: the factor a harmonic frequency is multiplied by before it is compared with a measured band,
#: by the first keyword of the level of theory (case does not matter).  The values are the
#: conventional ones for these methods, close to 0.97 for the composite r2SCAN-3c and 0.96 for
#: B3LYP in a small basis; they correct the harmonic approximation and the method together.  A
#: level not listed is not scaled.  The factor multiplies every band of a series by the same
#: number, so no comparison between descriptors depends on it.
FREQUENCY_SCALE = {"r2scan-3c": 0.97, "b3lyp": 0.96, "pbeh-3c": 0.95, "b97-3c": 0.98}


def frequency_scale(level: str) -> float:
    """The scaling factor of :data:`FREQUENCY_SCALE` for this level of theory, or 1."""
    first = (level.split() or [""])[0].lower()
    return float(FREQUENCY_SCALE.get(first, 1.0))


def make_frequency_input(level: str, charge: int, mult: int, xyz_name: str, nprocs: int = 1,
                         maxcore: int = 2000, solvent=None, opt: bool = True,
                         calc_hess: bool = False, max_cycles: int = 0) -> str:
    """Text of an ORCA input that computes the harmonic frequencies, after a geometry
    optimisation unless `opt` is False.

    A frequency on a geometry that is not a minimum of the same level is not defined, so the
    optimisation is the default; a caller that already holds a minimum of this level passes
    ``opt=False``.  `calc_hess` starts the optimisation from a computed Hessian instead of a
    model one, and `max_cycles` replaces ORCA's limit of optimisation cycles; both are for a
    molecule with a very soft motion, where the default optimisation does not settle.
    """
    keywords = [level] + (["Opt"] if opt else []) + ["Freq"]
    if mult > 1:
        keywords.append("UKS")
    solv_keywords, solv_block = solvent_lines(solvent)
    keywords += solv_keywords
    lines = ["! " + " ".join(keywords)]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.append(f"%maxcore {maxcore}")
    lines.append("%scf MaxIter 300 end")
    if opt and (calc_hess or max_cycles):
        words = (["Calc_Hess true"] if calc_hess else []) \
            + ([f"MaxIter {int(max_cycles)}"] if max_cycles else [])
        lines.append("%geom " + " ".join(words) + " end")
    lines += solv_block
    lines.append(f"* xyzfile {charge} {mult} {xyz_name}")
    return "\n".join(lines) + "\n"


CONTINUE_CYCLES = 200       # the limit of cycles of a continued optimisation
_CYCLE_LIMIT = "The optimization did not converge but reached the maximum"


def optimisation_hit_cycle_limit(jobdir: str, job: str, inp: str) -> bool:
    """True when the job `job` of `jobdir` was run with the input `inp` and its optimisation
    stopped at ORCA's limit of cycles, with the last geometry on disk.

    This is the one failure of an optimisation that is worth continuing, and only for a
    molecule that is still whole: with a very soft motion (a nearly free rotation, for example)
    the forces are small and the steps are not, and a model Hessian does not know how soft the
    motion is.  A start geometry that is wrong reaches the same limit for another reason, and
    more cycles do not repair it; :func:`changed_bonds` tells the two cases apart."""
    paths = [os.path.join(jobdir, job + ext) for ext in (".inp", ".out", ".xyz")]
    if not all(os.path.isfile(p) for p in paths):
        return False
    with open(paths[0]) as fh:
        if fh.read() != inp:
            return False
    with open(paths[1], errors="replace") as fh:
        return _CYCLE_LIMIT in fh.read()


#: a bond is whole when its length is between these multiples of the sum of the covalent radii
BOND_RATIO_LIMITS = (0.6, 1.5)


def bond_length_ratios(mol: Chem.Mol, coords: Optional[np.ndarray] = None) -> np.ndarray:
    """The length of every bond of `mol` divided by the sum of the covalent radii of its two
    atoms, in the order of the bonds.  `coords` replaces the conformer of the molecule.

    A single bond is near 1, a double or triple bond near 0.8.  A value far from 1 means that
    the coordinates do not show the molecule the bond table describes."""
    r = np.asarray(mol.GetConformer().GetPositions() if coords is None else coords, float)
    table = Chem.GetPeriodicTable()
    out = []
    for b in mol.GetBonds():
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        reference = table.GetRcovalent(a.GetAtomicNum()) + table.GetRcovalent(c.GetAtomicNum())
        out.append(float(np.linalg.norm(r[a.GetIdx()] - r[c.GetIdx()])) / reference)
    return np.asarray(out, float)


def changed_bonds(mol: Chem.Mol, coords: Optional[np.ndarray] = None,
                  low: float = BOND_RATIO_LIMITS[0],
                  high: float = BOND_RATIO_LIMITS[1]) -> List[str]:
    """The bonds of `mol` whose length is outside `low` to `high` times the sum of the covalent
    radii, each as a text such as ``S1-F4 2.62``; an empty list when the molecule is whole.

    The use is a check after a geometry optimisation.  An optimisation that starts from a wrong
    geometry can end in another molecule or in fragments, and its frequencies are then not the
    frequencies of the molecule that was asked for."""
    ratios = bond_length_ratios(mol, coords)
    out = []
    for b, q in zip(mol.GetBonds(), ratios):
        if not low <= q <= high:
            a, c = b.GetBeginAtom(), b.GetEndAtom()
            out.append(f"{a.GetSymbol()}{a.GetIdx() + 1}-{c.GetSymbol()}{c.GetIdx() + 1} {q:.2f}")
    return out


#: the limits of :func:`geometry_faults`: the range of a bond length as a multiple of the sum of
#: the covalent radii, and the shortest distance in angstrom of two atoms without a bond
STRUCTURE_LIMITS = {"low": 0.7, "high": 1.3, "contact": 1.3}


def geometry_faults(mol: Chem.Mol, coords: Optional[np.ndarray] = None,
                    low: float = STRUCTURE_LIMITS["low"], high: float = STRUCTURE_LIMITS["high"],
                    contact: float = STRUCTURE_LIMITS["contact"]) -> List[str]:
    """What is wrong with the coordinates of `mol` as a structure of that molecule; an empty
    list when nothing is.

    Two tests.  Every bond length has to be between `low` and `high` times the sum of the
    covalent radii of its two atoms (:func:`bond_length_ratios`); a bond outside the range is
    listed with its length and that ratio, as ``bond S8-F10 2.46 A (1.52 of the covalent
    radii)``.  And no two atoms without a bond between them may be closer than `contact`
    angstrom; the closest such pair is listed as ``no bond H24-F27 0.86 A``.  Two hydrogens on
    one carbon are 1.75 A apart, so 1.3 A is a distance that no pair of atoms of a real molecule
    has.

    The use is a check of a built structure before it is written to a file.  A force field
    without parameters for a group, an embedding that puts a counter-ion on the molecule and an
    optimisation that stops at a linear nitrogen all pass a model fit unnoticed, and all fail
    here."""
    r = np.asarray(mol.GetConformer().GetPositions() if coords is None else coords, float)
    if len(r) < 2:
        return []
    atoms = list(mol.GetAtoms())
    radius = np.array([_covalent_radius(a.GetAtomicNum()) for a in atoms])
    d = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2)
    bonded = Chem.GetAdjacencyMatrix(mol).astype(bool)

    def label(i: int, j: int) -> str:
        return f"{atoms[i].GetSymbol()}{i + 1}-{atoms[j].GetSymbol()}{j + 1}"

    out = []
    first, second = np.nonzero(np.triu(bonded, 1))
    ratio = d[first, second] / (radius[first] + radius[second])
    for k in np.flatnonzero((ratio < low) | (ratio > high)):
        i, j = int(first[k]), int(second[k])
        out.append(f"bond {label(i, j)} {float(d[i, j]):.2f} A "
                   f"({float(ratio[k]):.2f} of the covalent radii)")
    apart = np.where(bonded, np.inf, d)
    np.fill_diagonal(apart, np.inf)
    k = int(np.argmin(apart))
    i, j = divmod(k, len(r))
    if float(apart[i, j]) < contact:
        out.append(f"no bond {label(i, j)} {float(apart[i, j]):.2f} A")
    return out


_RADII: Dict[int, float] = {}


def _covalent_radius(z: int) -> float:
    """The covalent radius RDKit gives for the atomic number `z`, kept after the first call."""
    if z not in _RADII:
        _RADII[z] = float(Chem.GetPeriodicTable().GetRcovalent(int(z)))
    return _RADII[z]


def bond_from_smarts(mol: Chem.Mol, smarts: str) -> Optional[Tuple[int, int]]:
    """The first two atoms of the first match of `smarts`, as the bond whose stretch is wanted.

    ``[CX3](=O)[OX2H1]`` names the carbonyl carbon and its oxygen of a carboxylic acid; the
    first atom of the pattern is the one the band is written on.  None when nothing matches.
    """
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"not a SMARTS pattern: {smarts!r}")
    if patt.GetNumAtoms() < 2:
        raise ValueError("the pattern has to name at least two atoms")
    match = mol.GetSubstructMatch(patt)
    if not match:
        return None
    return int(match[0]), int(match[1])


def bond_stretch_mode(result: OrcaResult, i: int, j: int, low: float = 0.0,
                      high: float = float("inf")) -> Tuple[int, float, float, float]:
    """The normal mode that stretches the bond between atoms `i` and `j` most, among the modes
    whose frequency lies in ``[low, high]`` cm^-1.

    A band is assigned by its motion and not by its position alone: the carbonyl stretch is the
    mode in which the carbonyl carbon and oxygen move apart most, and the window only keeps a
    far-away mode that happens to move them a little from being chosen.  Returns the mode index,
    its frequency in cm^-1 (unscaled), its infrared intensity in km/mol and the *share*: half the
    squared projection of the unit mode vector on the stretch of this bond, a number between 0
    and 1 that is 1 when only these two atoms move and move exactly against each other, and
    says how much of the mode is this bond.  Raises :class:`OrcaError` when the job computed no
    modes or no mode lies in the window.
    """
    if result.frequencies is None or result.modes is None:
        raise OrcaError("the job computed no normal modes")
    if result.coords is None or max(i, j) >= len(result.coords):
        raise OrcaError("the job's coordinates do not hold both atoms of the bond")
    u = np.asarray(result.coords[j], float) - np.asarray(result.coords[i], float)
    norm = float(np.linalg.norm(u))
    if norm <= 0:
        raise OrcaError("the two atoms of the bond coincide")
    u = u / norm
    best = None
    for k, nu in enumerate(result.frequencies):
        if not (float(low) <= float(nu) <= float(high)):
            continue
        d_i = result.modes[3 * i:3 * i + 3, k]
        d_j = result.modes[3 * j:3 * j + 3, k]
        s = 0.5 * float(np.dot(u, d_j - d_i)) ** 2
        if best is None or s > best[3]:
            inten = (float(result.ir_intensities[k]) if result.ir_intensities is not None
                     else float("nan"))
            best = (int(k), float(nu), inten, s)
    if best is None:
        raise OrcaError(f"no mode between {low:g} and {high:g} cm^-1")
    return best


@dataclass
class BandOutcome:
    """What the frequency run of one molecule gave: the annotated copy, or the error."""
    index: int
    name: str
    mol: Optional[Chem.Mol] = None
    error: str = ""
    seconds: float = 0.0
    reused: bool = False
    frequency: float = float("nan")         # scaled, cm^-1
    unscaled: float = float("nan")
    intensity: float = float("nan")         # km/mol
    share: float = float("nan")
    mode: int = -1
    n_imaginary: int = 0


def run_band(mol: Chem.Mol, name: str, index: int, cfg: Config, bond: Tuple[int, int],
             prop: str = "band", low: float = 0.0, high: float = float("inf"),
             opt: bool = True) -> BandOutcome:
    """Optimise one molecule, compute its harmonic frequencies and write the band of one bond on
    the bond's first atom as a per-atom property (plan item SP2).

    The chain is the one of :func:`run_molecule` with a different job: the same cache key with
    the word ``Freq`` in the level, so that a property run and a frequency run on the same
    molecule never share a directory; the same status file; the same rule that a failed molecule
    is reported and never fatal.  The annotated copy carries the optimised geometry, the scaled
    band in ``atom.dprop.<prop>`` on the first atom of the bond and nothing on the other atoms,
    and as molecule properties the scaled and unscaled band, its intensity, its share, the mode
    index, the number of imaginary frequencies (a minimum has none) and the scaling factor.

    An optimisation that stops at ORCA's limit of cycles (:func:`optimisation_hit_cycle_limit`)
    is continued once: a second job, ``freq_more``, starts from the last geometry of the first
    with a computed Hessian and a limit of ``CONTINUE_CYCLES`` cycles.  The property ``qm_job``
    names the job the result comes from.

    The optimised geometry must still show the bonds of the molecule (:func:`changed_bonds`).
    A molecule whose optimisation broke or made a bond is a failure, because its band would be
    the band of another molecule.
    """
    t0 = time.time()
    outcome = BandOutcome(index=index, name=name)
    status_path = None
    key = None
    try:
        mol = prepare_molecule(mol)
        i, j = int(bond[0]), int(bond[1])
        if not (0 <= i < mol.GetNumAtoms() and 0 <= j < mol.GetNumAtoms()) or i == j:
            raise OrcaError(f"atoms {i} and {j} are not a bond of a molecule of "
                            f"{mol.GetNumAtoms()} atoms")
        charge, mult = state_from_properties(mol)
        lowest = multiplicity_for(mol, charge)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        coords0 = mol.GetConformer().GetPositions()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        key = cache_key(smiles, charge, cfg.level + " Freq", bool(opt), coords0,
                        solvent=cfg.solvent, mult=None if mult == lowest else mult)
        jobdir = os.path.join(cfg.workdir, f"{index:03d}_{safe_name(name)}")
        status_path = os.path.join(jobdir, "status.json")
        status = {}
        if os.path.isfile(status_path):
            with open(status_path) as fh:
                status = json.load(fh)
        if status.get("key") != key and os.path.isdir(jobdir):
            shutil.rmtree(jobdir)
            status = {}
        os.makedirs(jobdir, exist_ok=True)
        with open(os.path.join(jobdir, "geom.xyz"), "w", newline="\n") as fh:
            fh.write(xyz_block(symbols, coords0,
                               f"{name} charge={charge} mult={mult} (input geometry)"))
        outcome.reused = status.get("status") == "done"
        inp = make_frequency_input(cfg.level, charge, mult, "geom.xyz", cfg.nprocs,
                                   maxcore=cfg.maxcore, solvent=cfg.solvent, opt=opt)
        job, res = "freq", None
        if not (opt and optimisation_hit_cycle_limit(jobdir, job, inp)):
            try:
                res = run_job(jobdir, job, inp, cfg, need_converged=bool(opt))
            except OrcaError:
                if not (opt and optimisation_hit_cycle_limit(jobdir, job, inp)):
                    raise
        if res is None:
            # the forces are small and the steps are not: go on from the last geometry with a
            # computed Hessian, which knows how soft the soft motion is
            job = "freq_more"
            more = make_frequency_input(cfg.level, charge, mult, "freq.xyz", cfg.nprocs,
                                        maxcore=cfg.maxcore, solvent=cfg.solvent, opt=True,
                                        calc_hess=True, max_cycles=CONTINUE_CYCLES)
            res = run_job(jobdir, job, more, cfg, need_converged=True)
        if res.frequencies is None or res.modes is None:
            raise OrcaError("ORCA printed no normal modes")
        coords = coords0
        if opt:
            osyms, coords = read_xyz(os.path.join(jobdir, job + ".xyz"))
            if [s.upper() for s in osyms] != [s.upper() for s in symbols]:
                raise OrcaError("the optimised geometry lists different atoms")
            wrong = changed_bonds(mol, coords)
            if wrong:
                raise OrcaError("the optimisation changed the bonds of the molecule (bond, "
                                "length over the sum of the covalent radii): "
                                + ", ".join(wrong[:6]))
        if res.coords is None or len(res.coords) != len(symbols):
            res.coords = np.asarray(coords, float)
        k, nu, inten, share = bond_stretch_mode(res, i, j, low, high)
        scale = frequency_scale(cfg.level)
        n_imag = int(np.sum(res.frequencies < -5.0))
        outcome.frequency, outcome.unscaled = nu * scale, nu
        outcome.intensity, outcome.share, outcome.mode = inten, share, k
        outcome.n_imaginary = n_imag
        m = annotate(mol, {}, {prop: nu * scale, prop + "_unscaled": nu,
                               prop + "_intensity": inten, prop + "_share": share,
                               "E_total": res.energy},
                     coords=coords if opt else None,
                     extra={prop + "_mode": str(k), prop + "_atoms": f"{i} {j}",
                            "n_imaginary": str(n_imag), "frequency_scale": f"{scale:.4f}",
                            "qm_level": cfg.level,
                            "qm_geometry": "orca-opt" if opt else "input",
                            "qm_solvent": "gas" if cfg.solvent is None else str(cfg.solvent),
                            "qm_dir": os.path.basename(jobdir), "qm_job": job})
        m.GetAtomWithIdx(i).SetDoubleProp(prop, float(nu * scale))
        Chem.CreateAtomDoublePropertyList(m, prop)
        outcome.mol = m
        outcome.seconds = time.time() - t0
        seconds = status.get("seconds", outcome.seconds) if outcome.reused else outcome.seconds
        with open(status_path, "w") as fh:
            json.dump({"key": key, "status": "done", "name": name, "smiles": smiles,
                       "charge": charge, "multiplicity": mult, "level": cfg.level, "opt": bool(opt),
                       "freq": True, "bond": [i, j], "mode": k, "frequency": nu,
                       "scaled": nu * scale, "intensity": inten, "share": share,
                       "n_imaginary": n_imag,
                       "solvent": "gas" if cfg.solvent is None else str(cfg.solvent),
                       "seconds": seconds}, fh, indent=1)
    except Exception as exc:        # a failed molecule is reported, not fatal
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.seconds = time.time() - t0
        if status_path:
            try:
                with open(status_path, "w") as fh:
                    json.dump({"key": key, "status": "failed", "name": name,
                               "error": outcome.error}, fh, indent=1)
            except Exception:
                pass
    return outcome


def displaced_along_mode(coords: np.ndarray, mode: np.ndarray, step: float = 0.2) -> np.ndarray:
    """Coordinates moved along one normal mode: the atom that moves most moves by `step`
    angstrom.  The use is a stationary point that is not a minimum, where a step along the mode
    of the imaginary frequency followed by a new optimisation leads down to a minimum."""
    r = np.asarray(coords, float)
    d = np.asarray(mode, float).reshape(-1, 3)
    if d.shape != r.shape:
        raise ValueError(f"a mode of {len(d)} atoms on a geometry of {len(r)} atoms")
    top = float(np.max(np.linalg.norm(d, axis=1)))
    if top <= 0:
        raise ValueError("the mode moves no atom")
    return r + d * (float(step) / top)


def retry_band(outcome: BandOutcome, cfg: Config, bond: Tuple[int, int], prop: str = "band",
               low: float = 0.0, high: float = float("inf"), step: float = 0.2) -> BandOutcome:
    """One more optimisation for a molecule whose first one ended with an imaginary frequency.

    An optimisation stops at a stationary point, and a start geometry with a symmetric
    substituent can stop on a saddle point: a methyl or trifluoromethyl group eclipsed with the
    ring, for example.  The frequency run shows it as an imaginary frequency.  This function
    moves the optimised geometry along the mode of the lowest frequency
    (:func:`displaced_along_mode`) and repeats the job in a directory of its own, named after
    the molecule with ``retry`` added.  The second result replaces the first when it has fewer
    imaginary frequencies, or as many and a lower energy; it then carries the property
    ``<prop>_retry`` with the first band, so that the size of the correction can be read.  A
    molecule without an imaginary frequency, and one whose retry fails, is returned unchanged.
    """
    if outcome.mol is None or not outcome.n_imaginary:
        return outcome
    first = outcome.mol
    jobdir = os.path.join(cfg.workdir, first.GetProp("qm_dir"))
    job = first.GetProp("qm_job") if first.HasProp("qm_job") else "freq"
    with open(os.path.join(jobdir, job + ".out"), errors="replace") as fh:
        res = parse_orca_output(fh.read())
    if res.frequencies is None or res.modes is None:
        return outcome
    k = int(np.argmin(res.frequencies))
    start = Chem.Mol(first)
    conf = start.GetConformer()
    moved = displaced_along_mode(conf.GetPositions(), res.modes[:, k], step)
    for a, (x, y, z) in enumerate(moved):
        conf.SetAtomPosition(a, Point3D(float(x), float(y), float(z)))
    second = run_band(start, outcome.name + " retry", outcome.index, cfg, bond, prop=prop,
                      low=low, high=high, opt=True)
    if second.mol is None:
        return outcome
    lower = float(second.mol.GetProp("E_total")) < float(first.GetProp("E_total"))
    if not (second.n_imaginary < outcome.n_imaginary
            or (second.n_imaginary == outcome.n_imaginary and lower)):
        return outcome
    second.name = outcome.name
    second.seconds += outcome.seconds
    if first.HasProp("_Name"):
        second.mol.SetProp("_Name", first.GetProp("_Name"))
    second.mol.SetProp(prop + "_retry", f"{outcome.frequency:.4f}")
    return second


def band_positions(mols: Sequence[Chem.Mol], names: Sequence[str], cfg: Config,
                   bonds: Sequence[Optional[Tuple[int, int]]], prop: str = "band",
                   low: float = 0.0, high: float = float("inf"), jobs: int = 1,
                   opt: bool = True, progress=None, cancel=None, log=None) -> List[BandOutcome]:
    """:func:`run_band` over a set, `jobs` molecules at a time, as :func:`annotate_molecules`
    runs the property jobs.  A molecule whose entry of `bonds` is None is reported as skipped."""
    cfg = replace(cfg, cancel=cancel) if cancel is not None else cfg
    os.makedirs(cfg.workdir, exist_ok=True)
    done = [0]
    lock = threading.Lock()

    def work(k: int) -> BandOutcome:
        if cancel is not None and cancel():
            return BandOutcome(index=k + 1, name=names[k], error="cancelled")
        if bonds[k] is None:
            oc = BandOutcome(index=k + 1, name=names[k], error="no bond matches the pattern")
        else:
            oc = run_band(mols[k], names[k], k + 1, cfg, bonds[k], prop=prop, low=low,
                          high=high, opt=opt)
        with lock:
            done[0] += 1
            state = ("failed: " + oc.error if oc.error else
                     f"{'reused' if oc.reused else 'done'} in {oc.seconds:.0f} s, "
                     f"{oc.frequency:.1f} cm^-1 (mode {oc.mode}, share {oc.share:.2f}"
                     + (f", {oc.n_imaginary} imaginary" if oc.n_imaginary else "") + ")")
            if log is not None:
                log(f"[{done[0]}/{len(mols)}] {names[k]}: {state}")
            if progress is not None:
                progress(f"ORCA {done[0]}/{len(mols)}: {names[k]}", done[0] / max(len(mols), 1))
        return oc

    idx = list(range(len(mols)))
    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(work, idx))
    return [work(k) for k in idx]
