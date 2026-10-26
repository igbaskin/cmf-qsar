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
DEFAULT_ORCA = r"C:\ORCA_6.1.0\orca.exe"
HARTREE_EV = 27.211386245988

# per-atom property names written to the SDF (atom double property lists)
ATOM_PROPS_N = ["q_mulliken", "q_loewdin", "q_hirshfeld", "q_chelpg", "homo_pop", "lumo_pop"]
ATOM_PROPS_FUKUI = ["fukui_plus", "fukui_minus", "fukui_zero", "dual",
                    "softness_plus", "softness_minus", "local_electrophilicity",
                    "fukui_plus_mulliken", "fukui_minus_mulliken", "fukui_zero_mulliken",
                    "dual_mulliken"]
MOL_PROPS_N = ["E_total", "E_homo", "E_lumo", "gap", "dipole", "charge", "multiplicity"]
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


def make_input(level: str, charge: int, mult: int, xyz_name: str, nprocs: int = 1,
               opt: bool = False, populations: bool = True, chelpg: bool = True,
               maxcore: int = 2000) -> str:
    """Text of an ORCA input.

    `populations` adds the Hirshfeld analysis and the per-MO Mulliken atom populations; `chelpg`
    the CHELPG fit (needed for the N-electron state only).  An open-shell state is run
    unrestricted; a geometry optimisation prints nothing but what it needs.
    """
    keywords = [level]
    if opt:
        keywords.append("Opt")
    if mult > 1:
        keywords.append("UKS")
    if chelpg and not opt:
        keywords.append("CHELPG")
    lines = ["! " + " ".join(keywords)]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.append(f"%maxcore {maxcore}")
    lines.append("%scf MaxIter 300 end")
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


def cache_key(smiles: str, charge: int, level: str, opt: bool, coords: Optional[np.ndarray] = None) -> str:
    """Identity of a computation: the molecule, its state, the level and the geometry.

    The SMILES is canonicalised so that the same compound written two ways shares a key; the
    coordinates (to 1e-3 A) are part of it because a different starting geometry is a different
    result even when nothing else changed.
    """
    m = Chem.MolFromSmiles(smiles)
    can = Chem.MolToSmiles(m) if m is not None else smiles
    h = hashlib.sha1()
    h.update(f"{can}|{charge}|{' '.join(level.split())}|{int(bool(opt))}".encode())
    if coords is not None:
        h.update(np.round(np.asarray(coords, float), 3).tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# ORCA output
# ----------------------------------------------------------------------------------------------

_CHARGE_LINE = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s*:?\s+(-?\d+\.\d+)")
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
    homo: float = float("nan")              # Eh
    lumo: float = float("nan")              # Eh
    homo_index: int = -1
    lumo_index: int = -1
    mo_populations: Dict[int, Dict[int, float]] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    coords: Optional[np.ndarray] = None

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
        charge = Chem.GetFormalCharge(mol)
        mult = multiplicity_for(mol, charge)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        coords0 = mol.GetConformer().GetPositions()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        key = cache_key(smiles, charge, cfg.level, cfg.opt, coords0)
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
                             populations=False, chelpg=False, maxcore=cfg.maxcore)
            run_job(jobdir, "opt", inp, cfg, need_converged=True)
            osyms, coords = read_xyz(os.path.join(jobdir, "opt.xyz"))
            if [s.upper() for s in osyms] != [s.upper() for s in symbols]:
                raise OrcaError("the optimised geometry lists different atoms")
            xyz = "opt.xyz"
        res_n = run_job(jobdir, "sp_n", make_input(cfg.level, charge, mult, xyz, cfg.nprocs,
                                                   maxcore=cfg.maxcore), cfg)
        if res_n.n_atoms != len(symbols):
            raise OrcaError(f"ORCA reports {res_n.n_atoms} atoms, the molecule has {len(symbols)}")
        res_p = res_m = None
        if cfg.fukui:
            res_p = run_job(jobdir, "sp_np1",
                            make_input(cfg.level, charge - 1, multiplicity_for(mol, charge - 1), xyz,
                                       cfg.nprocs, chelpg=False, maxcore=cfg.maxcore), cfg)
            res_m = run_job(jobdir, "sp_nm1",
                            make_input(cfg.level, charge + 1, multiplicity_for(mol, charge + 1), xyz,
                                       cfg.nprocs, chelpg=False, maxcore=cfg.maxcore), cfg)
        atom_props, mol_props = derive_properties(res_n, res_p, res_m, charge=charge, mult=mult)
        extra = {"qm_level": cfg.level,
                 "qm_geometry": "orca-opt" if cfg.opt else "input",
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
