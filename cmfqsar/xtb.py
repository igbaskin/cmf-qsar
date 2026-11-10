"""
A cheap quantum stage: per-atom properties from GFN2-xTB (plan item D3).

`cmfqsar/orca.py` brings quantum-chemical per-atom quantities into the program as ``prop:``
fields, and it works, but a density-functional single point takes two to three minutes per
molecule on this machine.  That is affordable for one series and not for a collection.  GFN2-xTB
is a semi-empirical tight-binding method: it solves a parameterised one-electron problem instead
of the Kohn-Sham equations, so a molecule of forty atoms takes a fraction of a second, including
the two ionic states a Fukui index needs.  The question the plan asks is not whether the cheap
method is as good as the expensive one -- it is not -- but whether it is good enough to say, over
a whole collection of series, which series a quantum field helps at all.

What it computes, in **one** call of the program per molecule (``--gfn 2 --vfukui``):

=====================  ===============================================================
``q_xtb``              the GFN2 partial charge of the atom
``fukui_plus``         f+, the charge the atom gains when an electron is added
``fukui_minus``        f-, the charge it loses when an electron is removed
``fukui_zero``         f0, the mean of the two
``dual``               f+ - f-, positive where the atom is electrophilic
``alpha_xtb``          the atomic polarisability alpha(0), in atomic units
=====================  ===============================================================

and per molecule the total energy, the frontier orbital energies, their gap and the dipole.

**The Fukui convention is the same as ORCA's**, which is what makes the two engines
interchangeable in a ``prop:`` field name.  `cmfqsar/orca.py` forms ``f+ = q(N) - q(N+1)`` from
Hirshfeld charges; xtb prints f+ directly, and `tests/test_xtb.py` checks on water that the
printed value is the same difference of the program's own charges to three decimals.  What
differs is the population analysis behind the charges -- Hirshfeld there, a Mulliken-like
tight-binding population here -- so ``q_xtb`` is **not** ``q_hirshfeld`` and carries a name of its
own.  A model fitted on one must be predicted with the same one.

Three things are built in rather than left to the caller, as in `cmfqsar/orca.py`.  Every job is
**cached** by a key over the molecule, its charge, the method and the geometry, so an interrupted
run resumes.  A molecule that fails is **reported and skipped**, never fatal to the set.  And the
geometry is **not optimised** unless asked: a superimposed series optimised molecule by molecule
is no longer superimposed.

What this module does not provide is the ionisation potential and the electron affinity, and so
not the hardness, the softness or the electrophilicity index that `cmfqsar/orca.py` derives from
them.  The energies xtb prints for the two ionic states of a Fukui run are electronic energies of
the self-consistent charge step alone, not total energies, so a difference against the neutral's
total energy would not be an ionisation potential.  xtb computes those properly with ``--vipea``,
which needs a different Hamiltonian (GFN1 with the ipea parameters); mixing a GFN1 descriptor with
GFN2 fields would describe one molecule with two methods, so it is left out.

The command-line front end is ``examples/xtb_atom_properties.py``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger

from cmfqsar.orca import (annotate, cache_key, multiplicity_for, prepare_molecule, safe_name,
                          verify_sdf, xyz_block)

RDLogger.DisableLog("rdApp.*")

DEFAULT_XTB = r"C:\xtb-6.7.1\xtb-6.7.1\bin\xtb.exe"
DEFAULT_METHOD = 2                       # the GFN level; 2 is GFN2-xTB
HARTREE_EV = 27.211386245988

#: per-atom property names written to the SDF (atom double property lists)
ATOM_PROPS = ["q_xtb", "alpha_xtb"]
ATOM_PROPS_FUKUI = ["fukui_plus", "fukui_minus", "fukui_zero", "dual"]
#: per-molecule property names
MOL_PROPS = ["E_total", "E_homo", "E_lumo", "gap", "dipole", "charge", "multiplicity"]


class XtbError(RuntimeError):
    pass


WHERE_TO_GET_XTB = ("the Windows and Linux builds are released at "
                    "https://github.com/grimme-lab/xtb/releases")


def _resolve(cand: str) -> Optional[str]:
    """`cand` as an executable: a file as it stands, or a bare name found on the path."""
    if os.path.isfile(cand):
        return cand
    return shutil.which(cand)


def find_xtb(explicit: Optional[str] = None) -> str:
    """The xtb executable: the one given, then ``XTB_EXE``, then the path, then the default.

    A path that was given -- as `explicit` or in ``XTB_EXE`` -- is used or refused, never
    replaced. If it names neither a file nor a program on the path, the search stops there and
    the error quotes it. Running a different xtb instead would produce numbers from a program
    the caller did not ask for, and nothing in the result would say so.
    """
    for cand, source in ((explicit, "the path given for it"),
                         (os.environ.get("XTB_EXE"), "XTB_EXE")):
        if cand:
            found = _resolve(cand)
            if found:
                return found
            raise XtbError('xtb was not found at "%s", which is %s. Correct it, or leave '
                           'it out to search the path; %s' % (cand, source, WHERE_TO_GET_XTB))
    for cand in (shutil.which("xtb"), DEFAULT_XTB):
        if cand and os.path.isfile(cand):
            return cand
    raise XtbError("xtb was not found. Give its path, set XTB_EXE, or put it on the path; "
                   + WHERE_TO_GET_XTB)


def xtb_version(exe: str) -> str:
    """The version string the program reports, for the provenance of a result file."""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
    except Exception as exc:
        raise XtbError(f"xtb would not run: {exc}")
    for line in (out.stdout or "").splitlines():
        if "version" in line.lower():
            return " ".join(line.split())
    return "unknown"


# ----------------------------------------------------------------------------------------------
# the output
# ----------------------------------------------------------------------------------------------

#: "     1O       0.234    0.598    0.416"
_FUKUI_LINE = re.compile(r"^\s*(\d+)([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
#: "     1   8 O        1.611    -0.564    24.329     6.657"
_ATOM_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
                        r"\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
#: "         4        2.0000           -0.4474930             -12.1769 (HOMO)"
_FRONTIER = re.compile(r"^\s*\d+\s.*?(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\((HOMO|LUMO)\)")
_TOTAL_ENERGY = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)")
_GAP = re.compile(r"HOMO-LUMO GAP\s+(-?\d+\.\d+)")


@dataclass
class XtbResult:
    """What one xtb call reports; the Fukui entries are None without ``--vfukui``."""
    symbols: List[str]
    charges: np.ndarray
    polarisability: np.ndarray
    energy: float = float("nan")              # Hartree
    homo: float = float("nan")                # eV
    lumo: float = float("nan")                # eV
    gap: float = float("nan")                 # eV
    dipole: float = float("nan")              # Debye
    fukui_plus: Optional[np.ndarray] = None
    fukui_minus: Optional[np.ndarray] = None
    fukui_zero: Optional[np.ndarray] = None
    version: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def _dipole(lines: Sequence[str]) -> float:
    """The total dipole in Debye: the last number of the 'full:' line of the dipole block."""
    for i, line in enumerate(lines):
        if line.strip().startswith("molecular dipole"):
            for other in lines[i + 1:i + 5]:
                if other.strip().startswith("full:"):
                    nums = re.findall(r"-?\d+\.\d+", other)
                    if nums:
                        return float(nums[-1])
            break
    return float("nan")


def parse_output(text: str) -> XtbResult:
    """Read the quantities this module uses out of an xtb run.

    The partial charges are taken from the per-atom table of the output rather than from the
    ``charges`` file, so that a result can be parsed from the text alone; `run_molecule` prefers
    the file, which carries more digits.
    """
    if "normal termination of xtb" not in text:
        raise XtbError("xtb did not terminate normally")
    lines = text.splitlines()
    symbols, q, alpha = [], [], []
    seen_header = False
    for line in lines:
        if "covCN" in line and "C6AA" in line:
            symbols, q, alpha = [], [], []        # a later table replaces an earlier one
            seen_header = True
            continue
        if not seen_header:
            continue
        m = _ATOM_LINE.match(line)
        if m:
            symbols.append(m.group(3))
            q.append(float(m.group(5)))
            alpha.append(float(m.group(7)))
        elif symbols and not line.strip():
            seen_header = False                   # the table ended
    if not symbols:
        raise XtbError("the output carries no per-atom table")

    res = XtbResult(symbols=symbols, charges=np.array(q, float),
                    polarisability=np.array(alpha, float))
    m = _TOTAL_ENERGY.search(text)
    if m:
        res.energy = float(m.group(1))
    m = _GAP.search(text)
    if m:
        res.gap = float(m.group(1))
    for line in lines:
        m = _FRONTIER.match(line)
        if m:
            value = float(m.group(2))
            if m.group(3) == "HOMO":
                res.homo = value
            else:
                res.lumo = value
    res.dipole = _dipole(lines)

    f_plus, f_minus, f_zero, inside = [], [], [], False
    for line in lines:
        if line.strip().startswith("Fukui functions"):
            f_plus, f_minus, f_zero, inside = [], [], [], True
            continue
        if inside:
            m = _FUKUI_LINE.match(line)
            if m:
                f_plus.append(float(m.group(3)))
                f_minus.append(float(m.group(4)))
                f_zero.append(float(m.group(5)))
            elif f_plus:
                inside = False
    if f_plus:
        if len(f_plus) != len(symbols):
            raise XtbError(f"{len(f_plus)} Fukui rows for {len(symbols)} atoms")
        res.fukui_plus = np.array(f_plus, float)
        res.fukui_minus = np.array(f_minus, float)
        res.fukui_zero = np.array(f_zero, float)
    return res


def derive_properties(res: XtbResult, charge: int = 0,
                      mult: int = 1) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """(per-atom, per-molecule) properties from one result, named as `cmfqsar/orca.py` names them.

    ``dual`` is f+ - f-, the dual descriptor: positive where an atom takes an electron more
    readily than it gives one.  The Fukui entries are absent when the run did not compute them.
    """
    atom: Dict[str, np.ndarray] = {"q_xtb": res.charges, "alpha_xtb": res.polarisability}
    mol: Dict[str, float] = {"E_total": res.energy, "E_homo": res.homo, "E_lumo": res.lumo,
                             "gap": res.gap, "dipole": res.dipole,
                             "charge": float(charge), "multiplicity": float(mult)}
    if res.fukui_plus is not None:
        atom.update({"fukui_plus": res.fukui_plus, "fukui_minus": res.fukui_minus,
                     "fukui_zero": res.fukui_zero,
                     "dual": res.fukui_plus - res.fukui_minus})
    return atom, mol


# ----------------------------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------------------------

@dataclass
class Config:
    xtb: str = ""                    # empty: found by find_xtb
    gfn: int = DEFAULT_METHOD
    opt: bool = False
    fukui: bool = True
    solvent: str = ""                # an ALPB solvent name, e.g. "water"; empty is the gas phase
    accuracy: float = 1.0            # xtb's --acc; smaller is tighter
    timeout: Optional[float] = 600.0
    workdir: str = "xtb_work"
    #: asked between molecules; when it returns True the run stops and what is finished is kept
    cancel: Optional[object] = None

    def level(self) -> str:
        """One string naming the method, for the cache key and the provenance of the file."""
        parts = [f"GFN{self.gfn}-xTB", f"acc={self.accuracy:g}"]
        if self.solvent:
            parts.append(f"alpb={self.solvent}")
        if self.opt:
            parts.append("opt")
        return " ".join(parts)


def command(cfg: Config, xyz: str, charge: int, mult: int) -> List[str]:
    """The argument list for one molecule."""
    args = [find_xtb(cfg.xtb or None), xyz, "--gfn", str(int(cfg.gfn)),
            "--chrg", str(int(charge)), "--uhf", str(int(mult) - 1),
            "--acc", f"{float(cfg.accuracy):g}"]
    if cfg.solvent:
        args += ["--alpb", cfg.solvent]
    if cfg.opt:
        args.append("--opt")
    if cfg.fukui:
        args.append("--vfukui")
    return args


def _file_charges(res: XtbResult, jobdir: str) -> XtbResult:
    """Replace the charges of `res` by the ones of the ``charges`` file of the job, which are
    the same charges with more digits (8 decimals; the table of the output prints 3)."""
    charges_file = os.path.join(jobdir, "charges")
    if os.path.isfile(charges_file):
        with open(charges_file) as fh:
            vals = [float(x) for x in fh.read().split()]
        if len(vals) == res.n_atoms:
            res.charges = np.array(vals, float)
    return res


def run_job(jobdir: str, cfg: Config, charge: int, mult: int) -> XtbResult:
    """Run xtb in `jobdir` on ``geom.xyz``, or reuse the output already there.

    A reused job gives the same numbers as its first run: the charges are read from the
    ``charges`` file in both cases.
    """
    out_path = os.path.join(jobdir, "xtb.out")
    if os.path.isfile(out_path):
        with open(out_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if "normal termination of xtb" in text:
            return _file_charges(parse_output(text), jobdir)
    args = command(cfg, "geom.xyz", charge, mult)
    try:
        # xtb writes Greek letters and a middle dot, which the Windows code page cannot
        # decode; the output is read as UTF-8 throughout
        proc = subprocess.run(args, cwd=jobdir, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=cfg.timeout)
    except subprocess.TimeoutExpired:
        raise XtbError(f"xtb did not finish within {cfg.timeout:g} s")
    text = (proc.stdout or "") + (proc.stderr or "")
    with open(out_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(text)
    if "normal termination of xtb" not in text:
        tail = " / ".join(line.strip() for line in text.strip().splitlines()[-3:])
        raise XtbError(f"xtb failed (exit {proc.returncode}): {tail}")
    return _file_charges(parse_output(text), jobdir)


@dataclass
class MoleculeOutcome:
    index: int
    name: str
    mol: Optional[Chem.Mol] = None          # annotated, None on failure
    error: str = ""
    seconds: float = 0.0
    reused: bool = False


def run_molecule(mol: Chem.Mol, name: str, index: int, cfg: Config) -> MoleculeOutcome:
    """The whole chain for one molecule: geometry, the xtb call, parsing, the annotated copy."""
    t0 = time.time()
    outcome = MoleculeOutcome(index=index, name=name)
    status_path = key = None
    try:
        mol = prepare_molecule(mol)
        charge = Chem.GetFormalCharge(mol)
        mult = multiplicity_for(mol, charge)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        coords0 = mol.GetConformer().GetPositions()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        level = cfg.level() + (" fukui" if cfg.fukui else "")
        key = cache_key(smiles, charge, level, cfg.opt, coords0)
        jobdir = os.path.join(cfg.workdir, f"{index:05d}_{safe_name(name)}")
        status_path = os.path.join(jobdir, "status.json")
        status = {}
        if os.path.isfile(status_path):
            with open(status_path) as fh:
                status = json.load(fh)
        if status.get("key") != key and os.path.isdir(jobdir):
            shutil.rmtree(jobdir)           # the directory belongs to another computation
            status = {}
        os.makedirs(jobdir, exist_ok=True)
        with open(os.path.join(jobdir, "geom.xyz"), "w", newline="\n") as fh:
            fh.write(xyz_block(symbols, coords0, f"{name} charge={charge} mult={mult}"))
        outcome.reused = status.get("status") == "done"

        res = run_job(jobdir, cfg, charge, mult)
        if res.n_atoms != len(symbols):
            raise XtbError(f"xtb reports {res.n_atoms} atoms, the molecule has {len(symbols)}")
        atom_props, mol_props = derive_properties(res, charge=charge, mult=mult)
        coords = None
        if cfg.opt:
            from cmfqsar.orca import read_xyz
            opt_path = os.path.join(jobdir, "xtbopt.xyz")
            if not os.path.isfile(opt_path):
                raise XtbError("the optimised geometry was not written")
            osyms, coords = read_xyz(opt_path)
            if [s.upper() for s in osyms] != [s.upper() for s in symbols]:
                raise XtbError("the optimised geometry lists different atoms")
        extra = {"qm_level": cfg.level(), "qm_program": res.version or "xtb",
                 "qm_geometry": "xtb-opt" if cfg.opt else "input",
                 "qm_dir": os.path.basename(jobdir)}
        outcome.mol = annotate(mol, atom_props, mol_props, coords=coords, extra=extra)
        outcome.seconds = time.time() - t0
        seconds = status.get("seconds", outcome.seconds) if outcome.reused else outcome.seconds
        with open(status_path, "w") as fh:
            json.dump({"key": key, "status": "done", "name": name, "smiles": smiles,
                       "charge": charge, "multiplicity": mult, "level": cfg.level(),
                       "opt": cfg.opt, "fukui": cfg.fukui, "seconds": seconds}, fh, indent=1)
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


def process_molecules(mols: Sequence[Chem.Mol], names: Sequence[str], cfg: Config,
                      jobs: int = 1, log=print) -> List[MoleculeOutcome]:
    """Run every molecule and return the outcomes, in the order of the input."""
    os.makedirs(cfg.workdir, exist_ok=True)
    out: List[Optional[MoleculeOutcome]] = [None] * len(mols)

    def work(k):
        if cfg.cancel is not None and cfg.cancel():
            return
        oc = run_molecule(mols[k], names[k], k + 1, cfg)
        out[k] = oc
        if oc.error:
            log(f"[{k + 1}/{len(mols)}] {oc.name}: FAILED -- {oc.error}")
        return oc

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(work, range(len(mols))))
    else:
        for k in range(len(mols)):
            work(k)
    return [oc for oc in out if oc is not None]


def process_dataset(path: str, out: str, cfg: Config, limit: Optional[int] = None, jobs: int = 1,
                    log=print) -> List[MoleculeOutcome]:
    """Run every molecule of `path` and write the annotated SDF `out`; returns the outcomes."""
    from cmfqsar.dataio import read_molecules
    mols, names, _ = read_molecules(path)
    if limit:
        mols, names = mols[:limit], names[:limit]
    log(f"{len(mols)} molecules from {path}; {cfg.level()}, "
        f"{'optimised' if cfg.opt else 'input'} geometry, "
        f"Fukui {'on' if cfg.fukui else 'off'}, {jobs} at a time -> {cfg.workdir}")
    outcomes = process_molecules(mols, names, cfg, jobs=jobs, log=log)
    done = [oc for oc in outcomes if oc.mol is not None]
    if not done:
        raise XtbError(f"no molecule of {path} was computed")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with Chem.SDWriter(out) as w:
        for oc in done:
            w.write(oc.mol)
    expect = ATOM_PROPS + (ATOM_PROPS_FUKUI if cfg.fukui else [])
    verify_sdf(out, expect)
    total = sum(oc.seconds for oc in outcomes)
    log(f"{len(done)}/{len(outcomes)} molecules written to {out} "
        f"({total:.1f} s, {total / max(len(outcomes), 1):.2f} s per molecule)")
    return outcomes
