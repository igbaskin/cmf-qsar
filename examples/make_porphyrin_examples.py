"""
Substituted cobalt and iron porphyrins with computed HER and ORR descriptors.

Why this series is computed here
--------------------------------
The single-atom study (plan item EC8) measured CMF on metal sites in a graphene sheet. Its site is
symmetric, so a frame on the metal had nothing to resolve. A molecular macrocycle is the natural
next case: the ring itself can be substituted on one side, so the pocket around the metal becomes
asymmetric. A search of the literature on 15 November 2026 found no machine-readable series of
molecular catalysts that gives both a hydrogen descriptor and the oxygen descriptors with the
ring varied. The largest table found (444 complexes, Data S1 of a Cell Press article) varies
ligand parts on a grid, has no hydrogen descriptor and no structures. So the series is computed
here.

What is computed
----------------
Each member is a porphyrin with one metal (Co or Fe) in the ring and one adsorbate, or none, on
one face of the metal: H (the hydrogen evolution intermediate), and OH, O and OOH (the oxygen
reduction intermediates). The porphyrins differ by

* substituents on the meso carbons (positions 5, 10, 15, 20) or on the beta carbons (positions
  2, 3, 7, 8, 12, 13, 17, 18), one, two or all of them;
* nitrogen in place of meso carbons (5-aza, 5,15-diaza, the porphyrazine with all four), and the
  phthalocyanine.

The workflow for every species and spin state:

1. A three-dimensional free-base porphyrin from RDKit (ETKDG and MMFF). The two inner N-H
   hydrogens are removed, the metal is placed at the centre of the four inner nitrogens, and the
   adsorbate is placed above the metal along the normal of the ring.
2. A geometry optimisation with GFN2-xTB (xtb 6.7.1). The ORCA build on this machine is serial,
   and optimising 40 to 70 atoms with DFT is too slow for hundreds of species.
3. A geometry check (:func:`geometry_problems`): the adsorbate is still on the metal, OOH has not
   split, H has not moved to the ring, the ring bonds are those of the bare complex. A species
   that fails is kept in the table with its reason and gets no energy.
4. A DFT single point with ORCA on the xtb geometry, at the level in :data:`LEVEL`, in a
   continuum model of water.
5. For each species, the spin state of lowest DFT energy is used.

The descriptors are the free energies of adsorption in the computational hydrogen electrode
scheme (Norskov et al., J. Phys. Chem. B 108 (2004) 17886), with water and hydrogen as the
references:

    dG_H   = E(M-H)   - E(M) - 1/2 E(H2)                    + dZPE-TdS(H)
    dG_OH  = E(M-OH)  - E(M) - [E(H2O) - 1/2 E(H2)]         + dZPE-TdS(OH)
    dG_O   = E(M-O)   - E(M) - [E(H2O) - E(H2)]             + dZPE-TdS(O)
    dG_OOH = E(M-OOH) - E(M) - [2 E(H2O) - 3/2 E(H2)]       + dZPE-TdS(OOH)

The zero-point and entropy corrections are the same for every member of the series. They are
taken from GFN2-xTB frequencies of the unsubstituted complexes (stage ``--thermo``). A constant
shifts every value by the same amount, so it changes no comparison between methods.

From these follow the ORR limiting potential (the highest potential at which all four
proton-electron steps are downhill) and the overpotential 1.23 V - U_L, the HER free energy
dG_H, and the difference dG_H - dG_OH that the single-atom study used to compare the two
reactions on one site.

What the targets are, and what they are not
-------------------------------------------
All values are **computed**, at one level of theory, on xtb geometries. They are not measured
potentials. DFT spin-state energies of iron are known to depend on the functional; the level was
chosen with that in mind (see :data:`LEVEL`), and the spin state of every species is stored, so
a reader can see where it changed. The series says how well a representation orders complexes
that one calculation produced.

Usage
-----
    python examples/make_porphyrin_examples.py --build           # structures and xtb geometries
    python examples/make_porphyrin_examples.py --orca --jobs 8   # DFT single points
    python examples/make_porphyrin_examples.py --thermo          # xtb corrections, parents only
    python examples/make_porphyrin_examples.py --collect         # porphyrin/porphyrins.csv
    python examples/make_porphyrin_examples.py --summary

The work folders are under ``benchmarks/cache/porphyrin`` (not in the repository). Every job is
cached there, so a stopped run continues where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cmfqsar import orca as O  # noqa: E402
from cmfqsar import xtb as X  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT_DIR = os.path.join(HERE, "porphyrin")
WORK = os.path.join(ROOT, "benchmarks", "cache", "porphyrin")
HARTREE_EV = 27.211386245988

#: the DFT level of the single points; chosen on 16 Nov 2026 by timing three levels on the
#: parent complexes (see the README of the data set)
LEVEL = "r2SCAN-3c TightSCF CPCM(water)"

PORPHINE = "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3"
PHTHALOCYANINE = ("c1ccc2c(c1)-c1nc-2nc2[nH]c(nc3nc(nc4[nH]c(n1)c1ccccc41)-c1ccccc1-3)"
                  "c1ccccc21")

#: substituents, written with the attachment point as [*]
SUBSTITUENTS: Dict[str, str] = {
    "F": "[*]F", "Cl": "[*]Cl", "Br": "[*]Br", "CH3": "[*]C", "CF3": "[*]C(F)(F)F",
    "CN": "[*]C#N", "NO2": "[*][N+](=O)[O-]", "OCH3": "[*]OC", "NH2": "[*]N", "OH": "[*]O",
    "CHO": "[*]C=O", "CCH": "[*]C#C", "Ph": "[*]c1ccccc1", "NMe2": "[*]N(C)C",
    "CO2Me": "[*]C(=O)OC",
}

MESO = (5, 10, 15, 20)
BETA = (2, 3, 7, 8, 12, 13, 17, 18)


def _series() -> List[Tuple[str, Dict[int, str], Tuple[int, ...]]]:
    """(name, {position: substituent}, aza positions) for every porphyrin of the series."""
    s: List[Tuple[str, Dict[int, str], Tuple[int, ...]]] = [("porphine", {}, ())]
    for x in ("F", "Cl", "Br", "CH3", "CF3", "CN", "NO2", "OCH3", "NH2", "OH", "CHO", "CCH",
              "Ph", "NMe2", "CO2Me"):
        s.append((f"5-{x}", {5: x}, ()))
    for x in ("F", "Cl", "CH3", "CF3", "CN", "NO2", "OCH3", "NH2", "Ph"):
        s.append((f"5,15-{x}2", {5: x, 15: x}, ()))
    for x in ("CF3", "CN", "NH2"):
        s.append((f"5,10-{x}2", {5: x, 10: x}, ()))
    for x in ("F", "Cl", "CH3", "CF3", "CN"):
        s.append((f"5,10,15,20-{x}4", {p: x for p in MESO}, ()))
    for x in ("F", "Cl", "Br", "CH3", "CF3", "CN", "NO2", "OCH3", "NH2", "CHO"):
        s.append((f"2-{x}", {2: x}, ()))
    for x in ("Cl", "CH3", "CN"):
        s.append((f"2,3-{x}2", {2: x, 3: x}, ()))
    for x in ("F", "Cl", "CH3"):
        s.append((f"beta-{x}8", {p: x for p in BETA}, ()))
    s.append(("5-aza", {}, (5,)))
    s.append(("5,15-diaza", {}, (5, 15)))
    s.append(("porphyrazine", {}, MESO))
    s.append(("5-aza-15-CF3", {15: "CF3"}, (5,)))
    s.append(("phthalocyanine", {}, ()))
    return s


SERIES = _series()

#: the first batch (16 Nov 2026): the parent and 20 members that cover every kind of change --
#: meso and beta positions, donors and acceptors, one, two and four substituents, and the two
#: aza rings -- so that the spread of the descriptors can be seen before the rest is computed
FIRST_BATCH = ("porphine", "5-F", "5-Cl", "5-CH3", "5-CF3", "5-CN", "5-NO2", "5-OCH3", "5-NH2",
               "5-Ph", "5,15-CF32", "5,15-NH22", "5,10-CN2", "5,10,15,20-F4",
               "5,10,15,20-CF34", "2-CN", "2-NH2", "2-CF3", "beta-Cl8", "porphyrazine",
               "phthalocyanine")

#: a spin state is computed for the substituted members only when it lies within this many eV
#: of the lowest state of the same species on the unsubstituted parent
SPIN_WINDOW_EV = 0.25

METALS = ("Co", "Fe")
ADSORBATES = ("", "H", "OH", "O", "OOH")

#: spin multiplicities computed for each metal and adsorbate (the neutral complex)
SPINS: Dict[Tuple[str, str], Tuple[int, ...]] = {
    ("Co", ""): (2, 4), ("Co", "H"): (1, 3), ("Co", "OH"): (1, 3), ("Co", "O"): (2, 4),
    ("Co", "OOH"): (1, 3),
    ("Fe", ""): (1, 3, 5), ("Fe", "H"): (2, 4), ("Fe", "OH"): (2, 4, 6), ("Fe", "O"): (1, 3, 5),
    ("Fe", "OOH"): (2, 4, 6),
}


# ----------------------------------------------------------------------------------------------
# structures
# ----------------------------------------------------------------------------------------------

def perimeter_numbering(mol: Chem.Mol) -> Dict[int, int]:
    """Map the porphyrin position numbers 1-20 to atom indices.

    The 20 carbons of the porphyrin perimeter form one cycle: alpha (bonded to an inner N), beta,
    beta, alpha, meso, and so on. The walk starts at an alpha carbon and goes to its beta
    neighbour first, so positions 1, 4, 6, 9, ... are alpha, 2, 3, 7, 8, ... beta and 5, 10, 15,
    20 meso, as in the IUPAC numbering.
    """
    inner = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.IsInRingSize(5)}
    alpha = {a.GetIdx() for a in mol.GetAtoms()
             if a.GetSymbol() == "C" and any(n.GetIdx() in inner for n in a.GetNeighbors())}
    perim = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"}   # unsubstituted porphine
    if len(perim) != 20 or len(alpha) != 8:
        raise ValueError(f"not a porphyrin perimeter ({len(perim)} atoms, {len(alpha)} alpha)")

    def cnb(i):
        return [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in perim]

    start = min(alpha)
    first = [j for j in cnb(start) if j not in alpha and mol.GetAtomWithIdx(j).IsInRingSize(5)][0]
    order = [start, first]
    while len(order) < 20:
        nxt = [j for j in cnb(order[-1]) if j != order[-2]]
        order.append(nxt[0])
    if order[0] not in cnb(order[-1]):
        raise ValueError("the perimeter walk did not close")
    return {k + 1: idx for k, idx in enumerate(order)}


def _attach(mol: Chem.Mol, atom: int, frag_smiles: str) -> Chem.Mol:
    frag = Chem.MolFromSmiles(frag_smiles)
    dummy = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0][0]
    head = frag.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
    n0 = mol.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(mol, frag))
    combo.AddBond(atom, n0 + head, Chem.BondType.SINGLE)
    combo.RemoveAtom(n0 + dummy)
    a = combo.GetAtomWithIdx(atom)
    a.SetNumExplicitHs(0)
    a.SetNoImplicit(False)
    return combo.GetMol()


def free_base(name: str, subs: Dict[int, str], aza: Sequence[int]) -> Chem.Mol:
    """The free-base porphyrin of one member, as an RDKit molecule without hydrogens."""
    if name == "phthalocyanine":
        mol = Chem.MolFromSmiles(PHTHALOCYANINE)
        if mol is None:
            raise ValueError("phthalocyanine SMILES does not parse")
        return mol
    mol = Chem.MolFromSmiles(PORPHINE)
    num = perimeter_numbering(mol)
    # the substituents first: their atom indices are added at the end and do not move the ring
    for pos, x in sorted(subs.items()):
        mol = _attach(mol, num[pos], SUBSTITUENTS[x])
    rw = Chem.RWMol(mol)
    for pos in aza:
        a = rw.GetAtomWithIdx(num[pos])
        if a.GetDegree() != 2:
            raise ValueError(f"{name}: aza position {pos} carries a substituent")
        a.SetAtomicNum(7)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))


def embed(mol: Chem.Mol, seed: int = 7) -> Chem.Mol:
    m = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0:
            raise RuntimeError("embedding failed")
    AllChem.MMFFOptimizeMolecule(Chem.Mol(m), maxIters=0)   # a check that MMFF can type it
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    return m


def inner_nitrogens(mol: Chem.Mol) -> List[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.IsInRingSize(5)]


def species_geometry(fb: Chem.Mol, metal: str, ads: str) -> Tuple[List[str], np.ndarray]:
    """Atoms of the metal complex with `ads` on the +normal face, from an embedded free base.

    The ligand atoms keep the order of the free base (without the two inner N-H hydrogens); the
    metal follows, then the adsorbate atoms, metal-bound atom first.
    """
    pos = fb.GetConformer().GetPositions()
    inner = inner_nitrogens(fb)
    drop = {n.GetIdx() for i in inner for n in fb.GetAtomWithIdx(i).GetNeighbors()
            if n.GetSymbol() == "H"}
    if len(drop) != 2 or len(inner) != 4:
        raise ValueError("expected four inner nitrogens and two N-H hydrogens")
    keep = [i for i in range(fb.GetNumAtoms()) if i not in drop]
    sym = [fb.GetAtomWithIdx(i).GetSymbol() for i in keep]
    xyz = pos[keep]
    n4 = pos[inner]
    c = n4.mean(0)
    _, _, vt = np.linalg.svd(n4 - c)
    z, x = vt[2], vt[0]
    sym.append(metal)
    rows = [c]
    if ads == "H":
        sym += ["H"]; rows += [c + 1.50 * z]
    elif ads == "OH":
        sym += ["O", "H"]; rows += [c + 1.90 * z, c + 2.50 * z + 0.75 * x]
    elif ads == "O":
        sym += ["O"]; rows += [c + 1.65 * z]
    elif ads == "OOH":
        o1 = c + 1.90 * z
        o2 = o1 + 0.60 * z + 1.30 * x
        sym += ["O", "O", "H"]; rows += [o1, o2, o2 + 0.95 * z]
    elif ads:
        raise ValueError(ads)
    return sym, np.vstack([xyz] + [np.atleast_2d(r) for r in rows])


def write_xyz(path: str, sym: Sequence[str], xyz: np.ndarray, comment: str = "") -> None:
    with open(path, "w", newline="\n") as fh:
        fh.write(f"{len(sym)}\n{comment}\n")
        for s, p in zip(sym, xyz):
            fh.write(f"{s:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")


def read_xyz(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0])
    sym, rows = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        sym.append(p[0]); rows.append([float(v) for v in p[1:4]])
    return sym, np.array(rows)


# ----------------------------------------------------------------------------------------------
# geometry checks
# ----------------------------------------------------------------------------------------------

COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Cl": 1.02, "Br": 1.20}


def ligand_bonds(sym: Sequence[str], xyz: np.ndarray, n_lig: int) -> set:
    out = set()
    for i in range(n_lig):
        for j in range(i + 1, n_lig):
            cut = 1.25 * (COVALENT[sym[i]] + COVALENT[sym[j]])
            if np.linalg.norm(xyz[i] - xyz[j]) < cut:
                out.add((i, j))
    return out


def geometry_problems(sym: Sequence[str], xyz: np.ndarray, n_lig: int, ads: str,
                      ref_bonds: Optional[set] = None) -> List[str]:
    """Reasons why an optimised complex is not the species it should be (empty if none)."""
    bad: List[str] = []
    m = n_lig
    lig = xyz[:n_lig]
    ns = [i for i in range(n_lig) if sym[i] == "N"]
    d_n = sorted(np.linalg.norm(xyz[ns] - xyz[m], axis=1))[:4]
    if max(d_n) > 2.25:
        bad.append(f"metal-N {max(d_n):.2f} A")
    if ref_bonds is not None:
        b = ligand_bonds(sym, xyz, n_lig)
        if b != ref_bonds:
            bad.append(f"ring bonds changed ({len(b ^ ref_bonds)} differ)")
    if not ads:
        return bad
    a = list(range(m + 1, len(sym)))
    d_ma = np.linalg.norm(xyz[a[0]] - xyz[m])
    lim = (1.30, 1.85) if ads == "H" else (1.55, 2.25)
    if not lim[0] <= d_ma <= lim[1]:
        bad.append(f"metal-{sym[a[0]]} {d_ma:.2f} A")
    if ads == "H":
        d_lig = np.min(np.linalg.norm(lig - xyz[a[0]], axis=1))
        if d_lig < d_ma:
            bad.append(f"H closer to the ring ({d_lig:.2f} A) than to the metal")
    if ads == "OH" and np.linalg.norm(xyz[a[1]] - xyz[a[0]]) > 1.10:
        bad.append("O-H broken")
    if ads == "OOH":
        d_oo = np.linalg.norm(xyz[a[1]] - xyz[a[0]])
        d_h2 = np.linalg.norm(xyz[a[2]] - xyz[a[1]])
        d_h1 = np.linalg.norm(xyz[a[2]] - xyz[a[0]])
        if not 1.25 <= d_oo <= 1.60:
            bad.append(f"O-O {d_oo:.2f} A")
        if d_h2 > 1.10 or d_h1 < d_h2:
            bad.append("H not on the distal oxygen")
    for i in a:      # an adsorbate atom that has bonded to the ring
        d = np.min(np.linalg.norm(lig - xyz[i], axis=1))
        if sym[i] == "O" and d < 1.60:
            bad.append(f"adsorbate O bonded to the ring ({d:.2f} A)")
    return bad


# ----------------------------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------------------------

@dataclass
class Species:
    porphyrin: str
    metal: str
    ads: str
    mult: int

    @property
    def tag(self) -> str:
        return f"{self.metal}_{self.ads or 'bare'}_m{self.mult}"

    @property
    def folder(self) -> str:
        return os.path.join(WORK, O.safe_name(self.porphyrin, 60), self.tag)


def all_species() -> List[Species]:
    return [Species(p, m, a, s) for p, _, _ in SERIES for m in METALS for a in ADSORBATES
            for s in SPINS[(m, a)]]


def _fb_path(porphyrin: str) -> str:
    return os.path.join(WORK, O.safe_name(porphyrin, 60), "free_base.sdf")


def build_free_bases() -> None:
    for name, subs, aza in SERIES:
        path = _fb_path(name)
        if os.path.isfile(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mol = embed(free_base(name, subs, aza))
        mol.SetProp("_Name", name)
        w = Chem.SDWriter(path)
        w.write(mol)
        w.close()


def load_free_base(porphyrin: str) -> Chem.Mol:
    return Chem.MolFromMolFile(_fb_path(porphyrin), removeHs=False)


def xtb_species(sp: Species, cfg: X.Config) -> Tuple[Species, str]:
    os.makedirs(sp.folder, exist_ok=True)
    geom = os.path.join(sp.folder, "geom.xyz")
    if not os.path.isfile(geom):
        sym, xyz = species_geometry(load_free_base(sp.porphyrin), sp.metal, sp.ads)
        write_xyz(geom, sym, xyz, f"{sp.porphyrin} {sp.tag} start")
    try:
        X.run_job(sp.folder, cfg, 0, sp.mult)
    except X.XtbError as exc:
        return sp, f"xtb: {exc}"
    return sp, ""


def build(jobs: int = 8) -> None:
    build_free_bases()
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ.setdefault("OMP_STACKSIZE", "1G")
    cfg = X.Config(opt=True, fukui=False, timeout=3600)
    todo = all_species()
    t0 = time.time()
    failed = 0
    with ThreadPoolExecutor(jobs) as ex:
        futs = [ex.submit(xtb_species, sp, cfg) for sp in todo]
        for k, f in enumerate(as_completed(futs), 1):
            sp, err = f.result()
            if err:
                failed += 1
                print(f"  {sp.porphyrin} {sp.tag}: {err}", flush=True)
            if k % 100 == 0:
                print(f"  xtb {k}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"xtb: {len(todo)} species, {failed} failed, {time.time() - t0:.0f} s")


def checked_geometry(sp: Species) -> Tuple[Optional[Tuple[List[str], np.ndarray]], List[str]]:
    opt = os.path.join(sp.folder, "xtbopt.xyz")
    if not os.path.isfile(opt):
        return None, ["no xtb geometry"]
    sym, xyz = read_xyz(opt)
    n_lig = sym.index(sp.metal)
    start_sym, start_xyz = read_xyz(os.path.join(sp.folder, "geom.xyz"))
    ref = ligand_bonds(start_sym, start_xyz, n_lig)
    return (sym, xyz), geometry_problems(sym, xyz, n_lig, sp.ads, ref)


def orca_input(mult: int, maxcore: int, careful: bool = False) -> str:
    """Input of a single point.

    `careful` is the second try for a job that did not converge in the time allowed: slow
    convergence settings, a level shift and a lower iteration limit. It has its own input text
    and its own folder, so it never reuses the output of the first try.
    """
    kw = LEVEL + (" UKS" if mult > 1 else "")
    scf = "%scf MaxIter 500 end"
    if careful:
        kw += " SlowConv"
        scf = "%scf MaxIter 300\n  Shift Shift 0.3 ErrOff 0.1 end\nend"
    return (f"! {kw}\n%maxcore {maxcore}\n{scf}\n"
            f"* xyzfile 0 {mult} geom.xyz\n")


def orca_species(sp: Species, cfg: O.Config, careful: bool = False) -> Tuple[Species, str, float]:
    d = os.path.join(sp.folder, "dft_careful" if careful else "dft")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(sp.folder, "xtbopt.xyz"), os.path.join(d, "geom.xyz"))
    try:
        r = O.run_job(d, "sp", orca_input(sp.mult, cfg.maxcore, careful), cfg)
    except O.OrcaError as exc:
        return sp, str(exc)[:300], float("nan")
    return sp, "", r.energy


def gas_species() -> List[Tuple[str, str, int]]:
    """The reference molecules: (name, xyz text body, multiplicity)."""
    return [("H2", "H 0 0 0\nH 0 0 0.742", 1),
            ("H2O", "O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", 1)]


def orca_gas(cfg: O.Config) -> Dict[str, float]:
    out = {}
    for name, body, mult in gas_species():
        d = os.path.join(WORK, "_gas", name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "geom.xyz"), "w", newline="\n") as fh:
            fh.write(f"{len(body.splitlines())}\n{name}\n{body}\n")
        inp = orca_input(mult, cfg.maxcore).replace("! ", "! Opt ", 1)
        out[name] = O.run_job(d, "opt", inp, cfg, need_converged=True).energy
    return out


def series_spins() -> Dict[Tuple[str, str], Tuple[int, ...]]:
    """The spin states to compute for the substituted members: those within
    :data:`SPIN_WINDOW_EV` of the lowest state on the parent. Every state of a species whose
    parent scan is not complete."""
    e = dft_energies()
    out = {}
    for (metal, ads), mults in SPINS.items():
        have = {m: e[("porphine", metal, ads, m)] for m in mults
                if ("porphine", metal, ads, m) in e}
        if len(have) < len(mults):
            out[(metal, ads)] = mults
            continue
        low = min(have.values())
        out[(metal, ads)] = tuple(m for m in mults
                                  if (have[m] - low) * HARTREE_EV <= SPIN_WINDOW_EV)
    return out


#: states that are not tried again: the doublet of a cobalt oxo complex does not converge, and the
#: quartet is the lowest state of the parent (the doublet is 0.14 eV higher)
NO_RETRY = {("Co", "O", 2)}


def has_energy(sp: Species) -> bool:
    for sub in ("dft", "dft_careful"):
        f = os.path.join(sp.folder, sub, "sp.out")
        if os.path.isfile(f) and "ORCA TERMINATED NORMALLY" in open(f, errors="replace").read():
            return True
    return False


def run_orca(jobs: int = 8, maxcore: int = 1500, only: Optional[Sequence[str]] = None,
             retry: bool = False) -> None:
    """The DFT single points. With `retry`, only the species without an energy are run, once
    more with the careful settings of :func:`orca_input` and a limit of 3 hours per job."""
    cfg = O.Config(maxcore=maxcore, timeout=(3 if retry else 6) * 3600)
    gas = orca_gas(cfg)
    print("gas references (Eh):", {k: round(v, 6) for k, v in gas.items()}, flush=True)
    spins = series_spins()
    print("spin states for the substituted members:", spins, flush=True)
    todo = []
    for sp in all_species():
        if only and sp.porphyrin not in only:
            continue
        if sp.porphyrin != "porphine" and sp.mult not in spins[(sp.metal, sp.ads)]:
            continue
        geo, bad = checked_geometry(sp)
        if geo is None or bad:
            continue
        if retry and (has_energy(sp) or (sp.metal, sp.ads, sp.mult) in NO_RETRY):
            continue
        todo.append((len(geo[0]), sp))
    todo.sort(key=lambda t: -t[0])          # the largest first, so the tail is short
    t0 = time.time()
    failed = 0
    with ThreadPoolExecutor(jobs) as ex:
        futs = [ex.submit(orca_species, sp, cfg, retry) for _, sp in todo]
        for k, f in enumerate(as_completed(futs), 1):
            sp, err, _ = f.result()
            if err:
                failed += 1
                print(f"  {sp.porphyrin} {sp.tag}: {err}", flush=True)
            if k % 20 == 0 or k == len(todo):
                print(f"  ORCA {k}/{len(todo)} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
    print(f"ORCA: {len(todo)} species, {failed} failed")


# ----------------------------------------------------------------------------------------------
# free energies
# ----------------------------------------------------------------------------------------------

def _xtb_g(folder: str, charge: int, mult: int, xyz_name: str) -> float:
    """GFN2-xTB free energy (Eh) from a frequency job on an optimised geometry."""
    d = os.path.join(folder, "thermo")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(folder, xyz_name), os.path.join(d, "geom.xyz"))
    out = os.path.join(d, "xtb.out")
    if not (os.path.isfile(out) and "normal termination" in open(out, errors="replace").read()):
        args = [X.find_xtb(), "geom.xyz", "--gfn", "2", "--chrg", str(charge),
                "--uhf", str(mult - 1), "--hess"]
        p = subprocess.run(args, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=3600)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(p.stdout + p.stderr)
    text = open(out, encoding="utf-8", errors="replace").read()
    for line in text.splitlines():
        if "G(RRHO) contrib." in line:
            return float(line.split()[3])
    raise RuntimeError(f"no G(RRHO) in {out}")


def thermo() -> Dict[str, Dict[str, float]]:
    """The zero-point and entropy corrections dZPE-TdS (eV) of each adsorbate, per metal.

    G(RRHO) of the adsorbate complex minus that of the bare complex, minus the same for the gas
    references, from GFN2-xTB frequencies of the unsubstituted porphine complexes in their
    lowest DFT spin state. The gas references are optimised with xtb first.
    """
    os.environ["OMP_NUM_THREADS"] = "4"
    gas = {}
    for name, body, mult in gas_species():
        d = os.path.join(WORK, "_gas", name, "xtb")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "geom.xyz"), "w", newline="\n") as fh:
            fh.write(f"{len(body.splitlines())}\n{name}\n{body}\n")
        X.run_job(d, X.Config(opt=True, fukui=False), 0, mult)
        gas[name] = _xtb_g(d, 0, mult, "xtbopt.xyz")
    energies = dft_energies()
    out: Dict[str, Dict[str, float]] = {}
    for metal in METALS:
        g = {}
        for ads in ADSORBATES:
            best = lowest_spin(energies, "porphine", metal, ads)
            if best is None:
                raise RuntimeError(f"porphine {metal} {ads}: no DFT energy yet")
            sp = Species("porphine", metal, ads, best[0])
            g[ads] = _xtb_g(sp.folder, 0, sp.mult, "xtbopt.xyz")
        ref = {"H": 0.5 * gas["H2"],
               "OH": gas["H2O"] - 0.5 * gas["H2"],
               "O": gas["H2O"] - gas["H2"],
               "OOH": 2 * gas["H2O"] - 1.5 * gas["H2"]}
        out[metal] = {a: (g[a] - g[""] - ref[a]) * HARTREE_EV for a in ADSORBATES if a}
    path = os.path.join(WORK, "thermo.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return out


def dft_energies() -> Dict[Tuple[str, str, str, int], float]:
    out = {}
    for sp in all_species():
        for sub in ("dft", "dft_careful"):
            f = os.path.join(sp.folder, sub, "sp.out")
            if os.path.isfile(f):
                r = O.parse_orca_output(open(f, errors="replace").read())
                if r.terminated and np.isfinite(r.energy):
                    out[(sp.porphyrin, sp.metal, sp.ads, sp.mult)] = r.energy
                    break
    return out


def lowest_spin(energies, porphyrin: str, metal: str, ads: str):
    cands = [(m, energies[(porphyrin, metal, ads, m)]) for m in SPINS[(metal, ads)]
             if (porphyrin, metal, ads, m) in energies]
    if not cands:
        return None
    return min(cands, key=lambda t: t[1])


def collect() -> str:
    cfg = O.Config()
    gas = {}
    for name, _, _ in gas_species():
        f = os.path.join(WORK, "_gas", name, "opt.out")
        gas[name] = O.parse_orca_output(open(f, errors="replace").read()).energy
    corr = json.load(open(os.path.join(WORK, "thermo.json")))
    energies = dft_energies()
    ref = {"H": 0.5 * gas["H2"], "OH": gas["H2O"] - 0.5 * gas["H2"],
           "O": gas["H2O"] - gas["H2"], "OOH": 2 * gas["H2O"] - 1.5 * gas["H2"]}
    rows = []
    for name, subs, aza in SERIES:
        for metal in METALS:
            row: Dict[str, object] = {"name": f"{metal}-{name}", "porphyrin": name,
                                      "metal": metal}
            bare = lowest_spin(energies, name, metal, "")
            notes = []
            for ads in ADSORBATES:
                best = lowest_spin(energies, name, metal, ads)
                key = ads or "bare"
                if best is None:
                    probs = []
                    for m in SPINS[(metal, ads)]:
                        _, bad = checked_geometry(Species(name, metal, ads, m))
                        probs.append(f"m{m}: " + ("; ".join(bad) or "no DFT energy"))
                    notes.append(f"{key}: " + ", ".join(probs))
                    row[f"mult_{key}"] = ""
                    if ads:
                        row[f"dG_{ads}"] = ""
                    continue
                row[f"mult_{key}"] = best[0]
                if ads and bare is not None:
                    dg = (best[1] - bare[1] - ref[ads]) * HARTREE_EV + corr[metal][ads]
                    row[f"dG_{ads}"] = round(dg, 4)
            ok = all(row.get(f"dG_{a}") not in ("", None) for a in ("OH", "O", "OOH"))
            if ok:
                g_oh, g_o, g_ooh = (float(row[f"dG_{a}"]) for a in ("OH", "O", "OOH"))
                steps = [g_ooh - 4.92, g_o - g_ooh, g_oh - g_o, -g_oh]
                ul = -max(steps)
                row["U_L_ORR"] = round(ul, 4)
                row["eta_ORR"] = round(1.23 - ul, 4)
            else:
                row["U_L_ORR"] = row["eta_ORR"] = ""
            if row.get("dG_H") not in ("", None) and row.get("dG_OH") not in ("", None):
                row["dG_H_minus_dG_OH"] = round(float(row["dG_H"]) - float(row["dG_OH"]), 4)
            else:
                row["dG_H_minus_dG_OH"] = ""
            row["substituents"] = ";".join(f"{p}:{x}" for p, x in sorted(subs.items()))
            row["aza"] = ";".join(str(p) for p in aza)
            row["notes"] = " | ".join(notes)
            rows.append(row)
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["name", "porphyrin", "metal", "substituents", "aza", "dG_H", "dG_OH", "dG_O",
            "dG_OOH", "U_L_ORR", "eta_ORR", "dG_H_minus_dG_OH", "mult_bare", "mult_H",
            "mult_OH", "mult_O", "mult_OOH", "notes"]
    path = os.path.join(OUT_DIR, "porphyrins.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    write_structures(energies)
    print(f"wrote {path} ({len(rows)} complexes)")
    return path


def write_structures(energies) -> None:
    """One extended-xyz file with the lowest-spin geometry of every species that has energies."""
    path = os.path.join(OUT_DIR, "structures.extxyz")
    n = 0
    with open(path, "w", newline="\n") as fh:
        for name, _, _ in SERIES:
            for metal in METALS:
                for ads in ADSORBATES:
                    best = lowest_spin(energies, name, metal, ads)
                    if best is None:
                        continue
                    sp = Species(name, metal, ads, best[0])
                    sym, xyz = read_xyz(os.path.join(sp.folder, "xtbopt.xyz"))
                    fh.write(f"{len(sym)}\n")
                    fh.write(f'Properties=species:S:1:pos:R:3 name="{metal}-{name}" '
                             f'adsorbate="{ads or "none"}" multiplicity={best[0]} charge=0 '
                             f'E_dft_Eh={best[1]:.8f} pbc="F F F"\n')
                    for s, p in zip(sym, xyz):
                        fh.write(f"{s:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
                    n += 1
    print(f"wrote {path} ({n} structures)")


def summary() -> None:
    sps = all_species()
    n_geo = n_bad = 0
    bad_lines = []
    for sp in sps:
        geo, bad = checked_geometry(sp)
        if geo is not None:
            n_geo += 1
        if geo is not None and bad:
            n_bad += 1
            bad_lines.append(f"  {sp.porphyrin:22s} {sp.tag:14s} {'; '.join(bad)}")
    e = dft_energies()
    print(f"{len(SERIES)} porphyrins x {len(METALS)} metals; {len(sps)} species-spin states")
    print(f"xtb geometries: {n_geo}; failing the checks: {n_bad}; DFT energies: {len(e)}")
    for ln in bad_lines:
        print(ln)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--build", action="store_true", help="structures and xtb geometries")
    ap.add_argument("--orca", action="store_true", help="the DFT single points")
    ap.add_argument("--thermo", action="store_true", help="xtb corrections from the parents")
    ap.add_argument("--collect", action="store_true", help="write porphyrin/porphyrins.csv")
    ap.add_argument("--summary", action="store_true", help="what has been computed")
    ap.add_argument("--retry", action="store_true",
                    help="ORCA stage: only the species without an energy, with careful SCF settings")
    ap.add_argument("--only", default=None, help="one porphyrin by name (ORCA stage)")
    ap.add_argument("--batch", choices=("first", "all"), default="all",
                    help="'first': the parent and the 20 members of FIRST_BATCH (ORCA stage)")
    ap.add_argument("--jobs", type=int, default=8, help="jobs at a time (ORCA here is serial)")
    ap.add_argument("--maxcore", type=int, default=1500, help="MB per ORCA job")
    a = ap.parse_args()
    if a.build:
        build(a.jobs)
    if a.orca:
        only = FIRST_BATCH if a.batch == "first" else ([a.only] if a.only else None)
        run_orca(a.jobs, a.maxcore, only, a.retry)
    if a.thermo:
        thermo()
    if a.collect:
        collect()
    if a.summary or not (a.build or a.orca or a.thermo or a.collect):
        summary()


if __name__ == "__main__":
    main()
