"""Substituted benzenes in one frame, with their Hammett constants, for a field-based Hammett model.

A substituent constant is a property of a fragment attached to a fixed scaffold at a fixed
position, so the one hard requirement of CMF -- that the molecules share a frame -- is met by
construction: put the benzene ring of every X-C6H5 in the same place and every substituent sits
where the reaction centre of benzoic acid would see it.  With a linear kernel the CMF prediction
is a linear functional of the molecular field, so a Hammett constant becomes the integral of the
substituent's field against a learned weight function, the co-field, which is the sensitivity map
of benzoic acid ionisation to charge and bulk around the ring.  sigma_m and sigma_p should then
give co-fields that differ in *where* they sit, meta or para, and that is what the example is for.

The frame
---------
Ring centroid at the origin, the ipso carbon (the one carrying X) on the +x axis, the ring in the
xy plane.  The ring has C2v symmetry about the C1-C4 axis, so two reflections (y -> -y, z -> -z)
map the frame onto itself while turning a substituent conformer into an equally valid one; which of
the four images a conformer generator happens to produce is noise.  The single-conformer file
breaks the tie by a convention (the substituent's heavy atoms lean towards +y and +z); the
symmetrised file writes all four images of every molecule as conformers of one compound, so that
the ensemble mode of the field set (``--ensemble-by name --conformations ensemble``) averages them
and the field has the symmetry of the problem.

The constants
-------------
sigma_m, sigma_p and the Hansch hydrophobic constant pi are transcribed from the compilation of
Hansch, Leo and Taft, Chem. Rev. 1991, 91, 165 (Table I), from memory of that table rather than
from a machine-readable copy; the values are the familiar ones for the familiar substituents, but
check any single number against the table before quoting it.  Charged substituents (NMe3+, COO-,
O-) are left out: their electrostatic field would dominate the kernel.  pi is given where it is
well established and omitted otherwise.

Outputs
-------
    examples/hammett/hammett.sdf        one conformer per substituent, in frame
    examples/hammett/hammett_sym.sdf    the same, with the four symmetry images as conformers

Usage
-----
    python examples/make_hammett_examples.py
    python examples/make_hammett_examples.py --symmetrize examples/hammett/hammett_qm.sdf

The second form symmetrises any SDF already in frame -- the output of
``examples/orca_atom_properties.py`` run on hammett.sdf, say -- carrying every molecule and atom
property (quantum charges included; the images of an atom have the same charge) to the four
conformers.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

RDLogger.DisableLog("rdApp.*")

OUT_DIR = os.path.join(HERE, "hammett")
SOURCE = ("Hammett sigma_m / sigma_p and Hansch pi after Hansch, Leo & Taft, Chem. Rev. 1991, 91, "
          "165, Table I; structures X-C6H5 built with RDKit ETKDG + MMFF94 and placed with the ring "
          "centroid at the origin, the ipso carbon on +x and the ring in the xy plane "
          "(examples/make_hammett_examples.py)")

#: (label, substituent SMILES as written after "c1ccccc1", sigma_m, sigma_p, pi or None)
SUBSTITUENTS: List[Tuple[str, str, float, float, Optional[float]]] = [
    ("H", "", 0.00, 0.00, 0.00),
    # alkyl and related
    ("CH3", "C", -0.07, -0.17, 0.56),
    ("C2H5", "CC", -0.07, -0.15, 1.02),
    ("n-C3H7", "CCC", -0.06, -0.13, 1.55),
    ("i-C3H7", "C(C)C", -0.07, -0.15, 1.53),
    ("n-C4H9", "CCCC", -0.08, -0.16, 2.13),
    ("t-C4H9", "C(C)(C)C", -0.10, -0.20, 1.98),
    ("cyclopropyl", "C1CC1", -0.07, -0.21, 1.14),
    ("cyclohexyl", "C1CCCCC1", -0.05, -0.15, 2.51),
    ("CH2Ph", "Cc1ccccc1", -0.08, -0.09, 2.01),
    ("allyl", "CC=C", -0.11, -0.14, 1.10),
    ("CH2SiMe3", "C[Si](C)(C)C", -0.16, -0.21, None),
    ("vinyl", "C=C", 0.06, -0.04, 0.82),
    ("ethynyl", "C#C", 0.21, 0.23, 0.40),
    ("Ph", "c1ccccc1", 0.06, -0.01, 1.96),
    # halogenated alkyl
    ("CF3", "C(F)(F)F", 0.43, 0.54, 0.88),
    ("CHF2", "C(F)F", 0.29, 0.32, None),
    ("CH2F", "CF", 0.12, 0.11, None),
    ("CCl3", "C(Cl)(Cl)Cl", 0.40, 0.46, None),
    ("CHCl2", "C(Cl)Cl", 0.31, 0.32, None),
    ("CBr3", "C(Br)(Br)Br", 0.28, 0.29, None),
    ("CH2Cl", "CCl", 0.11, 0.12, 0.17),
    ("CH2Br", "CBr", 0.12, 0.14, 0.79),
    ("CH2I", "CI", 0.10, 0.11, None),
    ("C2F5", "C(F)(F)C(F)(F)F", 0.47, 0.52, None),
    ("C(CF3)3", "C(C(F)(F)F)(C(F)(F)F)C(F)(F)F", 0.55, 0.55, None),
    # functionalised methyl
    ("CH2OH", "CO", 0.00, 0.00, -1.03),
    ("CH2OCH3", "COC", 0.02, 0.03, None),
    ("CH2CN", "CC#N", 0.16, 0.18, None),
    ("CH2NH2", "CN", -0.03, -0.11, None),
    # halogens
    ("F", "F", 0.34, 0.06, 0.14),
    ("Cl", "Cl", 0.37, 0.23, 0.71),
    ("Br", "Br", 0.39, 0.23, 0.86),
    ("I", "I", 0.35, 0.18, 1.12),
    # oxygen
    ("OH", "O", 0.12, -0.37, -0.67),
    ("OCH3", "OC", 0.12, -0.27, -0.02),
    ("OC2H5", "OCC", 0.10, -0.24, 0.38),
    ("O-i-C3H7", "OC(C)C", 0.10, -0.45, None),
    ("OPh", "Oc1ccccc1", 0.25, -0.03, 2.08),
    ("OCF3", "OC(F)(F)F", 0.38, 0.35, 1.04),
    ("OCOCH3", "OC(C)=O", 0.39, 0.31, -0.64),
    ("OCOPh", "OC(=O)c1ccccc1", 0.21, 0.13, 1.46),
    ("OSO2CH3", "OS(C)(=O)=O", 0.39, 0.36, -0.88),
    # sulfur and selenium
    ("SH", "S", 0.25, 0.15, 0.39),
    ("SCH3", "SC", 0.15, 0.00, 0.61),
    ("SC2H5", "SCC", 0.18, 0.03, 1.07),
    ("SPh", "Sc1ccccc1", 0.23, 0.07, 2.32),
    ("SCF3", "SC(F)(F)F", 0.40, 0.50, 1.44),
    ("SCN", "SC#N", 0.51, 0.52, 0.41),
    ("SCOCH3", "SC(C)=O", 0.39, 0.44, None),
    ("SOCH3", "S(C)=O", 0.52, 0.49, -1.58),
    ("SO2CH3", "S(C)(=O)=O", 0.60, 0.72, -1.63),
    ("SO2CF3", "S(=O)(=O)C(F)(F)F", 0.83, 0.96, 0.55),
    ("SO2Ph", "S(=O)(=O)c1ccccc1", 0.62, 0.68, 0.27),
    ("SO2NH2", "S(N)(=O)=O", 0.53, 0.60, -1.82),
    ("SO2F", "S(F)(=O)=O", 0.80, 0.91, 0.05),
    ("SO2Cl", "S(Cl)(=O)=O", 1.20, 1.11, None),
    ("SF5", "S(F)(F)(F)(F)F", 0.61, 0.68, 1.23),
    ("SeCH3", "[Se]C", 0.10, 0.00, 0.74),
    # nitrogen
    ("NH2", "N", -0.16, -0.66, -1.23),
    ("NHCH3", "NC", -0.21, -0.70, -0.47),
    ("N(CH3)2", "N(C)C", -0.16, -0.83, 0.18),
    ("N(C2H5)2", "N(CC)CC", -0.23, -0.72, 1.18),
    ("NHNH2", "NN", -0.02, -0.55, -0.88),
    ("NHCHO", "NC=O", 0.19, 0.00, -0.98),
    ("NHCOCH3", "NC(C)=O", 0.21, 0.00, -0.97),
    ("NHCOCF3", "NC(=O)C(F)(F)F", 0.30, 0.12, 0.08),
    ("NHCONH2", "NC(N)=O", -0.03, -0.24, -1.30),
    ("NHSO2CH3", "NS(C)(=O)=O", 0.20, 0.03, -1.18),
    ("N(CH3)COCH3", "N(C)C(C)=O", 0.31, 0.26, None),
    ("N(CF3)2", "N(C(F)(F)F)C(F)(F)F", 0.47, 0.53, None),
    ("NO2", "[N+](=O)[O-]", 0.71, 0.78, -0.28),
    ("NO", "N=O", 0.62, 0.91, None),
    ("N3", "N=[N+]=[N-]", 0.37, 0.08, 0.46),
    ("NCO", "N=C=O", 0.27, 0.19, None),
    ("NCS", "N=C=S", 0.48, 0.38, 1.15),
    ("NC", "[N+]#[C-]", 0.48, 0.49, None),
    ("N=NPh", "N=Nc1ccccc1", 0.32, 0.39, 1.69),
    # carbon with heteroatoms
    ("CN", "C#N", 0.56, 0.66, -0.57),
    ("CHO", "C=O", 0.35, 0.42, -0.65),
    ("COCH3", "C(C)=O", 0.38, 0.50, -0.55),
    ("COPh", "C(=O)c1ccccc1", 0.34, 0.43, 1.05),
    ("COCF3", "C(=O)C(F)(F)F", 0.63, 0.80, 0.02),
    ("COOH", "C(=O)O", 0.37, 0.45, -0.32),
    ("COOCH3", "C(=O)OC", 0.37, 0.45, -0.01),
    ("COOC2H5", "C(=O)OCC", 0.37, 0.45, 0.51),
    ("CONH2", "C(N)=O", 0.28, 0.36, -1.49),
    ("COCl", "C(Cl)=O", 0.51, 0.61, None),
    ("COF", "C(F)=O", 0.55, 0.70, None),
    # silicon, boron, phosphorus
    ("Si(CH3)3", "[Si](C)(C)C", -0.04, -0.07, 2.59),
    ("B(OH)2", "B(O)O", -0.01, 0.12, -0.55),
    ("P(O)(OEt)2", "P(=O)(OCC)OCC", 0.55, 0.60, None),
    ("P(CH3)2", "P(C)C", 0.03, 0.06, None),
    ("P(O)(CH3)2", "P(C)(C)=O", 0.43, 0.50, None),
    ("PPh2", "P(c1ccccc1)c1ccccc1", 0.11, 0.19, None),
]

RING = "c1ccccc1"
RING_ATOMS = list(range(6))     # atom indices of the ring in RING + fragment
IPSO = 5                        # the ring atom the fragment is written after


def smiles_for(fragment: str) -> str:
    return RING + fragment


def _frame(coords: np.ndarray) -> np.ndarray:
    """Rotation matrix (rows = new axes) and translation putting the ring into the frame."""
    ring = coords[RING_ATOMS]
    centre = ring.mean(axis=0)
    x = coords[IPSO] - centre
    x /= np.linalg.norm(x)
    # the ring normal: the direction of least extent of the ring atoms
    _, _, vt = np.linalg.svd(ring - centre)
    z = vt[2]
    z -= x * np.dot(z, x)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.vstack([x, y, z]), centre


def place_in_frame(coords: np.ndarray, lean_positive: bool = True) -> np.ndarray:
    """Coordinates in the ring frame: centroid at the origin, ipso on +x, ring in z = 0.

    With ``lean_positive`` the two reflections that preserve the frame are used to put the centroid
    of the atoms outside the ring on y >= 0 and z >= 0, so that every molecule is written in the
    same image.  Both reflections are symmetries of the ring, so the result is a conformer of the
    same molecule, not its enantiomer in any sense that matters here.
    """
    R, centre = _frame(coords)
    out = (coords - centre) @ R.T
    if lean_positive and len(out) > 6:
        rest = np.delete(out, RING_ATOMS, axis=0)
        # the ring hydrogens lie in the plane and lean neither way; the substituent decides
        far = rest[np.linalg.norm(rest[:, :2], axis=1) > 1.6]
        probe = far if len(far) else rest
        c = probe.mean(axis=0)
        if c[1] < -1e-6:
            out[:, 1] *= -1
        if c[2] < -1e-6:
            out[:, 2] *= -1
    return out


def build(fragment: str, n_conformers: int = 30, seed: int = 2026) -> Chem.Mol:
    """X-C6H5 with hydrogens, MMFF-optimised, lowest-energy conformer, placed in the frame."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles_for(fragment)))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        raise RuntimeError(f"embedding failed for {smiles_for(fragment)}")
    # MMFF on a copy: its set-up re-perceives aromaticity in place
    work = Chem.Mol(mol)
    results = AllChem.MMFFOptimizeMoleculeConfs(work, maxIters=2000)
    energies = np.array([e if conv == 0 or e == e else np.inf for conv, e in results], float)
    if not np.isfinite(energies).any():
        energies = np.array([e for _, e in results], float)
    best = ids[int(np.argmin(energies))]
    coords = np.array(work.GetConformer(best).GetPositions(), float)
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(place_in_frame(coords)):
        conf.SetAtomPosition(i, p.tolist())
    out.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(out)
    out.SetDoubleProp("mmff_energy", float(energies.min()))
    return out


def make_dataset(n_conformers: int = 30, seed: int = 2026) -> List[Chem.Mol]:
    mols = []
    for label, frag, sm, sp, pi in SUBSTITUENTS:
        m = build(frag, n_conformers=n_conformers, seed=seed)
        m.SetProp("_Name", label)
        m.SetProp("substituent", label)
        m.SetProp("smiles", Chem.MolToSmiles(Chem.RemoveHs(m)))
        m.SetDoubleProp("sigma_m", sm)
        m.SetDoubleProp("sigma_p", sp)
        if pi is not None:
            m.SetDoubleProp("pi", pi)
        m.SetProp("source", SOURCE)
        mols.append(m)
    return mols


def symmetry_images(mol: Chem.Mol) -> List[Chem.Mol]:
    """The four images of a molecule in frame under the C2v symmetry of the ring frame.

    Every property of the molecule and of its atoms is carried over unchanged, which is right for
    anything scalar: a charge, a Fukui index, a hydrophobic increment.
    """
    base = np.array(mol.GetConformer().GetPositions(), float)
    out = []
    for sy, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        m = Chem.Mol(mol)
        conf = m.GetConformer()
        for i, p in enumerate(base * np.array([1.0, sy, sz])):
            conf.SetAtomPosition(i, p.tolist())
        m.SetProp("symmetry_image", f"y{sy:+d} z{sz:+d}")
        out.append(m)
    return out


def write_sdf(path: str, mols: Sequence[Chem.Mol], atom_props: Sequence[str] = ()) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = Chem.SDWriter(path)
    for m in mols:
        mm = Chem.Mol(m)
        for p in atom_props:
            if all(a.HasProp(p) for a in mm.GetAtoms()):
                Chem.CreateAtomDoublePropertyList(mm, p)
        w.write(mm)
    w.close()


def atom_property_names(mols: Sequence[Chem.Mol]) -> List[str]:
    """Numeric atom properties present on every atom of every molecule (private ones aside)."""
    names = None
    for m in mols:
        for a in m.GetAtoms():
            here = {k for k in a.GetPropNames() if not k.startswith("_")}
            names = here if names is None else names & here
    out = []
    for k in sorted(names or ()):
        try:
            float(mols[0].GetAtomWithIdx(0).GetProp(k))
            out.append(k)
        except ValueError:
            pass
    return out


def symmetrize_file(src: str, dst: str) -> int:
    """Write the four symmetry images of every molecule of an SDF already in frame."""
    suppl = Chem.SDMolSupplier(src, removeHs=False)
    mols = [m for m in suppl if m is not None]
    props = atom_property_names(mols)
    images = []
    for m in mols:
        images.extend(symmetry_images(m))
    write_sdf(dst, images, atom_props=props)
    return len(mols)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--conformers", type=int, default=30, help="ETKDG conformers tried per molecule")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--symmetrize", metavar="SDF",
                    help="instead of building: symmetrise this SDF (already in frame) into -o")
    ap.add_argument("-o", "--out", help="output of --symmetrize (default: <stem>_sym.sdf)")
    a = ap.parse_args(argv)
    if a.symmetrize:
        dst = a.out or os.path.splitext(a.symmetrize)[0] + "_sym.sdf"
        n = symmetrize_file(a.symmetrize, dst)
        print(f"{n} molecules x 4 images written to {dst}")
        return 0
    mols = make_dataset(n_conformers=a.conformers, seed=a.seed)
    single = os.path.join(a.out_dir, "hammett.sdf")
    sym = os.path.join(a.out_dir, "hammett_sym.sdf")
    write_sdf(single, mols)
    images = []
    for m in mols:
        images.extend(symmetry_images(m))
    write_sdf(sym, images)
    n_pi = sum(1 for m in mols if m.HasProp("pi"))
    print(f"{len(mols)} substituted benzenes written to {single} "
          f"(sigma_m, sigma_p for all, pi for {n_pi}); four symmetry images each in {sym}")
    rel = os.path.relpath(sym, os.path.join(HERE, ".."))
    print("  try:  python cmf_cli.py build " + rel + " --activity sigma_p --ensemble-by name "
          "--conformations ensemble --alphas 0.5,1,2 --lams 0.01,0.1,1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
