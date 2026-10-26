"""
Generate the synthetic benchmark data set ``examples/benzamides_synthetic.sdf``.

A congeneric series of substituted N-phenyl-benzamides is enumerated, 3D
structures are generated with RDKit (ETKDG + MMFF94), all molecules are aligned
on the common scaffold, and a *synthetic* activity is computed as the overlap
of the molecular fields with a hidden "pseudo-receptor" co-field consisting of
a few Gaussian zones (steric, electrostatic, hydrophobic and H-bond-acceptor
zones) plus Gaussian noise.  Because the activity is by construction a CMF-type
functional of the molecular fields, the data set is a good basic check for
the software: a CMF model should recover the hidden co-field zones.

Run:  python examples/make_example_dataset.py
"""
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.align import mcs_align, embed_3d  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, MoleculeFields, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "benzamides_synthetic.sdf")

# scaffold: ring A (left, carboxamide side) positions 3/4 ; ring B (anilide) positions 3'/4'
#                       R4                       R4'
#  R3 - c1cc(C(=O)Nc2ccc(R4')c(R3')c2)ccc1 - ...
SUBST = {
    "H": "[H]", "F": "F", "Cl": "Cl", "Br": "Br", "Me": "C", "Et": "CC", "iPr": "C(C)C", "tBu": "C(C)(C)C",
    "OMe": "OC", "OH": "O", "NH2": "N", "NMe2": "N(C)C", "NO2": "[N+](=O)[O-]", "CF3": "C(F)(F)F",
    "CN": "C#N", "COOMe": "C(=O)OC", "Ac": "C(C)=O", "SMe": "SC", "OEt": "OCC", "Ph": "c1ccccc1",
}

rng = np.random.RandomState(2013)


def make_smiles(r3, r4, r3p, r4p):
    def s(x):
        return "" if x == "H" else "(" + SUBST[x] + ")"
    return f"O=C(Nc1ccc{s(r4p)}c{s(r3p)}c1)c1ccc{s(r4)}c{s(r3)}c1"


def enumerate_series(n=70):
    keys = list(SUBST)
    seen = set()
    out = []
    # a few systematic members
    for k in keys:
        out.append(("H", k, "H", "H"))
        out.append(("H", "H", "H", k))
    while len(out) < n + 2 * len(keys):
        combo = tuple(rng.choice(keys, size=4, p=None))
        n_nonH = sum(1 for c in combo if c != "H")
        if n_nonH > 3:
            continue
        out.append(combo)
    res = []
    for combo in out:
        smi = make_smiles(*combo)
        can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if can in seen:
            continue
        seen.add(can)
        name = "-".join(f"{p}{c}" for p, c in zip(("3", "4", "3'", "4'"), combo) if c != "H") or "parent"
        res.append((name, smi))
    return res


def pseudo_receptor(fieldset):
    """Hidden co-field: Gaussian zones expressed as a pseudo-molecule (centres/weights/radii)."""
    # coordinates are defined relative to the aligned scaffold (see below: reference molecule frame)
    zones = [  # (field, x, y, z, weight, radius)
        ("steric", 6.3, 0.0, 0.0, +0.9, 1.9),      # bulky groups favourable at 4' (para of anilide)
        ("steric", -6.3, 0.0, 0.0, -0.9, 1.9),     # bulky groups unfavourable at 4 (para of benzoyl)
        ("electrostatic", 5.0, 2.3, 0.0, -2.5, 1.4),   # negative charge favourable near 3'
        ("hydrophobic", 6.3, 0.0, 0.0, +0.8, 1.9),  # hydrophobic favourable at 4'
        ("hba", -5.0, 2.3, 0.0, +1.2, 1.5),         # acceptor favourable at 3
    ]
    F = fieldset.n_fields
    names = fieldset.names
    coords, W, radii = [], [], []
    for fname, x, y, z, w, r in zones:
        coords.append([x, y, z])
        row = np.zeros(F)
        row[names.index(fname)] = w
        W.append(row)
        radii.append(r)
    return MoleculeFields("receptor", np.array(coords), np.array(W), np.array(radii),
                          ["X"] * len(zones), ["X"] * len(zones), np.zeros((0, 2), int), np.zeros(len(zones)))


def main():
    series = enumerate_series(70)
    print(f"{len(series)} unique molecules")
    mols = []
    for k, (name, smi) in enumerate(series):
        m = embed_3d(Chem.MolFromSmiles(smi), seed=k)
        m.SetProp("_Name", name)
        mols.append(m)

    # align on the parent compound (first entry) using the common substructure
    ref_idx = next(i for i, (n, _) in enumerate(series) if n == "parent")
    ref = mols[ref_idx]
    # put the reference into a canonical frame: amide C-N bond centre at the origin, long axis along x
    conf = ref.GetConformer()
    pos = np.array(conf.GetPositions())
    heavy = [a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() > 1]
    c = pos[heavy].mean(axis=0)
    pos -= c
    u, s, vt = np.linalg.svd(pos[heavy], full_matrices=False)
    pos = pos @ vt.T
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    for i, m in enumerate(mols):
        if i != ref_idx:
            mcs_align(m, ref, timeout=5)

    fs = FieldSet.create(physchem=("electrostatic", "steric", "hydrophobic", "hba", "hbd"), alpha=0.6)
    fs.specs[1].kind = "gauss"   # the pseudo-receptor uses plain Gaussians for every zone
    mf = [build_fields(m, fs, n) for m, (n, _) in zip(mols, series)]
    rec = pseudo_receptor(fs)
    # widths for the receptor zones: encode as plain Gaussians via the alpha/R^2 relation -> use kernel with kind gauss
    table = AtomTable.from_molecules(mf)
    rtab = AtomTable.from_molecules([rec])
    K = kernel_matrices(table, rtab, fs.specs)  # (F, n, 1)
    contrib = K[:, :, 0]
    y = contrib.sum(axis=0)
    y = (y - y.mean()) / y.std() * 1.0 + 6.5
    y += rng.normal(scale=0.25, size=len(y))

    w = Chem.SDWriter(OUT)
    for m, (name, smi), val, cont in zip(mols, series, y, contrib.T):
        m.SetProp("_Name", name)
        m.SetProp("pActivity", f"{val:.3f}")
        m.SetProp("SMILES", smi)
        m.SetProp("hidden_contributions", " ".join(f"{fs.names[f]}={cont[f]:.3f}" for f in range(len(cont))))
        w.write(m)
    w.close()
    print("activity range", y.min(), y.max(), "->", OUT)


if __name__ == "__main__":
    main()
