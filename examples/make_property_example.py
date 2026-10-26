"""Write an example data set carrying per-atom properties, for the `prop:` fields.

A field whose atomic weights are read from a property of the file is how a quantity the program
cannot compute itself -- a quantum charge, a Fukui index, a local softness, an atomic
polarisability -- becomes a molecular field.  Shipping quantum-chemical data with the program is
a different problem, so this script writes properties it can compute, in both conventions the
reader understands:

* ``qgasteiger`` as an **atom property**, which RDKit writes to the SDF as ``atom.dprop.qgasteiger``
  and reads back automatically.  It holds the same Gasteiger charges the electrostatic field uses,
  which makes the file a test as much as an example: a field built from this property must
  reproduce the electrostatic field exactly.
* ``crippen_logp`` as a **molecule property** holding one number per atom, separated by spaces,
  which is how most quantum-chemistry packages write such a table.  These are the Wildman-Crippen
  atomic logP contributions, so the field is a per-atom lipophilicity of a different kind from
  the program's own hydrophobic field.

Usage:
    python examples/make_property_example.py [--source examples/sutherland/THR.sdf] [--n 40]
"""
import argparse
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

RDLogger.DisableLog("rdApp.*")

SOURCE = ("per-atom properties for the prop: fields of cmfqsar: qgasteiger as an atom property, "
          "crippen_logp as a molecule property with one number per atom "
          "(examples/make_property_example.py)")


def annotate(mol: Chem.Mol) -> Chem.Mol:
    """Attach the two properties to a copy of `mol`."""
    m = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(m)
    for atom in m.GetAtoms():
        q = atom.GetDoubleProp("_GasteigerCharge")
        atom.SetDoubleProp("qgasteiger", 0.0 if q != q else float(q))      # q != q catches nan
    # RDKit writes atom properties to SDF only through a property list, which the reader then
    # turns back into atom properties; without this the numbers are not written to the file
    Chem.CreateAtomDoublePropertyList(m, "qgasteiger")
    contribs = [c[0] for c in Crippen.rdMolDescriptors._CalcCrippenContribs(m)]
    m.SetProp("crippen_logp", " ".join(f"{c:.4f}" for c in contribs))
    m.SetProp("source", SOURCE)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(HERE, "sutherland", "THR.sdf"))
    ap.add_argument("--n", type=int, default=40, help="how many molecules to take")
    ap.add_argument("--out", default=os.path.join(HERE, "thrombin_atom_properties.sdf"))
    a = ap.parse_args()
    suppl = Chem.SDMolSupplier(a.source, removeHs=False)
    w = Chem.SDWriter(a.out)
    n = 0
    for mol in suppl:
        if mol is None:
            continue
        w.write(annotate(mol))
        n += 1
        if n >= a.n:
            break
    w.close()
    print(f"{n} molecules with per-atom properties written to {a.out}")
    print("  try:  python cmf_cli.py build " + os.path.relpath(a.out, os.path.join(HERE, "..")) +
          " --fields steric,prop:qgasteiger,prop:crippen_logp --alphas 0.5 --lams 0.1")


if __name__ == "__main__":
    main()
