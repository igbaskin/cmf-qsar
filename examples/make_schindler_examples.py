"""
Build ``examples/schindler/<target>.sdf`` from the Merck KGaA FEP benchmark

    Schindler C.E.M., Baumann H., Blum A., et al. "Large-scale assessment of binding free energy
    calculations in active drug discovery projects", J. Chem. Inf. Model. 60 (2020) 5457-5474,
    https://github.com/MCompChem/fep-benchmark (MIT licence).

Eight congeneric series (CDK8, c-Met, Eg5, HIF-2a, PFKFB3, SHP-2, SYK, TNKS2; 264 ligands) plus two
variants (Eg5 with the alternative loop conformation, HIF-2a cycloalkyl[c]thiophenes, JMC 2023).
``ligands.sdf`` of each target holds the LigPrep-prepared ligand poses in the binding site, i.e. an
aligned congeneric set, with the measured IC50 in the tag ``IC50[uM]``; ``results_*.csv`` hold the
experimental binding free energies derived from them and the FEP+ predictions.

Written tags: ``pAffinity`` (-log10 of IC50 or Kd in M; the activity used by the program),
``affinity_type``, ``affinity_uM``,
``exp_dG_kcal`` and ``fep_pred_dG_20ns_kcal`` (from the CSV, for reference), ``target``, ``pdb``,
``series``, ``source``.  No train/test split is defined.

Usage:  python examples/make_schindler_examples.py --data-root <clone or download of the repository>
"""
import argparse
import csv
import math
import os

from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))

TARGETS = {  # folder: (label, pdb, note)
    "cdk8": ("CDK8 (cyclin-dependent kinase 8)", "5HNB", ""),
    "cmet": ("c-Met kinase", "4R1Y", ""),
    "eg5": ("Eg5 (kinesin spindle protein)", "3L9H", "loop adjacent to the phenol ligands remodelled"),
    "eg5_alternativeloop": ("Eg5, alternative loop conformation", "3L9H", "original X-ray loop; gives outliers for phenols"),
    "hif2a": ("HIF-2alpha", "5TBM", ""),
    "hif2a-jmc2023": ("HIF-2alpha, cycloalkyl[c]thiophenes (J. Med. Chem. 2023)", "8CK3", ""),
    "pfkfb3": ("PFKFB3 (6-phosphofructo-2-kinase/fructose-2,6-bisphosphatase 3)", "6HVI", ""),
    "shp2": ("SHP-2 phosphatase", "5EHR", ""),
    "syk": ("SYK (spleen tyrosine kinase)", "4PV0", ""),
    "tnks2": ("TNKS2 (tankyrase 2)", "4UI5", ""),
}
SOURCE = ("Schindler et al., J. Chem. Inf. Model. 60 (2020) 5457; Merck KGaA fep-benchmark "
          "(github.com/MCompChem/fep-benchmark, MIT licence); ligand poses prepared with LigPrep in the binding site")


def _affinity_tag(m):
    """(value in uM, p-type) from the heterogeneous affinity tags of the repository:
    IC50[uM], IC50[nM], IC50 uM, IC50[uM](SPA), ...ITC_Mean_KD_uM... ."""
    for tag in m.GetPropNames():
        t = tag.lower()
        if "ic50" in t or "kd" in t or "ki" in t:
            try:
                v = float(m.GetProp(tag).strip())
            except ValueError:
                continue
            if "nm" in t.replace("num", ""):
                v *= 1e-3
            elif "um" not in t:
                continue   # unknown unit
            ptype = "pIC50" if "ic50" in t else ("pKd" if "kd" in t else "pKi")
            if "itc" in t:
                ptype += " (ITC)"
            elif "spa" in t:
                ptype += " (SPA)"
            return v, ptype
    return None


def read_results(path):
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    keys = {k.strip(): k for k in rows[0].keys()}
    lig = keys.get("Ligand")
    exp = next((k for s, k in keys.items() if s.startswith("Exp")), None)
    pred = next((k for s, k in keys.items() if s.startswith("Pred.") and "Error" not in s), None)
    for r in rows:
        try:
            out[r[lig].strip()] = (float(r[exp]) if exp else float("nan"), float(r[pred]) if pred else float("nan"))
        except (ValueError, TypeError, KeyError):
            continue
    return out


def convert(data_root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for folder, (label, pdb, note) in TARGETS.items():
        sdf = os.path.join(data_root, folder, "ligands.sdf")
        if not os.path.isfile(sdf):
            print(f"{folder}: ligands.sdf not found, skipped")
            continue
        res20 = read_results(os.path.join(data_root, folder, "results_20ns.csv"))
        res5 = read_results(os.path.join(data_root, folder, "results_5ns.csv"))
        out = os.path.join(out_dir, f"{folder}.sdf")
        w = Chem.SDWriter(out)
        n = 0
        for m in Chem.SDMolSupplier(sdf, removeHs=False):
            if m is None:
                continue
            name = next((m.GetProp(k).strip() for k in ("Name", "s_user_Name", "ID") if m.HasProp(k)),
                        m.GetProp("_Name").strip())
            aff = _affinity_tag(m)
            if aff is None:
                continue
            value_uM, ptype = aff
            mm = Chem.Mol(m)
            for p in list(mm.GetPropNames()):
                mm.ClearProp(p)
            mm.SetProp("_Name", f"{folder}_{name}")
            mm.SetProp("pAffinity", f"{-math.log10(value_uM * 1e-6):.3f}")
            mm.SetProp("affinity_type", ptype)
            mm.SetProp("affinity_uM", f"{value_uM:g}")
            exp, pred = res20.get(name, res5.get(name, (float("nan"), float("nan"))))
            if exp == exp:
                mm.SetProp("exp_dG_kcal", f"{exp:.2f}")
            if pred == pred:
                mm.SetProp("fep_pred_dG_20ns_kcal" if name in res20 else "fep_pred_dG_5ns_kcal", f"{pred:.2f}")
            mm.SetProp("target", label)
            mm.SetProp("pdb", pdb)
            mm.SetProp("series", "Merck KGaA fep-benchmark (Schindler et al. 2020)")
            if note:
                mm.SetProp("note", note)
            mm.SetProp("source", SOURCE)
            w.write(mm)
            n += 1
        w.close()
        total += n
        print(f"{folder:20s} {n:3d} ligands -> {out}")
    print(f"total {total} ligands")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "schindler"))
    a = ap.parse_args()
    convert(a.data_root, a.out)
