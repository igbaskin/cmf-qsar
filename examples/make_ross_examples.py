"""
Build ``examples/ross/<group>/<series>.sdf`` from the Schrodinger public binding free energy benchmark

    Ross G.A., Lu C., Scarabelli G., et al. "The maximal and current accuracy of rigorous protein-ligand
    binding free energy calculations", Commun. Chem. 6 (2023) 222,
    https://github.com/schrodinger/public_binding_free_energy_benchmark (MIT licence).

The repository holds 92 congeneric series (1343 ligands) for 13 groups of test cases: the JACS R-group
set (Wang et al. 2015), the Merck set (Schindler et al. 2020), Janssen BACE series, GPCR ligands,
macrocycles (including the Bayer set), scaffold-hopping and core-hopping pairs, fragments, buried-water
cases, charge-change series and OPLS stress tests.  ``<series>_ligands.sdf`` of each case holds the
prepared ligand poses in the binding site, i.e. an aligned congeneric set, with the experimental binding
free energy in the tag ``r_exp_dg`` (kcal/mol).

This script converts dG to a p-scale activity, ``pAffinity = -dG / (2.303 R T)`` at T = 298.15 K, keeps
the coordinates untouched and writes one SDF per series with the tags ``pAffinity``, ``exp_dG_kcal``,
``protein``, ``pdb``, ``group``, ``series``, ``doi`` and ``source``.  Series with fewer than
``--min-ligands`` members (many test cases are 2-4 ligand pairs for scaffold hopping) are skipped.

Usage:  python examples/make_ross_examples.py --data-root <structure_inputs folder> [--min-ligands 5]
"""
import argparse
import csv
import os
import re

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))

RT_LN10 = 2.302585093 * 0.0019872041 * 298.15   # kcal/mol per log unit = 1.3642
SOURCE = ("Ross et al., Commun. Chem. 6 (2023) 222; Schrodinger public binding free energy benchmark "
          "(github.com/schrodinger/public_binding_free_energy_benchmark, MIT licence); "
          "prepared ligand poses in the binding site")
GROUP_NAMES = {
    "bayer_macrocycles": "Bayer macrocycles",
    "charge_annhil": "charge-change (charge annihilation) series",
    "fragments": "fragment series",
    "gpcrs": "GPCR ligands",
    "jacs_set": "FEP+ R-group set (Wang et al. 2015)",
    "janssen_bace": "Janssen BACE series",
    "macrocycles": "macrocycles",
    "mcs_docking": "MCS docking cases",
    "merck": "Merck KGaA set (Schindler et al. 2020)",
    "misc": "miscellaneous series",
    "opls_stress": "OPLS force-field stress tests",
    "scaffold_hopping": "scaffold- and core-hopping series",
    "waterset": "buried-water cases",
}


def read_metadata(path):
    """{file naming scheme: {protein, pdb, doi, group}}"""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r.get("Input file naming scheme") or "").strip()
            if key:
                out[key] = {
                    "protein": (r.get("Protein") or "").strip(),
                    "pdb": (r.get("Reference PDB") or "").strip(),
                    "doi": (r.get("DOI of source") or "").strip(),
                    "group": (r.get("Group name") or "").strip(),
                    "range": (r.get("Dynamic range (kcal/mol)") or "").strip(),
                }
    return out


def convert(data_root: str, out_dir: str, min_ligands: int = 5) -> None:
    meta = read_metadata(os.path.join(os.path.dirname(data_root.rstrip("/\\")), "benchmark_metadata.csv"))
    if not meta:
        meta = read_metadata(os.path.join(data_root, "benchmark_metadata.csv"))
    os.makedirs(out_dir, exist_ok=True)
    total, n_series, skipped = 0, 0, []
    for group in sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))):
        gdir = os.path.join(data_root, group)
        for fn in sorted(f for f in os.listdir(gdir) if f.endswith("_ligands.sdf")):
            series = fn[:-len("_ligands.sdf")]
            mols = [m for m in Chem.SDMolSupplier(os.path.join(gdir, fn), removeHs=False) if m is not None]
            mols = [m for m in mols if m.HasProp("r_exp_dg")]
            if len(mols) < min_ligands:
                skipped.append((group, series, len(mols)))
                continue
            info = meta.get(series, {})
            dg_all = [float(m.GetProp("r_exp_dg")) for m in mols]
            # a few OPLS stress-test series store RELATIVE free energies (dG > 0 for some ligands);
            # a constant offset does not affect a regression model (the intercept absorbs it), but the
            # absolute scale of pAffinity is then meaningless
            scale = "absolute" if max(dg_all) < 0 else "relative (arbitrary offset)"
            odir = os.path.join(out_dir, group)
            os.makedirs(odir, exist_ok=True)
            out = os.path.join(odir, f"{series}.sdf")
            w = Chem.SDWriter(out)
            for k, m in enumerate(mols):
                dg = float(m.GetProp("r_exp_dg"))
                mm = Chem.Mol(m)
                name = mm.GetProp("_Name").strip() or f"{series}_{k + 1}"
                for p in list(mm.GetPropNames()):
                    mm.ClearProp(p)
                mm.SetProp("_Name", name)
                mm.SetProp("pAffinity", f"{-dg / RT_LN10:.3f}")
                mm.SetProp("exp_dG_kcal", f"{dg:.2f}")
                mm.SetProp("series", series)
                mm.SetProp("group", GROUP_NAMES.get(group, group))
                if info.get("protein"):
                    mm.SetProp("protein", info["protein"])
                if info.get("pdb"):
                    mm.SetProp("pdb", info["pdb"])
                if info.get("doi"):
                    mm.SetProp("doi", info["doi"])
                mm.SetProp("affinity_scale", scale)
                mm.SetProp("source", SOURCE)
                w.write(mm)
            w.close()
            total += len(mols)
            n_series += 1
            print(f"{group:18s} {series:42s} {len(mols):3d} ligands  "
                  f"pAffinity {-max(dg_all) / RT_LN10:.1f}-{-min(dg_all) / RT_LN10:.1f}  "
                  f"{info.get('protein', '')}{'' if scale == 'absolute' else '  [relative dG]'}")
    print(f"\n{n_series} series, {total} ligands written to {out_dir}")
    if skipped:
        print(f"{len(skipped)} series with fewer than {min_ligands} ligands skipped: "
              + ", ".join(f"{s}({n})" for _, s, n in skipped))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="the structure_inputs folder of the repository")
    ap.add_argument("--out", default=os.path.join(HERE, "ross"))
    ap.add_argument("--min-ligands", type=int, default=5)
    a = ap.parse_args()
    convert(a.data_root, a.out, a.min_ligands)
