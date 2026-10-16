"""
Build ``examples/openff/<target>.sdf`` from the Open Force Field *protein-ligand-benchmark*
collection (https://github.com/openforcefield/protein-ligand-benchmark, MIT licence), which
bundles the congeneric free-energy benchmark series of

  * Wang et al., J. Am. Chem. Soc. 137 (2015) 2695  ("Schrodinger JACS" set: CDK2, MCL1, p38,
    PTP1B, thrombin, Tyk2 ...),
  * Schindler et al., J. Chem. Inf. Model. 60 (2020) 5457 ("Merck" set: CDK8, c-Met, Eg5,
    HIF-2a, PFKFB3, SHP-2, SYK, TNKS2),
  * further targets curated by the Open Force Field Initiative (e.g. PDE2).

Every ligand is posed in the binding site of the crystal structure of its target, so each series
is an aligned congeneric set; the repository gives the measured affinity (Ki, Kd or IC50) with its
source.  This script converts the affinity to a p-scale (-log10 of the molar value), keeps the
coordinates untouched and writes one SDF per target with the tags

    pAffinity        -log10(affinity / M)        <- activity used by the program
    affinity_type    pKi | pKd | pIC50
    affinity_value, affinity_unit, affinity_error, measurement_doi, measurement_comment
    target, pdb, series, source

No train/test split is defined (the original benchmarks have none); use "Random split" in the GUI
or ``--test-fraction`` in the CLI.

Usage:  python examples/make_fep_examples.py --data-root <folder with data/<target>/...>
        (the folder is a checkout/download of the repository's ``data`` tree)
"""
import argparse
import glob
import math
import os
import re
import sys

from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))

UNIT_TO_MOLAR = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9, "pM": 1e-12}
PTYPE = {"ki": "pKi", "kd": "pKd", "ic50": "pIC50", "ec50": "pEC50"}
SOURCE = ("Open Force Field protein-ligand-benchmark (github.com/openforcefield/protein-ligand-benchmark, "
          "MIT licence); ligand poses from the benchmark, affinities from the cited publications")


def _parse_ligands_yml(path: str) -> dict:
    """Parse the simple two-level ligands.yml (PyYAML if available, else a minimal parser)."""
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        pass
    out, cur, meas = {}, None, False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            key, _, val = line.strip().partition(":")
            val = val.strip().strip("'\"")
            if indent == 0:
                cur = key
                out[cur] = {"measurement": {}}
                meas = False
            elif indent == 2:
                meas = key == "measurement"
                if not meas:
                    out[cur][key] = val
            else:
                out[cur]["measurement"][key] = val
    return out


def _parse_target_yml(path: str) -> dict:
    info = {"pdb": "", "series": "", "comments": ""}
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    def list_after(i):
        items = []
        for nxt in lines[i + 1:]:
            if nxt.strip().startswith("- "):
                items.append(nxt.strip()[2:].strip().strip("'\""))
            elif nxt.strip() == "" or nxt.startswith(" "):
                continue
            else:
                break
        return items

    for i, line in enumerate(lines):
        if re.match(r"^pdb:", line):
            info["pdb"] = line.split(":", 1)[1].strip()
        elif line.startswith("associated_sets:"):
            info["series"] = "; ".join(list_after(i)).replace("KGgA", "KGaA")
        elif line.startswith("comments:"):
            rest = line.split(":", 1)[1].strip().strip("'\"")
            info["comments"] = rest or "; ".join(list_after(i))
    return info


def _combine_per_ligand_sdfs(tdir: str):
    """Releases <= 0.2.1 store one SDF per ligand (02_ligands/<lig>/crd/<lig>.sdf or <tdir>/<lig>.sdf).

    Returns the path of a combined temporary SDF, or a non-existing path when there are none.
    """
    parts = sorted(glob.glob(os.path.join(tdir, "lig_*.sdf")) +
                   glob.glob(os.path.join(tdir, "02_ligands", "*", "crd", "*.sdf")))
    if not parts:
        return os.path.join(tdir, "ligands.sdf")
    combined = os.path.join(tdir, "_ligands_combined.sdf")
    w = Chem.SDWriter(combined)
    for part in parts:
        for m in Chem.SDMolSupplier(part, removeHs=False):
            if m is None:
                continue
            if not m.GetProp("_Name").strip():
                m.SetProp("_Name", os.path.splitext(os.path.basename(part))[0])
            w.write(m)
    w.close()
    return combined


def convert(data_root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data = os.path.join(data_root, "data") if os.path.isdir(os.path.join(data_root, "data")) else data_root
    total = 0
    for folder in sorted(os.listdir(data)):
        tdir = os.path.join(data, folder)
        if not os.path.isdir(tdir):
            continue
        target = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", folder)   # "2020-02-04_bace" -> "bace"
        sdf = os.path.join(tdir, "ligands.sdf")
        if not os.path.isfile(sdf):
            sdf = os.path.join(tdir, "02_ligands", "ligands.sdf")
        if not os.path.isfile(sdf):
            sdf = _combine_per_ligand_sdfs(tdir)   # older releases: one SDF per ligand
        yml = os.path.join(tdir, "ligands.yml")
        if not os.path.isfile(yml):
            yml = os.path.join(tdir, "00_data", "ligands.yml")
        tyml = os.path.join(tdir, "target.yml")
        if not os.path.isfile(tyml):
            tyml = os.path.join(tdir, "00_data", "target.yml")
        if not (os.path.isfile(sdf) and os.path.isfile(yml)):
            continue
        ligs = _parse_ligands_yml(yml)
        tinfo = _parse_target_yml(tyml) if os.path.isfile(tyml) else {"pdb": "", "series": "", "comments": ""}
        out = os.path.join(out_dir, f"{target}.sdf")
        w = Chem.SDWriter(out)
        n, missing, types_used = 0, [], set()
        for m in Chem.SDMolSupplier(sdf, removeHs=False):
            if m is None:
                continue
            name = m.GetProp("_Name").strip()
            rec = ligs.get(name) or ligs.get("lig_" + name)   # some releases drop the "lig_" prefix in the SDF
            if rec is not None and name not in ligs:
                name = "lig_" + name
            if rec is None:
                missing.append(name)
                continue
            meas = rec["measurement"]
            value = float(meas["value"])
            unit = str(meas.get("unit", "") or "")
            mtype = str(meas["type"]).lower()
            if mtype in ("ki", "kd", "ic50", "ec50"):
                molar = value * UNIT_TO_MOLAR[unit]
                p = -math.log10(molar) if molar > 0 else float("nan")
                ptype = PTYPE[mtype]
            elif mtype in ("pki", "pkd", "pic50", "pec50"):          # already on the p-scale
                p = value
                ptype = "p" + mtype[1:].upper().replace("IC", "IC").replace("EC", "EC")
                ptype = {"pki": "pKi", "pkd": "pKd", "pic50": "pIC50", "pec50": "pEC50"}[mtype]
            elif mtype == "dg":                                      # binding free energy
                kj = value * (1.0 if unit.startswith("kilojoule") else 4.184)
                p = -kj / (8.314462618e-3 * 298.15 * math.log(10.0))
                ptype = "pKd (from dG)"
            else:                                                    # dh, tds ... are not affinities
                missing.append(f"{name} ({mtype})")
                continue
            types_used.add(ptype)
            mm = Chem.Mol(m)
            for prop in list(mm.GetPropNames()):      # drop Schrodinger bookkeeping tags (r_epik_*, s_lp_* ...)
                mm.ClearProp(prop)
            mm.SetProp("pAffinity", f"{p:.3f}")
            mm.SetProp("affinity_type", ptype)
            mm.SetProp("affinity_value", str(meas["value"]))
            mm.SetProp("affinity_unit", unit)
            if "error" in meas and meas["error"] not in (None, ""):
                mm.SetProp("affinity_error", str(meas["error"]))
            if "doi" in meas:
                mm.SetProp("measurement_doi", str(meas["doi"]))
            if "comment" in meas and meas["comment"]:
                mm.SetProp("measurement_comment", str(meas["comment"]))
            mm.SetProp("target", target)
            mm.SetProp("pdb", tinfo["pdb"])
            mm.SetProp("series", tinfo["series"] or "Open Force Field protein-ligand-benchmark")
            if tinfo.get("comments"):
                mm.SetProp("target_comment", tinfo["comments"])
            mm.SetProp("source", SOURCE)
            w.write(mm)
            n += 1
        w.close()
        total += n
        print(f"{target:9s} {n:3d} ligands  [{', '.join(sorted(types_used))}; {tinfo['series'] or 'OpenFF'}] -> {out}"
              + (f"   (skipped: {', '.join(missing)})" if missing else ""))
    print(f"total {total} ligands")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "openff"))
    a = ap.parse_args()
    convert(a.data_root, a.out)
