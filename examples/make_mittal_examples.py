"""
Build ``examples/mittal/<SET>.sdf`` from the 40 benchmark data sets of

    Mittal R.R., McKinnon R.A., Sorich M.J. "Comparison data sets for benchmarking QSAR
    methodologies in lead optimization", J. Chem. Inf. Model. 49 (2009) 1810-1820,
    Supporting Information ci900117m_si_001.zip (one ``<SET>.txt`` per set: SMILES + activity).

The supplementary data are two-dimensional.  For every set this script

  1. generates a 3D conformer for each molecule (RDKit ETKDGv3 + MMFF94 minimisation),
  2. chooses a reference molecule (the largest molecule among the most active quartile) and
     superimposes every other molecule on it by its maximum common substructure (MCS),
  3. writes an SDF with the tags ``activity`` (log units as given by Mittal et al.), ``set``,
     ``align_mcs_atoms`` and ``align_rmsd`` (size of the common substructure and RMSD of the fit,
     so that poorly aligned molecules can be recognised), ``reference`` and ``source``.

The conformations and alignments are therefore *automatic* and not the ones used in the original
3D-QSAR publications; the sets are meant for testing workflows and comparing methods under equal
conditions, not for reproducing literature statistics.  Two sets of the Supporting Information are corrupted (MT: every SMILES carries a "[Ce]" placeholder;
PLA2: dummy atoms "*") and are not produced.  Nine of the sets (ACE, ACHE, BZR, COX2, DHFR,
GPB, THERM, THR, STEROIDS) overlap with ``examples/sutherland`` and ``examples/steroids_cbg.sdf``,
where the original aligned coordinates are available.

Usage:  python examples/make_mittal_examples.py --si-dir <folder with the 40 .txt files> [--sets ACE,THR]
"""
import argparse
import os
import sys
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.align import mcs_align  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SOURCE = ("Mittal, McKinnon & Sorich, J. Chem. Inf. Model. 49 (2009) 1810, Supporting Information "
          "(SMILES + activity); 3D conformers: RDKit ETKDGv3 + MMFF94; alignment: MCS superposition "
          "on the reference molecule (automatic, not the original 3D-QSAR alignment)")


# Biological activity of each set (Table 1 of Mittal et al. 2009; N = number of molecules there)
DESCRIPTIONS = {
    "ACE": "inhibition of angiotensin-converting enzyme (N=114)",
    "ACHE": "inhibition of acetylcholinesterase (N=111)",
    "ACHE_TOX": "irreversible inhibition of acetylcholinesterase by organophosphates (N=30)",
    "AI": "inhibition of steroid aromatase (N=78)",
    "AMPH1": "binding affinity of peptides to the human amphiphysin-1 SH3 domain (N=130)",
    "APZ": "anti-cancer activity of anthrapyrazoles (N=17)",
    "ARB": "inhibition of the angiotensin II receptor (N=28)",
    "ATA": "anti-tuberculosis activity of quinolines (N=94)",
    "BZR": "binding affinity to the benzodiazepine receptor (N=163)",
    "CBRA": "binding affinity to the cannabinoid CB1 receptor (N=31)",
    "COMT": "inhibition of catechol-O-methyltransferase (N=92)",
    "COX2": "inhibition of cyclooxygenase-2 (N=322)",
    "D2A": "binding affinity to dopamine D2 receptors (N=26)",
    "DAT": "inhibition of the dopamine transporter (N=42)",
    "DHFR": "inhibition of dihydrofolate reductase (N=397)",
    "DIAZEPAM": "binding affinity to benzodiazepine receptor isoforms (N=42)",
    "DR": "inhibition of the dopamine D2 receptor, structurally diverse antagonists (N=38)",
    "ECR": "insecticidal activity of ecdysone agonists (N=50)",
    "EDC": "relative binding affinity to the estrogen receptor (N=123)",
    "EV71": "inhibition of enterovirus 71 (N=41)",
    "GHS": "median effective dose of growth hormone secretagogues (N=31)",
    "GPB": "inhibition of glycogen phosphorylase b (N=66)",
    "GSK3B": "inhibition of glycogen synthase kinase 3 (N=42)",
    "HIVPR": "inhibition of HIV protease (N=113)",
    "HIVRT": "inhibition of HIV-1 reverse transcriptase (N=101)",
    "HPTP": "inhibition of human protein tyrosine phosphatase 1B (N=135)",
    "KOA": "inhibition of the kappa opioid receptor (N=39)",
    "MT": "binding affinity to melatonin receptors (N=56)",
    "MX": "mutagenic potential of 3-chloro-4-(dichloromethyl)-5-hydroxy-2(5H)-furanone analogues, strain TA100 (N=29)",
    "PDE": "inhibition of phosphodiesterase IV (N=29)",
    "PLA2": "inhibition of phospholipase A2 (N=11)",
    "PTC": "enantiomeric selectivity of phase-transfer asymmetric catalysts (N=40)",
    "RYR": "binding affinity of ryanoids to the ryanodine receptor (N=18)",
    "STEROIDS": "binding affinity to corticosteroid-binding globulin (N=21)",
    "TCHK": "inhibition of Trypanosoma cruzi hexokinase (N=42)",
    "THERM": "inhibition of thermolysin (N=76)",
    "THR": "binding affinity to thrombin (N=88)",
    "TP2A": "inhibition of topoisomerase II (N=25)",
    "VATP": "inhibition of vacuolar-type (H+)-ATPases (N=23)",
    "YOPH": "inhibition of Yersinia protein tyrosine phosphatase YopH (N=39)",
}


def annotate(out_dir: str) -> None:
    """Add/refresh the ``description`` tag of every SDF in out_dir (no re-embedding)."""
    for fn in sorted(os.listdir(out_dir)):
        if not fn.lower().endswith(".sdf"):
            continue
        name = fn[:-4]
        path = os.path.join(out_dir, fn)
        mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
        w = Chem.SDWriter(path)
        for m in mols:
            m.SetProp("description", DESCRIPTIONS.get(name, ""))
            w.write(m)
        w.close()
        print(f"{name:9s} {len(mols):3d} molecules annotated: {DESCRIPTIONS.get(name, '?')}")


def read_set(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        for k, line in enumerate(fh):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((parts[0].strip(), float(parts[1])))
            except ValueError:
                continue
    return rows


def embed(smi, seed):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return None   # unparsable, or dummy-atom placeholders ("*") in the supplementary SMILES
    m = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.useRandomCoords = mol.GetNumHeavyAtoms() > 60   # random initial coordinates are faster for big molecules
    p.timeout = 60                                     # seconds per molecule
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=1000 if mol.GetNumHeavyAtoms() <= 60 else 200)
    except Exception:
        pass
    Chem.SanitizeMol(m)   # MMFF set-up alters aromaticity flags in place; restore RDKit's perception
    return m


def build_set(name, rows, out_path, mcs_timeout=2):
    t0 = time.time()
    mols, acts, smiles = [], [], []
    for k, (smi, y) in enumerate(rows):
        m = embed(smi, seed=1000 + k)
        if m is None:
            print(f"   {name}: could not embed {smi}")
            continue
        m.SetProp("_Name", f"{name}_{k + 1:03d}")
        mols.append(m)
        acts.append(y)
        smiles.append(smi)
    if len(mols) < 5:
        print(f"{name:9s} only {len(mols)} usable molecules - set skipped", flush=True)
        return
    acts = np.array(acts)
    # reference: the largest molecule among the most active quartile
    q = np.quantile(acts, 0.75)
    cand = [i for i, y in enumerate(acts) if y >= q] or list(range(len(mols)))
    ref = max(cand, key=lambda i: mols[i].GetNumHeavyAtoms())
    n_mcs = [0] * len(mols)
    rmsd = [0.0] * len(mols)
    for i, m in enumerate(mols):
        if i == ref:
            n_mcs[i] = m.GetNumHeavyAtoms()
            continue
        try:
            rmsd[i], n_mcs[i] = mcs_align(m, mols[ref], timeout=mcs_timeout)
        except Exception as exc:  # noqa: BLE001
            rmsd[i], n_mcs[i] = float("nan"), 0
            print(f"   {name}: alignment failed for {m.GetProp('_Name')}: {exc}")
    w = Chem.SDWriter(out_path)
    for m, y, smi, nm, r in zip(mols, acts, smiles, n_mcs, rmsd):
        m.SetProp("activity", f"{y:.3f}")
        m.SetProp("set", name)
        m.SetProp("SMILES", smi)
        m.SetProp("align_mcs_atoms", str(nm))
        m.SetProp("align_rmsd", "" if not np.isfinite(r) else f"{r:.2f}")
        m.SetProp("reference", mols[ref].GetProp("_Name"))
        m.SetProp("source", SOURCE)
        w.write(m)
    w.close()
    med = int(np.median(n_mcs))
    print(f"{name:9s} {len(mols):3d} molecules, reference {mols[ref].GetProp('_Name')} "
          f"({mols[ref].GetNumHeavyAtoms()} heavy atoms), median MCS {med} atoms, "
          f"{sum(1 for n in n_mcs if n < 6)} with MCS < 6, {time.time() - t0:.0f} s -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--si-dir", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "mittal"))
    ap.add_argument("--sets", default=None, help="comma-separated subset of set names")
    ap.add_argument("--mcs-timeout", type=int, default=2)
    ap.add_argument("--annotate-only", action="store_true", help="only (re)write the description tags")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.annotate_only:
        annotate(a.out)
        return
    if not a.si_dir:
        ap.error("--si-dir is required unless --annotate-only is given")
    names = sorted(f[:-4] for f in os.listdir(a.si_dir) if f.lower().endswith(".txt"))
    if a.sets:
        wanted = [s.strip().upper() for s in a.sets.split(",")]
        names = [n for n in names if n.upper() in wanted]
    for name in names:
        rows = read_set(os.path.join(a.si_dir, name + ".txt"))
        build_set(name, rows, os.path.join(a.out, f"{name}.sdf"), mcs_timeout=a.mcs_timeout)
    annotate(a.out)


if __name__ == "__main__":
    main()
