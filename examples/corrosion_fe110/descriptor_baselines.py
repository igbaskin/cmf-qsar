"""QSPR baselines on global molecular descriptors, for the field models to be measured against.

The corrosion-inhibitor literature models inhibition with the frontier-orbital indices of the
inhibitor: E_HOMO, E_LUMO, the gap, the dipole moment, hardness, softness, electrophilicity and
the fraction of electrons transferred to iron, dN = (chi_Fe - chi) / 2 (eta_Fe + eta) with
chi_Fe = 7 eV and eta_Fe = 0.  Those come from the ORCA run on the adsorbed poses
(157_benzimidazole_uma_qm.sdf); RDKit supplies the usual size, lipophilicity and topology
descriptors.  Three descriptor sets (quantum, RDKit, both), three learners (ridge, kernel ridge
with an RBF kernel, random forest), the ridge and kernel-ridge hyper-parameters chosen in an inner cross-validation (the forest fixed),
the same outer folds and repeats as the field models (5 x 10 for the node, leave-one-out for
Hu 2011).

Targets: the UMA adsorption energy (72 reliable molecules), the pooled potency (73) and the
potency of Hu 2011's series (18).

    python examples/corrosion_fe110/descriptor_baselines.py [--indices OTHER.sdf --tag gas]

With ``--indices`` the global quantum indices are read from another SDF of the same compounds
(matched by db_id), e.g. 157_benzimidazole_gas_qm.sdf with the indices computed at gas-phase
geometries, which removes the adsorbed geometry from the descriptors; the targets still come
from the UMA SDF.  Writes descriptor_baselines[_TAG].csv next to this script and prints the table.
"""
from __future__ import annotations

import csv
import os
import sys
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import potency_series as ps  # noqa: E402
from cmfqsar.model import _kfold_indices, r2_score, rmse  # noqa: E402

QM_SDF = os.path.join(HERE, "157_benzimidazole_uma_qm.sdf")
QM = ["E_homo", "E_lumo", "gap", "dipole", "IP", "EA", "hardness", "softness", "electrophilicity", "chemical_potential"]


def quantum(m, koopmans=False):
    """The eleven indices; with `koopmans` (or when the N+1/N-1 states were not computed) the
    reactivity indices come from the orbital energies, IP = -E_HOMO and EA = -E_LUMO, as the
    corrosion literature computes them."""
    if koopmans or not m.HasProp("IP"):
        homo, lumo = float(m.GetProp("E_homo")), float(m.GetProp("E_lumo"))
        ip, ea = -homo, -lumo
        eta = (ip - ea) / 2.0
        mu = -(ip + ea) / 2.0
        v = [homo, lumo, lumo - homo, float(m.GetProp("dipole")), ip, ea, eta,
             1.0 / (2.0 * eta), mu * mu / (2.0 * eta), mu]
    else:
        v = [float(m.GetProp(p)) for p in QM]
        mu, eta = float(m.GetProp("chemical_potential")), float(m.GetProp("hardness"))
    v.append((7.0 + mu) / (2.0 * eta))                      # dN towards iron, chi = -mu
    return v


QM_NAMES = QM + ["dN"]
RD_NAMES = ["MolWt", "MolLogP", "MolMR", "TPSA", "HBD", "HBA", "RotB", "HeavyAtoms", "AromRings", "Rings",
            "FracSP3", "nN", "nO", "nS", "nHal", "BalabanJ", "Kappa1", "Kappa2", "Chi0v", "Chi1v", "LabuteASA"]


def rdkit_desc(m):
    h = Chem.RemoveHs(m)
    return [Descriptors.MolWt(h), Crippen.MolLogP(h), Crippen.MolMR(h), rdMolDescriptors.CalcTPSA(h),
            Lipinski.NumHDonors(h), Lipinski.NumHAcceptors(h), rdMolDescriptors.CalcNumRotatableBonds(h),
            h.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumAromaticRings(h), rdMolDescriptors.CalcNumRings(h),
            rdMolDescriptors.CalcFractionCSP3(h),
            sum(a.GetAtomicNum() == 7 for a in h.GetAtoms()), sum(a.GetAtomicNum() == 8 for a in h.GetAtoms()),
            sum(a.GetAtomicNum() == 16 for a in h.GetAtoms()), sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in h.GetAtoms()),
            Descriptors.BalabanJ(h), Descriptors.Kappa1(h), Descriptors.Kappa2(h), Descriptors.Chi0v(h),
            Descriptors.Chi1v(h), Descriptors.LabuteASA(h)]


def learners():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def ridge(inner):
        return GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                            {"ridge__alpha": [0.01, 0.1, 1, 10, 100, 1000]}, cv=inner, scoring="neg_root_mean_squared_error")

    def krr(inner):
        return GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                            {"kernelridge__alpha": [0.01, 0.1, 1, 10], "kernelridge__gamma": [0.003, 0.01, 0.03, 0.1, 0.3]},
                            cv=inner, scoring="neg_root_mean_squared_error")

    def forest(inner):
        # a random forest is insensitive to its settings at this size; fixed, so that the 50 outer
        # fits per table row do not each carry an inner grid of forests
        return RandomForestRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=2, random_state=0)

    return {"ridge": ridge, "kernel ridge (RBF)": krr, "random forest": forest}


def nested(X, y, make, outer, repeats, seed):
    from sklearn.model_selection import KFold
    n = len(y)
    preds = np.zeros((repeats, n))
    for rep in range(repeats):
        folds = _kfold_indices(n, outer, None if seed is None else seed + rep)
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            inner = KFold(n_splits=min(10, max(2, len(tr) // 3)), shuffle=True, random_state=seed)
            est = make(inner)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X[tr], y[tr])
            preds[rep, te] = est.predict(X[te])
    q2 = [r2_score(y, p) for p in preds]
    return float(np.mean(q2)), float(np.std(q2)) if repeats > 1 else float("nan"), float(np.mean([rmse(y, p) for p in preds]))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("--indices", default=None, help="SDF whose global quantum indices replace the UMA SDF's")
    ap.add_argument("--tag", default="", help="suffix of the output table")
    ap.add_argument("--koopmans", action="store_true", help="reactivity indices from the orbital energies even when the Delta-SCF ones exist")
    a = ap.parse_args(argv)
    mols = [m for m in Chem.SDMolSupplier(QM_SDF, removeHs=False)]
    ids = np.array([int(m.GetProp("db_id")) for m in mols])
    if a.indices:
        other = {int(m.GetProp("db_id")): m for m in Chem.SDMolSupplier(a.indices, removeHs=False) if m.HasProp("gap")}
        missing = [i for i in ids if i not in other]
        print(f"indices from {a.indices}: {len(other)} molecules, {len(missing)} of the UMA set missing")
        Xq = np.array([quantum(other[i], a.koopmans) if i in other else [np.nan] * len(QM_NAMES) for i in ids])
        if not all(m.HasProp("IP") for m in other.values()):
            print("  no Delta-SCF states in that file: IP, EA, hardness, softness, electrophilicity and the chemical potential from the orbital energies (Koopmans)")
    else:
        Xq = np.array([quantum(m, a.koopmans) for m in mols])
    Xr = np.array([rdkit_desc(m) for m in mols])
    gas_ok = np.array([m.GetProp("gas_intact") == "1" for m in mols])
    e_ads = np.array([float(m.GetProp("E_ads_kJmol")) for m in mols])
    potency = np.array([float(m.GetProp("activity")) for m in mols])
    h = ps.rows_1m_hcl()
    hu = h[h.Refcode == "hu2011-2402"].groupby("db_id").MWILA.median()
    y_hu = np.array([hu.get(i, np.nan) for i in ids])
    targets = {"E_ads (kJ/mol)": (e_ads, gas_ok, 5, 10), "potency, pooled node": (potency, np.ones(len(mols), bool), 5, 10),
               "potency, Hu 2011": (y_hu, np.isfinite(y_hu), None, 1)}
    sets = {"quantum indices (11)": Xq, "RDKit descriptors (21)": Xr, "both (32)": np.hstack([Xq, Xr])}
    rows = []
    for tname, (y, keep, outer, repeats) in targets.items():
        idx = np.flatnonzero(keep & np.isfinite(y) & np.isfinite(Xq).all(axis=1))
        for sname, X in sets.items():
            for lname, make in learners().items():
                n_out = len(idx) if outer is None else outer
                q2, sd, rm = nested(X[idx], y[idx], make, n_out, repeats, 0)
                rows.append({"target": tname, "descriptors": sname, "learner": lname, "n": len(idx),
                             "q2": round(q2, 3), "q2_sd": round(sd, 3), "rmse": round(rm, 2), "sd_y": round(float(np.std(y[idx])), 2)})
                print(f"{tname:22s} {sname:24s} {lname:20s} n={len(idx):3d} q2 = {q2:6.3f} +- {sd:.3f}  rmse {rm:.2f} (sd y {np.std(y[idx]):.2f})", flush=True)
        # the single best-correlated index, the literature's favourite plot
        r = [(abs(np.corrcoef(Xq[idx, k], y[idx])[0, 1]), QM_NAMES[k]) for k in range(Xq.shape[1])]
        print(f"   strongest single quantum index for {tname}: {max(r)[1]} |r| = {max(r)[0]:.2f}")
    out = os.path.join(HERE, "descriptor_baselines" + (f"_{a.tag}" if a.tag else "") + ".csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
