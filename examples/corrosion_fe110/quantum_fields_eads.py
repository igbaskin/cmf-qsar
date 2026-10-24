"""The E_ads model with quantum-chemical fields: do proper charges and Fukui functions beat the
classical five, and what do their co-fields look like?

Input: 157_benzimidazole_uma_qm.sdf, the adsorbed poses with per-atom ORCA properties
(examples/orca_atom_properties.py on the UMA SDF, so the properties are computed at the
adsorbed geometries, in the surface frame).  The molecules are registered in the surface plane
as in compare_alignments.py (rotation about z and an x, y shift on the MCS atoms), the artefact
molecule is dropped, and the same nested cross-validation (5 outer x 10 inner, 10 repeats) is
run for several field sets:

- classical: electrostatic (Gasteiger), steric, hydrophobic, hba, hbd (the reference, q2 0.52);
- hirshfeld: the same with the Hirshfeld charges in place of the Gasteiger ones;
- classical + f-: the five plus the condensed Fukui function f- (where the molecule gives up an
  electron) as a sixth field;
- classical + f- + homo: plus the HOMO population;
- reactivity only: f-, f+, HOMO population, local softness s-;
- hirshfeld + f- + homo.

Then the best set is fitted once and its co-fields are drawn on the 2.0 and 3.5 A planes.

    python examples/corrosion_fe110/quantum_fields_eads.py

Writes quantum_fields_eads.csv and cofields_eads_qm.png next to this script.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import compare_alignments as ca  # noqa: E402
import cofields_eads as cf  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, grid_search, r2_score  # noqa: E402

QM_SDF = os.path.join(HERE, "157_benzimidazole_uma_qm.sdf")
CLASSICAL = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
SETS = {
    "classical": (CLASSICAL, []),
    "hirshfeld charges": (["steric", "hydrophobic", "hba", "hbd"], ["q_hirshfeld"]),
    "classical + f-": (CLASSICAL, ["fukui_minus"]),
    "classical + f- + homo": (CLASSICAL, ["fukui_minus", "homo_pop"]),
    "reactivity only": ([], ["fukui_minus", "fukui_plus", "homo_pop", "softness_minus"]),
    "hirshfeld + f- + homo": (["steric", "hydrophobic", "hba", "hbd"], ["q_hirshfeld", "fukui_minus", "homo_pop"]),
}


def load():
    ds = load_dataset(QM_SDF, activity_prop="E_ads_kJmol")
    mols = [Chem.Mol(m) for m in ds.mols]
    for m in mols[1:]:
        ca.register_in_plane(m, mols[0])
    gas_ok = np.array([m.GetProp("gas_intact") == "1" if m.HasProp("gas_intact") else True for m in mols])
    have = np.array([m.HasProp("atom.dprop.fukui_minus") or m.HasProp("atom.prop.fukui_minus") for m in mols])
    keep = np.flatnonzero(gas_ok & np.isfinite(ds.activity) & have)
    print(f"{len(ds)} molecules in the QM SDF, {have.sum()} with ORCA properties, {len(keep)} used")
    return [mols[i] for i in keep], [ds.names[i] for i in keep], np.asarray(ds.activity[keep], float)


def main():
    mols, names, y = load()
    rows = []
    best = None
    for label, (physchem, props) in SETS.items():
        fs = FieldSet.create(physchem=physchem, atom_properties=props)
        try:
            res, dt = ca.evaluate(mols, names, y, fs, 5, 10, 0)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:24s} FAILED: {exc}")
            continue
        rows.append({"fields": label, "n": len(y), "q2": round(res.q2_ex, 3), "q2_sd": round(res.q2_ex_sd, 3),
                     "rmse": round(res.rmse_ex, 2), "seconds": round(dt)})
        print(f"{label:24s} n={len(y)} q2 = {res.q2_ex:6.3f} +- {res.q2_ex_sd:.3f}  rmse {res.rmse_ex:.1f} kJ/mol  {dt:.0f} s", flush=True)
        if best is None or res.q2_ex > best[0]:
            best = (res.q2_ex, label, fs)
    out = os.path.join(HERE, "quantum_fields_eads.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("->", out)

    # co-fields of the best quantum set (and of hirshfeld + f- + homo if it is not the best)
    for label in dict.fromkeys([best[1], "hirshfeld + f- + homo"]):
        if label == "classical":
            continue
        physchem, props = SETS[label]
        fs = FieldSet.create(physchem=physchem, atom_properties=props)
        mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
        train = AtomTable.from_molecules(mf)
        model = CMFModel(fieldset=fs, method="krr", fit_bias=True)
        res = grid_search(train, y, model, ca.ALPHAS, lams=ca.LAMS, folds=10, seed=0)
        model.fit(train, y, cv_folds=10, cv_seed=0, reuse_kernels=True)
        print(f"co-fields of '{label}': alpha {res.best_alpha:g}, lambda {res.best_params.get('lam', float('nan')):g}, "
              f"q2 (10-fold) {res.best_q2:.3f}, fitted r2 {r2_score(y, model.predict(train)):.3f}")
        draw(model, fs, mols[0], names[0], y, res.best_q2, label,
             os.path.join(HERE, "cofields_eads_qm" + ("" if label == best[1] else "_hirshfeld") + ".png"))


def draw(model, fs, ref, ref_name, y, q2, label, out, planes=(2.0, 3.5), half=8.0, spacing=0.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(-half, half + 1e-9, spacing)
    fields = [s.name for s in fs.specs]
    conf = ref.GetConformer()
    heavy = {i: (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(ref.GetNumAtoms())
             if ref.GetAtomWithIdx(i).GetAtomicNum() > 1}
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in ref.GetBonds()
             if b.GetBeginAtomIdx() in heavy and b.GetEndAtomIdx() in heavy]
    fig, axes = plt.subplots(len(planes), len(fields), figsize=(3.1 * len(fields), 3.3 * len(planes)), squeeze=False)
    for r, z in enumerate(planes):
        slices = [cf.cofield_slice(model, f, z, xs, xs) for f in range(len(fields))]
        vmax = max(np.abs(s).max() for s in slices) or 1.0
        for f, name in enumerate(fields):
            ax = axes[r][f]
            im = ax.imshow(slices[f], origin="lower", extent=(xs[0], xs[-1], xs[0], xs[-1]),
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            for i, j in bonds:
                ax.plot([heavy[i][0], heavy[j][0]], [heavy[i][1], heavy[j][1]], color="black", lw=1.0)
            for i, (x0, y0) in heavy.items():
                sym = ref.GetAtomWithIdx(i).GetSymbol()
                ax.scatter([x0], [y0], s=14 if sym == "C" else 30, color="black" if sym == "C" else "darkgreen", zorder=3)
            ax.set_title(f"{name}, z = {z:.1f} A", fontsize=10)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("x, A", fontsize=8)
            if f == 0:
                ax.set_ylabel("y, A", fontsize=8)
        fig.colorbar(im, ax=axes[r].tolist(), shrink=0.8, label="h c C(r), kJ/mol per field unit")
    fig.suptitle(f"Co-fields of the E_ads model, fields: {label} (blue: stronger adsorption); q2 = {q2:.2f}, "
                 f"{len(y)} molecules; {ref_name} drawn (N in green)", fontsize=10)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    main()
