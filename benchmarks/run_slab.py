"""
A metal slab as the receptor: can complementarity place a molecule, and does it predict E_ads?

Plan item EC6.  The Fe(110) sets of ``examples/corrosion_fe110`` hold, for two nodes of the
corrosion-inhibitor scaffold tree, the adsorption energy of every molecule on bcc Fe(110) from
the UMA potential and the relaxed adsorbed pose in the surface frame (the normal along z, the top
Fe layer at z = 0).  Those are the reference poses no protein set has, so they are the gate of
this item, which has two halves that can fail separately:

1. **the pose.**  Does docking constrained to the surface -- height, tilt, rotation about the
   normal, translation in the plane -- recover the relaxed pose, starting from the rule pose that
   item EC2 built and from a conformer of the SMILES, which is what a user without a slab has?
2. **the energy.**  Does a complementarity score with the surface in place of the pocket track
   ``E_ads_kJmol`` across the set?

The second half has a hard control, and it was item EC2 that found it: the relaxed frame carries
the outcome of the relaxation, so a straight line on the lowest heavy-atom height of the relaxed
pose already predicts E_ads with a leave-one-out q2 of about 0.47 and 0.42 on the two nodes.  Any
score claimed here has to beat that line, and it is reported beside every model.

What the receptor is, and what it is not
----------------------------------------
``cmfqsar/slab.py`` turns the top layers of a slab, with their periodic images out to a cutoff,
into the same ``MoleculeFields`` that a protein pocket becomes.  On a one-element metal slab four
of the five physicochemical fields are identically zero, and not for want of parameters: a
one-element cell has no electronegativity differences, so every charge model gives zero, and the
hydrophobic and hydrogen-bond increments are fitted to organic fragments.  All four complementary
pairs of :mod:`cmfqsar.complementarity` are therefore zero against a clean metal, and what is
left is burial -- the steric overlap.  The ``receptor`` stage measures that rather than asserting
it.

What a metal has instead is its image charge, and ``slab.image_charge_energy`` computes it in
closed form.  Its one awkward property is measured here too: the image energy of a charge
distribution above an infinite plane is invariant to translation in the plane and to rotation
about the normal, so it can set a height and a tilt and can never say where on the lattice a
molecule sits.

    python benchmarks/run_slab.py --stage receptor
    python benchmarks/run_slab.py --stage pose
    python benchmarks/run_slab.py --stage score
    python benchmarks/run_slab.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, RDLogger                                            # noqa: E402

from cmfqsar.align import embed_3d, set_coords                              # noqa: E402
from cmfqsar.fields import FieldSet, build_fields                           # noqa: E402
from cmfqsar.model import r2_score                                          # noqa: E402
from cmfqsar.planealign import height_rmsd, in_plane_rmsd, rule_pose        # noqa: E402
from cmfqsar.slab import (SLAB_NONZERO_FIELDS, bcc110_slab,                 # noqa: E402
                          dock_on_surface, image_charge_energy, layer_heights,
                          slab_receptor, surface_score, surface_sites, top_layer_indices)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "slab")
NODES = {
    "benzimidazole": os.path.join(ROOT, "examples", "corrosion_fe110",
                                  "157_benzimidazole_uma.sdf"),
    "quinoline": os.path.join(ROOT, "examples", "corrosion_fe110", "158_quinoline_uma.sdf"),
}
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
ALPHA = 1.0
SEED = 0
#: the slab the reference poses were computed on (examples/uma_fe110_adsorption.py)
SLAB = dict(element="Fe", nx=6, ny=6, layers=4, a=2.78, vacuum=15.0)
#: how far beyond the primary surface cell the images are carried
IMAGE_CUTOFF = 12.0
#: the starting height of a rule pose, as item EC2 used
RULE_HEIGHT = 2.2


def fieldset(alpha: float = ALPHA) -> FieldSet:
    return FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger",
                           param_scheme="rdkit")


def receptor():
    slab = bcc110_slab(**SLAB)
    return slab, slab_receptor(slab, fieldset(), top_layers=2, cutoff=IMAGE_CUTOFF)


def load(node: str):
    mols, names, e_ads, bound, smiles = [], [], [], [], []
    for mol in Chem.SDMolSupplier(NODES[node], removeHs=False, sanitize=True):
        if mol is None or not mol.HasProp("E_ads_kJmol"):
            continue
        if mol.HasProp("gas_intact") and mol.GetProp("gas_intact") != "1":
            continue
        value = float(mol.GetProp("E_ads_kJmol"))
        if not np.isfinite(value):
            continue
        mols.append(Chem.Mol(mol))
        names.append(mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{len(mols)}")
        e_ads.append(value)
        bound.append(mol.HasProp("bound") and mol.GetProp("bound") == "1")
        smiles.append(mol.GetProp("smiles") if mol.HasProp("smiles") else Chem.MolToSmiles(mol))
    return mols, names, np.array(e_ads), np.array(bound), smiles


def heavy(mol) -> np.ndarray:
    return np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])


def xyz(mol) -> np.ndarray:
    return np.array(mol.GetConformer().GetPositions(), dtype=float)


# ------------------------------------------------------------------------------- the receptor

def receptor_stage() -> Dict[str, object]:
    slab, rec = receptor()
    fs = fieldset()
    out: Dict[str, object] = {
        "slab": dict(SLAB), "slab_atoms": int(len(slab)),
        "layer_heights": [round(h, 3) for h in layer_heights(slab)],
        "top_layer_atoms": int(len(top_layer_indices(slab, 1))),
        "receptor_atoms": int(rec.n_atoms), "image_cutoff": IMAGE_CUTOFF,
        "cell": [[float(v) for v in row[:2]] for row in np.asarray(slab.get_cell())[:2]],
    }
    nonzero = [s.name for k, s in enumerate(fs.specs) if np.abs(rec.weights[:, k]).sum() > 1e-12]
    out["nonzero_fields"] = nonzero
    out["zero_fields"] = [s.name for s in fs.specs if s.name not in nonzero]
    print(f"    the slab is {len(slab)} atoms in {len(out['layer_heights'])} layers; the "
          f"receptor is its top two layers with images, {rec.n_atoms} atoms", flush=True)
    print(f"    fields the slab carries: {', '.join(nonzero)}; identically zero: "
          f"{', '.join(out['zero_fields'])}", flush=True)
    assert nonzero == list(SLAB_NONZERO_FIELDS), nonzero

    # the image energy's invariances, measured rather than asserted
    mols, names, _e, _b, _s = load("benzimidazole")
    mf = build_fields(mols[0], fs, names[0])
    base = image_charge_energy(mf.coords, mf.charges)
    shifted = image_charge_energy(mf.coords + np.array([4.3, -2.7, 0.0]), mf.charges)
    psi = 1.1
    rot = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0],
                    [0.0, 0.0, 1.0]])
    turned = image_charge_energy(mf.coords @ rot.T, mf.charges)
    lifted = image_charge_energy(mf.coords + np.array([0.0, 0.0, 3.0]), mf.charges)
    out["image"] = {"at_its_pose_eV": base, "after_an_in_plane_shift_eV": shifted,
                    "after_a_turn_about_the_normal_eV": turned, "three_angstrom_higher_eV": lifted,
                    "in_plane_change": abs(shifted - base), "turn_change": abs(turned - base)}
    print(f"    the image energy of {names[0]} is {base:+.4f} eV, unchanged by an in-plane shift "
          f"({abs(shifted - base):.1e}) and by a turn about the normal "
          f"({abs(turned - base):.1e}), and {lifted:+.4f} eV three angstrom higher", flush=True)

    sites = surface_sites(slab)
    out["sites"] = {k: int(sites[0].count(k)) for k in sorted(set(sites[0]))}
    print(f"    surface sites of the top layer: "
          f"{', '.join(f'{v} {k}' for k, v in out['sites'].items())}", flush=True)
    rows = []
    for node in NODES:
        mols, names, _e, bound, _s = load(node)
        for i in (0, 1):
            mf = build_fields(mols[i], fs, names[i])
            sc = surface_score(mf, rec, fs, sites=sites)
            rows.append({"node": node, "name": names[i], "burial": sc.burial,
                         "image_eV": sc.image_eV, "height": sc.height, "sites": sc.sites})
            print(f"    {node}/{names[i][:22]:22s} {sc.describe()}", flush=True)
    out["examples"] = rows
    return out


# ------------------------------------------------------------------------------- the pose

def conformer_of(smiles: str, relaxed, seed: int = 0):
    """An ETKDG+MMFF conformer whose heavy atoms are in the order of `relaxed`."""
    conf = embed_3d(Chem.MolFromSmiles(smiles), seed=seed)
    ref_heavy = Chem.RemoveHs(relaxed)
    con_heavy = Chem.RemoveHs(conf)
    match = ref_heavy.GetSubstructMatch(con_heavy)
    if len(match) != con_heavy.GetNumAtoms():
        return None, None
    return conf, np.argsort(list(match))


def _posed(mol, coords):
    out = Chem.Mol(mol)
    set_coords(out, coords)
    return out


def pose_errors(posed_coords, posed_heavy, relaxed, perm=None):
    a = np.asarray(posed_coords, float)[posed_heavy]
    b = xyz(relaxed)[heavy(relaxed)]
    if perm is not None:
        a = a[perm]
    return in_plane_rmsd(a, b), height_rmsd(a, b)


def _summary(rows: Sequence[Tuple[float, float, bool]]) -> Dict[str, float]:
    arr = np.array([[r[0], r[1]] for r in rows], float)
    isb = np.array([r[2] for r in rows], bool)
    out = {"n": len(rows), "n_bound": int(isb.sum()),
           "in_plane_median": float(np.median(arr[:, 0])),
           "height_median": float(np.median(arr[:, 1])),
           "within_1A": float(np.mean(arr[:, 0] < 1.0))}
    if isb.any():
        out.update(in_plane_median_bound=float(np.median(arr[isb, 0])),
                   height_median_bound=float(np.median(arr[isb, 1])),
                   within_1A_bound=float(np.mean(arr[isb, 0] < 1.0)))
    return out


def pose_stage(limit: int = 0) -> Dict[str, object]:
    slab, rec = receptor()
    fs = fieldset()
    out: Dict[str, object] = {"variants": [], "nodes": {}}
    variants = [("rule", dict(dock=False)),
                ("docked", dict(dock=True, both_faces=False, image_weight=0.0)),
                ("docked-both-faces", dict(dock=True, both_faces=True, image_weight=0.0)),
                ("docked-with-image", dict(dock=True, both_faces=False, image_weight=0.1))]
    out["variants"] = [v for v, _ in variants] + ["conformer-rule", "conformer-docked"]
    for node in NODES:
        mols, names, _e, bound, smiles = load(node)
        if limit:
            mols, names, bound, smiles = (mols[:limit], names[:limit], bound[:limit],
                                          smiles[:limit])
        rec_node: Dict[str, object] = {"n": len(mols)}
        print(f"  === {node}: {len(mols)} molecules", flush=True)
        for label, how in variants:
            rows, t0 = [], time.time()
            for mol, name, isb in zip(mols, names, bound):
                coords, _anchor = rule_pose(mol, "flat", height=RULE_HEIGHT)
                if how["dock"]:
                    mf = build_fields(_posed(mol, coords), fs, name)
                    coords, _res = dock_on_surface(
                        mf, rec, fs, starts=8, seed=SEED,
                        both_faces=how["both_faces"], image_weight=how["image_weight"])
                ip, hz = pose_errors(coords, heavy(mol), mol)
                rows.append((ip, hz, bool(isb)))
            rec_node[label] = _summary(rows)
            rec_node[label]["seconds"] = time.time() - t0
            s = rec_node[label]
            print(f"    {label:<20} in-plane {s['in_plane_median']:.2f} A (bound "
                  f"{s.get('in_plane_median_bound', float('nan')):.2f}), height "
                  f"{s['height_median']:.2f} A, within 1 A {100 * s['within_1A']:.0f} % "
                  f"({s['seconds']:.0f} s)", flush=True)
        # from a conformer of the SMILES: what a user without a slab has
        for label, dock in (("conformer-rule", False), ("conformer-docked", True)):
            rows, t0, skipped = [], time.time(), 0
            for mol, name, isb, smi in zip(mols, names, bound, smiles):
                conf, perm = conformer_of(smi, mol, seed=SEED)
                if conf is None:
                    skipped += 1
                    continue
                coords, _anchor = rule_pose(conf, "flat", height=RULE_HEIGHT)
                if dock:
                    mf = build_fields(_posed(conf, coords), fs, name)
                    coords, _res = dock_on_surface(mf, rec, fs, starts=8, seed=SEED)
                ip, hz = pose_errors(coords, heavy(conf), mol, perm=perm)
                rows.append((ip, hz, bool(isb)))
            rec_node[label] = _summary(rows) if rows else {"n": 0}
            rec_node[label]["skipped"] = skipped
            rec_node[label]["seconds"] = time.time() - t0
            if rows:
                s = rec_node[label]
                print(f"    {label:<20} in-plane {s['in_plane_median']:.2f} A (bound "
                      f"{s.get('in_plane_median_bound', float('nan')):.2f}), height "
                      f"{s['height_median']:.2f} A, within 1 A {100 * s['within_1A']:.0f} % "
                      f"({s['seconds']:.0f} s, {skipped} skipped)", flush=True)
        out["nodes"][node] = rec_node
    return out


# ------------------------------------------------------------------------------- the energy

def _loo_q2(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out q2 of a straight line, the same measure item EC2's control uses."""
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float).reshape(len(y), -1)])
    pred = np.empty(len(y))
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        beta = np.linalg.lstsq(X[k], y[k], rcond=None)[0]
        pred[i] = X[i] @ beta
    return float(r2_score(y, pred))


def score_stage() -> Dict[str, object]:
    slab, rec = receptor()
    fs = fieldset()
    sites = surface_sites(slab)
    out: Dict[str, object] = {"nodes": {}}
    for node in NODES:
        mols, names, y, bound, _s = load(node)
        print(f"  === {node}: {len(mols)} molecules, E_ads {y.min():.0f} to {y.max():.0f} kJ/mol",
              flush=True)
        terms: Dict[str, List[float]] = {k: [] for k in
                                         ("burial", "image", "height", "burial_docked",
                                          "image_docked", "height_docked")}
        for mol, name in zip(mols, names):
            mf = build_fields(mol, fs, name)
            sc = surface_score(mf, rec, fs, sites=sites)
            terms["burial"].append(sc.burial)
            terms["image"].append(sc.image_eV)
            terms["height"].append(sc.height)
            coords, _anchor = rule_pose(mol, "flat", height=RULE_HEIGHT)
            posed = build_fields(_posed(mol, coords), fs, name)
            docked, res = dock_on_surface(posed, rec, fs, starts=8, seed=SEED)
            terms["burial_docked"].append(res.burial_after)
            terms["image_docked"].append(res.image_after)
            terms["height_docked"].append(res.height_after)
        rec_node: Dict[str, object] = {"n": len(mols), "n_bound": int(bound.sum())}
        for key, values in terms.items():
            v = np.array(values, float)
            rec_node[key] = {"pearson": float(np.corrcoef(v, y)[0, 1]) if v.std() > 1e-12 else 0.0,
                             "loo_q2": _loo_q2(v, y), "mean": float(v.mean()),
                             "sd": float(v.std())}
        # the two scores a user could form, and the controls item EC2 established
        rec_node["burial+image"] = {"loo_q2": _loo_q2(
            np.column_stack([terms["burial"], terms["image"]]), y)}
        rec_node["burial+image_docked"] = {"loo_q2": _loo_q2(
            np.column_stack([terms["burial_docked"], terms["image_docked"]]), y)}
        rec_node["control_relaxed_height"] = {"loo_q2": _loo_q2(terms["height"], y)}
        rec_node["control_bound_flag"] = {"loo_q2": _loo_q2(bound.astype(float), y)}
        for key in ("burial", "image", "height", "burial+image", "burial_docked",
                    "image_docked", "height_docked", "burial+image_docked",
                    "control_relaxed_height", "control_bound_flag"):
            q2 = rec_node[key]["loo_q2"]
            r = rec_node[key].get("pearson")
            print(f"    {key:<26} leave-one-out q2 {q2:+.3f}"
                  + (f"   r {r:+.3f}" if r is not None else ""), flush=True)
        out["nodes"][node] = rec_node
    return out


# ------------------------------------------------------------------------------- the measured sets

#: one CSV per scaffold-tree node of the iron corrosion-inhibitor database: SMILES, the MWILA
#: potency as ``activity``, and the Langmuir adsorption free energy as ``dG_ads``.  Unlike the
#: Fe(110) sets both targets are **measured**, which is what the Fe(110) half of this item could
#: not offer: there the target came from the same relaxation that produced the pose.
CORROSION = os.path.join(ROOT, "examples", "corrosion")
CORROSION_TARGETS = ("activity", "dG_ads")
#: a series smaller than this cannot carry a ten-fold cross-validation
CORROSION_MIN_ROWS = 15
#: the grids and the fold count of ``benchmarks/run_corrosion.py``, so that the numbers here can
#: be read against its published table (atom-type counts +0.189, the aligned five-field model
#: +0.269, the invariant kernel +0.278, means over the 23 series).  That convention chooses
#: (alpha, lambda) on the cross-validation it reports, which makes every model in the comparison
#: optimistic in the same way.
CORR_ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
CORR_LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
CORR_FOLDS = 10
#: the terms the slab receptor contributes, in the order the descriptor block has them
SURFACE_TERMS = ("burial", "image_eV", "height", "over_top", "over_bridge", "over_hollow")


def _standardise(train: np.ndarray, other: Optional[np.ndarray] = None) -> np.ndarray:
    mean, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    rows = train if other is None else other
    return (rows - mean) / sd


def _descriptor_kernel(Z: np.ndarray) -> np.ndarray:
    """The linear kernel of a descriptor block, scaled to a mean diagonal of one.

    Scaled rather than cosine-normalised, for the reason :mod:`cmfqsar.descriptors` gives: a
    molecule sitting at the mean in every column would otherwise become a zero row.
    """
    A = _standardise(np.asarray(Z, float))
    K = A @ A.T
    scale = float(np.mean(np.diag(K)))
    return K / (scale if scale > 1e-12 else 1.0)


def _unit_diagonal(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(np.asarray(K, float)), 1e-12, None))
    return np.asarray(K, float) / np.outer(d, d)


def _best_q2(kernels, y: np.ndarray):
    """The best ten-fold q2 over the kernels offered, in the convention of run_corrosion.py."""
    from cmfqsar.model import cv_predict
    best = None
    for key, K in kernels.items():
        for lam in CORR_LAMS:
            pred = cv_predict(np.asarray(K, float), y, "krr", {"lam": float(lam)},
                              folds=min(CORR_FOLDS, len(y)), seed=SEED)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (float(q2), key, float(lam))
    return best


def _surface_block(mols, names, rec, fs, sites) -> np.ndarray:
    """The slab receptor's terms for every molecule, at a pose this program produces itself.

    Each molecule is embedded, laid flat by the rule of item EC2 and then refined against the
    slab, which is the whole pipeline a user without a slab calculation has.  The Fe(110) half of
    this item measured that a score read at the *relaxed* pose carries the relaxation's own
    outcome, so there is no relaxed pose to fall back on here and none is wanted.
    """
    rows = []
    for mol, name in zip(mols, names):
        coords, _anchor = rule_pose(mol, "flat", height=RULE_HEIGHT)
        posed = build_fields(_posed(mol, coords), fs, name)
        docked, res = dock_on_surface(posed, rec, fs, starts=8, seed=SEED)
        sc = surface_score(build_fields(_posed(mol, docked), fs, name), rec, fs, sites=sites)
        rows.append([res.burial_after, res.image_after, res.height_after,
                     sc.sites.get("top", 0.0), sc.sites.get("bridge", 0.0),
                     sc.sites.get("hollow", 0.0)])
    return np.asarray(rows, float)


def corrosion_stage(limit_series: int = 0, min_rows: int = CORROSION_MIN_ROWS) -> Dict[str, object]:
    """Do the slab receptor's terms predict a *measured* adsorption, and do they add anything?

    The Fe(110) half of this item has a target that came from the same relaxation as the pose, so
    a score read at that pose cannot be told apart from a readout of the relaxation.  The
    corrosion sets have no such circularity: 23 congeneric series of iron corrosion inhibitors
    with a measured potency in acid and, for most of them, a measured Langmuir adsorption free
    energy.  The controls are the ones this repository already published on these very sets --
    atom-type counts with no geometry at all, and the five fields in a frame the program invents.
    """
    from cmfqsar.chemotypes import count_kernel
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable
    from cmfqsar.kernels import kernel_matrices
    import glob

    slab, rec = receptor()
    fs = fieldset()
    sites = surface_sites(slab)
    paths = sorted(glob.glob(os.path.join(CORROSION, "*.csv")))
    if limit_series:
        paths = paths[:limit_series]
    out: Dict[str, object] = {"n_series_offered": len(paths), "min_rows": min_rows,
                              "targets": list(CORROSION_TARGETS), "terms": list(SURFACE_TERMS),
                              "folds": CORR_FOLDS, "series": {}}
    for path in paths:
        tag = os.path.splitext(os.path.basename(path))[0]
        ds = load_dataset(path, activity_prop="activity")
        mols, names = ds.mols, ds.names
        if len(mols) < min_rows:
            continue
        t0 = time.time()
        Z = _surface_block(mols, names, rec, fs, sites)
        k_surface = {"surface": _descriptor_kernel(Z)}
        k_counts = {"counts": count_kernel(mols, names)}
        k_fields, k_both = {}, {}
        for alpha in CORR_ALPHAS:
            fs_a = fieldset(alpha)
            mf = [build_fields(_posed(m, rule_pose(m, "flat", height=RULE_HEIGHT)[0]), fs_a, n)
                  for m, n in zip(mols, names)]
            table = AtomTable.from_molecules(mf)
            K = _unit_diagonal(kernel_matrices(table, table, fs_a.specs).sum(axis=0))
            k_fields[alpha] = K
            k_both[alpha] = K + k_surface["surface"]
        rec_series: Dict[str, object] = {"n": len(mols), "seconds": time.time() - t0}
        print(f"  === {tag}: {len(mols)} molecules ({rec_series['seconds']:.0f} s of posing)",
              flush=True)
        for target in CORROSION_TARGETS:
            values = np.array([float(m.GetProp(target))
                               if m.HasProp(target) and str(m.GetProp(target)).strip() else np.nan
                               for m in mols])
            keep = np.flatnonzero(np.isfinite(values))
            if len(keep) < min_rows:
                continue
            y = values[keep]
            sub = {"counts": k_counts["counts"][np.ix_(keep, keep)],
                   "surface": k_surface["surface"][np.ix_(keep, keep)]}
            rec_target: Dict[str, object] = {"n": int(len(keep))}
            for label, kernels in (("counts", {"counts": sub["counts"]}),
                                   ("surface", {"surface": sub["surface"]}),
                                   ("fields", {a: K[np.ix_(keep, keep)]
                                               for a, K in k_fields.items()}),
                                   ("fields+surface", {a: K[np.ix_(keep, keep)]
                                                       for a, K in k_both.items()})):
                q2, key, lam = _best_q2(kernels, y)
                rec_target[label] = {"q2": q2, "alpha": None if key in ("counts", "surface")
                                     else float(key), "lam": lam}
            # which of the six terms carries the target on its own, as a correlation
            best_term, best_r = None, 0.0
            for k, name in enumerate(SURFACE_TERMS):
                col = Z[keep, k]
                if col.std() > 1e-12:
                    r = abs(float(np.corrcoef(col, y)[0, 1]))
                    if r > best_r:
                        best_term, best_r = name, r
            rec_target["best_term"] = best_term
            rec_target["best_term_abs_r"] = best_r
            print(f"    {target:<9} n {len(keep):3d}  counts {rec_target['counts']['q2']:+.3f}  "
                  f"fields {rec_target['fields']['q2']:+.3f}  "
                  f"surface {rec_target['surface']['q2']:+.3f}  "
                  f"fields+surface {rec_target['fields+surface']['q2']:+.3f}   "
                  f"(best term {best_term} |r| {best_r:.2f})", flush=True)
            rec_series[target] = rec_target
        out["series"][tag] = rec_series
    # the means over the series, which is how run_corrosion.py reports
    out["means"] = {}
    for target in CORROSION_TARGETS:
        rows = [s[target] for s in out["series"].values() if target in s]
        if not rows:
            continue
        means = {label: float(np.mean([r[label]["q2"] for r in rows]))
                 for label in ("counts", "fields", "surface", "fields+surface")}
        wins = {label: int(sum(1 for r in rows if r[label]["q2"] > r["counts"]["q2"]))
                for label in ("fields", "surface", "fields+surface")}
        adds = int(sum(1 for r in rows if r["fields+surface"]["q2"] > r["fields"]["q2"]))
        out["means"][target] = {"n_series": len(rows), "q2": means,
                                "beats_counts": wins, "surface_adds_to_fields": adds}
        print(f"  --- {target}: over {len(rows)} series, counts {means['counts']:+.3f}, "
              f"fields {means['fields']:+.3f}, surface {means['surface']:+.3f}, "
              f"fields+surface {means['fields+surface']:+.3f}; the surface block beats counts in "
              f"{wins['surface']} series and adds to the fields in {adds}", flush=True)
    return out


# ------------------------------------------------------------------------------- the table

def _read(base: str, name: str):
    path = os.path.join(base, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def write_table(path: Optional[str] = None) -> str:
    base = path or RESULTS
    os.makedirs(base, exist_ok=True)
    rec, pose, score = _read(base, "receptor"), _read(base, "pose"), _read(base, "score")
    corrosion = _read(base, "corrosion")
    out: List[str] = ["# A metal slab as the receptor", "",
                      "Plan item EC6. The Fe(110) sets hold the relaxed adsorbed pose and the",
                      "adsorption energy of every molecule from the UMA potential, so this item",
                      "has reference poses that no protein set has. Two halves: does docking",
                      "constrained to the surface recover the pose, and does a complementarity",
                      "score track the energy.", ""]
    if rec:
        out += ["## The receptor", "",
                f"The slab is {rec['slab_atoms']} atoms of "
                f"{rec['slab']['element']} in {len(rec['layer_heights'])} layers "
                f"({rec['slab']['nx']} x {rec['slab']['ny']} surface cell, a = "
                f"{rec['slab']['a']} angstrom), its top layer at z = 0. The receptor is its top",
                f"two layers with periodic images out to {rec['image_cutoff']} angstrom: "
                f"{rec['receptor_atoms']} atoms.", "",
                f"**Fields the slab carries:** {', '.join(rec['nonzero_fields'])}. "
                f"**Identically zero:** {', '.join(rec['zero_fields'])} — a one-element cell has "
                "no electronegativity differences, so every charge model gives zero, and the "
                "hydrophobic and hydrogen-bond increments are fitted to organic fragments. All "
                "four complementary pairs are therefore zero against a clean metal and what is "
                "left is burial.", ""]
        im = rec["image"]
        out += ["**The image charge**, which is what a metal has instead. For the first molecule "
                f"of the series it is {im['at_its_pose_eV']:+.4f} eV at its relaxed pose and "
                f"{im['three_angstrom_higher_eV']:+.4f} eV three angstrom higher. It changes by "
                f"{im['in_plane_change']:.1e} eV under an in-plane shift and "
                f"{im['turn_change']:.1e} eV under a turn about the normal, so it cannot say "
                "where on the lattice a molecule sits; only the shape term can.", "",
                "| surface sites of the top layer | count |", "|---|---|"]
        for k, v in rec["sites"].items():
            out.append(f"| {k} | {v} |")
        out += [""]
    if pose:
        out += ["## The pose", "",
                "Median error against the relaxed pose over heavy atoms: in-plane RMSD after the",
                "best rotation about the normal and translation in the plane (the lattice site a",
                "molecule sits over is not part of a pose), and the RMSD of the heights alone.",
                "`rule` is item EC2's flat rule pose; `docked` refines it against the slab.", ""]
        for node, block in pose["nodes"].items():
            out += [f"### {node} ({block['n']} molecules)", "",
                    "| start | in-plane (A) | in-plane, bound (A) | height (A) | within 1 A |",
                    "|---|---|---|---|---|"]
            for label in pose["variants"]:
                s = block.get(label)
                if not s or not s.get("n"):
                    continue
                out.append(f"| {label} | {s['in_plane_median']:.2f} | "
                           f"{s.get('in_plane_median_bound', float('nan')):.2f} | "
                           f"{s['height_median']:.2f} | {100 * s['within_1A']:.0f} % |")
            out += [""]
    if score:
        out += ["## The energy", "",
                "Leave-one-out q2 of a straight line on each term against `E_ads_kJmol`. The two",
                "controls are item EC2's: the lowest heavy-atom height of the relaxed pose, and",
                "the chemisorbed flag. A score that does not beat the height has found nothing",
                "the relaxation had not already put into the frame.", ""]
        for node, block in score["nodes"].items():
            out += [f"### {node} ({block['n']} molecules, {block['n_bound']} chemisorbed)", "",
                    "| term | leave-one-out q2 | r |", "|---|---|---|"]
            for key in ("burial", "image", "height", "burial+image", "burial_docked",
                        "image_docked", "height_docked", "burial+image_docked",
                        "control_relaxed_height", "control_bound_flag"):
                if key not in block:
                    continue
                r = block[key].get("pearson")
                out.append(f"| {key} | {block[key]['loo_q2']:+.3f} | "
                           + (f"{r:+.3f} |" if r is not None else "|"))
            out += [""]
    if corrosion:
        out += ["## The measured sets: 23 corrosion-inhibitor series", "",
                "The Fe(110) target came from the same relaxation as the pose, so a score read",
                "there cannot be told from a readout of the relaxation. These series have no such",
                "circularity: a measured potency in acid and, for most molecules, a measured",
                "Langmuir adsorption free energy. The slab terms are read at a pose this program",
                "produces itself: embedded, laid flat by the rule of item EC2, then refined",
                "against the slab. `counts` is atom-type counts with no geometry at all, and the",
                "grids, the fold count and the selection convention are those of",
                "`run_corrosion.py`, whose published means over these series are counts +0.189,",
                "the aligned five-field model +0.269 and the invariant kernel +0.278.", ""]
        for target, m in corrosion.get("means", {}).items():
            out += [f"### {target} ({m['n_series']} series)", "",
                    "| model | mean q2 over the series | series where it beats counts |",
                    "|---|---|---|"]
            for label in ("counts", "fields", "surface", "fields+surface"):
                beat = m["beats_counts"].get(label)
                out.append(f"| {label} | {m['q2'][label]:+.3f} | "
                           + (f"{beat} of {m['n_series']} |" if beat is not None else "|"))
            out += ["", f"The surface block adds to the five fields in "
                    f"{m['surface_adds_to_fields']} of {m['n_series']} series.", ""]
        out += ["### Every series", "",
                "| series | n | target | counts | fields | surface | fields+surface | "
                "best single term |", "|---|---|---|---|---|---|---|"]
        for tag, block in corrosion["series"].items():
            for target in corrosion["targets"]:
                if target not in block:
                    continue
                r = block[target]
                out.append(f"| {tag} | {r['n']} | {target} | {r['counts']['q2']:+.3f} | "
                           f"{r['fields']['q2']:+.3f} | {r['surface']['q2']:+.3f} | "
                           f"{r['fields+surface']['q2']:+.3f} | {r['best_term']} "
                           f"({r['best_term_abs_r']:.2f}) |")
        out += [""]
    text = "\n".join(out) + "\n"
    dest = os.path.join(base, "table.md")
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"written {dest}")
    return text


STAGES = ("receptor", "pose", "score", "corrosion")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=STAGES + ("all",), default="all")
    ap.add_argument("--limit", type=int, default=0, help="only the first N molecules of a node")
    ap.add_argument("--limit-series", type=int, default=0,
                    help="only the first N corrosion series")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        write_table()
        return
    for stage in (STAGES if args.stage == "all" else (args.stage,)):
        print(f"[{stage}]", flush=True)
        t0 = time.time()
        if stage == "pose":
            res = pose_stage(limit=args.limit)
        elif stage == "corrosion":
            res = corrosion_stage(limit_series=args.limit_series)
        else:
            res = {"receptor": receptor_stage, "score": score_stage}[stage]()
        res["seconds"] = time.time() - t0
        with open(os.path.join(RESULTS, f"{stage}.json"), "w", encoding="utf-8",
                  newline="\n") as fh:
            json.dump(res, fh, indent=1)
        print(f"  {stage} done in {res['seconds']:.0f} s", flush=True)
    write_table()


if __name__ == "__main__":
    main()
