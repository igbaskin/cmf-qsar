#!/usr/bin/env python
"""
The rest of plan item EC8 on the single-atom catalysts: the capped cluster, a co-field that can
be read, and a frame averaged over the symmetry of the site.

The pilot (`benchmarks/run_sac.py`) and plan item M5 (`benchmarks/run_sites.py`) measured the
site of the 133 single-atom catalysts of He, Wang and Ouyang (Chem. Mater. 2023) as an
environment cut from the periodic sheet and read with the inorganic fields.  EC8 asked for three
things they left open, and this script measures them on the same catalysts, targets and folds.

Stage gate: the cut-and-terminate step
--------------------------------------
The item's own route: a finite cluster cut from the sheet around the metal, every cut bond capped
with hydrogen, read as a metal complex by the path of plan item EC4 (`cmfqsar/clusters.py`, then
`complexes.build_complex` and the five classical molecular fields with Gasteiger charges and the
EC4 metal parameters).  Clusters of 3, 4 and 5 bond shells around the metal (25, 45 and 57
atoms).  The ligand's charge is the first that closes its shell, which for a capped divacancy
is minus the number of carbon donors, so the metal's formal charge is +4 on M@C4 and 0 on M@N4
whatever the metal (`cmfqsar/clusters.py` says why).  Three methods:

* the overlap kernel of the cluster in the shared frame of the sheet (metal at the origin above
  the sheet plane, normal along z);
* the rotation-invariant autocorrelation of the cluster;
* the cluster overlap kernel plus the kernel of the tabulated metal properties, because the
  molecular fields see the metal only through its radius and a charge that is the same for all.

The gate of the item, with the references the plan names and the pilot used: the periodic
autocorrelation of the whole cell; M5's site profile (the exact split of that autocorrelation
over the sites, read at the metal); the site overlap kernel of the pilot with the extended
inorganic fields; and the tabulated metal properties with the nitrogen count.  Every method
chooses its own settings (cluster size, attenuation, normalisation, ridge) in inner folds of the
same kind as the outer ones.  Three fold designs: random, one metal held out, one motif held out.

Stage cofield: the reading at a fixed smoothness
------------------------------------------------
The rerun of 5 November found that the co-field of the difference dE_H - dE_OH, read at the
metal, is +0.16 to +0.23 eV per unit at the pilot's smooth kernel (envelope 0.2, attenuation 0.5,
raw, ridge 0.1) and ten times larger with the sign changing between two subsets at the kernel
chosen for accuracy (attenuation 2, cosine, ridge 0.001).  Here both kernels are fitted on 100
random halves of the catalysts, and for every field the reading at the metal, the first and the
second shell is reported with how often its sign agrees with the full-set fit.

Stage symmetry: the frame averaged over the site
------------------------------------------------
Every sheet is the same divacancy in the same place, so the frame is free; but the site has the
symmetry of the divacancy (D2h: the identity, the turn by 180 degrees about the normal and the
two mirror planes through it), and a frame kernel that compares two sites in one orientation
only counts as different two arrangements that a symmetry operation maps onto each other.  The
averaged kernel K(a, b) = mean over the four operations g of K(a, g b), normalised afterwards, is
the device of plan item M14.  It is measured on the pilot's site overlap kernel and on the
cluster overlap kernel against the same kernels without the average.

Usage
-----
    python benchmarks/run_sac_clusters.py                 # every stage, then the table
    python benchmarks/run_sac_clusters.py --stage gate
    python benchmarks/run_sac_clusters.py --report
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_sac as R  # noqa: E402

from cmfqsar.clusters import cluster_complex  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

RESULTS = os.path.join(HERE, "results", "sac_clusters")
#: the cluster sizes, in bonds from the metal
HOPS = (3, 4, 5)
#: the molecular fields of the EC4 path
MOLECULAR = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
CLUSTER_ALPHAS = (0.5, 1.0, 2.0)
CLUSTER_SIGMAS = (0.5, 0.75)
#: weights of the metal-property kernel added to the cosine cluster kernel
PROPERTY_WEIGHTS = (0.5, 1.0, 2.0)
DESIGNS = ("random", "metal-out", "motif-out")
#: the two kernels of the co-field stage: the pilot's smooth one, and the one accuracy chose
SMOOTH = {"rho": 0.2, "alpha": 0.5, "normalize": False, "lam": 0.1}
SHARP = {"rho": 0.2, "alpha": 2.0, "normalize": True, "lam": 0.001}
HALVES = 100
STAGES = ("gate", "cofield", "symmetry")


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def jdump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=float)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cosine(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / np.outer(d, d)


# --------------------------------------------------------------------------- #
# The clusters
# --------------------------------------------------------------------------- #
def clusters(data, hops: int, log=print):
    """The capped cluster of every catalyst as an RDKit complex, in the frame of the sheet."""
    mols, stats = [], {"faults": [], "atoms": [], "caps": [], "metal_charge": []}
    for k, a in enumerate(data.structures):
        mi = R._metal_index(a)
        mol, cl, rep = cluster_complex(a, mi, hops=hops, origin=data.centres[k],
                                       flip_z=bool(data.flip[k]), name=data.names[k])
        if mol is None:
            stats["faults"].append(f"{data.names[k]}: {rep.fault}")
            mols.append(None)
            continue
        mols.append(mol)
        stats["atoms"].append(cl.n_atoms)
        stats["caps"].append(cl.n_caps)
        stats["metal_charge"].append(rep.metal_charge)
    return mols, stats


def molecular_fieldset(alpha: float) -> FieldSet:
    return FieldSet.create(physchem=list(MOLECULAR), alpha=alpha, charge_model="gasteiger",
                           param_scheme="rdkit")


def cluster_tables(data, log=print):
    """The field records of the clusters, by size and attenuation, and what the cut gave."""
    tables, stats = {}, {}
    for hops in HOPS:
        t0 = time.time()
        mols, st = clusters(data, hops, log)
        if st["faults"]:
            raise SystemExit(f"{len(st['faults'])} clusters of {hops} shells could not be built: "
                             + "; ".join(st["faults"][:5]))
        for alpha in CLUSTER_ALPHAS:
            fs = molecular_fieldset(alpha)
            mf = [build_fields(m, fs, n) for m, n in zip(mols, data.names)]
            tables[(hops, alpha)] = (AtomTable.from_molecules(mf), fs)
        by_motif = {}
        for m, q in zip(data.motifs, st["metal_charge"]):
            by_motif.setdefault(R.MOTIF_NAMES.get(m, m), set()).add(int(q))
        stats[hops] = {"atoms": sorted(set(st["atoms"])), "caps": sorted(set(st["caps"])),
                       "metal_charge_by_motif": {k: sorted(v) for k, v in by_motif.items()},
                       "seconds": time.time() - t0}
        log(f"  {hops} shells: {st['atoms'][0]} atoms with {st['caps'][0]} hydrogen caps; "
            f"metal charge by motif "
            + ", ".join(f"{k} {'/'.join(f'{q:+d}' for q in sorted(v))}"
                        for k, v in sorted(by_motif.items()))
            + f"; {time.time() - t0:.0f} s")
    return tables, stats


def cluster_families(data, tables) -> Dict[str, Dict[str, np.ndarray]]:
    overlap, invariant, mixed = {}, {}, {}
    props = R.descriptor_kernels(data)["gauss x1"]
    for (hops, alpha), (table, fs) in tables.items():
        for normalize in (False, True):
            K = R.overlap_kernel(table, fs, normalize)
            overlap[f"{hops} shells alpha={alpha:g} {'cosine' if normalize else 'raw'}"] = K
            if normalize:
                for w in PROPERTY_WEIGHTS:
                    mixed[f"{hops} shells alpha={alpha:g} + {w:g} x properties"] = K + w * props
        if alpha == 1.0:
            for s in CLUSTER_SIGMAS:
                invariant[f"{hops} shells sigma={s:g}"] = CMFModel(
                    fieldset=fs, kernel_source="autocorrelation", ac_sigma=s, ac_d_max=12.0,
                    normalize=True).training_kernel(table)
    return {"capped cluster, overlap (EC4 path)": overlap,
            "capped cluster, invariant": invariant,
            "capped cluster overlap + metal properties": mixed}


def reference_families(data, log=print) -> Dict[str, Dict[str, np.ndarray]]:
    import run_sites as S5
    from cmfqsar import sites as S
    t0 = time.time()
    sites = [R._metal_index(a) for a in data.structures]
    crystals = {kind: R.crystals_for(data, R.fieldset_for(kind, data.metals))
                for kind in ("classic", "extended", "indicator")}
    n_sites = [c.n_atoms for c in crystals["classic"]]
    index = S.one_site_targets(sites, np.zeros(len(data)), n_sites, name="binding")
    m5 = S5.sac_kernels(data, crystals, index, log=lambda *a: None)
    overlap, _inv, _tab = R.site_kernels(data, "extended", log=lambda *a: None)
    log(f"  references built in {time.time() - t0:.0f} s")
    return {"whole cell autocorrelation": R.periodic_kernels(data),
            "site profile (M5)": m5["site profile (M5)"],
            "site overlap, extended (pilot)": overlap,
            "metal properties and N count": R.descriptor_kernels(data)}


def stage_gate(args, log=print) -> dict:
    warnings.filterwarnings("ignore")
    data = R.load_set(log=lambda *a: None)
    log(f"gate: {len(data)} catalysts")
    tables, stats = cluster_tables(data, log)
    families = {**cluster_families(data, tables), **reference_families(data, log)}
    ours = [n for n in families if n.startswith("capped")]
    refs = [n for n in families if not n.startswith("capped")]
    out = {"n": len(data), "clusters": {str(h): s for h, s in stats.items()},
           "models": [], "pairs": [],
           "targets": {t: int(np.sum(np.isfinite(data.dE[a])))
                       for t, a in zip(R.TARGETS, R.ADSORBATES)}}
    for design in DESIGNS:
        for adsorbate in R.ADSORBATES:
            y = data.dE[adsorbate]
            target = f"dE_{adsorbate}"
            res = {}
            for name, fam in families.items():
                res[name] = R.cross_validate(name, fam, y, design, target, data)
                out["models"].append(res[name].entry())
                log("  " + res[name].summary())
            for a in ours:
                for b in refs:
                    d, lo, hi = R.bootstrap_difference(y, res[a].predicted, res[b].predicted,
                                                       runs=args.bootstrap)
                    out["pairs"].append({"design": design, "target": target, "method": a,
                                         "against": b, "delta": d, "low": lo, "high": hi})
    return out


# --------------------------------------------------------------------------- #
# Stage cofield
# --------------------------------------------------------------------------- #
def _cofield_readings(table, fs, normalize, lam, rows, data, points):
    """The co-field of dE_H, dE_OH and their difference at the named points, by field."""
    out = {}
    models = {}
    for a in ("H", "OH"):
        y = data.dE[a]
        have = np.array([i for i in rows if np.isfinite(y[i])])
        m = CMFModel(fieldset=fs.copy(), normalize=normalize, method="gp",
                     params={"lam": float(lam)})
        m.fit(table.subset(have), y[have], cv_folds=0)
        models[a] = m
    for fname in fs.names:
        if fname.startswith("cif:"):
            continue
        f = fs.names.index(fname)
        out[fname] = {k: float(np.mean(R.cofield_at(models["H"], f, v))
                               - np.mean(R.cofield_at(models["OH"], f, v)))
                      for k, v in points.items() if k != "2 A above the metal"}
    return out


def stage_cofield(args, log=print) -> dict:
    warnings.filterwarnings("ignore")
    data = R.load_set(log=lambda *a: None)
    both = [i for i in range(len(data)) if np.isfinite(data.dE["H"][i])
            and np.isfinite(data.dE["OH"][i])]
    out = {"n": len(both), "halves": args.halves, "kernels": {}}
    rng = np.random.default_rng(R.SEED)
    for label, kern in (("smooth (the pilot's)", SMOOTH), ("sharp (chosen for accuracy)", SHARP)):
        fs = R.fieldset_for("extended", data.metals, alpha=kern["alpha"])
        envs = R.site_environments(data, fs, kern["rho"])
        table = AtomTable.from_molecules(envs)
        ref_env = envs[data.names.index("Fe@N4")]
        points = R.named_points(ref_env, float(np.mean(data.height)))
        t0 = time.time()
        full = _cofield_readings(table, fs, kern["normalize"], kern["lam"], both, data, points)
        halves = []
        for _ in range(args.halves):
            pick = sorted(rng.choice(both, size=len(both) // 2, replace=False).tolist())
            halves.append(_cofield_readings(table, fs, kern["normalize"], kern["lam"], pick,
                                            data, points))
        entry = {"kernel": kern, "fields": {}}
        for fname, places in full.items():
            entry["fields"][fname] = {}
            for place, v in places.items():
                vals = np.array([h[fname][place] for h in halves])
                entry["fields"][fname][place] = {
                    "full": v, "median": float(np.median(vals)),
                    "low": float(np.percentile(vals, 5)), "high": float(np.percentile(vals, 95)),
                    "same_sign": float(np.mean(np.sign(vals) == np.sign(v)))}
        out["kernels"][label] = entry
        log(f"  co-field, {label}: {args.halves} halves in {time.time() - t0:.0f} s")
        for fname in ("electrostatic", "steric"):
            e = entry["fields"][fname]["metal"]
            log(f"    {fname} at the metal: {e['full']:+.3f} on all, half-sets 5-95 % "
                f"[{e['low']:+.3f}, {e['high']:+.3f}], same sign {100 * e['same_sign']:.0f} %")
    return out


# --------------------------------------------------------------------------- #
# Stage symmetry
# --------------------------------------------------------------------------- #
def site_axes(env) -> np.ndarray:
    """The in-plane axes of the divacancy: the principal axes of the four atoms that hold the
    metal, as the rows of a 3 x 3 matrix with the normal last."""
    xy = np.hypot(env.coords[:, 0], env.coords[:, 1])
    first = env.coords[(xy < 2.4) & (xy > 0.3)][:, :2]
    first = first - first.mean(axis=0)
    _, vecs = np.linalg.eigh(first.T @ first)
    e1 = np.append(vecs[:, 1], 0.0)
    e2 = np.append(vecs[:, 0], 0.0)
    return np.vstack([e1, e2, [0.0, 0.0, 1.0]])


def operations(axes: np.ndarray) -> List[np.ndarray]:
    """The four operations of D2h that keep the normal: identity, C2 about z, the two mirrors."""
    out = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            out.append(axes.T @ np.diag([sx, sy, 1.0]) @ axes)
    return out


def moved(rec, g: np.ndarray):
    return dataclasses.replace(rec, coords=np.asarray(rec.coords, float) @ g.T)


def averaged_kernel(records, fs, ops) -> np.ndarray:
    """mean over g of the raw overlap K(a, g b), from one joint kernel per operation."""
    n = len(records)
    acc = np.zeros((n, n))
    for g in ops:
        table = AtomTable.from_molecules(list(records) + [moved(r, g) for r in records])
        K = R.overlap_kernel(table, fs, False)
        acc += K[:n, n:]
    acc /= len(ops)
    return 0.5 * (acc + acc.T)


def stage_symmetry(args, log=print) -> dict:
    warnings.filterwarnings("ignore")
    data = R.load_set(log=lambda *a: None)
    ref = data.names.index("Fe@N4")
    out = {"n": len(data), "models": [], "pairs": []}
    families = {}
    t0 = time.time()
    plain, avg = {}, {}
    for alpha in R.ALPHAS:
        fs = R.fieldset_for("extended", data.metals, alpha=alpha)
        envs = R.site_environments(data, fs, SMOOTH["rho"])
        ops = operations(site_axes(envs[ref]))
        out["operations"] = len(ops)
        raw = R.overlap_kernel(AtomTable.from_molecules(envs), fs, False)
        sym = averaged_kernel(envs, fs, ops)
        plain[f"alpha={alpha:g} raw"], plain[f"alpha={alpha:g} cosine"] = raw, cosine(raw)
        avg[f"alpha={alpha:g} raw"], avg[f"alpha={alpha:g} cosine"] = sym, cosine(sym)
    families["site overlap, extended, one frame"] = plain
    families["site overlap, extended, averaged over D2h"] = avg
    log(f"  site kernels with and without the average: {time.time() - t0:.0f} s")
    t0 = time.time()
    mols, _st = clusters(data, 4)
    plain, avg = {}, {}
    for alpha in CLUSTER_ALPHAS:
        fs = molecular_fieldset(alpha)
        mf = [build_fields(m, fs, n) for m, n in zip(mols, data.names)]
        ops = operations(site_axes(mf[ref]))
        raw = R.overlap_kernel(AtomTable.from_molecules(mf), fs, False)
        sym = averaged_kernel(mf, fs, ops)
        plain[f"alpha={alpha:g} raw"], plain[f"alpha={alpha:g} cosine"] = raw, cosine(raw)
        avg[f"alpha={alpha:g} raw"], avg[f"alpha={alpha:g} cosine"] = sym, cosine(sym)
    families["capped cluster (4 shells), one frame"] = plain
    families["capped cluster (4 shells), averaged over D2h"] = avg
    log(f"  cluster kernels with and without the average: {time.time() - t0:.0f} s")
    for design in DESIGNS:
        for adsorbate in R.ADSORBATES:
            y = data.dE[adsorbate]
            target = f"dE_{adsorbate}"
            res = {}
            for name, fam in families.items():
                res[name] = R.cross_validate(name, fam, y, design, target, data)
                out["models"].append(res[name].entry())
                log("  " + res[name].summary())
            for a, b in (("site overlap, extended, averaged over D2h",
                          "site overlap, extended, one frame"),
                         ("capped cluster (4 shells), averaged over D2h",
                          "capped cluster (4 shells), one frame")):
                d, lo, hi = R.bootstrap_difference(y, res[a].predicted, res[b].predicted,
                                                   runs=args.bootstrap)
                out["pairs"].append({"design": design, "target": target, "method": a,
                                     "against": b, "delta": d, "low": lo, "high": hi})
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _q2_table(models, names) -> List[str]:
    L = []
    for design in DESIGNS:
        L.append(f"**{design}**\n")
        L.append("| Method | dE_H | dE_O | dE_OH | dE_OOH | Mean |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for name in names:
            q = [next((m["q2"] for m in models if m["name"] == name and m["design"] == design
                       and m["target"] == t), float("nan")) for t in R.TARGETS]
            L.append(f"| {name} | " + " | ".join(f"{v:+.2f}" for v in q)
                     + f" | {np.nanmean(q):+.3f} |")
        L.append("")
    return L


def _pairs_table(pairs, methods) -> List[str]:
    L = ["| Method | Against | Design | dE_H | dE_O | dE_OH | dE_OOH |",
         "|---|---|---|---:|---:|---:|---:|"]
    for a in methods:
        against = sorted({p["against"] for p in pairs if p["method"] == a})
        for b in against:
            for design in DESIGNS:
                cells = []
                for t in R.TARGETS:
                    p = next(p for p in pairs if p["method"] == a and p["against"] == b
                             and p["design"] == design and p["target"] == t)
                    clear = p["low"] > 0 or p["high"] < 0
                    txt = f"{p['delta']:+.2f} [{p['low']:+.2f}, {p['high']:+.2f}]"
                    cells.append(f"**{txt}**" if clear else txt)
                L.append(f"| {a} | {b} | {design} | " + " | ".join(cells) + " |")
    return L


def report(results_dir: str = RESULTS) -> str:
    L = ["# The rest of EC8: capped clusters, a readable co-field, a symmetry-averaged frame\n"]
    prov_path = os.path.join(results_dir, "run.json")
    if os.path.exists(prov_path):
        prov = jload(prov_path)
        L.append("`" + " ".join(["python benchmarks/run_sac_clusters.py"] + list(prov["argv"]))
                 + f"`, written {prov['written']}.\n")
    path = os.path.join(results_dir, "gate.json")
    if os.path.exists(path):
        d = jload(path)
        L.append(f"## The gate: the capped cluster against the references\n")
        L.append(f"{d['n']} catalysts; the targets have "
                 + ", ".join(f"{t} {n}" for t, n in d["targets"].items()) + " values. The "
                 "ligand of every cluster takes the first charge that closes its shell, and the "
                 "metal the rest of the cluster's zero charge. q2 in cross-validation; every "
                 "method chooses its settings and its ridge in inner folds of the same kind as "
                 "the outer ones.\n")
        L.append("| Bond shells around the metal | Atoms | Hydrogen caps | "
                 "Formal charge of the metal, by motif |")
        L.append("|---:|---:|---:|---|")
        for h, s in d["clusters"].items():
            L.append(f"| {h} | {', '.join(map(str, s['atoms']))} | "
                     f"{', '.join(map(str, s['caps']))} | "
                     + ", ".join(f"{k} {'/'.join(f'{q:+d}' for q in v)}"
                                 for k, v in sorted(s["metal_charge_by_motif"].items()))
                     + " |")
        L.append("")
        names = list(dict.fromkeys(m["name"] for m in d["models"]))
        L += _q2_table(d["models"], names)
        L.append("Differences in q2 with a 95 % bootstrap interval over the catalysts, the "
                 "cluster method minus the reference; bold where the interval excludes zero.\n")
        L += _pairs_table(d["pairs"], [n for n in names if n.startswith("capped")])
        L.append("")
    path = os.path.join(results_dir, "cofield.json")
    if os.path.exists(path):
        d = jload(path)
        L.append("## The co-field of dE_H - dE_OH: smooth against sharp\n")
        L.append(f"The difference of the two co-fields, in eV per unit of the field, read at "
                 f"the metal and averaged over the first and the second shell of Fe@N4, fitted "
                 f"on all {d['n']} catalysts and on {d['halves']} random halves. 'Same sign' is "
                 "how often a half gives the sign of the full fit.\n")
        L.append("| Kernel | Field | Place | All | Halves, 5 to 95 % | Same sign |")
        L.append("|---|---|---|---:|---|---:|")
        for label, entry in d["kernels"].items():
            for fname, places in entry["fields"].items():
                for place, e in places.items():
                    L.append(f"| {label} | {fname} | {place} | {e['full']:+.3f} | "
                             f"[{e['low']:+.3f}, {e['high']:+.3f}] | "
                             f"{100 * e['same_sign']:.0f} % |")
        L.append("")
    path = os.path.join(results_dir, "symmetry.json")
    if os.path.exists(path):
        d = jload(path)
        L.append("## The frame averaged over the symmetry of the site\n")
        L.append(f"The overlap kernels averaged over the {d.get('operations', 4)} operations of "
                 "the divacancy that keep the normal (identity, the turn by 180 degrees, the two "
                 "mirrors), against the same kernels in one frame.\n")
        names = list(dict.fromkeys(m["name"] for m in d["models"]))
        L += _q2_table(d["models"], names)
        L += _pairs_table(d["pairs"], sorted({p["method"] for p in d["pairs"]}))
        L.append("")
    return "\n".join(L)


def write_table(results_dir: str = RESULTS) -> str:
    path = os.path.join(results_dir, "table.md")
    text = report(results_dir)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--bootstrap", type=int, default=R.RUNS,
                    help="bootstrap resamples of every difference (default %(default)s)")
    ap.add_argument("--halves", type=int, default=HALVES,
                    help="random halves of the co-field stage (default %(default)s)")
    ap.add_argument("--force", action="store_true", help="redo a stage that is already stored")
    ap.add_argument("--report", action="store_true", help="rebuild the table and stop")
    return ap


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    if args.report:
        print(write_table(RESULTS))
        return
    t0 = time.time()
    lines: List[str] = []

    def log(s=""):
        print(s, flush=True)
        lines.append(str(s))

    jobs = {"gate": stage_gate, "cofield": stage_cofield, "symmetry": stage_symmetry}
    for name in STAGES:
        if name not in args.stage:
            continue
        path = result(f"{name}.json")
        if os.path.exists(path) and not args.force:
            log(f"{name}: already done")
            continue
        jdump(jobs[name](args, log), path)
    with open(result("log.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    jdump({"argv": list(argv) if argv is not None else sys.argv[1:],
           "minutes": round((time.time() - t0) / 60, 2),
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}, result("run.json"))
    print(write_table(RESULTS))
    print(f"Elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
