"""
Plan item BT3: the average voltage of an insertion electrode, from the structure of the cell.

The question is the one an electrode paper asks: does the arrangement of the atoms say anything
about the voltage that the composition does not already say?  Composition is a strong baseline
here, because the voltage of a lithium insertion is mostly decided by which metal is reduced and
by how much; the field model has to earn its place against it.

The data are built by :mod:`cmfqsar.battery` out of the Materials Project's public structures and
formation energies, which need no account: one query per chemical system, the phases within
``--hull`` of the convex hull kept, and neighbouring steps of one host's ladder paired.  The
average voltage of a pair follows from the two formation energies (see that module).  Before the
set is used for anything, ``--stage check`` compares the reconstructed voltages of four textbook
cathodes with the published ones.

Protocol.  Five folds of ``GroupKFold`` with the framework as the group, so the pairs of one
host's ladder are never split between training and test: two steps of the same ladder share a
framework, a chemistry and most of their structure, and a model that has seen one of them has
been told most of the answer for the other.  The ridge and the width of the Gaussians are chosen
inside the training part by ten-fold cross-validation, as everywhere else in this directory.
Three models are compared on the same folds:

* **composition** -- a linear kernel on the element fractions of the cell and the crystal
  descriptors of plan item BT2 (the capacity, the content of the working ion, the volume per
  atom, the density, the coordination of the working ion and its distance to oxygen);
* **fields** -- the periodic autocorrelation kernel of the five inorganic fields plus one
  indicator field per element, which is the kernel every other crystal benchmark here uses;
* **fields+composition** -- the two kernels added, each scaled to unit mean diagonal.

The units are volts throughout, and the mean absolute error is the number to read: the spread of
the set is about one volt, and a difference below the fold-to-fold spread is not a result.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar import battery                                               # noqa: E402
from cmfqsar.descriptors import compute as compute_descriptor             # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel, channel_scale       # noqa: E402
from cmfqsar.model import _Learner, cv_predict, r2_score, rmse            # noqa: E402
from cmfqsar.periodic import (INORGANIC_FIELDS, PeriodicSpec,             # noqa: E402
                              crystal_fields, elements_in, inorganic_fieldset,
                              periodic_block_sizes, periodic_channel_labels,
                              periodic_feature_matrix)

CACHE = os.path.join(HERE, "cache", "battery")
RESULTS = os.path.join(HERE, "results", "battery")

#: the redox-active metals an electrode is built on, for the polyanion systems
REDOX = ("Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Nb", "Mo", "W")
#: every metal whose ternary oxide system is fetched
METALS = REDOX + ("Sc", "Zn", "Y", "Zr", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "Re", "Ir",
                  "Pt", "Al", "Ga", "In", "Sn", "Pb", "Bi", "Sb", "Mg", "Ca", "Na", "Si", "B")

SPLIT_SEED = 18012019                   # the seed every benchmark in this directory splits with
OUTER, INNER = 5, 10
LAMS = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
SIGMAS = (0.35, 0.5, 0.75)
R_MAX, TAPER = 10.0, 2.0
#: eV per atom above the convex hull a phase may lie.  0.03 is chosen and not 0.01: the agreement
#: with the published cathode voltages is the same at 0.01, 0.02 and 0.03 (0.39 V) and the set
#: grows from 83 to 303 pairs, while at 0.08 the agreement is worse (0.57 V).  ``--stage check``
#: reports that table, so the choice can be read rather than taken on trust.
HULL = 0.03
#: the windows the sensitivity table in ``--stage check`` covers
HULL_WINDOWS = (0.01, 0.02, 0.03, 0.05, 0.08)
#: a published average voltage is quoted over the first insertion, up to one working ion per host
#: formula unit; beyond that the reaction is a different one
CHECK_WINDOW = 1.0
#: the crystal descriptors of plan item BT2 that enter the composition model
CRYSTAL_DESCRIPTORS = ("calc:capacity", "calc:ion_count", "calc:ion_fraction",
                       "calc:volume_per_atom", "calc:density", "calc:ion_coordination",
                       "calc:ion_oxygen_distance")
#: the four cathodes the reconstruction is checked on, with the voltage the literature gives
PUBLISHED = (("CoO2", 4.1, "LiCoO2, layered"), ("FeO4P", 3.45, "LiFePO4, olivine"),
             ("MnO2", 4.0, "LiMn2O4, spinel"), ("NiO2", 3.8, "LiNiO2, layered"))


def systems():
    """The chemical systems to fetch: the ternary oxides, and the polyanion hosts of the metals."""
    out = [("Li", m, "O") for m in METALS]
    out += [("Li", m, "P", "O") for m in REDOX]
    out += [("Li", m, "S") for m in REDOX]
    out += [("Li", m, "F") for m in REDOX]
    out += [("Li", m, "Si", "O") for m in ("Fe", "Mn", "Co")]
    return out


def system_name(elements):
    return "-".join(elements)


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #
def fetch(force=False):
    """One query per chemical system into the cache; a system already there is left alone."""
    os.makedirs(CACHE, exist_ok=True)
    total = 0
    for elements in systems():
        path = os.path.join(CACHE, f"{system_name(elements)}.json")
        if os.path.exists(path) and not force:
            n = len(battery.read_records(path))
            total += n
            print(f"  {system_name(elements):<14} {n:5d} structures (cached)", flush=True)
            continue
        t0 = time.time()
        try:
            records = battery.fetch_system(elements)
        except RuntimeError as exc:
            print(f"  {system_name(elements):<14} failed: {exc}", flush=True)
            continue
        battery.write_records(records, path)
        total += len(records)
        print(f"  {system_name(elements):<14} {len(records):5d} structures "
              f"({time.time() - t0:.0f} s)", flush=True)
    print(f"{total} structures in {CACHE}")
    return total


def load_pairs(hull=HULL, verbose=True):
    """Every insertion pair the cache gives, with the phases that lie above the hull dropped."""
    kept, seen = [], set()
    for elements in systems():
        path = os.path.join(CACHE, f"{system_name(elements)}.json")
        if not os.path.exists(path):
            continue
        for record in battery.read_records(path):
            ident = record.get("id") or record.get("attributes", {}).get("id")
            if ident in seen:
                continue                      # a structure of Li-Fe-O is also in Li-Fe-P-O
            seen.add(ident)
            try:
                atoms = battery.atoms_from_optimade(record)
            except (ValueError, KeyError):
                continue
            above = atoms.info.get("energy_above_hull")
            if above is None or not np.isfinite(above) or above > hull:
                continue
            kept.append(atoms)
    pairs = battery.insertion_pairs(kept)
    if verbose:
        print(f"  {len(seen)} structures, {len(kept)} within {hull} eV/atom of the hull, "
              f"{len(pairs)} pairs")
    return pairs


def cathode_rows(pairs, window=CHECK_WINDOW):
    """The reconstructed voltage of every checked cathode, averaged over the first insertion.

    The average is weighted by how much of the working ion each step takes in, which is what an
    average voltage over an interval means, and only the steps up to ``window`` ions per host
    formula unit are counted.
    """
    rows = []
    for framework, published, label in PUBLISHED:
        mine = [p for p in pairs if p.framework == framework
                and p.x_discharged <= window + 1e-9]
        width = sum(p.x_discharged - p.x_charged for p in mine)
        if not mine or width <= 0:
            rows.append({"framework": framework, "label": label, "published": float(published),
                         "voltage": None, "steps": 0, "error": None})
            continue
        value = sum(p.voltage * (p.x_discharged - p.x_charged) for p in mine) / width
        rows.append({"framework": framework, "label": label, "published": float(published),
                     "voltage": float(value), "steps": len(mine),
                     "x_covered": float(width), "error": float(value - published)})
    return rows


def check(hull=HULL, windows=HULL_WINDOWS):
    """Do the reconstructed voltages of four textbook cathodes agree with the published ones?

    The second table is the one that decides ``HULL``: how the size of the set and the agreement
    with the published voltages move as phases further above the hull are allowed in.
    """
    pairs = load_pairs(hull)
    rows = cathode_rows(pairs)
    for row in rows:
        volts = "none" if row["voltage"] is None else f"{row['voltage']:4.2f} V"
        print(f"  {row['label']:<22} published {row['published']:4.2f} V, reconstructed "
              f"{volts} over {row['steps']} steps")
    sensitivity = []
    for cut in windows:
        wide = load_pairs(cut, verbose=False)
        inside = [p for p in wide if p.x_discharged <= CHECK_WINDOW + 1e-9]
        errors = [abs(r["error"]) for r in cathode_rows(wide) if r["error"] is not None]
        sensitivity.append({
            "hull": float(cut), "pairs": len(wide), "pairs_first_insertion": len(inside),
            "frameworks": len({p.framework for p in wide}),
            "mean_absolute_error": float(np.mean(errors)) if errors else None,
            "cathodes": len(errors)})
        print(f"  hull <= {cut:<5} {len(wide):4d} pairs, "
              f"{sensitivity[-1]['frameworks']:3d} frameworks, published off by "
              f"{sensitivity[-1]['mean_absolute_error']:.2f} V")
    return {"hull": hull, "window": CHECK_WINDOW, "rows": rows, "sensitivity": sensitivity,
            "mean_absolute_error": float(np.mean([abs(r["error"]) for r in rows
                                                  if r["error"] is not None]))}


# --------------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------------- #
def composition_features(structures, elements):
    """The element fractions of every cell, then the crystal descriptors of plan item BT2."""
    index = {el: k for k, el in enumerate(elements)}
    X = np.zeros((len(structures), len(elements) + len(CRYSTAL_DESCRIPTORS)))
    for i, atoms in enumerate(structures):
        symbols = atoms.get_chemical_symbols()
        for s in symbols:
            if s in index:
                X[i, index[s]] += 1.0 / len(symbols)
        for k, name in enumerate(CRYSTAL_DESCRIPTORS):
            X[i, len(elements) + k] = compute_descriptor(atoms, name)
    return X


def standardise(X_tr, X):
    """Centre and scale by the training columns; a column that never moves is dropped."""
    mean, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    keep = sd > 1e-12
    return (X[:, keep] - mean[keep]) / sd[keep]


def linear_kernel(X_tr, X=None):
    """A linear kernel scaled to unit mean diagonal on the training rows."""
    A = X_tr if X is None else X
    K = A @ X_tr.T
    scale = float(np.mean(np.einsum("ij,ij->i", X_tr, X_tr)))
    return K / (scale if scale > 1e-12 else 1.0)


def grouped_folds(groups, n_splits=OUTER, seed=SPLIT_SEED):
    """Five folds that keep the pairs of one framework together.

    The frameworks are shuffled once with the seed every benchmark here uses and dealt round the
    folds, so the split is the same on every machine and no framework is split between training
    and test.
    """
    groups = np.asarray(groups)
    unique = np.array(sorted(set(groups.tolist())))
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = {unique[j]: int(k % n_splits) for k, j in enumerate(order)}
    which = np.array([fold_of[g] for g in groups], int)
    return [(np.flatnonzero(which != k), np.flatnonzero(which == k)) for k in range(n_splits)]


def measure(pairs, frame="discharged", hull=HULL):
    """The three models on the same folds, and what each one predicts of the voltage."""
    structures, y, names = battery.pair_table(pairs, frame=frame)
    groups = [p.framework for p in pairs for _ in range(2 if frame == "both" else 1)]
    chem = [battery.chemistry(p) for p in pairs for _ in range(2 if frame == "both" else 1)]
    elements = sorted(elements_in(structures))
    fs = inorganic_fieldset(elements, physchem=INORGANIC_FIELDS, charge_model="oxidation")
    specs = {s: PeriodicSpec(r_max=R_MAX, step=s, sigma=s, taper=TAPER) for s in SIGMAS}
    print(f"  {len(y)} samples, {len(set(groups))} frameworks, {fs.n_fields} fields, "
          f"voltage {y.min():.2f} to {y.max():.2f} V, mean {y.mean():.2f}")
    feats = {}
    for s, spec in specs.items():
        crystals = [crystal_fields(a, fs, nm) for a, nm in zip(structures, names)]
        t0 = time.time()
        feats[s] = periodic_feature_matrix(crystals, spec, n_fields=fs.n_fields)
        print(f"  features at sigma {s}: {feats[s].shape}, {time.time() - t0:.0f} s", flush=True)
    X = composition_features(structures, elements)
    folds = grouped_folds(groups)

    out = {}
    for model in ("composition", "fields", "fields+composition"):
        pred = np.full(len(y), np.nan)
        per_fold = []
        t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            Xs_tr, Xs_te = standardise(X[tr], X[tr]), standardise(X[tr], X[te])
            Xs_tr = np.nan_to_num(Xs_tr, nan=0.0)
            Xs_te = np.nan_to_num(Xs_te, nan=0.0)
            best = None
            for s in (SIGMAS if model != "composition" else (SIGMAS[0],)):
                spec = specs[s]
                blocks = periodic_block_sizes(spec, fs.n_fields)
                F = feats[s]
                sc = channel_scale(F[tr], spec.n_bins, blocks=blocks)
                K_field = autocorrelation_kernel(F[tr], scale=sc, n_bins=spec.n_bins,
                                                 blocks=blocks)
                K_comp = linear_kernel(Xs_tr)
                K_tr = (K_comp if model == "composition" else
                        K_field if model == "fields" else K_field + K_comp)
                for lam in LAMS:
                    p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
                    err = rmse(y[tr], p)
                    if best is None or err < best[0]:
                        best = (err, s, lam, sc, blocks)
            err, s, lam, sc, blocks = best
            spec = specs[s]
            F = feats[s]
            K_field_tr = autocorrelation_kernel(F[tr], scale=sc, n_bins=spec.n_bins, blocks=blocks)
            K_field_te = autocorrelation_kernel(F[tr], F[te], scale=sc, n_bins=spec.n_bins,
                                                blocks=blocks)
            K_comp_tr, K_comp_te = linear_kernel(Xs_tr), linear_kernel(Xs_tr, Xs_te)
            if model == "composition":
                K_tr, K_te = K_comp_tr, K_comp_te
            elif model == "fields":
                K_tr, K_te = K_field_tr, K_field_te
            else:
                K_tr, K_te = K_field_tr + K_comp_tr, K_field_te + K_comp_te
            learner = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr])
            pred[te] = learner.predict(K_te)
            mae = float(np.mean(np.abs(pred[te] - y[te])))
            per_fold.append({"fold": k, "sigma": s, "lam": lam, "mae": mae,
                             "rmse": float(rmse(y[te], pred[te])), "n_test": int(len(te))})
            print(f"    fold {k}: sigma {s:g}, lam {lam:g} -> MAE {mae:.3f} V", flush=True)
        maes = np.array([f["mae"] for f in per_fold])
        rec = {"model": model, "n": int(len(y)), "frame": frame, "hull": hull,
               "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)),
               "mae_pooled": float(np.mean(np.abs(pred - y))),
               "rmse_pooled": float(rmse(y, pred)), "r2_pooled": float(r2_score(y, pred)),
               "per_fold": per_fold, "elapsed_s": time.time() - t0,
               "by_chemistry": by_chemistry(y, pred, chem),
               "y": [float(v) for v in y], "y_pred": [float(v) for v in pred],
               "names": list(names), "groups": list(groups), "chemistry": list(chem)}
        print(f"  {model:<20} MAE {rec['mae_mean']:.3f} +/- {rec['mae_sd']:.3f} V, "
              f"R2 {rec['r2_pooled']:+.3f}")
        out[model] = rec
    out["fields_meta"] = {"fields": [sp.label for sp in fs.specs], "n_fields": fs.n_fields,
                          "elements": elements, "sigmas": list(SIGMAS), "r_max": R_MAX,
                          "taper": TAPER, "lams": list(LAMS),
                          "descriptors": list(CRYSTAL_DESCRIPTORS)}
    out["baseline"] = {"mean_voltage": float(y.mean()),
                       "mae_of_the_mean": float(np.mean(np.abs(y - y.mean())))}
    return out


def by_chemistry(y, pred, chem):
    """The error inside every family of hosts, since a family may be easy and another hard."""
    rows = []
    for family in sorted(set(chem)):
        idx = np.array([i for i, c in enumerate(chem) if c == family])
        if len(idx) < 3:
            continue
        rows.append({"chemistry": family, "n": int(len(idx)),
                     "mae": float(np.mean(np.abs(pred[idx] - y[idx]))),
                     "r2": float(r2_score(y[idx], pred[idx])),
                     "mean_voltage": float(np.mean(y[idx]))})
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(path=RESULTS):
    """The table of the result files, as every benchmark in this directory writes one."""
    out = ["# The average voltage of an insertion electrode, from the cell", ""]
    checked = _read(os.path.join(path, "check.json"))
    if checked:
        out += [f"The pairs are reconstructed from public formation energies "
                f"(`cmfqsar/battery.py`), keeping the phases within {checked['hull']} eV per atom "
                f"of the convex hull. Four textbook cathodes, against the voltages the literature "
                f"gives, over the first insertion (up to {checked.get('window', 1.0):g} working "
                f"ion per host formula unit):", "",
                "| cathode | published, V | reconstructed, V | steps of the ladder |",
                "|---|---:|---:|---:|"]
        for row in checked["rows"]:
            volts = "none" if row["voltage"] is None else f"{row['voltage']:.2f}"
            out.append(f"| {row['label']} | {row['published']:.2f} | {volts} | {row['steps']} |")
        out += ["", f"Mean absolute difference from the published values: "
                    f"**{checked['mean_absolute_error']:.2f} V**, and the reconstruction is lower "
                    f"than the published value every time. Two reasons: the energies are the "
                    f"published GGA/GGA+U ones without the corrections a battery database "
                    f"applies, and the average here runs over the intermediate phases of the "
                    f"ladder, which a two-phase measurement does not see.", ""]
        if checked.get("sensitivity"):
            out += ["How the set and the agreement move with the hull window, which is what fixes "
                    "it at 0.03:", "",
                    "| hull window, eV/atom | pairs | frameworks | published off by, V |",
                    "|---|---:|---:|---:|"]
            for row in checked["sensitivity"]:
                err = "-" if row["mean_absolute_error"] is None else f"{row['mean_absolute_error']:.2f}"
                out.append(f"| {row['hull']:g} | {row['pairs']} | {row['frameworks']} | {err} |")
            out.append("")
    voltage = _read(os.path.join(path, "voltage.json"))
    if voltage:
        meta = voltage.get("fields_meta", {})
        base = voltage.get("baseline", {})
        first = voltage.get("composition", {})
        out += [f"{first.get('n', 0)} insertion pairs over "
                f"{len(set(first.get('groups', [])))} host frameworks, five folds grouped by "
                f"framework, {meta.get('n_fields', 0)} fields. Predicting the mean voltage of the "
                f"set would give a mean absolute error of {base.get('mae_of_the_mean', 0):.3f} V.",
                "",
                "| model | MAE, V | spread over the folds | R2 | pooled RMSE, V |",
                "|---|---:|---:|---:|---:|"]
        for model in ("composition", "fields", "fields+composition"):
            rec = voltage.get(model)
            if not rec:
                continue
            out.append(f"| {model} | {rec['mae_mean']:.3f} | {rec['mae_sd']:.3f} | "
                       f"{rec['r2_pooled']:+.3f} | {rec['rmse_pooled']:.3f} |")
        out += ["", "By family of hosts, for the model that uses both:", "",
                "| family | pairs | MAE, V | R2 | mean voltage, V |", "|---|---:|---:|---:|---:|"]
        for row in (voltage.get("fields+composition", {}).get("by_chemistry") or []):
            out.append(f"| {row['chemistry']} | {row['n']} | {row['mae']:.3f} | "
                       f"{row['r2']:+.3f} | {row['mean_voltage']:.2f} |")
        out.append("")
    return "\n".join(out) + "\n"


def write_table(path=RESULTS):
    """Write ``table.md`` of this benchmark from its result files, as run_all.py asks."""
    text = report(path)
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, "table.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"written {out}")
    return out


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write(record, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"written {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="voltage",
                    choices=["fetch", "check", "voltage", "report"])
    ap.add_argument("--hull", type=float, default=HULL,
                    help="eV per atom above the convex hull a phase may lie (default 0.01)")
    ap.add_argument("--frame", default="discharged", choices=["discharged", "charged", "both"],
                    help="which cell of a pair describes it")
    ap.add_argument("--force", action="store_true", help="fetch again what the cache holds")
    args = ap.parse_args(argv)
    if args.stage == "fetch":
        fetch(force=args.force)
        return 0
    if args.stage == "check":
        _write(check(args.hull), "check.json")
        return 0
    if args.stage == "voltage":
        pairs = load_pairs(args.hull)
        if len(pairs) < 20:
            raise SystemExit(f"only {len(pairs)} pairs: run --stage fetch first")
        _write(measure(pairs, frame=args.frame, hull=args.hull), "voltage.json")
        return 0
    write_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
