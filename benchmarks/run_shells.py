"""
Plan item BT5: does an ensemble of sampled solvation shells beat one optimised complex?

Item BT4 builds a single lithium complex and optimises it. This benchmark asks the question item
BT5 sets: on a transport property of a real electrolyte, does the ensemble of shells that a bulk
simulation samples describe the electrolyte better than that single complex, by more than the
interval of rule 4?

The data are 62 LiPF6 electrolytes with measured conductivities (``examples/electrolyte/``,
derived from CALiSol-23) and the shells sampled for each of them.
``examples/make_electrolyte_shells.py`` is the recipe that produced them.

Two targets, both read off each electrolyte's own measurements at several temperatures:

* **log10 of the conductivity at 298.15 K**, in mS/cm. What an electrolyte is chosen for.
* **The activation energy at 298.15 K**, in kJ/mol. The quantity a solvation shell has the better
  claim on, because an ion moves by leaving one shell and entering the next.

Six models, each on the same electrolytes and the same folds
------------------------------------------------------------
Every model sees the molality, because conductivity depends on it strongly and a model that
cannot see it is not a competitor but a straw man. What the models differ in is the description
of the chemistry beside it:

===============  ===========================================================================
``conditions``   the molality alone. The control that says how much the rest adds.
``composition``  the molality and the solvent weight fractions. This is the control items BT1
                 to BT3 earned: on an average voltage the composition beat every field.
``solvent``      the free solvent molecules, a mole-fraction-weighted ensemble, through the
                 alignment-free kernel. The electrolyte without its ion, which is the model of
                 item EC7.
``complex``      one lithium complex per solvent composition, built and optimised as item BT4
                 does, in the cation frame. **This is what item BT5 has to beat.**
``shells``       the ensemble of shells sampled from the bulk simulation, in the cation frame.
``shells-free``  the same shells through the alignment-free kernel, as the control on the frame
                 that item BT4 found necessary.
===============  ===========================================================================

The field kernel and the condition kernel are mixed in equal parts, both normalised to unit
diagonal, with no weight chosen anywhere. That is deliberate. Letting the weight be chosen inside
each fold would let every model discard its field and keep the molality, so all six would land on
the same number and the item's question would go unanswered; item M10 met that trap from the other
side. The ``--mix-selected`` reading repeats the measurement with the weight chosen in the fold,
so the size of that effect is reported rather than hidden.

Two splits, and which one is the gate
-------------------------------------
``random``
    five random folds. A solvent composition appears in training and in testing at different
    molalities, so this measures interpolation along the concentration axis.
``by-composition``
    folds grouped so that no solvent composition is in both. This is the question that matters --
    describe an electrolyte whose solvent mixture was not in the training set -- and it is the
    gate. The ridge and the field width are chosen on an inner split that is grouped in the same
    way, because item M10 found that choosing them on a random inner split is what breaks a model
    meant to generalise.

Stages
------
    python benchmarks/run_shells.py --stage data      # the series and the shell statistics
    python benchmarks/run_shells.py --stage gate      # the six models, both targets, both splits
    python benchmarks/run_shells.py --stage cofield   # what the shell co-field points at
    python benchmarks/run_shells.py --stage report    # the table
    python benchmarks/run_shells.py                   # all of them
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger                                             # noqa: E402

from cmfqsar.cofields import cofield_volume, make_grid                       # noqa: E402
from cmfqsar.electrolyte import SOLVENTS, molar_mass, template               # noqa: E402
from cmfqsar.fields import (AtomTable, FieldSet, build_fields,               # noqa: E402
                            combine_members)
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,   # noqa: E402
                               block_sizes, channel_scale, feature_matrix)
from cmfqsar.model import CMFModel, _Learner, r2_score, rmse                 # noqa: E402
from cmfqsar.shells import (DONOR_ELEMENTS, ensemble_from_molecules,         # noqa: E402
                            shell_cutoff)
from cmfqsar import solvation as SV                                          # noqa: E402

import kernel_cache as KC                                                    # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "electrolyte")
RESULTS = os.path.join(HERE, "results", "shells")
WORK = os.path.join(HERE, "cache", "shells")

SEED = 0
OUTER, INNER = 5, 10
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
ALPHAS = (0.3, 0.5, 1.0, 2.0)
MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
ION = "Li"
#: molecules in the complex item BT4's model is built on, the coordination number it measured
COMPLEX_SIZE = 4
#: starting arrangements tried per complex, each optimised, the lowest kept, as item BT4 does
STARTS = 5
#: the field kernel and the condition kernel are mixed in these parts, with nothing selected
MIX = 0.5

TARGETS = {"log10_k_298": ("log10 conductivity at 298 K", ""),
           "ea_298": ("activation energy at 298 K", "kJ/mol")}
MODELS = ("conditions", "composition", "solvent", "complex", "complex-free", "shells",
          "shells-free")


# --------------------------------------------------------------------------------- the data


def load_series(limit: int = 0) -> List[Dict[str, object]]:
    """The electrolytes that have both a target and a shell file."""
    path = os.path.join(DATA, "formulations.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is not here; run examples/make_electrolyte_shells.py --select")
    diag: Dict[str, Dict[str, object]] = {}
    jpath = os.path.join(DATA, "shells.json")
    if os.path.exists(jpath):
        with open(jpath, encoding="utf-8") as fh:
            diag = {str(d["id"]): d for d in json.load(fh)["runs"]}
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sdf = os.path.join(DATA, "shells", f"{row['id']}.sdf")
            if not os.path.exists(sdf):
                continue
            d = diag.get(row["id"], {})
            if "failed" in d:
                continue
            solvents = {k[2:]: float(v) for k, v in row.items()
                        if k.startswith("w_") and float(v) > 0}
            out.append({"id": row["id"], "salt": row["salt"],
                        "molality": float(row["molality"]),
                        "solvents": solvents, "sdf": sdf,
                        "composition": "+".join(f"{s}{v:.2f}" for s, v in sorted(
                            solvents.items())),
                        "log10_k_298": float(row["log10_k_298"]),
                        "ea_298": float(row["ea_298"]),
                        "diagnostics": d})
    out.sort(key=lambda r: str(r["id"]))
    return out[:limit] if limit else out


def mole_fractions(solvents: Dict[str, float]) -> Dict[str, float]:
    """Weight fractions turned into mole fractions, which is what a shell is made of."""
    share = {s: w / molar_mass(SOLVENTS[s][0]) for s, w in solvents.items()}
    total = sum(share.values())
    return {s: v / total for s, v in share.items()}


def data_stage(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """What the series is: its spread, its solvents, and how well the shells are sampled."""
    print(f"  {len(rows)} electrolytes with a target and a shell file")
    comps = sorted({str(r["composition"]) for r in rows})
    solv: Dict[str, int] = {}
    for r in rows:
        for s in r["solvents"]:
            solv[s] = solv.get(s, 0) + 1
    print(f"  {len(comps)} solvent compositions over {len(solv)} solvents: "
          + ", ".join(f"{s} ({n})" for s, n in sorted(solv.items(), key=lambda t: -t[1])))
    out: Dict[str, object] = {"n": len(rows), "compositions": len(comps),
                              "solvents": solv, "ids": [r["id"] for r in rows]}
    for key, (label, unit) in TARGETS.items():
        y = np.array([float(r[key]) for r in rows], float)
        spread = float(np.mean(np.abs(y - np.mean(y))))
        print(f"  {label}: {y.min():.3f} to {y.max():.3f} {unit}, "
              f"predicting the mean costs {spread:.3f} {unit}")
        out[key] = {"min": float(y.min()), "max": float(y.max()),
                    "mean": float(y.mean()), "sd": float(y.std(ddof=1)),
                    "mean_model_mae": spread}
    m = np.array([float(r["molality"]) for r in rows], float)
    print(f"  molality {m.min():.3f} to {m.max():.3f} mol/kg")
    out["molality"] = {"min": float(m.min()), "max": float(m.max())}
    # How much of the series one optimised complex can tell apart. Four molecules cannot carry
    # an arbitrary mixture, so two nearby compositions round to the same shell and the BT4
    # descriptor gives them the same field. This is a property of that descriptor, not of this
    # implementation, and it belongs beside the result.
    shapes = {}
    for r in rows:
        key = tuple(sorted(complex_counts(r).items()))
        shapes.setdefault(key, set()).add(str(r["composition"]))
    shared = sum(1 for v in shapes.values() if len(v) > 1)
    print(f"  one optimised complex of {COMPLEX_SIZE} molecules gives {len(shapes)} distinct "
          f"shells for {len(comps)} compositions; {shared} of those shells are shared by more "
          f"than one composition")
    out["complex_shapes"] = {"distinct": len(shapes), "shared": shared,
                             "shapes": {" ".join(f"{s}:{n}" for s, n in k): sorted(v)
                                        for k, v in shapes.items()}}
    for key, label, fmt in (("mean_molecules", "molecules in a shell", "{:.2f}"),
                            ("mean_coordination", "donors at the ion", "{:.2f}"),
                            ("ion_pair_share", "shells that hold an anion", "{:.3f}"),
                            ("density", "density the box settled at, g/cm3", "{:.3f}"),
                            ("shells", "shells cut per electrolyte", "{:.0f}"),
                            ("block_error", "block error of the mean field", "{:.5f}"),
                            ("thinning_error", "cost of keeping 16 members", "{:.5f}")):
        vs = np.array([float(r["diagnostics"].get(key, np.nan)) for r in rows], float)
        vs = vs[np.isfinite(vs)]
        if len(vs) == 0:
            continue
        print(f"  {label:34s} " + fmt.format(float(vs.mean())) + " +/- "
              + fmt.format(float(vs.std(ddof=0))) + "  ("
              + fmt.format(float(vs.min())) + " to " + fmt.format(float(vs.max())) + ")")
        out[key] = {"mean": float(vs.mean()), "sd": float(vs.std(ddof=0)),
                    "min": float(vs.min()), "max": float(vs.max())}
    # Three correlations with the molality. The first is the physics the sampled shells are
    # supposed to see and the single optimised complex cannot: more anions in the shell as the
    # salt gets more concentrated. The other two are checks that the descriptor does not get
    # quietly worse along the same axis, which would look like the physics.
    for key, label in (("ion_pair_share", "the share of shells holding an anion"),
                       ("mean_coordination", "the donors at the ion"),
                       ("thinning_error", "the cost of keeping 16 members")):
        v = np.array([float(r["diagnostics"].get(key, np.nan)) for r in rows], float)
        keep = np.isfinite(v)
        if keep.sum() > 3 and float(np.std(v[keep])) > 0:
            c = float(np.corrcoef(m[keep], v[keep])[0, 1])
            print(f"  {label} against the molality: {c:+.3f}")
            out[f"{key}_vs_molality"] = c
    return out


# ------------------------------------------------------------------------------ the molecules


def shell_molecules(rows: Sequence[Dict[str, object]]) -> List[List[Chem.Mol]]:
    """The representative shells of every electrolyte, read back from their files."""
    out = []
    for r in rows:
        members = [m for m in Chem.SDMolSupplier(str(r["sdf"]), removeHs=False) if m is not None]
        if not members:
            raise SystemExit(f"{r['sdf']} holds no readable record")
        out.append(members)
    return out


def solvent_members(row: Dict[str, object]) -> Tuple[List[Chem.Mol], List[float]]:
    """The free solvent molecules of one electrolyte and their mole fractions."""
    x = mole_fractions(dict(row["solvents"]))
    mols = [template(SOLVENTS[s][0], s, seed=i) for i, s in enumerate(sorted(x))]
    return mols, [x[s] for s in sorted(x)]


def complex_counts(row: Dict[str, object], size: int = COMPLEX_SIZE) -> Dict[str, int]:
    """How many molecules of each solvent go into the one optimised complex.

    The mole fractions decide it, by largest remainder, so a mixture gets a complex whose shell
    has the mixture's composition as closely as four molecules can.
    """
    x = mole_fractions(dict(row["solvents"]))
    raw = {s: size * v for s, v in x.items()}
    counts = {s: int(np.floor(v)) for s, v in raw.items()}
    left = size - sum(counts.values())
    for s in sorted(raw, key=lambda s: -(raw[s] - np.floor(raw[s]))):
        if left <= 0:
            break
        counts[s] += 1
        left -= 1
    return {s: n for s, n in counts.items() if n > 0}


def optimised_complexes(rows: Sequence[Dict[str, object]]) -> Dict[str, Chem.Mol]:
    """One optimised lithium complex per solvent composition, as item BT4 builds them.

    Cached as an SDF, because this is the only part of the benchmark that calls a quantum
    program. The complex does not depend on the molality -- that is the point of the comparison,
    and it is why one complex serves every molality of a composition.
    """
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, "complexes.sdf")
    done: Dict[str, Chem.Mol] = {}
    if os.path.exists(path):
        for m in Chem.SDMolSupplier(path, removeHs=False):
            if m is not None and m.HasProp("_Name"):
                done[m.GetProp("_Name")] = m
    wanted = sorted({str(r["composition"]) for r in rows})
    todo = [c for c in wanted if c not in done]
    if todo:
        engine = SV.Engine(workdir=os.path.join(WORK, "xtb"))
        by_comp = {str(r["composition"]): r for r in rows}
        for i, comp in enumerate(todo, 1):
            row = by_comp[comp]
            counts = complex_counts(row)
            mols = []
            for s, n in sorted(counts.items()):
                mols.extend([template(SOLVENTS[s][0], s, seed=k) for k in range(n)])
            best = None
            for start in range(STARTS):
                try:
                    cx = SV.place_shell(mols, ion=ION, seed=start)
                    opt, energy = SV.optimise(cx.mol, name=f"{comp}_{start}", engine=engine)
                except Exception as err:         # noqa: BLE001 - one start may fail
                    print(f"    {comp} start {start}: {type(err).__name__}: {err}", flush=True)
                    continue
                if best is None or energy < best[1]:
                    best = (opt, energy)
            if best is None:
                raise SystemExit(f"no complex could be built for {comp}")
            opt, energy = best
            mol = SV.with_frame(opt, index=0, ion=ION)
            mol.SetProp("_Name", comp)
            mol.SetProp("complex_counts", " ".join(f"{s}:{n}" for s, n in sorted(counts.items())))
            mol.SetProp("energy_ev", f"{energy:.6f}")
            done[comp] = mol
            print(f"    [{i}/{len(todo)}] {comp}: "
                  f"{'+'.join(f'{s}x{n}' for s, n in sorted(counts.items()))}, "
                  f"{energy:.3f} eV", flush=True)
        writer = Chem.SDWriter(path)
        for comp in sorted(done):
            writer.write(done[comp])
        writer.close()
    return done


# ------------------------------------------------------------------------------- the kernels


def _ensemble_of_free_solvents(row: Dict[str, object], fieldset: FieldSet):
    mols, weights = solvent_members(row)
    members = [build_fields(m, fieldset, name=f"{row['id']}_{i}") for i, m in enumerate(mols)]
    sizes = [m.GetNumAtoms() for m in mols]
    maps, offset = [], 0
    first = {}
    for i, m in enumerate(mols):
        if i == 0:
            maps.append(list(range(sizes[0])))
            first[Chem.MolToSmiles(m)] = 0
        else:
            key = Chem.MolToSmiles(m)
            base = first.get(key)
            maps.append(list(range(base, base + sizes[i])) if base is not None
                        else [-1] * sizes[i])
        offset += sizes[i]
    return combine_members(members, weights, kind="solvents", atom_maps=maps,
                           name=str(row["id"]))


def field_tables(rows: Sequence[Dict[str, object]], model: str,
                 shells: Sequence[Sequence[Chem.Mol]],
                 complexes: Dict[str, Chem.Mol]) -> Dict[float, object]:
    """One prepared kernel (or feature matrix, for an alignment-free model) per field width."""
    out: Dict[float, object] = {}
    spec = AutocorrelationSpec()
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger")
        if model == "shells":
            mf = [ensemble_from_molecules(members, fs, name=str(r["id"]))
                  for r, members in zip(rows, shells)]
        elif model == "shells-free":
            mf = [ensemble_from_molecules(members, fs, name=str(r["id"]))
                  for r, members in zip(rows, shells)]
        elif model in ("complex", "complex-free"):
            mf = [build_fields(complexes[str(r["composition"])], fs, name=str(r["id"]))
                  for r in rows]
        elif model == "solvent":
            mf = [_ensemble_of_free_solvents(r, fs) for r in rows]
        else:
            raise ValueError(f"{model} has no field")
        if model in ("shells-free", "complex-free", "solvent"):
            F = feature_matrix(mf, spec, fieldset=fs)
            out[alpha] = (F, spec, block_sizes(spec, fs.n_fields))
        else:
            table = AtomTable.from_molecules(mf)
            cmf = CMFModel(fieldset=fs, method="krr", normalize=True)
            out[alpha] = _unit_diagonal(KC.joint_training_kernel(cmf, table))
    return out


def _unit_diagonal(K: np.ndarray) -> np.ndarray:
    """The same kernel with a unit diagonal, so that two kernels can be mixed in equal parts.

    The joint overlap kernel sums one normalised block per field that carries weight, so its
    diagonal is the number of such fields -- four for these shells, because an ether and a
    carbonate have no hydrogen-bond donor. The alignment-free kernel is already cosine
    normalised. Dividing by the diagonal puts both on the same scale, which is what makes the
    fixed mixture below mean the same thing for every model.
    """
    d = np.sqrt(np.clip(np.diag(np.asarray(K, float)), 1e-12, None))
    return np.asarray(K, float) / np.outer(d, d)


def _kernel_at(entry, tr, te=None):
    """Training and test blocks of one prepared kernel or feature matrix."""
    if isinstance(entry, tuple):
        F, spec, blocks = entry
        sc = channel_scale(F[tr], spec.n_bins, blocks=blocks)
        K_tr = autocorrelation_kernel(F[tr], scale=sc, n_bins=spec.n_bins, blocks=blocks)
        if te is None:
            return K_tr, None
        return K_tr, autocorrelation_kernel(F[tr], F[te], scale=sc, n_bins=spec.n_bins,
                                            blocks=blocks)
    K = entry
    if te is None:
        return K[np.ix_(tr, tr)], None
    return K[np.ix_(tr, tr)], K[np.ix_(te, tr)]


def condition_kernel(Z: np.ndarray, tr: np.ndarray, te: Optional[np.ndarray] = None):
    """A Gaussian kernel on the descriptors, standardised and scaled on the training fold only.

    The bandwidth is the median squared distance inside the training fold, the usual median
    heuristic, so nothing about it is chosen on the data being predicted.
    """
    Z = np.asarray(Z, float)
    mu, sd = Z[tr].mean(axis=0), Z[tr].std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A = (Z[tr] - mu) / sd
    d2 = np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=-1)
    med = float(np.median(d2[np.triu_indices(len(A), k=1)])) if len(A) > 1 else 1.0
    gamma = 1.0 / max(med, 1e-9)
    K_tr = np.exp(-gamma * d2)
    if te is None:
        return K_tr, None
    B = (Z[te] - mu) / sd
    d2te = np.sum((B[:, None, :] - A[None, :, :]) ** 2, axis=-1)
    return K_tr, np.exp(-gamma * d2te)


def descriptors(rows: Sequence[Dict[str, object]], model: str) -> np.ndarray:
    """The side information a model is given: the molality, and for one model the composition."""
    m = np.array([[float(r["molality"])] for r in rows], float)
    if model != "composition":
        return m
    names = sorted({s for r in rows for s in r["solvents"]})
    W = np.array([[float(r["solvents"].get(s, 0.0)) for s in names] for r in rows], float)
    return np.hstack([m, W])


# --------------------------------------------------------------------------------- the folds


def random_folds(n: int, k: int = OUTER, seed: int = SEED):
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    return [(np.setdiff1d(np.arange(n), g), g) for g in np.array_split(order, k)]


def grouped_folds(labels: Sequence[str], k: int = OUTER, seed: int = SEED):
    """Folds in which no group is split, so a test electrolyte's solvent mixture is unseen.

    Groups are dealt to the folds largest first, each to the fold that holds fewest samples, so
    the folds come out as even as groups of unequal size allow.
    """
    labels = [str(v) for v in labels]
    groups: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    order = sorted(groups, key=lambda g: (-len(groups[g]), g))
    rng = np.random.RandomState(seed)
    order = [order[i] for i in rng.permutation(len(order))]
    order.sort(key=lambda g: -len(groups[g]))
    buckets: List[List[int]] = [[] for _ in range(k)]
    for g in order:
        t = int(np.argmin([len(b) for b in buckets]))
        buckets[t].extend(groups[g])
    n = len(labels)
    return [(np.setdiff1d(np.arange(n), np.asarray(b, int)), np.asarray(b, int))
            for b in buckets if b]


def inner_predict(K: np.ndarray, y: np.ndarray, lam: float, labels: Optional[Sequence[str]],
                  k: int = INNER, seed: int = SEED) -> np.ndarray:
    """Cross-validated predictions inside a training fold, split the same way as the outer one.

    Item M10 found that choosing the ridge on a random inner split is what breaks a model meant
    to generalise, so when the outer split is grouped the inner one is grouped too.
    """
    n = len(y)
    design = (grouped_folds(labels, k=min(k, len(set(labels))), seed=seed) if labels is not None
              else random_folds(n, k=min(k, n), seed=seed))
    out = np.full(n, np.nan)
    for tr, te in design:
        learner = _Learner("krr", {"lam": lam}, True).fit(K[np.ix_(tr, tr)], y[tr])
        out[te] = learner.predict(K[np.ix_(te, tr)])
    return out


def nested(entry, Z: np.ndarray, y: np.ndarray, design, labels: Optional[Sequence[str]],
           mixes: Sequence[float], label: str = "", unit: str = "") -> Dict[str, object]:
    """Out-of-fold predictions, the ridge and the field width chosen inside each fold."""
    pred = np.full(len(y), np.nan)
    per_fold = []
    t0 = time.time()
    for k, (tr, te) in enumerate(design):
        inner_labels = None if labels is None else [labels[i] for i in tr]
        K_cond_tr, K_cond_te = condition_kernel(Z, tr, te)
        best = None
        alphas = list(entry.keys()) if entry is not None else [0.0]
        # a model with no field has nothing to mix: its kernel is the condition kernel alone
        use = (1.0,) if entry is None else tuple(mixes)
        for alpha in alphas:
            K_f_tr = None if entry is None else _kernel_at(entry[alpha], tr)[0]
            for w in use:
                K_tr = (K_cond_tr if entry is None else (1.0 - w) * K_f_tr + w * K_cond_tr)
                for lam in LAMS:
                    p = inner_predict(K_tr, y[tr], lam, inner_labels)
                    ok = np.isfinite(p)
                    err = rmse(y[tr][ok], p[ok])
                    if best is None or err < best[0]:
                        best = (err, alpha, lam, w)
        _e, alpha, lam, w = best
        if entry is None:
            K_tr, K_te = K_cond_tr, K_cond_te
        else:
            K_f_tr, K_f_te = _kernel_at(entry[alpha], tr, te)
            K_tr = (1.0 - w) * K_f_tr + w * K_cond_tr
            K_te = (1.0 - w) * K_f_te + w * K_cond_te
        learner = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr])
        pred[te] = learner.predict(K_te)
        per_fold.append({"fold": k, "alpha": alpha, "lam": lam, "mix": w,
                         "n_test": int(len(te)),
                         "mae": float(np.mean(np.abs(pred[te] - y[te])))})
    maes = np.array([f["mae"] for f in per_fold])
    rec = {"n": int(len(y)), "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)),
           "mae_pooled": float(np.mean(np.abs(pred - y))),
           "r2_pooled": float(r2_score(y, pred)), "per_fold": per_fold,
           "abs_errors": [float(v) for v in np.abs(pred - y)],
           "elapsed_s": time.time() - t0}
    print(f"    {label:<24} MAE {rec['mae_mean']:.3f} +/- {rec['mae_sd']:.3f} {unit}, "
          f"R2 {rec['r2_pooled']:+.3f}  ({rec['elapsed_s']:.0f} s)", flush=True)
    return rec


def paired_interval(a, b, seed=SEED, draws=2000):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(d)), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ----------------------------------------------------------------------------------- stages


def gate_stage(rows: List[Dict[str, object]], mix_selected: bool = False) -> Dict[str, object]:
    """The six models, both targets, both splits, with the paired intervals of rule 4."""
    shells = shell_molecules(rows)
    print("  building the optimised complexes of item BT4")
    complexes = optimised_complexes(rows)
    prepared: Dict[str, Dict[float, object]] = {}
    for model in MODELS:
        if model in ("conditions", "composition"):
            continue
        t0 = time.time()
        prepared[model] = field_tables(rows, model, shells, complexes)
        print(f"  {model} kernels in {time.time() - t0:.0f} s")

    labels = [str(r["composition"]) for r in rows]
    splits = {"random": (random_folds(len(rows)), None),
              "by-composition": (grouped_folds(labels), labels)}
    mixes = MIXES if mix_selected else (MIX,)
    out: Dict[str, object] = {"mix": "selected in the fold" if mix_selected else MIX,
                             "models": {}, "comparisons": {}}
    for target, (tlabel, unit) in TARGETS.items():
        y = np.array([float(r[target]) for r in rows], float)
        for split, (design, group_labels) in splits.items():
            print(f"  {tlabel}, {split} folds "
                  f"(predicting the mean costs "
                  f"{float(np.mean(np.abs(y - y.mean()))):.3f} {unit})")
            recs = {}
            for model in MODELS:
                Z = descriptors(rows, model)
                entry = prepared.get(model)
                recs[model] = nested(entry, Z, y, design, group_labels, mixes,
                                     label=model, unit=unit)
            out["models"][f"{target}|{split}"] = recs
            # The item's gate is "the sampled shells beat the single optimised complex". That
            # has to be read in each kernel separately, because the kernel is not part of the
            # question: the overlap kernel needs the cation frame and the autocorrelation kernel
            # needs no alignment at all, and item BT4 found the alignment-free model the better
            # of the two. So "shells vs complex" below is framed against framed, and
            # "shells-free vs complex-free" is alignment-free against alignment-free.
            pairs = {"shells vs complex": ("shells", "complex"),
                     "shells-free vs complex-free": ("shells-free", "complex-free"),
                     "shells vs solvent": ("shells", "solvent"),
                     "shells-free vs solvent": ("shells-free", "solvent"),
                     "shells-free vs composition": ("shells-free", "composition"),
                     "shells vs composition": ("shells", "composition"),
                     "shells vs conditions": ("shells", "conditions"),
                     "complex vs conditions": ("complex", "conditions"),
                     "the frame on the shells": ("shells", "shells-free"),
                     "the frame on the complex": ("complex", "complex-free")}
            comp = {}
            for name, (a, b) in pairs.items():
                # a positive gain means the first model has the smaller error
                iv = paired_interval(recs[b]["abs_errors"], recs[a]["abs_errors"])
                comp[name] = iv
                mark = "clear of zero" if iv["excludes_zero"] else "includes zero"
                print(f"      {name:<30} {iv['mean']:+.4f} "
                      f"[{iv['low']:+.4f}, {iv['high']:+.4f}] {unit} ({mark})")
            out["comparisons"][f"{target}|{split}"] = comp
    return out


def cofield_stage(rows: List[Dict[str, object]], target: str = "ea_298") -> Dict[str, object]:
    """Where the co-field of the shell model has its weight, read from the ion.

    The alignment-free kernel has no co-field; the cation frame of item BT4 is what gives one.
    The reading is the same as that item's: the point of largest magnitude of each field, how far
    it is from the ion, and how far it is from the nearest donor atom. The second distance is
    needed to read the first. A point 0.7 A from the ion and a point 2.0 A from it are both
    "inside the first shell", but only the second one is near anything an atom occupies.
    """
    shells = shell_molecules(rows)
    alpha = ALPHAS[0]
    fs = FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger")
    mf = [ensemble_from_molecules(members, fs, name=str(r["id"]))
          for r, members in zip(rows, shells)]
    table = AtomTable.from_molecules(mf)
    y = np.array([float(r[target]) for r in rows], float)
    lam = 0.1
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": lam})
    model.fit(table, y, cv_folds=0)
    grid = make_grid(np.vstack([m.coords for m in mf[:12]]), margin=2.0, spacing=0.5)
    points = grid.points()
    out: Dict[str, object] = {"target": target, "alpha": alpha, "lam": lam, "spacing": 0.5}
    # Where the donor atoms of these shells actually sit. Every shell is framed on the ion, so a
    # donor's distance from the origin is its coordination distance, and the spread of those
    # distances is what "the first shell" means here.
    radius = shell_cutoff(ION)
    donors = []
    for members in shells:
        for m in members:
            conf = m.GetConformer()
            for a in m.GetAtoms():
                if a.GetIdx() == 0 or a.GetSymbol() not in DONOR_ELEMENTS:
                    continue
                pos = conf.GetAtomPosition(a.GetIdx())
                xyz = np.array([pos.x, pos.y, pos.z], float)
                if np.linalg.norm(xyz) <= radius:
                    donors.append(xyz)
    D = np.asarray(donors, float)
    out["first_shell_donors"] = int(len(D))
    out["donor_radius"] = {"mean": float(np.mean(np.linalg.norm(D, axis=1))),
                           "sd": float(np.std(np.linalg.norm(D, axis=1)))}
    print(f"    the donors of these shells sit {out['donor_radius']['mean']:.2f} +/- "
          f"{out['donor_radius']['sd']:.2f} A from the ion ({len(D)} of them)", flush=True)
    rows_out = []
    for f, spec in enumerate(fs.specs):
        flat = np.asarray(cofield_volume(model, f, grid), float).ravel()
        k = int(np.argmax(np.abs(flat)))
        xyz = points[k]
        # the ion is at the origin of every framed shell, so the distance to it is the norm
        d_ion = float(np.linalg.norm(xyz))
        d_donor = float(np.min(np.linalg.norm(D - xyz, axis=1)))
        rows_out.append({"field": spec.label, "peak": float(flat[k]),
                         "distance_from_ion": d_ion,
                         "distance_from_donor": d_donor,
                         "point": [float(v) for v in xyz]})
        print(f"    {spec.label:<16} largest magnitude {flat[k]:+.4f} at {d_ion:.2f} A "
              f"from the ion and {d_donor:.2f} A from the nearest donor", flush=True)
    out["fields"] = rows_out
    return out


def write_table(path: Optional[str] = None) -> str:
    """The table of every stage that has run, read back from its JSON."""
    base = path or RESULTS
    os.makedirs(base, exist_ok=True)
    data = _read(base, "data")
    gate = _read(base, "gate")
    selected = _read(base, "gate_mix_selected")
    cofield = _read(base, "cofield")
    path = os.path.join(base, "table.md")
    lines = ["# Item BT5: sampled solvation shells against one optimised complex", ""]
    if data:
        lines += [f"{data['n']} LiPF6 electrolytes over {data['compositions']} solvent "
                  f"compositions, with measured conductivities from CALiSol-23 and shells "
                  f"sampled from one bulk simulation each.", ""]
        lines += ["| What | Value |", "|---|---|"]
        for key, label, fmt in (("mean_molecules", "Molecules in a shell", "{:.2f}"),
                                ("mean_coordination", "Donor atoms at the ion", "{:.2f}"),
                                ("ion_pair_share", "Share of shells holding an anion", "{:.3f}"),
                                ("density", "Density the boxes settled at, g/cm3", "{:.3f}"),
                                ("block_error", "Block error of the mean field", "{:.5f}"),
                                ("thinning_error", "Cost of keeping 16 members", "{:.5f}")):
            if key in data:
                d = data[key]
                lines.append(f"| {label} | " + fmt.format(d["mean"]) + " +/- "
                             + fmt.format(d["sd"]) + " |")
        if "complex_shapes" in data:
            d = data["complex_shapes"]
            lines.append(f"| Distinct shells one optimised complex gives | {d['distinct']} for "
                         f"{data['compositions']} compositions |")
        for key, label in (("ion_pair_share_vs_molality",
                            "Anion in the shell against molality, correlation"),
                           ("mean_coordination_vs_molality",
                            "Donors at the ion against molality, correlation"),
                           ("thinning_error_vs_molality",
                            "Cost of 16 members against molality, correlation")):
            if key in data:
                lines.append(f"| {label} | {data[key]:+.3f} |")
        lines.append("")
    if gate:
        for key, recs in gate.get("models", {}).items():
            target, split = key.split("|")
            label, unit = TARGETS[target]
            lines += [f"## {label} ({unit or 'log10 mS/cm'}), {split} folds", "",
                      "| Model | Mean absolute error | R2 |", "|---|---|---|"]
            for model in MODELS:
                r = recs.get(model)
                if r is None:
                    continue
                lines.append(f"| {model} | {r['mae_mean']:.3f} +/- {r['mae_sd']:.3f} | "
                             f"{r['r2_pooled']:+.3f} |")
            lines.append("")
            comp = gate.get("comparisons", {}).get(key, {})
            if comp:
                lines += ["| Comparison | Gain of the first model | 95 per cent interval |",
                          "|---|---|---|"]
                for name, iv in comp.items():
                    lines.append(f"| {name} | {iv['mean']:+.4f} | "
                                 f"[{iv['low']:+.4f}, {iv['high']:+.4f}] |")
                lines.append("")
    if selected.get("comparisons"):
        lines += ["## The same comparison with the mixture chosen inside each fold", "",
                  "Every model may now keep as much or as little of its field as the inner "
                  "split prefers. A model that drops its field altogether lands on the "
                  "`conditions` model, so this reading says how much of each result above "
                  "survives being allowed to do that.", "",
                  "| Target and split | Comparison | Gain of the first model | "
                  "95 per cent interval |", "|---|---|---|---|"]
        for key, comp in selected["comparisons"].items():
            target, split = key.split("|")
            for name, iv in comp.items():
                lines.append(f"| {TARGETS[target][0]}, {split} | {name} | {iv['mean']:+.4f} | "
                             f"[{iv['low']:+.4f}, {iv['high']:+.4f}] |")
        lines.append("")
    if cofield and cofield.get("fields"):
        lines += ["## The co-field of the shell model, read from the ion", ""]
        if "donor_radius" in cofield:
            lines += [f"The donor atoms of these shells sit "
                      f"{cofield['donor_radius']['mean']:.2f} +/- "
                      f"{cofield['donor_radius']['sd']:.2f} A from the ion, which is what the "
                      f"last column is to be read against.", ""]
        lines += ["| Field | Largest magnitude | Distance from the ion, A | "
                  "Distance from the nearest donor atom, A |", "|---|---|---|---|"]
        for r in cofield["fields"]:
            lines.append(f"| {r['field']} | {r['peak']:+.4f} | "
                         f"{r['distance_from_ion']:.2f} | "
                         + (f"{r['distance_from_donor']:.2f} |"
                            if "distance_from_donor" in r else "not measured |"))
        lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote {path}")
    return path


def _read(base: str, name: str) -> Dict[str, object]:
    p = os.path.join(base, f"{name}.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def save(name: str, payload: Dict[str, object]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"  wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stage", choices=("data", "gate", "cofield", "report", "all"),
                    default="all")
    ap.add_argument("--limit", type=int, default=0, help="use at most this many electrolytes")
    ap.add_argument("--mix-selected", action="store_true",
                    help="choose the field-to-condition mixture inside each fold as well")
    args = ap.parse_args()

    rows = load_series(args.limit)
    if not rows:
        raise SystemExit("no electrolyte has both a target and a shell file")
    stages = ("data", "gate", "cofield", "report") if args.stage == "all" else (args.stage,)
    out: Dict[str, Dict[str, object]] = {}
    for stage in stages:
        print(f"[{stage}]", flush=True)
        if stage == "data":
            out["data"] = data_stage(rows)
            save("data", out["data"])
        elif stage == "gate":
            name = "gate_mix_selected" if args.mix_selected else "gate"
            out["gate"] = gate_stage(rows, mix_selected=args.mix_selected)
            save(name, out["gate"])
        elif stage == "cofield":
            out["cofield"] = cofield_stage(rows)
            save("cofield", out["cofield"])
        else:
            write_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
