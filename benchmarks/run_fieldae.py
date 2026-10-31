#!/usr/bin/env python
"""
Can pseudo-atoms reconstruct a molecular field better than kernel PCA? (plan item E11)

Kernel PCA on the CMF kernel is the linear autoencoder of the field space, and it reconstructs a
field as a weighted sum of training fields. That sum can be drawn, but it is not a set of atoms:
nothing can be sampled from it and no design step can take it as a target. Item E11 replaces the
linear decoder with a network whose output *is* a set of atoms -- a fixed number of pseudo-atoms,
each with a position and one weight per field -- and trains it on the closed-form distance
between the two fields in the field Hilbert space,

    ||f - f'||^2 = k(f, f) - 2 k(f, f') + k(f', f'),

so no grid enters the training (rule 3). The construction is in `cmfqsar.fieldae`.

The item makes the first gate a stopping rule: if the nonlinear decoder does not reconstruct
better than kernel PCA at the same number of latent dimensions, there is no nonlinearity to use
and the item closes. This script measures that, and the second gate beside it.

The data
--------
COMPAS-1D: the 8,678 cata-condensed polybenzenoid hydrocarbons optimised at B3LYP-D3BJ/def2-SVP,
in the frame given by rule (`cmfqsar.pbh.canonical_frame`). The item asks for a set that is large
and shares a frame, because kernel PCA on thirty molecules has as many components as it wants and
any network is underdetermined. The fields, the attenuations, the split and the field scaling are
the model of record of `benchmarks/run_compas.py`: field set ABCD (steric, GFN2-xTB charge,
two atom-class indicator fields, four ring-class indicator fields), attenuation 0.5 for the
steric field and 0.1 for the others, the random0 split of 6,942 / 868 / 868 molecules, and every
field kernel scaled to unit mean diagonal over the whole set.

What is measured
----------------
**Gate (a), the stopping rule.** The relative reconstruction error ``||f - f'|| / ||f||`` on the
868 held-out molecules, for the autoencoder and for kernel PCA with the same number of latent
dimensions. Both are fitted on the same 6,942 training molecules. The comparison is not quite
even and the unevenness is in kernel PCA's favour: it gets the training mean for free on top of
its components, and its reconstruction need not be a set of atoms.

Beside the two, a third number: the **floor**, the error that the same number of pseudo-atoms
reaches when they are fitted to one molecule directly, with no code in between
(`cmfqsar.fieldae.fit_pseudo_atoms`). It separates the two ways a decoder can fail -- the code is
too narrow, or the pseudo-atoms cannot represent the field at all -- and the gate cannot be read
without it.

**Gate (b), the applicability domain.** The reconstruction error as an estimate of how wrong a
prediction will be, against the five one-class scores of `cmfqsar.oneclass`, under the protocol
of item E5 (`cmfqsar.errorpred`: rank correlation, risk-coverage efficiency, and the error of the
half each estimate trusts against the half it doubts). The predictions are the model of record's
own, read from `benchmarks/results/compas/fields_ABCD_1D.json`, so nothing is refitted here.
One of the five, the kernel PCA distance from the principal subspace, *is* the linear
reconstruction error, which makes this gate the same question as gate (a) asked of a different
quantity.

Usage
-----
    python benchmarks/run_fieldae.py                     # every stage, then the table
    python benchmarks/run_fieldae.py --stage prepare     # fields, scales and landmark features
    python benchmarks/run_fieldae.py --stage gate_a --codes 8 --pseudo 32
    python benchmarks/run_fieldae.py --stage floor
    python benchmarks/run_fieldae.py --stage gate_b
    python benchmarks/run_fieldae.py --stage figure
    python benchmarks/run_fieldae.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402

from cmfqsar import fieldae, oneclass  # noqa: E402
from cmfqsar.errorpred import compare  # noqa: E402
from cmfqsar.fields import AtomTable  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "fieldae")
CACHE = os.path.join(HERE, "cache", "fieldae")

SET = "1D"
GROUPS = "ABCD"
SPLIT = "random0"
TARGET = "GAP_eV"
#: the attenuations the model of record chose for this field set and split (per group)
ALPHAS = {"A": 0.5, "B": 0.1, "C": 0.1, "D": 0.1}
#: landmarks for the encoder: the atoms of a few molecules, thinned (item B6)
LANDMARKS = 256
LANDMARK_MOLECULES = 24
#: the grid of gate (a): the code dimension at a fixed number of pseudo-atoms, and the number of
#: pseudo-atoms at a fixed code dimension
CODES = (2, 4, 8, 16, 32)
PSEUDO = (8, 16, 32, 64)
REF_CODE = 8
REF_PSEUDO = 32
EPOCHS = 150
BATCH = 128
LR = 1e-3
PATIENCE = 20
HIDDEN = (256, 256)
REPULSION = 1e-3
MIN_DISTANCE = 0.75
#: molecules used for the floor, and the steps of the direct fit
FLOOR_MOLECULES = 40
FLOOR_STEPS = 500
#: threads for PyTorch.  Measured on this machine, one training step of 128 molecules: 313 ms on
#: one thread, 29 ms on four, 55 ms on eight and 81 ms on all twenty-four.  The tensors of one
#: batch are small -- a distance block of 128 by 76 by 32 -- so beyond a few threads the cost of
#: starting them is larger than the arithmetic they share.
THREADS = 4


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_default)
        fh.write("\n")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def result(name: str) -> str:
    return os.path.join(RESULTS, name)


# --------------------------------------------------------------------------- #
# Stage: prepare
# --------------------------------------------------------------------------- #

def cache_path() -> str:
    return os.path.join(CACHE, f"{SET}_{GROUPS}.npz")


def prepare(force: bool = False) -> dict:
    """The fields, the field scaling, the padded tensors and the landmark features.

    The field scaling is **read from the stored kernel record** of `run_compas.py` rather than
    recomputed, so that every number here is in the same units as the model of record; the
    function checks that the closed-form self-overlap agrees with the diagonal of the stored
    kernel, which is an independent test of the whole field pipeline on 8,678 molecules.
    """
    path = cache_path()
    rec_path = result("prepared.json")
    if os.path.isfile(path) and os.path.isfile(rec_path) and not force:
        log(f"prepare: {path} is there")
        return jload(rec_path)
    import run_compas as rc

    t0 = time.time()
    ds, meta = rc.load_prepared(SET)
    fieldset = rc.make_fieldset(GROUPS, meta, alpha_by_group=ALPHAS)
    names = [s.name for s in fieldset.specs]
    log(f"{len(ds.mols)} molecules, fields {names}")
    table = rc.build_table(ds, fieldset, ring_cache={})
    log(f"{table.coords.shape[0]} atoms in {time.time() - t0:.0f} s")

    store = rc.KernelStore(SET, max_loaded=1)
    stored = store.scales(ALPHAS)
    missing = [n for n in names if n not in stored]
    if missing:
        raise SystemExit(f"the stored kernels of COMPAS-{SET} do not cover {missing}; run "
                         f"benchmarks/run_compas.py --stage kernels first")
    scale = np.array([stored[n] for n in names], float)
    geom = fieldae.geometry(fieldset, scale=scale)
    tens = fieldae.tensors(table, fieldset, geom)

    diag = store.diag(ALPHAS, np.arange(table.n_mol))
    rel = float(np.max(np.abs(tens.self_k - diag) / np.maximum(np.abs(diag), 1e-30)))
    log(f"self-overlap against the stored kernel diagonal: {rel:.2e} relative, worst of "
        f"{table.n_mol}")

    t1 = time.time()
    feats, low = fieldae.landmark_features(table, fieldset, n_landmarks=LANDMARKS,
                                          molecules=LANDMARK_MOLECULES)
    log(f"features {feats.shape} (rank {low.rank}) in {time.time() - t1:.0f} s")

    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, coords=tens.coords.astype(np.float32),
                        weights=tens.weights.astype(np.float32),
                        radii=tens.radii.astype(np.float32), self_k=tens.self_k,
                        counts=tens.counts, features=feats.astype(np.float32),
                        scale=scale, alphas=geom.alphas,
                        kinds=np.array(geom.kinds), names=np.array(names),
                        mol_names=np.array(tens.names), gap=np.asarray(meta["gap"], float),
                        n_rings=np.asarray(meta["n_rings"], int))
    rec = {"set": SET, "groups": GROUPS, "fields": names, "alphas": ALPHAS,
           "scale": scale.tolist(), "n_molecules": int(table.n_mol),
           "atoms": int(table.coords.shape[0]), "max_atoms": int(tens.max_atoms),
           "landmarks": int(low.rank), "landmark_atoms": int(low.landmarks.n_mol or 0),
           "features": list(feats.shape),
           "self_overlap_vs_stored_kernel": rel,
           "box": tens.box().tolist(), "weight_scale": tens.weight_scale().tolist(),
           "seconds": time.time() - t0, "cache": path}
    jdump(rec, rec_path)
    return rec


def load_prepared():
    """(the tensors, the geometry, the features, the extra columns) from the cache."""
    path = cache_path()
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is not there; run --stage prepare first")
    z = np.load(path, allow_pickle=False)
    geom = fieldae.FieldGeometry(names=[str(s) for s in z["names"]],
                                 kinds=[str(s) for s in z["kinds"]],
                                 alphas=z["alphas"], scale=z["scale"])
    tens = fieldae.FieldTensors(coords=z["coords"].astype(float), weights=z["weights"].astype(float),
                                radii=z["radii"].astype(float), self_k=z["self_k"],
                                counts=z["counts"], names=[str(s) for s in z["mol_names"]])
    extra = {"gap": z["gap"], "n_rings": z["n_rings"]}
    return tens, geom, z["features"].astype(float), extra


def split_rows():
    """The train / validation / test indices of the split of record."""
    path = os.path.join(HERE, "results", "compas", f"splits_{SET}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is not there; run benchmarks/run_compas.py --stage prepare")
    s = jload(path)[SPLIT]
    return (np.asarray(s["train"], int), np.asarray(s["valid"], int), np.asarray(s["test"], int))


def joint_kernel_blocks(rows_a, rows_b=None):
    """The stored joint kernel of the model of record for the given rows, built group by group."""
    import run_compas as rc

    store = rc.KernelStore(SET, max_loaded=1)
    return store.joint(ALPHAS, np.asarray(rows_a, int),
                       None if rows_b is None else np.asarray(rows_b, int))


# --------------------------------------------------------------------------- #
# Stage: gate (a)
# --------------------------------------------------------------------------- #

def one_fit(tens, geom, feats, train, valid, test, code: int, n_pseudo: int,
            epochs: int = EPOCHS, seed: int = 0) -> dict:
    """Fit one autoencoder and measure it on the three parts of the split."""
    ae = fieldae.FieldAutoencoder(geom, feats.shape[1], code=int(code), n_pseudo=int(n_pseudo),
                                  hidden=HIDDEN, box=tens.box(),
                                  weight_scale=tens.weight_scale(), seed=seed)
    t0 = time.time()
    hist = fieldae.fit(ae, feats, tens, train, valid, epochs=int(epochs), batch=BATCH, lr=LR,
                       repulsion=REPULSION, min_distance=MIN_DISTANCE, patience=PATIENCE,
                       seed=seed, progress=None)
    err = fieldae.relative_errors(ae, feats, tens, test)
    pos, _w = fieldae.decode_rows(ae, feats, test[:256])
    out = {"code": int(code), "pseudo": int(n_pseudo), "seed": int(seed),
           "epochs": hist.epochs, "best_epoch": hist.best_epoch,
           "train": hist.train_error, "valid": hist.valid_error,
           "test_mean": float(np.mean(err)), "test_median": float(np.median(err)),
           "test_p90": float(np.quantile(err, 0.9)), "test_max": float(np.max(err)),
           "min_pseudo_distance": float(fieldae.min_pseudo_distance(pos)),
           "seconds": time.time() - t0, "parameters": int(sum(
               p.numel() for p in ae.parameters() if p.requires_grad))}
    log(f"  code {code:3d} pseudo {n_pseudo:3d}: test {out['test_mean']:.4f} "
        f"(train {out['train']:.4f}, valid {out['valid']:.4f}), {hist.epochs} epochs, "
        f"{out['seconds']:.0f} s")
    return out


def gate_a(codes=CODES, pseudo=PSEUDO, epochs: int = EPOCHS, force: bool = False,
           seeds=(0,)) -> dict:
    """The autoencoder against kernel PCA at the same number of latent dimensions."""
    path = result("gate_a.json")
    done = jload(path) if os.path.isfile(path) and not force else {"runs": [], "kpca": {}}
    tens, geom, feats, _extra = load_prepared()
    train, valid, test = split_rows()
    fieldae.require()

    wanted = [(c, REF_PSEUDO) for c in codes] + [(REF_CODE, p) for p in pseudo]
    seen = {(r["code"], r["pseudo"], r["seed"]) for r in done["runs"]}
    for code, n_pseudo in dict.fromkeys(wanted):
        for seed in seeds:
            if (code, n_pseudo, seed) in seen:
                continue
            done["runs"].append(one_fit(tens, geom, feats, train, valid, test, code, n_pseudo,
                                        epochs=epochs, seed=seed))
            jdump(done, path)

    want_q = sorted({r["code"] for r in done["runs"]})
    if any(str(q) not in done["kpca"] for q in want_q):
        log("kernel PCA on the same training molecules")
        t0 = time.time()
        K_tr = joint_kernel_blocks(train)
        K_te = joint_kernel_blocks(test, train)
        d_te = tens.self_k[test]
        log(f"  the training kernel is {K_tr.shape}, read in {time.time() - t0:.0f} s")
        t1 = time.time()
        errs = fieldae.kpca_errors(K_tr, K_te, d_te, want_q)
        seconds = (time.time() - t1) / max(len(want_q), 1)
        for q, e in errs.items():
            done["kpca"][str(q)] = {"components": int(q), "test_mean": float(np.mean(e)),
                                    "test_median": float(np.median(e)),
                                    "test_p90": float(np.quantile(e, 0.9)),
                                    "seconds": seconds}
            log(f"  kernel PCA with {q:3d} components: test {np.mean(e):.4f}")
        log(f"  one eigendecomposition for every number of components: "
            f"{time.time() - t1:.0f} s")
        jdump(done, path)
        del K_tr, K_te
    done["n_train"] = int(len(train))
    done["n_test"] = int(len(test))
    jdump(done, path)
    return done


# --------------------------------------------------------------------------- #
# Stage: the floor
# --------------------------------------------------------------------------- #

def floor(pseudo=PSEUDO, molecules: int = FLOOR_MOLECULES, steps: int = FLOOR_STEPS,
          force: bool = False) -> dict:
    """What the same pseudo-atoms reach when they are fitted to one molecule directly."""
    path = result("floor.json")
    done = jload(path) if os.path.isfile(path) and not force else {"rows": []}
    tens, geom, _feats, _extra = load_prepared()
    _train, _valid, test = split_rows()
    rows = test[:int(molecules)]
    seen = {r["pseudo"] for r in done["rows"]}
    for m in pseudo:
        if int(m) in seen:
            continue
        t0 = time.time()
        rel, pos, _w = fieldae.fit_pseudo_atoms(tens, rows, geom, n_pseudo=int(m),
                                                steps=int(steps), repulsion=REPULSION,
                                                min_distance=MIN_DISTANCE)
        done["rows"].append({"pseudo": int(m), "molecules": int(len(rows)),
                             "steps": int(steps), "mean": float(np.mean(rel)),
                             "median": float(np.median(rel)), "max": float(np.max(rel)),
                             "min_pseudo_distance": float(fieldae.min_pseudo_distance(pos)),
                             "atoms_mean": float(np.mean(tens.counts[rows])),
                             "seconds": time.time() - t0})
        log(f"  floor at {m:3d} pseudo-atoms: {np.mean(rel):.4f} over {len(rows)} molecules "
            f"of {np.mean(tens.counts[rows]):.0f} atoms ({time.time() - t0:.0f} s)")
        jdump(done, path)
    return done


# --------------------------------------------------------------------------- #
# Stage: gate (b)
# --------------------------------------------------------------------------- #

def model_of_record():
    """(the test indices, the measured gap, the predicted gap) of the model of record."""
    path = os.path.join(HERE, "results", "compas", f"fields_{GROUPS}_{SET}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is not there; run benchmarks/run_compas.py --stage fields")
    rec = jload(path)[f"{TARGET}|{GROUPS}|{SPLIT}"]
    cfg = rec["configs"][rec["chosen"]]
    _train, _valid, test = split_rows()
    y_pred = np.asarray(cfg["y_test_pred"], float)
    if len(y_pred) != len(test):
        raise SystemExit("the stored predictions and the split do not have the same length")
    return test, cfg, y_pred


def reference_ae(tens, geom, feats, train, valid, code: int = REF_CODE,
                 n_pseudo: int = REF_PSEUDO, epochs: int = EPOCHS, force: bool = False):
    """The one fitted autoencoder that gate (b) and the figure share, kept in the cache.

    Both of them need the same network, and fitting it twice would cost the same again and risk
    reporting a figure of one fit beside the numbers of another.
    """
    import torch

    fieldae.require()
    path = os.path.join(CACHE, f"ae_{SET}_{GROUPS}_c{code}_m{n_pseudo}.pt")
    rec_path = os.path.join(CACHE, f"ae_{SET}_{GROUPS}_c{code}_m{n_pseudo}.json")
    ae = fieldae.FieldAutoencoder(geom, feats.shape[1], code=int(code), n_pseudo=int(n_pseudo),
                                  hidden=HIDDEN, box=tens.box(),
                                  weight_scale=tens.weight_scale(), seed=0)
    if os.path.isfile(path) and os.path.isfile(rec_path) and not force:
        ae.load_state_dict(torch.load(path, weights_only=True))
        log(f"the reference autoencoder is read from {os.path.basename(path)}")
        return ae, jload(rec_path)
    hist = fieldae.fit(ae, feats, tens, train, valid, epochs=int(epochs), batch=BATCH, lr=LR,
                       repulsion=REPULSION, min_distance=MIN_DISTANCE, patience=PATIENCE)
    os.makedirs(CACHE, exist_ok=True)
    torch.save(ae.state_dict(), path)
    rec = {"code": int(code), "pseudo": int(n_pseudo), "epochs": hist.epochs,
           "best_epoch": hist.best_epoch, "train": hist.train_error, "valid": hist.valid_error,
           "min_distance": hist.min_distance, "seconds": hist.seconds}
    jdump(rec, rec_path)
    return ae, rec


def gate_b(code: int = REF_CODE, n_pseudo: int = REF_PSEUDO, epochs: int = EPOCHS,
           oneclass_train: int = 0, force: bool = False) -> dict:
    """The reconstruction error as an applicability-domain score, under the protocol of E5."""
    path = result("gate_b.json")
    if os.path.isfile(path) and not force:
        log("gate_b: already done")
        return jload(path)
    tens, geom, feats, extra = load_prepared()
    train, valid, test = split_rows()
    test_of_record, cfg, y_pred = model_of_record()
    if not np.array_equal(test, test_of_record):
        raise SystemExit("the split of record is not the split used here")
    y = extra["gap"][test]
    resid = y - y_pred
    r2 = 1.0 - float(np.sum(resid ** 2) / np.sum((y - np.mean(y)) ** 2))
    log(f"the model of record on the held-out part: R2 {r2:.4f} against the stored "
        f"{cfg['test']['r2']:.4f}")

    ae, hist = reference_ae(tens, geom, feats, train, valid, code, n_pseudo, epochs)
    rel = fieldae.relative_errors(ae, feats, tens, test)
    absolute = (rel ** 2) * tens.self_k[test]
    log(f"the autoencoder: test relative error {np.mean(rel):.4f} after {hist['epochs']} epochs")

    fit_rows = train if not oneclass_train else train[:int(oneclass_train)]
    log(f"the one-class scores on {len(fit_rows)} training molecules")
    K_tr = joint_kernel_blocks(fit_rows)
    K_te = joint_kernel_blocks(test, fit_rows)
    d_te = tens.self_k[test]
    estimates = {"field autoencoder, relative": rel,
                 "field autoencoder, absolute": absolute}
    seconds = {}
    for method in oneclass.ONECLASS_METHODS:
        t0 = time.time()
        if method == "ocsvm":
            from sklearn.svm import OneClassSVM

            svm = OneClassSVM(kernel="precomputed", nu=0.1).fit(K_tr)
            value = -np.asarray(svm.decision_function(K_te), float).ravel()
        else:
            sc = oneclass.make(method, nu=0.1).fit(K_tr)
            value = -np.asarray(sc.decision(K_te, d_te), float).ravel()
        estimates[oneclass.LABELS[method]] = value
        seconds[method] = time.time() - t0
        log(f"  {method}: {time.time() - t0:.0f} s")
    del K_tr, K_te

    scores = compare(estimates, y, y_pred)
    out = {"code": int(code), "pseudo": int(n_pseudo), "n_train": int(len(train)),
           "n_oneclass_train": int(len(fit_rows)), "n_test": int(len(test)),
           "r2_of_record": r2, "r2_stored": float(cfg["test"]["r2"]),
           "rmse_of_record": float(np.sqrt(np.mean(resid ** 2))),
           "autoencoder": {"test_mean": float(np.mean(rel)), "epochs": hist["epochs"],
                           "valid": hist["valid"]},
           "seconds": seconds,
           "scores": [{"name": s.name, "spearman": s.spearman, "kendall": s.kendall,
                       "efficiency": s.efficiency, "rmse_trusted": s.rmse_trusted,
                       "rmse_doubted": s.rmse_doubted, "constant": bool(s.constant)}
                      for s in scores],
           "aurc_oracle": float(scores[0].aurc_oracle) if scores else float("nan"),
           "aurc_random": float(scores[0].aurc_random) if scores else float("nan")}
    jdump(out, path)
    return out


# --------------------------------------------------------------------------- #
# Stage: the figure
# --------------------------------------------------------------------------- #

def figure(code: int = REF_CODE, n_pseudo: int = REF_PSEUDO, epochs: int = EPOCHS,
           field: str = "ring:terminal", out: str = "") -> str:
    """One held-out molecule: its own field, the decoded field, and the pseudo-atoms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cmfqsar.cofields import Grid

    tens, geom, feats, _extra = load_prepared()
    train, valid, test = split_rows()
    ae, _hist = reference_ae(tens, geom, feats, train, valid, code, n_pseudo, epochs)
    err = fieldae.relative_errors(ae, feats, tens, test)
    order = np.argsort(err)
    picks = [int(order[len(order) // 2]), int(order[-1])]      # the median and the worst
    f = geom.names.index(field) if field in geom.names else 0
    spacing = 0.3
    pos, wts = fieldae.decode_rows(ae, feats, test[picks])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    for row, pick in enumerate(picks):
        idx = int(test[pick])
        k = int(tens.counts[idx])
        real = fieldae.pre_image_fields(tens.coords[idx, :k], tens.weights[idx, :k])
        real.radii[:] = tens.radii[idx, :k]
        decoded = fieldae.pre_image_fields(pos[row], wts[row])
        lo = np.minimum(real.coords.min(axis=0), decoded.coords.min(axis=0)) - 2.5
        hi = np.maximum(real.coords.max(axis=0), decoded.coords.max(axis=0)) + 2.5
        shape = (int((hi[0] - lo[0]) / spacing) + 1, int((hi[1] - lo[1]) / spacing) + 1, 1)
        grid = Grid(origin=np.array([lo[0], lo[1], 0.0]), spacing=spacing, shape=shape)
        extent = [lo[0], lo[0] + spacing * (shape[0] - 1),
                  lo[1], lo[1] + spacing * (shape[1] - 1)]
        panels = [(mf, fieldae.decoded_volume(mf, geom, f, grid)[:, :, 0].T)
                  for mf in (real, decoded)]
        vmax = max(float(np.max(np.abs(v))) for _mf, v in panels) or 1.0
        for col, (mf, v) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(v, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="equal")
            ax.scatter(mf.coords[:, 0], mf.coords[:, 1], s=10,
                       c="k" if col == 0 else "limegreen", linewidths=0)
            ax.set_title(f"{tens.names[idx]}: the molecule" if col == 0
                         else f"{n_pseudo} pseudo-atoms, relative error {err[pick]:.3f}",
                         fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Field {geom.names[f]} of a held-out molecule and of the field it decodes to "
                 f"(code {code}). Top: the median molecule of the held-out part; "
                 f"bottom: the worst.", fontsize=10)
    fig.tight_layout()
    out = out or result("decoded.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"written to {out}")

    cube = result("decoded.cube")
    fieldae.export_decoded(cube, fieldae.pre_image_fields(pos[0], wts[0]), geom, f)
    log(f"written to {cube}")
    return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

def report(out: str = "") -> str:
    prep = jload(result("prepared.json")) if os.path.isfile(result("prepared.json")) else {}
    a = jload(result("gate_a.json")) if os.path.isfile(result("gate_a.json")) else None
    fl = jload(result("floor.json")) if os.path.isfile(result("floor.json")) else None
    b = jload(result("gate_b.json")) if os.path.isfile(result("gate_b.json")) else None
    lines = ["# A field autoencoder with an analytic decoder (item E11)", "",
             "The relative reconstruction error `||f - f'|| / ||f||` of a decoder of "
             "pseudo-atoms, against kernel PCA with the same number of latent dimensions. "
             "COMPAS-1D, field set ABCD of `run_compas.py` in the same scaling and the same "
             "random0 split; lower is better everywhere in this file.", ""]
    if prep:
        lines += [f"{prep['n_molecules']} molecules, {prep['atoms']} atoms "
                  f"({prep['max_atoms']} at most in one molecule), fields "
                  f"{', '.join('`' + n + '`' for n in prep['fields'])}, "
                  f"encoder input {prep['features'][1]} columns from "
                  f"{prep['landmarks']} landmark features.", "",
                  f"The closed-form self-overlap of `cmfqsar.fieldae` agrees with the diagonal "
                  f"of the stored kernel of `run_compas.py` to "
                  f"{prep['self_overlap_vs_stored_kernel']:.1e} relative, worst of "
                  f"{prep['n_molecules']} molecules.", ""]

    if a and a.get("runs"):
        runs = a["runs"]
        kpca = a.get("kpca", {})
        ref = [r for r in runs if r["pseudo"] == REF_PSEUDO]
        lines += ["## Gate (a): the autoencoder against kernel PCA", "",
                  f"The code dimension, at {REF_PSEUDO} pseudo-atoms. "
                  f"{a.get('n_train', 0)} training and {a.get('n_test', 0)} held-out molecules.",
                  "", "| latent dimensions | autoencoder, mean | autoencoder, median | "
                  "kernel PCA, mean | kernel PCA, median | the autoencoder is |",
                  "|---:|---:|---:|---:|---:|---|"]
        for r in sorted(ref, key=lambda r: r["code"]):
            k = kpca.get(str(r["code"]))
            if k is None:
                continue
            gain = k["test_mean"] - r["test_mean"]
            verdict = ("better" if gain > 0.001 else "worse" if gain < -0.001 else "the same")
            lines.append(f"| {r['code']} | {r['test_mean']:.4f} | {r['test_median']:.4f} | "
                         f"{k['test_mean']:.4f} | {k['test_median']:.4f} | "
                         f"{verdict} by {abs(gain):.4f} |")
        lines += ["", f"## The number of pseudo-atoms, at code {REF_CODE}", "",
                  "| pseudo-atoms | mean | median | 90th percentile | worst | "
                  "closest two pseudo-atoms, A | epochs |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sorted([r for r in runs if r["code"] == REF_CODE], key=lambda r: r["pseudo"]):
            lines.append(f"| {r['pseudo']} | {r['test_mean']:.4f} | {r['test_median']:.4f} | "
                         f"{r['test_p90']:.4f} | {r['test_max']:.4f} | "
                         f"{r['min_pseudo_distance']:.2f} | {r['epochs']} |")
        lines.append("")

    if fl and fl.get("rows"):
        lines += ["## The floor: the same pseudo-atoms fitted to one molecule directly", "",
                  "No code, no sharing between molecules: the best these pseudo-atoms can do on "
                  "a field of this set. It says whether a decoder's error is the bottleneck of "
                  "the code or the limit of the representation.", "",
                  "| pseudo-atoms | mean | median | worst | molecules | atoms per molecule |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for r in sorted(fl["rows"], key=lambda r: r["pseudo"]):
            lines.append(f"| {r['pseudo']} | {r['mean']:.4f} | {r['median']:.4f} | "
                         f"{r['max']:.4f} | {r['molecules']} | {r['atoms_mean']:.0f} |")
        lines.append("")

    if b:
        lines += ["## Gate (b): the reconstruction error as an applicability domain", "",
                  f"The model of record predicts the held-out part with R2 "
                  f"{b['r2_of_record']:.3f} and RMSE {b['rmse_of_record']:.3f} eV. Every "
                  f"estimate below is scored against the error that model actually made, under "
                  f"the protocol of item E5: rank correlation (rho), risk-coverage efficiency "
                  f"(1 = the ordering the errors themselves give, 0 = no better than ignoring "
                  f"the estimate), and the RMSE of the half it trusts against the half it "
                  f"doubts. The one-class scores are fitted on {b['n_oneclass_train']} training "
                  f"molecules.", "",
                  "| estimate | rho | tau | efficiency | RMSE trusted | RMSE doubted |",
                  "|---|---:|---:|---:|---:|---:|"]
        for s in b["scores"]:
            if s["constant"]:
                lines.append(f"| {s['name']} | -- | -- | constant | "
                             f"{s['rmse_trusted']:.3f} | {s['rmse_doubted']:.3f} |")
                continue
            lines.append(f"| {s['name']} | {s['spearman']:+.3f} | {s['kendall']:+.3f} | "
                         f"{s['efficiency']:+.3f} | {s['rmse_trusted']:.3f} | "
                         f"{s['rmse_doubted']:.3f} |")
        lines += ["", f"The ordering the errors themselves give has an area under the "
                  f"risk-coverage curve of {b['aurc_oracle']:.3f}; ignoring the estimate gives "
                  f"{b['aurc_random']:.3f}.", ""]

    text = "\n".join(lines) + "\n"
    out = out or result("table.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)
    print(f"written to {out}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("prepare", "gate_a", "floor", "gate_b", "figure"),
                    default=None)
    ap.add_argument("--codes", nargs="+", type=int, default=list(CODES))
    ap.add_argument("--pseudo", nargs="+", type=int, default=list(PSEUDO))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--floor-molecules", type=int, default=FLOOR_MOLECULES)
    ap.add_argument("--floor-steps", type=int, default=FLOOR_STEPS)
    ap.add_argument("--oneclass-train", type=int, default=0,
                    help="training molecules for the one-class scores; 0 uses them all")
    ap.add_argument("--threads", type=int, default=THREADS,
                    help=f"threads for PyTorch; {THREADS} is the fastest here (see the source)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return 0
    if not fieldae.available():
        raise SystemExit(fieldae.INSTALL)
    if args.threads:
        import torch

        torch.set_num_threads(int(args.threads))
    stages = (args.stage,) if args.stage else ("prepare", "gate_a", "floor", "gate_b", "figure")
    for stage in stages:
        log(f"=== {stage}")
        if stage == "prepare":
            prepare(force=args.force)
        elif stage == "gate_a":
            gate_a(codes=args.codes, pseudo=args.pseudo, epochs=args.epochs, force=args.force)
        elif stage == "floor":
            floor(pseudo=args.pseudo, molecules=args.floor_molecules, steps=args.floor_steps,
                  force=args.force)
        elif stage == "gate_b":
            gate_b(epochs=args.epochs, oneclass_train=args.oneclass_train, force=args.force)
        elif stage == "figure":
            figure(epochs=args.epochs)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
