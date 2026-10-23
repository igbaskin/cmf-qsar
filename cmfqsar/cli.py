#!/usr/bin/env python
"""
Command-line interface for building CMF 3D-QSAR models (no GUI needed).

Installed, it is the ``cmf`` command (``pyproject.toml``); from a checkout, ``python cmf_cli.py``
runs the same ``main``.

Examples
--------
  python cmf_cli.py build examples/benzamides_synthetic.sdf --activity pActivity \
        --alphas 0.3,0.5,0.8,1.2,2 --lams 1e-3,1e-2,0.1,1,10 --test-fraction 0.25 --save model.cmf

  # literature data set with MOL2 charges, a separate activity table carrying the published
  # train/test split (set column 1/2) and the protocol of the 2013 paper
  python cmf_cli.py build ace-all.mol2 --activity-file ace-data.txt --charges file --param tripos \
        --raw-scale --cv-no-shuffle --lam-search brent --alphas 0.01,0.05,0.1,0.25,0.5,0.75,1,2,3,5,10 \
        --lams 0.01,20 --nested --nested-on all

  python cmf_cli.py predict model.cmf new_molecules.sdf --out predictions.csv
"""
import argparse
import json
import sys
import time

import numpy as np

from cmfqsar.dataio import load_dataset, save_model, load_model, write_predictions_csv
from cmfqsar.fields import AtomTable, FieldSet, build_fields, cif_types_in
from cmfqsar.chemotypes import BASELINE_ALPHA
from cmfqsar.model import (CMFModel, cv_predict, grid_search, nested_cv,
                           r2_score, rmse, screening_scores)


def _split_fields(text):
    """Split a --fields list into the built-in fields and the per-atom property fields.

    "prop:NAME" asks for a field whose atomic weights are read from a property called NAME that
    the structure file carries, one number per atom: quantum charges, a Fukui index, a local
    softness, anything of that shape.
    """
    physchem, props = [], []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        (props if item.startswith("prop:") else physchem).append(
            item[5:] if item.startswith("prop:") else item)
    return physchem, props


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _load_dataset(args):
    ds = load_dataset(args.dataset, activity_prop=args.activity, activity_file=args.activity_file,
                      ensemble_by=(args.ensemble_by or None))
    if args.decoys:
        dec = load_dataset(args.decoys, activity_prop=args.activity, subset="decoy")
        dec.subset = np.array(["decoy"] * len(dec), dtype=object)
        ds.append(dec, subset="decoy")
        print(f"  {len(dec)} background molecules (decoys) loaded from {args.decoys}")
    if args.test_file:
        te = load_dataset(args.test_file, activity_prop=args.activity or (None if args.activity_file else ds.activity_prop),
                          activity_file=args.activity_file, subset="test")
        te.subset = np.array(["test"] * len(te), dtype=object)
        ds.append(te, subset="test")
    return ds


def _report_validation(args, model, ds, mf, tr, te, target):
    """y-randomisation, the applicability domain and bootstrap intervals, when asked for.

    All three work on kernels the fit has already computed, so nothing here rebuilds a field.
    The results are printed and stashed in ``model.stats.extra`` so that --json carries them.
    """
    from cmfqsar.validation import bootstrap_ci, bootstrap_refit, domain, scramble_kernel
    st = model.stats
    y = np.asarray(target, float)
    K = None
    if args.y_scramble or args.bootstrap_refit:
        K = model.joint_kernel(model.K_fields_train) if model.uses_fields else \
            model.training_kernel(model.train)
    if args.y_scramble:
        print()
        print(f"y-randomisation: {args.y_scramble} shuffles of the activities ...")
        if args.y_scramble_select:
            # the strict form: every shuffle gets the whole search, so the test covers the
            # selection and not only the fit. The kernels do not depend on the activities, so
            # they are built once here and reused by every shuffle.
            from cmfqsar.validation import scramble_selection
            train = model.train
            # the loop below walks the alpha grid, which moves the field set off the alpha the
            # search chose; remember it and put it back afterwards, or everything downstream
            # (the domain, the saved model, the predictions) would use the last grid point
            chosen_alphas = [s.alpha for s in model.fieldset.specs]
            grids, alphas = [], _floats(args.alphas)
            for a in (alphas or [None]):
                if a is not None:
                    model.fieldset.set_alpha(a)
                grids.append(model.training_kernel(train))
            params = [{"lam": float(l)} for l in _floats(args.lams)] or [dict(model.params)]
            print(f"  the search is repeated inside every shuffle: {len(grids)} alphas x "
                  f"{len(params)} lambdas")
            res = scramble_selection(grids, y[tr], method=model.method, param_grid=params,
                                     folds=args.folds, cv_seed=args.seed,
                                     runs=int(args.y_scramble), seed=args.seed)
            for spec, a in zip(model.fieldset.specs, chosen_alphas):
                spec.alpha = float(a)
            model.training_kernel(train)        # back on the kernel the search chose
        else:
            res = scramble_kernel(K, y[tr], method=model.method, params=model.params,
                                  folds=args.folds, cv_seed=args.seed,
                                  runs=int(args.y_scramble), seed=args.seed)
        print("  " + res.describe())
        note = res.caution()
        if note:
            print("  " + note)
        st.extra["scramble"] = {"statistic": res.statistic, "observed": res.observed,
                                "mean": res.mean, "sd": res.sd, "max": float(max(res.scrambled))
                                if len(res.scrambled) else float("nan"),
                                "z": res.z_score, "p": res.p_value, "runs": res.runs,
                                "values": [float(v) for v in res.scrambled]}
    if args.bootstrap:
        print()
        rows = []
        if "y_cv" in st.extra:
            rows.append(("q2 (cross-validation)", y[tr], np.asarray(st.extra["y_cv"], float), None))
        if te and "y_test_pred" in st.extra:
            rows.append(("R2pred (test set, deviations from the training mean)", y[te], np.asarray(st.extra["y_test_pred"], float),
                         float(np.mean(y[tr]))))
        for label, obs, pred, ref in rows:
            b = bootstrap_ci(obs, pred, statistic=label, runs=int(args.bootstrap),
                             seed=args.seed, y_ref_mean=ref)
            print("  " + b.describe())
            st.extra.setdefault("bootstrap", {})[label] = {"value": b.value, "lo": b.lo,
                                                           "hi": b.hi, "runs": b.runs}
        if args.bootstrap_refit:
            b = bootstrap_refit(K, y[tr], method=model.method, params=model.params,
                                runs=int(args.bootstrap_refit), seed=args.seed)
            print("  " + b.describe())
            st.extra.setdefault("bootstrap", {})["q2_oob"] = {"value": b.value, "lo": b.lo,
                                                              "hi": b.hi, "runs": b.runs}
    if args.domain and model.uses_fields:
        print()
        scored = [i for i in range(len(ds)) if i not in set(tr)]
        if not scored:
            scored = list(tr)
            where = "the training molecules themselves (no test set)"
        else:
            where = f"{len(scored)} molecules outside the training set"
        table = AtomTable.from_molecules([mf[i] for i in scored])
        K_xt = model.test_kernel(table)
        dom = domain(K_xt, model.joint_self(table), model.training_kernel(model.train),
                     k=args.domain_k, quantile=args.domain_quantile)
        print(f"Applicability domain, {where}:")
        print("  " + dom.describe())
        outside = [(ds.names[i], float(s)) for i, s in zip(scored, dom.mean_top)
                   if s < dom.threshold]
        for name, s in sorted(outside, key=lambda kv: kv[1])[:10]:
            print(f"    outside: {name[:40]:40s} similarity {s:.3f}")
        st.extra["domain"] = {"threshold": dom.threshold, "k": dom.k,
                              "names": [ds.names[i] for i in scored],
                              "similarity": [float(v) for v in dom.mean_top],
                              "nearest": [float(v) for v in dom.similarity],
                              "distance": [float(v) for v in dom.distance],
                              "inside": [bool(v) for v in dom.inside]}


def cmd_build(args):
    t0 = time.time()
    ds = _load_dataset(args)
    print(f"Loaded {len(ds)} molecules from {args.dataset}; activity: {ds.activity_prop}")
    ok = np.isfinite(ds.activity)
    if ok.sum() < len(ds):
        print(f"  {len(ds) - ok.sum()} molecules without activity are ignored")
    dec_idx = [int(i) for i in ds.decoy_indices]
    if args.test_fraction > 0:
        keep = ds.subset.copy()
        ds.random_split(args.test_fraction, seed=args.seed)
        ds.subset[dec_idx] = "decoy"                 # the background is never trained or tested on
    tr = [i for i in ds.train_indices if ok[i]]
    te = [i for i in ds.test_indices if ok[i]]
    print(f"  training set: {len(tr)} molecules, test set: {len(te)} molecules"
          + (f", decoys: {len(dec_idx)}" if dec_idx else ""))
    seed = None if args.cv_no_shuffle else args.seed

    physchem, atom_props = _split_fields(args.fields)
    cif = []
    if args.cif:
        cif = cif_types_in([ds.mols[i] for i in tr], args.cif_scheme, args.cif_min)
        if args.cif_types:
            cif = [t for t in args.cif_types.split(",") if t]
        print(f"  CIF atom types ({args.cif_scheme}): {', '.join(cif)}")
    fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, cif_types=cif, alpha=1.0, charge_model=args.charges,
                         prop_missing=args.prop_missing,
                         cif_scheme=args.cif_scheme, param_scheme=args.param,
                         conformers=args.conformations,
                         conformer_weighting=args.conformer_weighting,
                         conformer_kt=args.kt, max_conformers=args.max_conformers,
                         descriptors=[d for d in (args.descriptors or "").split(",") if d],
                         descriptor_h=args.descriptor_h,
                         probes=[p for p in (args.probes or "").split(",") if p],
                         probe_terms=args.probe_terms)
    if args.probes:
        from cmfqsar.probes import fit_report, probe_names
        unknown = [p for p in args.probes.split(",") if p and p not in probe_names()]
        if unknown:
            raise SystemExit(f"unknown probe(s) {', '.join(unknown)}; the ones defined are "
                             f"{', '.join(probe_names())}")
        for name in [p for p in args.probes.split(",") if p]:
            print("  " + fit_report(name, n_terms=args.probe_terms).replace("\n", "\n  "))
    if fs.descriptors:
        from cmfqsar.descriptors import _resolve, available
        for name in fs.descriptors:
            if name.startswith("calc:"):
                try:                       # a mistyped descriptor should stop here, not mid-fit
                    _resolve(name[5:])
                except KeyError as exc:
                    raise SystemExit(str(exc).strip('"'))
        have = available(ds.mols)
        unknown = [d for d in fs.descriptors if d not in have]
        print(f"  non-field descriptors: {', '.join(fs.descriptors)} (h = {fs.descriptor_h:g})"
              + (f"; NOT numeric on any molecule: {', '.join(unknown)}" if unknown else ""))
    if args.conformers:
        from cmfqsar.conformers import generate_for_dataset
        where = {"self": "superimposed on each molecule's own input pose",
                 "core": "superimposed on each molecule's own input pose over the common scaffold",
                 "reference": "superimposed on the reference molecule of the series",
                 "field": "superimposed on the reference molecule and refined on the field"}
        print(f"Generating up to {args.conformers} conformations per molecule "
              f"(pruning below {args.prune_rmsd:g} A), "
              f"{where[args.conformer_align]}")
        def conf_progress(msg, fr):
            print("\r  [%5.1f%%] %-44s" % (fr * 100, msg), end="", flush=True)

        ds.mols, failures = generate_for_dataset(
            ds.mols, n=args.conformers, seed=args.seed, prune_rmsd=args.prune_rmsd,
            align_to=args.conformer_align, fieldset=fs, window_rmsd=args.conformer_window,
            reference_index=(None if args.conformer_align_ref < 0 else args.conformer_align_ref),
            progress=conf_progress)
        print()
        for line in failures[:5]:
            print("  " + line)
    if fs.is_ensemble:
        from cmfqsar.conformers import ensemble_summary
        info = ensemble_summary(ds.mols, fs)
        if info:
            print(f"  4D mode: {info['conformations_mean']:.1f} conformations per molecule on "
                  f"average (max {info['conformations_max']:.0f}), effectively "
                  f"{info['effective_mean']:.1f} after weighting; kernels cost about "
                  f"{info['kernel_cost_factor']:.0f} times the single-conformation ones")
    mf = [build_fields(ds.mols[i], fs, ds.names[i]) for i in range(len(ds))]
    train = AtomTable.from_molecules([mf[i] for i in tr])
    target = ds.activity
    if args.task in ("classification", "oneclass"):
        if args.class_property:
            target = ds.classes_from_property(args.class_property)
            print(f"  classes from the property '{args.class_property}'")
        elif args.threshold is not None:
            target = ds.classes_from_threshold(args.threshold, active_above=not args.active_below)
            rel = "<=" if args.active_below else ">="
            print(f"  classes from the activity: active if {ds.activity_prop} {rel} {args.threshold:g}")
        else:
            raise SystemExit("classification needs --threshold or --class-property")
        n_pos = int(np.sum(target[tr] > 0))
        print(f"  training set: {n_pos} active, {len(tr) - n_pos} inactive")
        if args.task == "oneclass":
            tr = [i for i in tr if target[i] > 0]
            train = AtomTable.from_molecules([mf[i] for i in tr])
            print(f"  one-class model fitted on the {len(tr)} active molecules only")
        if args.method in ("krr", "gp", "svr", "nusvr"):
            args.method = "ocsvm" if args.task == "oneclass" else "svc"
            print(f"  method set to {args.method} for this task")
        if args.method == "gpc" and args.task == "oneclass":
            raise SystemExit("the Gaussian-process classifier needs both classes; use ocsvm")
        if args.task == "oneclass" and not args.normalize:
            # without cosine normalisation the decision value scales with the size of the molecule,
            # so the ranking measures molecular size rather than similarity to the target class
            args.normalize = True
            print("  kernel normalisation switched on (a one-class score on raw kernels is "
                  "dominated by molecular size)")
    if dec_idx and args.align_decoys:
        from cmfqsar.decoys import align_decoys, align_decoys_flexible
        act = [i for i in tr if not np.isfinite(target[i]) or target[i] > 0] or tr
        ref_tab = AtomTable.from_molecules([mf[i] for i in act])
        prog = lambda i, n: (print("  %d/%d decoys aligned" % (i, n), flush=True)
                             if (i % 10 == 0 or i == n) else None)
        n_conf = max(1, int(args.decoy_conformers))
        print(f"Aligning {len(dec_idx)} decoys to the {len(act)} training molecules "
              f"({args.decoy_restarts} restarts"
              + (f", {n_conf} conformations each" if n_conf > 1 else "") + ") ...")
        if n_conf > 1:
            chosen, res = align_decoys_flexible([ds.mols[i] for i in dec_idx],
                                                [ds.names[i] for i in dec_idx], ref_tab, fs,
                                                n_conformers=n_conf,
                                                restarts=args.decoy_restarts, progress=prog)
            for k, i in enumerate(dec_idx[:len(chosen)]):
                mf[i] = chosen[k]
        else:
            res = align_decoys([mf[i] for i in dec_idx], ref_tab, fs,
                               restarts=args.decoy_restarts, progress=prog)
        print()
        before = np.array([r.similarity_before for r in res])
        after = np.array([r.similarity_after for r in res])
        print(f"  field similarity to the nearest training molecule: "
              f"{before.mean():.3f} -> {after.mean():.3f} on average, best {after.max():.3f}")
        if n_conf > 1:
            moved = sum(1 for r in res if r.conformer > 0)
            print(f"  a generated conformation beat the one in the file for {moved} of "
                  f"{len(res)} decoys")
        for m, mol in zip([ds.mols[i] for i in dec_idx], [mf[i] for i in dec_idx]):
            conf = m.GetConformer()
            for a, xyz in enumerate(mol.coords):
                conf.SetAtomPosition(a, [float(v) for v in xyz])
    y = target[tr]
    model = CMFModel(fieldset=fs, task=args.task, method=args.method, normalize=args.normalize,
                     scale_fields=not args.raw_scale, fit_bias=not args.no_bias,
                     kernel_layer=args.kernel_layer, kernel_source=args.kernel,
                     ac_sigma=args.ac_sigma, ac_d_max=args.ac_dmax)
    if args.kernel == "autocorrelation":
        print(f"Kernel: field autocorrelation up to {args.ac_dmax:g} A in steps of "
              f"{args.ac_sigma:g} A, invariant to rotation and translation. No alignment is "
              f"assumed; the attenuation grid is not used, and there are no co-fields or atom "
              f"contributions, only the contributions of the training molecules.")
    if args.kernel_layer != "linear":
        print(f"Nonlinear kernel layer '{args.kernel_layer}': co-fields and atom contributions "
              "are not available for this model (the model is no longer linear in the field).")

    def prog(msg, fr):
        print(f"\r  [{fr * 100:5.1f}%] {msg:<70}", end="", flush=True)

    alphas, lams = _floats(args.alphas), _floats(args.lams)
    nm_start = _floats(args.nm_start)
    if args.task == "oneclass":
        # no cross-validation objective exists without negatives: nu is the expected outlier
        # fraction and alpha the level of generalisation, both chosen by the user
        a0, nu0 = alphas[0], _floats(args.nus)[0]
        model.fieldset.set_alpha(a0)
        model.params["nu"] = nu0
        model.params.update({"k": args.oc_k, "components": args.oc_components,
                             "oc_lam": args.oc_lam, "oc_score": args.oc_score})
        if args.method != "ocsvm":
            from cmfqsar.oneclass import LABELS
            print(f"  description of the actives: {LABELS[args.method]}")
        print(f"One-class model: alpha = {a0:g}, nu = {nu0:g} (not selected by cross-validation; "
              f"alpha sets the level of generalisation, nu the expected fraction of outliers)")
        model.fit(train, y, cv_folds=0)
        if te:
            model.evaluate_test(AtomTable.from_molecules([mf[i] for i in te]), target[te])
        print(model.stats.summary())
        scores = np.full(len(ds), np.nan)
        for k, i in enumerate(tr):
            scores[i] = model.stats.extra["y_fit"][k]
        for k, i in enumerate(te):
            if "y_test_pred" in model.stats.extra:
                scores[i] = model.stats.extra["y_test_pred"][k]
        if dec_idx:
            d_scores = model.predict(AtomTable.from_molecules([mf[i] for i in dec_idx]))
            for k, i in enumerate(dec_idx):
                scores[i] = d_scores[k]
            act = [i for i in te if target[i] > 0] or [i for i in tr]
            y_scr = np.concatenate([np.ones(len(act)), -np.ones(len(dec_idx))])
            d_scr = np.concatenate([scores[act], d_scores])
            model.stats.screening = screening_scores(y_scr, d_scr)
            for line in model.stats._screening_lines():
                print(line)
        order = [i for i in np.argsort(-np.where(np.isfinite(scores), scores, -np.inf))
                 if np.isfinite(scores[i])]
        print("Top 10 by decision value:")
        for rank, i in enumerate(order[:10], 1):
            lab = "" if not np.isfinite(target[i]) else ("active" if target[i] > 0 else "inactive")
            print(f"  {rank:3d}. {ds.names[i]:28s} {scores[i]:+8.4f}  {ds.subset[i]:5s} {lab}")
        if args.predictions:
            write_predictions_csv(args.predictions, ds.names, target, scores, ds.subset,
                                  value_header="decision_value", truth_header="class")
            print(f"Scores written to {args.predictions}")
        if args.json:
            out = {
                "dataset": args.dataset, "task": args.task, "activity": ds.activity_prop,
                "n_train": len(tr), "n_scored": int(np.isfinite(scores).sum()),
                "settings": {k: v for k, v in vars(args).items() if k not in ("func",)},
                "alpha": a0, "nu": nu0, "inlier_fraction": model.stats.inlier_fraction,
                "n_support": model.stats.n_support, "test": model.stats.test, "cif_types": cif,
                "names": list(ds.names), "subset": [str(x) for x in ds.subset],
                "classes": [None if not np.isfinite(v) else float(v) for v in target],
                "scores": {ds.names[i]: float(scores[i]) for i in range(len(ds)) if np.isfinite(scores[i])},
                "elapsed_s": time.time() - t0,
            }
            with open(args.json, "w") as fh:
                json.dump(out, fh, indent=1,
                          default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
            print(f"Results written to {args.json}")
        if args.save:
            save_model(args.save, model, extra={"dataset": args.dataset, "task": args.task,
                                                "activity": ds.activity_prop})
            print(f"Model saved to {args.save}")
        print(f"Elapsed {time.time() - t0:.1f} s")
        return
    res = grid_search(train, y, model, alphas, lams=lams, Cs=_floats(args.Cs),
                      epsilons=_floats(args.epsilons), nus=_floats(args.nus),
                      gammas=_floats(args.gammas), folds=args.folds, seed=seed,
                      optimize_h=args.optimize_h, mkl=args.mkl or False, mkl_rho=args.mkl_rho, progress=prog,
                      lam_search=args.lam_search,
                      alpha_search=args.alpha_search, nm_start=nm_start)
    print()
    knob = (f"sigma = {args.ac_sigma:g} A" if args.kernel == "autocorrelation"
            else f"alpha = {res.best_alpha:g}")
    if args.task == "classification":
        print(f"Best: {knob}, {res.best_params}, "
              f"1-AUCcv = {res.best_rmse:.3f}, balanced accuracy = {res.best_q2:.3f}")
    else:
        print(f"Best: {knob}, {res.best_params}, "
              f"RMSEcv = {res.best_rmse:.3f}, q2 = {res.best_q2:.3f}")
    if args.optimize_h or args.mkl:
        print("  mixing coefficients:", {k: round(v, 3) for k, v in res.best_h.items()})
    model.fit(train, y, cv_folds=args.folds, cv_seed=seed, reuse_kernels=True)
    if args.method == "gpc":
        from cmfqsar.gpc import brier_score
        p_fit = model.predict_proba(train)
        print(f"  log marginal likelihood {model.log_marginal_likelihood:.2f}, "
              f"Brier score of the fitted probabilities {brier_score(y, p_fit):.3f} "
              f"(0.25 is a coin)")
    if args.align_train_variance is not None:
        from cmfqsar.varalign import cv_align_training_set
        guard = None if args.align_train_variance <= 0 else args.align_train_variance
        print(f"Realigning the {len(tr)} training molecules by prediction confidence "
              f"({args.align_train_rounds} round(s), {args.align_train_folds} folds, "
              + (f"max displacement {guard} A)" if guard else "no displacement limit)"))
        rows = cv_align_training_set([mf[i] for i in tr], y, model, folds=args.align_train_folds,
                                     rounds=args.align_train_rounds, max_rmsd=guard, seed=seed,
                                     progress=prog)
        print()
        for r in rows:
            print("  " + r.describe())
        print("  NOTE: q2 below is computed on the geometry that has just been changed and is not a "
              "valid criterion for this step; judge it on a test set that was not realigned.")
        train = AtomTable.from_molecules([mf[i] for i in tr])
        res = grid_search(train, y, model, alphas, lams=lams, Cs=_floats(args.Cs),
                          epsilons=_floats(args.epsilons), nus=_floats(args.nus),
                          gammas=_floats(args.gammas), folds=args.folds,
                          seed=seed, optimize_h=args.optimize_h, mkl=args.mkl or False, mkl_rho=args.mkl_rho, progress=prog,
                          lam_search=args.lam_search, alpha_search=args.alpha_search, nm_start=nm_start)
        print()
        print(f"After realignment: alpha = {res.best_alpha:g}, {res.best_params}, "
              f"RMSEcv = {res.best_rmse:.3f}, q2 = {res.best_q2:.3f}")
        model.fit(train, y, cv_folds=args.folds, cv_seed=seed, reuse_kernels=True)
    if te:
        test = AtomTable.from_molecules([mf[i] for i in te])
        model.evaluate_test(test, target[te])
    ncv = None
    if args.nested and args.task != "regression":
        print("External cross-validation is implemented for regression only; skipped.")
        args.nested = False
    if args.nested:
        if args.nested_on == "all":
            idx = [i for i in range(len(ds)) if ok[i]]
            print(f"External cross-validation on all {len(idx)} molecules ...")
        else:
            idx = tr
            print(f"External cross-validation on the {len(idx)} training molecules ...")
        table = AtomTable.from_molecules([mf[i] for i in idx])
        # A repeated external CV reshuffles the molecules before every repeat even with
        # --cv-no-shuffle: the original scripts permute the molecules once per repeat and then use
        # interleaved outer *and* inner folds on the permuted order, i.e. random folds in both loops.
        # With a single repeat --cv-no-shuffle gives interleaved folds in file order in both loops.
        outer_seed = args.seed if args.repeats > 1 else seed
        ncv = nested_cv(table, ds.activity[idx], model, alphas, lams=lams, Cs=_floats(args.Cs),
                        epsilons=_floats(args.epsilons), nus=_floats(args.nus), outer_folds=args.outer_folds,
                        inner_folds=args.folds, repeats=args.repeats, seed=outer_seed, inner_seed=outer_seed,
                        optimize_h=args.optimize_h, mkl=args.mkl or False, mkl_rho=args.mkl_rho, gammas=_floats(args.gammas),
                        progress=prog, lam_search=args.lam_search,
                        alpha_search=args.alpha_search, nm_start=nm_start)
        print()
        model.stats.q2_ex, model.stats.rmse_ex = ncv.q2_ex, ncv.rmse_ex
        model.stats.q2_ex_sd, model.stats.rmse_ex_sd = ncv.q2_ex_sd, ncv.rmse_ex_sd
        if args.repeats > 1:
            model.stats.q2_ex_aggr, model.stats.rmse_ex_aggr = ncv.q2_ex_aggr, ncv.rmse_ex_aggr
    print(model.stats.summary())
    if model.desc_train_ is not None:
        # how much of the model rests on the descriptors rather than on the geometry: the mean
        # absolute contribution of each, over the training molecules
        import collections
        totals = collections.Counter()
        for i in tr:
            for name, v in model.descriptor_contributions(mf[i]):
                totals[name] += abs(v) / len(tr)
        print("Non-field descriptors (mean |contribution| per training molecule):")
        for name, v in totals.most_common():
            print(f"  {name:28s} {v:.4f}")
        if model.desc_scaler_.constant:
            print(f"  constant over the training set, so inert: "
                  f"{', '.join(model.desc_scaler_.constant)}")
    _report_validation(args, model, ds, mf, tr, te, target)
    if args.save:
        save_model(args.save, model, extra={"dataset": args.dataset, "activity": ds.activity_prop,
                                            "train_names": [ds.names[i] for i in tr]})
        print(f"Model saved to {args.save}")
    if args.predictions:
        yp = np.full(len(ds), np.nan)
        yp[tr] = model.stats.extra["y_fit"]
        if te:
            yp[te] = model.stats.extra["y_test_pred"]
        ycv = np.full(len(ds), np.nan)
        if "y_cv" in model.stats.extra:
            ycv[tr] = model.stats.extra["y_cv"]
        dom = model.stats.extra.get("domain")
        ad = None
        if dom:
            # carry the domain into the file, so that an extrapolation stays marked as one
            by_name = dict(zip(dom["names"], dom["similarity"]))
            ad = np.array([by_name.get(nm, np.nan) for nm in ds.names], dtype=float)
        write_predictions_csv(args.predictions, ds.names, ds.activity, yp, ds.subset, y_cv=ycv,
                              domain=ad,
                              domain_threshold=dom["threshold"] if dom else None)
        print(f"Predictions written to {args.predictions}")
    if args.json:
        st = model.stats
        if args.task != "regression":
            out = {
                "dataset": args.dataset, "task": args.task, "activity": ds.activity_prop,
                "n_train": len(tr), "n_test": len(te),
                "settings": {k: v for k, v in vars(args).items() if k not in ("func",)},
                "best_alpha": res.best_alpha, "best_params": res.best_params, "best_h": res.best_h,
                "fit": st.fit, "cv": st.cv, "test": st.test,
                "inlier_fraction": st.inlier_fraction, "cif_types": cif, "search_table": res.table,
                "names": list(ds.names), "subset": [str(s) for s in ds.subset],
                "classes": [None if not np.isfinite(v) else float(v) for v in target],
                "decision_fit": {ds.names[i]: float(v) for i, v in zip(tr, st.extra.get("y_fit", []))},
                "decision_cv": {ds.names[i]: float(v) for i, v in zip(tr, st.extra.get("y_cv", []))},
                "decision_test": {ds.names[i]: float(v) for i, v in zip(te, st.extra.get("y_test_pred", []))},
                "elapsed_s": time.time() - t0,
            }
            with open(args.json, "w") as fh:
                json.dump(out, fh, indent=1,
                          default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
            print(f"Results written to {args.json}")
            print(f"Elapsed {time.time() - t0:.1f} s")
            return
        out = {
            "dataset": args.dataset, "activity": ds.activity_prop, "n_train": len(tr), "n_test": len(te),
            "settings": {k: v for k, v in vars(args).items() if k not in ("func",)},
            "best_alpha": res.best_alpha, "best_params": res.best_params, "best_h": res.best_h,
            "r2": st.r2, "rmse_fit": st.rmse_fit, "q2": st.q2, "rmse_cv": st.rmse_cv,
            "r2_pred": st.r2_pred, "r2_pred_train_mean": st.r2_pred_train_mean,
            "r2_pred_train_var": st.r2_pred_train_var, "rmse_pred": st.rmse_pred,
            "q2_ex": st.q2_ex, "rmse_ex": st.rmse_ex, "q2_ex_sd": st.q2_ex_sd, "rmse_ex_sd": st.rmse_ex_sd,
            "q2_ex_aggr": st.q2_ex_aggr, "rmse_ex_aggr": st.rmse_ex_aggr,
            "cif_types": cif,
            "search_table": res.table,
            "names": list(ds.names), "subset": [str(s) for s in ds.subset], "y": ds.activity.tolist(),
            "y_fit": {ds.names[i]: float(v) for i, v in zip(tr, st.extra["y_fit"])},
            "y_cv": {ds.names[i]: float(v) for i, v in zip(tr, st.extra.get("y_cv", []))},
            "y_test_pred": {ds.names[i]: float(v) for i, v in zip(te, st.extra.get("y_test_pred", []))},
            "elapsed_s": time.time() - t0,
        }
        if ncv is not None:
            idx = [i for i in range(len(ds)) if ok[i]] if args.nested_on == "all" else tr
            out["y_ex_pred"] = {ds.names[i]: float(v) for i, v in zip(idx, ncv.y_pred)}
            out["nested_per_repeat"] = ncv.per_repeat
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        print(f"Results written to {args.json}")
    print(f"Elapsed {time.time() - t0:.1f} s")


def _check_contract(model, mf, allow=False):
    """Say how the model's training set was prepared, and stop if these molecules were not."""
    from cmfqsar.contract import check_molecules, summary
    print("Model: " + summary(model))
    findings = check_molecules(model, mf)
    for f in findings:
        print(f"  {f}")
    refused = [f for f in findings if f.level == "refuse"]
    if refused and not allow:
        raise SystemExit("prediction refused (" + ", ".join(f.code for f in refused)
                         + "); --allow-mismatch predicts anyway")
    return findings


def cmd_predict(args):
    model, extra = load_model(args.model)
    ds = load_dataset(args.dataset, activity_prop=args.activity, activity_file=args.activity_file)
    mf = [build_fields(m, model.fieldset, n) for m, n in zip(ds.mols, ds.names)]
    _check_contract(model, mf, allow=args.allow_mismatch)
    if args.align_variance is not None:
        guard = None if args.align_variance <= 0 else args.align_variance
        how = "rigid"
        if args.align_flexible:
            how = "rigid motion and the rotatable bonds"
            if args.align_conformers:
                how += f", from {args.align_conformers} generated conformations"
        print(f"Aligning {len(mf)} molecules by prediction confidence ({how})"
              + (f" (max displacement {guard} A)" if guard else " (no displacement limit)") + " ...")
        if args.align_flexible:
            from cmfqsar.flexalign import flexible_align
            res = flexible_align(mf, list(ds.mols), model, max_rmsd=guard,
                                 restarts=args.align_restarts,
                                 energy_slack=args.align_energy_slack,
                                 energy_weight=args.align_energy_weight,
                                 max_torsions=args.align_max_torsions,
                                 conformers=args.align_conformers)
        else:
            from cmfqsar.varalign import variance_align
            res = variance_align(mf, model, max_rmsd=guard, restarts=args.align_restarts)
        for r in res:
            print(f"  {r.name:28s} variance {r.variance_before:8.4f} -> {r.variance_after:8.4f}   "
                  f"moved {r.rmsd:5.2f} A   prediction {r.prediction_before:7.3f} -> {r.prediction_after:7.3f}"
                  + ("   [at the displacement limit]" if r.limited else ""))
        if args.aligned_out:
            from rdkit import Chem
            w = Chem.SDWriter(args.aligned_out)
            for m, mol, nm in zip(ds.mols, mf, ds.names):
                mm = Chem.Mol(m); conf = mm.GetConformer()
                for i, xyz in enumerate(mol.coords):
                    conf.SetAtomPosition(i, [float(v) for v in xyz])
                mm.SetProp("_Name", nm); w.write(mm)
            w.close()
            print(f"Aligned structures written to {args.aligned_out}")
    table = AtomTable.from_molecules(mf)
    if model.is_linear and model.method in ("krr", "gp"):
        yp, var = model.predict_with_variance(table)
        for n, v, s in zip(ds.names, yp, var):
            print(f"{n:30s} {v:8.3f}   (GP var {s:.3g})")
    else:
        # the posterior variance is defined for the linear kernel only
        yp, var = model.predict(table), None
        for n, v in zip(ds.names, yp):
            print(f"{n:30s} {v:8.3f}")
    if np.isfinite(ds.activity).sum() >= 2:
        model.evaluate_test(table, ds.activity)
        print(model.stats.summary())
    if args.out:
        write_predictions_csv(args.out, ds.names, ds.activity, yp, ["test"] * len(ds), var=var)
        print(f"Predictions written to {args.out}")


def cmd_explain(args):
    """Shapley decomposition of a prediction: works for linear and nonlinear kernels alike."""
    from cmfqsar.shapley import (atom_field_shapley, atom_shapley, common_core, field_shapley,
                                 substituent_groups)
    model, extra = load_model(args.model)
    ds = load_dataset(args.dataset, activity_prop=args.activity)
    mf = [build_fields(m, model.fieldset, n) for m, n in zip(ds.mols, ds.names)]
    _check_contract(model, mf, allow=args.allow_mismatch)
    if args.molecule is None:
        idx = list(range(len(ds)))
    else:
        want = {w.strip() for w in args.molecule.split(",")}
        idx = [i for i, n in enumerate(ds.names) if n in want]
        if not idx:
            raise SystemExit(f"no molecule named {args.molecule} in {args.dataset}")
    rows = []
    for i in idx:
        mol, name = mf[i], ds.names[i]
        if args.players == "fields":
            res = field_shapley(model, mol, interactions=args.interactions)
            base = "no fields"
        elif args.players == "atomfields":
            res = atom_field_shapley(model, mol, samples=args.samples, seed=args.seed,
                                     interactions=args.interactions)
            base = "the empty field"
        elif args.players == "groups":
            refs = [m for k, m in enumerate(ds.mols) if k != i][:20]
            if not refs:          # a single molecule was given: fall back to the training set
                from rdkit import Chem
                refs = [Chem.MolFromMolBlock(t.molblock, removeHs=False)
                        for t in model.train.molecules[:20]]
                refs = [m for m in refs if m is not None]
            if not refs:
                raise SystemExit("the common scaffold needs reference molecules; give a file with "
                                 "the series, or use --players atoms")
            core = common_core(ds.mols[i], refs)
            groups, labels = substituent_groups(ds.mols[i], core)
            if not groups:
                print(f"{name}: this molecule is the common scaffold itself; skipped")
                continue
            res = atom_shapley(model, mol, groups=groups, core=core, labels=labels,
                               samples=args.samples, seed=args.seed, interactions=args.interactions)
            base = f"the common scaffold of {len(core)} atoms"
        else:
            res = atom_shapley(model, mol, samples=args.samples, seed=args.seed,
                               interactions=args.interactions)
            base = "the empty field"
        how = "exact" if res.exact else f"{args.samples} permutations"
        print(f"\n{name}: prediction {res.prediction:.3f} = {res.baseline:.3f} for {base} "
              f"+ the values below ({how}, {res.n_evaluations} coalition evaluations)")
        for label, v, se in res.table()[:args.top]:
            print(f"   {label:24s} {v:+8.4f}" + ("" if np.isnan(se) else f"  +/- {se:.4f}"))
        for a, b, v in res.top_interactions(args.top):
            print(f"   interaction {a:18s} x {b:18s} {v:+8.4f}")
        for label, v, se in res.table():
            rows.append((name, label, v, se))
        if args.cube and res.player_fields is not None:
            from cmfqsar.cofields import export_cube, make_grid, shapley_volume
            vals = res.field_atom_values(mol.n_atoms, model.fieldset.n_fields)
            widths = [s.alpha if s.kind == "gauss" else s.alpha / float(mol.radii.max()) ** 2
                      for s in model.fieldset.specs]
            grid = make_grid(mol.coords, margin=max(3.0, 2.5 / np.sqrt(max(min(widths), 1e-3))),
                             spacing=args.cube_spacing)
            dv = grid.spacing ** 3
            for k, spec in enumerate(model.fieldset.specs):
                if not spec.enabled or not np.any(vals[k]):
                    continue
                vol = shapley_volume(mol, spec, vals[k], grid)
                out = f"{args.cube}_{name}_{spec.name.replace(':', '_')}.cube"
                export_cube(out, grid, vol, mol, title=f"CMF Shapley field {spec.name} of {name}")
                print(f"   {out}  integral {vol.sum() * dv:+.4f}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["molecule", "player", "shapley_value", "std_error"])
            for r in rows:
                w.writerow([r[0], r[1], f"{r[2]:.6f}", "" if np.isnan(r[3]) else f"{r[3]:.6f}"])
        print(f"\nShapley values written to {args.out}")


def cmd_complement(args):
    """Field complementarity of a data set with a protein pocket."""
    from cmfqsar.complementarity import (cofield_agreement, occupancy, pocket_fields,
                                         series_scores)
    from cmfqsar.protein import fetch_pdb, find_pocket, read_structure
    ds = load_dataset(args.dataset, activity_prop=args.activity)
    path = args.structure
    if not path:
        codes = [m.GetProp("pdb") for m in ds.mols if m is not None and m.HasProp("pdb")]
        path = find_pocket(args.dataset, codes[0] if codes else None)
        if not path and codes:
            print(f"Fetching {codes[0].upper()} from RCSB")
            path = fetch_pdb(codes[0])
    if not path:
        raise SystemExit("give a structure with --structure, or use a data set that names a "
                         "PDB entry")
    prot = read_structure(path, keep_waters=False)
    print(f"Structure {prot.name}: {prot.n_atoms} atoms, {prot.n_residues} residues, from {path}")
    physchem, atom_props = _split_fields(args.fields)
    fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, alpha=args.alpha, charge_model=args.charges)
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    pocket = pocket_fields(prot, fs)
    print(f"Pocket fields: {pocket.n_atoms} atoms after adding hydrogens and ionising the "
          f"side chains")

    def prog(msg, fr):
        print("\r  [%5.1f%%] %-48s" % (fr * 100, msg), end="", flush=True)

    res, stats = series_scores(mf, pocket, fs, activity=ds.activity,
                               surface=not args.no_surface, progress=prog)
    print()
    for c in res[:args.top]:
        print("  " + c.describe())
    if len(res) > args.top:
        print(f"  ... {len(res) - args.top} more")
    if stats:
        print("\nAgainst the activity:")
        for k, v in sorted(stats.items()):
            print(f"  {k:30s} {v:+.3f}")
    profiles = None
    if args.residues:
        from cmfqsar.complementarity import (InteractionProfile, _pearson, consensus_profile,
                                             interaction_profile, pocket_residues,
                                             profile_similarity)
        labels, order = pocket_residues(prot, pocket)
        profiles = [interaction_profile(m, pocket, fs, labels, order) for m in mf]
        cols = list(profiles[0].labels)
        print(f"\nBy residue: {len(order)} residues, {len(cols)} interaction types "
              f"({', '.join(cols)})")
        for p in profiles[:args.top]:
            print("  " + p.describe())
        if len(profiles) > args.top:
            print(f"  ... {len(profiles) - args.top} more")
        names_ds = list(ds.names)
        if args.reference:
            if args.reference not in names_ds:
                raise SystemExit(f"no molecule called {args.reference!r}; the file has "
                                 f"{', '.join(names_ds[:5])}...")
            query = profiles[names_ds.index(args.reference)]
            what = f"the profile of {args.reference}"
        else:
            freq, strength = consensus_profile(profiles, threshold=args.residue_threshold)
            query = InteractionProfile(
                name="consensus", residues=order, labels=cols,
                terms=np.nan_to_num(strength).reshape(len(order), len(cols)),
                contacts=profiles[0].contacts)
            what = f"the consensus of the {len(profiles)} molecules"
            shared = [(float(freq[k]), order[k // len(cols)], cols[k % len(cols)])
                      for k in range(len(freq)) if freq[k] > 0.5]
            shared.sort(key=lambda t: -t[0])
            print("  conserved in more than half of them: "
                  + (", ".join(f"{r} {c} ({f:.0%})" for f, r, c in shared[:10]) or "none"))
        sim = np.array([profile_similarity(p, query) for p in profiles])
        tani = np.array([profile_similarity(p, query, binary=True,
                                            threshold=args.residue_threshold)
                         for p in profiles])
        print(f"\nAgainst {what}:")
        for nm, s, t in list(zip(names_ds, sim, tani))[:args.top]:
            print(f"  {nm:30s} cosine {s:+.3f}   Tanimoto {t:.3f}")
        ok = np.isfinite(ds.activity)
        if ok.sum() >= 3:
            print(f"  against the activity: cosine r {_pearson(sim[ok], ds.activity[ok]):+.3f}, "
                  f"Tanimoto r {_pearson(tani[ok], ds.activity[ok]):+.3f}")
    if args.model:
        model, _ = load_model(args.model)
        if not model.is_linear:
            print("\nThe saved model is not linear, so it has no co-field to compare.")
        else:
            from cmfqsar.cofields import make_grid
            g = make_grid(model.train.coords, margin=3.0, spacing=args.spacing)
            w = occupancy(model.train.molecules, g).reshape(-1)
            pf = pocket_fields(prot, model.fieldset)
            agree = cofield_agreement(model, pf, g, w, statistic=args.statistic)
            print(f"\nCo-fields against the pocket ({args.statistic}):")
            for k, v in agree.items():
                print(f"  {k:30s} {v:+.3f}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            writer = csv.writer(fh)
            keys = list(res[0].terms)
            surf = list(res[0].surface)
            writer.writerow(["molecule", "activity", "score", "burial", "clashes"] + keys
                            + [f"surface_{k}" for k in surf])
            for c, y in zip(res, ds.activity):
                writer.writerow([c.name, f"{y:.4f}", f"{c.score:.4f}", f"{c.burial:.4f}",
                                 c.clashes]
                                + [f"{c.terms.get(k, float('nan')):.4f}" for k in keys]
                                + [f"{c.surface.get(k, float('nan')):.4f}" for k in surf])
        print(f"\nScores written to {args.out}")
    if args.profile_out:
        if not profiles:
            raise SystemExit("--profile-out needs --residues")
        import csv
        with open(args.profile_out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["molecule", "residue", "contact"] + list(profiles[0].labels))
            for p in profiles:
                t = np.asarray(p.terms, float)
                for i, res in enumerate(p.residues):
                    writer.writerow([p.name, res, f"{p.contacts[i]:.2f}"]
                                    + [f"{v:.5f}" for v in t[i]])
        print(f"Interaction profiles written to {args.profile_out}")


def cmd_mil(args):
    """Multi-instance learning over conformations: let the model choose the pose."""
    import time
    from cmfqsar.conformers import generate_for_dataset
    from cmfqsar.mil import (cv_predict as mil_cv, flatten_conformers, learn_weights,
                             predict_bags)
    from cmfqsar.model import r2_score, rmse
    t0 = time.time()
    ds = load_dataset(args.dataset, activity_prop=args.activity, activity_file=args.activity_file,
                      ensemble_by=(args.ensemble_by or None))
    ok = np.isfinite(ds.activity)
    tr = [int(i) for i in ds.train_indices if ok[i]]
    te = [int(i) for i in ds.test_indices if ok[i]]
    if len(tr) < 4:
        raise SystemExit("the training set needs at least four molecules with activities")
    print(f"Loaded {len(ds)} molecules: {len(tr)} training, {len(te)} test")
    mols = list(ds.mols)
    if args.conformers:
        print(f"Generating up to {args.conformers} conformations per molecule")

        def prog(msg, fr):
            print("\r  [%5.1f%%] %-44s" % (fr * 100, msg), end="", flush=True)

        mols, failures = generate_for_dataset(mols, n=args.conformers, seed=args.seed,
                                              prune_rmsd=args.prune_rmsd,
                                              align_to=args.conformer_align,
                                              window_rmsd=args.conformer_window, progress=prog)
        print()
        for line in failures[:5]:
            print("  " + line)
    if args.dock:
        from cmfqsar.complementarity import pocket_fields
        from cmfqsar.docking import dock_dataset, summarise
        from cmfqsar.protein import fetch_pdb, find_pocket, read_structure
        path = args.structure
        if not path:
            codes = [m.GetProp("pdb") for m in ds.mols if m is not None and m.HasProp("pdb")]
            path = find_pocket(args.dataset, codes[0] if codes else None)
            if not path and codes:
                path = fetch_pdb(codes[0])
        if not path:
            raise SystemExit("docking needs a structure: give --structure, or use a data set "
                             "that names a PDB entry")
        prot = read_structure(path, keep_waters=False)
        fs_dock = FieldSet.create(physchem=[f for f in args.fields.split(",") if f],
                                  alpha=0.5, charge_model=args.charges)
        pocket = pocket_fields(prot, fs_dock)
        print(f"Docking every conformation into {prot.name} ({pocket.n_atoms} atoms with "
              f"hydrogens), {args.dock_restarts} restart(s)")

        def dock_prog(msg, fr):
            print("\r  [%5.1f%%] %-44s" % (fr * 100, msg), end="", flush=True)

        mols, dock_res = dock_dataset(mols, pocket, fs_dock, restarts=args.dock_restarts,
                                      seed=args.seed, keep_best=args.dock_keep,
                                      progress=dock_prog)
        print()
        info = summarise(dock_res)
        if info:
            print(f"  score {info['score_before']:+.2f} -> {info['score_after']:+.2f}, "
                  f"clash {info['clash_before']:.1f} -> {info['clash_after']:.1f}, "
                  f"poses moved {info['moved']:.1f} A on average")
    counts = [mols[i].GetNumConformers() for i in tr]
    if max(counts) < 2:
        raise SystemExit("every molecule has one conformation; nothing to choose between. "
                         "Use --conformers N, or a multi-conformer file with --ensemble-by name")
    print(f"  {np.mean(counts):.1f} conformations per molecule on average, up to {max(counts)}")
    physchem, atom_props = _split_fields(args.fields)
    y = ds.activity[tr]
    rules = [("ensemble (beta = 0)", 0.0, "soft")]
    rules += [(f"soft, beta = {b:g}", float(b), "soft") for b in _floats(args.betas) if b > 0]
    if args.hard:
        rules.append(("hard selection", 0.0, "hard"))
    best = None
    for alpha in _floats(args.alphas):
        fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, alpha=alpha, charge_model=args.charges,
                             param_scheme=args.param)
        flat_tr, bags_tr = flatten_conformers([mols[i] for i in tr], fs, [ds.names[i] for i in tr],
                                              max_conformers=args.max_conformers)
        model = CMFModel(fieldset=fs, method="krr", normalize=not args.raw_kernels)
        K_conf = model.joint_kernel(model.prepare_training_kernels(
            AtomTable.from_molecules(flat_tr)))
        for lam in _floats(args.lams):
            for label, beta, sel in rules:
                ycv = mil_cv(K_conf, bags_tr, y, beta=beta, lam=lam, folds=args.folds,
                             seed=args.seed, selection=sel, init=args.init, rounds=args.rounds)
                q2, err = r2_score(y, ycv), rmse(y, ycv)
                print(f"  alpha {alpha:<5g} lambda {lam:<6g} {label:22s} q2cv {q2:+.3f}  "
                      f"RMSEcv {err:.3f}", flush=True)
                if best is None or err < best[0]:
                    best = (err, q2, alpha, lam, beta, sel, label, ycv)
    err, q2, alpha, lam, beta, sel, label, ycv = best
    print(f"\nBest: alpha {alpha:g}, lambda {lam:g}, {label}: q2cv {q2:.3f}, RMSEcv {err:.3f}")
    fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, alpha=alpha, charge_model=args.charges,
                         param_scheme=args.param)
    flat_tr, bags_tr = flatten_conformers([mols[i] for i in tr], fs, [ds.names[i] for i in tr],
                                          max_conformers=args.max_conformers)
    model = CMFModel(fieldset=fs, method="krr", normalize=not args.raw_kernels)
    K_conf = model.joint_kernel(model.prepare_training_kernels(AtomTable.from_molecules(flat_tr)))
    fit = learn_weights(K_conf, bags_tr, y, beta=beta, lam=lam, selection=sel, init=args.init,
                        rounds=args.rounds)
    eff = fit.effective_counts()
    print(f"  conformations that actually count: {eff.mean():.2f} of {np.mean(counts):.1f} "
          f"on average, {fit.rounds} rounds of re-weighting")
    if te:
        flat_te, bags_te = flatten_conformers([mols[i] for i in te], fs, [ds.names[i] for i in te],
                                              max_conformers=args.max_conformers)
        K_te = model.joint_kernel(model.test_kernels(AtomTable.from_molecules(flat_te)))
        pred = predict_bags(K_te, fit, bags_te)
        print(f"  Test set: R2pred {r2_score(ds.activity[te], pred):.3f}, "
              f"RMSEpred {rmse(ds.activity[te], pred):.3f}")
    if args.out:
        import csv
        chosen = fit.chosen()
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["molecule", "activity", "predicted_cv", "conformations",
                        "effective", "chosen_conformation"])
            for b, i in enumerate(tr):
                first = int(np.flatnonzero(bags_tr == b)[0])
                w.writerow([ds.names[i], f"{ds.activity[i]:.4f}", f"{ycv[b]:.4f}",
                            int(np.sum(bags_tr == b)), f"{eff[b]:.3f}",
                            int(chosen[b] - first)])
        print(f"  Chosen conformations written to {args.out}")
    print(f"Elapsed {time.time() - t0:.1f} s")


def cmd_topomer(args):
    """Model a series by its substituents, each posed by rule instead of superimposed."""
    from rdkit import Chem
    from cmfqsar.topomer import (TopomerSpec, choose_core, cuttable_fraction, series_labels,
                                 topomer_fields, unique_substituents)
    ds = load_dataset(args.dataset, activity_prop=args.activity,
                      activity_file=args.activity_file)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    yy = y[ok]
    core = Chem.MolFromSmiles(args.core) if args.core else None
    if core is None and args.core_smarts:
        core = Chem.MolFromSmarts(args.core_smarts)
    how = "given"
    if core is None:
        core, how = choose_core(mols, mode=args.core_mode)
    if core is None:
        raise SystemExit("no scaffold found: give one with --core or --core-smarts")
    labels = series_labels(mols, core)
    print(f"Loaded {len(mols)} molecules with activities from {args.dataset}")
    print(f"  core {Chem.MolToSmiles(core)} ({how})  ({len(labels)} attachment points, "
          f"{len(unique_substituents(mols, core))} distinct substituents)")
    print(f"  the core comes off cleanly in {cuttable_fraction(mols, core):.0%} of them")
    physchem, atom_props = _split_fields(args.fields)
    best = None
    for alpha in _floats(args.alphas):
        fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, alpha=alpha,
                             charge_model=args.charges, param_scheme=args.param)
        mf, log = topomer_fields(mols, names, fs, core=core, labels=labels,
                                 spec=TopomerSpec(torsion=args.torsion, seed=args.seed))
        for line in log[:4]:
            print("   " + line[:150])
        keep = [i for i, f in enumerate(mf) if f is not None]
        if len(keep) < 5:
            raise SystemExit("too few molecules could be cut into substituents")
        model = CMFModel(fieldset=fs, method="krr", normalize=not args.raw_kernels)
        K = model.joint_kernel(model.prepare_training_kernels(
            AtomTable.from_molecules([mf[i] for i in keep])))
        for lam in _floats(args.lams):
            pred = cv_predict(K, yy[keep], "krr", {"lam": float(lam)}, folds=args.folds, seed=0)
            q2 = r2_score(yy[keep], pred)
            if best is None or q2 > best[0]:
                best = (q2, alpha, lam, len(keep), rmse(yy[keep], pred))
    q2, alpha, lam, kept, err = best
    print(f"Topomer model: alpha {alpha:g}, lambda {lam:g}")
    print(f"  q2 ({args.folds}-fold CV) {q2:+.3f}   RMSEcv {err:.3f}   on {kept}/{len(mols)} "
          f"molecules")
    print("  the poses are a convention, not conformations: the same substituent always lands "
          "the same way, which is what makes this comparable without any alignment")


def cmd_difference(args):
    """What one molecule's fields have that another's do not."""
    from cmfqsar.cofields import difference_volume, export_cube, make_grid
    from cmfqsar.fielddiff import (difference_features, mismatch, rank_against, tversky,
                                   tversky_similarity)
    tv = None
    if args.tversky:
        parts = [p for p in str(args.tversky).replace(" ", "").split(",") if p]
        if len(parts) != 2:
            raise SystemExit("--tversky takes two numbers, as in --tversky 0,1")
        tv = (float(parts[0]), float(parts[1]))
    ds = load_dataset(args.dataset, activity_prop=args.activity)
    fs = FieldSet.create(physchem=[f for f in args.fields.split(",") if f], alpha=args.alpha,
                         charge_model=args.charges, param_scheme=args.param)
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    names = list(ds.names)
    if args.reference not in names:
        raise SystemExit(f"no molecule called {args.reference!r}; the file has "
                         f"{', '.join(names[:5])}...")
    ref = mf[names.index(args.reference)]
    print(f"Reference: {args.reference}  ({len(ds)} molecules, fields "
          f"{', '.join(s.label for s in fs.specs)})")
    if not args.candidate:
        how = f"Tversky alpha={tv[0]:g}, beta={tv[1]:g}" if tv else "field cosine"
        print(f"  the {args.top} molecules whose fields come closest to it ({how}):")
        ranked = rank_against(ref, mf, fs, tv[0], tv[1]) if tv else rank_against(ref, mf, fs)
        for i, cos in ranked[:args.top]:
            mark = "  <- the reference itself" if names[i] == args.reference else ""
            print(f"    {cos:+6.3f}  {names[i]}{mark}")
        return
    if args.candidate not in names:
        raise SystemExit(f"no molecule called {args.candidate!r}")
    cand = mf[names.index(args.candidate)]
    grid = make_grid(np.vstack([cand.coords, ref.coords]), margin=args.margin,
                     spacing=args.spacing)
    rows = mismatch(cand, ref, fs, grid=grid)
    print(f"Candidate: {args.candidate}")
    for r in rows:
        print("   " + r.describe())
    tvrows = None
    if tv:
        tvrows = tversky(cand, ref, fs, tv[0], tv[1])
        overall = tversky_similarity(cand, ref, fs, tv[0], tv[1])
        print(f"  Tversky's index at alpha={tv[0]:g}, beta={tv[1]:g} "
              f"(all fields together: {overall:+.3f}):")
        for t in tvrows:
            print("   " + t.describe())
    feats = difference_features(cand, ref, fs, grid=grid, level=args.level,
                                max_features=args.top)
    if feats:
        print("  where they differ, strongest first:")
        for f in feats:
            print("   " + f.describe())
    if args.out:
        import csv as _csv
        with open(args.out, "w", newline="") as fh:
            w = _csv.writer(fh)
            head = ["field", "cosine", "distance", "norm_reference", "norm_candidate",
                    "missing", "added"]
            w.writerow(head + (["tversky"] if tvrows else []))
            for j, r in enumerate(rows):
                w.writerow([r.label, f"{r.cosine:.4f}", f"{r.distance:.4f}",
                            f"{r.norm_ref:.4f}", f"{r.norm_cand:.4f}",
                            f"{r.missing:.4f}", f"{r.added:.4f}"]
                           + ([f"{tvrows[j].index:.4f}"] if tvrows else []))
        print(f"  table written to {args.out}")
    if args.cube:
        for f, spec in enumerate(fs.specs):
            d = difference_volume(cand, ref, spec, f, grid)
            path = f"{args.cube}_{spec.name}.cube"
            export_cube(path, grid, d, cand)
            print(f"  {spec.label} difference written to {path}")


def cmd_pharmacophore(args):
    """Discretise the co-fields of a fitted model into typed features with tolerance spheres."""
    from cmfqsar.pharmacophore import (extract, pocket_exclusions, summary, write_csv,
                                       write_pdb)
    from cmfqsar.cofields import make_grid
    model = load_model(args.model)
    if isinstance(model, tuple):
        model = model[0]
    grid = make_grid(model.train.coords, margin=args.margin, spacing=args.spacing)
    print(f"Model {args.model}: {model.train.n_mol} training molecules, "
          f"grid {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]} at {grid.spacing:.2f} A")
    feats = extract(model, grid=grid, level=args.level, max_features=args.max_features,
                    min_radius=args.min_radius, spacing=args.spacing, margin=args.margin,
                    fields=[f for f in args.fields.split(",") if f] if args.fields else None)
    if args.structure:
        from cmfqsar.protein import read_structure
        prot = read_structure(args.structure, keep_waters=False)
        pocket = pocket_exclusions(prot, grid, max_spheres=args.pocket_spheres)
        print(f"  {prot.name}: {len(pocket)} exclusion spheres from the receptor")
        feats = feats + pocket
    print("  " + summary(feats))
    for f in feats:
        print("   " + f.describe())
    if args.out:
        write_csv(args.out, feats)
        print(f"  features written to {args.out}")
    if args.pdb:
        write_pdb(args.pdb, feats)
        print(f"  pseudo-atoms written to {args.pdb}")


def cmd_hetero(args):
    """The workflow for a series of several chemotypes, in the order the problem imposes."""
    import time
    from cmfqsar.chemotypes import (align_within_groups, chemotype_labels, evaluate,
                                    grouped_folds, place_groups, summarise_groups)
    t0 = time.time()
    ds = load_dataset(args.dataset, activity_prop=args.activity, activity_file=args.activity_file)
    labels, group_names = chemotype_labels(ds.mols, generic=args.generic_scaffolds,
                                           min_size=args.min_group)
    folds = grouped_folds(labels, max_folds=args.max_folds)
    print(f"Loaded {len(ds)} molecules from {args.dataset}")
    print("  " + summarise_groups(labels, group_names))
    print(f"  leave-one-chemotype-out folds: {[len(f) for f in folds]}")
    if len(folds) < 2:
        raise SystemExit("one chemotype only: this is a congeneric series, use 'build'")
    physchem, atom_props = _split_fields(args.fields)
    cif = cif_types_in(ds.mols, args.cif_scheme, args.cif_min) if args.cif else []
    if cif:
        print(f"  CIF atom types ({args.cif_scheme}): {', '.join(cif)}")
    fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, cif_types=cif, alpha=1.0, charge_model=args.charges,
                         cif_scheme=args.cif_scheme, param_scheme=args.param)
    if args.align_within or args.place:
        from rdkit import Chem
        ds.mols = [Chem.Mol(m) for m in ds.mols]
    if args.align_within:
        print("Aligning inside every chemotype on its largest molecule")
        log = align_within_groups(ds.mols, labels)
        bad = [l for l in log if "FAILED" in l]
        print(f"  {len(log) - len(bad)} aligned, {len(bad)} failed")
        for line in bad[:3]:
            print("   " + line)
    if args.place:
        pocket = None
        if args.place == "dock":
            from cmfqsar.complementarity import pocket_fields
            from cmfqsar.protein import fetch_pdb, find_pocket, read_structure
            path = args.structure
            if not path:
                codes = [m.GetProp("pdb") for m in ds.mols if m is not None and m.HasProp("pdb")]
                path = find_pocket(args.dataset, codes[0] if codes else None)
                if not path and codes:
                    path = fetch_pdb(codes[0])
            if not path:
                raise SystemExit("placing by docking needs a structure: give --structure")
            pocket = pocket_fields(read_structure(path, keep_waters=False), fs)
            print(f"  pocket {pocket.n_atoms} atoms from {path}")
        print(f"Placing the chemotypes relative to one another ({args.place})")
        for line in place_groups(ds.mols, labels, fs, mode=args.place,
                                 reference_group=args.reference_group, pocket=pocket,
                                 restarts=args.restarts, seed=args.seed):
            print("   " + line[:120])
    if args.out_sdf:
        write_sdf(args.out_sdf, ds.mols, ds.names, ds.activity, ds.activity_prop or "activity")
        print(f"  realigned structures written to {args.out_sdf}")

    def prog(msg, fr):
        print("\r  [%5.1f%%] %-48s" % (fr * 100, msg), end="", flush=True)

    res = evaluate(ds.mols, ds.names, ds.activity, labels, group_names, fs,
                   alphas=_floats(args.alphas), lams=_floats(args.lams),
                   max_folds=args.max_folds, normalize=not args.raw_kernels,
                   baseline_alpha=args.baseline_alpha, progress=prog)
    print()
    print(f"Aligned model (alpha {res.alpha:g}, lambda {res.lam:g}):")
    print(f"  q2 with random folds            {res.q2_random:+.3f}   <- flatters a mixed series")
    print(f"  q2 leave-one-chemotype-out      {res.q2_grouped:+.3f}   RMSE {res.rmse_grouped:.3f}")
    how = f"alpha {args.baseline_alpha:g}" if args.baseline_alpha else "atom counts, the limit"
    print(f"  alignment-free fragment model   {res.q2_baseline:+.3f}   (indicator fields, {how})")
    print(f"  alignment-free, geometry-aware  {res.q2_invariant:+.3f}   (field autocorrelation: "
          f"rotation-invariant, no superposition)")
    print(f"  verdict: {res.verdict()}")
    if res.per_group:
        print("  per chemotype, held out in turn:")
        for g in sorted(res.per_group, key=lambda d: -d["n"])[:args.top]:
            print(f"    {g['n']:4d} molecules  RMSE {g['rmse']:5.2f}  bias {g['bias']:+6.2f}  "
                  f"{g['name'][:60]}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["molecule", "activity", "chemotype", "scaffold"])
            for i, nm in enumerate(ds.names):
                g = int(labels[i])
                w.writerow([nm, f"{ds.activity[i]:.4f}", g,
                            group_names[g] if g < len(group_names) else ""])
        print(f"  chemotype assignment written to {args.out}")
    print(f"Elapsed {time.time() - t0:.1f} s")


def cmd_unalign(args):
    """Give every molecule of a structure file a random pose (control for alignment methods)."""
    import numpy as np
    from rdkit import Chem
    from cmfqsar.align import random_rigid_transform

    ds = load_dataset(args.dataset, activity_prop=args.activity, activity_file=args.activity_file)
    rng = np.random.RandomState(args.seed)
    rot = None if args.max_rotation is None or args.max_rotation < 0 else args.max_rotation
    moved = []
    w = Chem.SDWriter(args.out)
    for m, name in zip(ds.mols, ds.names):
        mm = Chem.Mol(m)
        conf = mm.GetConformer()
        X0 = np.array(conf.GetPositions(), dtype=float)
        X = random_rigid_transform(X0, rng, max_rotation=rot, max_translation=args.max_translation)
        for i, xyz in enumerate(X):
            conf.SetAtomPosition(i, [float(v) for v in xyz])
        mm.SetProp("_Name", name)
        moved.append(float(np.sqrt(((X - X0) ** 2).sum(1).mean())))
        w.write(mm)
    w.close()
    print(f"Randomised {len(moved)} molecules ("
          + ("random orientation" if rot is None else f"rotation up to {rot:g} deg")
          + f", displacement up to {args.max_translation:g} A, seed {args.seed}): "
            f"RMSD {np.mean(moved):.2f} A on average, max {np.max(moved):.2f} A")
    print(f"Written to {args.out}")


def cmd_map(args):
    """Chemical space map and activity landscape (kernel PCA + GTM) from the CMF kernels."""
    import csv as _csv
    from cmfqsar.gtm import CMFMap, predict_from_landscape

    ds = load_dataset(args.dataset, activity_prop=args.activity, activity_file=args.activity_file)
    ok = np.isfinite(ds.activity)
    idx = [i for i in range(len(ds)) if ok[i]] if args.all else [i for i in ds.train_indices if ok[i]]
    if len(idx) < 8:
        raise SystemExit("a map needs at least eight molecules with activity values")
    physchem, atom_props = _split_fields(args.fields)
    cif = cif_types_in([ds.mols[i] for i in idx], args.cif_scheme, args.cif_min) if args.cif else []
    fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, cif_types=cif, alpha=args.alpha, charge_model=args.charges,
                         cif_scheme=args.cif_scheme, param_scheme=args.param)
    print(f"{len(idx)} molecules, fields: {', '.join(s.label for s in fs.specs)}, alpha = {args.alpha:g}")
    mf = [build_fields(ds.mols[i], fs, ds.names[i]) for i in idx]
    cmap = CMFMap(n_components=args.components, grid=args.grid, rbf=args.rbf, normalize=True)
    cmap.fit(AtomTable.from_molecules(mf), fs,
             progress=lambda m_, f_: print("  " + m_, flush=True) if "iteration" not in m_ else None)
    ev = cmap.kpca.explained_variance_ratio
    ll = cmap.gtm.log_likelihood_
    print(f"kernel variance explained by the first components: {np.round(ev[:5], 3)} "
          f"(first two: {100 * ev[:2].sum():.0f} %)")
    print(f"log-likelihood {ll[0]:.0f} -> {ll[-1]:.0f} in {len(ll)} EM iterations")
    land = cmap.activity_landscape(ds.activity[idx])
    print(f"{int(np.isfinite(land).sum())} of {args.grid ** 2} nodes carry data; "
          f"landscape spans {np.nanmin(land):.2f}..{np.nanmax(land):.2f}")
    pred = predict_from_landscape(cmap.R_, land)
    good = np.isfinite(pred)
    if good.sum() > 2:
        from cmfqsar.model import r2_score, rmse
        print(f"landscape fit on the mapped molecules: R2 {r2_score(ds.activity[idx][good], pred[good]):.3f}, "
              f"RMSE {rmse(ds.activity[idx][good], pred[good]):.3f}")
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["name", "subset", "activity", "x", "y"])
            for k, i in enumerate(idx):
                w.writerow([ds.names[i], ds.subset[i], f"{ds.activity[i]:.4f}",
                            f"{cmap.coords_[k, 0]:.4f}", f"{cmap.coords_[k, 1]:.4f}"])
        print(f"Map coordinates written to {args.out}")
    if args.landscape_out:
        with open(args.landscape_out, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["x", "y", "landscape", "density"])
            dens = cmap.density()
            for k, (x, y) in enumerate(cmap.gtm.latent):
                w.writerow([f"{x:.4f}", f"{y:.4f}",
                            "" if not np.isfinite(land[k]) else f"{land[k]:.4f}", f"{dens[k]:.4f}"])
        print(f"Landscape written to {args.landscape_out}")


def main(argv=None):
    p = argparse.ArgumentParser(description="CMF 3D-QSAR command line")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a model")
    b.add_argument("dataset")
    b.add_argument("--activity", default=None, help="SD property with the activity values")
    b.add_argument("--activity-file", default=None,
                   help="text table 'name activity [set]' with the activities (and the train/test split: 1/2)")
    b.add_argument("--test-file", default=None, help="separate structure file with the test set")
    b.add_argument("--align-decoys", action="store_true",
                   help="place every decoy in the frame of the training set by maximising its field "
                        "similarity to it, so that no decoy is rejected merely for being posed badly")
    b.add_argument("--decoy-restarts", type=int, default=6,
                   help="random starting orientations per decoy (default 6)")
    b.add_argument("--decoy-conformers", type=int, default=1, metavar="N",
                   help="generate N conformations of every decoy and keep the one that poses "
                        "best, so that a decoy is not rejected for the conformation its file "
                        "happened to carry. The pose from the file is always among the "
                        "candidates, and the cost is N times the alignment")
    b.add_argument("--decoys", default=None,
                   help="structure file with background molecules (presumed inactive) used only to "
                        "evaluate a screening model: reports AUC and enrichment at the top 1/5/10 %%")
    b.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    b.add_argument("--cif", action="store_true", help="add continuous indicator fields")
    b.add_argument("--cif-scheme", default="sybyl", choices=["sybyl", "element", "element+aromatic"])
    b.add_argument("--cif-min", type=int, default=5, help="min. number of molecules containing an atom type")
    b.add_argument("--cif-types", default=None,
                   help="explicit comma-separated list of CIF atom types (overrides the --cif-min selection)")
    b.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    b.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"],
                   help="parameterisation of the hydrophobic/H-bond fields: RDKit (Crippen + built-in rules) "
                        "or the Sybyl-type increments of the original CMF scripts (tripos; -legacy reproduces "
                        "the hydrogen-counting bug of the 2013 scripts)")
    b.add_argument("--method", default="krr",
                   choices=["krr", "gp", "svr", "nusvr", "svc", "nusvc", "gpc",
                            "ocsvm", "ocmean", "ocknn", "ocgp", "ockpca"],
                   help="gpc is the Gaussian-process classifier (Laplace approximation): it "
                        "gives calibrated probabilities, a posterior variance per molecule and "
                        "the marginal likelihood, and takes its amplitude from the C grid")
    b.add_argument("--prop-missing", default="error", choices=["error", "zero"],
                   help="what to do when a molecule carries no value for a prop: field; "
                        "'error' names the molecule, 'zero' treats it as having none")
    b.add_argument("--probes", default=None, metavar="P,Q",
                   help="GRID-like probe fields: what a water, methyl, carbonyl, amine or cation "
                        "probe would gain by sitting at each point. Each atom carries several "
                        "Gaussians fitted to that attraction, so the kernel stays analytic; the "
                        "repulsive wall is left to the steric field, which already is one")
    b.add_argument("--probe-terms", type=int, default=4, metavar="K",
                   help="Gaussians per atom in a probe field (default 4). The kernel costs the "
                        "square of this, so 3 is the cheap setting and 5 the careful one")
    b.add_argument("--descriptors", default=None, metavar="A,B",
                   help="molecule properties to use as non-field descriptors, beside the fields: "
                        "one number per molecule, standardised on the training set and added to "
                        "the kernel as a block of its own. For what a field cannot represent - a "
                        "measured logP, a medium, a temperature, an energy from a quantum "
                        "calculation. A name beginning with calc: is computed instead of read: "
                        "calc:mw, calc:clogp, calc:tpsa, calc:rotatable, or any of RDKit's own "
                        "descriptors by its RDKit name (calc:NumHAcceptors, calc:FractionCSP3, "
                        "calc:BalabanJ, ...). Do not pass anything derived from the activity")
    b.add_argument("--descriptor-h", type=float, default=1.0, metavar="H",
                   help="mixing coefficient of the descriptor block (default 1, like a field); "
                        "0 switches it off without removing it from the model")
    b.add_argument("--kernel", default="fields", choices=["fields", "autocorrelation"],
                   help="where the kernel comes from: the overlap integrals of the fields, which "
                        "needs the molecules to share a frame, or the rotation-invariant "
                        "autocorrelation of those fields, which needs no alignment at all")
    b.add_argument("--ac-sigma", type=float, default=0.75, metavar="S",
                   help="autocorrelation: width in angstrom along the distance axis, which also "
                        "sets the spacing of the channels")
    b.add_argument("--ac-dmax", type=float, default=12.0, metavar="D",
                   help="autocorrelation: longest interatomic distance resolved")
    b.add_argument("--oc-k", type=int, default=5, metavar="K",
                   help="ocknn: how many nearest actives the distance is taken to")
    b.add_argument("--oc-components", type=int, default=5, metavar="Q",
                   help="ockpca: dimension of the principal subspace of the actives")
    b.add_argument("--oc-lam", type=float, default=0.1, metavar="L",
                   help="ocgp: ridge on the constant target")
    b.add_argument("--oc-score", default="mean", choices=["mean", "mean_over_sd"],
                   help="ocgp: the predictive mean, or the mean divided by its standard "
                        "deviation, which also punishes a molecule the model knows nothing about")
    b.add_argument("--task", default="regression",
                   choices=["regression", "classification", "oneclass"],
                   help="regression (default), binary classification, or one-class / novelty detection "
                        "for similarity-based virtual screening")
    b.add_argument("--threshold", type=float, default=None,
                   help="classification: activity threshold separating active from inactive")
    b.add_argument("--active-below", action="store_true",
                   help="classification: active means activity <= threshold (e.g. for IC50 in nM)")
    b.add_argument("--class-property", default=None,
                   help="classification: SD property holding the class (1/-1, active/inactive)")
    b.add_argument("--normalize", action="store_true", help="cosine-normalise field kernels")
    b.add_argument("--raw-scale", action="store_true", help="do not rescale field kernels to unit mean diagonal")
    b.add_argument("--no-bias", action="store_true", help="KRR without bias term (b = 0)")
    b.add_argument("--alphas", default="0.2,0.3,0.5,0.7,1.0,1.5,2.0,3.0")
    b.add_argument("--lams", default="1e-4,1e-3,1e-2,0.1,1,10")
    b.add_argument("--lam-search", default="grid", choices=["grid", "brent"],
                   help="brent: continuous minimisation of RMSEcv over lambda within [min, max] of --lams")
    b.add_argument("--alpha-search", default="grid", choices=["grid", "nelder-mead"],
                   help="nelder-mead: simultaneous minimisation of RMSEcv over (alpha, lambda) with R's "
                        "Nelder-Mead simplex on kernels interpolated between the --alphas grid points "
                        "(protocol of the 2015 CIF paper); --lams is then ignored")
    b.add_argument("--nm-start", default="0.25,5",
                   help="starting point alpha,lambda of the Nelder-Mead search (default: that of the R scripts)")
    b.add_argument("--Cs", default="0.1,1,10,100")
    b.add_argument("--epsilons", default="0.05,0.1,0.2")
    b.add_argument("--nus", default="0.3,0.5,0.7")
    b.add_argument("--optimize-h", action="store_true", help="optimise mixing coefficients (model-selection bias!)")
    b.add_argument("--mkl", nargs="?", const="align", default=None, choices=["align", "alignf"],
                   help="multiple kernel learning: set the mixing coefficients h_f by centred "
                        "kernel alignment, either per field ('align', default) or jointly "
                        "('alignf'); the kernel stays linear, so co-fields remain valid")
    b.add_argument("--mkl-rho", type=float, default=1.0,
                   help="shrinkage of the joint alignment towards equal weights (0 = none)")
    b.add_argument("--kernel-layer", default="linear",
                   choices=["linear", "gauss", "arccos1", "arccos2"],
                   help="nonlinear layer on top of the CMF kernel: a radial basis function in the "
                        "field Hilbert space ('gauss') or the arc-cosine kernel of an infinitely "
                        "wide network with one or two hidden layers; co-fields and atom "
                        "contributions are then unavailable")
    b.add_argument("--gammas", default="0.25,0.5,1,2,4",
                   help="bandwidth grid of the Gaussian layer in units of the inverse median "
                        "squared distance between the training molecules")
    b.add_argument("--conformations", default="single", choices=["single", "ensemble"],
                   help="how a molecule is represented: by one conformation (the default, as in "
                        "the papers) or by the weighted mean field of a conformational ensemble, "
                        "which is the fourth dimension of 4D-QSAR")
    b.add_argument("--conformers", type=int, default=0, metavar="N",
                   help="generate up to N conformations per molecule before building, keeping the "
                        "input pose as the first one")
    b.add_argument("--conformer-window", type=float, default=0.0, metavar="A",
                   help="keep only conformations within this heavy-atom RMSD of the pose the "
                        "molecule arrived in, measured where they stand, so that the ensemble "
                        "covers the flexibility around an aligned pose rather than the whole "
                        "conformational surface (0 = no window)")
    b.add_argument("--prune-rmsd", type=float, default=0.5, metavar="A",
                   help="discard a generated conformation within this heavy-atom RMSD of a better one")
    b.add_argument("--conformer-align", default="self",
                   choices=["self", "core", "reference", "field"],
                   help="where the generated conformations are put, which is what the kernel "
                        "compares: 'self' superimposes them on the molecule's own input pose "
                        "(the default, as before), 'core' does the same but only over the "
                        "scaffold shared with the reference, so the published pose of the "
                        "scaffold survives and only the flexible part moves, 'reference' "
                        "superimposes every conformation on one reference molecule of the series "
                        "through their common substructure, and 'field' refines that by "
                        "maximising the CMF kernel with the reference")
    b.add_argument("--conformer-align-ref", type=int, default=-1, metavar="I",
                   help="molecule to align the conformations to (0-based index in the data set); "
                        "the default takes the largest molecule, whose common substructure with "
                        "the others is the most likely to be the scaffold of the series")
    b.add_argument("--conformer-weighting", default="uniform",
                   choices=["uniform", "boltzmann", "property"],
                   help="weights of the conformations in the mean field")
    b.add_argument("--kt", type=float, default=0.6, metavar="KCAL",
                   help="kT for the Boltzmann weights (default 0.6 kcal/mol, about RT at 300 K)")
    b.add_argument("--max-conformers", type=int, default=0, metavar="N",
                   help="use at most the N heaviest conformations of each molecule (0 = all)")
    b.add_argument("--ensemble-by", default=None, metavar="KEY",
                   help="group the records of a multi-conformer file into ensembles, by 'name' or "
                        "by the named property")
    b.add_argument("--folds", type=int, default=10)
    b.add_argument("--cv-no-shuffle", action="store_true",
                   help="interleaved folds in file order (molecule i -> fold i mod k) instead of shuffled folds")
    b.add_argument("--test-fraction", type=float, default=0.0)
    b.add_argument("--align-train-variance", type=float, nargs="?", const=1.0, default=None,
                   metavar="MAX_RMSD",
                   help="before the final fit, realign the TRAINING set by minimising the posterior "
                        "variance, cross-validated so that no molecule is aligned against a model that "
                        "contains it; the value is the displacement limit in A (default 1.0, <=0 none). "
                        "The gain is not reliable - judge it on a test set, not on q2")
    b.add_argument("--align-train-rounds", type=int, default=1)
    b.add_argument("--align-train-folds", type=int, default=5)
    b.add_argument("--y-scramble", type=int, default=0, metavar="N",
                   help="y-randomisation: refit the model on N shuffles of the activities and "
                        "report where the real q2 falls among them. 20 is enough to see a "
                        "chance model, 100 for a p-value worth quoting")
    b.add_argument("--y-scramble-select", action="store_true",
                   help="repeat the whole search inside every shuffle instead of keeping the "
                        "chosen hyper-parameters, so that y-randomisation tests the procedure "
                        "and not only the fit. One cross-validation per grid point per shuffle; "
                        "the kernels are computed once because they do not depend on the "
                        "activities")
    b.add_argument("--bootstrap", type=int, default=0, metavar="N",
                   help="percentile confidence intervals for q2 and R2pred from N bootstrap "
                        "resamples of the (observed, predicted) pairs")
    b.add_argument("--bootstrap-refit", type=int, default=0, metavar="N",
                   help="Cramer's bootstrap as well: N resamples of the training set, each "
                        "refitted and scored on the molecules left out of the bag")
    b.add_argument("--domain", action="store_true",
                   help="report the applicability domain of the molecules outside the training "
                        "set: their similarity to it, against a threshold taken from the "
                        "training set's own spread")
    b.add_argument("--domain-k", type=int, default=3, metavar="K",
                   help="how many nearest training molecules the similarity is averaged over")
    b.add_argument("--domain-quantile", type=float, default=0.05, metavar="Q",
                   help="the quantile of the training set's own similarities that becomes the "
                        "threshold (default 0.05)")
    b.add_argument("--nested", action="store_true", help="external (nested) cross-validation")
    b.add_argument("--nested-on", default="train", choices=["train", "all"],
                   help="run the external CV on the training set or on all molecules (train + test)")
    b.add_argument("--outer-folds", type=int, default=5)
    b.add_argument("--repeats", type=int, default=1)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--save", default=None)
    b.add_argument("--predictions", default=None)
    b.add_argument("--json", default=None, help="write all statistics and predictions to a JSON file")
    b.set_defaults(func=cmd_build)
    q = sub.add_parser("predict", help="predict with a saved model")
    q.add_argument("model")
    q.add_argument("dataset")
    q.add_argument("--activity", default=None)
    q.add_argument("--activity-file", default=None)
    q.add_argument("--out", default=None)
    q.add_argument("--align-variance", type=float, nargs="?", const=2.0, default=None, metavar="MAX_RMSD",
                   help="before predicting, align every molecule by minimising the posterior variance "
                        "of its prediction (Baskin & Zhokhova 2019); the optional value is the maximum "
                        "displacement in A from the input pose (default 2.0, <=0 for no limit)")
    q.add_argument("--align-restarts", type=int, default=2,
                   help="random restarts of the alignment search (default 2)")
    q.add_argument("--align-flexible", action="store_true",
                   help="turn the rotatable bonds as well as moving the molecule, with the force "
                        "field as a prior on the conformation (the 2019 proposal in full); the "
                        "displacement limit then applies to the rigid motion only, since the "
                        "strain prior is what governs the conformation")
    q.add_argument("--align-conformers", type=int, default=5, metavar="N",
                   help="with --align-flexible, start the search from N generated conformations "
                        "besides the input one (default 5, 0 for the input one alone); a torsion "
                        "search is local, and on the FEP benchmark these are what remove the "
                        "conformational part of the error -- 0.86 A against 1.41 A median, and "
                        "better on 15 of 16 sets")
    q.add_argument("--align-energy-slack", type=float, default=10.0, metavar="KCAL",
                   help="with --align-flexible, how far the conformation may rise above the one "
                        "it started in before the strain prior penalises it (default 10)")
    q.add_argument("--align-energy-weight", type=float, default=1.0,
                   help="with --align-flexible, the weight of that penalty (0 switches it off)")
    q.add_argument("--align-max-torsions", type=int, default=12,
                   help="with --align-flexible, the largest number of rotatable bonds to search")
    q.add_argument("--aligned-out", default=None, help="write the aligned structures to this SDF")
    q.add_argument("--allow-mismatch", action="store_true",
                   help="predict even when the molecules were not prepared the way the model's "
                        "training set was (the check that otherwise refuses is printed either way)")
    q.set_defaults(func=cmd_predict)
    u = sub.add_parser("unalign", help="give every molecule a random pose (control experiment)")
    u.add_argument("dataset")
    u.add_argument("out", help="output SDF")
    u.add_argument("--activity", default=None)
    u.add_argument("--activity-file", default=None)
    u.add_argument("--max-rotation", type=float, default=None, metavar="DEG",
                   help="maximum rotation angle in degrees; omitted means a uniformly random "
                        "orientation, which destroys the alignment completely")
    u.add_argument("--max-translation", type=float, default=2.0, metavar="A",
                   help="maximum displacement in A (default 2.0)")
    u.add_argument("--seed", type=int, default=0)
    u.set_defaults(func=cmd_unalign)
    g = sub.add_parser("map", help="chemical space map and activity landscape (kernel PCA + GTM)")
    g.add_argument("dataset")
    g.add_argument("--activity", default=None)
    g.add_argument("--activity-file", default=None)
    g.add_argument("--all", action="store_true", help="map every molecule, not only the training set")
    g.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    g.add_argument("--cif", action="store_true")
    g.add_argument("--cif-scheme", default="sybyl", choices=["sybyl", "element", "element+aromatic"])
    g.add_argument("--cif-min", type=int, default=5)
    g.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    g.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"])
    g.add_argument("--alpha", type=float, default=0.5,
                   help="attenuation factor; a small value (0.02) with --cif gives an "
                        "alignment-free map, because CIFs then tend to atom counts")
    g.add_argument("--grid", type=int, default=20, help="map nodes per side (default 20)")
    g.add_argument("--rbf", type=int, default=5, help="radial basis functions per side (default 5)")
    g.add_argument("--components", type=int, default=15,
                   help="functional principal components kept before mapping (default 15)")
    g.add_argument("--out", default=None, help="CSV with the map coordinates of every molecule")
    g.add_argument("--landscape-out", default=None, help="CSV with the landscape on the node grid")
    g.set_defaults(func=cmd_map)

    e = sub.add_parser("explain", help="Shapley decomposition of a prediction (any kernel)")
    e.add_argument("--allow-mismatch", action="store_true",
                   help="go on when the molecules were not prepared the way the model's training set was")
    e.add_argument("model")
    e.add_argument("dataset")
    e.add_argument("--activity", default=None)
    e.add_argument("--molecule", default=None,
                   help="name of the molecule to explain, or several separated by commas "
                        "(default: every molecule of the file)")
    e.add_argument("--players", default="atoms",
                   choices=["atoms", "groups", "fields", "atomfields"],
                   help="what the prediction is divided between: atoms, the substituents of the "
                        "common scaffold of the file, the field types, or atoms crossed with "
                        "field types, which is the basis of a Shapley field")
    e.add_argument("--cube", default=None, metavar="PREFIX",
                   help="with --players atomfields, write one Gaussian cube per field type, "
                        "named <PREFIX>_<molecule>_<field>.cube")
    e.add_argument("--cube-spacing", type=float, default=0.5, metavar="A")
    e.add_argument("--samples", type=int, default=400,
                   help="permutations sampled when the players are too many to enumerate")
    e.add_argument("--interactions", action="store_true",
                   help="also report second-order Shapley interaction indices")
    e.add_argument("--top", type=int, default=15, help="rows printed per molecule")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None, help="CSV with all values")
    e.set_defaults(func=cmd_explain)

    i = sub.add_parser("mil", help="multi-instance learning over conformations (4D, selection)")
    i.add_argument("dataset")
    i.add_argument("--activity", default=None)
    i.add_argument("--activity-file", default=None)
    i.add_argument("--ensemble-by", default=None, metavar="KEY",
                   help="group the records of a multi-conformer file into bags, by 'name' or by "
                        "the named property")
    i.add_argument("--conformers", type=int, default=0, metavar="N",
                   help="generate up to N conformations per molecule first, keeping the input "
                        "pose as the first one")
    i.add_argument("--prune-rmsd", type=float, default=0.5, metavar="A")
    i.add_argument("--conformer-align", default="self",
                   choices=["self", "core", "reference", "field"],
                   help="where the generated conformations are placed")
    i.add_argument("--conformer-window", type=float, default=0.0, metavar="A",
                   help="keep only conformations within this distance of the arrival pose")
    i.add_argument("--max-conformers", type=int, default=0, metavar="N",
                   help="use at most the first N conformations of every molecule")
    i.add_argument("--dock", action="store_true",
                   help="place every conformation in the pocket by rigid-body optimisation "
                        "before learning, so that the bag is bound-like rather than gas-phase")
    i.add_argument("--structure", default=None,
                   help="PDB or mmCIF file of the protein; by default the pocket stored beside "
                        "the data set is used, or the entry it names is fetched")
    i.add_argument("--dock-restarts", type=int, default=1, metavar="N",
                   help="bounded random restarts per conformation (0 = optimise from where it is)")
    i.add_argument("--dock-keep", type=int, default=0, metavar="N",
                   help="keep only the N best-scoring poses of each molecule (0 = all)")
    i.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    i.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    i.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"])
    i.add_argument("--raw-kernels", action="store_true",
                   help="do not cosine-normalise the conformation kernels")
    i.add_argument("--alphas", default="0.3,0.5,1.0")
    i.add_argument("--lams", default="1e-2,0.1,1")
    i.add_argument("--betas", default="1,3,10",
                   help="hardness of the weighting; 0, the plain ensemble, is always included "
                        "and a large value approaches choosing one conformation")
    i.add_argument("--hard", action="store_true",
                   help="also try hard selection: exactly one conformation per molecule")
    i.add_argument("--init", default="first", choices=["first", "uniform"],
                   help="where the alternation starts: the arrival pose, or equal weights")
    i.add_argument("--rounds", type=int, default=10,
                   help="maximum rounds of fit and re-weight")
    i.add_argument("--folds", type=int, default=10)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--out", default=None,
                   help="CSV with the chosen conformation of every training molecule")
    i.set_defaults(func=cmd_mil)

    h = sub.add_parser("hetero", help="series of several chemotypes: cluster, align, place, judge")
    h.add_argument("dataset")
    h.add_argument("--activity", default=None)
    h.add_argument("--activity-file", default=None)
    h.add_argument("--generic-scaffolds", action="store_true",
                   help="group by the generic framework, ignoring elements and bond orders, "
                        "which merges chemotypes that differ only in heteroatoms")
    h.add_argument("--min-group", type=int, default=3, metavar="N",
                   help="scaffolds with fewer than N molecules are pooled into one group")
    h.add_argument("--max-folds", type=int, default=0, metavar="N",
                   help="bundle the chemotypes into N folds instead of holding out one at a time")
    h.add_argument("--align-within", action="store_true",
                   help="superimpose the molecules of each chemotype on its largest member")
    h.add_argument("--place", default=None, choices=["field", "dock"],
                   help="place the chemotypes relative to one another, each moved as a rigid "
                        "body: by field similarity with the reference chemotype, or by docking "
                        "its reference into the binding site")
    h.add_argument("--reference-group", type=int, default=0, metavar="G",
                   help="which chemotype stays still when placing by field similarity")
    h.add_argument("--structure", default=None, help="protein structure for --place dock")
    h.add_argument("--restarts", type=int, default=2)
    h.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd",
                   help="physico-chemical fields; the hydrophobic and hydrogen-bond ones are "
                        "atomic increments fitted to logP and Abraham's constants, so they are "
                        "property fields. Pass an empty list to compare with --cif alone")
    h.add_argument("--cif", action="store_true",
                   help="also use continuous indicator fields, one per atom type")
    h.add_argument("--cif-min", type=int, default=5, metavar="N",
                   help="an indicator field needs atoms of its type in at least N molecules")
    h.add_argument("--cif-scheme", default="sybyl",
                   choices=["sybyl", "element", "element+aromatic"])
    h.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    h.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"])
    h.add_argument("--raw-kernels", action="store_true")
    h.add_argument("--alphas", default="0.3,0.5,1.0")
    h.add_argument("--lams", default="1e-2,0.1,1")
    h.add_argument("--baseline-alpha", type=float, default=None, metavar="A",
                   help="attenuation of the alignment-free indicator fields; omitted, the "
                        "baseline is the limit these fields tend to, taken from the atom counts "
                        "themselves, which is exactly pose-free and much faster. A value takes "
                        "the Gaussian route at that attenuation instead, which has to be far "
                        "smaller than a modelling one, since the dependence on the poses dies "
                        f"only linearly in it ({BASELINE_ALPHA:g} is close enough)")
    h.add_argument("--top", type=int, default=10)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--out", default=None, help="CSV with the chemotype of every molecule")
    h.add_argument("--out-sdf", default=None, help="write the realigned structures")
    h.set_defaults(func=cmd_hetero)

    ph = sub.add_parser("pharmacophore",
                        help="read a pharmacophore out of the co-fields of a fitted model")
    ph.add_argument("model", help="a model saved with 'build --save'")
    ph.add_argument("--level", type=float, default=0.5, metavar="F",
                    help="a feature is a region where the co-field exceeds this fraction of its "
                         "largest value; larger means tighter and fewer features")
    ph.add_argument("--max-features", type=int, default=12, metavar="N",
                    help="how many favourable features to keep, strongest first (exclusion "
                         "volumes are counted separately)")
    ph.add_argument("--min-radius", type=float, default=0.6, metavar="R",
                    help="ignore lobes smaller than this tolerance radius")
    ph.add_argument("--fields", default=None,
                    help="restrict to these field types (default: every enabled field)")
    ph.add_argument("--spacing", type=float, default=0.6)
    ph.add_argument("--margin", type=float, default=3.0)
    ph.add_argument("--structure", default=None,
                    help="a protein structure; its atoms inside the box become exclusion volumes "
                         "of the harder kind, the ones a ligand-only model cannot know")
    ph.add_argument("--pocket-spheres", type=int, default=30, metavar="N")
    ph.add_argument("--out", default=None, help="write the features as CSV")
    ph.add_argument("--pdb", default=None, help="write the features as pseudo-atoms in PDB")
    ph.set_defaults(func=cmd_pharmacophore)

    df = sub.add_parser("difference",
                        help="what one molecule's fields have that another's do not")
    df.add_argument("dataset")
    df.add_argument("--reference", required=True, metavar="NAME",
                    help="the molecule to be imitated")
    df.add_argument("--candidate", default=None, metavar="NAME",
                    help="the molecule to compare with it; without this the whole file is "
                         "ranked by how closely its fields match the reference")
    df.add_argument("--activity", default=None)
    df.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    df.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    df.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"])
    df.add_argument("--alpha", type=float, default=0.5)
    df.add_argument("--level", type=float, default=0.5,
                    help="contour bounding a lobe of the difference map, as a fraction of its "
                         "largest value")
    df.add_argument("--spacing", type=float, default=0.5)
    df.add_argument("--margin", type=float, default=3.0)
    df.add_argument("--top", type=int, default=10)
    df.add_argument("--tversky", default=None, metavar="ALPHA,BETA",
                    help="also report Tversky's index with these weights, and rank by it: "
                         "ALPHA charges the candidate for what the reference has and it lacks, "
                         "BETA for what it adds.  1,1 is Tanimoto, 0.5,0.5 Dice, and 0,1 asks "
                         "how much of the candidate the reference accounts for, which is the "
                         "setting for a library of molecules smaller than the reference")
    df.add_argument("--out", default=None, help="write the per-field table as CSV")
    df.add_argument("--cube", default=None, metavar="PREFIX",
                    help="write the difference map of every field as a Gaussian cube")
    df.set_defaults(func=cmd_difference)

    tp = sub.add_parser("topomer",
                        help="model a series by its substituents, each posed by rule "
                             "(topomer CMF, after Cramer's topomer CoMFA)")
    tp.add_argument("dataset")
    tp.add_argument("--activity", default=None)
    tp.add_argument("--activity-file", default=None)
    tp.add_argument("--core", default=None, metavar="SMILES",
                    help="the scaffold to cut the substituents from; without it the maximum "
                         "common substructure of the series is used")
    tp.add_argument("--core-smarts", default=None, metavar="SMARTS")
    tp.add_argument("--core-mode", default="framework", choices=["framework", "mcs"],
                    help="how to find the core when none is given: 'framework' (the default) "
                         "takes the Bemis-Murcko framework of the largest chemotype, which is "
                         "the scaffold of one series; 'mcs' takes the maximum common "
                         "substructure of everything, which suits a congeneric set and fails on "
                         "a mixture of chemotypes - on GPB it is CCO and cuts nothing off")
    tp.add_argument("--torsion", type=float, default=180.0,
                    help="the angle every rotatable torsion is set to (Cramer's anti rule)")
    tp.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    tp.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    tp.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"])
    tp.add_argument("--alphas", default="0.3,0.5,1.0")
    tp.add_argument("--lams", default="1e-2,0.1,1")
    tp.add_argument("--folds", type=int, default=10)
    tp.add_argument("--raw-kernels", action="store_true")
    tp.add_argument("--seed", type=int, default=0xC0FFEE)
    tp.set_defaults(func=cmd_topomer)

    k = sub.add_parser("complement", help="field complementarity with a protein pocket")
    k.add_argument("dataset")
    k.add_argument("--structure", default=None,
                   help="PDB or mmCIF file of the protein; by default the pocket stored beside "
                        "the data set is used, or the entry it names is fetched")
    k.add_argument("--model", default=None,
                   help="saved model whose co-fields are compared with what the pocket offers")
    k.add_argument("--activity", default=None)
    k.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    k.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    k.add_argument("--alpha", type=float, default=0.5)
    k.add_argument("--spacing", type=float, default=0.6, metavar="A",
                   help="grid spacing for the co-field comparison")
    k.add_argument("--statistic", default="sign", choices=["sign", "cosine", "pearson"],
                   help="how a co-field is compared with the pocket (default: sign agreement)")
    k.add_argument("--no-surface", action="store_true",
                   help="skip the surface scores, which are the slow part")
    k.add_argument("--top", type=int, default=15, help="molecules printed")
    k.add_argument("--residues", action="store_true",
                   help="resolve every score by receptor residue: the interaction profile of "
                        "each molecule, the pattern conserved across the series, and the "
                        "similarity of each molecule to that pattern")
    k.add_argument("--reference", default=None, metavar="NAME",
                   help="with --residues, compare every profile with this molecule's instead "
                        "of with the consensus of the series")
    k.add_argument("--residue-threshold", type=float, default=0.02, metavar="X",
                   help="a term above this counts as an interaction when a profile is reduced "
                        "to bits for the consensus and the Tanimoto comparison")
    k.add_argument("--out", default=None, help="CSV with every score")
    k.add_argument("--profile-out", default=None, metavar="FILE",
                   help="CSV of the interaction profiles: one row per molecule and residue")
    k.set_defaults(func=cmd_complement)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
