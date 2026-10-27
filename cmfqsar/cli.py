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
import os
import sys
import time

import numpy as np

from cmfqsar.dataio import load_dataset, save_model, load_model, write_predictions_csv
from cmfqsar.fields import AtomTable, FieldSet, build_fields, cif_types_in
from cmfqsar.chemotypes import BASELINE_ALPHA
from cmfqsar.ranking import make_pairs
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


def _block_weights(text):
    """--block-weights: None, 'balanced', 'align', 'alignf', or 'KIND=W,KIND=W' with a kind a
    group name ('physchem', 'indicator', or a group of --block-groups) or two joined by '*'."""
    if text is None or not str(text).strip():
        return None
    text = str(text).strip()
    if text in ("balanced", "align", "alignf"):
        return text
    out = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise SystemExit(f"--block-weights: expected KIND=W, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _block_groups(text, fieldset):
    """--block-groups 'GROUP=field,field;GROUP=cif:*': field name -> group, patterns allowed;
    fields not named keep the default grouping (indicator fields against the rest)."""
    if text is None or not str(text).strip():
        return None
    import fnmatch
    names = [sp.name for sp in fieldset.specs]
    out = {}
    for part in str(text).split(";"):
        if not part.strip():
            continue
        if "=" not in part:
            raise SystemExit(f"--block-groups: expected GROUP=field,field, got {part!r}")
        group, pats = part.split("=", 1)
        for pat in pats.split(","):
            pat = pat.strip()
            hits = [n for n in names if fnmatch.fnmatchcase(n, pat)]
            if not hits:
                raise SystemExit(f"--block-groups: no field matches {pat!r} (fields: "
                                 f"{', '.join(names)})")
            for n in hits:
                out[n] = group.strip()
    return out


def _print_block_weights(model):
    rows = model.block_weight_report()
    if not rows:
        return
    print("Block weights (mixing coefficients of the channel blocks, by kind of block):")
    for r in rows:
        print(f"  {r['kind']:28s} {r['n_blocks']:5d} blocks  weight {r['mean']:.3f} "
              f"[{r['min']:.3f}, {r['max']:.3f}]  share of the kernel {r['share']:.2f}")


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
    The results are printed and stored in ``model.stats.extra`` so that --json carries them.
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
            # the loop below goes through the alpha grid, which moves the field set off the alpha the
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
    pharm = []
    if getattr(args, "pharm", False):
        from cmfqsar.pharmtypes import coverage, definitions_used, features_in
        pharm = features_in([ds.mols[i] for i in tr], args.pharm_min)
        if args.pharm_features:
            pharm = [p for p in args.pharm_features.split(",") if p]
        cov = coverage([ds.mols[i] for i in tr])
        print(f"  pharmacophore features ({definitions_used()}): "
              + ", ".join(f"{p} ({cov[p]['molecules']:.0%} of the molecules)" for p in pharm))
    fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, cif_types=cif,
                         pharm_features=pharm, alpha=1.0, charge_model=args.charges,
                         prop_missing=args.prop_missing,
                         cif_scheme=args.cif_scheme, param_scheme=args.param,
                         conformers=args.conformations,
                         conformer_weighting=args.conformer_weighting,
                         conformer_kt=args.kt, max_conformers=args.max_conformers,
                         descriptors=[d for d in (args.descriptors or "").split(",") if d],
                         descriptor_h=args.descriptor_h,
                         probes=[p for p in (args.probes or "").split(",") if p],
                         probe_terms=args.probe_terms,
                         protonation_ph=args.ph, protonation_states=args.ph_states,
                         protonation_min=args.ph_min_population,
                         tautomers=(args.tautomers or ""), tautomer_forms=args.tautomer_forms)
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
    if args.increments or args.learn_increments:
        from cmfqsar.learnincrements import (coverage, fit_increments, read_increments,
                                             types_in)
        if not fs.param_scheme.startswith("tripos"):
            raise SystemExit("fitted increments only apply to the Tripos parameterisation: "
                             "add --param tripos")
        legacy = fs.param_scheme == "tripos-legacy"
        if args.increments:
            fs.increments = read_increments(args.increments)
            cov = coverage(fs.increments, ds.mols, legacy=legacy)
            print(f"  increments read from {args.increments}: {len(fs.increments)} types, "
                  f"{cov['exact']:.0%} of the atoms matched by their own type, "
                  f"{cov['other']:.0%} by the 'other' group, {cov['missing']:.0%} not covered")
        else:
            keys, counts = types_in([ds.mols[i] for i in tr], legacy=legacy)
            print(f"  fitting the increments of {len(keys)} atom types on the training set "
                  f"(alpha {args.learn_increments_alpha:g}, lambda "
                  f"{args.learn_increments_lam:g}, penalty {args.learn_increments_rho:g})")
            fit = fit_increments([ds.mols[i] for i in tr], [ds.names[i] for i in tr],
                                 ds.activity[tr], alpha=args.learn_increments_alpha,
                                 lam=args.learn_increments_lam, rho=args.learn_increments_rho,
                                 folds=args.folds, seed=args.seed, keys=keys, fieldset=fs,
                                 legacy=legacy)
            fs.increments = fit.increments
            cos = fit.cosine()
            print(f"    cross-validated squared error {fit.loss_published:.3f} -> {fit.loss:.3f} "
                  f"in {fit.iterations} iterations; the fitted increments still point the same "
                  "way as the published ones for "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in cos.items()))
            print("    WARNING: the increments were fitted on the same cross-validation that "
                  "reports q2 below, so that q2 is optimistic. Use a test set, or --nested, or "
                  "the numbers in the manual for how large the difference is.")
            if args.save_increments:
                fit.to_json(args.save_increments)
                print(f"    written to {args.save_increments}")
    if fs.is_tautomer_ensemble:
        from cmfqsar.tautomers import ambiguous_share, available, forms_table
        if not available():
            raise SystemExit("--tautomers needs RDKit's tautomer enumerator, which this RDKit "
                             "build does not have")
        forms = forms_table([ds.mols[i] for i in tr], weighting=args.tautomers,
                            max_forms=args.tautomer_forms)
        print(f"  tautomer ensembles weighted '{args.tautomers}': "
              f"{np.mean([len(e) for e in forms]):.1f} forms per molecule on average, "
              f"{ambiguous_share(forms):.0%} of the training molecules with more than one form")
        print("    the weights are not populations: a tautomer has one only if the free energies "
              "are known, and none are computed here")
    if fs.is_protonation_ensemble:
        from cmfqsar.protonation import available, ensemble_table, mixture_share
        if not available():
            raise SystemExit("--ph needs Dimorphite-DL for the pKa values and the sites: "
                             "pip install dimorphite-dl")
        ens = ensemble_table([ds.mols[i] for i in tr], [ds.names[i] for i in tr], ph=args.ph,
                             min_population=args.ph_min_population, max_states=args.ph_states)
        share = mixture_share(ens)
        print(f"  protonation ensembles at pH {args.ph:g}: "
              f"{np.mean([len(e) for e in ens]):.1f} states per molecule on average, "
              f"{share:.0%} of the training molecules a real mixture, mean charge "
              f"{np.mean([e.mean_charge for e in ens]):+.2f}")
        if share == 0.0:
            print("    every molecule is in one state at this pH, so this is the single-state "
                  "model with the states chosen by pKa")
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
            print(f"  a generated conformation posed better than the one in the file for {moved} of "
                  f"{len(res)} decoys")
        for m, mol in zip([ds.mols[i] for i in dec_idx], [mf[i] for i in dec_idx]):
            conf = m.GetConformer()
            for a, xyz in enumerate(mol.coords):
                conf.SetAtomPosition(a, [float(v) for v in xyz])
    y = target[tr]
    model = CMFModel(fieldset=fs, task=args.task, method=args.method, normalize=args.normalize,
                     scale_fields=not args.raw_scale, fit_bias=not args.no_bias,
                     kernel_layer=args.kernel_layer, kernel_source=args.kernel,
                     ac_sigma=args.ac_sigma, ac_d_max=args.ac_dmax,
                     ac_chirality=args.ac_chirality,
                     ac_chirality_shells=args.ac_chirality_shells,
                     surf_z_min=args.surface_z[0], surf_z_max=args.surface_z[1],
                     surf_z_step=args.surface_z[2],
                     ac_block_weights=_block_weights(args.block_weights),
                     ac_block_groups=_block_groups(args.block_groups, fs),
                     ac_block_rho=args.block_rho)
    if args.method == "ranksvm":
        model.params["gap"] = float(args.rank_gap)
        model.params["max_pairs"] = int(args.rank_pairs)
        if args.rank_groups:
            g = [m.GetProp(args.rank_groups) if m.HasProp(args.rank_groups) else ""
                 for m in ds.mols]
            if len(set(g)) < 2:
                raise SystemExit(f"--rank-groups {args.rank_groups!r}: every molecule has the "
                                 f"same value, so grouping changes nothing")
            model.rank_groups = np.asarray(g, dtype=object)[tr]
            print(f"Ranking: pairs formed inside each of {len(set(g))} groups of "
                  f"{args.rank_groups!r}, never across them")
        n_pairs = len(make_pairs(target[tr], gap=float(args.rank_gap),
                                 groups=getattr(model, "rank_groups", None),
                                 max_pairs=int(args.rank_pairs)))
        print(f"Ranking SVM: {n_pairs} pairs"
              + (f" after dropping those closer than {args.rank_gap:g}" if args.rank_gap else "")
              + ". The score has no units, so the model is judged by rho, tau and the share of "
                "pairs it orders correctly, and r2 is not reported.")
    if model.ac_block_weights is not None and model.uses_fields:
        raise SystemExit("--block-weights belongs to the invariant kernels (--kernel "
                         "autocorrelation or surface); the overlap kernel mixes its fields with "
                         "--mkl or --optimize-h")
    if args.kernel == "surface":
        print(f"Kernel: surface autocorrelation, in-plane distances up to {args.ac_dmax:g} A in "
              f"steps of {args.ac_sigma:g} A, heights {args.surface_z[0]:g}-{args.surface_z[1]:g} A "
              f"in steps of {args.surface_z[2]:g} A. The surface is z = 0 with the normal along "
              f"+z; no in-plane alignment is assumed, and there are no co-fields.")
    if args.kernel == "autocorrelation":
        print(f"Kernel: field autocorrelation up to {args.ac_dmax:g} A in steps of "
              f"{args.ac_sigma:g} A"
              + (f", with the signed three-point channels over {args.ac_chirality_shells} radial "
                 f"shells, so enantiomers are distinguishable" if args.ac_chirality else "")
              + ", invariant to rotation and translation. No alignment is "
              f"assumed and the attenuation grid is not used. There is no co-field, because "
              f"there is no frame to draw one in, but the prediction still splits over the atoms "
              f"of a molecule, because every channel is a sum over atom pairs (plan item SC6): "
              f"`report` lists the largest contributions and the viewer colours the atoms by "
              f"them.")
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
                             "oc_lam": args.oc_lam, "oc_score": args.oc_score,
                             "fusion_threshold": args.fusion_threshold, "rrf_k": args.rrf_k})
        if args.method.startswith("fuse-"):
            from cmfqsar.fusion import LABELS as FUSION_LABELS
            rule = args.method[len("fuse-"):]
            print(f"  description of the actives: {FUSION_LABELS[rule]}")
            if rule in ("anz", "mnz") and args.fusion_threshold == 0.0:
                print("  warning: at a threshold of 0 this rule counts every active for every "
                      "molecule, so it collapses onto fuse-sum; set --fusion-threshold")
        elif args.method != "ocsvm":
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
                      lam_search=args.lam_search, select=args.select,
                      alpha_search=args.alpha_search, nm_start=nm_start)
    print()
    knob = (f"sigma = {args.ac_sigma:g} A" if args.kernel == "autocorrelation"
            else f"alpha = {res.best_alpha:g}")
    if res.best_alphas is not None:
        knob = "alpha = " + ", ".join(f"{n} {a:g}" for n, a in res.best_alphas.items())
    if args.task == "classification":
        print(f"Best: {knob}, {res.best_params}, "
              f"1-AUCcv = {res.best_rmse:.3f}, balanced accuracy = {res.best_q2:.3f}")
    elif args.select == "evidence":
        # the search minimised the negative log evidence, so that is what to report
        print(f"Best: {knob}, {res.best_params}, log evidence = {res.best_q2:.1f} "
              f"(chosen without holding anything out)")
        pf = res.per_field
        if pf is not None:
            # what the extra freedom gained, in the units the descent was scored in
            print(f"Per-field widths: {pf.gain:+.1f} in log evidence over the shared "
                  f"alpha = {pf.shared_alpha:g}, in {pf.n_eval} factorisations"
                  + (" (the descent came back to one shared width)" if pf.is_shared else "")
                  + (f"; {', '.join(pf.skipped)} left alone, no width to search"
                     if pf.skipped else ""))
    elif args.method == "ranksvm":
        # the search minimises 1 - rho for a ranking model, so that is what to call it
        print(f"Best: {knob}, {res.best_params}, rho(cv) = {res.best_q2:+.3f}")
    else:
        print(f"Best: {knob}, {res.best_params}, "
              f"RMSEcv = {res.best_rmse:.3f}, q2 = {res.best_q2:.3f}")
    if args.optimize_h or args.mkl:
        print("  mixing coefficients:", {k: round(v, 3) for k, v in res.best_h.items()})
    model.fit(train, y, cv_folds=args.folds, cv_seed=seed, reuse_kernels=True)
    second = None
    if args.second_endpoint:
        second = _fit_second_endpoint(model, ds, tr, train, y, args)
    if args.method == "gpc":
        from cmfqsar.gpc import brier_score
        p_fit = model.predict_proba(train)
        print(f"  log marginal likelihood {model.log_marginal_likelihood:.2f}, "
              f"Brier score of the fitted probabilities {brier_score(y, p_fit):.3f} "
              f"(0.25 is no better than chance)")
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
    _print_block_weights(model)
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
            print(f"  constant over the training set, so without effect: "
                  f"{', '.join(model.desc_scaler_.constant)}")
    _report_validation(args, model, ds, mf, tr, te, target)
    conf = None
    if args.conformal is not None:
        from cmfqsar.conformal import coverage as _coverage
        try:
            conf = model.calibrate_conformal(level=float(args.conformal),
                                             normalise=bool(args.conformal_normalise))
        except ValueError as exc:
            raise SystemExit(f"--conformal: {exc}")
        print(conf.describe())
        if te and "y_test_pred" in model.stats.extra:
            # the test set was never part of the calibration, so this is the reliable check
            yp_te = np.asarray(model.stats.extra["y_test_pred"], float)
            d_te = None
            if conf.difficulty is not None:
                from cmfqsar.conformal import difficulty_from_kernel
                K_train = model._linear_train_kernel()
                table_te = AtomTable.from_molecules([mf[i] for i in te])
                d_te = difficulty_from_kernel(model.test_kernel(table_te),
                                              model.joint_self(table_te), K_train)
            lo, hi = conf.interval(yp_te, d_te)
            got = _coverage(ds.activity[te], lo, hi)
            print(f"  on the {got['n']} test molecules it covered {got['coverage']:.0%}, "
                  f"mean width {got['mean_width']:.3f}")
    if args.error_predictors:
        from cmfqsar.errorpred import compare as _ep_compare, estimates_for, table as _ep_table
        if not te or "y_test_pred" not in model.stats.extra:
            print("--error-predictors needs a test set: what the estimates are scored against is "
                  "the error on molecules the model never saw")
        else:
            table_te = AtomTable.from_molecules([mf[i] for i in te])
            est = estimates_for(model, table_te)
            print()
            print(f"Error predictors on the {len(te)} test molecules. Efficiency is 1 for the "
                  f"ordering the errors themselves give and 0 for no better than ignoring it:")
            scores = _ep_compare(est, ds.activity[te],
                                 np.asarray(model.stats.extra["y_test_pred"], float))
            print(_ep_table(scores))
            if "conformal width" not in est:
                print("  (the conformal width is missing: it needs --conformal, and only "
                      "--conformal-normalise gives it a width that varies)")
            model.stats.extra["error_predictors"] = [dict(vars(sc), ratio=sc.ratio)
                                                     for sc in scores]
    if args.save:
        bundle = {"dataset": args.dataset, "activity": ds.activity_prop,
                  "train_names": [ds.names[i] for i in tr]}
        if second is not None:
            bundle["second_endpoint"] = second
        save_model(args.save, model, extra=bundle)
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
        bounds = None
        if conf is not None and conf.usable:
            # one width everywhere unless the rule was normalised, in which case the molecules
            # outside the training set get a wider one; nan where there is no prediction
            lo, hi = conf.interval(np.nan_to_num(yp, nan=0.0))
            lo = np.where(np.isfinite(yp), lo, np.nan)
            hi = np.where(np.isfinite(yp), hi, np.nan)
            bounds = (lo, hi)
        write_predictions_csv(args.predictions, ds.names, ds.activity, yp, ds.subset, y_cv=ycv,
                              domain=ad, interval=bounds,
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
            # a ranking model has these instead, and nan for everything above
            "rank_fit": st.rank_fit, "rank_cv": st.rank_cv, "rank_test": st.rank_test,
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


def _second_endpoint_values(ds, column):
    """The second endpoint, read from a molecule property of the same file."""
    vals = []
    for m in ds.mols:
        v = float("nan")
        if m.HasProp(column):
            try:
                v = float(str(m.GetProp(column)).strip().split()[0])
            except (ValueError, IndexError):
                v = float("nan")
        vals.append(v)
    return np.array(vals, float)


def _fit_second_endpoint(model, ds, tr, train, y, args):
    """Fit both endpoints on one kernel and install the first one's coefficients (item SC1).

    The two endpoints share the molecules' kernel and are tied by one correlation, so a molecule
    measured against one of them informs the other.  The fitted model keeps predicting the
    primary endpoint, with the coefficients the two-task fit gives it; the second endpoint and
    the selectivity travel with the model file.
    """
    from cmfqsar import multitask as MT

    y2_all = _second_endpoint_values(ds, args.second_endpoint)
    y2 = y2_all[tr]
    if np.isfinite(y2).sum() < 5:
        raise SystemExit(f"--second-endpoint {args.second_endpoint}: only "
                         f"{int(np.isfinite(y2).sum())} training molecules carry it; the "
                         f"properties this file has are "
                         f"{', '.join(ds.properties[:12]) or 'none'}")
    names = (ds.activity_prop or "primary", args.second_endpoint)
    obs = MT.Observations.from_columns(np.column_stack([y, y2]), names=names)
    both = int(np.sum(np.isfinite(y) & np.isfinite(y2)))
    print(f"Second endpoint '{args.second_endpoint}': {int(np.isfinite(y2).sum())} of "
          f"{len(tr)} training molecules carry it, {both} carry both")
    K = model._linear_train_kernel()
    lam = float(model.params.get("lam", 1.0))
    if args.task_rho is not None:
        rho = float(args.task_rho)
        how = "given"
    else:
        rhos = _floats(args.task_rhos) if args.task_rhos else ()
        fit_rho = MT.evidence_rho(K, obs, rhos=rhos, lams=[lam])
        rho = float(fit_rho.correlations[0, 1])
        how = f"chosen by the evidence ({fit_rho.note})"
    print(f"  task correlation {rho:+.3f}, {how}")
    T = MT.task_covariance(2, rho)
    fit = MT.fit(K, obs, lam=lam, T=T)
    joint = MT.score_by_task(obs, MT.cv_predict(K, obs, lam=lam, T=T, folds=args.folds,
                                                seed=args.seed))
    alone = MT.score_by_task(obs, MT.cv_predict(K, obs, rho=0.0, lam=lam, folds=args.folds,
                                                seed=args.seed))
    print("  ten-fold cross-validation, whole molecules held out:")
    for j, t in enumerate(joint):
        print(f"    {t['task']:28s} n {t['n']:5d}  q2 {t['q2']:+.3f} with the second endpoint, "
              f"{alone[j]['q2']:+.3f} on its own  ({t['q2'] - alone[j]['q2']:+.3f})")
    sel = MT.score_selectivity(K, obs, 0, 1, rho=rho, lam=lam, folds=args.folds, seed=args.seed)
    if sel.get("n", 0):
        print(f"    selectivity ({names[0]} - {names[1]}) on the {sel['n']} molecules measured "
              f"against both: q2 {sel['q2']:+.3f}, RMSE {sel['rmse_cv']:.3f}")
    model.coef = fit.task_coefficients(0)
    model.bias = float(fit.bias[0])
    print("  the model now predicts the primary endpoint with the two-task coefficients")
    return {"column": args.second_endpoint, "names": list(names), "rho": rho, "lam": lam,
            "coef": fit.task_coefficients(1).tolist(), "bias": float(fit.bias[1]),
            "selectivity_coef": fit.selectivity_coefficients(0, 1).tolist(),
            "selectivity_bias": float(fit.bias[0] - fit.bias[1]),
            "scores": joint, "scores_alone": alone,
            "selectivity": {k: v for k, v in sel.items()
                            if k in ("n", "q2", "rmse_cv")},
            "n_both": both}


def cmd_build_crystals(args):
    """A model on periodic structures: the periodic field autocorrelation of cmfqsar/periodic.py."""
    from cmfqsar.dataio import read_activities_csv
    from cmfqsar.periodic import (crystal_table, elements_in, inorganic_fieldset,
                                  periodic_channel_labels, read_structures, top_channels)
    t0 = time.time()
    atoms = read_structures(args.dataset)
    names = [str(a.info.get("name", f"{a.get_chemical_formula()}_{i}")) for i, a in enumerate(atoms)]
    if args.target_file:
        tab = read_activities_csv(args.target_file)
        y = np.array([tab.get(nm, np.nan) for nm in names], float)
    elif args.target:
        y = np.array([float(a.info.get(args.target, np.nan)) for a in atoms])
    else:
        raise SystemExit("build-crystals needs --target KEY or --target-file CSV")
    ok = np.isfinite(y)
    print(f"Loaded {len(atoms)} structures from {args.dataset}; target: {args.target or args.target_file}")
    if ok.sum() < len(atoms):
        print(f"  {len(atoms) - ok.sum()} structures without a target are ignored")
    idx = np.flatnonzero(ok)
    rng = np.random.default_rng(args.seed)
    te = np.array([], int)
    if args.test_fraction > 0:
        perm = rng.permutation(idx)
        n_te = int(round(args.test_fraction * len(idx)))
        te, idx = np.sort(perm[:n_te]), np.sort(perm[n_te:])
    physchem = [f for f in args.fields.split(",") if f]
    if args.elements == "auto":
        elements = elements_in([atoms[i] for i in idx], args.min_structures)
    elif args.elements == "none":
        elements = []
    else:
        elements = [e for e in args.elements.split(",") if e]
    fs = inorganic_fieldset(elements, physchem=physchem, charge_model=args.charges)
    print(f"  fields: {', '.join(sp.label for sp in fs.specs)}")
    print(f"  training set: {len(idx)} structures" + (f", test set: {len(te)}" if len(te) else ""))
    train = crystal_table([atoms[i] for i in idx], fs, [names[i] for i in idx])
    sources = sorted({m.charge_source for m in train.molecules if m.charge_source})
    if sources:
        print(f"  electrostatic weights from: {', '.join(sources)}")
    model = CMFModel(fieldset=fs, method=args.method, normalize=True, kernel_source="periodic",
                     ac_sigma=args.sigma, ac_d_max=args.cutoff, pbc_taper=args.taper,
                     pbc_background=not args.no_background, kernel_layer=args.kernel_layer,
                     ac_block_weights=_block_weights(args.block_weights),
                     ac_block_groups=_block_groups(args.block_groups, fs),
                     ac_block_rho=args.block_rho)
    taper = "no taper" if args.taper <= 0 else f"{args.taper:g} A taper"
    print(f"Kernel: periodic field autocorrelation, neighbours to {args.cutoff:g} A, channels every "
          f"{args.sigma:g} A, {taper}, background {'kept' if args.no_background else 'removed'}; "
          f"per atom of the cell. Invariant to the choice of cell, origin and orientation; no "
          f"co-fields, but a co-pattern over the channels.")
    seed = None if args.cv_no_shuffle else args.seed

    def prog(msg, fr):
        print(f"\r  [{fr * 100:5.1f}%] {msg:<70}", end="", flush=True)

    res = grid_search(train, y[idx], model, [1.0], lams=_floats(args.lams),
                      epsilons=_floats(args.epsilons), nus=_floats(args.nus),
                      gammas=_floats(args.gammas), folds=args.folds, seed=seed, progress=prog)
    print()
    print(f"Best: {res.best_params}, RMSEcv = {res.best_rmse:.4g}, q2 = {res.best_q2:.3f}")
    model.fit(train, y[idx], cv_folds=args.folds, cv_seed=seed)
    print(model.stats.summary())
    _print_block_weights(model)
    if args.external_cv:
        ncv = nested_cv(train, y[idx], model, [1.0], lams=_floats(args.lams),
                        epsilons=_floats(args.epsilons), nus=_floats(args.nus),
                        outer_folds=args.external_cv, inner_folds=args.folds,
                        repeats=args.external_repeats, seed=seed, precompute=False, progress=prog)
        print()
        print(f"External {args.external_cv}-fold CV x {args.external_repeats}: q2ex = {ncv.q2_ex:.3f} "
              f"+/- {ncv.q2_ex_sd:.3f}, RMSEcvex = {ncv.rmse_ex:.4g} +/- {ncv.rmse_ex_sd:.4g}")
        model.stats.extra["external_cv"] = {"q2_ex": ncv.q2_ex, "rmse_ex": ncv.rmse_ex,
                                            "q2_ex_sd": ncv.q2_ex_sd, "rmse_ex_sd": ncv.rmse_ex_sd}
    if len(te):
        test = crystal_table([atoms[i] for i in te], fs, [names[i] for i in te])
        model.evaluate_test(test, y[te])
        print(model.stats.summary())
    if model.is_linear:
        labels = periodic_channel_labels(fs, model._ac_spec())
        print(f"Co-pattern: the {args.top} channels that move the prediction most "
              f"(weight x spread over the training set):")
        for lab, w, sd in top_channels(model, labels, n=args.top):
            print(f"  {w * sd:+9.3f}  {lab}")
    if args.predictions:
        pred = np.full(len(atoms), np.nan)
        pred[idx] = model.stats.extra["y_fit"]
        if len(te) and "y_test_pred" in model.stats.extra:
            pred[te] = model.stats.extra["y_test_pred"]
        subset = np.array(["train"] * len(atoms), dtype=object)
        subset[te] = "test"
        write_predictions_csv(args.predictions, names, y, pred, subset)
        print(f"Predictions written to {args.predictions}")
    if args.save:
        save_model(args.save, model, extra={"dataset": args.dataset, "task": "regression",
                                            "activity": args.target or args.target_file})
        print(f"Model saved to {args.save}")
    print(f"Elapsed {time.time() - t0:.1f} s")


def cmd_diagnose(args):
    """Ask a data set the questions that should precede modelling it (item C6)."""
    import json as _json

    from cmfqsar.diagnose import ALPHAS, diagnose

    ds = load_dataset(args.dataset, activity_prop=args.activity,
                      activity_file=args.activity_file)
    y = np.asarray(ds.activity, float)
    alphas = tuple(float(a) for a in args.alphas.split(",")) if args.alphas else ALPHAS
    routes = tuple(args.routes) if args.routes else ("aligned", "counts", "invariant",
                                                     "descriptors")
    d = diagnose(ds.mols, ds.names, y, alphas=alphas, folds=args.folds, seed=args.seed,
                 charge_model=args.charges, param_scheme=args.params, routes=routes,
                 activity_prop=ds.activity_prop,
                 progress=(lambda m: print(f"  ... {m}", flush=True)) if args.verbose else None)
    print(d.report())
    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(d.to_dict(), fh, indent=1)
        print(f"\nwritten to {args.json}")
    return 0


#: the default block size of the screen subcommand, from cmfqsar.screen
SCREEN_CHUNK = 500


def _engine_note(engine) -> str:
    """What to print about the kernel implementation a run will use."""
    from cmfqsar import fastkernel
    from cmfqsar.kernels import use_compiled

    if not use_compiled(engine):
        return ", NumPy kernels"
    return f", compiled kernels on {fastkernel.threads()} threads"


def cmd_screen(args):
    """Score a library block by block, never holding more than one block of it."""
    from cmfqsar.screen import ScreenSpec, screen

    model, _extra = load_model(args.model)
    if model.train is None:
        raise SystemExit("the model file carries no training molecules, so nothing can be scored "
                         "against it")
    cutoff = args.cutoff
    if cutoff is not None and str(cutoff).lower() not in ("auto", "none", "off"):
        cutoff = float(cutoff)
    elif cutoff is not None and str(cutoff).lower() in ("none", "off"):
        cutoff = None
    spec = ScreenSpec(chunk=max(1, int(args.chunk)), top=max(0, int(args.top)),
                      beta=float(args.beta), larger_is_better=not args.smaller_is_better,
                      pose=bool(args.pose), restarts=int(args.pose_restarts),
                      conformers=int(args.pose_conformers), cutoff=cutoff,
                      engine=args.engine)
    print(f"Screening {args.library} against {os.path.basename(args.model)}: "
          f"{model.train.n_mol} training molecules, blocks of {spec.chunk}"
          + (", posing every molecule first" if spec.pose else "")
          + (f", ranking by mean - {spec.beta:g} sd" if spec.beta else "")
          + (f", cutoff {spec.cutoff}" if spec.cutoff is not None else "")
          + _engine_note(spec.engine))

    last = [0.0]

    def progress(n, seconds):
        if seconds - last[0] >= 2.0:
            last[0] = seconds
            print(f"  {n} scored ({n / seconds:.0f} per second)", flush=True)

    rep = screen(model, args.library, spec, out=args.out, progress=progress)
    print(rep.describe())
    if rep.failures:
        print("molecules that could not be read or typed: " + ", ".join(rep.failures[:5])
              + (" ..." if len(rep.failures) > 5 else ""))
    if rep.out_path:
        print(f"written {rep.out_path}")


ALIGN_WAYS = ("mcs", "field", "variance", "pocket")


def _write_sdf(path, rdmols, fields, names):
    """Write the molecules with the coordinates their fields now carry."""
    from rdkit import Chem
    with Chem.SDWriter(path) as w:
        for m, mol, nm in zip(rdmols, fields, names):
            mm = Chem.Mol(m)
            conf = mm.GetConformer()
            for i, xyz in enumerate(mol.coords):
                conf.SetAtomPosition(i, [float(v) for v in xyz])
            mm.SetProp("_Name", nm)
            w.write(mm)


def _reference_index(names, spec, y=None):
    """Which molecule the others are aligned onto: an index, a name, or the default rule."""
    if spec is None:
        import numpy as _np
        if y is not None and _np.isfinite(y).sum() >= 2:
            # the largest molecule of the most active quarter, the rule make_mittal_examples uses
            thr = float(_np.nanpercentile(y, 75))
            cand = [i for i in range(len(names)) if _np.isfinite(y[i]) and y[i] >= thr]
            return cand[0] if cand else 0
        return 0
    try:
        return int(spec)
    except (TypeError, ValueError):
        pass
    for i, n in enumerate(names):
        if n == spec:
            return i
    raise SystemExit(f"no molecule called {spec!r}; give an index or one of: "
                     + ", ".join(names[:6]) + (" ..." if len(names) > 6 else ""))


def cmd_align(args):
    """Put a set of molecules in one frame, by whichever of the four criteria is asked for."""
    import numpy as np

    ds = load_dataset(args.dataset, activity_prop=args.activity,
                      activity_file=args.activity_file)
    y = np.asarray(ds.activity, float)
    model = None
    if args.model:
        model, _extra = load_model(args.model)
        fieldset = model.fieldset
    else:
        physchem, props = _split_fields(args.fields) if args.fields else (["electrostatic", "steric", "hydrophobic", "hba", "hbd"], [])
        fieldset = FieldSet.create(physchem=physchem, alpha=args.alpha,
                                   atom_properties=props)
    mf = [build_fields(m, fieldset, n) for m, n in zip(ds.mols, ds.names)]
    before = [np.array(f.coords, copy=True) for f in mf]
    way = args.by
    if way in ("variance", "pocket") and model is None:
        raise SystemExit(f"--by {way} aligns by what a model says, so it needs --model")
    if way == "pocket" and not args.pocket:
        raise SystemExit("--by pocket needs --pocket STRUCTURE")
    guard = None if args.max_rmsd is not None and args.max_rmsd <= 0 else args.max_rmsd

    if way == "mcs":
        from cmfqsar.align import align_dataset
        ref = _reference_index(list(ds.names), args.reference, y)
        print(f"Aligning {len(mf)} molecules onto {ds.names[ref]} by maximum common substructure"
              + (", then refining on the field similarity" if args.field_refine else ""))
        log = align_dataset(list(ds.mols), ref, use_mcs=True, field_refine=args.field_refine,
                            fieldset=fieldset)
        mf = [build_fields(m, fieldset, n) for m, n in zip(ds.mols, ds.names)]
        for nm, line in zip(ds.names, log):
            print(f"  {nm:30s} {line}")
    elif way == "field":
        from cmfqsar.decoys import align_decoys
        ref = _reference_index(list(ds.names), args.reference, y)
        keep = [i for i in range(len(mf)) if i != ref] if args.reference is not None else None
        reference = AtomTable.from_molecules([mf[ref]]) if args.reference is not None \
            else (model.train if model is not None else AtomTable.from_molecules(mf))
        what = (f"the pose of {ds.names[ref]}" if args.reference is not None
                else ("the model's training set" if model is not None else "the set itself"))
        print(f"Aligning {len(mf)} molecules by maximum field similarity to {what} "
              f"({args.restarts} restarts)")
        res = align_decoys(mf, reference, fieldset, restarts=args.restarts, seed=args.seed)
        for nm, r in zip(ds.names, res):
            print(f"  {nm:30s} similarity {r.similarity_before:.3f} -> "
                  f"{r.similarity_after:.3f}")
    elif way == "variance":
        if args.flexible:
            from cmfqsar.flexalign import flexible_align
            print(f"Aligning {len(mf)} molecules by prediction confidence, rigid motion and the "
                  f"rotatable bonds, from {args.conformers} conformations")
            res = flexible_align(mf, list(ds.mols), model, max_rmsd=guard,
                                 restarts=args.restarts, conformers=args.conformers)
        else:
            from cmfqsar.varalign import variance_align
            print(f"Aligning {len(mf)} molecules by prediction confidence (rigid)"
                  + (f", at most {guard} A from the input pose" if guard else ""))
            res = variance_align(mf, model, max_rmsd=guard, restarts=args.restarts)
        for r in res:
            print(f"  {r.name:28s} variance {r.variance_before:8.4f} -> "
                  f"{r.variance_after:8.4f}   moved {r.rmsd:5.2f} A"
                  + ("   [at the displacement limit]" if r.limited else ""))
    else:
        from cmfqsar.complementarity import pocket_fields
        from cmfqsar.pocketalign import pocket_align, summarise
        from cmfqsar.protein import read_structure
        prot = read_structure(args.pocket, keep_waters=False)
        pocket = pocket_fields(prot, fieldset)
        print(f"Aligning {len(mf)} molecules inside the pocket of "
              f"{prot.name or args.pocket} ({pocket.n_atoms} atoms), complementarity weighted "
              f"{args.pocket_weight:g} against the confidence")
        res = pocket_align(mf, model, pocket, pocket_weight=args.pocket_weight,
                           max_rmsd=guard, restarts=args.restarts)
        for r in res:
            print("  " + r.describe())
        st = summarise(res)
        print(f"  median: variance {st['variance_before']:.4f} -> {st['variance']:.4f}, "
              f"pocket {st['pocket_before']:+.3f} -> {st['pocket']:+.3f}, "
              f"moved {st['moved']:.2f} A, {st['at_the_guard']} of {st['n']} at the guard")

    moved = [float(np.sqrt(np.mean(np.sum((f.coords - b) ** 2, axis=1))))
             for f, b in zip(mf, before)]
    print(f"Moved: median {float(np.median(moved)):.2f} A, largest "
          f"{float(np.max(moved)):.2f} A")
    if args.judge and model is not None:
        from cmfqsar.varalign import ensemble_similarity
        print(f"Mean field similarity within the set: "
              f"{ensemble_similarity(mf, fieldset.specs):.3f}")
    if args.out:
        _write_sdf(args.out, list(ds.mols), mf, list(ds.names))
        print(f"Aligned structures written to {args.out}")
    else:
        print("Nothing written: give --out to keep the result.")


def cmd_cofield(args):
    """The co-fields of a saved model on a grid, as cube files and as numbers."""
    import numpy as np
    from cmfqsar.cofields import (cofield_volume, cofield_with_error, export_cube, make_grid,
                                  molfield_volume, noise_level, overlap_volume)

    model, _extra = load_model(args.model)
    model._require_linear("co-fields")
    if model.train is None:
        raise SystemExit("the model file carries no training molecules, so it has no grid")
    endpoint = getattr(args, "endpoint", "primary") or "primary"
    if endpoint != "primary":
        # a two-endpoint model carries the other endpoint's dual coefficients and their
        # difference; swapping them in turns every co-field into that endpoint's (item SC1)
        second = _extra.get("second_endpoint") if isinstance(_extra, dict) else None
        if second is None:
            raise SystemExit(f"--endpoint {endpoint} needs a model built with "
                             "--second-endpoint; this one has a single endpoint")
        key = "coef" if endpoint == "second" else "selectivity_coef"
        bias = "bias" if endpoint == "second" else "selectivity_bias"
        model.coef = np.asarray(second[key], float)
        model.bias = float(second[bias])
        which = (second["names"][1] if endpoint == "second"
                 else f"{second['names'][0]} - {second['names'][1]}")
        print(f"  co-fields of '{which}' (correlation {second['rho']:+.3f})")
    names = [sp.name for sp in model.fieldset.specs]
    asked = set(_split_fields(args.fields)[0]) if args.fields else None
    wanted = names if asked is None else [n for n in names if n in asked]
    if not wanted:
        raise SystemExit(f"none of those fields is in the model; it has {', '.join(names)}")
    grid = make_grid(model.train.coords, margin=args.margin, spacing=args.spacing)
    print(f"{os.path.basename(args.model)}: {model.train.n_mol} training molecules, "
          f"grid {grid.shape} at {args.spacing} A = {int(np.prod(grid.shape))} points")
    mol = None
    if args.molecule is not None:
        ds = load_dataset(args.molecule)
        mol = build_fields(ds.mols[0], model.fieldset, ds.names[0])
        print(f"  against {ds.names[0]} from {args.molecule}")
    sigma = None
    if args.z > 0:
        sigma, src = noise_level(model)
        print(f"  masking below {args.z:g} standard deviations, noise level {sigma:.3f} ({src})")
    rows = []
    for f, name in enumerate(names):
        if name not in wanted:
            continue
        if args.kind == "molecule":
            if mol is None:
                raise SystemExit("--kind molecule needs --molecule FILE")
            vol = molfield_volume(model, mol, f, grid)
        elif args.kind == "overlap":
            if mol is None:
                raise SystemExit("--kind overlap needs --molecule FILE")
            vol = overlap_volume(model, mol, f, grid)
        else:
            vol = cofield_volume(model, f, grid)
        kept = float("nan")
        if args.z > 0 and args.kind == "cofield":
            ce = cofield_with_error(model, f, grid, sigma=sigma, z=args.z)
            covered = ce.sd > 0
            kept = float(np.mean(np.abs(ce.mean[covered]) > args.z * ce.sd[covered])) \
                if covered.any() else 0.0
            vol = ce.masked
        peak = float(np.max(np.abs(vol))) if vol.size else 0.0
        signed = float(vol.flat[int(np.argmax(np.abs(vol)))]) if vol.size else 0.0
        rows.append((name, peak, signed, kept))
        print(f"  {name:16s} peak {peak:10.4f} (signed {signed:+10.4f})"
              + (f"   {kept:.0%} of the covered grid above {args.z:g} sd"
                 if kept == kept else ""))
        if args.out_prefix:
            path = f"{args.out_prefix}_{args.kind}_{name.replace(':', '_')}.cube"
            export_cube(path, grid, vol, mol,
                        title=f"CMF {args.kind} {name} of {os.path.basename(args.model)}")
            print(f"    written {path}")
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["field", "peak", "signed_peak", "share_above_z"])
            for r in rows:
                w.writerow([r[0], f"{r[1]:.6g}", f"{r[2]:.6g}",
                            "" if r[3] != r[3] else f"{r[3]:.4f}"])
        print(f"written {args.csv}")
    if not args.out_prefix and not args.csv:
        print("Nothing written: give --out-prefix for cube files or --csv for the table.")


def cmd_benchmark(args):
    """Run, list or check the benchmarks, which live in benchmarks/run_all.py."""
    import runpy
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "benchmarks", "run_all.py")
    if not os.path.isfile(path):
        raise SystemExit("the benchmarks are part of the source checkout and are not installed "
                         f"with the package; {path} is missing")
    argv = [path]
    for flag in ("list", "check", "tables", "manifest", "readme"):
        if getattr(args, flag, False):
            argv.append("--" + flag)
    if args.data_root:
        argv += ["--data-root", args.data_root]
    if args.orca:
        argv.append("--orca")
    argv += list(args.name or [])
    saved = list(sys.argv)
    sys.argv = argv
    sys.path.insert(0, os.path.join(root, "benchmarks"))
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as exc:
        # run_all.py ends with sys.exit(main()); a zero is how it says it finished, and a
        # subcommand that raised SystemExit(0) would look like a failure to a caller
        code = exc.code if exc.code is not None else 0
        if code:
            raise
    finally:
        sys.argv = saved


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
        elif args.align_pocket:
            from cmfqsar.complementarity import pocket_fields
            from cmfqsar.pocketalign import pocket_align, summarise
            from cmfqsar.protein import read_structure
            prot = read_structure(args.align_pocket, keep_waters=False)
            pocket = pocket_fields(prot, model.fieldset)
            print(f"  inside the pocket of {prot.name or args.align_pocket}: "
                  f"{pocket.n_atoms} atoms, complementarity weighted {args.pocket_weight:g} "
                  f"against the confidence")
            res = pocket_align(mf, model, pocket, pocket_weight=args.pocket_weight,
                               max_rmsd=guard, restarts=args.align_restarts)
            for r in res:
                print("  " + r.describe())
            s = summarise(res)
            print(f"  median: variance {s['variance_before']:.4f} -> {s['variance']:.4f}, "
                  f"pocket {s['pocket_before']:+.3f} -> {s['pocket']:+.3f}, "
                  f"moved {s['moved']:.2f} A, {s['at_the_guard']} of {s['n']} at the guard")
            res = []
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
        if args.rank_bound:
            # rank by a confidence bound rather than by the point estimate: a library is mostly
            # unlike the training set, and for those molecules a point estimate is the training
            # mean presented as a prediction
            score, yp, sd = model.rank_score(table, beta=float(args.rank_bound),
                                             larger_is_better=not args.smaller_is_better)
            var = sd ** 2
            order = np.argsort(-score if not args.smaller_is_better else score)
            print(f"Ranked by prediction {'-' if not args.smaller_is_better else '+'} "
                  f"{args.rank_bound:g} sd:")
            for i in order:
                print(f"{ds.names[i]:30s} {score[i]:8.3f}   = {yp[i]:7.3f} "
                      f"{'-' if not args.smaller_is_better else '+'} "
                      f"{args.rank_bound:g} x {sd[i]:.3f}")
        else:
            yp, var = model.predict_with_variance(table)
            for n, v, s in zip(ds.names, yp, var):
                print(f"{n:30s} {v:8.3f}   (GP var {s:.3g})")
    else:
        # the posterior variance is defined for the linear kernel only
        if args.rank_bound:
            raise SystemExit("--rank-bound needs a posterior, so it needs the linear kernel and "
                             "the krr or gp method; this model has neither")
        yp, var = model.predict(table), None
        for n, v in zip(ds.names, yp):
            print(f"{n:30s} {v:8.3f}")
    second = extra.get("second_endpoint") if isinstance(extra, dict) else None
    yp2 = sel = None
    if second is not None:
        # the second endpoint rides on the same kernel: only its dual coefficients differ
        if not model.is_linear:
            raise SystemExit("a two-endpoint model is fitted on the linear kernel; this one has "
                             f"the '{model.kernel_layer}' layer")
        K_cross = model.test_kernel(table)
        yp2 = K_cross @ np.asarray(second["coef"], float) + float(second["bias"])
        sel = (K_cross @ np.asarray(second["selectivity_coef"], float)
               + float(second["selectivity_bias"]))
        a, b = second["names"]
        print()
        print(f"Both endpoints (correlation {second['rho']:+.3f}):")
        print(f"{'molecule':30s} {a[:12]:>12s} {b[:12]:>12s} {'difference':>12s}")
        for n, v1, v2, d in zip(ds.names, yp, yp2, sel):
            print(f"{n:30s} {v1:12.3f} {v2:12.3f} {d:+12.3f}")
    if np.isfinite(ds.activity).sum() >= 2:
        model.evaluate_test(table, ds.activity)
        print(model.stats.summary())
    bounds = None
    conf = getattr(model, "conformal_", None)
    if conf is not None and args.conformal is not None:
        from cmfqsar.conformal import Conformal
        conf = Conformal(level=float(args.conformal), residuals=conf.residuals, kind=conf.kind,
                         difficulty=conf.difficulty, beta=conf.beta)
    if conf is not None:
        if not conf.usable:
            print(f"Conformal: {conf.describe()}")
        else:
            _yp, lo, hi = model.predict_interval(table)
            bounds = (lo, hi)
            print(f"Conformal: {conf.describe()}")
            for n, v, a, b in zip(ds.names, yp, lo, hi):
                print(f"{n:30s} {v:8.3f}   [{a:7.3f}, {b:7.3f}]")
    if args.out:
        write_predictions_csv(args.out, ds.names, ds.activity, yp, ["test"] * len(ds), var=var,
                              interval=bounds,
                              extra_columns=(None if yp2 is None else
                                             {second["names"][1]: yp2, "difference": sel}))
        print(f"Predictions written to {args.out}")


def cmd_report(args):
    """One self-contained HTML file describing a saved model: item F4 of the plan.

    Everything comes out of the model file, so the report cannot disagree with the model and
    costs no refitting; what the file does not carry (a validation that was never run) is
    reported as missing rather than quietly left out.
    """
    from cmfqsar.report import ReportSpec, write_report

    model, extra = load_model(args.model)
    out = args.out or (os.path.splitext(args.model)[0] + ".html")
    spec = ReportSpec(spacing=args.spacing, margin=args.margin, level=args.level,
                      max_features=args.max_features, top_atoms=args.top_atoms,
                      cofield_z=args.cofield_z,
                      cofields=not args.no_cofields,
                      pharmacophore=not args.no_pharmacophore)
    t0 = time.time()
    write_report(out, model, extra=extra, title=args.title, spec=spec, source=args.model,
                 progress=lambda msg, frac: print(f"  {msg}", flush=True))
    size = os.path.getsize(out) / 1024.0
    print(f"Report written to {out} ({size:.0f} kB, self-contained) in {time.time() - t0:.1f} s")
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(out)}")


def cmd_suggest(args):
    """Rank posed candidates by an acquisition rule: item E2 of the plan."""
    from cmfqsar.active import ACQUISITIONS, PURPOSE, suggest

    model, extra = load_model(args.model)
    ds = load_dataset(args.dataset, activity_prop=args.activity)
    mf = [build_fields(m, model.fieldset, n) for m, n in zip(ds.mols, ds.names)]
    _check_contract(model, mf, allow=args.allow_mismatch)
    table = AtomTable.from_molecules(mf)
    best = float(np.max(np.asarray(model.y, float)))
    print(f"Model {args.model}: {model.train.n_mol} molecules made so far, best activity "
          f"{best:.3f}")
    print(f"{len(ds)} candidates, ranked by {args.acquire}: {PURPOSE[args.acquire]}")
    picks = suggest(model, table, kind=args.acquire, n=args.top, beta=args.beta, xi=args.xi,
                    names=list(ds.names), max_similarity=args.max_similarity, seed=args.seed)
    for k, p in enumerate(picks, start=1):
        print(f"  {k:3d}. " + p.describe(args.acquire))
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "name", "predicted", "sd", args.acquire, "nearest_made"])
            for k, p in enumerate(picks, start=1):
                w.writerow([k, p.name, f"{p.mean:.4f}", f"{p.sd:.4f}", f"{p.score:.6f}",
                            f"{p.distance:.4f}"])
        print(f"  written to {args.out}")
    if args.acquire in ("greedy",):
        print("  note: this rule ignores the posterior entirely. A candidate unlike anything "
              "made so far is predicted at the training mean, and greedy acquisition cannot "
              "tell that answer from a real one.")


def cmd_design(args):
    """What to put at a position of a parent compound, ranked by the model: item E3.

    The model is a topomer model built here from the series, because the suggestion has to be
    posed by the same rule as the molecules it is being compared with, and because a model
    fitted on superimposed whole molecules has no way to place a group nobody has made.

    ``--kernel autocorrelation`` takes the other route (plan item SC7): the rotation-invariant
    kernel needs no frame, so the series needs no common scaffold and a candidate is built as a
    whole molecule and embedded once.  That is what makes a design loop possible on a table of
    guests, which share no scaffold at all; the core is then only what names the position being
    varied on the parent.
    """
    from rdkit import Chem
    from cmfqsar.design import design, fragment_smiles, read_fragments
    from cmfqsar.evidence import log_evidence
    from cmfqsar.topomer import (TopomerSpec, attachment_label, choose_core, cuttable_fraction,
                                 series_labels, substituents, topomer_fields)

    ds = load_dataset(args.dataset, activity_prop=args.activity, activity_file=args.activity_file)
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
    share = cuttable_fraction(mols, core)
    print(f"{len(mols)} molecules; core {Chem.MolToSmiles(core)} ({how}), "
          f"{len(labels)} attachment points, cut cleanly in {share:.0%} of them")
    if share < 0.6:
        print("  warning: this core is fused to the rest of most of these molecules, so what "
              "comes off is not a substituent and the suggestions will not mean much")

    # what the series puts where, which decides the default position and names the known groups
    at_site = {}
    for mol in mols:
        for frag in substituents(mol, core):
            at_site.setdefault(attachment_label(frag), set()).add(Chem.MolToSmiles(frag))
    if args.position is not None:
        label = int(args.position)
        if label not in labels:
            raise SystemExit(f"position {label} is not one this series substitutes: "
                             f"{', '.join(str(k) + ' (' + str(len(v)) + ' groups)' for k, v in sorted(at_site.items()))}")
    else:
        label = max(at_site, key=lambda k: len(at_site[k]))
        print(f"  position {label} varies most ({len(at_site[label])} distinct groups); "
              f"--position picks another")

    spec = TopomerSpec(torsion=args.torsion, seed=args.seed)
    physchem, atom_props = _split_fields(args.fields)
    if args.kernel == "autocorrelation":
        return _design_invariant(args, ds, mols, names, yy, core, labels, at_site, physchem,
                                 atom_props, spec)
    best = None
    for alpha in _floats(args.alphas):
        fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, alpha=alpha,
                             charge_model=args.charges, param_scheme=args.param)
        mf, log = topomer_fields(mols, names, fs, core=core, labels=labels, spec=spec)
        keep = [i for i, f in enumerate(mf) if f is not None]
        if len(keep) < 5:
            continue
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        table = AtomTable.from_molecules([mf[i] for i in keep])
        K = model.joint_kernel(model.prepare_training_kernels(table))
        for lam in _floats(args.lams):
            ev = log_evidence(K, yy[keep], float(lam), fit_bias=True)
            if np.isfinite(ev) and (best is None or ev > best[0]):
                best = (ev, float(alpha), float(lam), fs, table, keep, log)
    if best is None:
        raise SystemExit("too few molecules could be cut into substituents to fit anything")
    ev, alpha, lam, fs, table, keep, log = best
    for line in log[:3]:
        print("   " + line[:150])
    model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=True)
    model.fit(table, yy[keep], cv_folds=args.folds, cv_seed=0)
    print(f"Topomer model on {len(keep)} molecules: alpha {alpha:g}, lambda {lam:g} (by the "
          f"evidence), q2 {model.stats.q2:+.3f}, RMSEcv {model.stats.rmse_cv:.3f}")

    # only a molecule the rule can take apart can be a parent: the candidates are built by
    # replacing one of its groups, so the others have to be poseable as they are
    if args.parent:
        if args.parent not in names:
            raise SystemExit(f"no molecule called {args.parent!r} in {args.dataset}")
        p = names.index(args.parent)
        if p not in set(keep):
            raise SystemExit(f"{args.parent} cannot be cut into substituents by this core "
                             f"(it is one of the {len(mols) - len(keep)} that could not be "
                             f"posed), so it cannot be a parent; name another or give a core "
                             f"that comes off it at single acyclic bonds")
    else:
        p = int(keep[int(np.argmax(yy[keep]))])
        print(f"  parent: {names[p]}, the most active of those that can be posed "
              f"({yy[p]:.2f}); --parent picks another")
    frags = []
    if args.fragments in ("series", "both"):
        frags += sorted(at_site.get(label, []))
    if args.fragments in ("common", "both"):
        frags += read_fragments("common")
    if args.fragments not in ("series", "common", "both"):
        frags = read_fragments(args.fragments)
    seen = {fragment_smiles(f) for f in at_site.get(label, [])}

    res = design(model, mols[p], core, label, fragments=frags, labels=labels,
                 acquire=args.acquire, beta=args.beta, xi=args.xi, spec=spec,
                 parent_name=names[p], series=mols, series_names=names, series_y=yy,
                 known=seen, with_fields=args.fields_for)
    print()
    print(f"Parent {res.parent}: measured {res.parent_measured:.3f}, "
          f"predicted {res.parent_mean:.3f} +- {res.parent_sd:.3f}")
    print("  " + res.describe())
    for k, c in enumerate(res.top(args.top), start=1):
        print(f"  {k:3d}. " + c.describe())
        for row in c.fields:
            print("        " + row.describe())
    if res.failed:
        print(f"  could not be posed: {', '.join(res.failed[:6])}"
              + (" ..." if len(res.failed) > 6 else ""))
    print("  the poses are the topomer convention, not conformations: a suggestion is a "
          "statement about which group belongs here, not about how it will sit")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "fragment", "position", "predicted", "sd", args.acquire,
                        "delta", "known", "made", "measured"])
            for k, c in enumerate(res.candidates, start=1):
                w.writerow([k, c.smiles, c.label, f"{c.mean:.4f}", f"{c.sd:.4f}",
                            f"{c.score:.6f}", f"{c.delta:+.4f}", int(c.known), c.made or "",
                            "" if not np.isfinite(c.measured) else f"{c.measured:.4f}"])
        print(f"  written to {args.out}")


def _design_invariant(args, ds, mols, names, yy, core, labels, at_site, physchem, atom_props,
                      spec):
    """The design loop on a model that needs no frame (plan item SC7).

    Nothing is superimposed and nothing is cut except the parent, whose group at the chosen
    position is the one being replaced; every molecule of the table is a training molecule,
    whether or not a scaffold comes off it.
    """
    from rdkit import Chem
    from cmfqsar.design import design, fragment_smiles, read_fragments
    from cmfqsar.evidence import log_evidence
    from cmfqsar.topomer import attachment_label, substituents

    fs = FieldSet.create(atom_properties=atom_props, physchem=physchem, alpha=1.0,
                         charge_model=args.charges, param_scheme=args.param)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True,
                     kernel_source="autocorrelation", ac_sigma=args.ac_sigma,
                     ac_d_max=args.ac_dmax)
    K = model.training_kernel(table)
    best = None
    for lam in _floats(args.lams):
        ev = log_evidence(K, yy, float(lam), fit_bias=True)
        if np.isfinite(ev) and (best is None or ev > best[0]):
            best = (ev, float(lam))
    if best is None:
        raise SystemExit("the evidence could not be evaluated on any lambda")
    lam = best[1]
    model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=True,
                     kernel_source="autocorrelation", ac_sigma=args.ac_sigma,
                     ac_d_max=args.ac_dmax)
    model.fit(table, yy, cv_folds=args.folds, cv_seed=0)
    print(f"Invariant model on {len(mols)} molecules: lambda {lam:g} (by the evidence), "
          f"q2 {model.stats.q2:+.3f}, RMSEcv {model.stats.rmse_cv:.3f}")
    if args.parent:
        if args.parent not in names:
            raise SystemExit(f"no molecule called {args.parent!r} in {args.dataset}")
        p = names.index(args.parent)
    else:
        with_core = [i for i, m in enumerate(mols) if m.GetSubstructMatch(core)]
        if not with_core:
            raise SystemExit("no molecule of this table carries the core; give --parent or "
                             "another --core")
        p = int(with_core[int(np.argmax(yy[with_core]))])
        print(f"  parent: {names[p]}, the most active molecule that carries the core "
              f"({yy[p]:.2f}); --parent picks another")
    here = [f for f in substituents(mols[p], core)]
    if not here:
        raise SystemExit(f"{names[p]} does not come apart at this core, so there is no position "
                         f"to vary; name another parent or core")
    label = int(args.position) if args.position is not None else attachment_label(here[0])
    frags = []
    if args.fragments in ("series", "both"):
        frags += sorted(at_site.get(label, []))
    if args.fragments in ("common", "both"):
        frags += read_fragments("common")
    if args.fragments not in ("series", "common", "both"):
        frags = read_fragments(args.fragments)
    seen = {fragment_smiles(f) for f in at_site.get(label, [])}
    res = design(model, mols[p], core, label, fragments=frags, labels=labels,
                 acquire=args.acquire, beta=args.beta, xi=args.xi, spec=spec,
                 parent_name=names[p], series=mols, series_names=names, series_y=yy,
                 known=seen, with_fields=args.fields_for, pose="molecule", seed=args.seed)
    print()
    print(f"Parent {res.parent}: measured {res.parent_measured:.3f}, "
          f"predicted {res.parent_mean:.3f} +- {res.parent_sd:.3f}")
    print("  " + res.describe())
    for k, c in enumerate(res.top(args.top), start=1):
        print(f"  {k:3d}. " + c.describe())
        if c.molecule:
            print(f"        {c.molecule}")
    if res.failed:
        print(f"  could not be built: {', '.join(res.failed[:6])}"
              + (" ..." if len(res.failed) > 6 else ""))
    print("  each candidate is one embedded conformation of the whole molecule; the kernel is "
          "invariant to rotation, so nothing was superimposed on anything")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "fragment", "molecule", "position", "predicted", "sd",
                        args.acquire, "delta", "known", "made", "measured", "group",
                        "group_of_total"])
            for k, c in enumerate(res.candidates, start=1):
                w.writerow([k, c.smiles, c.molecule, c.label, f"{c.mean:.4f}", f"{c.sd:.4f}",
                            f"{c.score:.6f}", f"{c.delta:+.4f}", int(c.known), c.made or "",
                            "" if not np.isfinite(c.measured) else f"{c.measured:.4f}",
                            "" if not np.isfinite(c.group_value) else f"{c.group_value:+.4f}",
                            "" if not np.isfinite(c.group_total) else f"{c.group_total:+.4f}"])
        print(f"  written to {args.out}")


def cmd_atoms(args):
    """What a file says about its atoms: a target that belongs to them (plan item LP1).

    There is no model over atoms yet -- the local kernel is the item that adds one -- so what
    this does is the part that has to be right first: read the target with its blanks, say what
    was measured and where, show the folds that hold out whole molecules, and write the whole
    thing back in the form every program can read.
    """
    from cmfqsar.atomtargets import AtomTargets, write_atom_predictions

    ds = load_dataset(args.dataset, activity_prop=args.activity)
    print(f"{len(ds)} molecules from {args.dataset}")
    if args.list:
        from cmfqsar.atomtargets import candidate_targets
        seen = candidate_targets(ds.mols)
        if not seen:
            print("  nothing here could be a per-atom target")
        for name, (count, atoms) in sorted(seen.items(), key=lambda kv: -kv[1][1]):
            print(f"  {name:24s} on {count} of {len(ds)} molecules, {atoms} atoms measured")
        return
    if not args.target:
        raise SystemExit("name the target with --target, or ask for --list")
    t = ds.set_atom_activity(args.target)
    print("  " + t.describe())
    if not len(t):
        raise SystemExit(f"no atom carries '{args.target}': --list says what the file has")
    counts = t.counts()
    worst = sorted(counts.items(), key=lambda kv: -kv[1])[:args.top]
    print(f"  the {len(worst)} molecules with the most measured atoms:")
    for k, n in worst:
        print(f"    {ds.names[k]:30s} {n:4d} atoms")
    folds = t.folds(args.folds, seed=args.seed)
    if folds:
        sizes = [len(f) for f in folds]
        print(f"  {len(folds)} folds holding out whole molecules: {min(sizes)} to {max(sizes)} "
              f"atoms each, and no molecule is split between two of them")
    if args.out:
        written = write_atom_predictions(args.out, ds.mols, t, names=ds.names)
        print(f"  {written} molecules written to {args.out}, the target on the atoms it "
              f"belongs to")


def cmd_local(args):
    """A model over atoms: the field around an atom against a count around it (plan item LP2).

    The sample is an atom in a molecule, not the molecule. The descriptor is the molecule's
    fields under a Gaussian envelope about the atom, turned into the rotation-invariant
    autocorrelation, so nothing has to be aligned. The folds hold out whole molecules, and
    `--reference` adds the atom-centred count that every such model has to beat.
    """
    from cmfqsar.localenv import (DEFAULT_RHO, choose_lam, count_features, count_kernel,
                                  environments, envelope_width, kept_share, local_features,
                                  local_kernel, paired_interval, types_of)

    ds = load_dataset(args.dataset, activity_prop=args.activity)
    t = ds.set_atom_activity(args.target)
    print(f"{len(ds)} molecules from {args.dataset}")
    print("  " + t.describe())
    if not len(t):
        raise SystemExit(f"no atom carries '{args.target}': 'cmf atoms --list' says what the "
                         f"file has")
    physchem, atom_props = _split_fields(args.fields)
    cif = cif_types_in(ds.mols, "sybyl", args.cif_min) if args.cif else []
    if cif:
        print(f"  {len(cif)} indicator fields: " + ", ".join(cif))
    fieldset = FieldSet.create(physchem=physchem, atom_properties=atom_props, cif_types=cif,
                               charge_model=args.charges, param_scheme=args.param)
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(ds.mols, ds.names)]

    reference = None
    if args.reference:
        types = types_of(fields, min_atoms=max(5, len(t) // 100))
        reference = choose_lam(count_kernel(count_features(fields, t, types)), t,
                               folds=args.folds, seed=args.seed, name="atom-centred counts")
        print("  " + reference.summary())

    best = None
    for rho in (args.rho or [DEFAULT_RHO]):
        if args.align:
            r, used, extra = _local_aligned(args, ds, fields, t, fieldset, rho)
        else:
            envs = environments(fields, t, rho=rho)
            r = choose_lam(local_kernel(local_features(envs)), t, folds=args.folds,
                           seed=args.seed, name=f"local field, rho {rho:g}")
            used, extra = t, f"{np.mean([e.n_atoms for e in envs]):.1f} atoms per sample"
        kept = float(np.mean([kept_share(fields[int(used.molecule[i])], int(used.atom[i]), rho)
                              for i in range(len(used))]))
        print("  " + r.summary())
        print(f"    envelope width {envelope_width(rho):.2f} A, {extra}, "
              f"{kept:.1%} of the enveloped weight kept")
        if reference is not None and len(used) == len(t):
            diff, lo, hi = paired_interval(r, reference, t, runs=args.bootstrap, seed=args.seed)
            word = ("beats" if lo > 0 else "loses to" if hi < 0 else "is no better than")
            print(f"    against the counts {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]: it {word} them")
        elif reference is not None:
            print("    not compared with the counts: the site rule left out some of the atoms "
                  "the reference was fitted on")
        if best is None or r.q2 > best[0].q2:
            best = (r, rho, used)
    if args.out and best is not None:
        from cmfqsar.atomtargets import write_atom_predictions
        written = write_atom_predictions(args.out, ds.mols, best[2], best[0].predicted,
                                         names=ds.names)
        print(f"  {written} molecules written to {args.out}: the measured value and the "
              f"out-of-fold prediction on the atom each belongs to")


def _local_aligned(args, ds, fields, targets, fieldset, rho):
    """The posed overlap kernel of plan item LP3, and the co-field it allows."""
    from cmfqsar.localalign import overlap_kernel, posed_environments, ring_frames
    from cmfqsar.localenv import choose_lam

    frames_of = ring_frames if args.site == "ring" else None
    used = targets
    if frames_of is not None:
        from cmfqsar.localenv import environment
        keep = [r for r in range(len(targets))
                if frames_of(environment(fields[int(targets.molecule[r])],
                                         int(targets.atom[r]), rho))]
        if not keep:
            raise SystemExit("the ring site rule accepts none of these atoms: it wants an "
                             "aromatic carbon of a six-membered ring with two aromatic neighbours")
        if len(keep) < len(targets):
            print(f"  the ring site rule accepts {len(keep)} of {len(targets)} atoms; the rest "
                  f"are left out")
        used = targets.subset(keep)
    envs = posed_environments(fields, used, rho=rho, frames_of=frames_of)
    model, table, K = overlap_kernel(envs, fieldset)
    r = choose_lam(K, used, folds=args.folds, seed=args.seed,
                   name=f"posed overlap, rho {rho:g}")
    poses = float(np.mean([e.n_conformers for e in envs]))
    extra = (f"{np.mean([e.n_atoms for e in envs]):.1f} atoms per sample, "
             f"{poses:.1f} poses averaged")
    if args.cofield:
        _write_local_cofield(args, model, table, envs, used, r.lam)
    return r, used, extra


def _write_local_cofield(args, model, table, envs, targets, lam):
    """The co-field around the centre, as a cube file in the frame of the pose."""
    from cmfqsar.cofields import cofield_volume, export_cube, make_grid

    names = model.fieldset.names
    if args.cofield_field not in names:
        raise SystemExit(f"this model has no field called '{args.cofield_field}'; it has "
                         + ", ".join(names))
    model.params = dict(model.params)
    model.params["lam"] = float(lam)
    model.fit(table, np.asarray(targets.y, float), cv_folds=args.folds, cv_seed=args.seed,
              reuse_kernels=True)
    # a box around the centre, not around every atom the envelope kept: a co-field this far out
    # is read off a handful of atoms and says nothing
    reach = float(min(6.0, np.percentile(np.linalg.norm(
        np.vstack([e.coords for e in envs[:200]]), axis=1), 90)))
    grid = make_grid(np.array([[-reach, -reach, -reach], [reach, reach, reach]]), margin=0.0,
                     spacing=0.3)
    f = names.index(args.cofield_field)
    volume = cofield_volume(model, f, grid, include_scale=True)
    export_cube(args.cofield, grid, volume, envs[0])
    print(f"    co-field of the {args.cofield_field} field written to {args.cofield}, in the "
          f"frame of the pose: the centre is at the origin")


def cmd_transfer(args):
    """How much two models ask the same thing of the chemistry: item F3 of the plan."""
    from cmfqsar.transfer import cofield_overlap, describe_matrix, overlap_matrix

    models, names = [], []
    for path in args.models:
        model, extra = load_model(path)
        models.append(model)
        names.append(os.path.splitext(os.path.basename(path))[0])
    for name, model in zip(names, models):
        print(f"{name:20s} {model.train.n_mol:4d} molecules, "
              f"{', '.join(s.label for s in model.fieldset.specs if s.enabled)}, "
              f"alpha {', '.join('%g' % s.alpha for s in model.fieldset.specs if s.enabled)}")
    print()
    if len(models) == 2:
        res = cofield_overlap(models[0], models[1], names[0], names[1], strict=False)
        if res.notes and not np.isfinite(res.cosine):
            raise SystemExit("these two models cannot be compared: " + res.notes[0])
        print(res.describe())
        for row in res.fields:
            print("  " + row.describe())
        for note in res.notes:
            print("  note: " + note)
    else:
        M, pairs = overlap_matrix(models, names)
        print("Cosine of the co-fields, model against model:")
        print(describe_matrix(M, names))
        bad = [p for p in pairs if p.notes]
        for p in bad[:5]:
            print(f"  {p.name_a} / {p.name_b}: {p.notes[0]}")
        if args.fields:
            for f, spec in enumerate(models[0].fieldset.specs):
                if not spec.enabled:
                    continue
                Mf = np.full((len(models), len(models)), np.nan)
                for p in pairs:
                    i, j = names.index(p.name_a), names.index(p.name_b)
                    Mf[i, j] = Mf[j, i] = p.by_field().get(spec.name, float("nan"))
                np.fill_diagonal(Mf, 1.0)
                print()
                print(f"{spec.label}:")
                print(describe_matrix(Mf, names))
    print()
    print("  A cosine compares the shapes of two demands, not their sizes: it does not move "
          "when either activity is rescaled. It says nothing about whether either model is "
          "right, and it means anything at all only if the two training sets were posed in one "
          "frame.")
    if args.out and len(models) > 2:
        import csv
        M, _pairs = overlap_matrix(models, names)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([""] + names)
            for i, n in enumerate(names):
                w.writerow([n] + [f"{v:.4f}" if np.isfinite(v) else "" for v in M[i]])
        print(f"  matrix written to {args.out}")


def cmd_mixing(args):
    """The posterior over the field weights: item E6 of the plan."""
    from cmfqsar.mixing import drop_one, evidence_mixing, sample_mixing
    from cmfqsar.model import align_mixing

    model, extra = load_model(args.model)
    if not (model.is_linear and model.uses_fields):
        raise SystemExit("the mixing coefficients weigh the per-field overlap kernels; this "
                         "model uses another kernel and has none to weigh")
    if model.K_fields_train is None:
        model.K_fields_train = model._field_kernels(model.train, model.train, model.self_diag,
                                                    model.self_diag)
    Kf = np.asarray(model.K_fields_train, float)
    y = np.asarray(model.y, float)
    labels = [s.label for s in model.fieldset.specs]
    on = [s.enabled for s in model.fieldset.specs]
    print(f"Model {args.model}: {model.train.n_mol} molecules, {int(sum(on))} fields")
    print(f"  as fitted      {', '.join(f'{l} {s.h:g}' for l, s in zip(labels, model.fieldset.specs))}")
    print(f"  by alignment   {', '.join(f'{l} {h:.2f}' for l, h in zip(labels, align_mixing(Kf, y, model.scale, on)))}")
    print(f"  by the evidence {', '.join(f'{l} {h:.2f}' for l, h in zip(labels, evidence_mixing(Kf, y, model.scale, on)))}")
    print()
    print(f"Posterior over each field's share of the kernel ({args.draws} draws after "
          f"{args.burn} discarded):")
    post = sample_mixing(Kf, y, labels, model.scale, on, draws=args.draws, burn=args.burn,
                         seed=args.seed,
                         progress=(lambda m, f: None) if args.quiet else
                         (lambda m, f: print(f"  {m}", end="\r", flush=True)))
    if not args.quiet:
        print(" " * 60, end="\r")
    print(post.describe(level=args.level))
    small = post.carries_less_than_its_share(level=args.level)
    if small:
        print(f"  carrying less than an equal share, with {int(100 * args.level)} % confidence: "
              + ", ".join(small))
    print()
    print("How much log evidence the model loses without each field (the Bayes factor against "
          "'this field does nothing'):")
    for name, loss in sorted(drop_one(Kf, y, labels, model.scale, on).items(),
                             key=lambda kv: -kv[1]):
        verdict = ("it is carrying something no other field carries" if loss > 3 else
                   "worth keeping" if loss > 1 else
                   "the others already carry whatever it carries" if loss > -1 else
                   "the model is better off without it")
        print(f"  {name:18s} {loss:+7.1f}   {verdict}")
    print()
    print("  The shares are what the evidence can see, and on a congeneric series the field "
          "kernels are strongly correlated, so wide intervals are the usual answer rather than "
          "a failure: several mixtures explain the activities about equally well. A narrow "
          "interval is the informative case.")
    if args.out:
        import csv
        iv = post.interval(level=args.level)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["field", "share_mean", "share_lo", "share_hi", "share_map"])
            for f, name in enumerate(labels):
                w.writerow([name, f"{post.mean[f]:.4f}", f"{iv[f, 0]:.4f}", f"{iv[f, 1]:.4f}",
                            f"{post.map_shares[f]:.4f}"])
        print(f"  written to {args.out}")


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


def _thread(coords, centre, axis):
    """A guest in a channel: centroid on the centre, longest axis along it (plan item SC3)."""
    pos = np.asarray(coords, float)
    pos = pos - pos.mean(axis=0)
    if len(pos) > 2:
        _u, _s, vt = np.linalg.svd(pos, full_matrices=False)
        v = np.cross(vt[0], axis)
        c = float(np.dot(vt[0], axis))
        if np.linalg.norm(v) > 1e-8 and c > -1 + 1e-8:
            k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            pos = pos @ (np.eye(3) + k + k @ k / (1.0 + c)).T
    return pos + centre


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
    if args.place_in_cavity:
        from cmfqsar.docking import dock_coords
        from cmfqsar.protein import cavity_frame
        centre, axis = cavity_frame(prot)
        print(f"Placing {len(mf)} molecules in the cavity at "
              f"({centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f}), axis "
              f"({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f}): each one's centroid on the "
              f"centre and its longest axis along the channel, then refined in the site")
        moved = []
        for k, mol in enumerate(mf):
            mol.coords = _thread(mol.coords, centre, axis)
            coords, res = dock_coords(mol, pocket, fs, restarts=args.place_restarts, seed=0)
            mol.coords = coords
            moved.append(res.moved)
            if (k + 1) % 50 == 0:
                print("  [%5.1f%%] placed %d/%d" % (100.0 * (k + 1) / len(mf), k + 1,
                                                      len(mf)), flush=True)
        print(f"  the refinement moved a molecule by {np.median(moved):.2f} A (median)")

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
    print("  the poses are a convention, not conformations: the same substituent is always placed "
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
                                       write_pdb, write_query)
    from cmfqsar.cofields import make_grid
    model = load_model(args.model)
    if isinstance(model, tuple):
        model = model[0]
    grid = make_grid(model.train.coords, margin=args.margin, spacing=args.spacing)
    print(f"Model {args.model}: {model.train.n_mol} training molecules, "
          f"grid {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]} at {grid.spacing:.2f} A")
    if args.min_z > 0:
        print(f"  lobes below {args.min_z:g} standard deviations of the activities' noise are "
              f"masked out before the features are read")
    feats = extract(model, grid=grid, level=args.level, max_features=args.max_features,
                    min_radius=args.min_radius, spacing=args.spacing, margin=args.margin,
                    min_z=args.min_z, sigma=args.sigma,
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
    if args.query:
        try:
            left = write_query(args.query, feats, enable_top=args.enable_top)
        except ValueError as exc:
            raise SystemExit(str(exc))
        on = min(args.enable_top, sum(1 for f in feats
                                      if f.kind not in ("exclusion", "pocket")))
        print(f"  query written to {args.query}: {len(feats) - len(left)} features, "
              f"{on} of them switched on, the rest disabled for you to choose from")
        if left:
            kinds = sorted({f.kind for f in left})
            print(f"  left out, because a query has no way to say them: {', '.join(kinds)}. "
                  f"'bulk' asks for atoms of any kind, and the disfavoured features name a "
                  f"group that is undesired, which only an exclusion sphere could carry and it "
                  f"would exclude everything else with it.")
        print("  a hydrogen-bond feature is written without a direction: a co-field says that a "
              "donor here raises the activity, not where its hydrogen points")


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
    print(f"  q2 with random folds            {res.q2_random:+.3f}   <- overstates the quality of a mixed series")
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


def _fieldset_from(args, mols):
    """The field set a comparison or a clustering uses, built exactly as a model's would be."""
    physchem, atom_props = _split_fields(args.fields)
    cif = cif_types_in(mols, args.cif_scheme, args.cif_min) if args.cif else []
    return FieldSet.create(atom_properties=atom_props, physchem=physchem, cif_types=cif,
                           alpha=args.alpha, charge_model=args.charges,
                           cif_scheme=args.cif_scheme, param_scheme=args.param)


def cmd_compare(args):
    """Are two sets of molecules drawn from the same population?  A kernel two-sample test."""
    from cmfqsar.mmd import assemble, kernel_over, mmd_test, witness

    ds_a = load_dataset(args.dataset)
    if args.model:
        model, extra = load_model(args.model)
        fs = model.fieldset
        print(f"Model {args.model}: {model.train.n_mol} training molecules, "
              f"fields {', '.join(s_.label for s_ in fs.specs)}")
        mf = [build_fields(m, fs, n) for m, n in zip(ds_a.mols, ds_a.names)]
        table = AtomTable.from_molecules(mf)
        probe = CMFModel(fieldset=fs, method="krr", normalize=model.normalize)
        Kyy = probe.joint_kernel(probe.prepare_training_kernels(table))
        K, mask = assemble(model._linear_train_kernel(), model.test_kernel(table), Kyy)
        train_names = list(extra.get("train_names")
                           or getattr(model.train, "names", None) or [])
        if len(train_names) != model.train.n_mol:
            train_names = [str(i) for i in range(model.train.n_mol)]
        names = [f"train:{n}" for n in train_names] + list(ds_a.names)
        label_a, label_b = "the training set", args.dataset
    else:
        if not args.other:
            raise SystemExit("give a second data set, or --model to compare with a model's "
                             "training set")
        ds_b = load_dataset(args.other)
        fs = _fieldset_from(args, list(ds_a.mols) + list(ds_b.mols))
        K, mask = kernel_over(ds_a.mols, ds_a.names, ds_b.mols, ds_b.names, fs)
        names = list(ds_a.names) + list(ds_b.names)
        label_a, label_b = args.dataset, args.other
        print(f"Fields: {', '.join(s_.label for s_ in fs.specs)}, alpha = {args.alpha:g}")

    res = mmd_test(K, mask, permutations=args.permutations, seed=args.seed)
    print(f"\n{label_a} against {label_b}")
    print("  " + res.describe())
    if np.isfinite(res.null_sd):
        print(f"  the shuffled labels gave {res.null_mean:+.5f} +/- {res.null_sd:.5f}")
    w = witness(K, mask)
    order = np.argsort(w)
    print(f"\nMost unlike {label_a} (the witness, most negative first):")
    for i in order[:args.top]:
        print(f"  {names[i]:34s} {w[i]:+.4f}")
    print(f"Most typical of it:")
    for i in order[::-1][:args.top]:
        print(f"  {names[i]:34s} {w[i]:+.4f}")
    if args.out:
        import csv as _csv
        with open(args.out, "w", newline="") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["name", "set", "witness"])
            for i, nm in enumerate(names):
                wr.writerow([nm, label_a if mask[i] else label_b, f"{w[i]:.6f}"])
        print(f"Witness written to {args.out}")


def cmd_cluster(args):
    """Cluster a set by the field kernel itself, and compare that with its scaffolds."""
    from cmfqsar.chemotypes import chemotype_labels
    from cmfqsar.spectral import adjusted_rand, spectral_clusters

    ds = load_dataset(args.dataset, activity_prop=args.activity)
    fs = _fieldset_from(args, list(ds.mols))
    print(f"{len(ds)} molecules, fields: {', '.join(s_.label for s_ in fs.specs)}, "
          f"alpha = {args.alpha:g}")
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    probe = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = probe.joint_kernel(probe.prepare_training_kernels(AtomTable.from_molecules(mf)))
    fit = spectral_clusters(K, k=args.clusters, k_max=args.max_clusters, seed=args.seed)
    print(fit.describe())
    print(f"  eigenvalues {np.round(fit.eigenvalues[:min(8, len(fit.eigenvalues))], 4)}")

    scaffolds, scaffold_smiles = chemotype_labels(ds.mols, min_size=args.min_scaffold)
    ari = adjusted_rand(fit.labels, scaffolds)
    print(f"\n{len(set(scaffolds.tolist()))} scaffold groups; agreement with the field clusters: "
          f"adjusted Rand index {ari:+.3f}")
    if ari > 0.5:
        print("  the two partitions largely agree, so grouping by scaffold groups the kernel too")
    elif ari > 0.2:
        print("  the two partitions agree in part: the scaffold split is not the kernel's split")
    else:
        print("  the two partitions disagree: what the chemist calls a series and what the model "
              "finds similar are different things here, and a leave-one-scaffold-out figure is "
              "not testing what it appears to")
    if np.isfinite(ds.activity).sum() >= 3:
        for c in sorted(fit.sizes):
            sel = fit.labels == c
            y = ds.activity[sel]
            y = y[np.isfinite(y)]
            if len(y):
                print(f"  cluster {c}: {int(sel.sum()):4d} molecules, activity "
                      f"{y.mean():+.2f} +/- {y.std():.2f}")
    if args.out:
        import csv as _csv
        with open(args.out, "w", newline="") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["name", "field_cluster", "scaffold_group", "scaffold"])
            for i, nm in enumerate(ds.names):
                g = int(scaffolds[i])
                wr.writerow([nm, int(fit.labels[i]), g,
                             scaffold_smiles[g] if g < len(scaffold_smiles) else ""])
        print(f"Clusters written to {args.out}")


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
    b.add_argument("--pharm", action="store_true",
                   help="add indicator fields of the six pharmacophore features (donor, "
                        "acceptor, aromatic, hydrophobe, cation, anion), which unlike the Sybyl "
                        "indicators are each carried by most molecules of a series and read "
                        "directly as a pharmacophore")
    b.add_argument("--pharm-min", type=int, default=5, metavar="N",
                   help="a feature is used when at least this many training molecules carry it")
    b.add_argument("--pharm-features", default=None,
                   help="explicit comma-separated list of features (overrides --pharm-min)")
    b.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    b.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"],
                   help="parameterisation of the hydrophobic/H-bond fields: RDKit (Crippen + built-in rules) "
                        "or the Sybyl-type increments of the original CMF scripts (tripos; -legacy reproduces "
                        "the hydrogen-counting bug of the 2013 scripts)")
    b.add_argument("--method", default="krr",
                   choices=["krr", "gp", "svr", "nusvr", "ranksvm", "svc", "nusvc", "gpc",
                            "ocsvm", "ocmean", "ocknn", "ocgp", "ockpca",
                            "fuse-max", "fuse-min", "fuse-sum", "fuse-med",
                            "fuse-anz", "fuse-mnz", "fuse-euc", "fuse-rrf"],
                   help="gpc is the Gaussian-process classifier (Laplace approximation): it "
                        "gives calibrated probabilities, a posterior variance per molecule and "
                        "the marginal likelihood, and takes its amplitude from the C grid. The "
                        "fuse-* methods are the data-fusion rules for the one-class task: one "
                        "score from the similarities to the individual actives, with nothing "
                        "fitted; fuse-max is the similarity to the best-matching active and "
                        "fuse-sum is the centroid description under another name")
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
                        "square of this, so 3 is the fast setting and 5 the accurate one")
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
    b.add_argument("--kernel", default="fields", choices=["fields", "autocorrelation", "surface"],
                   help="where the kernel comes from: the overlap integrals of the fields, which "
                        "needs the molecules to share a frame; the rotation-invariant "
                        "autocorrelation of those fields, which needs no alignment at all; or "
                        "the surface autocorrelation, for molecules over a surface at z = 0 "
                        "(normal +z): invariant in the plane, heights kept, no in-plane alignment")
    b.add_argument("--surface-z", type=float, nargs=3, default=(1.0, 6.0, 1.0),
                   metavar=("ZMIN", "ZMAX", "ZSTEP"),
                   help="surface kernel: the height channels, from ZMIN to ZMAX in steps of "
                        "ZSTEP angstrom (default 1 6 1)")
    b.add_argument("--ac-sigma", type=float, default=0.75, metavar="S",
                   help="autocorrelation: width in angstrom along the distance axis, which also "
                        "sets the spacing of the channels")
    b.add_argument("--ac-dmax", type=float, default=12.0, metavar="D",
                   help="autocorrelation: longest interatomic distance resolved")
    b.add_argument("--ac-chirality", action="store_true",
                   help="autocorrelation: add the three-point signed channels, which are what "
                        "makes enantiomers distinguishable. A distance distribution is the same "
                        "for a molecule and its mirror image; the signed volume spanned by three "
                        "field-weighted moments is not. Off by default, so a model built without "
                        "it is unchanged")
    b.add_argument("--ac-chirality-shells", type=int, default=4, metavar="S",
                   help="autocorrelation: radial shells the signed volume is resolved over; "
                        "1 gives the single global invariant per field triple")
    b.add_argument("--block-weights", default=None, metavar="W",
                   help="autocorrelation / surface: mixing coefficients of the channel blocks "
                        "(one block per pair of field types), the analogue of the h_f of the "
                        "overlap kernel. 'balanced' gives every kind of block -- within a group "
                        "of fields, across two groups -- the same share of the kernel; 'align' "
                        "or 'alignf' set them by centred kernel alignment of the per-block "
                        "kernels, re-derived inside every cross-validation fold; KIND=W,... "
                        "names them (e.g. indicator=0.3,indicator*physchem=0.5; unnamed kinds "
                        "weigh 1, 0 removes a kind). Default: every block weighs 1")
    b.add_argument("--block-groups", default=None, metavar="G",
                   help="the groups of fields the kinds of block are made of, as "
                        "'GROUP=field,field;GROUP=cif:*' (patterns allowed); fields not named "
                        "fall into the default grouping: the continuous indicator fields "
                        "('indicator') against the rest ('physchem')")
    b.add_argument("--block-rho", type=float, default=1.0, metavar="R",
                   help="--block-weights alignf: shrinkage towards equal block weights (0 = none)")
    b.add_argument("--oc-k", type=int, default=5, metavar="K",
                   help="ocknn: how many nearest actives the distance is taken to")
    b.add_argument("--oc-components", type=int, default=5, metavar="Q",
                   help="ockpca: dimension of the principal subspace of the actives")
    b.add_argument("--oc-lam", type=float, default=0.1, metavar="L",
                   help="ocgp: ridge on the constant target")
    b.add_argument("--oc-score", default="mean", choices=["mean", "mean_over_sd"],
                   help="ocgp: the predictive mean, or the mean divided by its standard "
                        "deviation, which also lowers the score of a molecule the model knows "
                        "nothing about")
    b.add_argument("--fusion-threshold", type=float, default=0.0, metavar="S",
                   help="fuse-anz / fuse-mnz: what counts as a match. These two rules divide and "
                        "multiply by the number of actives a molecule matches, so on a field "
                        "similarity, where nothing is near zero, the default of 0 makes that "
                        "count a constant and both rules collapse onto fuse-sum exactly; give a "
                        "threshold inside the range the similarities actually span")
    b.add_argument("--rrf-k", type=float, default=60.0, metavar="K",
                   help="fuse-rrf: the constant of reciprocal rank fusion")
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
    b.add_argument("--alpha-search", default="grid", choices=["grid", "nelder-mead", "per-field"],
                   help="nelder-mead: simultaneous minimisation of RMSEcv over (alpha, lambda) with R's "
                        "Nelder-Mead simplex on kernels interpolated between the --alphas grid points "
                        "(protocol of the 2015 CIF paper); --lams is then ignored. "
                        "per-field: one attenuation per field by coordinate descent on the evidence, "
                        "starting from the best shared width (needs --select evidence)")
    b.add_argument("--nm-start", default="0.25,5",
                   help="starting point alpha,lambda of the Nelder-Mead search (default: that of the R scripts)")
    b.add_argument("--Cs", default="0.1,1,10,100")
    b.add_argument("--epsilons", default="0.05,0.1,0.2")
    b.add_argument("--nus", default="0.3,0.5,0.7")
    b.add_argument("--optimize-h", action="store_true", help="optimise mixing coefficients (model-selection bias!)")
    b.add_argument("--mkl", nargs="?", const="align", default=None,
                   choices=["align", "alignf", "evidence"],
                   help="multiple kernel learning: set the mixing coefficients h_f by centred "
                        "kernel alignment, either per field ('align', default) or jointly "
                        "('alignf'), or by maximising the marginal likelihood of the mixture "
                        "('evidence'); the kernel stays linear, so co-fields remain valid")
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
                        "molecule arrived in, measured where they are placed, so that the ensemble "
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
    b.add_argument("--second-endpoint", default=None, metavar="COLUMN",
                   help="model a second endpoint from this property of the same file together "
                        "with the activity (plan item SC1). The two share the molecules' kernel "
                        "and are tied by one correlation, so a molecule measured against one of "
                        "them informs the other; `predict` then reports both and their "
                        "difference, and `cofield --endpoint` draws either or the selectivity")
    b.add_argument("--task-rho", type=float, default=None, metavar="R",
                   help="the correlation between the two endpoints; the default chooses it by "
                        "the marginal likelihood")
    b.add_argument("--task-rhos", default=None, metavar="LIST",
                   help="the correlations the evidence chooses among (default: 25 values from "
                        "nearly -1 to 0.95)")
    b.add_argument("--increments", default=None, metavar="FILE",
                   help="read the atomic increments of the hydrophobic and hydrogen-bond fields "
                        "from a JSON table instead of the published one (plan item D5). Needs "
                        "--param tripos")
    b.add_argument("--learn-increments", action="store_true",
                   help="fit those increments on the training set instead of taking them from "
                        "the table. The q2 of the same cross-validation is then optimistic; "
                        "measure on a test set or with --nested")
    b.add_argument("--learn-increments-alpha", type=float, default=0.5, metavar="A",
                   help="the attenuation the increments are fitted at (default 0.5)")
    b.add_argument("--learn-increments-lam", type=float, default=0.1, metavar="L",
                   help="the ridge the increments are fitted at (default 0.1)")
    b.add_argument("--learn-increments-rho", type=float, default=1.0, metavar="R",
                   help="how strongly the fit is held near the published increments (default 1)")
    b.add_argument("--save-increments", default=None, metavar="FILE",
                   help="write the fitted increments to this JSON file")
    b.add_argument("--tautomers", default=None, choices=["score", "uniform", "given"],
                   help="represent every molecule by the mean field of its tautomers instead of "
                        "by the form the file gives (plan item D4). 'score' keeps the forms "
                        "RDKit scores highest, 'uniform' keeps all of them, 'given' reads "
                        "relative energies in kcal/mol from the molecule property "
                        "'tautomer_energies'. The first two are assumptions, not populations")
    b.add_argument("--tautomer-forms", type=int, default=8, metavar="N",
                   help="forms kept per molecule at most (default 8)")
    b.add_argument("--ph", type=float, default=None, metavar="PH",
                   help="represent every molecule by the pKa-weighted mean field of its "
                        "protonation states at this pH instead of by the state the file gives "
                        "(plan item D4). Needs Dimorphite-DL, whose patterns find the ionisable "
                        "sites and whose regression pKa values set the weights")
    b.add_argument("--ph-states", type=int, default=8, metavar="N",
                   help="states kept per molecule at most, the likeliest ones (default 8)")
    b.add_argument("--ph-min-population", type=float, default=0.01, metavar="F",
                   help="drop a state that holds less than this share of the molecule "
                        "(default 0.01)")
    b.add_argument("--max-conformers", type=int, default=0, metavar="N",
                   help="use at most the N conformations of each molecule with the largest weights "
                        "(0 = all)")
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
    b.add_argument("--select", default="cv", choices=["cv", "evidence"],
                   help="how alpha and lambda are chosen: by cross-validation, or by the Gaussian-process marginal likelihood, which costs one factorisation per candidate instead of one per fold and holds nothing out (regression with krr or gp only)")
    b.add_argument("--rank-gap", type=float, default=0.0, metavar="D",
                   help="ranksvm: ignore pairs whose activities differ by less than this. A pair "
                        "closer than the experimental error is a matter of chance, and learning it is "
                        "learning noise; half a log unit suits pooled literature data")
    b.add_argument("--rank-pairs", type=int, default=4000, metavar="N",
                   help="ranksvm: the largest number of pairs used. The solver works on the "
                        "kernel between pairs, so the cost is quadratic in this number and the "
                        "default already asks for 128 MB; a larger set is subsampled once, "
                        "reproducibly")
    b.add_argument("--rank-groups", default=None, metavar="PROP",
                   help="ranksvm: a molecule property naming the assay, paper or series. Pairs "
                        "are then formed only inside a group, so an offset between two sources "
                        "cancels instead of being learnt")
    b.add_argument("--conformal", type=float, default=None, metavar="LEVEL",
                   help="calibrate a prediction interval at this level (0.9) on the errors of "
                        "the cross-validation, and store it with the model")
    b.add_argument("--error-predictors", action="store_true",
                   help="score the GP variance, the applicability domain and the conformal width "
                        "against the errors actually made on the test set, by rank correlation, "
                        "risk-coverage efficiency and the RMSE of the trusted against the "
                        "doubted half")
    b.add_argument("--conformal-normalise", action="store_true",
                   help="with --conformal: widen the interval for molecules far from the "
                        "training set, instead of one width everywhere")
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
    bc = sub.add_parser("build-crystals",
                        help="build a model on periodic structures (CIF, POSCAR, extended XYZ, "
                             "ASE JSON): the periodic field autocorrelation of cmfqsar/periodic.py")
    bc.add_argument("dataset", help="an ASE-readable file with one or more structures")
    bc.add_argument("--target", default=None, metavar="KEY",
                    help="per-structure info key with the property (extended XYZ / ASE JSON)")
    bc.add_argument("--target-file", default=None, metavar="CSV",
                    help="CSV 'name,value' with the property, matched on the structures' 'name' "
                         "info key (default name: <formula>_<index>)")
    bc.add_argument("--fields", default="electrostatic,steric,prop:electronegativity",
                    help="physico-chemical fields: electrostatic (oxidation states), steric (atomic "
                         "volume), prop:electronegativity, prop:mass, prop:inverse_sqrt_mass (the "
                         "field of the light atoms), or prop:<array> for any per-atom array of the "
                         "file (Bader charges, magnetic moments, ...)")
    bc.add_argument("--elements", default="auto",
                    help="indicator fields: 'auto' (one per element in at least --min-structures "
                         "training structures, plus one for the rest), 'none', or a list Na,Cl,O")
    bc.add_argument("--min-structures", type=int, default=5, metavar="N")
    bc.add_argument("--charges", default="oxidation", choices=["oxidation", "electronegativity", "file"],
                    help="electrostatic weights: oxidation states guessed from the composition, the "
                         "electronegativity proxy, or the 'oxi_states'/'charges' array of the file")
    bc.add_argument("--cutoff", type=float, default=10.0, metavar="R",
                    help="neighbours are summed out to this radius (angstrom)")
    bc.add_argument("--sigma", type=float, default=0.5, metavar="S",
                    help="width and spacing of the distance channels (angstrom)")
    bc.add_argument("--taper", type=float, default=2.0, metavar="T",
                    help="width of the cosine taper below the cutoff; 0 for a hard cutoff")
    bc.add_argument("--no-background", action="store_true",
                    help="keep the uniform-density term instead of subtracting it")
    bc.add_argument("--block-weights", default=None, metavar="W",
                    help="mixing coefficients of the channel blocks (one per pair of field "
                         "types): 'balanced' gives the physico-chemical pairs, the indicator "
                         "pairs and the pairs across the two the same share of the kernel "
                         "however many blocks each kind has; 'align' / 'alignf' set every "
                         "block's weight by centred kernel alignment, re-derived inside every "
                         "cross-validation fold; KIND=W,... names them "
                         "(indicator=0.3,indicator*physchem=0.5; 0 removes a kind). "
                         "Default: every block weighs 1")
    bc.add_argument("--block-groups", default=None, metavar="G",
                    help="the groups the kinds of block are made of, 'GROUP=field,field;"
                         "GROUP=cif:*' (patterns allowed); unnamed fields fall into the default "
                         "grouping, the indicator fields ('indicator') against the rest "
                         "('physchem')")
    bc.add_argument("--block-rho", type=float, default=1.0, metavar="R",
                    help="--block-weights alignf: shrinkage towards equal block weights")
    bc.add_argument("--method", default="krr", choices=["krr", "gp", "svr", "nusvr"])
    bc.add_argument("--kernel-layer", default="linear", choices=["linear", "gauss", "arccos1", "arccos2"])
    bc.add_argument("--lams", default="1e-4,1e-3,1e-2,0.1,1")
    bc.add_argument("--epsilons", default="0.1")
    bc.add_argument("--nus", default="0.5")
    bc.add_argument("--gammas", default="0.25,0.5,1,2,4")
    bc.add_argument("--folds", type=int, default=10)
    bc.add_argument("--external-cv", type=int, default=0, metavar="K",
                    help="external K-fold cross-validation with the regularisation chosen inside")
    bc.add_argument("--external-repeats", type=int, default=1)
    bc.add_argument("--test-fraction", type=float, default=0.0)
    bc.add_argument("--seed", type=int, default=0)
    bc.add_argument("--cv-no-shuffle", action="store_true")
    bc.add_argument("--top", type=int, default=15, help="channels of the co-pattern to print")
    bc.add_argument("--predictions", default=None, help="CSV with the fitted/test predictions")
    bc.add_argument("--save", default=None)
    bc.set_defaults(func=cmd_build_crystals)
    dg = sub.add_parser("diagnose", help="ask a data set what it can support, before modelling it")
    dg.add_argument("dataset")
    dg.add_argument("--activity", default=None, help="the activity property to use")
    dg.add_argument("--activity-file", default=None, help="a table of activities")
    dg.add_argument("--routes", nargs="*", default=None,
                    choices=["aligned", "counts", "invariant", "descriptors"],
                    help="which comparisons to run (all four by default)")
    dg.add_argument("--alphas", default=None, help="comma-separated attenuations to search")
    dg.add_argument("--folds", type=int, default=10)
    dg.add_argument("--seed", type=int, default=0)
    dg.add_argument("--charges", default="gasteiger", help="gasteiger | mmff | file")
    dg.add_argument("--params", default="rdkit", help="rdkit | tripos | tripos-legacy")
    dg.add_argument("--json", default=None, help="write the diagnosis to this file as well")
    dg.add_argument("--verbose", action="store_true")
    dg.set_defaults(func=cmd_diagnose)

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
    q.add_argument("--align-pocket", default=None, metavar="STRUCTURE",
                   help="with --align-variance: put the receptor in the objective as well, so a "
                        "pose must both fit the pocket and sit inside the model's domain (PDB "
                        "or mmCIF of the complex or the site)")
    q.add_argument("--pocket-weight", type=float, default=1.0, metavar="W",
                   help="how much a unit of pocket complementarity is worth against a unit of "
                        "posterior variance; 0 is the confidence alignment alone")
    q.add_argument("--conformal", type=float, default=None, metavar="LEVEL",
                   help="report the model's calibrated interval at this level instead of the "
                        "one it was calibrated at; the model must carry a calibration")
    q.add_argument("--rank-bound", type=float, default=0.0, metavar="BETA",
                   help="rank the molecules by a confidence bound, prediction minus BETA "
                        "posterior standard deviations, instead of by the prediction itself. A "
                        "screening library is mostly unlike the training set, and a kernel "
                        "method returns the training mean presented as a prediction for those "
                        "molecules; "
                        "subtracting the spread demotes them. One standard deviation is the "
                        "conventional starting point")
    q.add_argument("--smaller-is-better", action="store_true",
                   help="with --rank-bound, the activity is one where a smaller value is more "
                        "active, so the bound is added rather than subtracted")
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

    cp = sub.add_parser("compare", help="are two sets of molecules the same population? (MMD)")
    cp.add_argument("dataset")
    cp.add_argument("other", nargs="?", default=None,
                    help="the second data set; omit it and give --model instead")
    cp.add_argument("--model", default=None,
                    help="compare the data set with a saved model's training set, using that "
                         "model's own fields")
    cp.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    cp.add_argument("--cif", action="store_true")
    cp.add_argument("--cif-scheme", default="sybyl",
                    choices=["sybyl", "element", "element+aromatic"])
    cp.add_argument("--cif-min", type=int, default=5)
    cp.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    cp.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"])
    cp.add_argument("--alpha", type=float, default=0.5)
    cp.add_argument("--permutations", type=int, default=999,
                    help="shuffles of the labels that make the null distribution")
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--top", type=int, default=8, help="molecules listed at each end")
    cp.add_argument("--out", default=None, help="CSV of the witness value of every molecule")
    cp.set_defaults(func=cmd_compare)

    cl = sub.add_parser("cluster", help="cluster a set by its field kernel (spectral clustering)")
    cl.add_argument("dataset")
    cl.add_argument("--activity", default=None)
    cl.add_argument("--clusters", type=int, default=None,
                    help="how many; by default the eigengap decides")
    cl.add_argument("--max-clusters", type=int, default=10,
                    help="the largest number the eigengap may choose")
    cl.add_argument("--min-scaffold", type=int, default=3,
                    help="scaffold groups smaller than this are merged before the comparison")
    cl.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    cl.add_argument("--cif", action="store_true")
    cl.add_argument("--cif-scheme", default="sybyl",
                    choices=["sybyl", "element", "element+aromatic"])
    cl.add_argument("--cif-min", type=int, default=5)
    cl.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    cl.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"])
    cl.add_argument("--alpha", type=float, default=0.5)
    cl.add_argument("--seed", type=int, default=0)
    cl.add_argument("--out", default=None, help="CSV of the cluster of every molecule")
    cl.set_defaults(func=cmd_cluster)

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
    h.add_argument("--place", default=None, choices=["field", "fieldgraph", "dock"],
                   help="place the chemotypes relative to one another, each moved as a rigid "
                        "body: by field similarity with the reference chemotype, by matching "
                        "their typed field extrema, which needs no overlap to start from, or by "
                        "docking its reference into the binding site")
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
                        "smaller than a modelling one, since the dependence on the poses decreases "
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
    ph.add_argument("--query", default=None, metavar="FILE",
                    help="write the features as a pharmacophore query a screening program can "
                         "read: .json for a Pharmit session, .pml for the LigandScout-style XML")
    ph.add_argument("--enable-top", type=int, default=4, metavar="N",
                    help="with --query: how many features to switch on in the written query. A "
                         "query is a conjunction, so a dozen at once matches nothing; the rest "
                         "are written disabled, to be switched on in the other program")
    ph.add_argument("--min-z", type=float, default=0.0, metavar="Z",
                    help="mask the co-field where it lies less than Z standard deviations "
                         "from zero before reading features off it, so that a lobe the noise in "
                         "the activities could have produced never becomes a feature (try 2)")
    ph.add_argument("--sigma", type=float, default=None, metavar="S",
                    help="with --min-z: the noise level of the activities (default: the "
                         "model's own out-of-fold RMSE)")
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
                         "ALPHA penalises the candidate for what the reference has and it lacks, "
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
    k.add_argument("--place-in-cavity", action="store_true",
                   help="put every molecule in the cavity before scoring: centroid on the "
                        "centre of the ring of units, longest axis along the channel, then the "
                        "local rigid-body refinement. For a macrocyclic host read from a "
                        "molecule file, whose guests are not already in the site (plan item SC3)")
    k.add_argument("--place-restarts", type=int, default=2, metavar="N",
                   help="perturbed restarts of that refinement (default 2)")
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

    al = sub.add_parser("align", help="put a set of molecules in one frame, by substructure, "
                                      "by field similarity, by prediction confidence or in a "
                                      "pocket")
    al.add_argument("dataset")
    al.add_argument("--by", default="mcs", choices=list(ALIGN_WAYS),
                    help="mcs: superimpose on the maximum common substructure with a reference "
                         "molecule. field: maximise the CMF field similarity to a reference. "
                         "variance: minimise the posterior variance of the prediction, which "
                         "needs a model. pocket: the confidence criterion with a complementarity "
                         "term, which needs a model and a structure")
    al.add_argument("--out", default=None, metavar="SDF",
                    help="where to write the aligned structures; without it nothing is kept")
    al.add_argument("--model", default=None, metavar="FILE",
                    help="a saved model: its field set is used, and --by variance and pocket "
                         "align by what it predicts")
    al.add_argument("--reference", default=None, metavar="INDEX|NAME",
                    help="the molecule the others are aligned onto (default: the first of the "
                         "most active quarter, or the first molecule)")
    al.add_argument("--pocket", default=None, metavar="STRUCTURE",
                    help="a PDB or mmCIF file, for --by pocket")
    al.add_argument("--pocket-weight", type=float, default=1.0, metavar="W")
    al.add_argument("--max-rmsd", type=float, default=2.0, metavar="A",
                    help="how far a molecule may move from its input pose; <=0 for no limit")
    al.add_argument("--restarts", type=int, default=2)
    al.add_argument("--flexible", action="store_true",
                    help="with --by variance, also turn the rotatable bonds")
    al.add_argument("--conformers", type=int, default=5, metavar="N",
                    help="with --flexible, how many conformations to start from")
    al.add_argument("--field-refine", action="store_true",
                    help="with --by mcs, refine the substructure fit on the field similarity")
    al.add_argument("--fields", default=None,
                    help="field set to measure with when no model is given (default: the five)")
    al.add_argument("--alpha", type=float, default=0.5)
    al.add_argument("--activity", default=None)
    al.add_argument("--activity-file", default=None)
    al.add_argument("--judge", action="store_true",
                    help="also report the mean field similarity within the set afterwards")
    al.add_argument("--seed", type=int, default=0)
    al.set_defaults(func=cmd_align)

    cf = sub.add_parser("cofield", help="the co-fields of a saved model on a grid, as cube files "
                                       "and as numbers, without the application")
    cf.add_argument("model", help="a model file written by build --save")
    cf.add_argument("--out-prefix", default=None, metavar="PREFIX",
                    help="write <prefix>_<kind>_<field>.cube for every field")
    cf.add_argument("--csv", default=None, metavar="FILE",
                    help="write the peak of every field as a table")
    cf.add_argument("--kind", default="cofield", choices=["cofield", "molecule", "overlap"],
                    help="cofield: the model's own map. molecule: one molecule's field in the "
                         "model's frame. overlap: that molecule's field weighted by the co-field, "
                         "which is where it agrees with the model")
    cf.add_argument("--molecule", default=None, metavar="FILE",
                    help="a structure file whose first molecule --kind molecule and overlap use")
    cf.add_argument("--fields", default=None, help="only these fields (default: all of them)")
    cf.add_argument("--spacing", type=float, default=0.5, metavar="A")
    cf.add_argument("--margin", type=float, default=3.0, metavar="A")
    cf.add_argument("--z", type=float, default=2.0, metavar="Z",
                    help="zero the co-field where it lies closer than this many standard "
                         "deviations to zero; 0 draws the unmasked map")
    cf.add_argument("--endpoint", default="primary",
                    choices=["primary", "second", "selectivity"],
                    help="for a model built with --second-endpoint: whose co-field to draw. "
                         "'selectivity' is the difference of the two, which is the map of what "
                         "separates them (plan item SC1)")
    cf.set_defaults(func=cmd_cofield)

    at = sub.add_parser("atoms", help="a target that belongs to the atoms: what a file carries "
                                      "and how it would be split")
    at.add_argument("dataset")
    at.add_argument("--target", default=None,
                    help="the per-atom target: a property on the atoms, or a molecule property "
                         "holding one field per atom with a blank where the atom was not "
                         "measured")
    at.add_argument("--list", action="store_true",
                    help="what per-atom properties the file carries, and on how many molecules")
    at.add_argument("--folds", type=int, default=10,
                    help="folds to describe; they hold out whole molecules (default 10)")
    at.add_argument("--seed", type=int, default=0)
    at.add_argument("--top", type=int, default=5, help="molecules listed (default 5)")
    at.add_argument("--activity", default=None, help="the molecule-level activity, if any")
    at.add_argument("--out", default=None, metavar="FILE",
                    help="write an SDF carrying the target on its atoms")
    at.set_defaults(func=cmd_atoms)

    lo = sub.add_parser("local", help="a model over atoms: the field around an atom against a "
                                      "count around it")
    lo.add_argument("dataset")
    lo.add_argument("--target", required=True,
                    help="the per-atom target, as 'cmf atoms --list' reports it")
    lo.add_argument("--rho", nargs="*", type=float, default=None, metavar="RHO",
                    help="envelope width parameters to try, in A^-2; the envelope falls to 1/e "
                         "at 1/sqrt(rho), so a larger value looks less far (default 0.5)")
    lo.add_argument("--reference", action="store_true",
                    help="also fit the atom-centred count model on the same folds and bootstrap "
                         "the difference over molecules")
    lo.add_argument("--align", action="store_true",
                    help="pose every environment by a rule and use the overlap kernel instead of "
                         "the rotation-invariant one (plan item LP3); this is what makes a "
                         "co-field around the atom possible")
    lo.add_argument("--site", default="ring", choices=["ring", "any"],
                    help="which rule poses the environment: 'ring' is an aromatic carbon of a "
                         "six-membered ring, with the bond leaving the ring on +x, and leaves out "
                         "every atom it does not fit; 'any' uses the centre's first shell "
                         "whatever it is (default ring)")
    lo.add_argument("--cofield", default=None, metavar="FILE.cube",
                    help="with --align: write the co-field around the centre as a cube file, in "
                         "the frame of the pose")
    lo.add_argument("--cofield-field", default="electrostatic", metavar="NAME",
                    help="which field's co-field to write (default electrostatic)")
    lo.add_argument("--folds", type=int, default=10,
                    help="folds; they hold out whole molecules (default 10)")
    lo.add_argument("--seed", type=int, default=0)
    lo.add_argument("--bootstrap", type=int, default=2000,
                    help="bootstrap samples for the interval (default 2000)")
    lo.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd",
                    help="the fields the environment is built from; prop:NAME takes a per-atom "
                         "property of the file, a quantum charge or a Fukui index")
    lo.add_argument("--cif", action="store_true",
                    help="add one continuous indicator field per atom type. With --align this is "
                         "what lets a co-field say what an atom of a given type at a given place "
                         "does, which a charge cannot: a donor and an acceptor carry opposite "
                         "charges and move a shift the same way (plan item LP3)")
    lo.add_argument("--cif-min", type=int, default=10, metavar="N",
                    help="an indicator field is made for a type carried by at least N molecules")
    lo.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    lo.add_argument("--param", default="rdkit", choices=["rdkit", "tripos", "tripos-legacy"])
    lo.add_argument("--activity", default=None, help="the molecule-level activity, if any")
    lo.add_argument("--out", default=None, metavar="FILE",
                    help="write an SDF carrying the measured and the predicted value on the "
                         "atoms they belong to")
    lo.set_defaults(func=cmd_local)

    bm = sub.add_parser("benchmark", help="run, list or check the benchmarks of the checkout")
    bm.add_argument("name", nargs="*", help="which benchmarks to run (default: all of them)")
    bm.add_argument("--list", action="store_true", help="what there is and what it needs")
    bm.add_argument("--check", action="store_true",
                    help="do the tables and the documents match the stored results?")
    bm.add_argument("--tables", action="store_true", help="rebuild every table from its results")
    bm.add_argument("--manifest", action="store_true",
                    help="record the hashes of the data and result files")
    bm.add_argument("--readme", action="store_true", help="refresh the spliced README blocks")
    bm.add_argument("--data-root", default=None, metavar="DIR",
                    help="where the data a benchmark needs but the repository does not ship is")
    bm.add_argument("--orca", action="store_true", help="also run the benchmarks that need ORCA")
    bm.set_defaults(func=cmd_benchmark)

    sc = sub.add_parser("screen", help="score a library against a saved model, reading it in "
                                       "blocks so that its size does not matter")
    sc.add_argument("model", help="a model file written by build --save")
    sc.add_argument("library", help="the molecules to score: an SD file, read as a stream, or a "
                                    "SMILES file, whose molecules are embedded as they are read")
    sc.add_argument("--out", default=None, metavar="FILE",
                    help="CSV of name, score, prediction and standard deviation, written as the "
                         "run goes, so an interrupted run keeps what it had scored")
    sc.add_argument("--chunk", type=int, default=SCREEN_CHUNK, metavar="N",
                    help=f"molecules per block (default {SCREEN_CHUNK}); the peak memory is one "
                         f"block of fields and one N x n_train kernel")
    sc.add_argument("--top", type=int, default=25, metavar="N",
                    help="how many of the best molecules to print and keep in memory")
    sc.add_argument("--beta", type=float, default=0.0, metavar="B",
                    help="rank by the lower confidence bound mean - B * sd instead of by the "
                         "prediction; 1.0 is the conventional starting point and is what keeps a "
                         "molecule the model knows nothing about out of the top of the list")
    sc.add_argument("--smaller-is-better", action="store_true",
                    help="the endpoint improves as it falls, so the bound is mean + B * sd")
    sc.add_argument("--pose", action="store_true",
                    help="place every molecule in the training set's frame first, by maximising "
                         "its field similarity to the training molecules; about half a second per "
                         "molecule, and optimistic, since a real library has to choose a pose "
                         "without knowing the actives")
    sc.add_argument("--pose-restarts", type=int, default=4, metavar="N")
    sc.add_argument("--pose-conformers", type=int, default=0, metavar="N",
                    help="generate N conformations of each molecule and pose every one, keeping "
                         "the best (the conformation in the file is always among them)")
    sc.add_argument("--cutoff", default=None, metavar="AUTO|R",
                    help="drop atom pairs further apart than R angstroms, or let 'auto' decide "
                         "per field whether a neighbour list is faster. The kernel is then an "
                         "approximation whose dropped pairs were each below 1e-6 of a peak "
                         "overlap; without this the sum is exact")
    sc.add_argument("--engine", default="auto", choices=["numpy", "numba", "auto"],
                    help="which atom-pair sum to use: 'numpy' (the reference, single-threaded), "
                         "'numba' (a compiled threaded loop, six to ten times faster on a "
                         "many-core machine, differing by about 4e-15), or 'auto', the default "
                         "here, which uses the compiled one where Numba is installed")
    sc.add_argument("--allow-mismatch", action="store_true",
                    help="score molecules the model's contract would reject")
    sc.set_defaults(func=cmd_screen)

    rp = sub.add_parser("report", help="one HTML file describing a saved model, for a colleague "
                                       "who will never start the application")
    rp.add_argument("model", help="a model file written by build --save")
    rp.add_argument("--out", default=None, metavar="FILE",
                    help="where to write it (default: the model's name with .html)")
    rp.add_argument("--title", default=None, help="the heading of the report")
    rp.add_argument("--spacing", type=float, default=0.6, metavar="A",
                    help="grid spacing for the co-field projections and the pharmacophore")
    rp.add_argument("--margin", type=float, default=3.0, metavar="A",
                    help="margin around the training molecules for the same grid")
    rp.add_argument("--level", type=float, default=0.5, metavar="F",
                    help="fraction of a co-field's peak that bounds a pharmacophore feature")
    rp.add_argument("--max-features", type=int, default=12)
    rp.add_argument("--cofield-z", type=float, default=2.0, metavar="Z",
                    help="draw only the part of each co-field that lies this far from zero "
                         "once the noise in the activities is propagated through the fit, and "
                         "read the pharmacophore from what is left; 0 draws everything")
    rp.add_argument("--top-atoms", type=int, default=8,
                    help="atom contributions listed for each of the two extreme molecules")
    rp.add_argument("--no-cofields", action="store_true",
                    help="leave out the co-field pictures, which are the slow part")
    rp.add_argument("--no-pharmacophore", action="store_true")
    rp.add_argument("--open", action="store_true", help="open it in the default browser")
    rp.set_defaults(func=cmd_report)

    sg = sub.add_parser("suggest", help="which posed candidate to make next (active learning)")
    sg.add_argument("model", help="a model saved with 'build --save'")
    sg.add_argument("dataset", help="posed candidates, in the training set's own frame")
    sg.add_argument("--acquire", default="ei",
                    choices=["ei", "ucb", "greedy", "variance", "maxmin", "random"],
                    help="ei: expected improvement over the best made so far (the usual choice "
                         "for finding actives); ucb: prediction plus beta standard deviations; "
                         "greedy: the prediction alone; variance: what most improves the model; "
                         "maxmin: furthest in field space from everything made, no model needed; "
                         "random: the baseline")
    sg.add_argument("--top", type=int, default=10, help="how many to print")
    sg.add_argument("--beta", type=float, default=1.0, metavar="B",
                    help="with --acquire ucb: how many standard deviations of optimism")
    sg.add_argument("--xi", type=float, default=0.0, metavar="X",
                    help="with --acquire ei: how much a candidate must exceed the best made by "
                         "before the improvement counts")
    sg.add_argument("--max-similarity", type=float, default=0.98, metavar="C",
                    help="a candidate more similar than this to one already picked is skipped, "
                         "so that a batch is not one molecule several times")
    sg.add_argument("--activity", default=None)
    sg.add_argument("--allow-mismatch", action="store_true")
    sg.add_argument("--seed", type=int, default=0)
    sg.add_argument("--out", default=None, metavar="FILE", help="CSV of the ranked candidates")
    sg.set_defaults(func=cmd_suggest)

    dg = sub.add_parser("design",
                        help="what to put at a position of a parent compound, ranked by the "
                             "model (co-field-guided design)")
    dg.add_argument("dataset", help="the series, with activities")
    dg.add_argument("--parent", default=None,
                    help="the compound to modify (default: the most active)")
    dg.add_argument("--position", type=int, default=None, metavar="LABEL",
                    help="which attachment point to vary (default: the one the series varies "
                         "most); the numbers are core atom indices and are printed above")
    dg.add_argument("--fragments", default="both",
                    help="'common' (a built-in list of medicinal-chemistry substituents), "
                         "'series' (the groups this series already uses at that position), "
                         "'both' (the default), or a file of one SMILES per line")
    dg.add_argument("--acquire", default="ei",
                    choices=["ei", "ucb", "greedy", "variance", "random"],
                    help="how to rank: expected improvement by default (see 'suggest')")
    dg.add_argument("--beta", type=float, default=1.0)
    dg.add_argument("--xi", type=float, default=0.0)
    dg.add_argument("--top", type=int, default=15, help="how many to print")
    dg.add_argument("--fields-for", type=int, default=3, metavar="N",
                    help="print the field-by-field difference from the parent for the top N")
    dg.add_argument("--core", default=None, help="the scaffold as SMILES")
    dg.add_argument("--core-smarts", default=None, help="the scaffold as SMARTS")
    dg.add_argument("--core-mode", default="framework", choices=["framework", "mcs"])
    dg.add_argument("--torsion", type=float, default=180.0,
                    help="the convention angle every rotatable torsion is set to")
    dg.add_argument("--fields", default="electrostatic,steric,hydrophobic,hba,hbd")
    dg.add_argument("--charges", default="gasteiger", choices=["gasteiger", "mmff", "file"])
    dg.add_argument("--param", default="rdkit", choices=["rdkit", "tripos"])
    dg.add_argument("--alphas", default="0.3,0.5,1.0,2.0")
    dg.add_argument("--lams", default="1e-3,1e-2,0.1,1")
    dg.add_argument("--kernel", default="fields", choices=["fields", "autocorrelation"],
                    help="fields: a topomer model, which poses every candidate by the topomer "
                         "rule. autocorrelation: the rotation-invariant kernel, which needs no "
                         "frame, so the series needs no common scaffold and a candidate is built "
                         "as a whole molecule and embedded once (plan item SC7)")
    dg.add_argument("--ac-sigma", type=float, default=0.75, metavar="A",
                    help="with --kernel autocorrelation: the width along the distance axis")
    dg.add_argument("--ac-dmax", type=float, default=12.0, metavar="A",
                    help="with --kernel autocorrelation: the longest distance resolved")
    dg.add_argument("--folds", type=int, default=10)
    dg.add_argument("--activity", default=None)
    dg.add_argument("--activity-file", default=None)
    dg.add_argument("--seed", type=int, default=0)
    dg.add_argument("--out", default=None, metavar="FILE", help="CSV of every candidate")
    dg.set_defaults(func=cmd_design)

    tr = sub.add_parser("transfer",
                        help="compare models by their co-fields: do they ask the same thing of "
                             "the chemistry?")
    tr.add_argument("models", nargs="+", help="two or more model files written by build --save")
    tr.add_argument("--fields", action="store_true",
                    help="with more than two models, print one matrix per field as well")
    tr.add_argument("--out", default=None, metavar="FILE", help="CSV of the matrix")
    tr.set_defaults(func=cmd_transfer)

    mx = sub.add_parser("mixing",
                        help="the posterior over the field weights: which fields this series "
                             "needs, with an interval")
    mx.add_argument("model", help="a model saved with 'build --save'")
    mx.add_argument("--draws", type=int, default=400, help="posterior draws to keep")
    mx.add_argument("--burn", type=int, default=100, help="draws to discard first")
    mx.add_argument("--level", type=float, default=0.9, help="credible level of the intervals")
    mx.add_argument("--seed", type=int, default=0)
    mx.add_argument("--quiet", action="store_true", help="no progress line")
    mx.add_argument("--out", default=None, metavar="FILE", help="CSV of the shares")
    mx.set_defaults(func=cmd_mixing)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
