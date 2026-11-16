#!/usr/bin/env python
"""
Does the field around an atom predict a property of that atom? (plan item LP2)

Every other model in this program predicts one number per molecule.  A carbon shift and an
acidity belong to an atom, so the sample has to be the atom and its surroundings:
`cmfqsar/localenv.py` builds that sample by multiplying the molecule's fields by a Gaussian
envelope about the atom, and the rotation-invariant field autocorrelation of
`cmfqsar/invariant.py` turns it into a descriptor that needs no alignment.

Two properties are measured, because they ask different things of the envelope.

* **Carbon shifts**, 13C spectra of nmrshiftdb2 measured in deuterated chloroform, with every
  signal assigned to an atom.  A shift is a short-range property: the atom, its bonds and the next
  shell decide most of it.
* **Carboxylic acid pKa**, literature and ChEMBL values collected by Baltruschat and Czodrowski
  (F1000Research 9:113, 2020), on molecules with exactly one carboxylic acid group.  Acidity is a
  response to losing a proton, so the sample is the **difference field** of the acid and its anion
  (`difference_field`), which is what `cmfqsar/fielddiff.py` compares in its kernels and what the
  free energy of deprotonation is a field picture of.  The environment of the plain acid is
  measured beside it, to say whether the difference is what carries the signal.

`examples/make_local_examples.py` prepares both files from the free downloads; the prepared sets
are in `examples/local/`.

How it is measured
------------------
* **The reference under rule 1 is an atom-centred count**: the type of the centre itself and the
  counts of atom types in distance shells around it, which is what a per-atom increment scheme is.
  It uses the same molecules, the same samples and the same folds, so the only difference between
  the two numbers is the descriptor.
* **The folds hold out whole molecules** (`AtomTargets.folds`).  Two carbons of one molecule, one
  in the training set and one in the test set, would let the model see the molecule it is being
  tested on.
* **The ridge is searched the same way for both descriptors**, on the same folds.  That search
  makes both numbers optimistic by the same kind of amount; what is compared is the difference.
* **The envelope width rho is reported twice**: at the module's default
  (`localenv.DEFAULT_RHO`, fixed before either set here was measured) and at the best value of the
  scan.  Only the first is a fair comparison with the reference, because the reference has no such
  parameter to tune; the second says how much the width matters.
* **The interval is bootstrapped over molecules, not atoms** (rule 4).  The atoms of one molecule
  are not independent samples, so a bootstrap sample is a sample of molecules with all of their
  atoms.

Usage
-----
    python benchmarks/run_local.py                      # both sets
    python benchmarks/run_local.py --set shifts
    python benchmarks/run_local.py --set pka --rho 0.25 0.5 1.0
    python benchmarks/run_local.py --max-molecules 60   # a quick look
    python benchmarks/run_local.py --report
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localenv import (DEFAULT_RHO, LOCAL_SPEC, count_features,  # noqa: E402
                              count_kernel, choose_lam, difference_field, ensemble_environment,
                              environment, local_features, local_kernel, paired_interval,
                              types_of)

DATA = os.path.join(ROOT, "examples", "local")
RESULTS = os.path.join(HERE, "results", "local")
#: envelope widths tried, per set.  A shift is a short-range property, so the scan stays narrow;
#: the plan expects an acidity to be felt further away, so that scan reaches a width of 4.5 A.
RHOS = (0.25, 0.5, 1.0, 2.0)
SETS = {
    "shifts": dict(file="shifts_cdcl3.sdf", target="shift", unit="ppm",
                   what="13C shifts in chloroform (nmrshiftdb2)", rhos=RHOS),
    "pka": dict(file="carboxylic_pka.sdf", target="pka", unit="pKa units",
                what="carboxylic acid pKa (Baltruschat and Czodrowski 2020)",
                rhos=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0)),
}


#: conformations generated per molecule for the ensemble half (plan item LP4)
CONFORMERS = 5
#: the envelope width each set is measured at there: the one that was best for it above
ENSEMBLE_RHO = {"shifts": 1.0, "pka": 0.1}


def _say(text):
    print(text, flush=True)


def read_set(key, max_molecules=0):
    """The molecules, their fields and the per-atom target of one set."""
    spec = SETS[key]
    path = os.path.join(DATA, spec["file"])
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing. Run examples/make_local_examples.py first; its "
                         f"docstring says where the free downloads are.")
    ds = load_dataset(path, atom_activity=spec["target"])
    mols = list(ds.mols)
    names = list(ds.names)
    if max_molecules and len(mols) > max_molecules:
        mols, names = mols[:max_molecules], names[:max_molecules]
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    targets = AtomTargets.from_molecules(mols, spec["target"], names=names)
    return mols, fields, targets, fieldset


def anion_of(mol, carbon):
    """The carboxylate of an acid: the same geometry with the acid proton taken away."""
    for nb in mol.GetAtomWithIdx(int(carbon)).GetNeighbors():
        if nb.GetSymbol() != "O":
            continue
        hs = [x.GetIdx() for x in nb.GetNeighbors() if x.GetSymbol() == "H"]
        if len(hs) != 1:
            continue
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(nb.GetIdx()).SetFormalCharge(-1)
        rw.GetAtomWithIdx(nb.GetIdx()).SetNoImplicit(True)
        rw.RemoveAtom(hs[0])
        out = rw.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            return None
        return out
    return None


def difference_environments(mols, fields, targets, fieldset, rho):
    """The environment of the acid-minus-anion field about every measured carboxyl carbon."""
    out = []
    for r in range(len(targets)):
        k, i = int(targets.molecule[r]), int(targets.atom[r])
        anion = anion_of(mols[k], i)
        if anion is None:
            out.append(environment(fields[k], i, rho))
            continue
        b = build_fields(anion, fieldset, name=f"{fields[k].name}-anion")
        out.append(environment(difference_field(fields[k], b), i, rho))
    return out


def plain_environments(fields, targets, rho):
    return [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
            for r in range(len(targets))]


def alike(K):
    """The mean kernel value between two different samples, a number in [-1, 1].

    A cosine-normalised kernel of 1 between two samples says the descriptor cannot tell them
    apart.  When that mean sits at almost 1 the model has nothing to work with, whatever the
    regression does afterwards, so this one number explains a result that is no better than the
    mean of the training set.
    """
    n = len(K)
    if n < 2:
        return float("nan")
    return float((K.sum() - np.trace(K)) / (n * (n - 1)))


def one_row(name, envs, targets, folds, seed):
    """Cross-validate one kind of environment."""
    t0 = time.time()
    X = local_features(envs)
    K = local_kernel(X)
    r = choose_lam(K, targets, folds=folds, seed=seed, name=name)
    r.note = f"{X.shape[1]} channels, {time.time() - t0:.0f} s"
    return r, alike(K)


def run_set(key, rhos=None, folds=10, seed=0, max_molecules=0, results=RESULTS):
    spec = SETS[key]
    rhos = list(rhos) if rhos else list(spec["rhos"])
    _say(f"=== {spec['what']}")
    mols, fields, targets, fieldset = read_set(key, max_molecules)
    _say(f"{len(targets)} measured atoms in {targets.n_measured_molecules} molecules")

    types = types_of(fields, min_atoms=10)
    Kc = count_kernel(count_features(fields, targets, types))
    ref = choose_lam(Kc, targets, folds=folds, seed=seed, name="atom-centred counts")
    ref_alike = alike(Kc)
    ref.note = f"{len(types)} atom types in 6 shells"
    _say("  " + ref.summary())

    rows, kinds = [], ["local field"]
    if key == "pka":
        kinds.append("local difference field")
    for rho in rhos:
        for label in kinds:
            if label == "local difference field":
                envs = difference_environments(mols, fields, targets, fieldset, rho)
            else:
                envs = plain_environments(fields, targets, rho)
            r, how_alike = one_row(f"{label}, rho {rho:g}", envs, targets, folds, seed)
            diff, lo, hi = paired_interval(r, ref, targets, runs=2000, seed=seed)
            rows.append(dict(kind=label, rho=float(rho), q2=r.q2, error=r.error, mae=r.mae,
                             lam=r.lam, note=r.note, difference=diff, low=lo, high=hi,
                             alike=how_alike,
                             atoms_per_environment=float(np.mean([e.n_atoms for e in envs]))))
            _say("  " + r.summary() + f"; against the counts {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    out = dict(set=key, what=spec["what"], unit=spec["unit"], target=spec["target"],
               n_samples=len(targets), n_molecules=targets.n_measured_molecules, folds=folds,
               seed=seed, fixed_rho=DEFAULT_RHO, spec=dict(d_max=LOCAL_SPEC.d_max,
                                                           step=LOCAL_SPEC.step,
                                                           sigma=LOCAL_SPEC.sigma),
               reference=dict(name=ref.name, q2=ref.q2, error=ref.error, mae=ref.mae,
                              lam=ref.lam, note=ref.note, alike=ref_alike),
               rows=rows,
               y=dict(low=float(np.min(targets.y)), high=float(np.max(targets.y)),
                      mean=float(np.mean(targets.y)),
                      spread=float(np.std(targets.y))))
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, f"{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def conformer_members(mol, fieldset, n, seed, name):
    """The fields of one molecule in each of its conformations, the given one first.

    The deposited geometry is kept as the first conformation, so the single-conformation model
    this is compared with uses exactly that one and the ensemble only adds to it.
    """
    from cmfqsar.conformers import generate_conformers, single_conformer

    try:
        ens = generate_conformers(mol, n=int(n), seed=int(seed))
    except Exception:
        ens = mol
    k = max(ens.GetNumConformers(), 1)
    out = []
    for i in range(k):
        one = single_conformer(ens, i) if ens.GetNumConformers() > 1 else ens
        out.append(build_fields(one, fieldset, name=f"{name}#{i}"))
    from cmfqsar.conformers import get_conformer_energies
    energies = get_conformer_energies(ens)
    return out, (None if energies is None or len(energies) != k else np.asarray(energies, float))


def run_conformers(key, n=CONFORMERS, rho=None, folds=10, seed=0, max_molecules=0,
                   results=RESULTS):
    """A local target against the conformations its molecules visit (plan item LP4).

    A measured shift is an average over the conformations a molecule visits, and the single
    geometry that distance geometry returns is a stand-in for that average.  Plan items LP2 and
    LP3 both named it as the weakest part of their measurement; this is the measurement of how
    much it costs.  The ensemble needs no new kernel: an environment is a field, so the mean of
    the per-conformation environments is a field too, and the kernel on it is the mean-map kernel
    (`cmfqsar.localenv.ensemble_environment`).
    """
    spec = SETS[key]
    rho = float(ENSEMBLE_RHO[key] if rho is None else rho)
    _say(f"=== {spec['what']}: conformations, rho {rho:g}")
    mols, fields, targets, fieldset = read_set(key, max_molecules)
    _say(f"{len(targets)} measured atoms in {targets.n_measured_molecules} molecules, "
         f"{n} conformations asked for")

    t0 = time.time()
    members, energies, kept = [], [], []
    for k, m in enumerate(mols):
        got, e = conformer_members(m, fieldset, n, seed + k, targets.name)
        members.append(got)
        energies.append(e)
        kept.append(len(got))
    _say(f"  conformations generated in {time.time() - t0:.0f} s: "
         f"{float(np.mean(kept)):.1f} per molecule on average, "
         f"{sum(1 for x in kept if x == 1)} molecules with only one")

    # which member is the lowest in energy, per molecule
    lowest = [int(np.argmin(e)) if e is not None and len(e) == len(members[k]) else 0
              for k, e in enumerate(energies)]
    _say(f"  the lowest-energy conformation is the given one for "
         f"{sum(1 for x in lowest if x == 0)} of {len(lowest)} molecules")

    rows = []
    for how in ("single", "lowest", "uniform", "boltzmann"):
        t0 = time.time()
        if how == "single":
            envs = [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
                    for r in range(len(targets))]
        elif how == "lowest":
            # one conformation again, but a different one: this separates what averaging does
            # from what simply changing the geometry does
            envs = [environment(members[int(targets.molecule[r])][lowest[int(targets.molecule[r])]],
                                int(targets.atom[r]), rho) for r in range(len(targets))]
        else:
            envs = []
            for r in range(len(targets)):
                k, i = int(targets.molecule[r]), int(targets.atom[r])
                w = member_weights(energies[k], len(members[k]), how)
                envs.append(ensemble_environment(members[k], i, rho, weights=w))
        res = choose_lam(local_kernel(local_features(envs)), targets, folds=folds, seed=seed,
                         name=f"local field, {how}")
        res.note = f"{np.mean([e.n_atoms for e in envs]):.0f} atoms per sample, " \
                   f"{time.time() - t0:.0f} s"
        rows.append(dict(how=how, q2=res.q2, error=res.error, mae=res.mae, lam=res.lam,
                         note=res.note,
                         atoms=float(np.mean([e.n_atoms for e in envs])),
                         result=res))
        _say("  " + res.summary())
    base = rows[0]["result"]
    for row in rows[1:]:
        d = paired_interval(row["result"], base, targets, runs=2000, seed=seed)
        row["against_single"] = d
        _say(f"    {row['how']} against the single conformation: {d[0]:+.3f} "
             f"[{d[1]:+.3f}, {d[2]:+.3f}]")
    out = dict(set=key, what=spec["what"], unit=spec["unit"], rho=rho, conformers=int(n),
               n_samples=len(targets), n_molecules=targets.n_measured_molecules, folds=folds,
               seed=seed, kept=dict(mean=float(np.mean(kept)), one=int(sum(1 for x in kept
                                                                          if x == 1))),
               lowest_is_given=int(sum(1 for x in lowest if x == 0)),
               rows=[{k: v for k, v in row.items() if k != "result"} for row in rows])
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, f"{key}_conformers.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def member_weights(energies, k, how):
    """Uniform weights, or Boltzmann weights from the energies the generator stored.

    The energies are the force field's, in kcal/mol, which is enough to order conformations and
    not enough to trust their populations; that is why both weightings are measured rather than
    one being chosen.
    """
    if how != "boltzmann" or k <= 1 or energies is None:
        return np.ones(k)
    e = np.asarray(energies, float)
    if len(e) != k or not np.isfinite(e).all():
        return np.ones(k)
    w = np.exp(-(e - float(np.min(e))) / 0.6)    # kcal/mol, about RT at 300 K
    return w if w.sum() > 0 else np.ones(k)


# --------------------------------------------------------------------------- #
# The reference a spectroscopist would use: a HOSE-code lookup (added 16 November 2026)
# --------------------------------------------------------------------------- #
#
# The count reference above is the one rule 1 asked for, but it is not how 13C shifts are
# predicted in practice.  The classical method is the HOSE code of Bremser (1978): describe the
# atom sphere by sphere of bonds, and predict the mean shift of the training atoms that share the
# deepest sphere the new atom has in common with any of them.  It needs no geometry and fits
# nothing.  RDKit's Morgan environments are such codes: one identifier per atom and radius.
#
# The stage also asks what the field models gain once that reference is in place, and whether
# the gain is the fields at all: each geometric descriptor is measured once with the five fields
# as its channels and once with five element channels (H, C, N, O, other) in their place, which
# keep the geometry and drop the physical chemistry.

#: the deepest sphere of bonds the lookup tries; the measurement reports 2 to HOSE_DEPTH
HOSE_DEPTH = 5
#: the depth reported as the reference (four spheres, the depth HOSE codes are usually cut at)
HOSE_MAIN = 4
#: element channels standing in for the five fields, with one more for every other element
ELEMENT_CHANNELS = ("H", "C", "N", "O")
#: the ridge grid of this stage reaches below the module's 1e-6, where a stored search stopped
WIDE_LAMS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
#: LP5's best three-distance settings (benchmarks/results/triangles/shifts.json, "best")
TRIANGLE = dict(r_max=5.0, step=0.5, sigma=0.35, rho=0.5, power=1)
#: the envelope width of LP3's best posed overlap on the aromatic ring carbons
POSED_RHO = 0.25


def environment_ids(mols, targets, depth=HOSE_DEPTH):
    """`ids[r, k]`: the Morgan environment identifier of measured atom `r` at radius `k`.

    Redundant environments are kept, so every atom has an identifier at every radius, even when
    its environment stopped growing before that radius.
    """
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(depth),
                                                    includeRedundantEnvironments=True)
    ids = np.zeros((len(targets), int(depth) + 1), dtype=np.int64)
    for k in np.unique(targets.molecule):
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetSparseCountFingerprint(mols[int(k)], additionalOutput=ao)
        table = {}
        for bit, hits in ao.GetBitInfoMap().items():
            for atom, rad in hits:
                table[(int(atom), int(rad))] = int(bit)
        for r in np.where(np.asarray(targets.molecule) == k)[0]:
            for rad in range(int(depth) + 1):
                ids[r, rad] = table[(int(targets.atom[r]), rad)]
    return ids


def hose_lookup(ids, y, folds):
    """Out-of-fold HOSE predictions, and the radius of the sphere each one was read at.

    For a test atom the deepest radius whose identifier occurs among the training atoms is
    used, and the prediction is the mean target of those training atoms; an atom that shares
    nothing, not even its own type, gets the training mean and a radius of -1.
    """
    y = np.asarray(y, float)
    pred = np.zeros(len(y))
    depth = np.full(len(y), -1, int)
    top = ids.shape[1] - 1
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        tables = []
        for rad in range(top + 1):
            sums = {}
            for i in train:
                s = sums.setdefault(int(ids[i, rad]), [0.0, 0])
                s[0] += y[i]
                s[1] += 1
            tables.append({key: s[0] / s[1] for key, s in sums.items()})
        fallback = float(np.mean(y[train]))
        for i in test:
            pred[i] = fallback
            for rad in range(top, -1, -1):
                v = tables[rad].get(int(ids[i, rad]))
                if v is not None:
                    pred[i], depth[i] = v, rad
                    break
    return pred, depth


def as_local_result(name, pred, targets, folds, note=""):
    """A `LocalResult` for predictions that came from no kernel, for `paired_interval`."""
    from cmfqsar.localenv import LocalResult
    from cmfqsar.model import r2_score, rmse

    y = np.asarray(targets.y, float)
    return LocalResult(name=name, predicted=np.asarray(pred, float), q2=float(r2_score(y, pred)),
                       error=float(rmse(y, pred)), mae=float(np.mean(np.abs(y - pred))),
                       lam=float("nan"), n_samples=len(y),
                       n_molecules=int(len(np.unique(targets.molecule))), folds=len(folds),
                       note=note, molecule=np.asarray(targets.molecule, int))


def element_environment(env, rho):
    """The same environment with five element channels in place of the five fields.

    The channels carry the envelope as the field weights do, so only what the channels say
    changes: which element sits where, instead of the charge, size, lipophilicity and hydrogen
    bonding of what sits there.
    """
    from cmfqsar.triangles import element_weights, with_weights
    return with_weights(env, element_weights(env, ELEMENT_CHANNELS, rho=rho))


def _row_of(res, **extra):
    out = dict(name=res.name, q2=float(res.q2), error=float(res.error), mae=float(res.mae),
               lam=(None if not np.isfinite(res.lam) else float(res.lam)))
    out.update(extra)
    return out


def _pair(a, b, targets, seed):
    d = paired_interval(a, b, targets, runs=2000, seed=seed)
    return [float(d[0]), float(d[1]), float(d[2])]


def _stored_q2(path, pick):
    """A q2 the earlier stages stored, for the control of this one; None if it is not there."""
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        return pick(json.load(fh))


def run_hose(folds=10, seed=0, max_molecules=0, results=RESULTS, triangles=True):
    """The HOSE lookup against the local models of LP2, LP3 and LP5, on the same folds."""
    from cmfqsar.localalign import overlap_kernel, posed, ring_frames
    from cmfqsar.triangles import TriangleSpec, triangle_kernel_matrix, with_weights

    out = dict(folds=folds, seed=seed, depth=HOSE_DEPTH, main=HOSE_MAIN,
               element_channels=list(ELEMENT_CHANNELS), lams=list(WIDE_LAMS),
               triangle=TRIANGLE, posed_rho=POSED_RHO)

    # 1. every measured carbon (LP2 and LP5)
    _say("=== HOSE lookup: every measured carbon")
    mols, fields, targets, fieldset = read_set("shifts", max_molecules)
    fl = targets.folds(folds, seed)
    ids = environment_ids(mols, targets)
    lookups = {}
    for top in range(2, HOSE_DEPTH + 1):
        pred, depth = hose_lookup(ids[:, :top + 1], targets.y, fl)
        res = as_local_result(f"HOSE lookup, up to {top} bonds", pred, targets, fl)
        lookups[top] = (res, depth)
        _say("  " + res.summary())
    hose, depth = lookups[HOSE_MAIN]
    types = types_of(fields, min_atoms=10)
    counts = choose_lam(count_kernel(count_features(fields, targets, types)), targets,
                        folds=folds, seed=seed, name="atom-centred counts")
    _say("  " + counts.summary())
    two = {}
    for rho in (DEFAULT_RHO, 1.0):
        two[rho] = choose_lam(local_kernel(local_features(plain_environments(fields, targets, rho))),
                              targets, lams=WIDE_LAMS, folds=folds, seed=seed,
                              name=f"two-point local field, rho {rho:g}")
        _say("  " + two[rho].summary())
    tri = {}
    if triangles:
        base = plain_environments(fields, targets, TRIANGLE["rho"])
        spec = TriangleSpec(r_max=TRIANGLE["r_max"], step=TRIANGLE["step"],
                            sigma=TRIANGLE["sigma"])
        nf = len(fieldset.specs)
        ne = len(ELEMENT_CHANNELS) + 1
        el = [element_environment(e, TRIANGLE["rho"]) for e in base]
        both = [with_weights(e, np.hstack([e.weights, x.weights])) for e, x in zip(base, el)]
        for label, envs, n, scaling in (("fields", base, nf, "field"),
                                        ("elements", el, ne, "field"),
                                        ("fields and elements", both, nf + ne, "none")):
            t0 = time.time()
            K = triangle_kernel_matrix(envs, spec, n_fields=n, power=TRIANGLE["power"],
                                       scaling=scaling)
            tri[label] = choose_lam(K, targets, lams=WIDE_LAMS, folds=folds, seed=seed,
                                    name=f"three-distance, {label}")
            tri[label].note = f"{time.time() - t0:.0f} s"
            _say("  " + tri[label].summary())
            del K
    shares = {str(k): float(np.mean(depth == k)) for k in range(-1, HOSE_MAIN + 1)}
    out["shifts"] = dict(
        n_samples=len(targets), n_molecules=targets.n_measured_molecules,
        spread=float(np.std(targets.y)),
        lookups=[_row_of(r, depth=int(k)) for k, (r, _d) in sorted(lookups.items())],
        deepest=shares, counts=_row_of(counts),
        two_point=[_row_of(two[rho], rho=float(rho)) for rho in sorted(two)],
        three_distance=[_row_of(r, channels=k) for k, r in tri.items()],
        against_hose=dict(
            counts=_pair(counts, hose, targets, seed),
            **{f"two_point_{rho:g}": _pair(two[rho], hose, targets, seed) for rho in two},
            **{f"three_distance_{k.replace(' ', '_')}": _pair(r, hose, targets, seed)
               for k, r in tri.items()}),
        fields_against_elements=(_pair(tri["fields"], tri["elements"], targets, seed)
                                 if tri else None),
        control=dict(
            two_point_default=[float(two[DEFAULT_RHO].q2), _stored_q2(
                os.path.join(results, "shifts.json"),
                lambda d: next(r["q2"] for r in d["rows"] if abs(r["rho"] - DEFAULT_RHO) < 1e-9))],
            two_point_1=[float(two[1.0].q2), _stored_q2(
                os.path.join(results, "shifts.json"),
                lambda d: next(r["q2"] for r in d["rows"] if abs(r["rho"] - 1.0) < 1e-9))],
            counts=[float(counts.q2), _stored_q2(os.path.join(results, "shifts.json"),
                                                 lambda d: d["reference"]["q2"])]))
    del fields

    # 2. the aromatic ring carbons of LP3, posed by the ring rule
    _say("=== HOSE lookup: aromatic ring carbons (LP3)")
    from run_local_cofield import aromatic_samples
    rmols, rnames, rt = aromatic_samples(max_molecules=max_molecules)
    rfl = rt.folds(folds, seed)
    fs = FieldSet.create()
    rfields = [build_fields(m, fs, name=n) for m, n in zip(rmols, rnames)]
    if len(fs.specs) != len(ELEMENT_CHANNELS) + 1:
        raise RuntimeError("the element channels stand in for the fields one for one; the field "
                           f"set now has {len(fs.specs)} fields")
    pred, rdepth = hose_lookup(environment_ids(rmols, rt, HOSE_MAIN), rt.y, rfl)
    rhose = as_local_result(f"HOSE lookup, up to {HOSE_MAIN} bonds", pred, rt, rfl)
    _say("  " + rhose.summary())
    rcounts = choose_lam(count_kernel(count_features(rfields, rt, types_of(rfields, min_atoms=10))),
                         rt, folds=folds, seed=seed, name="atom-centred counts")
    _say("  " + rcounts.summary())
    posed_rows = {}
    for label in ("fields", "elements"):
        t0 = time.time()
        envs = []
        for r in range(len(rt)):
            env = environment(rfields[int(rt.molecule[r])], int(rt.atom[r]), POSED_RHO)
            frames = ring_frames(env)
            if label == "elements":
                env = element_environment(env, POSED_RHO)
            envs.append(posed(env, frames=frames))
        _m, _t, K = overlap_kernel(envs, fs)
        posed_rows[label] = choose_lam(K, rt, lams=WIDE_LAMS, folds=folds, seed=seed,
                                       name=f"posed overlap, {label}, rho {POSED_RHO:g}")
        posed_rows[label].note = f"{time.time() - t0:.0f} s"
        _say("  " + posed_rows[label].summary())
        del K
    out["aromatic"] = dict(
        n_samples=len(rt), n_molecules=rt.n_measured_molecules, spread=float(np.std(rt.y)),
        hose=_row_of(rhose, depth=HOSE_MAIN),
        deepest={str(k): float(np.mean(rdepth == k)) for k in range(-1, HOSE_MAIN + 1)},
        counts=_row_of(rcounts),
        posed=[_row_of(r, channels=k) for k, r in posed_rows.items()],
        against_hose=dict(counts=_pair(rcounts, rhose, rt, seed),
                          **{f"posed_{k}": _pair(r, rhose, rt, seed)
                             for k, r in posed_rows.items()}),
        fields_against_elements=_pair(posed_rows["fields"], posed_rows["elements"], rt, seed),
        control=dict(posed_fields=[float(posed_rows["fields"].q2), _stored_q2(
            os.path.join(HERE, "results", "local_cofield", "summary.json"),
            lambda d: next((r["aligned"]["q2"] for r in d.get("rows", [])
                            if abs(r["rho"] - POSED_RHO) < 1e-9), None))]))
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "hose.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


#: shares of each fold's training molecules the learning curve trains on
CURVE_SHARES = (0.25, 0.5, 1.0)
#: random draws of the training molecules per share
CURVE_DRAWS = 5


def _subsampled_predictions(K, y, molecule, folds, share, lam, rng):
    """Out-of-fold predictions when each fold trains on a random share of its training molecules.

    The test part of every fold is the same as in the full cross-validation; only the training
    part shrinks, and it shrinks by whole molecules, so no molecule is split.
    """
    from cmfqsar.model import krr_fit

    pred = np.zeros(len(y))
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        mols = np.unique(molecule[train])
        keep = rng.choice(mols, max(2, int(round(share * len(mols)))), replace=False)
        tr = train[np.isin(molecule[train], keep)]
        coef, bias = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
        pred[test] = K[np.ix_(test, tr)] @ coef + bias
    return pred, tr


def _subsampled_lookup(ids, y, molecule, folds, share, rng):
    """The HOSE lookup with the same shrunken training parts."""
    pred = np.zeros(len(y))
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        mols = np.unique(molecule[train])
        keep = rng.choice(mols, max(2, int(round(share * len(mols)))), replace=False)
        tr = train[np.isin(molecule[train], keep)]
        sub_ids = np.vstack([ids[tr], ids[test]])
        sub_y = np.concatenate([y[tr], np.zeros(len(test))])
        p, _ = hose_lookup(sub_ids, sub_y, [np.arange(len(tr), len(tr) + len(test))])
        pred[test] = p[len(tr):]
    return pred


def run_hose_curve(folds=10, seed=0, max_molecules=0, results=RESULTS,
                   shares=CURVE_SHARES, draws=CURVE_DRAWS):
    """Fields against element channels, and both against the lookup, as the training set grows.

    On LP3's ring carbons the two channel sets tie on the full set, but a live run on half of the
    file put the fields ahead; scored on the same molecules, models trained on the full set tie
    again, so the difference belongs to the size of the training set. This measures it: each
    fold trains on a random share of its training molecules, the test parts stay the same, and
    the ridge of each model and share is the one that cross-validates best, as everywhere here.
    """
    from cmfqsar.localalign import overlap_kernel, posed, ring_frames
    from cmfqsar.model import r2_score

    _say("=== learning curve: aromatic ring carbons, fields against element channels")
    from run_local_cofield import aromatic_samples
    rmols, rnames, rt = aromatic_samples(max_molecules=max_molecules)
    fl = rt.folds(folds, seed)
    y = np.asarray(rt.y, float)
    molecule = np.asarray(rt.molecule, int)
    fs = FieldSet.create()
    rfields = [build_fields(m, fs, name=n) for m, n in zip(rmols, rnames)]
    kernels = {}
    for label in ("fields", "elements"):
        t0 = time.time()
        envs = []
        for r in range(len(rt)):
            env = environment(rfields[int(rt.molecule[r])], int(rt.atom[r]), POSED_RHO)
            frames = ring_frames(env)
            if label == "elements":
                env = element_environment(env, POSED_RHO)
            envs.append(posed(env, frames=frames))
        kernels[label] = overlap_kernel(envs, fs)[2]
        _say(f"  {label} kernel in {time.time() - t0:.0f} s")
    ids = environment_ids(rmols, rt, HOSE_MAIN)

    rows = []
    for share in shares:
        per = dict(fields=[], elements=[], hose=[], difference=[], fields_vs_hose=[],
                   train_molecules=[])
        for d in range(draws):
            preds = {}
            for label, K in kernels.items():
                best = None
                for lam in WIDE_LAMS:
                    rng = np.random.RandomState(seed + 1000 * d + 1)
                    p, tr = _subsampled_predictions(K, y, molecule, fl, share, lam, rng)
                    q = r2_score(y, p)
                    if best is None or q > best[0]:
                        best = (q, p, lam)
                preds[label] = best[1]
                per[label].append(float(best[0]))
            rng = np.random.RandomState(seed + 1000 * d + 1)
            preds["hose"] = _subsampled_lookup(ids, y, molecule, fl, share, rng)
            per["hose"].append(float(r2_score(y, preds["hose"])))
            per["train_molecules"].append(int(round(share * len(np.unique(molecule)) * (folds - 1)
                                                   / folds)))
            per["difference"].append(_pair(as_local_result("f", preds["fields"], rt, fl),
                                           as_local_result("e", preds["elements"], rt, fl),
                                           rt, seed))
            per["fields_vs_hose"].append(_pair(as_local_result("f", preds["fields"], rt, fl),
                                               as_local_result("h", preds["hose"], rt, fl),
                                               rt, seed))
        row = dict(share=float(share), train_molecules=int(np.median(per["train_molecules"])),
                   fields=float(np.mean(per["fields"])), elements=float(np.mean(per["elements"])),
                   hose=float(np.mean(per["hose"])),
                   difference=[float(np.mean([x[0] for x in per["difference"]])),
                               float(min(x[0] for x in per["difference"])),
                               float(max(x[0] for x in per["difference"]))],
                   clear_for_fields=int(sum(1 for x in per["difference"] if x[1] > 0)),
                   clear_for_elements=int(sum(1 for x in per["difference"] if x[2] < 0)),
                   fields_vs_hose=float(np.mean([x[0] for x in per["fields_vs_hose"]])),
                   draws=[dict(fields=f, elements=e, hose=h, difference=dd)
                          for f, e, h, dd in zip(per["fields"], per["elements"], per["hose"],
                                                 per["difference"])])
        rows.append(row)
        _say(f"  share {share:g} (about {row['train_molecules']} training molecules): fields "
             f"{row['fields']:+.3f}, elements {row['elements']:+.3f}, lookup {row['hose']:+.3f}; "
             f"fields minus elements {row['difference'][0]:+.3f} (draws "
             f"{row['difference'][1]:+.3f} to {row['difference'][2]:+.3f}; interval clear for "
             f"the fields in {row['clear_for_fields']} of {draws}, for the elements in "
             f"{row['clear_for_elements']})")
    out = dict(n_samples=len(rt), n_molecules=rt.n_measured_molecules, folds=folds, seed=seed,
               draws=draws, rho=POSED_RHO, rows=rows)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "hose_curve.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _curve_lines(results=RESULTS):
    path = os.path.join(results, "hose_curve.json")
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as fh:
        c = json.load(fh)
    out = ["### The same ring carbons as the training set grows", "",
           f"Each fold trains on a random share of its training molecules; the test parts are the "
           f"same as above, and each model's ridge is the one that cross-validates best at that "
           f"share. {c['draws']} random draws per share; the q2 values are their means. The last "
           f"column counts the draws whose 95 % interval, bootstrapped over molecules, excludes "
           f"zero.", "",
           "| share of the training molecules | training molecules per fold | HOSE lookup | posed "
           "overlap, element channels | posed overlap, fields | fields minus elements: mean "
           "(lowest to highest draw) | draws clear for the fields / for the elements |",
           "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in c["rows"]:
        d = r["difference"]
        out.append(f"| {r['share']:g} | {r['train_molecules']} | {r['hose']:+.3f} | "
                   f"{r['elements']:+.3f} | {r['fields']:+.3f} | {d[0]:+.3f} ({d[1]:+.3f} to "
                   f"{d[2]:+.3f}) | {r['clear_for_fields']} / {r['clear_for_elements']} |")
    small, full = c["rows"][0], c["rows"][-1]
    out += ["", f"With {small['train_molecules']} training molecules per fold the fields lead "
                f"the element channels by {small['difference'][0]:+.3f} on average and the lookup "
                f"by {small['fields_vs_hose']:+.3f}; with {full['train_molecules']} the lead over "
                f"the elements is {full['difference'][0]:+.3f} and over the lookup "
                f"{full['fields_vs_hose']:+.3f}. "
            + ("The fields are worth something exactly where data are scarce: they say in "
               "advance that atoms of similar charge, size and lipophilicity act alike, which "
               "the element channels have to learn from examples."
               if small["difference"][0] > 0 and small["clear_for_fields"] > c["draws"] // 2
               and small["difference"][0] > full["difference"][0] else
               f"The curve does not show a lead for the fields that grows as the data shrink: at "
               f"{small['train_molecules']} molecules the draws run from "
               f"{small['difference'][1]:+.3f} to {small['difference'][2]:+.3f}, and the interval "
               f"is clear for the fields in {small['clear_for_fields']} of {c['draws']} draws. "
               f"What does grow is the lead of either geometric model over the lookup, which "
               f"needs a near copy of an atom's surroundings among the training atoms and finds "
               f"fewer of them in a smaller set."), ""]
    return out


def _hose_lines(results=RESULTS):
    """The report section of the HOSE stage, or nothing when it has not been run."""
    path = os.path.join(results, "hose.json")
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as fh:
        h = json.load(fh)

    def cell(d):
        return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"

    def lam(r):
        return "--" if r.get("lam") is None else f"{r['lam']:g}"

    s, a = h["shifts"], h["aromatic"]
    main = next(r for r in s["lookups"] if r["depth"] == h["main"])
    out = ["## The reference a spectroscopist would use: a HOSE-code lookup", "",
           "Added 16 November 2026. The count reference above is the one rule 1 asked for, but it "
           "is not how 13C shifts are predicted in practice. The classical method is the HOSE "
           "code (Bremser 1978): describe the atom sphere by sphere of bonds, and predict the "
           "mean shift of the training atoms that share the deepest sphere the new atom has in "
           "common with any of them. It needs no geometry and fits nothing. The codes here are "
           "RDKit's Morgan environments, one identifier per atom and number of bonds. Every model "
           "below uses the same folds, which hold out whole molecules, and the field models a "
           f"ridge grid that reaches down to {min(h['lams']):g}. The differences are in q2, "
           "against the lookup up to "
           f"{h['main']} bonds, with a 95 % interval bootstrapped over molecules.", "",
           f"### Every measured carbon: {s['n_samples']} in {s['n_molecules']} molecules", "",
           "| model | q2 | error (ppm) | mean absolute error (ppm) | ridge | difference against "
           "the lookup |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in s["lookups"]:
        out.append(f"| {r['name']} | {r['q2']:+.3f} | {r['error']:.2f} | {r['mae']:.2f} | -- | "
                   f"-- |")
    c = s["counts"]
    out.append(f"| {c['name']} | {c['q2']:+.3f} | {c['error']:.2f} | {c['mae']:.2f} | {lam(c)} | "
               f"{cell(s['against_hose']['counts'])} |")
    for r in s["two_point"]:
        key = "two_point_%g" % r["rho"]
        out.append(f"| {r['name']} | {r['q2']:+.3f} | {r['error']:.2f} | {r['mae']:.2f} | "
                   f"{lam(r)} | {cell(s['against_hose'][key])} |")
    for r in s["three_distance"]:
        key = "three_distance_" + r["channels"].replace(" ", "_")
        out.append(f"| {r['name']} (r_max {h['triangle']['r_max']:g} A, rho "
                   f"{h['triangle']['rho']:g}, sigma {h['triangle']['sigma']:g}) | "
                   f"{r['q2']:+.3f} | {r['error']:.2f} | {r['mae']:.2f} | {lam(r)} | "
                   f"{cell(s['against_hose'][key])} |")
    share = ", ".join(f"{100 * v:.1f} % at {k} bonds" for k, v in s["deepest"].items()
                      if float(v) > 0 and k != "-1")
    out += ["", f"The lookup up to {h['main']} bonds read its prediction at: {share}."]
    if s.get("fields_against_elements"):
        out.append(f"In the three-distance kernel the five fields against five element channels "
                   f"(H, C, N, O, other) in their place: "
                   f"{cell(s['fields_against_elements'])}.")
    ctrl = s["control"]
    ok = all(v[1] is not None and abs(v[0] - v[1]) < 5e-4 for v in ctrl.values())
    out += ["", "Control: the counts and the two-point rows of this stage against the same rows "
                "stored by the stage above, " + ", ".join(
                    f"{k.replace('_', ' ')} {v[0]:+.3f} against {v[1]:+.3f}" if v[1] is not None
                    else f"{k.replace('_', ' ')} {v[0]:+.3f} (nothing stored)"
                    for k, v in ctrl.items())
            + (" -- the same." if ok else " -- NOT the same; read the rows with care."), ""]

    out += [f"### Aromatic ring carbons (plan item LP3): {a['n_samples']} in "
            f"{a['n_molecules']} molecules", "",
            "The ring carbons and the pose of LP3: the carbon at the origin, the bond that leaves "
            "the ring on +x, the ring in the xy plane, both mirror images averaged. The element "
            "channels are swapped in before the pose, so both rows go through the same frames.",
            "",
            "| model | q2 | error (ppm) | mean absolute error (ppm) | ridge | difference against "
            "the lookup |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| {a['hose']['name']} | {a['hose']['q2']:+.3f} | {a['hose']['error']:.2f} | "
            f"{a['hose']['mae']:.2f} | -- | -- |",
            f"| {a['counts']['name']} | {a['counts']['q2']:+.3f} | {a['counts']['error']:.2f} | "
            f"{a['counts']['mae']:.2f} | {lam(a['counts'])} | {cell(a['against_hose']['counts'])} |"]
    for r in a["posed"]:
        out.append(f"| {r['name']} | {r['q2']:+.3f} | {r['error']:.2f} | {r['mae']:.2f} | "
                   f"{lam(r)} | {cell(a['against_hose']['posed_' + r['channels']])} |")
    pc = a["control"]["posed_fields"]
    out += ["", f"The five fields against five element channels in the same posed overlap: "
                f"{cell(a['fields_against_elements'])}. Control: the posed overlap with the "
                f"fields gives q2 {pc[0]:+.3f} here"
            + (f" and {pc[1]:+.3f} in `results/local_cofield/`." if pc[1] is not None
               else "; `results/local_cofield/` stores no row to compare with."), ""]

    # the reading, generated from the numbers
    two_best = max(s["two_point"], key=lambda r: r["q2"])
    d2 = s["against_hose"][f"two_point_{two_best['rho']:g}"]
    words = []
    words.append(
        "**The two-point field of LP2 "
        + ("beats" if d2[1] > 0 else "loses to" if d2[2] < 0 else "does not beat")
        + f" the lookup** at its best width ({cell(d2)}). The count reference, which it beat "
          f"by a wide margin, is itself far behind the lookup ({cell(s['against_hose']['counts'])}"
          f"), so that margin measured a weak reference.")
    if s["three_distance"]:
        dt = s["against_hose"]["three_distance_fields"]
        fe = s["fields_against_elements"]
        words.append(
            "**The three-distance form of LP5 "
            + ("beats" if dt[1] > 0 else "does not beat") + f" it** ({cell(dt)}), "
            + ("and the fields are not why: element channels in their place do as well "
               if fe[1] <= 0 <= fe[2] else "and the fields add to element channels ")
            + f"({cell(fe)}).")
    dp = a["against_hose"]["posed_fields"]
    fe2 = a["fields_against_elements"]
    words.append(
        "**On the aromatic ring carbons the posed overlap "
        + ("beats" if dp[1] > 0 else "does not beat") + f" the lookup by {cell(dp)}**, "
        + ("and again element channels do as well as the fields "
           if fe2[1] <= 0 <= fe2[2] else "and the fields add to element channels ")
        + f"({cell(fe2)}).")
    out += ["**What this shows.** " + " ".join(words) + " What predicts a shift beyond the "
            "lookup is where the atoms around the carbon sit, which the envelope, the pose and the "
            "three-distance kernel all carry. Put in place of the element of each atom, what the "
            "fields say about it does no better. (Added to the elements, the fields gain a little "
            "in the three-distance kernel: +0.002 [+0.001, +0.004] in "
            "`benchmarks/results/triangles/`.)", ""]
    return out


def _verdict(data):
    """Does the local kernel beat the count reference by more than the interval?"""
    fixed = [r for r in data["rows"] if abs(r["rho"] - data["fixed_rho"]) < 1e-9]
    best = max(data["rows"], key=lambda r: r["q2"])
    pairs = [("at the module's default width", r) for r in fixed]
    if best not in fixed:
        pairs.append(("at the best width of the scan", best))
    out = []
    for label, r in pairs:
        if not np.isfinite(r["low"]):
            word = "cannot be judged: too few molecules for an interval"
        elif r["low"] > 0:
            word = "beats the counts"
        elif r["high"] < 0:
            word = "loses to the counts"
        else:
            word = "is no better than the counts"
        out.append(f"{label} (rho {r['rho']:g}, {r['kind']}) it {word}: "
                   f"{r['difference']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]")
    return out


def report(results=RESULTS):
    out = ["# The field around an atom against a count around an atom (plan item LP2)", "",
           "The sample is an atom in a molecule. The local descriptor is the molecule's fields "
           "under a Gaussian envelope about that atom, turned into the rotation-invariant "
           "autocorrelation, so no alignment is needed. The reference of rule 1 is an "
           "atom-centred count: the type of the centre and the atom types in distance shells "
           "about it, which is what a per-atom increment scheme is. The folds hold out whole "
           "molecules, and the interval is bootstrapped over molecules.", ""]
    for key in ("shifts", "pka"):
        path = os.path.join(results, f"{key}.json")
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        ref = d["reference"]
        out += [f"## {d['what']}", "",
                f"{d['n_samples']} measured atoms in {d['n_molecules']} molecules; the target "
                f"runs from {d['y']['low']:.2f} to {d['y']['high']:.2f} {d['unit']} with a "
                f"standard deviation of {d['y']['spread']:.2f}. {d['folds']} folds holding out "
                f"whole molecules.", "",
                "| descriptor | envelope width rho | q2 | error | mean absolute error | ridge | "
                "atoms per sample | mean kernel between two samples | difference in q2 against "
                "the counts |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {ref['name']} ({ref['note']}) | -- | {ref['q2']:+.3f} | {ref['error']:.3f} | "
                f"{ref['mae']:.3f} | {ref['lam']:g} | -- | {ref.get('alike', float('nan')):.3f} | "
                f"-- |"]
        for r in d["rows"]:
            out.append(f"| {r['kind']} | {r['rho']:g} | {r['q2']:+.3f} | {r['error']:.3f} | "
                       f"{r['mae']:.3f} | {r['lam']:g} | {r['atoms_per_environment']:.1f} | "
                       f"{r.get('alike', float('nan')):.3f} | "
                       f"{r['difference']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] |")
        out += ["", f"The error is in {d['unit']}. A mean kernel close to 1 says the descriptor "
                    f"cannot tell two samples apart, whatever the regression does with it.", ""]
        for line in _verdict(d):
            out.append(f"* {line}")
        out.append("")
    for key in ("shifts", "pka"):
        path = os.path.join(results, f"{key}_conformers.json")
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="utf-8") as fh:
            c = json.load(fh)
        out += [f"## {c['what']}: the conformations the molecules visit", "",
                f"A measured value of this kind is an average over the conformations a molecule "
                f"visits, and the single geometry that distance geometry returns is a stand-in "
                f"for that average. {c['conformers']} conformations were asked for per molecule "
                f"and {c['kept']['mean']:.1f} were kept on average "
                f"({c['kept']['one']} molecules gave only one); the deposited geometry is the "
                f"first of them, so the single-conformation row below is exactly the model the "
                f"sections above measured. Envelope width rho {c['rho']:g}, "
                f"{c['n_samples']} atoms in {c['n_molecules']} molecules, {c['folds']} folds "
                f"holding out whole molecules.", "",
                "| conformations | weighting | q2 | error (" + c["unit"] + ") | ridge | atoms "
                "per sample | difference in q2 against one conformation |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in c["rows"]:
            d = r.get("against_single")
            cell = (f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else "--")
            how = ("one, as prepared" if r["how"] == "single" else
                   "one, lowest in energy" if r["how"] == "lowest" else
                   f"{c['conformers']} asked for")
            weighting = ("--" if r["how"] in ("single", "lowest") else r["how"])
            out.append(f"| {how} | {weighting} | {r['q2']:+.3f} | {r['error']:.3f} | "
                       f"{r['lam']:g} | {r['atoms']:.0f} | {cell} |")
        ens = [r for r in c["rows"] if r["how"] in ("uniform", "boltzmann")
               and r.get("against_single")]
        low = next((r for r in c["rows"] if r["how"] == "lowest" and r.get("against_single")),
                   None)
        if ens:
            best = max(ens, key=lambda r: r["against_single"][0])
            d = best["against_single"]
            word = ("helps" if d[1] > 0 else "costs" if d[2] < 0 else
                    "changes nothing that can be measured")
            out += ["", f"Averaging over the conformations {word}: the better of the two "
                        f"weightings is {best['how']}, {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] "
                        "against the conformation the set was prepared with."]
            if low is not None:
                e = low["against_single"]
                out += ["", "The row that says whether that is about averaging or about the "
                            "geometries themselves is the second one: a single conformation "
                            "again, the lowest in energy of the same pool, which scores "
                            f"{e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}]. "
                        + ("Changing which single conformation is used costs about as much as "
                           "averaging does, so what the ensemble adds is not the averaging but "
                           "the arbitrariness of the geometries it averages."
                           if e[2] < 0 and abs(e[0]) > 0.4 * abs(d[0]) else
                           "One conformation is one conformation whichever it is, so the loss "
                           "belongs to the averaging rather than to the geometries.")]
            out.append("")
    out += _hose_lines(results)
    out += _curve_lines(results)
    return "\n".join(out)


def write_table(results=RESULTS) -> str:
    """Rebuild ``table.md`` from the result files and return its text."""
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="which", nargs="*", default=["shifts", "pka"],
                    choices=sorted(SETS))
    ap.add_argument("--rho", nargs="*", type=float, default=None,
                    help="envelope width parameters to try, in A^-2; the default is the list the "
                         "set names")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-molecules", type=int, default=0,
                    help="use only the first n molecules of each set")
    ap.add_argument("--conformers", type=int, default=CONFORMERS,
                    help="conformations per molecule for the ensemble half (plan item LP4)")
    ap.add_argument("--only", default="all",
                    choices=["all", "widths", "conformers", "hose", "hose-curve"],
                    help="which part to compute: the envelope-width scan, the conformations, the "
                         "HOSE-code reference against the local models of LP2, LP3 and LP5 "
                         "with its learning curve (about an hour and a half), the learning curve "
                         "alone, or all of them (default)")
    ap.add_argument("--skip-triangles", action="store_true",
                    help="in the HOSE part, leave out the three-distance kernels, which take "
                         "most of its time")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    os.makedirs(RESULTS, exist_ok=True)
    for key in a.which:
        if a.only in ("all", "widths"):
            run_set(key, rhos=a.rho, folds=a.folds, seed=a.seed, max_molecules=a.max_molecules)
        if a.only in ("all", "conformers"):
            run_conformers(key, n=a.conformers, folds=a.folds, seed=a.seed,
                           max_molecules=a.max_molecules)
    if a.only in ("all", "hose"):
        run_hose(folds=a.folds, seed=a.seed, max_molecules=a.max_molecules,
                 triangles=not a.skip_triangles)
    if a.only in ("all", "hose", "hose-curve"):
        run_hose_curve(folds=a.folds, seed=a.seed, max_molecules=a.max_molecules)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
