#!/usr/bin/env python
"""
A co-field around one atom: where, and in which field, a feature moves its shift (plan item LP3)

Plan item LP2 made the sample an atom and its surroundings and measured it with the
rotation-invariant autocorrelation, which needs no frame.  That kernel cannot say **where** around
the atom a feature acts, for the same reason the invariant kernel for whole molecules cannot: with
no frame there is no co-field.

For one site the frame costs nothing.  An aromatic carbon of a six-membered ring, with its two
ring neighbours, fixes a frame to within a mirror image: put the carbon at the origin, the bond
that leaves the ring on +x, the ring in the xy plane, and keep both mirror images
(`cmfqsar/localalign.py`).  Every such carbon of every molecule is then in one frame, with nothing
fitted and nothing to converge, and the ordinary overlap kernel and co-field of the program apply
to the environment exactly as they apply to a molecule.

This script asks two things of that.

1. **Does the posed overlap kernel predict a shift as well as the invariant one?**  Same samples,
   same folds holding out whole molecules, same ridge search, and the atom-centred count of plan
   item LP2 as the reference of rule 1.
2. **Does the co-field sit where substituent effects on an aromatic carbon are known to sit?**
   In the frame above, the positions a substituent occupies on the ring are fixed numbers
   (`localalign.RING_POSITIONS`): ipso at 1.4 A on +x, then ortho, meta and para.  The effect of a
   substituent on an aromatic carbon alternates with the ring position -- ipso, ortho and para
   conjugate with the measured carbon and meta does not, which is why the substituent-induced
   shifts of a group such as methoxy are large at ipso, ortho and para (+31, -14, -8 ppm) and
   almost nothing at meta (+1 ppm).  So the co-field should be large at the ipso, ortho and para
   positions and small at the meta ones.  The statistic is

       alternation = (|C(ortho)| + |C(para)|) / 2 - |C(meta)|, divided by their sum,

   which is +1 when the meta position carries nothing and -1 when it carries everything.  The null
   is a y-scramble: the same geometry, the same frame and the same kernel, with the shifts
   shuffled between the carbons, which destroys only which carbon has which shift.

Usage
-----
    python benchmarks/run_local_cofield.py
    python benchmarks/run_local_cofield.py --rho 0.05 0.1 --cofield-rho 0.1
    python benchmarks/run_local_cofield.py --max-molecules 40      # a quick look
    python benchmarks/run_local_cofield.py --report
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

from cmfqsar.atomtargets import AtomTargets, read_atom_target  # noqa: E402
from cmfqsar.cofields import (atom_effect, atom_effect_covariance,  # noqa: E402
                              cofield_at_points, contrast_at, dual_covariance_factor)
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields, cif_types_in  # noqa: E402
from cmfqsar.localalign import (RING_POSITIONS, overlap_kernel, posed_environments,  # noqa: E402
                                ring_frames, well_posed)
from cmfqsar.localenv import (choose_lam, count_features, count_kernel,  # noqa: E402
                              environment, local_features, local_kernel,
                              paired_interval, types_of)

DATA = os.path.join(ROOT, "examples", "local", "shifts_cdcl3.sdf")
RESULTS = os.path.join(HERE, "results", "local_cofield")
#: envelope widths tried.  The list reaches 1.0 A^-2, which is a width of 1.0 A, because that
#: is where the invariant local kernel of plan item LP2 was best on these shifts: the
#: comparison has to show that kernel at its own best, not only where the co-field needs it.
RHOS = (0.05, 0.1, 0.25, 0.5, 1.0)
COFIELD_RHO = 0.1
BENZENE = Chem.MolFromSmarts("c1ccccc1")
#: measured substituent-induced 13C shifts on benzene, in ppm, by ring position, for one
#: pi-donor and one pi-acceptor (the values tabulated in Pretsch, Buhlmann and Badertscher,
#: Structure Determination of Organic Compounds).  The first atom of the donor carries a negative
#: partial charge and the first atom of the acceptor a positive one, so a shift that followed the
#: charge alone would give the two rows opposite signs; it does not, which is the resonance part
#: that fragment charges cannot see and which the Hammett co-fields of this repository ran into
#: for sigma_p.
SCS = {"methoxy (donor)": {"ipso": 31.4, "ortho": -14.4, "meta": 1.0, "para": -7.7},
       "nitro (acceptor)": {"ipso": 19.9, "ortho": -4.9, "meta": 0.9, "para": 6.1}}
#: the sign a co-field needs to reproduce a row of SCS: a donor's first atom is negative, so the
#: co-field has to be minus the shift the donor causes; an acceptor's is positive, so plus
SCS_SIGN = {"methoxy (donor)": -1.0, "nitro (acceptor)": +1.0}
POSITIONS = ("ipso", "ortho", "meta", "para")
SCRAMBLES = 20


#: measured substituent-induced 13C shifts on benzene, in ppm, by ring position, for substituents
#: whose first atom carries these Sybyl types.  Transcribed from the standard tables (Pretsch,
#: Buhlmann and Badertscher, Structure Determination of Organic Compounds); as with the Hammett
#: constants of `examples/hammett`, check a number against the book before quoting it.  The point
#: of the list is the contrast between the rows, which is not in doubt: a donor (O.3, N.pl3)
#: lowers the para carbon and an acceptor (C.2, as in acetophenone) raises it, while both raise
#: the carbon they sit on.
TYPE_SCS = {
    "O.3": ((31.4, -14.4, 1.0, -7.7), "methoxy (anisole)"),
    "N.pl3": ((18.0, -13.3, 0.9, -9.8), "amino (aniline)"),
    "C.3": ((9.3, 0.7, -0.1, -2.9), "methyl (toluene)"),
    "C.2": ((9.3, 0.2, 0.2, 4.2), "acetyl (acetophenone)"),
    "Cl": ((6.2, 0.4, 1.3, -1.9), "chlorine (chlorobenzene)"),
    "Br": ((-5.4, 3.3, 2.2, -1.0), "bromine (bromobenzene)"),
}
#: the envelope width the indicator co-fields are read at: wider than the one the physico-chemical
#: co-fields use, because the para position is 4.2 A from the centre and an envelope that damps it
#: to nothing cannot be asked what sits there
INDICATOR_RHO = 0.05
#: a type needs at least this many samples carrying it at a position before that number is read
MIN_SEEN = 20


def _say(text):
    print(text, flush=True)


def aromatic_samples(path=DATA, prop="shift", max_molecules=0):
    """The ring carbons the site rule applies to, with their shifts.

    A carbon counts when it is aromatic, belongs to an all-carbon six-membered aromatic ring and
    has exactly two aromatic neighbours.  The last condition leaves out the carbons where two
    rings meet, which have three and whose frame the ring rule cannot fix.
    """
    mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    if max_molecules:
        mols = mols[:max_molecules]
    names = [m.GetProp("_Name") if m.HasProp("_Name") else f"m{k}" for k, m in enumerate(mols)]
    per, mol_i, atom_i, vals = [], [], [], []
    for k, m in enumerate(mols):
        v = read_atom_target(m, prop)
        rings = [set(r) for r in m.GetSubstructMatches(BENZENE)]
        keep = np.full(len(v), np.nan)
        for i in np.where(np.isfinite(v))[0]:
            i = int(i)
            a = m.GetAtomWithIdx(i)
            if not a.GetIsAromatic() or not any(i in r for r in rings):
                continue
            if sum(1 for nb in a.GetNeighbors() if nb.GetIsAromatic()) != 2:
                continue
            keep[i] = v[i]
            mol_i.append(k)
            atom_i.append(i)
            vals.append(float(v[i]))
        per.append(keep)
    targets = AtomTargets(np.asarray(mol_i, int), np.asarray(atom_i, int), np.asarray(vals, float),
                          len(mols), prop, per)
    return mols, names, targets


def cofield_at(model, f, points):
    """h_f c_f C_f at a few points: what the prediction gains per unit of field there."""
    return cofield_at_points(model, f, points, include_scale=True)


def position_errors(model, f, pts, values, sigma):
    """What the noise in the shifts allows: the error of each position and of the comparisons.

    The co-field is linear in the activities, so noise of level `sigma` in them propagates to it
    in closed form (`cmfqsar.cofields.cofield_covariance_at`).  Two nearby positions are built
    from the same training fields and their errors are correlated, which is why the comparisons
    are computed as weighted combinations rather than from the two error bars separately.
    """
    s = {k: float(np.sign(values[k]) or 1.0) for k in POSITIONS}
    out = {}
    for k in POSITIONS:
        w = np.zeros(4)
        w[POSITIONS.index(k)] = s[k]
        _v, sd = contrast_at(model, f, pts, w, sigma=sigma)
        out["sd_" + k] = sd
        out["z_" + k] = abs(values[k]) / sd if sd > 0 else float("nan")
    # |para| - |meta| and (|ortho| + |para|)/2 - |meta|, each with its own error
    for name, w in (("para_minus_meta", {"para": 1.0, "meta": -1.0}),
                    ("conjugated_minus_meta", {"ortho": 0.5, "para": 0.5, "meta": -1.0})):
        w_vec = np.zeros(4)
        for k, c in w.items():
            w_vec[POSITIONS.index(k)] = c * s[k]
        v, sd = contrast_at(model, f, pts, w_vec, sigma=sigma)
        out[name] = v
        out[name + "_sd"] = sd
        out[name + "_z"] = v / sd if sd > 0 else float("nan")
    return out


def alternation(values):
    """(ortho + para)/2 against meta, as a number in [-1, 1]; +1 means meta carries nothing."""
    a = 0.5 * (abs(values["ortho"]) + abs(values["para"]))
    b = abs(values["meta"])
    return float((a - b) / (a + b)) if (a + b) > 0 else 0.0


def para_over_meta(values):
    """Para against meta, in [-1, 1].  The distance works against para, which is the point.

    The para position is further from the measured carbon (4.19 A) than the meta one (3.70 A), so
    the envelope and the overlap integral both favour meta.  A co-field that is nonetheless larger
    at para says something the geometry does not: that the para position conjugates with the
    measured carbon and the meta position does not.
    """
    a, b = abs(values["para"]), abs(values["meta"])
    return float((a - b) / (a + b)) if (a + b) > 0 else 0.0


def scs_match(values, row, sign):
    """How well a co-field follows a measured row of substituent-induced shifts.

    `sign` is -1 for a donor, whose first atom is negatively charged, and +1 for an acceptor.
    Returns the correlation over the four positions and how many of the four signs agree.
    """
    c = np.array([values[k] for k in POSITIONS], float)
    want = sign * np.array([row[k] for k in POSITIONS], float)
    agree = int(np.sum(np.sign(c) == np.sign(want)))
    if np.std(c) <= 0 or np.std(want) <= 0:
        return float("nan"), agree
    return float(np.corrcoef(c, want)[0, 1]), agree


def _statistics(values):
    """Every statistic read off one co-field, as a flat dictionary."""
    out = {k: float(values[k]) for k in POSITIONS}
    out["alternation"] = alternation(values)
    out["para_over_meta"] = para_over_meta(values)
    for label, row in SCS.items():
        r, agree = scs_match(values, row, SCS_SIGN[label])
        out["r_" + label] = r
        out["signs_" + label] = agree
    return out


#: the statistics a y-scramble gives a null for
NULLED = ("alternation", "para_over_meta") + tuple("r_" + k for k in SCS)


def ring_cofields(model, table, y, lam, names, sigma, runs=SCRAMBLES, seed=0):
    """The co-field at the four ring positions, per field, with a y-scramble null.

    The scramble keeps the geometry, the frame, the kernel and the ridge and shuffles the shifts
    between the carbons, so it says what each statistic is worth before any chemistry: a position
    nearer the centre carries more co-field whatever the target is, and only the comparison with
    this null separates the chemistry from the geometry.
    """
    pts = np.array([RING_POSITIONS[k] for k in POSITIONS])
    model.params = dict(model.params)
    model.params["lam"] = float(lam)

    def read():
        return {name: _statistics(dict(zip(POSITIONS, cofield_at(model, f, pts))))
                for f, name in enumerate(names)}

    model.fit(table, y, cv_folds=5, cv_seed=0, reuse_kernels=True)
    out = read()
    for f, name in enumerate(names):
        out[name].update(position_errors(model, f, pts, out[name], sigma))
    rng = np.random.RandomState(int(seed))
    null = {name: {k: [] for k in NULLED} for name in names}
    for _ in range(int(runs)):
        model.fit(table, y[rng.permutation(len(y))], cv_folds=5, cv_seed=0, reuse_kernels=True)
        got = read()
        for name in names:
            for k in NULLED:
                null[name][k].append(got[name][k])
    for name in names:
        for k in NULLED:
            a = np.asarray([x for x in null[name][k] if np.isfinite(x)], float)
            mean = float(a.mean()) if len(a) else float("nan")
            sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
            out[name][k + "_null"] = mean
            out[name][k + "_null_sd"] = sd
            out[name][k + "_z"] = (float((out[name][k] - mean) / sd)
                                   if sd and np.isfinite(sd) and sd > 0 else float("nan"))
    model.fit(table, y, cv_folds=5, cv_seed=0, reuse_kernels=True)      # leave it on the real y
    return out


def plot_cofield(model, names, path, fields=("electrostatic", "steric"), z=0.0):
    """The co-field in the ring plane, with the ring and the four positions drawn in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(-6.0, 3.6, 0.1)
    ys = np.arange(-4.2, 4.2, 0.1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, float(z))])
    show = [f for f in fields if f in names]
    fig, axes = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.0), squeeze=False)
    for j, fname in enumerate(show):
        f = list(names).index(fname)
        v = cofield_at(model, f, pts).reshape(X.shape)
        vmax = float(np.abs(v).max()) or 1.0
        ax = axes[0, j]
        im = ax.imshow(v, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="equal")
        _draw_ring(ax)
        ax.set_title(f"{model.fieldset.specs[f].label}, z = {z:g} A", fontsize=10)
        ax.set_xlabel("x / A", fontsize=8)
        ax.set_ylabel("y / A", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle("Co-field around an aromatic carbon (red: raises its shift per unit of field). "
                 "The measured carbon is at the origin.", fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_ring(ax):
    from cmfqsar.localalign import RING_RADIUS
    c = np.array([-RING_RADIUS, 0.0])
    ring = np.array([c + RING_RADIUS * np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])
                     for t in range(0, 360, 60)])
    for k in range(6):
        a, b = ring[k], ring[(k + 1) % 6]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.2)
    ax.scatter(ring[:, 0], ring[:, 1], s=20, color="black", zorder=3)
    ax.scatter([0.0], [0.0], s=55, facecolor="none", edgecolor="black", lw=1.6, zorder=4)
    for name, p in RING_POSITIONS.items():
        ax.annotate(name, (p[0], p[1]), ha="center", va="center", fontsize=8, color="dimgray",
                    fontweight="bold")
        if name in ("ortho", "meta"):
            ax.annotate(name, (p[0], -p[1]), ha="center", va="center", fontsize=8,
                        color="dimgray", fontweight="bold")


def run(rhos=RHOS, cofield_rho=COFIELD_RHO, folds=10, seed=0, max_molecules=0, results=RESULTS,
        scrambles=SCRAMBLES):
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} is missing; run examples/make_local_examples.py first")
    mols, names, targets = aromatic_samples(max_molecules=max_molecules)
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    _say(f"{len(targets)} aromatic ring carbons in {targets.n_measured_molecules} molecules")
    poses = well_posed(fields, targets, rho=cofield_rho, frames_of=ring_frames)
    _say(f"poses per sample: {dict(sorted(poses.items()))}")

    types = types_of(fields, min_atoms=10)
    ref = choose_lam(count_kernel(count_features(fields, targets, types)), targets, folds=folds,
                     seed=seed, name="atom-centred counts")
    ref.note = f"{len(types)} atom types in 6 shells"
    _say("  " + ref.summary())

    rows, kept = [], []
    for rho in rhos:
        t0 = time.time()
        envs = posed_environments(fields, targets, rho=rho, frames_of=ring_frames)
        model, table, K = overlap_kernel(envs, fieldset)
        aligned = choose_lam(K, targets, folds=folds, seed=seed, name=f"posed overlap, rho {rho:g}")
        aligned.note = f"{time.time() - t0:.0f} s"
        d_a = paired_interval(aligned, ref, targets, runs=2000, seed=seed)
        plain = [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
                 for r in range(len(targets))]
        inv = choose_lam(local_kernel(local_features(plain)), targets, folds=folds, seed=seed,
                         name=f"invariant local, rho {rho:g}")
        d_i = paired_interval(inv, ref, targets, runs=2000, seed=seed)
        d_ai = paired_interval(aligned, inv, targets, runs=2000, seed=seed)
        rows.append(dict(rho=float(rho),
                         aligned=dict(q2=aligned.q2, error=aligned.error, mae=aligned.mae,
                                      lam=aligned.lam, note=aligned.note,
                                      atoms=float(np.mean([e.n_atoms for e in envs]))),
                         invariant=dict(q2=inv.q2, error=inv.error, mae=inv.mae, lam=inv.lam),
                         against_counts=dict(aligned=d_a, invariant=d_i),
                         aligned_against_invariant=d_ai))
        _say("  " + aligned.summary() + f"; against the counts {d_a[0]:+.3f} "
                                        f"[{d_a[1]:+.3f}, {d_a[2]:+.3f}]")
        _say("  " + inv.summary() + f"; against the counts {d_i[0]:+.3f} "
                                    f"[{d_i[1]:+.3f}, {d_i[2]:+.3f}]")
        _say(f"    posed against invariant {d_ai[0]:+.3f} [{d_ai[1]:+.3f}, {d_ai[2]:+.3f}]")
        kept.append((rho, aligned, inv))

    # each kernel at its own best width, which is the comparison a reader asks for next: the
    # equal-width rows above hold the envelope fixed, and the two kernels do not want the same one
    best_a = max(kept, key=lambda k: k[1].q2)
    best_i = max(kept, key=lambda k: k[2].q2)
    best_pair = dict(aligned_rho=float(best_a[0]), invariant_rho=float(best_i[0]),
                     aligned_q2=best_a[1].q2, invariant_q2=best_i[2].q2,
                     difference=paired_interval(best_a[1], best_i[2], targets, runs=2000,
                                                seed=seed))
    _say(f"  best posed (rho {best_pair['aligned_rho']:g}, q2 {best_pair['aligned_q2']:+.3f}) "
         f"against best invariant (rho {best_pair['invariant_rho']:g}, q2 "
         f"{best_pair['invariant_q2']:+.3f}): {best_pair['difference'][0]:+.3f} "
         f"[{best_pair['difference'][1]:+.3f}, {best_pair['difference'][2]:+.3f}]")

    # the co-field, at the width that still reaches the ring
    _say(f"co-field at rho {cofield_rho:g}")
    envs = posed_environments(fields, targets, rho=cofield_rho, frames_of=ring_frames)
    model, table, K = overlap_kernel(envs, fieldset)
    best = choose_lam(K, targets, folds=folds, seed=seed, name="co-field model")
    names_f = fieldset.names
    co = ring_cofields(model, table, np.asarray(targets.y, float), best.lam, names_f,
                       sigma=best.error, runs=scrambles, seed=seed)
    for name in names_f:
        d = co[name]
        _say(f"  {name:14s} ipso {d['ipso']:+.4f} ortho {d['ortho']:+.4f} "
             f"meta {d['meta']:+.4f} para {d['para']:+.4f}")
        _say(f"    |para| - |meta| {d['para_minus_meta']:+.3f} +- {d['para_minus_meta_sd']:.3f} "
             f"(z {d['para_minus_meta_z']:+.1f}); conjugated - meta "
             f"{d['conjugated_minus_meta']:+.3f} +- {d['conjugated_minus_meta_sd']:.3f} "
             f"(z {d['conjugated_minus_meta_z']:+.1f})")
        _say(f"    donor r {d['r_methoxy (donor)']:+.3f} "
             f"(scramble z {d['r_methoxy (donor)_z']:+.1f}), "
             f"signs {d['signs_methoxy (donor)']}/4; acceptor r "
             f"{d['r_nitro (acceptor)']:+.3f}, signs {d['signs_nitro (acceptor)']}/4")
    os.makedirs(results, exist_ok=True)
    plot_cofield(model, names_f, os.path.join(results, "cofield.png"))

    reach = {k: float(np.exp(-cofield_rho * float(np.sum(np.asarray(v) ** 2))))
             for k, v in RING_POSITIONS.items()}
    out = dict(n_samples=len(targets), n_molecules=targets.n_measured_molecules, folds=folds,
               seed=seed, poses={str(k): v for k, v in sorted(poses.items())},
               cofield_rho=float(cofield_rho), cofield_lam=float(best.lam),
               cofield_q2=best.q2, cofield_error=best.error,
               envelope_at=reach, scrambles=int(scrambles),
               reference=dict(name=ref.name, q2=ref.q2, error=ref.error, mae=ref.mae,
                              lam=ref.lam, note=ref.note),
               rows=rows, best_pair=best_pair, cofield=co, scs=SCS,
               y=dict(low=float(np.min(targets.y)), high=float(np.max(targets.y)),
                      spread=float(np.std(targets.y))))
    with io.open(os.path.join(results, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def run_pka(rhos=RHOS, folds=10, seed=0, max_molecules=0, results=RESULTS):
    """The other site the plan names: the carboxyl carbon of a carboxylic acid.

    Plan item LP2 found no signal there -- on 912 acids the invariant local kernel was no better
    than an atom-centred count, and the difference field of the acid and its anion was far worse.
    The question this answers is whether that was the missing frame: a carboxyl carbon has two
    oxygens and one carbon of its own, which the atom types separate, so the pose is unique and
    needs no averaging at all.  Only the model is measured here, not a co-field: a co-field read
    off a model that does not predict would be an interpretation of noise.
    """
    path = os.path.join(ROOT, "examples", "local", "carboxylic_pka.sdf")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing; run examples/make_local_examples.py first")
    ds = load_dataset(path, atom_activity="pka")
    mols, names = list(ds.mols), list(ds.names)
    if max_molecules:
        mols, names = mols[:max_molecules], names[:max_molecules]
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    targets = AtomTargets.from_molecules(mols, "pka", names=names)
    _say(f"=== carboxylic acid pKa: {len(targets)} carboxyl carbons in "
         f"{targets.n_measured_molecules} molecules")
    poses = well_posed(fields, targets, rho=0.1)
    _say(f"poses per sample: {dict(sorted(poses.items()))}")

    types = types_of(fields, min_atoms=10)
    ref = choose_lam(count_kernel(count_features(fields, targets, types)), targets, folds=folds,
                     seed=seed, name="atom-centred counts")
    ref.note = f"{len(types)} atom types in 6 shells"
    _say("  " + ref.summary())
    rows = []
    for rho in rhos:
        envs = posed_environments(fields, targets, rho=rho)
        model, _table, K = overlap_kernel(envs, fieldset)
        aligned = choose_lam(K, targets, folds=folds, seed=seed,
                             name=f"posed overlap, rho {rho:g}")
        d_a = paired_interval(aligned, ref, targets, runs=2000, seed=seed)
        plain = [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
                 for r in range(len(targets))]
        inv = choose_lam(local_kernel(local_features(plain)), targets, folds=folds, seed=seed,
                         name=f"invariant local, rho {rho:g}")
        d_i = paired_interval(inv, ref, targets, runs=2000, seed=seed)
        d_ai = paired_interval(aligned, inv, targets, runs=2000, seed=seed)
        rows.append(dict(rho=float(rho),
                         aligned=dict(q2=aligned.q2, error=aligned.error, mae=aligned.mae,
                                      lam=aligned.lam,
                                      atoms=float(np.mean([e.n_atoms for e in envs])),
                                      poses=float(np.mean([e.n_conformers for e in envs]))),
                         invariant=dict(q2=inv.q2, error=inv.error, mae=inv.mae, lam=inv.lam),
                         against_counts=dict(aligned=d_a, invariant=d_i),
                         aligned_against_invariant=d_ai))
        _say("  " + aligned.summary() + f"; against the counts {d_a[0]:+.3f} "
                                        f"[{d_a[1]:+.3f}, {d_a[2]:+.3f}]")
        _say("  " + inv.summary() + f"; against the counts {d_i[0]:+.3f} "
                                    f"[{d_i[1]:+.3f}, {d_i[2]:+.3f}]")
        _say(f"    posed against invariant {d_ai[0]:+.3f} [{d_ai[1]:+.3f}, {d_ai[2]:+.3f}]")
    out = dict(n_samples=len(targets), n_molecules=targets.n_measured_molecules, folds=folds,
               seed=seed, poses={str(k): v for k, v in sorted(poses.items())},
               reference=dict(name=ref.name, q2=ref.q2, error=ref.error, mae=ref.mae,
                              lam=ref.lam, note=ref.note),
               rows=rows,
               y=dict(low=float(np.min(targets.y)), high=float(np.max(targets.y)),
                      spread=float(np.std(targets.y))))
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "pka.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def occupancy(fields, targets, rho, tol=0.75):
    """How many samples carry an atom of each type at each ring position.

    A co-field is read at a place; what it is worth there depends on how often the training set
    put anything there.  This counts it, by posing every sample and looking for an atom within
    `tol` of each of the four positions, on both sides of the x axis for the two that have a
    mirror image.
    """
    seen = {}
    for r in range(len(targets)):
        k, i = int(targets.molecule[r]), int(targets.atom[r])
        env = environment(fields[k], i, rho)
        frames = ring_frames(env)
        if not frames:
            continue
        c = np.asarray(env.coords, float) @ frames[0].T
        for name, p in RING_POSITIONS.items():
            for sign in (1.0, -1.0):
                q = np.array([p[0], sign * p[1], p[2]])
                d = np.linalg.norm(c - q, axis=1)
                j = int(np.argmin(d))
                if d[j] < tol:
                    key = (str(env.atom_types[j]), name)
                    seen[key] = seen.get(key, 0) + 1
                if abs(p[1]) < 1e-9:
                    break
    return seen


def indicator_effects(model, table, y, lam, fieldset, rho, sigma, seen):
    """What one atom of each indicator type at each ring position does to the predicted shift.

    The co-field says what the prediction gains per unit of field, which is not a number anyone
    can check.  `cofields.atom_effect` turns it into the effect of **one atom** placed there, in
    ppm, which is what a table of substituent-induced shifts holds.  The atom enters with the
    weight the envelope gives it at that distance -- an atom at the para position counts for
    exp(-rho d^2) of itself in the sample the model was fitted on -- so the number is what a real
    substituent at that position contributes, not what an undamped one would.
    """
    pts = np.array([RING_POSITIONS[k] for k in POSITIONS])
    damp = np.exp(-float(rho) * np.sum(pts ** 2, axis=1))
    model.params = dict(model.params)
    model.params["lam"] = float(lam)
    model.fit(table, y, cv_folds=5, cv_seed=0, reuse_kernels=True)
    factor = dual_covariance_factor(model)
    out = {}
    for f, name in enumerate(fieldset.names):
        if not name.startswith("cif:"):
            continue
        t = name[4:]
        row = {"type": t, "seen": {p: int(seen.get((t, p), 0)) for p in POSITIONS}}
        for k, p in enumerate(POSITIONS):
            v = float(atom_effect(model, f, pts[k:k + 1], weight=float(damp[k]))[0])
            cov = atom_effect_covariance(model, f, pts[k:k + 1], weight=float(damp[k]),
                                         sigma=sigma, factor=factor)
            row[p] = v
            row["sd_" + p] = float(np.sqrt(max(float(cov[0, 0]), 0.0)))
        # para against meta, with the covariance between the two places
        cov2 = atom_effect_covariance(model, f, pts[[2, 3]], weight=1.0, sigma=sigma,
                                      factor=factor)
        w = np.array([-float(np.sign(row["meta"]) or 1.0) * damp[2],
                      float(np.sign(row["para"]) or 1.0) * damp[3]])
        row["para_minus_meta"] = abs(row["para"]) - abs(row["meta"])
        row["para_minus_meta_sd"] = float(np.sqrt(max(float(w @ cov2 @ w), 0.0)))
        row["para_minus_meta_z"] = (row["para_minus_meta"] / row["para_minus_meta_sd"]
                                    if row["para_minus_meta_sd"] > 0 else float("nan"))
        if t in TYPE_SCS:
            want = np.array(TYPE_SCS[t][0], float)
            got = np.array([row[p] for p in POSITIONS], float)
            row["scs"] = list(want)
            row["scs_what"] = TYPE_SCS[t][1]
            row["signs"] = int(np.sum(np.sign(got) == np.sign(want)))
            row["r"] = (float(np.corrcoef(got, want)[0, 1])
                        if np.std(got) > 0 and np.std(want) > 0 else float("nan"))
            row["rms"] = float(np.sqrt(np.mean((got - want) ** 2)))
        out[t] = row
    return out


def run_indicator(rho=INDICATOR_RHO, folds=10, seed=0, max_molecules=0, results=RESULTS,
                  min_molecules=10):
    """The same co-field, read from the indicator fields instead of the five physico-chemical ones.

    The electrostatic field carries a partial charge, and a partial charge cannot tell a pi-donor
    from a pi-acceptor: methoxy and nitro both raise the carbon they sit on while their first
    atoms carry opposite charges, which is why no model linear in that field reproduced the ring
    picture.  A continuous indicator field carries the atom **type** instead, so "an ether oxygen
    at the ortho position" and "a carbonyl carbon at the ortho position" are different features
    and may have different co-fields.  That is how a table of substituent effects is written, and
    this asks whether the model writes the same table.
    """
    mols, names, targets = aromatic_samples(max_molecules=max_molecules)
    cif = cif_types_in(mols, "sybyl", min_molecules=min_molecules)
    fieldset = FieldSet.create(cif_types=cif)
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    _say(f"=== indicator fields: {len(cif)} types on {len(targets)} carbons in "
         f"{targets.n_measured_molecules} molecules, rho {rho:g}")
    t0 = time.time()
    envs = posed_environments(fields, targets, rho=rho, frames_of=ring_frames)
    model, table, K = overlap_kernel(envs, fieldset)
    _say(f"  kernel of {fieldset.n_fields} fields in {time.time() - t0:.0f} s")
    best = choose_lam(K, targets, folds=folds, seed=seed, name="posed overlap with indicators")
    _say("  " + best.summary())
    seen = occupancy(fields, targets, rho)
    eff = indicator_effects(model, table, np.asarray(targets.y, float), best.lam, fieldset, rho,
                            best.error, seen)
    for t, row in sorted(eff.items(), key=lambda kv: -sum(kv[1]["seen"].values())):
        if max(row["seen"].values()) < MIN_SEEN:
            continue
        _say(f"  {t:7s} " + "  ".join(f"{p} {row[p]:+6.1f} +- {row['sd_' + p]:4.1f} "
                                      f"(n {row['seen'][p]})" for p in POSITIONS))
        if "r" in row:
            _say(f"          measured {row['scs_what']}: "
                 + ", ".join(f"{v:+.1f}" for v in row["scs"])
                 + f"; r {row['r']:+.2f}, signs {row['signs']}/4, "
                   f"root mean square difference {row['rms']:.1f} ppm")
    out = dict(n_samples=len(targets), n_molecules=targets.n_measured_molecules, folds=folds,
               seed=seed, rho=float(rho), types=list(cif), n_fields=fieldset.n_fields,
               lam=float(best.lam), q2=best.q2, error=best.error, min_seen=MIN_SEEN,
               envelope_at={k: float(np.exp(-float(rho) * float(np.sum(np.asarray(v) ** 2))))
                            for k, v in RING_POSITIONS.items()},
               effects=eff)
    os.makedirs(results, exist_ok=True)
    draw = [n for n in ("cif:O.3", "cif:C.2") if n in fieldset.names]
    if draw:
        plot_cofield(model, fieldset.names, os.path.join(results, "indicator.png"), fields=draw)
    with io.open(os.path.join(results, "indicator.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def report(results=RESULTS):
    path = os.path.join(results, "summary.json")
    if not os.path.exists(path):
        raise SystemExit("nothing to report: run the benchmark first")
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    ref = d["reference"]
    out = ["# A co-field around one atom (plan item LP3)", "",
           "The sample is an aromatic carbon of a six-membered ring and its surroundings, posed "
           "by a rule: the carbon at the origin, the bond that leaves the ring on +x, the ring in "
           "the xy plane, both mirror images kept and averaged. In that frame the overlap kernel "
           "and the co-field of the program apply to an environment as they apply to a molecule. "
           f"{d['n_samples']} carbons in {d['n_molecules']} molecules, shifts from "
           f"{d['y']['low']:.1f} to {d['y']['high']:.1f} ppm (standard deviation "
           f"{d['y']['spread']:.1f}); {d['folds']} folds holding out whole molecules.", "",
           "## Does the pose cost anything?", "",
           "| descriptor | envelope width rho | q2 | error (ppm) | ridge | atoms per sample | "
           "difference in q2 against the counts |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
           f"| {ref['name']} ({ref['note']}) | -- | {ref['q2']:+.3f} | {ref['error']:.3f} | "
           f"{ref['lam']:g} | -- | -- |"]
    for r in d["rows"]:
        a, i = r["aligned"], r["invariant"]
        da, di = r["against_counts"]["aligned"], r["against_counts"]["invariant"]
        out.append(f"| posed overlap | {r['rho']:g} | {a['q2']:+.3f} | {a['error']:.3f} | "
                   f"{a['lam']:g} | {a['atoms']:.1f} | {da[0]:+.3f} [{da[1]:+.3f}, "
                   f"{da[2]:+.3f}] |")
        out.append(f"| invariant local | {r['rho']:g} | {i['q2']:+.3f} | {i['error']:.3f} | "
                   f"{i['lam']:g} | -- | {di[0]:+.3f} [{di[1]:+.3f}, {di[2]:+.3f}] |")
    out += ["", "The posed kernel against the invariant one, at the same envelope width:", ""]
    for r in d["rows"]:
        p = r["aligned_against_invariant"]
        out.append(f"* rho {r['rho']:g} (width {1.0 / float(r['rho']) ** 0.5:.1f} A): "
                   f"{p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}]")
    bp = d.get("best_pair")
    if bp:
        p = bp["difference"]
        out += ["", f"And each at its own best width -- posed at rho {bp['aligned_rho']:g} "
                    f"({bp['aligned_q2']:+.3f}) against invariant at rho "
                    f"{bp['invariant_rho']:g} ({bp['invariant_q2']:+.3f}): {p[0]:+.3f} "
                    f"[{p[1]:+.3f}, {p[2]:+.3f}]."]
    out += ["", "## Where the co-field sits", "",
            f"Fitted on every sample at rho {d['cofield_rho']:g} (q2 {d['cofield_q2']:+.3f}, "
            f"error {d['cofield_error']:.3f} ppm out of fold, ridge {d['cofield_lam']:g}), which "
            "is wide enough for the envelope to reach the ring: the factor exp(-rho d^2) is "
            + ", ".join(f"{k} {v:.2f}" for k, v in d["envelope_at"].items()) + ".", "",
            "The co-field is read at the four positions a substituent occupies on the ring. A "
            "positive value means that a positive field there raises the shift of the carbon at "
            "the origin.", "",
            "| field | ipso | ortho | meta | para |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for name, c in d["cofield"].items():
        out.append(f"| {name} | {c['ipso']:+.4f} | {c['ortho']:+.4f} | {c['meta']:+.4f} | "
                   f"{c['para']:+.4f} |")
    out += ["", "**Is the difference between the positions real?** The co-field is linear in "
                "the shifts, so the noise in them propagates to it in closed form, and the "
                "positions are correlated because they are built from the same training fields: "
                "the comparisons below are weighted combinations with the whole covariance taken "
                "into account, not two error bars added. The noise level is the out-of-fold error "
                f"of the model itself, {d['cofield_error']:.2f} ppm.", "",
            "* **|para| - |meta|** is the comparison the plan asks for. The para position is "
            "further from the measured carbon (4.19 A) than the meta one (3.70 A), so the "
            "envelope and the overlap both favour meta; a co-field larger at para says what the "
            "geometry does not.",
            "* **conjugated - meta** is (|ortho| + |para|) / 2 - |meta|, the same question with "
            "both conjugated positions.", "",
            "| field | |para| - |meta| | z | conjugated - meta | z |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for name, c in d["cofield"].items():
        out.append(f"| {name} | {c['para_minus_meta']:+.3f} +- {c['para_minus_meta_sd']:.3f} | "
                   f"{c['para_minus_meta_z']:+.1f} | {c['conjugated_minus_meta']:+.3f} +- "
                   f"{c['conjugated_minus_meta_sd']:.3f} | {c['conjugated_minus_meta_z']:+.1f} |")
    out += ["", "**Does it follow a measured substituent?** The correlation, over the four "
                "positions, of the co-field with the measured shifts a group causes: minus them "
                "for the donor, whose oxygen is negatively charged, plus them for the acceptor, "
                "whose nitrogen is positive. The null here is "
                f"{d['scrambles']} y-scrambles -- the same geometry, frame, kernel and ridge with "
                "the shifts shuffled between the carbons.", "",
            "| field | r (donor) | scramble z | signs | r (acceptor) | signs |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    don, acc = "methoxy (donor)", "nitro (acceptor)"
    for name, c in d["cofield"].items():
        out.append(f"| {name} | {c['r_' + don]:+.3f} | {c['r_' + don + '_z']:+.1f} | "
                   f"{c['signs_' + don]}/4 | {c['r_' + acc]:+.3f} | {c['signs_' + acc]}/4 |")
    rows = [", ".join(f"{k} {v:+.1f}" for k, v in row.items()) for row in d["scs"].values()]
    out += ["", "The measured shifts being compared with are " + rows[0] + " ppm for methoxy and "
                + rows[1] + " ppm for nitro, in that order of position. Both are large at ipso, "
                "ortho and para and almost nothing at meta, and they do **not** have opposite "
                "signs although the charges of their first atoms do: that part is resonance, "
                "which fragment charges cannot carry, and it is the same wall the Hammett "
                "co-fields of this repository met for sigma_p. `cofield.png` draws the co-field "
                "in the ring plane.", ""]
    strong = [n for n, c in d["cofield"].items() if c["para_minus_meta_z"] > 2]
    weak = [n for n, c in d["cofield"].items() if c["para_minus_meta_z"] < -2]
    follow = [n for n, c in d["cofield"].items()
              if c["signs_" + don] == 4 and c["r_" + don + "_z"] > 2]
    if strong:
        out.append("**The co-field puts more at the para position than at the meta one, against "
                   "what the distance would give and by more than the noise allows**, in "
                   + ", ".join(strong) + ".")
    else:
        out.append("**No field puts more of its co-field at the para position than at the meta "
                   "one** by more than the noise in the shifts allows.")
    if weak:
        out.append("The opposite is true, by more than the noise allows, in "
                   + ", ".join(weak) + ".")
    if follow:
        out.append("**It also follows the measured methoxy shifts in sign at all four positions, "
                   "by more than the scramble null allows**, in " + ", ".join(follow) + ".")
    out.append("")
    path = os.path.join(results, "pka.json")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as fh:
            p = json.load(fh)
        ref = p["reference"]
        out += ["## The other site: the carboxyl carbon of an acid", "",
                "Plan item LP2 found no signal on this set: on " + str(p["n_samples"]) + " acids "
                "the invariant local kernel was no better than an atom-centred count, and the "
                "difference field of the acid and its anion was far worse. A carboxyl carbon has "
                "two oxygens and one carbon of its own, which the atom types separate, so its "
                "pose is unique and nothing has to be averaged. The question here is whether the "
                "missing frame was the reason. Only the model is measured: a co-field read off a "
                "model that does not predict would be an interpretation of noise.", "",
                "| descriptor | envelope width rho | q2 | error (pKa units) | ridge | "
                "difference in q2 against the counts |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| {ref['name']} ({ref['note']}) | -- | {ref['q2']:+.3f} | {ref['error']:.3f} | "
                f"{ref['lam']:g} | -- |"]
        for r in p["rows"]:
            for key, label in (("aligned", "posed overlap"), ("invariant", "invariant local")):
                x, da = r[key], r["against_counts"][key]
                out.append(f"| {label} | {r['rho']:g} | {x['q2']:+.3f} | {x['error']:.3f} | "
                           f"{x['lam']:g} | {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}] |")
        best = max(p["rows"], key=lambda r: r["aligned"]["q2"])
        d = best["against_counts"]["aligned"]
        word = ("beats the count" if d[1] > 0 else "loses to the count" if d[2] < 0
                else "is no better than the count")
        out += ["", f"At its best width the posed kernel {word}: {d[0]:+.3f} [{d[1]:+.3f}, "
                    f"{d[2]:+.3f}]. Against the invariant kernel at the same width it is "
                    f"{best['aligned_against_invariant'][0]:+.3f} "
                    f"[{best['aligned_against_invariant'][1]:+.3f}, "
                    f"{best['aligned_against_invariant'][2]:+.3f}].", ""]
    path = os.path.join(results, "indicator.json")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as fh:
            q = json.load(fh)
        out += ["## The same co-field, read from the indicator fields", "",
                "A partial charge cannot tell a pi-donor from a pi-acceptor: methoxy and nitro "
                "both raise the carbon they sit on while their first atoms carry opposite "
                "charges. A continuous indicator field carries the atom **type** instead, so an "
                "ether oxygen at the ortho position and a carbonyl carbon at the ortho position "
                "are different features with their own co-fields. That is how a table of "
                "substituent effects is written, and this asks whether the model writes the same "
                f"table. {q['n_fields']} fields ({len(q['types'])} types and the five "
                f"physico-chemical ones), envelope width rho {q['rho']:g}, q2 {q['q2']:+.3f}, "
                f"error {q['error']:.2f} ppm out of fold.", "",
                "The numbers are **what one atom of that type at that position does to the "
                "predicted shift, in ppm** (`cofields.atom_effect`), with the weight the envelope "
                "gives an atom at that distance, so they are what a real substituent there "
                "contributes. `n` is how many of the carbons had an atom of that type at that "
                f"position; a row is shown when one position reaches {q['min_seen']}.", "",
                "| type | ipso | ortho | meta | para | measured |",
                "| --- | ---: | ---: | ---: | ---: | --- |"]
        rows = sorted(q["effects"].values(), key=lambda r: -sum(r["seen"].values()))
        for r in rows:
            if max(r["seen"].values()) < q["min_seen"]:
                continue
            cells = " | ".join(f"{r[p]:+.1f} +- {r['sd_' + p]:.1f} (n {r['seen'][p]})"
                               for p in POSITIONS)
            got = (", ".join(f"{v:+.1f}" for v in r["scs"]) + f" ({r['scs_what']})"
                   if "scs" in r else "--")
            out.append(f"| {r['type']} | {cells} | {got} |")
        with_scs = [r for r in rows if "r" in r and max(r["seen"].values()) >= q["min_seen"]]
        if with_scs:
            out += ["", "| type | r with the measured row | signs | root mean square difference "
                        "(ppm) |",
                    "| --- | ---: | ---: | ---: |"]
            for r in with_scs:
                out.append(f"| {r['type']} | {r['r']:+.2f} | {r['signs']}/4 | {r['rms']:.1f} |")
            got = np.concatenate([[r[p] for p in POSITIONS] for r in with_scs])
            want = np.concatenate([r["scs"] for r in with_scs])
            overall = float(np.corrcoef(got, want)[0, 1])
            out += ["", f"Over all {len(got)} (type, position) pairs with a measured row the "
                        f"correlation is {overall:+.2f}.", ""]
            donor = next((r for r in with_scs if r["type"] == "O.3"), None)
            acceptor = next((r for r in with_scs if r["type"] == "C.2"), None)
            if donor and acceptor:
                out.append(
                    "The sharpest question these fields can answer is the one the electrostatic "
                    "field could not: a donor lowers the para carbon and an acceptor raises it. "
                    f"Here the ether oxygen gives {donor['para']:+.1f} +- {donor['sd_para']:.1f} "
                    f"at para and the sp2 carbon {acceptor['para']:+.1f} +- "
                    f"{acceptor['sd_para']:.1f}, against {donor['scs'][3]:+.1f} and "
                    f"{acceptor['scs'][3]:+.1f} measured.")
            out.append("`indicator.png` draws the two co-fields in the ring plane side by "
                       "side. Both are positive where a substituent sits on the measured "
                       "carbon and negative at the ortho positions; what separates the "
                       "donor from the acceptor is the weaker pattern further out, which "
                       "the table above reads off at the points rather than by eye.")
            out.append("")
    return "\n".join(out)


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", nargs="*", type=float, default=list(RHOS))
    ap.add_argument("--cofield-rho", type=float, default=COFIELD_RHO)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scrambles", type=int, default=SCRAMBLES)
    ap.add_argument("--max-molecules", type=int, default=0)
    ap.add_argument("--only", default="all", choices=["all", "ring", "pka", "indicator"],
                    help="which part to compute: the aromatic ring site, the carboxyl site, the "
                         "indicator co-fields, or all of them (default)")
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
    if a.only in ("all", "ring"):
        run(rhos=a.rho, cofield_rho=a.cofield_rho, folds=a.folds, seed=a.seed,
            max_molecules=a.max_molecules, scrambles=a.scrambles)
    if a.only in ("all", "indicator"):
        run_indicator(folds=a.folds, seed=a.seed, max_molecules=a.max_molecules)
    if a.only in ("all", "pka"):
        run_pka(rhos=a.rho, folds=a.folds, seed=a.seed, max_molecules=a.max_molecules)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
