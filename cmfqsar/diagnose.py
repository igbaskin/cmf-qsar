"""
What to ask of a data set before modelling it, and which route it should take.

Every expensive mistake this program has recorded was visible before the model was built, in a
number that took seconds to compute.  The corrosion benzenes were modelled with five fields and an
automatic frame, and atom-type counts -- no geometry at all -- predicted better than them; the
ACE set of the 2013 paper is predicted better by four two-dimensional descriptors than by the
whole 3D model; a
thrombin model in the exposure study scored +0.059 with alpha stuck at the top of its grid, which
is what a mis-specified model looks like from outside.  None of these is subtle.  They are simply
never asked, because asking means running three or four baselines by hand.

This module asks them.  It is not a model and it does not improve one: it is the half hour of
scepticism that should precede the modelling, packaged as :func:`diagnose` and ``cmf diagnose``.

Five questions, and the numbers that answer each:

1. **Is there anything to model?**  The size of the set and the spread of the activity.  Twenty
   molecules with an activity range of half a log unit cannot support a conclusion, whatever q2
   comes out.
2. **Can a frame exist at all?**  The largest Bemis-Murcko framework and how much of the set and
   of each molecule it covers, the number of chemotypes, and whether the molecules are already
   superimposed.  A series of several scaffolds has no common frame to be aligned into, and the
   program's own alignment will invent one.
3. **Does the frame carry signal?**  The aligned model against the two baselines that need no
   frame: the atom-type count kernel, which has thrown the geometry away entirely, and the
   rotation-invariant field autocorrelation, which keeps it without a pose.  If neither is
   outperformed, the alignment adds nothing and the cheaper route is also the more reliable one.
4. **Does the geometry add anything?**  The same comparison against descriptors alone -- what the
   file carries plus the few the program computes.  This is the question the ACE result forces.
5. **Was the model specified sensibly?**  Whether the chosen alpha sits on the edge of its grid,
   which means the search needed a value it was not offered.

Every comparison is paired over the molecules and reported with a bootstrap interval, because on
twenty to thirty molecules a difference below about 0.1 in q2 is not a difference.  The
recommendation at the end is a reading of those intervals and nothing more: it says which route to
take and which question the set has not answered, never that a model is good.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

from .fields import AtomTable, FieldSet, build_fields
from .model import CMFModel, cv_predict, r2_score, rmse

#: the attenuations and ridges the diagnosis searches, kept small because it is a diagnosis
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
#: a difference smaller than this, or than the bootstrap interval, is not a difference
MEANINGFUL = 0.05


@dataclass
class Route:
    """One way of comparing the molecules, and how well it predicted."""
    name: str
    q2: float = float("nan")
    rmse_cv: float = float("nan")
    alpha: float = float("nan")
    lam: float = float("nan")
    predicted: np.ndarray = field(default_factory=lambda: np.zeros(0))
    note: str = ""
    #: columns removed because they reproduce the activity rather than describe the molecule
    dropped: List[str] = field(default_factory=list)
    #: the descriptors that correlate most strongly with the activity, for the leak warning
    correlations: List[Tuple[str, float]] = field(default_factory=list)
    #: set when the baseline turned out to be reading the activity rather than describing it
    leaked: bool = False
    #: the molecules this route could cover, when it cannot cover them all
    subset: Optional[List[int]] = None

    def row(self) -> dict:
        return {"route": self.name, "q2": self.q2, "rmse_cv": self.rmse_cv,
                "alpha": self.alpha, "lam": self.lam, "note": self.note}


@dataclass
class Finding:
    """One answer, with the numbers behind it."""
    code: str
    level: str          # "ok" | "note" | "warn"
    text: str
    numbers: dict = field(default_factory=dict)

    def __str__(self) -> str:
        mark = {"ok": "  ", "note": "- ", "warn": "! "}.get(self.level, "  ")
        return f"{mark}{self.text}"


@dataclass
class Diagnosis:
    n: int = 0
    activity_sd: float = float("nan")
    activity_range: Tuple[float, float] = (float("nan"), float("nan"))
    routes: Dict[str, Route] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    recommended: str = ""
    why: str = ""
    nested: bool = True

    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    def report(self) -> str:
        lines = [f"{self.n} molecules, activity from {self.activity_range[0]:.2f} to "
                 f"{self.activity_range[1]:.2f} (sd {self.activity_sd:.2f})", ""]
        if self.nested:
            lines.append("Hyper-parameters chosen inside each fold and scored outside it.")
            lines.append("")
        if self.routes:
            lines.append("How well each route predicts, 10-fold CV:")
            width = max(len(r) for r in self.routes)
            for r in self.routes.values():
                extra = ""
                if np.isfinite(r.alpha):
                    extra = f"  (alpha {r.alpha:g}, lambda {r.lam:g})"
                elif np.isfinite(r.lam):
                    extra = f"  (lambda {r.lam:g})"
                lines.append(f"  {r.name:{width}s}  q2 {r.q2:+.3f}  RMSEcv {r.rmse_cv:.3f}{extra}"
                             + (f"   {r.note}" if r.note else ""))
            lines.append("")
        lines.append("What the set says:")
        lines += [str(f) for f in self.findings]
        lines += ["", f"Recommended route: {self.recommended}", f"  {self.why}"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"n": self.n, "activity_sd": self.activity_sd,
                "activity_range": list(self.activity_range),
                "routes": {k: v.row() for k, v in self.routes.items()},
                "findings": [{"code": f.code, "level": f.level, "text": f.text,
                              "numbers": f.numbers} for f in self.findings],
                "recommended": self.recommended, "why": self.why, "nested": self.nested}


# ----------------------------------------------------------------------------------------------
# the pieces
# ----------------------------------------------------------------------------------------------

def _flat_best(kernels, y, lams=LAMS, folds=10, seed=0):
    """(q2, rmse, alpha, lam, predictions) of the best point of a grid, scored where it was chosen.

    Optimistic by construction, and kept only for comparison: the routes differ in how many
    hyper-parameters they have, so the bias is not the same for each and the comparison between
    them is biased too.
    """
    best = None
    for alpha, K in kernels:
        for lam in lams:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=folds, seed=seed)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (q2, rmse(y, pred), float(alpha), float(lam), pred)
    return best


def _nested_best(kernels, y, lams=LAMS, folds=10, inner: int = 5, seed=0):
    """The same, with (alpha, lambda) chosen inside every training part and scored outside it.

    This is the difference between "how well does the best of thirty settings fit this
    cross-validation" and "how well does the procedure predict a molecule it has not seen".  The
    diagnosis compares five routes carrying different numbers of hyper-parameters, and without
    nesting the one with the largest grid gains a little of its lead merely by searching harder.
    """
    from .model import krr_fit

    y = np.asarray(y, float)
    n = len(y)
    if n < 8 or not kernels:
        return _flat_best(kernels, y, lams, folds, seed)
    order = np.random.RandomState(seed).permutation(n)
    parts = np.array_split(order, max(2, min(int(folds), n)))
    pred = np.full(n, np.nan)
    picks = []
    for part in parts:
        te = np.asarray(part, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[te] = False
        tr = np.where(mask)[0]
        if len(tr) < 4 or not len(te):
            continue
        best = None
        for alpha, K in kernels:
            Ktr = K[np.ix_(tr, tr)]
            for lam in lams:
                p = cv_predict(Ktr, y[tr], "krr", {"lam": float(lam)},
                               folds=max(2, min(int(inner), len(tr))), seed=seed)
                q = r2_score(y[tr], p)
                if best is None or q > best[0]:
                    best = (q, float(alpha), float(lam), K)
        if best is None:
            continue
        _q, alpha, lam, K = best
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], float(lam), True)
        pred[te] = K[np.ix_(te, tr)] @ a + b
        picks.append((alpha, lam))
    ok = np.isfinite(pred)
    if not ok.any() or not picks:
        return _flat_best(kernels, y, lams, folds, seed)
    alpha = float(np.median([p[0] for p in picks]))
    lam = float(np.median([p[1] for p in picks]))
    return (float(r2_score(y[ok], pred[ok])), float(rmse(y[ok], pred[ok])), alpha, lam, pred)


#: whether the routes nest their hyper-parameter search; False reproduces the optimistic numbers
NESTED = True


def _best(kernels, y, lams=LAMS, folds=10, seed=0, nested: Optional[bool] = None):
    """The route score: nested by default, flat when asked for the comparison."""
    use = NESTED if nested is None else bool(nested)
    return (_nested_best(kernels, y, lams, folds, seed=seed) if use
            else _flat_best(kernels, y, lams, folds, seed))


def aligned_route(mols, names, y, fieldset=None, alphas=ALPHAS, folds=10, seed=0,
                  charge_model="gasteiger", param_scheme="rdkit") -> Route:
    """The ordinary model: the five fields in the frame the molecules arrive in."""
    kernels = []
    for alpha in alphas:
        fs = (fieldset.copy() if fieldset is not None
              else FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model=charge_model,
                                   param_scheme=param_scheme))
        if fieldset is not None:
            for s in fs.specs:
                s.alpha = alpha
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        kernels.append((alpha, model.joint_kernel(
            model.prepare_training_kernels(AtomTable.from_molecules(mf)))))
    q2, err, alpha, lam, pred = _best(kernels, y, folds=folds, seed=seed)
    return Route("aligned", q2, err, alpha, lam, pred)


def counts_route(mols, names, y, folds=10, seed=0) -> Route:
    """Atom-type counts: no geometry at all, and no frame to get wrong."""
    from .chemotypes import count_kernel

    K = count_kernel(list(mols), list(names))
    q2, err, _a, lam, pred = _best([(float("nan"), K)], y, folds=folds, seed=seed)
    return Route("counts", q2, err, float("nan"), lam, pred)


def invariant_route(mols, names, y, alphas=ALPHAS, folds=10, seed=0,
                    charge_model="gasteiger", param_scheme="rdkit") -> Route:
    """The field autocorrelation: the geometry kept, the frame not needed."""
    from .invariant import AutocorrelationSpec, autocorrelation_kernel, feature_matrix

    acs = AutocorrelationSpec()
    kernels = []
    for alpha in alphas:
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model=charge_model,
                             param_scheme=param_scheme)
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        kernels.append((alpha, autocorrelation_kernel(feature_matrix(mf, acs),
                                                      n_bins=acs.n_bins)))
    q2, err, alpha, lam, pred = _best(kernels, y, folds=folds, seed=seed)
    return Route("invariant", q2, err, alpha, lam, pred)


def topomer_route(mols, names, y, alphas=ALPHAS, folds=10, seed=0,
                  charge_model="gasteiger", param_scheme="rdkit") -> Route:
    """The substituents cut at the framework and posed by rule: geometry without a frame.

    Only the molecules that can be cut at the common framework take part, so the route carries
    the indices it covers and the diagnosis compares it with an aligned model fitted on those
    same molecules -- comparing a subset against the whole set is the mistake this repository
    already made once, and it overstates the quality of whichever route happens to drop the hard
    compounds.
    """
    from .topomer import TopomerSpec, choose_core, cuttable, series_labels, topomer_fields

    core, _how = choose_core(list(mols), mode="framework")
    if core is None:
        return Route("topomer", float("nan"), float("nan"), note="no common framework to cut at")
    sel = [i for i, m in enumerate(mols) if cuttable(m, core)]
    if len(sel) < 10:
        return Route("topomer", float("nan"), float("nan"), subset=sel,
                     note=f"only {len(sel)} of {len(mols)} molecules can be cut at the framework")
    labels = series_labels([mols[i] for i in sel], core)
    spec = TopomerSpec(torsion=180.0)
    kernels, keep = [], None
    for alpha in alphas:
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model=charge_model,
                             param_scheme=param_scheme)
        tf, _log = topomer_fields([mols[i] for i in sel], [names[i] for i in sel], fs,
                                  core=core, labels=labels, spec=spec)
        k = [sel[j] for j, f in enumerate(tf) if f is not None]
        if keep is None:
            keep = k
        mf = [f for f in tf if f is not None]
        if len(mf) < 10 or len(k) != len(keep):
            return Route("topomer", float("nan"), float("nan"), subset=keep or sel,
                         note="the substituents could not be built consistently")
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        kernels.append((alpha, model.joint_kernel(
            model.prepare_training_kernels(AtomTable.from_molecules(mf)))))
    yk = np.asarray(y, float)[keep]
    q2, err, alpha, lam, pred = _best(kernels, yk, folds=folds, seed=seed)
    return Route("topomer", q2, err, alpha, lam, pred, subset=list(keep),
                 note=f"{len(keep)} of {len(mols)} molecules can be cut at the framework")


def descriptor_route(mols, y, names_d: Optional[Sequence[str]] = None, folds=10, seed=0,
                     exclude: Sequence[str] = (), leak: float = 0.95) -> Route:
    """Descriptors alone: the question the ACE result forces on every set.

    Most files store the activity as a molecule property, so the available descriptors usually
    *include the quantity being predicted*; given it, this baseline returns q2 = 1.000 and the
    diagnosis recommends descriptors for every set in the world.  Anything in `exclude`, and
    anything correlating with the activity above `leak`, is therefore dropped and named.

    `leak` is 0.95 and not something nearer 1 because the leak is rarely a copy.  The corrosion
    files carry the potency as ``activity`` and again as ``mwila_pooled_raw``, two reductions of
    the same measurements, which correlate at 0.989: excluding the activity by name left the
    other, and the descriptor route came back at q2 = 0.999 on four sets.  A genuine descriptor
    can of course reach 0.95 on a congeneric series, so the dropped columns are named in the
    note and raised as a warning rather than quietly discarded -- if one of them really is a
    descriptor, name it with `names_d` and run again.
    """
    from .descriptors import DescriptorScaler, available, computed_names, descriptor_kernel, \
        read_values

    want = list(names_d) if names_d else (
        list(computed_names()) + list(available(list(mols))))   # already carry the calc: prefix
    drop = {str(e) for e in exclude}
    want = [w for w in dict.fromkeys(want) if w not in drop]
    if not want:
        return Route("descriptors", float("nan"), float("nan"), note="no descriptors available")
    Z = np.vstack([read_values(m, want) for m in mols])
    keep = [j for j in range(Z.shape[1]) if np.all(np.isfinite(Z[:, j])) and np.std(Z[:, j]) > 0]
    leaked = []
    for j in list(keep):
        r = abs(float(np.corrcoef(Z[:, j], y)[0, 1])) if np.std(Z[:, j]) > 0 else 0.0
        if np.isfinite(r) and r >= float(leak):
            keep.remove(j)
            leaked.append(want[j])
    if not keep:
        return Route("descriptors", float("nan"), float("nan"),
                     note=("every descriptor was the activity itself (" + ", ".join(leaked) + ")"
                           if leaked else "no usable descriptors"))
    Z0 = Z
    Z = Z[:, keep]
    Zs = DescriptorScaler.fit(Z, tuple(want[j] for j in keep)).transform(Z)
    K = descriptor_kernel(Zs, Zs)
    K = K / max(float(np.mean(np.diag(K))), 1e-12)
    q2, err, _a, lam, pred = _best([(float("nan"), K)], y, folds=folds, seed=seed)
    note = f"{len(keep)} descriptors"
    if leaked:
        note += "; dropped as the activity under another name: " + ", ".join(leaked)
    r = Route("descriptors", q2, err, float("nan"), lam, pred, note=note)
    r.dropped = list(leaked)
    r.correlations = sorted(
        ((want[j], abs(float(np.corrcoef(Z0[:, j], y)[0, 1]))) for j in keep),
        key=lambda kv: -kv[1])[:5]
    return r


def paired_interval(a: Route, b: Route, y, runs: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """(difference in q2, low, high) for route `a` against route `b`, bootstrapped over molecules.

    Resampling the molecules rather than comparing two point estimates is what keeps a difference
    of 0.04 on twenty-five molecules from being read as a result.
    """
    y = np.asarray(y, float)
    if not (len(a.predicted) == len(b.predicted) == len(y)) or len(y) < 4:
        return (a.q2 - b.q2, float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(int(runs)):
        idx = rng.randint(0, len(y), len(y))
        yy = y[idx]
        if np.std(yy) <= 0:
            continue
        out.append(r2_score(yy, a.predicted[idx]) - r2_score(yy, b.predicted[idx]))
    if not out:
        return (a.q2 - b.q2, float("nan"), float("nan"))
    return (a.q2 - b.q2, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def frame_findings(mols: Sequence[Chem.Mol], names: Sequence[str]) -> List[Finding]:
    """Whether a common frame can exist, and whether the molecules are already in one."""
    from .chemotypes import chemotype_labels
    from .topomer import choose_core, cuttable_fraction

    out: List[Finding] = []
    labels, _scaffolds = chemotype_labels(list(mols))
    groups = len(set(labels.tolist()))
    biggest = max(int(np.sum(labels == g)) for g in set(labels.tolist()))
    out.append(Finding(
        "chemotypes", "ok" if groups <= 2 else "warn",
        f"{groups} chemotype(s); the largest holds {biggest} of {len(mols)} molecules"
        + ("" if groups <= 2 else " -- a series of several scaffolds has no common frame, and an "
                                 "alignment will invent one"),
        {"chemotypes": groups, "largest": biggest}))
    core, how = choose_core(list(mols), mode="framework")
    if core is None:
        out.append(Finding("core", "warn", "no common framework was found at all", {}))
    else:
        frac = cuttable_fraction(list(mols), core)
        heavy = core.GetNumHeavyAtoms()
        median_size = float(np.median([m.GetNumHeavyAtoms() for m in mols]))
        out.append(Finding(
            "core", "ok" if frac > 0.6 else "note",
            f"the common framework ({how}) has {heavy} heavy atoms, about "
            f"{heavy / max(median_size, 1):.0%} of a typical molecule, and "
            f"{frac:.0%} of the set can be cut at it",
            {"core_atoms": heavy, "cuttable_fraction": frac}))
    out.append(_superimposed_finding(mols, core))
    return out


def _superimposed_finding(mols: Sequence[Chem.Mol], core) -> Finding:
    """Are the molecules in one frame?  Asked of the shared atoms, not of the centroids.

    A centroid cannot answer it: ``EmbedMolecule`` leaves every molecule centred on the origin,
    so a set of independently generated conformations has a centroid spread of 0.02 A and looks
    perfectly superimposed while every molecule points a different way.  The question is whether
    the atoms the molecules *share* lie in the same place, so that is what is measured -- the
    framework atoms against the first molecule that carries them, with no superposition.
    """
    coords = [np.asarray(m.GetConformer().GetPositions(), float) for m in mols]
    size = float(np.median([np.ptp(c, axis=0).max() for c in coords]))
    all_matches = []
    if core is not None:
        for m, c in zip(mols, coords):
            try:
                hits = m.GetSubstructMatches(core, uniquify=False, maxMatches=64)
            except Exception:                   # noqa: BLE001
                hits = ()
            keep = [c[list(h)] for h in hits if len(h) >= 3]
            all_matches.append(keep or None)
    have = [x for x in all_matches if x]
    if len(have) >= 3 and all(x[0].shape == have[0][0].shape for x in have):
        ref = have[0][0]
        # a small framework -- a benzene ring, say -- matches in several places and in several
        # orientations, and the first match RDKit returns is an arbitrary one.  Asking whether
        # the molecules share a frame means asking whether *some* assignment puts the shared
        # atoms together, so every match is tried and the best one counts; with the first match
        # alone an author-aligned Sutherland set read as 11.6 A apart.
        devs = [min(float(np.sqrt(np.mean(np.sum((x - ref) ** 2, axis=1)))) for x in xs)
                for xs in have[1:]]
        med = float(np.median(devs))
        n_core = ref.shape[0]
        coverage = n_core / max(float(np.median([m.GetNumHeavyAtoms() for m in mols])), 1.0)
        ambiguous = float(np.median([len(xs) for xs in have]))
        # The test can be misled both ways, so it may give no answer.  A framework that is a
        # small symmetric ring matches in a dozen orientations, and two rings near the origin
        # will coincide to under an angstrom by chance -- a set of independently embedded
        # molecules passed that way.  A framework that is a small part of the molecule may also
        # match a *different* copy in each molecule, which made an author-aligned Sutherland set
        # read as 9 A apart.  So a conclusion is drawn only when the framework is a large part of
        # the molecule and matches it in few ways.
        # positive and negative evidence are not symmetric.  Six atoms lying on six atoms to
        # within half an angstrom is not luck, however many ways the framework matches, so a
        # small deviation is a conclusion on its own.  A large one is only a conclusion when the
        # framework is a large part of the molecule, since otherwise it may be matching a
        # different copy of itself in each molecule.
        conclusive = coverage >= 0.4
        where = (f"the {n_core} shared framework atoms sit {med:.1f} A apart between molecules, "
                 f"over the best of their matches ({len(have)} of {len(mols)} carry them)")
        if med < 0.5:
            return Finding("superimposed", "ok", where, {"core_rmsd": med, "coverage": coverage,
                                                         "matches": ambiguous, "extent": size})
        if conclusive and med >= 1.0:
            return Finding("superimposed", "warn", where + " -- the molecules are not in a "
                           "common frame", {"core_rmsd": med, "coverage": coverage,
                                            "matches": ambiguous, "extent": size})
        return Finding(
            "superimposed", "note",
            where + f"; but that framework is only {coverage:.0%} of a typical molecule and "
            f"matches it {ambiguous:.0f} ways on average, so this cannot say whether the set "
            f"shares a frame -- look at it",
            {"core_rmsd": med, "coverage": coverage, "matches": ambiguous, "extent": size})
    centres = np.array([c.mean(axis=0) for c in coords])
    spread = float(np.median(np.linalg.norm(centres - centres.mean(axis=0), axis=1)))
    return Finding(
        "superimposed", "note",
        f"no shared framework to check the frame with; the centroids lie {spread:.1f} A apart "
        f"against a molecular extent of {size:.1f} A, which cannot distinguish a frame from a "
        f"heap of molecules at the origin",
        {"centroid_spread": spread, "extent": size})


# ----------------------------------------------------------------------------------------------
# the diagnosis
# ----------------------------------------------------------------------------------------------

def route_kernels(route, mols, names, alphas=ALPHAS, charge_model="gasteiger",
                  param_scheme="rdkit", fieldset=None, y=None, activity_prop="",
                  leak=0.95):
    """[(alpha, K)] over all the molecules for one route, and the molecules it covers.

    The routes are scored inside :func:`diagnose` on their own cross-validation; to *follow* the
    advice one needs the kernels themselves, so that a model can be fitted on part of a set and
    asked about the rest.
    """
    from .chemotypes import count_kernel
    from .descriptors import (DescriptorScaler, available, computed_names, descriptor_kernel,
                              read_values)
    from .invariant import AutocorrelationSpec, autocorrelation_kernel, feature_matrix

    mols, names = list(mols), list(names)
    if route == "counts":
        return [(float("nan"), count_kernel(mols, names))], None
    if route == "descriptors":
        want = [w for w in (list(computed_names()) + list(available(mols)))
                if w != activity_prop]
        Z = np.vstack([read_values(m, want) for m in mols])
        keep = [j for j in range(Z.shape[1])
                if np.all(np.isfinite(Z[:, j])) and np.std(Z[:, j]) > 0]
        if y is not None:
            # the same guard the descriptor baseline applies: a column that reproduces the
            # activity is the activity, and without this the held-out score is 1.000
            yy = np.asarray(y, float)
            keep = [j for j in keep
                    if abs(float(np.corrcoef(Z[:, j], yy)[0, 1])) < float(leak)]
        if not keep:
            return [], None
        Zs = DescriptorScaler.fit(Z[:, keep], tuple(want[j] for j in keep)).transform(Z[:, keep])
        K = descriptor_kernel(Zs, Zs)
        return [(float("nan"), K / max(float(np.mean(np.diag(K))), 1e-12))], None
    if route == "topomer":
        from .topomer import TopomerSpec, choose_core, cuttable, series_labels, topomer_fields
        core, _how = choose_core(mols, mode="framework")
        if core is None:
            return [], None
        sel = [i for i, m in enumerate(mols) if cuttable(m, core)]
        if len(sel) < 10:
            return [], sel
        labels = series_labels([mols[i] for i in sel], core)
        out, keep = [], None
        for alpha in alphas:
            fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model=charge_model,
                                 param_scheme=param_scheme)
            tf, _log = topomer_fields([mols[i] for i in sel], [names[i] for i in sel], fs,
                                      core=core, labels=labels, spec=TopomerSpec(torsion=180.0))
            k = [sel[j] for j, f in enumerate(tf) if f is not None]
            if keep is None:
                keep = k
            if k != keep:
                return [], keep
            model = CMFModel(fieldset=fs, method="krr", normalize=True)
            out.append((alpha, model.joint_kernel(model.prepare_training_kernels(
                AtomTable.from_molecules([f for f in tf if f is not None])))))
        return out, keep
    acs = AutocorrelationSpec() if route == "invariant" else None
    out = []
    for alpha in alphas:
        fs = (fieldset.copy() if fieldset is not None
              else FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model=charge_model,
                                   param_scheme=param_scheme))
        if fieldset is not None:
            for sp in fs.specs:
                sp.alpha = alpha
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        if route == "invariant":
            out.append((alpha, autocorrelation_kernel(feature_matrix(mf, acs),
                                                      n_bins=acs.n_bins)))
        else:
            model = CMFModel(fieldset=fs, method="krr", normalize=True)
            out.append((alpha, model.joint_kernel(
                model.prepare_training_kernels(AtomTable.from_molecules(mf)))))
    return out, None


def _fit_predict(kernels, y, train, test, lams=LAMS, inner=5, seed=0):
    """Choose (alpha, lambda) inside `train` and predict `test`."""
    from .model import krr_fit

    y = np.asarray(y, float)
    train = np.asarray(train, dtype=int)
    test = np.asarray(test, dtype=int)
    best = None
    for alpha, K in kernels:
        Ktr = K[np.ix_(train, train)]
        for lam in lams:
            p = cv_predict(Ktr, y[train], "krr", {"lam": float(lam)},
                           folds=max(2, min(int(inner), len(train))), seed=seed)
            q = r2_score(y[train], p)
            if best is None or q > best[0]:
                best = (q, alpha, float(lam), K)
    if best is None:
        return None
    _q, _alpha, lam, K = best
    a, b = krr_fit(K[np.ix_(train, train)], y[train], lam, True)
    return K[np.ix_(test, train)] @ a + b


def validate_advice(mols, names, y, test_fraction=0.3, seeds=(0, 1, 2), alphas=ALPHAS,
                    charge_model="gasteiger", param_scheme="rdkit", activity_prop="",
                    folds=5, progress=None):
    """Take the advice, build that model, and see whether it did better than the aligned one.

    The diagnosis is run on the training part alone, so the advice never sees the molecules it
    will be judged on; every route is then fitted on that part and asked about the rest.  This
    is the measurement that decides whether the recommendation is worth printing: a tool that
    reports four baselines is useful whatever its advice, but advice that has never been
    followed is only a rule its author liked.
    """
    mols, names = list(mols), list(names)
    y = np.asarray(y, float)
    n = len(mols)
    routes = ("aligned", "counts", "invariant", "topomer", "descriptors")
    kernels = {}
    for r in routes:
        try:
            kernels[r] = route_kernels(r, mols, names, alphas, charge_model, param_scheme,
                                       y=y, activity_prop=activity_prop)
        except Exception as exc:                 # noqa: BLE001
            kernels[r] = ([], None)
            if progress:
                progress(f"{r}: {exc}")
    rows = []
    for seed in seeds:
        order = np.random.RandomState(seed).permutation(n)
        k = max(3, int(round(test_fraction * n)))
        test, train = sorted(order[:k].tolist()), sorted(order[k:].tolist())
        d = diagnose([mols[i] for i in train], [names[i] for i in train], y[train],
                     alphas=alphas, folds=folds, seed=seed, charge_model=charge_model,
                     param_scheme=param_scheme, activity_prop=activity_prop)
        advice = d.recommended.split(",")[0].split(" or ")[0].strip()
        # the diagnosis decides, on the training part, whether the descriptor baseline is
        # reading the activity; if it is, that route is not a route the advice could have taken
        # and must not count as the one it should have chosen
        dr = d.routes.get("descriptors")
        unusable = {"descriptors"} if (dr is not None and getattr(dr, "leaked", False)) else set()
        scores = {}
        for r in routes:
            ks, subset = kernels[r]
            if not ks:
                continue
            if subset is None:
                tr, te, yy = np.asarray(train, int), np.asarray(test, int), y
            else:
                # the topomer route covers only what can be cut, so the split is remapped into
                # that subset and the route is scored on the molecules it actually holds
                pos = {int(i): j for j, i in enumerate(subset)}
                tr = np.array([pos[i] for i in train if i in pos], dtype=int)
                te = np.array([pos[i] for i in test if i in pos], dtype=int)
                if len(tr) < 8 or len(te) < 3:
                    continue
                yy = y[np.asarray(subset, dtype=int)]
            pred = _fit_predict(ks, yy, tr, te, seed=seed)
            if pred is None:
                continue
            scores[r] = float(r2_score(yy[te], pred, float(np.mean(yy[tr]))))
        rows.append({"seed": seed, "advice": advice, "n_train": len(train),
                     "n_test": len(test), "held_out": scores,
                     "unusable": sorted(unusable)})
        if progress:
            progress(f"seed {seed}: advised {advice}; held out "
                     + ", ".join(f"{a} {b:+.3f}" for a, b in scores.items()))
    return {"splits": rows, "test_fraction": test_fraction}


def diagnose(mols: Sequence[Chem.Mol], names: Sequence[str], y: Sequence[float],
             fieldset: Optional[FieldSet] = None, alphas=ALPHAS, folds: int = 10,
             seed: int = 0, charge_model: str = "gasteiger", param_scheme: str = "rdkit",
             routes: Sequence[str] = ("aligned", "counts", "invariant", "topomer",
                                      "descriptors"),
             activity_prop: str = "", nested: bool = True, progress=None) -> Diagnosis:
    """Ask a data set the questions that should precede modelling it."""
    mols, names = list(mols), list(names)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if not np.all(ok):
        mols = [m for m, g in zip(mols, ok) if g]
        names = [n for n, g in zip(names, ok) if g]
        y = y[ok]
    global NESTED
    was, NESTED = NESTED, bool(nested)
    try:
        return _diagnose(mols, names, y, fieldset, alphas, folds, seed, charge_model,
                         param_scheme, routes, activity_prop, bool(nested), progress)
    finally:
        NESTED = was


def _diagnose(mols, names, y, fieldset, alphas, folds, seed, charge_model, param_scheme,
              routes, activity_prop, nested, progress) -> Diagnosis:
    d = Diagnosis(n=len(mols), activity_sd=float(np.std(y)), nested=nested,
                  activity_range=(float(np.min(y)), float(np.max(y))))

    if len(mols) < 10 or np.std(y) <= 0:
        d.findings.append(Finding(
            "size", "warn",
            f"{len(mols)} molecules with an activity spread of {np.std(y):.2f}: too little to "
            f"support any conclusion", {"n": len(mols), "sd": float(np.std(y))}))
        d.recommended = "none"
        d.why = "there is not enough here to model"
        return d
    if np.std(y) < 0.5:
        d.findings.append(Finding(
            "spread", "warn",
            f"the activity spans only {np.ptp(y):.2f} log units (sd {np.std(y):.2f}); a good q2 "
            f"on this range still means little", {"sd": float(np.std(y))}))

    d.findings += frame_findings(mols, names)

    todo = [r for r in routes]
    for name in todo:
        if progress:
            progress(f"the {name} route")
        if name == "aligned":
            d.routes[name] = aligned_route(mols, names, y, fieldset, alphas, folds, seed,
                                           charge_model, param_scheme)
        elif name == "counts":
            d.routes[name] = counts_route(mols, names, y, folds, seed)
        elif name == "invariant":
            d.routes[name] = invariant_route(mols, names, y, alphas, folds, seed,
                                             charge_model, param_scheme)
        elif name == "topomer":
            d.routes[name] = topomer_route(mols, names, y, alphas, folds, seed,
                                           charge_model, param_scheme)
        elif name == "descriptors":
            d.routes[name] = descriptor_route(mols, y, folds=folds, seed=seed,
                                              exclude=[activity_prop] if activity_prop else ())

    aligned = d.routes.get("aligned")
    if aligned is not None and np.isfinite(aligned.alpha):
        if aligned.alpha in (min(alphas), max(alphas)):
            where = "bottom" if aligned.alpha == min(alphas) else "top"
            d.findings.append(Finding(
                "alpha-edge", "warn",
                f"the chosen attenuation {aligned.alpha:g} is at the {where} of the grid: the "
                f"search needed a value it was not offered, so widen it before believing the model",
                {"alpha": aligned.alpha}))

    desc = d.routes.get("descriptors")
    if desc is not None and np.isfinite(desc.q2) and desc.q2 >= 0.95:
        near = ", ".join(f"{n} (r {r:.2f})" for n, r in getattr(desc, "correlations", [])[:4])
        d.findings.append(Finding(
            "descriptor-leak", "warn",
            f"the descriptor baseline reproduces the activity almost exactly (q2 {desc.q2:+.3f}); "
            f"on a set this size that is not chemistry but the measurement itself, reached "
            f"through the file's own columns"
            + (f" -- the nearest are {near}" if near else "")
            + ". No descriptor conclusion can be drawn until they are excluded.",
            {"q2": desc.q2, "nearest": getattr(desc, "correlations", [])}))
        desc.note += "; not usable as a baseline (see the warning)"
        desc.leaked = True
    if desc is not None and getattr(desc, "dropped", None):
        d.findings.append(Finding(
            "activity-as-descriptor", "warn",
            "dropped from the descriptor baseline because they reproduce the activity rather "
            "than describe the molecule: " + ", ".join(desc.dropped)
            + " -- if one of these is a real descriptor, name it and run again",
            {"dropped": list(desc.dropped)}))

    for other in ("counts", "invariant", "topomer", "descriptors"):
        r = d.routes.get(other)
        if aligned is None or r is None or not np.isfinite(r.q2):
            continue
        against, y_against = aligned, y
        if r.subset is not None and len(r.subset) < len(y):
            # the topomer route covers only what can be cut, so the aligned model is refitted on
            # those same molecules; a subset against the whole set is not a comparison
            against = aligned_route([mols[i] for i in r.subset], [names[i] for i in r.subset],
                                    y[r.subset], fieldset, alphas, folds, seed, charge_model,
                                    param_scheme)
            against.name = "aligned (same molecules)"
            y_against = y[r.subset]
            d.routes[f"aligned on the {len(r.subset)} cuttable"] = against
        diff, lo, hi = paired_interval(against, r, y_against, seed=seed)
        beaten = np.isfinite(lo) and lo <= 0.0 <= hi or diff < MEANINGFUL
        text = (f"the aligned model is {diff:+.3f} in q2 against {other}"
                + (f" (95% interval {lo:+.3f} to {hi:+.3f})" if np.isfinite(lo) else ""))
        if other == "topomer":
            d.findings.append(Finding(
                "frame-vs-topomer", "warn" if beaten else "ok",
                text + (" -- cutting the molecules and posing the pieces by rule does as well as "
                        "the frame, on the molecules that can be cut" if beaten
                        else " -- the frame does better than the topomer rule"),
                {"difference": diff, "low": lo, "high": hi,
                 "n": len(r.subset) if r.subset is not None else len(y)}))
        elif other == "descriptors":
            d.findings.append(Finding(
                "decoration", "warn" if beaten else "ok",
                text + (" -- descriptors alone do as well, so the geometry adds nothing on this "
                        "set" if beaten else " -- the geometry adds something"),
                {"difference": diff, "low": lo, "high": hi}))
        else:
            d.findings.append(Finding(
                f"frame-vs-{other}", "warn" if beaten else "ok",
                text + (f" -- the frame gains nothing over {other}" if beaten
                        else f" -- the frame does better than {other}"),
                {"difference": diff, "low": lo, "high": hi}))

    d.recommended, d.why = _recommend(d)
    return d


def _recommend(d: Diagnosis) -> Tuple[str, str]:
    """Read the intervals, and say which route the set should take."""
    aligned = d.routes.get("aligned")
    inv = d.routes.get("invariant")
    counts = d.routes.get("counts")
    desc = d.routes.get("descriptors")
    if aligned is None:
        return "unknown", "the aligned model was not run"
    usable = [r for r in d.routes.values()
              if np.isfinite(r.q2) and not getattr(r, "leaked", False)]
    best = max(usable, key=lambda r: r.q2) if usable else aligned
    if best.q2 < 0.1:
        return "none", (f"no route reaches q2 0.1 (best {best.name} at {best.q2:+.3f}); the set "
                        f"does not support a model as it is")
    topo = d.routes.get("topomer")
    frameless = [r for r in (inv, counts) if r is not None and np.isfinite(r.q2)]
    beaten_by_frameless = any(aligned.q2 - r.q2 < MEANINGFUL for r in frameless)
    # only a conclusive "not in a common frame" counts; "cannot tell" must not drive a route
    not_superimposed = any(f.code == "superimposed" and f.level == "warn" for f in d.findings)
    many_chemotypes = any(f.code == "chemotypes" and f.level == "warn" for f in d.findings)
    leaked = getattr(desc, "leaked", False) if desc is not None else False
    if (desc is not None and not leaked and np.isfinite(desc.q2)
            and aligned.q2 - desc.q2 < MEANINGFUL):
        return ("descriptors", "descriptors alone match the 3D model, so a field model here is an "
                               "expensive way to count atoms; use the fields only if the co-field "
                               "is the purpose")
    if beaten_by_frameless:
        route = max(frameless, key=lambda r: r.q2)
        return (route.name, f"the frame gains nothing over {route.name}, which needs no alignment "
                            f"to curate and cannot be done wrongly")
    topo_beaten = any(f.code == "frame-vs-topomer" and f.level == "ok" for f in d.findings)
    if not_superimposed:
        if topo is not None and np.isfinite(topo.q2) and topo_beaten:
            return ("aligned after curating the frame",
                    "the molecules are not superimposed, so the frame measured above is whatever "
                    "the file happens to hold -- but the topomer rule, which needs no frame, is "
                    f"measurably worse ({topo.q2:+.3f}), so the frame is worth curating rather "
                    "than abandoning")
        if topo is not None and np.isfinite(topo.q2):
            return ("topomer",
                    "the molecules are not superimposed, so the frame measured above is whatever "
                    "the file happens to hold; the topomer rule needs none and reaches q2 "
                    f"{topo.q2:+.3f} on the molecules it can cut")
        return ("aligned after curating the frame",
                "the frame carries signal, but the molecules are not superimposed, so the one "
                "measured above is whatever the file happens to hold; curate it -- the topomer "
                "route, which would need no frame, could not be scored here")
    if many_chemotypes:
        return ("aligned, with the frame checked",
                "the frame carries signal and the molecules are superimposed, so a frame exists "
                "whoever made it; the several Murcko scaffolds only mean the program could not "
                "have built it, not that it is wrong")
    return ("aligned", "the frame carries signal that neither the frameless kernels nor the "
                       "descriptors reproduce")
