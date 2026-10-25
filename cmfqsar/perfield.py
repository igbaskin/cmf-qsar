"""A separate attenuation for every field, chosen by the evidence (plan item D6).

The attenuation alpha_f decides how far a field reaches: large alpha makes it sharp, so two
molecules must match closely before the overlap integral calls them similar, and small alpha makes
it diffuse.  There is no reason for one number to be right for all five fields.  Electrostatics is
a long-range property and its field is the one a crystallographer would draw widest; a hydrogen-bond
donor is a point with a direction, and a field around it that reaches 4 A is making a claim about
the molecule that the chemistry does not support.  The 2013 protocol nevertheless shared one alpha
across the fields, and said why: a grid over five widths is five times the search, and a grid over
one width per field is 5^5 of it.

What makes the second grid affordable is two observations rather than a faster computer.

The first is that the per-field kernels are independent.  The joint kernel is a weighted sum,
``K = sum_f h_f c_f K_f``, and K_f depends on alpha_f alone, so the kernels for a grid of A widths
over F fields are the same FA matrices the shared search already computes -- it simply throws away
every combination but the diagonal one.  Assembling the kernel for any assignment of widths to
fields is then a weighted sum of matrices already in memory, not a rebuild.

The second is that the evidence (:mod:`cmfqsar.evidence`) scores an assignment with one
factorisation instead of ten, which is what item B4 measured.  A coordinate descent over the
widths -- hold the others, try every width for one field, keep the best, and go round again --
costs ``passes * F * A * L`` factorisations, a few hundred, where the same descent on a
cross-validated score would cost ten times that and the full grid 5^5 times.

What this does not do is hold anything out, and over 5^5 assignments the evidence has room to
please itself.  The descent is therefore deliberately short (two passes by default, from the best
shared width rather than from nothing), and whether the extra freedom pays is a question for a
test set, not for the criterion that chose it: see ``benchmarks/run_perfield.py``.

Fields whose kernel does not depend on alpha at all -- the probe fields, whose widths come from
the expansion in :mod:`cmfqsar.probes` -- are detected and skipped rather than searched in vain.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evidence import log_evidence


@dataclass
class PerFieldResult:
    """The outcome of the descent: the widths, the regularisation that goes with them, and the
    trace, including the shared-width configuration it started from so that the gain in evidence
    is readable off the result."""

    alphas: Dict[str, float]
    params: Dict[str, float]
    log_evidence: float
    shared_alpha: float
    shared_params: Dict[str, float]
    shared_log_evidence: float
    tuned: List[str] = _field(default_factory=list)
    skipped: List[str] = _field(default_factory=list)
    rows: List[dict] = _field(default_factory=list)
    n_eval: int = 0
    passes_used: int = 0

    @property
    def gain(self) -> float:
        """How much log evidence the per-field widths bought over the shared one."""
        return float(self.log_evidence - self.shared_log_evidence)

    @property
    def is_shared(self) -> bool:
        """True when the descent came back to one width for every field."""
        return len(set(round(v, 12) for v in self.alphas.values())) <= 1


def assemble(cache: Dict[float, tuple], choice: Sequence[float]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The per-field kernel stack for one width per field, taken out of the shared-search cache.

    `cache` maps a width to ``(K_fields, scale, self_diag)`` as :func:`cmfqsar.model.grid_search`
    stores it, with the fields along the first axis; `choice` gives the width of each field in
    that order.  The normalisation diagonals are selected field by field in the same way, because
    a field normalised at one width and kerneled at another is not a kernel at all.
    """
    K = np.stack([np.asarray(cache[a][0])[f] for f, a in enumerate(choice)])
    diags = [cache[a][2] for a in choice]
    if any(d is None for d in diags):
        return K, None
    return K, np.stack([np.asarray(d)[f] for f, d in enumerate(diags)])


def tunable_fields(cache: Dict[float, tuple], alphas: Sequence[float],
                   enabled: Sequence[bool]) -> Tuple[List[int], List[int]]:
    """Which fields the width actually moves: the enabled ones whose kernel changes with alpha.

    A probe field's kernel is built from its own expansion and ignores alpha, so searching over
    its width spends factorisations to rediscover the same number; a disabled field (h = 0) is
    multiplied out of the joint kernel entirely.  Both are reported as skipped rather than
    silently tuned.
    """
    tune, skip = [], []
    a0 = alphas[0]
    F = np.asarray(cache[a0][0]).shape[0]
    for f in range(F):
        if not enabled[f]:
            skip.append(f)
            continue
        moves = any(not np.allclose(np.asarray(cache[a0][0])[f], np.asarray(cache[a][0])[f],
                                    rtol=1e-10, atol=1e-12) for a in alphas[1:])
        (tune if moves else skip).append(f)
    return tune, skip


def descend(train, y: np.ndarray, model, cache: Dict[float, tuple], alphas: Sequence[float],
            grid: Sequence[Dict[str, float]], start_alpha: float,
            start_params: Dict[str, float], start_evidence: Optional[float] = None,
            passes: int = 2, progress: Optional[Callable[[str, float], None]] = None,
            cancel: Optional[Callable[[], bool]] = None) -> PerFieldResult:
    """Coordinate descent on the evidence over one attenuation per field.

    Starts from `start_alpha` for every field -- the width the shared search chose, so the descent
    can only improve on the configuration the program would have used -- and visits the tunable
    fields in turn, trying every width of `alphas` and every regularisation of `grid` and keeping
    whatever the evidence likes best.  A pass that changes nothing ends the search.

    The model is left holding the chosen widths, the kernels assembled from them and the
    regularisation that goes with them, so a fit follows immediately.  `cache` is not modified.
    """
    y = np.asarray(y, float)
    names = [s.name for s in model.fieldset.specs]
    enabled = [bool(s.enabled) and float(s.h) != 0.0 for s in model.fieldset.specs]
    alphas = [float(a) for a in alphas]
    grid = [dict(p) for p in grid] or [dict(start_params)]
    missing = [a for a in alphas if a not in cache]
    if missing:
        raise ValueError(f"no cached kernels for the widths {missing}; the per-field search reuses "
                         "the kernels of the shared grid and cannot build its own")

    def score(choice: Sequence[float], params: Dict[str, float]) -> float:
        Kf, sd = assemble(cache, choice)
        model.set_training_kernels(train, Kf, sd)
        K = model.joint_kernel(Kf, train.descriptors)
        return float(log_evidence(K, y, float(params.get("lam", 1.0)), fit_bias=model.fit_bias))

    tune, skip = tunable_fields(cache, alphas, enabled)
    choice = [float(start_alpha)] * len(names)
    params = dict(start_params)
    best = float(start_evidence) if start_evidence is not None else score(choice, params)
    shared_ev = best
    rows: List[dict] = []
    n_eval = 0
    used = 0

    total = max(1, passes * max(1, len(tune)) * len(alphas) * len(grid))
    for p in range(passes if tune and len(alphas) > 1 else 0):
        used = p + 1
        changed = False
        for f in tune:
            local = None
            for a in alphas:
                if a == choice[f]:
                    continue
                trial = list(choice)
                trial[f] = a
                for pr in grid:
                    if cancel and cancel():
                        from .model import Cancelled
                        raise Cancelled()
                    ev = score(trial, pr)
                    n_eval += 1
                    if progress:
                        progress(f"per-field pass {p + 1}: {names[f]} alpha = {a:g}, "
                                 f"log evidence = {ev:.1f}", min(1.0, n_eval / total))
                    if np.isfinite(ev) and (local is None or ev > local[0]):
                        local = (ev, a, dict(pr))
            if local is not None and local[0] > best + 1e-9:
                best, choice[f], params = local[0], local[1], local[2]
                changed = True
                rows.append({"pass": p + 1, "field": names[f], "alpha": local[1],
                             **local[2], "log_evidence": local[0]})
        if not changed:
            break

    # leave the model in the configuration the descent chose
    for f, a in enumerate(choice):
        model.fieldset.set_alpha(a, [names[f]])
    Kf, sd = assemble(cache, choice)
    model.set_training_kernels(train, Kf, sd)
    model.params.update(params)
    return PerFieldResult(alphas={n: float(a) for n, a in zip(names, choice)}, params=params,
                          log_evidence=float(best), shared_alpha=float(start_alpha),
                          shared_params=dict(start_params), shared_log_evidence=float(shared_ev),
                          tuned=[names[f] for f in tune], skipped=[names[f] for f in skip],
                          rows=rows, n_eval=n_eval, passes_used=used)
