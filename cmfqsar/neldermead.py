"""
Nelder-Mead simplex minimiser, a direct port of R's ``nmmin`` (``optim(method =
"Nelder-Mead")``, R 2.x-4.x), which the original CMF R scripts use to optimise the
attenuation factor alpha and the regularisation parameter gamma simultaneously
(``cmf_krr_train_mem`` with ``alpha_grid_search = FALSE``).

The port reproduces R's sequence of function evaluations exactly: the initial simplex
with step ``0.1 * max |x0|``, the reflection/expansion/contraction/shrink rules and
the convergence test on the spread of the function values only
(``VH <= VL + reltol * (|f(x0)| + reltol)``).
"""
from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple


def nmmin(fn: Callable[[List[float]], float], x0: Sequence[float], abstol: float = -math.inf,
          reltol: float = 1.490116e-08, maxit: int = 500, alpha: float = 1.0, bet: float = 0.5,
          gamm: float = 2.0) -> Tuple[List[float], float, int, int]:
    """Minimise ``fn`` from ``x0``; returns (x_min, f_min, n_evaluations, fail).

    ``fail`` is 0 on convergence, 1 when ``maxit`` evaluations were exceeded and 10 when
    the simplex could not shrink any further (the codes of R's ``optim``).
    """
    big = 1.0e35
    x0 = [float(v) for v in x0]
    n = len(x0)
    n1 = n + 1
    C = n + 2
    # P[i][j]: coordinate i of vertex j (j < n1); row n holds the function values;
    # column C-1 is the centroid.
    P = [[0.0] * (n + 2) for _ in range(n1)]
    f = fn(list(x0))
    if not math.isfinite(f):
        raise ValueError("function cannot be evaluated at initial parameters")
    funcount = 1
    convtol = reltol * (abs(f) + reltol)
    P[n1 - 1][0] = f
    for i in range(n):
        P[i][0] = x0[i]
    L = 1
    size = 0.0
    step = 0.0
    for i in range(n):
        if 0.1 * abs(x0[i]) > step:
            step = 0.1 * abs(x0[i])
    if step == 0.0:
        step = 0.1
    for j in range(2, n1 + 1):
        for i in range(n):
            P[i][j - 1] = x0[i]
        trystep = step
        while P[j - 2][j - 1] == x0[j - 2]:
            P[j - 2][j - 1] = x0[j - 2] + trystep
            trystep *= 10
        size += trystep
    oldsize = size
    calcvert = True
    fail = 0
    Bvec = list(x0)
    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    for i in range(n):
                        Bvec[i] = P[i][j]
                    f = fn(list(Bvec))
                    if not math.isfinite(f):
                        f = big
                    funcount += 1
                    P[n1 - 1][j] = f
            calcvert = False
        VL = P[n1 - 1][L - 1]
        VH = VL
        H = L
        for j in range(1, n1 + 1):
            if j != L:
                f = P[n1 - 1][j - 1]
                if f < VL:
                    L = j
                    VL = f
                if f > VH:
                    H = j
                    VH = f
        if VH <= VL + convtol or VL <= abstol:
            break
        for i in range(n):
            temp = -P[i][H - 1]
            for j in range(n1):
                temp += P[i][j]
            P[i][C - 1] = temp / n
        for i in range(n):
            Bvec[i] = (1.0 + alpha) * P[i][C - 1] - alpha * P[i][H - 1]
        f = fn(list(Bvec))
        if not math.isfinite(f):
            f = big
        funcount += 1
        VR = f
        if VR < VL:
            P[n1 - 1][C - 1] = f
            for i in range(n):
                f = gamm * Bvec[i] + (1 - gamm) * P[i][C - 1]
                P[i][C - 1] = Bvec[i]
                Bvec[i] = f
            f = fn(list(Bvec))
            if not math.isfinite(f):
                f = big
            funcount += 1
            if f < VR:
                for i in range(n):
                    P[i][H - 1] = Bvec[i]
                P[n1 - 1][H - 1] = f
            else:
                for i in range(n):
                    P[i][H - 1] = P[i][C - 1]
                P[n1 - 1][H - 1] = VR
        else:
            if VR < VH:
                for i in range(n):
                    P[i][H - 1] = Bvec[i]
                P[n1 - 1][H - 1] = VR
            for i in range(n):
                Bvec[i] = (1 - bet) * P[i][H - 1] + bet * P[i][C - 1]
            f = fn(list(Bvec))
            if not math.isfinite(f):
                f = big
            funcount += 1
            if f < P[n1 - 1][H - 1]:
                for i in range(n):
                    P[i][H - 1] = Bvec[i]
                P[n1 - 1][H - 1] = f
            else:
                if VR >= VH:
                    calcvert = True
                    size = 0.0
                    for j in range(n1):
                        if j + 1 != L:
                            for i in range(n):
                                P[i][j] = bet * (P[i][j] - P[i][L - 1]) + P[i][L - 1]
                                size += abs(P[i][j] - P[i][L - 1])
                    if size < oldsize:
                        oldsize = size
                    else:
                        fail = 10
                        break
        if funcount > maxit:
            break
    fmin = P[n1 - 1][L - 1]
    x = [P[i][L - 1] for i in range(n)]
    if funcount > maxit:
        fail = 1
    return x, fmin, funcount, fail


def interp_terms(alpha: float, grid: Sequence[float]) -> List[Tuple[float, float]]:
    """Linear interpolation weights of ``alpha`` between the points of a sorted ``grid``.

    Returns ``[(grid_lo, c_lo), (grid_hi, c_hi)]`` (or a single ``(grid_end, 1.0)`` term
    when ``alpha`` lies outside the grid), as ``cmf_kernels_interpolate`` of the original
    R scripts: kernel matrices are only available on the alpha grid and are interpolated
    linearly in between, with the end values used beyond the ends.
    """
    g = sorted(float(a) for a in grid)
    if not g:
        return [(float(alpha), 1.0)]
    if alpha <= g[0]:
        return [(g[0], 1.0)]
    if alpha >= g[-1]:
        return [(g[-1], 1.0)]
    for lo, hi in zip(g[:-1], g[1:]):
        if lo <= alpha <= hi:
            if hi == lo:
                return [(lo, 1.0)]
            c2 = (alpha - lo) / (hi - lo)
            if c2 == 0.0:
                return [(lo, 1.0)]
            return [(lo, 1.0 - c2), (hi, c2)]
    return [(g[-1], 1.0)]  # pragma: no cover
