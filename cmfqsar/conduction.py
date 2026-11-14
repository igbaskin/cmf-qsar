"""From hop barriers to a diffusion coefficient and a conductivity (plan item MS8).

The hops of a crystal form a periodic network of sites. With a rate on every hop, an ion on
that network performs a random walk, and its diffusion coefficient over long times is what a
hop simulation (kinetic Monte Carlo) would measure. For a network whose rates are the same in
both directions of a hop, that long-time limit has an exact form (homogenisation of a random
walk on a periodic graph). Find the correction ``chi`` of every site that makes the mean
displacement out of each site zero,

    sum over the hops out of i of  rate * (d + chi_b - chi_i) = 0,

which is one linear system with the graph Laplacian of the network. Then

    D = 1 / (2 N) * sum over hops of  rate * (d + chi_b - chi_a) (d + chi_b - chi_a)^T,

with ``N`` the number of sites in the cell and ``d`` the displacement of the hop. A network
that does not cross the cell gives D = 0, because the correction then cancels every
displacement. :func:`network_diffusion` solves this; :func:`simulate_walk` is the kinetic
Monte Carlo simulation it replaces, used to check it.

**The rates.** A hop of barrier ``E`` has the rate ``nu exp(-E / kT)``. The two directions of
a hop get the mean of their two barriers, so that the walk is reversible with every site
equally likely, which is what a network of barriers without site energies says.

**Thirty orders of magnitude.** At room temperature the rates of one network can differ by a
factor of 10^30, more than a double-precision linear system can hold. The conductance of such a
network is set by the hops near its percolation barrier (the critical-path argument of
Ambegaokar, Halperin and Langer), so the rates are divided by the rate of the critical hop,
hops more than ``window`` times slower are dropped (the network bypasses them) and hops more
than ``window`` times faster are capped (they act as short circuits). With the default window of
10^5 the change in D is below 10^-4 of its value, which a test checks; a direction whose own
percolation barrier lies more than ``kT ln(window)`` (0.30 eV at 298 K) above the lowest one
gets D = 0 in that direction instead of a value 10^5 times smaller.

**The conductivity.** The sites hold the mobile ion at a mean occupancy ``c``. In a lattice gas
whose only interaction is that a site holds one ion, the conductivity is

    sigma = q^2 (N / V) c (1 - c) D / (k T),

the Nernst-Einstein relation with the carriers ``c N / V`` and the share ``1 - c`` of empty
sites to move into (:func:`lattice_gas_conductivity`). A fully occupied network does not
conduct in this picture.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Boltzmann constant in eV/K and J/K, and the elementary charge in C
KB_EV = 8.617333262e-5
KB_J = 1.380649e-23
CHARGE = 1.602176634e-19
#: the rates kept around the critical one, as a factor either way
WINDOW = 1e5


def pair_barriers(edges: Sequence[Tuple[int, int, Tuple[int, int, int]]],
                  barriers: Sequence[float]) -> np.ndarray:
    """The barrier of each hop with its two directions averaged.

    `edges` lists hops as (site a, site b, cell offset of b); a hop and its reverse are
    (a, b, o) and (b, a, -o). A hop whose reverse is missing keeps its own barrier.
    """
    b = np.asarray(barriers, float)
    index = {(int(a), int(c), tuple(int(x) for x in o)): k for k, (a, c, o) in enumerate(edges)}
    out = b.copy()
    for k, (a, c, o) in enumerate(edges):
        r = index.get((int(c), int(a), tuple(-int(x) for x in o)))
        if r is not None and np.isfinite(b[r]) and np.isfinite(b[k]):
            out[k] = 0.5 * (b[k] + b[r])
    return out


def critical_rate(n_sites: int, edges, rates: np.ndarray) -> float:
    """The largest rate at which the hops at or above it still cross the cell (0 if none do).

    Hops are joined from the fastest down, with the union-find of the percolation barrier
    (:class:`cmfqsar.hops._Wrapping`), until a site meets one of its own images.
    """
    from .hops import _Wrapping

    uf = _Wrapping(int(n_sites))
    for k in np.argsort(-np.asarray(rates, float), kind="stable"):
        if rates[k] <= 0:
            break
        a, b, o = edges[k]
        uf.union(int(a), int(b), np.asarray(o, int))
        if uf.dimension() >= 1:
            return float(rates[k])
    return 0.0


def network_diffusion(n_sites: int, edges: Sequence[Tuple[int, int, Tuple[int, int, int]]],
                      vectors: np.ndarray, rates: np.ndarray, window: float = WINDOW,
                      site_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """The long-time diffusion tensor (3, 3) of one ion on a periodic network, in the units of
    ``vectors`` squared times those of ``rates`` (A^2/s for A and 1/s).

    Sites are 0 .. n_sites - 1. `edges` and `vectors` give every directed hop (a, b, offset)
    and its displacement; `rates` its rate. Without `site_weights` every site is equally likely
    and a hop and its reverse must have the same rate. With them (the probability of finding
    the ion on each site, summing to one) the walk must be reversible: weight(a) rate(a to b)
    equals weight(b) rate(b to a), as it is for rates from site and saddle energies
    (:func:`landscape_rates`). That product is the conductance of the hop, and only it enters.
    See the module text for the window around the critical conductance.
    """
    vectors = np.asarray(vectors, float)
    rates = np.asarray(rates, float)
    out = np.zeros((3, 3))
    if n_sites < 1 or len(rates) == 0:
        return out
    a_all = np.array([e[0] for e in edges], int)
    pi = (np.full(n_sites, 1.0 / n_sites) if site_weights is None
          else np.asarray(site_weights, float))
    cond = pi[a_all] * rates
    crit = critical_rate(n_sites, edges, cond)
    if crit <= 0:
        return out
    rel = cond / crit
    keep = rel >= 1.0 / window
    w = np.minimum(rel[keep], window)
    a = a_all[keep]
    b = np.array([edges[k][1] for k in np.flatnonzero(keep)], int)
    d = vectors[keep]
    L = np.zeros((n_sites, n_sites))
    np.add.at(L, (a, a), w)
    np.add.at(L, (a, b), -w)
    rhs = np.zeros((n_sites, 3))
    np.add.at(rhs, a, w[:, None] * d)
    chi = np.linalg.lstsq(L, rhs, rcond=None)[0]
    u = d + chi[b] - chi[a]
    out = (w[:, None, None] * u[:, :, None] * u[:, None, :]).sum(axis=0) / 2.0
    return out * crit


def landscape_rates(edges: Sequence[Tuple[int, int, Tuple[int, int, int]]],
                    site_energy: np.ndarray, saddle: np.ndarray, nu: float = 1e13,
                    temperature: float = 298.15) -> Tuple[np.ndarray, np.ndarray]:
    """Rates and site weights of a landscape with an energy on every site and every saddle.

    The rate of a hop from a is ``nu exp(-(saddle - E_a) / kT)`` and the weight of a site is
    its Boltzmann factor, normalised; a hop with an unknown saddle gets rate 0. Returns
    ``(rates, site_weights)`` for :func:`network_diffusion`.
    """
    kt = KB_EV * temperature
    E = np.asarray(site_energy, float)
    known = np.isfinite(E)
    if not known.any():
        raise ValueError("no site has an energy")
    e0 = float(E[known].min())
    pi = np.where(known, np.exp(-(np.where(known, E, e0) - e0) / kt), 0.0)
    pi = pi / pi.sum()
    s = np.asarray(saddle, float)
    a = np.array([e[0] for e in edges], int)
    rates = np.zeros(len(edges))
    ok = np.isfinite(s) & known[a]
    rates[ok] = nu * np.exp(-(s[ok] - E[a[ok]]) / kt)
    return rates, pi


def lattice_gas_conductivity(D: float, n_sites: int, occupancy: float, volume: float,
                             temperature: float = 298.15, charge: float = 1.0) -> float:
    """The conductivity in S/cm of a lattice gas of mobile ions.

    `D` is the diffusion coefficient of one ion on the empty network in A^2/s, `n_sites` the
    number of sites in a cell of `volume` A^3, `occupancy` their mean occupancy (0 to 1).
    """
    c = float(occupancy)
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"an occupancy of {c} is not between 0 and 1")
    density = n_sites / (volume * 1e-30)                     # sites per m^3
    sigma = (charge * CHARGE) ** 2 * density * c * (1.0 - c) * (D * 1e-20) \
        / (KB_J * temperature)                               # S/m
    return sigma / 100.0


def rates_from_barriers(barriers: np.ndarray, nu: float = 1e13,
                        temperature: float = 298.15) -> np.ndarray:
    """``nu exp(-E / kT)`` for barriers in eV; an infinite or missing barrier gives rate 0."""
    E = np.asarray(barriers, float)
    out = np.zeros_like(E)
    ok = np.isfinite(E)
    out[ok] = nu * np.exp(-E[ok] / (KB_EV * temperature))
    return out


def simulate_walk(n_sites: int, edges, vectors: np.ndarray, rates: np.ndarray,
                  walkers: int = 2000, steps: int = 2000, seed: int = 0) -> np.ndarray:
    """Kinetic Monte Carlo of independent walkers; the diffusion tensor from their spread.

    Each walker starts on a site chosen uniformly, picks a hop out of its site with a
    probability proportional to the rate, and advances its clock by an exponential waiting
    time. The tensor is ``<R R^T> / (2 t)`` at the end. This is the simulation that
    :func:`network_diffusion` solves in closed form; it is kept to check that form.
    """
    rng = np.random.default_rng(seed)
    vectors = np.asarray(vectors, float)
    rates = np.asarray(rates, float)
    out_of: Dict[int, np.ndarray] = {}
    for i in range(n_sites):
        out_of[i] = np.array([k for k, e in enumerate(edges) if e[0] == i and rates[k] > 0], int)
    site = rng.integers(0, n_sites, size=walkers)
    R = np.zeros((walkers, 3))
    t = np.zeros(walkers)
    for _ in range(steps):
        for i in np.unique(site):
            ks = out_of[i]
            mine = np.flatnonzero(site == i)
            if ks.size == 0:
                continue
            r = rates[ks]
            total = r.sum()
            pick = ks[rng.choice(ks.size, size=mine.size, p=r / total)]
            R[mine] += vectors[pick]
            t[mine] += rng.exponential(1.0 / total, size=mine.size)
            site[mine] = np.array([edges[k][1] for k in pick], int)
    T = float(np.mean(t))
    return (R[:, :, None] * R[:, None, :]).mean(axis=0) / (2.0 * T)
