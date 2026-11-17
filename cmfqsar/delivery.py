"""
What a predicted binding constant means for a delivery formulation (plan item SC2).

A host such as a cyclodextrin carries a guest (a drug) in solution.  Three questions a
formulation asks are functions of the 1:1 association constant K of the complex alone, and this
module answers them from K:

* **Solubility gain.**  A guest of intrinsic solubility S0 (its solubility without host) is
  dissolved in a solution of host at total concentration H_t, with solid guest in excess.  The
  mass balance of the 1:1 complex gives the total dissolved guest

      S_t = S0 + K S0 H_t / (1 + K S0)

  which is the straight line of an A_L phase-solubility diagram (Higuchi and Connors).  Its slope
  K S0 / (1 + K S0) is the fraction of the host that carries a guest.  Without S0 the gain
  S_t / S0 can only be bounded: 1 + K H_t, reached when K S0 is small.
* **Fraction bound after dilution.**  A solution of guest G_t and host H_t is diluted D times,
  as a formulation is when it is injected.  The complex concentration C solves the mass balance
  K (G - C)(H - C) = C with G = G_t / D and H = H_t / D, whose root in [0, min(G, H)] is

      C = (G + H + 1/K - sqrt((G + H + 1/K)^2 - 4 G H)) / 2

  and the fraction of the guest still bound is C / G.  With a guest much more dilute than the
  host this tends to K H / (1 + K H).
* **Release half-life.**  K = k_on / k_off.  Two spheres of radii r_A and r_B that diffuse
  freely meet at k_D = (2 R T / 3 eta) (r_A + r_B)^2 / (r_A r_B) (Smoluchowski, with the
  Stokes-Einstein diffusion coefficients).  It is smallest, 8 R T / (3 eta), for equal radii;
  in water at 25 C that is about 7.4e9 per molar per second, and it is the default here.  Then
  k_off = k_on / K, and a complex releases its guest with a half-life t_1/2 = ln 2 / k_off.  An
  association slower than k_D makes k_off smaller and the complex longer-lived, so the
  half-life from k_D is a lower bound.  A guest much smaller than its host can meet it somewhat
  faster than 8RT/(3 eta) -- 12.5 % faster at a ratio of radii of 2 -- which lowers that bound
  by the same factor; ``k_on`` sets the rate when it is known.

Every quantity grows with K, so the ends of an interval of K give the ends of the interval of
each quantity.  The concentrations are in mol/L, the constant in L/mol, the times in seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

#: the gas constant, J/(mol K)
R_GAS = 8.314462618
#: the viscosity of water at 25 C, in mPa s
WATER_VISCOSITY = 0.890
#: the scales an activity can be on, and what each means
SCALES = {
    "log10K": "log10 of K in L/mol (the scale of the cyclodextrin sets)",
    "lnK": "natural logarithm of K in L/mol",
    "K": "K itself, in L/mol",
    "dG": "the binding free energy in kJ/mol, negative for binding: K = exp(-dG / RT)",
}


def constant_from_activity(values, scale: str = "log10K", temperature: float = 298.15):
    """The association constant K (L/mol) from activities on `scale` (see ``SCALES``)."""
    v = np.asarray(values, float)
    if scale == "log10K":
        return 10.0 ** v
    if scale == "lnK":
        return np.exp(v)
    if scale == "K":
        return v.copy()
    if scale == "dG":
        return np.exp(-1000.0 * v / (R_GAS * float(temperature)))
    raise ValueError(f"unknown activity scale {scale!r}; the scales are {', '.join(SCALES)}")


def diffusion_limit(temperature: float = 298.15, viscosity_mpas: float = WATER_VISCOSITY) -> float:
    """The diffusion-limited rate constant 8RT/(3 eta) of two equal spheres, in L/(mol s)."""
    if viscosity_mpas <= 0 or temperature <= 0:
        raise ValueError("the temperature and the viscosity must be positive")
    per_m3 = 8.0 * R_GAS * float(temperature) / (3.0 * float(viscosity_mpas) * 1e-3)
    return per_m3 * 1000.0                       # m^3/(mol s) to L/(mol s)


def solubility_gain(K, host_total: float, intrinsic: Optional[float] = None):
    """S_t / S0 for an A_L phase-solubility diagram; without S0, the bound 1 + K H_t."""
    K = np.asarray(K, float)
    if host_total < 0:
        raise ValueError("the host concentration cannot be negative")
    if intrinsic is None:
        return 1.0 + K * host_total
    if intrinsic <= 0:
        raise ValueError("the intrinsic solubility must be positive")
    return 1.0 + K * host_total / (1.0 + K * intrinsic)


def bound_fraction(K, guest_total: Optional[float], host_total: float, dilution: float = 1.0):
    """The fraction of the guest in the complex after diluting `dilution` times.

    `guest_total` None is the limit of a guest much more dilute than the host.
    """
    K = np.asarray(K, float)
    if dilution < 1:
        raise ValueError("the dilution factor is 1 (none) or more")
    H = float(host_total) / float(dilution)
    if guest_total is None:
        return K * H / (1.0 + K * H)
    G = float(guest_total) / float(dilution)
    if G <= 0:
        raise ValueError("the guest concentration must be positive")
    b = G + H + 1.0 / np.maximum(K, 1e-300)
    disc = np.maximum(b * b - 4.0 * G * H, 0.0)
    # the smaller root, written to keep its digits when 4GH is much smaller than b^2
    C = 2.0 * G * H / (b + np.sqrt(disc))
    return C / G


def half_life(K, k_on: float):
    """ln 2 / k_off with k_off = k_on / K, in seconds."""
    if k_on <= 0:
        raise ValueError("the association rate constant must be positive")
    return math.log(2.0) * np.asarray(K, float) / float(k_on)


@dataclass
class Delivery:
    """The conditions of a formulation, and what a constant means under them."""

    host_total: float                       # mol/L
    intrinsic: Optional[float] = None       # mol/L, the solubility of the guest without host
    guest_total: Optional[float] = None     # mol/L, before dilution
    dilution: float = 1.0
    k_on: Optional[float] = None            # L/(mol s); None: the diffusion limit
    temperature: float = 298.15
    viscosity: float = WATER_VISCOSITY      # mPa s
    scale: str = "log10K"
    notes: List[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return float(self.k_on) if self.k_on else diffusion_limit(self.temperature,
                                                                 self.viscosity)

    def guest_for_dilution(self, K) -> Optional[np.ndarray]:
        """The guest concentration before dilution: the one given, else the saturated
        solubility of the formulation when S0 is known, else None (a dilute guest)."""
        if self.guest_total is not None:
            return np.full(np.shape(K), float(self.guest_total))
        if self.intrinsic is not None:
            return self.intrinsic * solubility_gain(K, self.host_total, self.intrinsic)
        return None

    def quantities(self, activity) -> Dict[str, np.ndarray]:
        """Every quantity, one value per activity."""
        K = constant_from_activity(activity, self.scale, self.temperature)
        out = {"K": K, "solubility_gain": solubility_gain(K, self.host_total, self.intrinsic)}
        if self.intrinsic is not None:
            out["solubility"] = self.intrinsic * out["solubility_gain"]
        G = self.guest_for_dilution(K)
        if G is None:
            out["bound_fraction"] = bound_fraction(K, None, self.host_total, self.dilution)
        else:
            out["bound_fraction"] = np.array([float(bound_fraction(k, g, self.host_total,
                                                                    self.dilution))
                                              for k, g in zip(np.ravel(K), np.ravel(G))])
        out["half_life_s"] = half_life(K, self.rate)
        return out

    def describe(self) -> str:
        """The conditions in words, with what each quantity assumes."""
        lines = [f"host {self.host_total:g} mol/L"]
        if self.intrinsic is not None:
            lines.append(f"intrinsic solubility of the guest {self.intrinsic:g} mol/L: the "
                         f"solubility gain is that of an A_L phase-solubility diagram")
        else:
            lines.append("no intrinsic solubility given: the solubility gain is the upper "
                         "bound 1 + K [host], reached when K S0 is small")
        if self.guest_total is not None:
            guest = f"guest {self.guest_total:g} mol/L"
        elif self.intrinsic is not None:
            guest = "guest at the solubility the host gives it"
        else:
            guest = "a guest much more dilute than the host"
        lines.append(f"fraction bound after diluting {self.dilution:g} times, {guest}")
        how = ("as given" if self.k_on else
               f"the diffusion limit 8RT/(3 eta) at {self.temperature:g} K and "
               f"{self.viscosity:g} mPa s")
        lines.append(f"half-life from k_on = {self.rate:.3g} L/(mol s), {how}; an association "
                     f"slower than that gives a longer life, so the half-life is a lower bound")
        return "; ".join(lines)


def format_time(seconds: float) -> str:
    """A time in the largest unit that keeps it above 1, for a reader."""
    s = float(seconds)
    if not np.isfinite(s):
        return "-"
    for unit, size in (("h", 3600.0), ("min", 60.0), ("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if s >= size:
            return f"{s / size:.3g} {unit}"
    return f"{s * 1e9:.3g} ns"


def rows(names: Sequence[str], activity, delivery: Delivery, low=None, high=None) -> List[Dict]:
    """One record per molecule: the quantities, with their interval when the ends are given."""
    q = delivery.quantities(activity)
    lo = delivery.quantities(low) if low is not None else None
    hi = delivery.quantities(high) if high is not None else None
    if delivery.scale == "dG" and lo is not None:
        lo, hi = hi, lo                     # a lower free energy is a larger constant
    out = []
    for i, n in enumerate(names):
        r = {"name": n}
        for k, v in q.items():
            r[k] = float(np.ravel(v)[i])
            if lo is not None:
                r[k + "_lo"] = float(np.ravel(lo[k])[i])
                r[k + "_hi"] = float(np.ravel(hi[k])[i])
        out.append(r)
    return out


__all__ = ["Delivery", "R_GAS", "SCALES", "WATER_VISCOSITY", "bound_fraction",
           "constant_from_activity", "diffusion_limit", "format_time", "half_life", "rows",
           "solubility_gain"]
