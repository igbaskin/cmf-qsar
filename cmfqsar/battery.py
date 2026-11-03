"""
Electrode data for the periodic kernel: insertion pairs, and the voltage between them.

A cathode is measured by its **average voltage**: the work per electron of taking the working ion
out of the host.  For the insertion reaction

    Li_x Host  +  (y - x) Li  ->  Li_y Host

the average voltage over that interval is

    V = - [ E(Li_y Host) - E(Li_x Host) - (y - x) E(Li metal) ] / (y - x),

with the energies per host formula unit and in electronvolts, so that V comes out in volts.  The
reference to lithium metal is what makes the number a voltage against a lithium anode.

This module builds those pairs out of public data, without an account anywhere.  The Materials
Project serves its structures and their **formation energies** through OPTIMADE, which needs no
API key, and a formation energy already carries the elemental reference: it is the energy of the
cell minus the energies of the elements it is made of.  Elemental lithium is one of those
elements, so in the difference above it cancels, and

    V = - [ E_f(Li_y Host) - E_f(Li_x Host) ] / (y - x)

with E_f the formation energy per host formula unit.  Nothing else is needed, and the result is
the same quantity the battery databases publish, up to the corrections they apply (see
*What this is not*).

**How a pair is formed.**  One query per chemical system (``Li``, one metal, ``O`` -- the ternary
lithium-metal-oxygen systems) returns every structure of that system, the lithium-free host
included.  Each structure is reduced to its **host framework** (its composition with the working
ion removed, divided by the common factor) and its **lithium content per host formula unit**.
Structures that share a framework form a ladder in that content; the lowest structure above the
convex hull is kept at each step, and neighbouring steps make a pair.

**What this is not.**  The pairs are matched by host *composition*, not by host *structure*: two
cells with the same framework formula and different frameworks can be paired, which a topotactic
criterion would refuse.  The energies are the Materials Project's GGA/GGA+U values without the
corrections a battery database applies to them, so a voltage here can differ from the published
one by of the order of 0.1 V.  Both limits are measured rather than argued in
``benchmarks/run_battery.py``, which checks the lithium cobalt oxide and lithium iron phosphate
voltages against the published ones before it uses the set for anything.

The official insertion-electrode documents, with topotactic pairs and the corrections, need a
free Materials Project API key and the ``mp-api`` package; :func:`insertion_electrodes` fetches
them when a key is given, and the benchmark uses them in place of the reconstructed pairs if it
finds one.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: the Materials Project's OPTIMADE endpoint, which serves structures without an API key
MP_OPTIMADE = "https://optimade.materialsproject.org/v1"

#: what a structure must bring: its cell, its sites, and the thermodynamic record that holds the
#: formation energy
OPTIMADE_FIELDS = ("id", "chemical_formula_reduced", "lattice_vectors", "cartesian_site_positions",
                   "species", "species_at_sites", "nsites", "_mp_stability")

#: which of the Materials Project's functionals to read the formation energy from, best first
FUNCTIONALS = ("gga_gga+u", "gga_gga+u_r2scan", "r2scan")

#: the working ion of an insertion electrode unless another is named
WORKING_ION = "Li"


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
#: who the requests say they are; a service that is given no name answers some clients with 403
USER_AGENT = "cmfqsar/1.0 (https://github.com/igbaskin/cmf-qsar) python-urllib"


def _get(url: str, timeout: float = 90.0, tries: int = 3, pause: float = 2.0) -> dict:
    """One OPTIMADE request, with a short wait and a retry, as a parsed document."""
    import urllib.error
    import urllib.request
    last: Optional[Exception] = None
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as fh:  # noqa: S310 - fixed host
                return json.loads(fh.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"the structure service did not answer: {last}")


def fetch_system(elements: Sequence[str], endpoint: str = MP_OPTIMADE,
                 page_limit: int = 100, max_pages: int = 100,
                 timeout: float = 90.0) -> List[dict]:
    """Every structure whose elements are among those named, as OPTIMADE entries.

    ``elements HAS ONLY`` is the OPTIMADE way of asking for one chemical system: the lithium-free
    host, the fully lithiated phase and everything between them come back in one answer, which is
    what a ladder needs.
    """
    import urllib.parse
    listed = ",".join(f'"{el}"' for el in elements)
    query = {"filter": f"elements HAS ONLY {listed}",
             "response_fields": ",".join(OPTIMADE_FIELDS),
             "page_limit": str(int(page_limit))}
    url = f"{endpoint}/structures?" + urllib.parse.urlencode(query)
    out: List[dict] = []
    for _ in range(max_pages):
        doc = _get(url, timeout=timeout)
        out.extend(doc.get("data", []))
        nxt = (doc.get("links") or {}).get("next")
        if isinstance(nxt, dict):
            nxt = nxt.get("href")
        if not nxt:
            break
        url = nxt
    return out


def insertion_electrodes(key: Optional[str] = None, working_ion: str = WORKING_ION,
                         limit: int = 10000) -> List[dict]:
    """The Materials Project's own insertion-electrode documents, which need a free API key.

    The key is taken from the argument, then from ``MP_API_KEY`` or ``PMG_MAPI_KEY`` in the
    environment.  Without one this raises, and the caller falls back to the pairs this module
    reconstructs from formation energies, which need no key.
    """
    key = key or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        raise RuntimeError(
            "the insertion-electrode documents need a Materials Project API key: make an account "
            "at materialsproject.org, copy the key from the dashboard and set MP_API_KEY. The "
            "voltages this module reconstructs from formation energies need no key.")
    from mp_api.client import MPRester
    with MPRester(key) as mpr:
        docs = mpr.materials.insertion_electrodes.search(working_ion=working_ion, num_chunks=None)
    return [d.dict() if hasattr(d, "dict") else dict(d) for d in docs[:limit]]


def write_records(records: Sequence[dict], path: str) -> str:
    """Keep the fetched entries, so that a measurement does not depend on a service being up."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(list(records), fh)
    return path


def read_records(path: str) -> List[dict]:
    """The entries written by :func:`write_records`."""
    with open(path, encoding="utf-8") as fh:
        return list(json.load(fh))


# --------------------------------------------------------------------------- #
# Structures and energies
# --------------------------------------------------------------------------- #
def atoms_from_optimade(entry: dict):
    """One OPTIMADE entry as an ASE ``Atoms``, carrying its identifier and its energies.

    What travels in ``atoms.info``: ``id``, ``formation_energy_per_atom`` and
    ``energy_above_hull`` in eV, and ``functional``, the name of the calculation they came from.
    The descriptor reader of :mod:`cmfqsar.descriptors` looks there, so any of them can enter a
    model as a non-field descriptor.
    """
    from ase import Atoms
    attrs = entry.get("attributes", entry)
    names = list(attrs["species_at_sites"])
    symbol_of = {}
    for sp in attrs["species"]:
        elements = list(sp.get("chemical_symbols", []))
        if len(elements) != 1:
            raise ValueError(f"site species {sp.get('name')!r} of {entry.get('id')} shares its "
                             f"site between {len(elements)} elements; this reader takes ordered "
                             f"structures only")
        symbol_of[sp["name"]] = elements[0]
    atoms = Atoms(symbols=[symbol_of[nm] for nm in names],
                  positions=np.asarray(attrs["cartesian_site_positions"], float),
                  cell=np.asarray(attrs["lattice_vectors"], float), pbc=True)
    atoms.info["id"] = entry.get("id") or attrs.get("id", "")
    e_f, e_hull, functional = _energies(attrs)
    atoms.info["formation_energy_per_atom"] = e_f
    atoms.info["energy_above_hull"] = e_hull
    atoms.info["functional"] = functional
    return atoms


def _energies(attrs: dict) -> Tuple[float, float, str]:
    """The formation energy per atom, the energy above the hull, and which calculation gave them."""
    stability = attrs.get("_mp_stability") or {}
    for name in FUNCTIONALS:
        rec = stability.get(name)
        if isinstance(rec, dict) and rec.get("formation_energy_per_atom") is not None:
            return (float(rec["formation_energy_per_atom"]),
                    float(rec.get("energy_above_hull") or np.nan), name)
    return float("nan"), float("nan"), ""


def host_framework(atoms, ion: str = WORKING_ION) -> Tuple[Tuple[str, int], ...]:
    """The composition of the cell with the working ion removed, divided by its common factor.

    Two cells of one ladder share this: ``CoO2`` for both ``LiCoO2`` and ``Li0.5CoO2``, however
    large the cell each was computed in.
    """
    from collections import Counter
    from math import gcd
    counts = Counter(s for s in atoms.get_chemical_symbols() if s != ion)
    if not counts:
        return ()
    g = 0
    for c in counts.values():
        g = gcd(g, int(c))
    g = max(g, 1)
    return tuple(sorted((el, int(n) // g) for el, n in counts.items()))


def host_units(atoms, ion: str = WORKING_ION) -> float:
    """How many host formula units the cell holds."""
    from collections import Counter
    counts = Counter(s for s in atoms.get_chemical_symbols() if s != ion)
    framework = dict(host_framework(atoms, ion))
    if not framework:
        return float(len(atoms))
    return min(counts[el] / n for el, n in framework.items())


def ion_content(atoms, ion: str = WORKING_ION) -> float:
    """Working ions per host formula unit: the *x* of Li_x Host."""
    units = host_units(atoms, ion)
    n_ion = float(sum(1 for s in atoms.get_chemical_symbols() if s == ion))
    return n_ion / units if units > 0 else float("nan")


def formation_energy_per_host(atoms, ion: str = WORKING_ION) -> float:
    """The formation energy of the cell divided by the number of host formula units, in eV."""
    per_atom = float(atoms.info.get("formation_energy_per_atom", np.nan))
    units = host_units(atoms, ion)
    return per_atom * len(atoms) / units if units > 0 else float("nan")


def average_voltage(charged, discharged, ion: str = WORKING_ION) -> float:
    """The average voltage between two steps of one ladder, in volts.

    ``charged`` is the cell with less of the working ion in it.  Both formation energies are taken
    per host formula unit, so the cells need not be the same size.
    """
    x_c, x_d = ion_content(charged, ion), ion_content(discharged, ion)
    if not np.isfinite(x_c) or not np.isfinite(x_d) or x_d <= x_c:
        return float("nan")
    e_c = formation_energy_per_host(charged, ion)
    e_d = formation_energy_per_host(discharged, ion)
    return float(-(e_d - e_c) / (x_d - x_c))


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #
@dataclass
class InsertionPair:
    """Two steps of one host's ladder and the average voltage between them."""
    framework: str
    charged: object                      # the ASE Atoms with less of the working ion
    discharged: object                   # the ASE Atoms with more of it
    x_charged: float
    x_discharged: float
    voltage: float
    #: the relative volume change per host formula unit from charged to discharged
    volume_change: float = float("nan")
    #: how far each step lies above the convex hull, in eV per atom
    hull_charged: float = float("nan")
    hull_discharged: float = float("nan")
    ion: str = WORKING_ION
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """A label a table can carry: the framework and the two contents."""
        return f"{self.framework} x={self.x_charged:g}->{self.x_discharged:g}"

    @property
    def capacity(self) -> float:
        """The specific capacity of the interval, mAh per gram of the discharged cell."""
        mass = float(np.sum(self.discharged.get_masses())) / host_units(self.discharged, self.ion)
        delta = self.x_discharged - self.x_charged
        return 26801.5 * delta / mass if mass > 0 else float("nan")


def framework_label(framework: Sequence[Tuple[str, int]]) -> str:
    """``(("Co", 1), ("O", 2))`` as ``CoO2``."""
    return "".join(f"{el}{n if n != 1 else ''}" for el, n in framework)


def ladders(structures: Sequence, ion: str = WORKING_ION,
            tol: float = 1e-6) -> Dict[str, Dict[float, object]]:
    """Group the structures by host framework and keep the most stable cell at each content.

    The key of the inner mapping is the working-ion content per host formula unit, rounded to six
    decimals so that two cells of the same stoichiometry in different cell sizes meet.  The cell
    kept is the one lowest above the convex hull, which is the phase a measurement would see.
    """
    out: Dict[str, Dict[float, object]] = {}
    for atoms in structures:
        framework = host_framework(atoms, ion)
        if not framework:
            continue                                  # elemental lithium: the reference, not a host
        x = ion_content(atoms, ion)
        if not np.isfinite(x):
            continue
        if not np.isfinite(atoms.info.get("formation_energy_per_atom", np.nan)):
            continue
        key = framework_label(framework)
        step = round(float(x), 6)
        rung = out.setdefault(key, {})
        old = rung.get(step)
        if old is None or _hull(atoms) < _hull(old) - tol:
            rung[step] = atoms
    return out


def _hull(atoms) -> float:
    """How far a cell lies above the convex hull; a cell without the number goes last."""
    value = float(atoms.info.get("energy_above_hull", np.nan))
    return value if np.isfinite(value) else float("inf")


def insertion_pairs(structures: Sequence, ion: str = WORKING_ION,
                    min_delta: float = 1e-3, max_voltage: float = 6.0,
                    min_voltage: float = -1.0) -> List[InsertionPair]:
    """Every neighbouring pair of every ladder, with its average voltage.

    A pair is refused when the two steps hold the same amount of the working ion, when either
    formation energy is missing, or when the voltage falls outside ``min_voltage`` to
    ``max_voltage``: outside that range the pair is not an insertion electrode but two unrelated
    phases that happen to share a framework formula.
    """
    pairs: List[InsertionPair] = []
    for label, rung in sorted(ladders(structures, ion).items()):
        steps = sorted(rung)
        for x_c, x_d in zip(steps, steps[1:]):
            if x_d - x_c < min_delta:
                continue
            charged, discharged = rung[x_c], rung[x_d]
            volts = average_voltage(charged, discharged, ion)
            if not np.isfinite(volts) or not (min_voltage <= volts <= max_voltage):
                continue
            v_c = _volume_per_host(charged, ion)
            v_d = _volume_per_host(discharged, ion)
            pairs.append(InsertionPair(
                framework=label, charged=charged, discharged=discharged,
                x_charged=float(x_c), x_discharged=float(x_d), voltage=float(volts),
                volume_change=float((v_d - v_c) / v_c) if v_c > 0 else float("nan"),
                hull_charged=_hull(charged), hull_discharged=_hull(discharged), ion=ion,
                info={"id_charged": charged.info.get("id", ""),
                      "id_discharged": discharged.info.get("id", "")}))
    return pairs


def _volume_per_host(atoms, ion: str) -> float:
    try:
        volume = float(atoms.get_volume())
    except Exception:                                 # noqa: BLE001 - a structure without a cell
        return float("nan")
    units = host_units(atoms, ion)
    return volume / units if units > 0 else float("nan")


def pair_table(pairs: Sequence[InsertionPair], frame: str = "discharged"):
    """The structures, the voltages and the labels of a set of pairs, ready for ``crystal_table``.

    ``frame`` says which cell of the pair describes it: ``"discharged"`` (the lithiated one),
    ``"charged"``, or ``"both"``, which returns the two cells of every pair in order, each with
    the pair's voltage, so a kernel sees the ladder from both ends.
    """
    if frame not in ("discharged", "charged", "both"):
        raise ValueError(f"frame must be discharged, charged or both, not {frame!r}")
    structures, y, names = [], [], []
    for p in pairs:
        chosen = ([p.discharged] if frame == "discharged" else
                  [p.charged] if frame == "charged" else [p.charged, p.discharged])
        for atoms in chosen:
            structures.append(atoms)
            y.append(p.voltage)
            names.append(p.name)
    return structures, np.asarray(y, float), names


def chemistry(pair: InsertionPair) -> str:
    """Which family the host belongs to: an oxide, a phosphate, a sulfide, and so on.

    The families are the ones an electrode review groups by, and a pair that is in none of them is
    reported as ``other`` rather than forced into one.
    """
    elements = {el for el, _ in host_framework(pair.discharged, pair.ion)}
    if "P" in elements and "O" in elements:
        return "phosphate"
    if "Si" in elements and "O" in elements:
        return "silicate"
    if "S" in elements and "O" not in elements:
        return "sulfide"
    if "F" in elements and "O" not in elements:
        return "fluoride"
    if "O" in elements:
        return "oxide"
    return "other"
