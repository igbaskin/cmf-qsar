"""
Transport coefficients of a simulated electrolyte, for plan item BT9.

A cell model reads the salt diffusion coefficient of its electrolyte, and this program has no
prediction of it. Measured values exist for a few solvent mixtures only, so the labels a model
could be fitted on have to come from simulation, and a simulated label has to be tested against
a measurement before anything is fitted on it (rule 2 of items BT9 to BT11). This module makes
the label and the quantities it is tested by.

:mod:`cmfqsar.electrolyte` already runs a box of a named formulation, for the solvation shells of
item BT5. That run is right for the shape of a shell and wrong for a transport coefficient in
three ways, and :func:`run_transport` changes each of them.

* **The volume is fixed during production.** A barostat rescales the positions, which adds a
  displacement that is not diffusion. The box is equilibrated at constant pressure, set to its
  mean volume, and then run at constant volume.
* **The thermostat conserves momentum.** A Langevin thermostat adds a friction to every atom and
  slows diffusion by an amount that depends on the friction chosen. The production run uses a
  Nose-Hoover chain, which changes the kinetic energy without a drag on single atoms.
* **Unwrapped centres are stored, often.** A displacement needs positions that are not folded
  back into the box. The centre of mass of every molecule is stored every picosecond, unwrapped,
  which is small (tens of megabytes for ten nanoseconds) and is all that every quantity here
  needs. No atom trajectory is written.

What is computed from the centres, with the convention each one uses.

* **A self-diffusion coefficient** (:func:`diffusion`) is the slope of the mean squared
  displacement of one species against time, divided by six. The slope is fitted in a window of
  lag times that starts late enough for the motion to be diffusive, and the exponent of the mean
  squared displacement in that window is reported next to it: a value near one says the window
  is in the diffusive regime, and a value below it says the run is too short.
* **The error of a coefficient** is the standard error over blocks of time origins. The run is
  cut into contiguous blocks, the mean squared displacement is computed from the origins of
  each block at the same lags, and each block gives a slope. The error therefore belongs to the
  estimate it stands next to: the same lags, the same window. (Fitting each block in a shorter
  window of its own gives a different and more precise estimator, and its scatter understates
  the error of the reported one by a factor of three on a random walk.) It is the error of the
  mean over the time that was run and says nothing about a bias of the force field.
* **The finite box** (:func:`size_factor`, :func:`extrapolate_size`). A diffusion coefficient
  from a periodic box is too small by ``xi k T / (6 pi eta L)`` with ``xi`` = 2.837 (Yeh and
  Hummer, J. Phys. Chem. B 108 (2004) 15873), ``eta`` the viscosity and ``L`` the side of the
  box. The viscosity is not measured here. Boxes of several sizes give the limit of an infinite
  box and, from the slope against ``1 / L``, the viscosity; together these give a hydrodynamic
  radius, and with that radius the correction of a box of any size is a factor.
* **The salt diffusion coefficient** (:func:`salt_diffusivity`) of a dilute, ideal solution is
  ``2 D+ D- / (D+ + D-)``. For a concentrated solution it is multiplied by the thermodynamic
  factor, which is item BT11's and is passed in.
* **The conductivity** two ways. :func:`nernst_einstein_conductivity` is what the two
  self-diffusion coefficients would give if every ion moved on its own. :func:`conductivity`
  is the Einstein relation for the displacement of the total charge, which includes the motion
  of ions together: a cation and an anion that move as a pair carry no current. The second is
  the one to compare with a measurement, and its ratio to the first says how far the ions are
  from moving on their own. Both use the formal charges of the ions, also when the force field
  scales them.

**Scaled ion charges.** A force field with fixed charges has no polarisation and binds a small
cation too strongly, which slows it. Multiplying the charges of the ions by a factor near 0.8 is
the usual repair in the literature. Here it is a setting (`ion_charge_scale`) to be tested against
the measured conductivity, never a default.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import electrolyte as E

BOLTZMANN = 1.380649e-23            # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
AVOGADRO = 6.02214076e23
#: the constant of the finite-box correction for a cubic box (Yeh and Hummer, 2004)
YEH_HUMMER = 2.837297
#: one square angstrom per picosecond in square metres per second
A2_PER_PS = 1.0e-8


class TransportError(RuntimeError):
    """A transport coefficient cannot be computed from what was given."""


# ----------------------------------------------------------------------------------------------
# mean squared displacements
# ----------------------------------------------------------------------------------------------

def msd(positions, max_lag: Optional[int] = None, batch: int = 64) -> np.ndarray:
    """The mean squared displacement of a set of particles at every lag, in the square of the
    unit of the positions.

    `positions` has the shape (frames, particles, 3) and must be unwrapped. The mean is over the
    particles and over every time origin, computed with the Fourier-transform form of the sum, so
    a run of ten thousand frames costs seconds and not hours. Entry `m` is the lag of `m` frames.
    """
    r = np.asarray(positions, float)
    if r.ndim != 3 or r.shape[2] != 3:
        raise TransportError(f"positions have the shape (frames, particles, 3), not {r.shape}")
    frames, particles = r.shape[0], r.shape[1]
    if frames < 2 or particles < 1:
        raise TransportError("a displacement needs at least two frames and one particle")
    top = frames - 1 if max_lag is None else min(int(max_lag), frames - 1)
    n = 1 << (2 * frames - 1).bit_length()
    norm = (frames - np.arange(frames)).astype(float)
    out = np.zeros(top + 1)
    for start in range(0, particles, int(batch)):
        x = r[:, start:start + int(batch), :]
        f = np.fft.rfft(x, n=n, axis=0)
        s2 = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:frames].sum(axis=2) / norm[:, None]
        d = np.concatenate([(x ** 2).sum(axis=2), np.zeros((1, x.shape[1]))], axis=0)
        q = 2.0 * d.sum(axis=0)
        s1 = np.empty((frames, x.shape[1]))
        for m in range(frames):
            q = q - d[m - 1] - d[frames - m]
            s1[m] = q / norm[m]
        out += (s1 - 2.0 * s2)[:top + 1].sum(axis=1)
    out /= particles
    out[0] = 0.0
    return out


def remove_drift(centres: np.ndarray, masses: Sequence[float]) -> np.ndarray:
    """The centres with the motion of the whole box taken out, frame by frame.

    `centres` holds every molecule of the box. A drift of the centre of mass of the box is not
    diffusion; the simulation removes it, and this removes what is left of it.
    """
    c = np.asarray(centres, float)
    m = np.asarray(masses, float).ravel()
    if c.shape[1] != len(m):
        raise TransportError(f"{c.shape[1]} molecules and {len(m)} masses")
    com = (c * m[None, :, None]).sum(axis=1) / m.sum()
    return c - com[:, None, :]


def default_window(total: float) -> Tuple[float, float]:
    """The lag times a slope is fitted between, for a run of the length `total`: from a
    fortieth of the run, and no earlier than 50, to a tenth of it.

    The start leaves out the short times at which a molecule is still inside its cage. The end
    is early on purpose: few independent stretches of the run contribute to a long lag, and on
    a random walk a window that ends at a tenth of the run gives a coefficient about twice as
    precise as one that ends at a quarter. A run too short for a start at 50 is fitted between
    a twentieth and a quarter of its length instead.
    """
    total = float(total)
    low = max(50.0, 0.025 * total)
    high = 0.10 * total
    if high <= low:
        low, high = 0.05 * total, 0.25 * total
    return low, high


def fit_slope(times, values, window: Tuple[float, float]) -> Dict[str, float]:
    """The slope of `values` against `times` inside `window`, by least squares, and the exponent
    of `values` there (the slope of its logarithm against the logarithm of the time)."""
    t = np.asarray(times, float)
    v = np.asarray(values, float)
    keep = (t >= window[0]) & (t <= window[1]) & (t > 0)
    if keep.sum() < 3:
        raise TransportError(f"fewer than three lag times lie between {window[0]:g} and "
                             f"{window[1]:g}; the run is too short for this window")
    slope, intercept = np.polyfit(t[keep], v[keep], 1)
    positive = keep & (v > 0)
    exponent = (float(np.polyfit(np.log(t[positive]), np.log(v[positive]), 1)[0])
                if positive.sum() >= 3 else float("nan"))
    return {"slope": float(slope), "intercept": float(intercept), "exponent": exponent,
            "points": int(keep.sum())}


def _origin_blocks(r: np.ndarray, lags: np.ndarray, blocks: int) -> Tuple[np.ndarray, np.ndarray]:
    """The mean squared displacement at the lags `lags`, separately for the time origins of each
    of `blocks` contiguous parts of the run. Returns the table (blocks, lags) and how many
    origins each entry is the mean of."""
    table = np.empty((int(blocks), len(lags)))
    count = np.empty((int(blocks), len(lags)))
    for j, lag in enumerate(lags):
        d = r[lag:] - r[:-lag]
        per_origin = (d ** 2).sum(axis=2).mean(axis=1)
        edges = np.linspace(0, len(per_origin), int(blocks) + 1).astype(int)
        for b, (a, e) in enumerate(zip(edges[:-1], edges[1:])):
            table[b, j] = per_origin[a:e].mean()
            count[b, j] = e - a
    return table, count


def diffusion(positions, dt_ps: float, window: Optional[Tuple[float, float]] = None,
              blocks: int = 5, points: int = 16) -> Dict[str, object]:
    """The self-diffusion coefficient of one species, in m2/s, with its error over blocks.

    `positions` are unwrapped centres in angstrom, shape (frames, particles, 3), `dt_ps` the
    time between frames. `window` is the range of lag times, in picoseconds, the slope is fitted
    in; without one, :func:`default_window` of the length of the run. The mean squared
    displacement is computed at `points` lags spread evenly over the window, from every time
    origin, and the coefficient is the slope of a straight line through them, divided by six.

    The error is the standard error of the same slope over `blocks` blocks of time origins, so
    it is the error of this estimate and of no other. With one block there is no error.
    """
    r = np.asarray(positions, float)
    if r.ndim != 3 or r.shape[2] != 3:
        raise TransportError(f"positions have the shape (frames, particles, 3), not {r.shape}")
    frames = r.shape[0]
    dt = float(dt_ps)
    total = (frames - 1) * dt
    w = default_window(total) if window is None else (float(window[0]), float(window[1]))
    lags = np.unique(np.round(np.linspace(w[0] / dt, w[1] / dt, int(points))).astype(int))
    lags = lags[lags >= 1]
    blocks = max(1, int(blocks))
    if len(lags) < 3 or lags[-1] > frames - 1 - blocks:
        raise TransportError(f"a run of {total:g} ps is too short for a slope between "
                             f"{w[0]:g} and {w[1]:g} ps")
    table, count = _origin_blocks(r, lags, blocks)
    times = lags * dt
    curve = (table * count).sum(axis=0) / count.sum(axis=0)
    fit = fit_slope(times, curve, (times[0], times[-1]))
    per_block = ([float(np.polyfit(times, row, 1)[0] / 6.0 * A2_PER_PS) for row in table]
                 if blocks > 1 else [])
    se = (float(np.std(per_block, ddof=1) / np.sqrt(len(per_block))) if len(per_block) > 1
          else float("nan"))
    return {"d": fit["slope"] / 6.0 * A2_PER_PS, "se": se, "exponent": fit["exponent"],
            "window_ps": [w[0], w[1]], "blocks": per_block, "particles": int(r.shape[1]),
            "length_ps": total, "lags_ps": [float(t) for t in times],
            "msd": [float(v) for v in curve]}


# ----------------------------------------------------------------------------------------------
# from self-diffusion to what a cell model and a conductivity cell read
# ----------------------------------------------------------------------------------------------

def salt_diffusivity(d_cation: float, d_anion: float, factor: float = 1.0) -> float:
    """The diffusion coefficient of the salt: ``2 D+ D- / (D+ + D-)`` for a dilute, ideal
    solution of two singly charged ions, times `factor`, the thermodynamic factor."""
    dp, dm = float(d_cation), float(d_anion)
    if dp <= 0 or dm <= 0:
        raise TransportError("both self-diffusion coefficients have to be above zero")
    return 2.0 * dp * dm / (dp + dm) * float(factor)


def transport_number(d_cation: float, d_anion: float) -> float:
    """The share of the current the cation would carry if every ion moved on its own,
    ``D+ / (D+ + D-)``. It ignores the motion of ions together, so it is not the transference
    number a cell model reads (plan item BT10); it is that item's first reference."""
    return float(d_cation) / (float(d_cation) + float(d_anion))


def molarity(pairs: int, volume_a3: float) -> float:
    """Moles of salt per litre, from the number of ion pairs and the volume of the box in
    cubic angstrom."""
    return float(pairs) / AVOGADRO / (float(volume_a3) * 1.0e-27)


def nernst_einstein_conductivity(d_cation: float, d_anion: float, pairs: int, volume_a3: float,
                                 temperature: float, charge: float = 1.0) -> float:
    """The conductivity, in S/m, that the two self-diffusion coefficients would give if every
    ion moved on its own: ``q^2 e^2 N (D+ + D-) / (V k T)``."""
    v = float(volume_a3) * 1.0e-30
    return float((float(charge) * ELEMENTARY_CHARGE) ** 2 * int(pairs)
                 * (float(d_cation) + float(d_anion)) / (v * BOLTZMANN * float(temperature)))


def conductivity(cations, anions, dt_ps: float, volume_a3: float, temperature: float,
                 charge: float = 1.0, window: Optional[Tuple[float, float]] = None,
                 blocks: int = 5) -> Dict[str, object]:
    """The conductivity, in S/m, from the displacement of the total charge (the Einstein
    relation), with its error over blocks.

    The charge that has crossed the box is ``q e (sum of the cation displacements - sum of the
    anion displacements)``, and its mean squared displacement grows as ``6 V k T sigma t``.
    Unlike a self-diffusion coefficient this is one quantity per box and not one per ion, so it
    converges much more slowly, and its block error is the number to look at first.
    """
    cp = np.asarray(cations, float)
    am = np.asarray(anions, float)
    if cp.shape[0] != am.shape[0]:
        raise TransportError("the two species have different numbers of frames")
    total = float(charge) * (cp.sum(axis=1) - am.sum(axis=1))          # (frames, 3), e angstrom
    res = diffusion(total[:, None, :], dt_ps, window=window, blocks=blocks)
    # diffusion() returns slope / 6 in m2/s for a "particle" whose position is in e angstrom
    scale = ELEMENTARY_CHARGE ** 2 / (float(volume_a3) * 1.0e-30 * BOLTZMANN
                                      * float(temperature))
    return {"sigma": res["d"] * scale, "se": res["se"] * scale, "exponent": res["exponent"],
            "window_ps": res["window_ps"], "blocks": [b * scale for b in res["blocks"]],
            "length_ps": res["length_ps"]}


# ----------------------------------------------------------------------------------------------
# the finite box
# ----------------------------------------------------------------------------------------------

def yeh_hummer_correction(side_a: float, viscosity: float, temperature: float) -> float:
    """What a periodic cubic box of the side `side_a` (angstrom) takes off a diffusion
    coefficient, in m2/s, at the viscosity `viscosity` (Pa s)."""
    return float(YEH_HUMMER * BOLTZMANN * float(temperature)
                 / (6.0 * np.pi * float(viscosity) * float(side_a) * 1.0e-10))


def extrapolate_size(d: Sequence[float], sides_a: Sequence[float], temperature: float,
                     se: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """The diffusion coefficient of an infinite box, from boxes of several sizes.

    A straight line is fitted to the coefficients against one over the side of the box,
    weighted by their errors when `se` is given. Its value at zero is the coefficient of an
    infinite box, and its slope gives the viscosity, because the correction of Yeh and Hummer
    is ``xi k T / (6 pi eta L)``. With two boxes the line passes through both and has no error;
    three or more are needed for the scatter to say anything.
    """
    y = np.asarray(d, float)
    x = 1.0 / (np.asarray(sides_a, float) * 1.0e-10)
    if len(y) != len(x) or len(y) < 2:
        raise TransportError("the extrapolation needs at least two boxes of different size")
    w = None
    if se is not None:
        s = np.asarray(se, float)
        if np.all(np.isfinite(s)) and np.all(s > 0):
            w = 1.0 / s
    slope, intercept = np.polyfit(x, y, 1, w=w)
    viscosity = (YEH_HUMMER * BOLTZMANN * float(temperature) / (6.0 * np.pi * -slope)
                 if slope < 0 else float("nan"))
    return {"d_infinite": float(intercept), "slope": float(slope), "viscosity": float(viscosity),
            "residual": float(np.sqrt(np.mean((np.polyval([slope, intercept], x) - y) ** 2)))}


def hydrodynamic_radius(d_infinite: float, viscosity: float, temperature: float) -> float:
    """The radius, in angstrom, of the sphere that would diffuse as fast in a liquid of that
    viscosity (Stokes and Einstein, with stick boundary conditions)."""
    return float(BOLTZMANN * float(temperature)
                 / (6.0 * np.pi * float(viscosity) * float(d_infinite)) * 1.0e10)


def size_factor(radius_a: float, side_a: float) -> float:
    """The factor that turns the diffusion coefficient of a periodic box into that of an
    infinite one, for a species of the hydrodynamic radius `radius_a` in a box of the side
    `side_a`: ``1 / (1 - xi R / L)``.

    This is the correction of Yeh and Hummer with the viscosity written through the
    coefficient itself, which is what makes it a factor. It lets one size study correct boxes
    of other formulations, on the assumption that the radius of a species changes less from one
    formulation to another than the viscosity does.
    """
    share = YEH_HUMMER * float(radius_a) / float(side_a)
    if not (0.0 <= share < 1.0):
        raise TransportError(f"a box of {side_a:g} angstrom is too small for a species of "
                             f"{radius_a:g} angstrom: the correction would be the whole value")
    return float(1.0 / (1.0 - share))


# ----------------------------------------------------------------------------------------------
# the simulation
# ----------------------------------------------------------------------------------------------

@dataclass
class TransportSettings:
    """What one transport run does."""

    temperature: float = 298.15
    pressure: float = 1.0
    timestep_fs: float = 2.0
    cutoff_nm: float = 1.0
    n_solvent: int = 250
    min_ions: int = 4
    #: at constant pressure with a Langevin thermostat; the volume is averaged over the second
    #: half
    equilibration_ps: float = 500.0
    #: at constant volume with the production thermostat, before anything is stored
    settle_ps: float = 50.0
    production_ps: float = 4000.0
    #: how often the centre of every molecule is stored
    sample_ps: float = 1.0
    seed: int = 1
    platform: str = "CUDA"
    charge_model: str = "openff-gnn-am1bcc-0.1.0-rc.3.pt"
    small_molecule_forcefield: str = "gaff-2.11"
    ion_forcefield: str = "amber14/tip3p.xml"
    #: the partial charges of the cation and the anion are multiplied by this
    ion_charge_scale: float = 1.0
    #: "nose-hoover" conserves momentum; "langevin" is there to measure what a friction costs
    thermostat: str = "nose-hoover"
    #: the collision frequency of the Nose-Hoover chain, or the friction of the Langevin
    #: thermostat, per picosecond
    coupling_per_ps: float = 1.0

    def steps(self, picoseconds: float) -> int:
        return max(1, int(round(float(picoseconds) * 1.0e3 / float(self.timestep_fs))))


def scale_ion_charges(system, atoms: Sequence[int], scale: float) -> None:
    """Multiply the partial charges of the atoms in `atoms` by `scale`, in place.

    The pair interactions the force field lists one by one (the scaled interactions between
    atoms three bonds apart) carry the product of two charges, and they are scaled as well: by
    `scale` for each of their two atoms that is in `atoms`.
    """
    import openmm

    chosen = set(int(a) for a in atoms)
    scale = float(scale)
    for force in system.getForces():
        if not isinstance(force, openmm.NonbondedForce):
            continue
        for i in chosen:
            q, sigma, eps = force.getParticleParameters(i)
            force.setParticleParameters(i, q * scale, sigma, eps)
        for k in range(force.getNumExceptions()):
            a, b, qq, sigma, eps = force.getExceptionParameters(k)
            power = int(a in chosen) + int(b in chosen)
            if power:
                force.setExceptionParameters(k, a, b, qq * scale ** power, sigma, eps)


def _molecule_table(topology, system) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """For every molecule (a residue of the topology): its name, the index of its first atom,
    its mass, and the masses of all atoms. Atoms of one molecule are contiguous by construction
    in :func:`cmfqsar.electrolyte.build`, which is what lets the centres be summed in one pass."""
    from openmm import unit

    names, starts = [], []
    expected = 0
    for res in topology.residues():
        idx = [a.index for a in res.atoms()]
        if idx != list(range(expected, expected + len(idx))):
            raise TransportError("the atoms of a molecule are not contiguous in the topology")
        names.append(res.name)
        starts.append(expected)
        expected += len(idx)
    atom_mass = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                          for i in range(system.getNumParticles())], float)
    starts = np.array(starts, int)
    return names, starts, np.add.reduceat(atom_mass, starts), atom_mass


def _centres(positions_a: np.ndarray, starts: np.ndarray, molecule_mass: np.ndarray,
             atom_mass: np.ndarray) -> np.ndarray:
    return (np.add.reduceat(positions_a * atom_mass[:, None], starts, axis=0)
            / molecule_mass[:, None])


def run_transport(form: E.Formulation, workdir: str,
                  settings: Optional[TransportSettings] = None,
                  progress=None) -> Dict[str, object]:
    """Pack, equilibrate and run one formulation for transport; returns what was done.

    The centres of all molecules, unwrapped, go to ``centres.npz`` in `workdir` (angstrom, one
    frame per `sample_ps`), with the name, the mass and the net charge of every molecule, and
    ``run.json`` holds the record. A `workdir` that already holds a ``run.json`` is not run
    again, so one directory is one setting.
    """
    s = settings or TransportSettings()
    if s.thermostat not in ("nose-hoover", "langevin"):
        raise TransportError(f"the thermostat is nose-hoover or langevin, not {s.thermostat!r}")
    E.require()
    import openmm
    from openmm import app, unit

    os.makedirs(workdir, exist_ok=True)
    done = os.path.join(workdir, "run.json")
    if os.path.exists(done):
        with open(done, encoding="utf-8") as fh:
            return json.load(fh)

    def say(text: str) -> None:
        if progress:
            progress(f"{form.name}: {text}")

    base = E.Settings(temperature=s.temperature, pressure=s.pressure, timestep_fs=s.timestep_fs,
                      cutoff_nm=s.cutoff_nm, n_solvent=s.n_solvent, min_ions=s.min_ions,
                      seed=s.seed, platform=s.platform, charge_model=s.charge_model,
                      small_molecule_forcefield=s.small_molecule_forcefield,
                      ion_forcefield=s.ion_forcefield)
    counts = form.counts(s.n_solvent, s.min_ions)
    packed_pdb, side = E.pack(form, counts, workdir, seed=s.seed)
    top, system, positions = E.build(form, counts, packed_pdb, side, base)
    names, starts, molecule_mass, atom_mass = _molecule_table(top, system)
    ion_res = E.ION_RESIDUE.get(form.cation, form.cation.upper())
    ion_atoms = [a.index for res in top.residues() if res.name in (ion_res, "ANI")
                 for a in res.atoms()]
    if s.ion_charge_scale != 1.0:
        scale_ion_charges(system, ion_atoms, s.ion_charge_scale)
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    atom_charge = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                            for i in range(system.getNumParticles())], float)
    molecule_charge = np.add.reduceat(atom_charge, starts)

    kelvin = float(s.temperature) * unit.kelvin
    step = float(s.timestep_fs) * unit.femtosecond
    try:
        platform = openmm.Platform.getPlatformByName(s.platform)
    except Exception:                        # noqa: BLE001 - fall back rather than fail
        platform = None

    def simulation(integrator):
        return (app.Simulation(top, system, integrator, platform) if platform is not None
                else app.Simulation(top, system, integrator))

    # 1. constant pressure, to find the volume
    t0 = time.time()
    first = openmm.LangevinMiddleIntegrator(kelvin, 1.0 / unit.picosecond, step)
    first.setRandomNumberSeed(int(s.seed))
    sim = simulation(first)
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=5000)
    sim.context.setVelocitiesToTemperature(kelvin, int(s.seed))
    say(f"equilibrating {s.equilibration_ps:.0f} ps at constant pressure")
    sim.step(s.steps(0.5 * s.equilibration_ps))
    volumes = []
    for _ in range(max(1, int(round(0.5 * s.equilibration_ps)))):
        sim.step(s.steps(1.0))
        volumes.append(sim.context.getState().getPeriodicBoxVolume()
                       .value_in_unit(unit.nanometer ** 3))
    mean_volume = float(np.mean(volumes))                                   # nm3
    state = sim.context.getState(getPositions=True, getVelocities=True)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    factor = (mean_volume / float(abs(np.linalg.det(box)))) ** (1.0 / 3.0)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    # every molecule is moved as a whole, so that no bond is stretched by the change of volume
    centre = _centres(pos, starts, molecule_mass, atom_mass)
    sizes = np.diff(np.append(starts, len(atom_mass)))
    pos = pos + np.repeat((factor - 1.0) * centre, sizes, axis=0)
    velocities = state.getVelocities(asNumpy=True)
    del sim

    # 2. constant volume, with the production thermostat
    for k in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(k), openmm.MonteCarloBarostat):
            system.removeForce(k)
    if s.thermostat == "nose-hoover":
        second = openmm.NoseHooverIntegrator(kelvin, float(s.coupling_per_ps) / unit.picosecond,
                                             step)
    else:
        second = openmm.LangevinMiddleIntegrator(
            kelvin, float(s.coupling_per_ps) / unit.picosecond, step)
        second.setRandomNumberSeed(int(s.seed) + 1)
    sim = simulation(second)
    sim.context.setPeriodicBoxVectors(*(box * factor * unit.nanometer))
    sim.context.setPositions(pos * unit.nanometer)
    sim.context.setVelocities(velocities)
    say(f"settling {s.settle_ps:.0f} ps at constant volume")
    sim.step(s.steps(s.settle_ps))
    equilibration_seconds = time.time() - t0

    # 3. production: the centre of every molecule, unwrapped
    frames = max(1, int(round(float(s.production_ps) / float(s.sample_ps))))
    per_frame = s.steps(s.sample_ps)
    centres = np.empty((frames + 1, len(names), 3), np.float32)
    degrees = (3 * system.getNumParticles() - system.getNumConstraints() - 3)
    kinetic = []

    def record(k: int) -> None:
        st = sim.context.getState(getPositions=True, getEnergy=(k % 10 == 0))
        p = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        centres[k] = _centres(p, starts, molecule_mass, atom_mass)
        if k % 10 == 0:
            ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            kinetic.append(2.0 * ke / (degrees * 8.314462618e-3))

    say(f"running {s.production_ps:.0f} ps")
    t0 = time.time()
    record(0)
    for k in range(1, frames + 1):
        sim.step(per_frame)
        record(k)
    seconds = time.time() - t0

    side_a = float(mean_volume ** (1.0 / 3.0) * 10.0)
    volume_a3 = mean_volume * 1.0e3
    density = form.mass(counts) / AVOGADRO / (volume_a3 * 1.0e-24)
    np.savez_compressed(os.path.join(workdir, "centres.npz"), centres=centres,
                        names=np.array(names), masses=molecule_mass, charges=molecule_charge,
                        dt_ps=float(s.sample_ps), side=side_a)
    out = {
        "name": form.name, "salt": form.salt, "solvents": form.solvents,
        "molality": float(form.molality),
        "realised_molality": form.realised_molality(counts),
        "molarity": molarity(counts["ION"], volume_a3),
        "temperature": float(s.temperature),
        "mean_kinetic_temperature": float(np.mean(kinetic)),
        "counts": counts, "pairs": int(counts["ION"]),
        "atoms": int(system.getNumParticles()), "molecules": int(len(names)),
        "side": side_a, "volume": volume_a3, "density": float(density),
        "volume_sd_share": float(np.std(volumes) / mean_volume),
        "ion_residue": ion_res, "anion_residue": "ANI",
        "production_ps": float(s.production_ps), "frames": int(frames + 1),
        "seconds": float(seconds), "equilibration_seconds": float(equilibration_seconds),
        "ns_per_day": float(s.production_ps * 1.0e-3 / (seconds / 86400.0)),
        "centres": os.path.join(workdir, "centres.npz"),
        "settings": asdict(s),
    }
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return out


# ----------------------------------------------------------------------------------------------
# reading a run
# ----------------------------------------------------------------------------------------------

def read_centres(path: str) -> Dict[str, object]:
    """The stored centres of a run: `centres` (frames, molecules, 3) in angstrom, with the name,
    mass and net charge of every molecule, the time between frames and the side of the box."""
    with np.load(path, allow_pickle=False) as z:
        return {"centres": np.asarray(z["centres"], float),
                "names": [str(n) for n in z["names"]], "masses": np.asarray(z["masses"], float),
                "charges": np.asarray(z["charges"], float), "dt_ps": float(z["dt_ps"]),
                "side": float(z["side"])}


def analyse(run: Dict[str, object], first_ps: float = 0.0, last_ps: Optional[float] = None,
            blocks: int = 5) -> Dict[str, object]:
    """Everything this module computes for one run, from its stored centres.

    `first_ps` and `last_ps` cut the run, which is how the dependence on its length is read
    from one run instead of from several. The diffusion coefficients are those of the periodic
    box; the correction for its size is applied by the caller, who knows the radius to use.
    """
    data = read_centres(str(run["centres"]))
    dt = data["dt_ps"]
    a = int(round(float(first_ps) / dt))
    b = None if last_ps is None else int(round(float(last_ps) / dt)) + 1
    centres = remove_drift(data["centres"][a:b], data["masses"])
    names = np.array(data["names"])
    cation = centres[:, names == str(run["ion_residue"]), :]
    anion = centres[:, names == str(run["anion_residue"]), :]
    if cation.shape[1] == 0 or anion.shape[1] == 0:
        raise TransportError("the run holds no cation or no anion under the names it gives")
    temperature = float(run["temperature"])
    volume = float(run["volume"])
    pairs = int(cation.shape[1])
    dp = diffusion(cation, dt, blocks=blocks)
    dm = diffusion(anion, dt, blocks=blocks)
    out = {"length_ps": dp["length_ps"], "pairs": pairs, "side": float(run["side"]),
           "molarity": molarity(pairs, volume), "temperature": temperature,
           "cation": dp, "anion": dm, "solvents": {}}
    for name in sorted(set(names) - {str(run["ion_residue"]), str(run["anion_residue"])}):
        out["solvents"][name] = diffusion(centres[:, names == name, :], dt, blocks=blocks)
    out["salt"] = salt_diffusivity(dp["d"], dm["d"])
    out["transport_number"] = transport_number(dp["d"], dm["d"])
    out["nernst_einstein"] = nernst_einstein_conductivity(dp["d"], dm["d"], pairs, volume,
                                                          temperature)
    out["conductivity"] = conductivity(cation, anion, dt, volume, temperature, blocks=blocks)
    if out["nernst_einstein"] > 0:
        out["ionicity"] = out["conductivity"]["sigma"] / out["nernst_einstein"]
    return out


__all__ = ["A2_PER_PS", "AVOGADRO", "BOLTZMANN", "ELEMENTARY_CHARGE", "TransportError",
           "TransportSettings", "YEH_HUMMER", "analyse", "conductivity", "default_window",
           "diffusion", "extrapolate_size", "fit_slope", "hydrodynamic_radius", "molarity",
           "msd", "nernst_einstein_conductivity", "read_centres", "remove_drift",
           "run_transport", "salt_diffusivity", "scale_ion_charges", "size_factor",
           "transport_number", "yeh_hummer_correction"]
