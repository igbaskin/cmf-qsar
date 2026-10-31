#!/usr/bin/env python
"""
The periodic overlap kernel of crystals against the periodic autocorrelation (plan item M4).

A crystal series on one parent lattice is the solid-state form of a congeneric series: the cells
share a frame, so the fields can be compared point by point and the program's own kernel -- the
overlap integral -- applies, with the integral taken over one cell and the periodic images summed
in (`cmfqsar.periodic.periodic_overlap_kernels`). What that gains over the rotation-invariant
autocorrelation the module already had is a **co-field inside the cell**: which site, and which
region around it, raises the property and which lowers it. An autocorrelation is a distance
distribution and can never say where.

The data
--------
`matbench_perovskites`: 18,928 ABO3 perovskite cells of five atoms with their formation energy
from DFT (Castelli et al. 2012, as distributed by Matbench v0.1). It is the largest public set
whose members share a cell topology -- A at the corner, B at the centre, O on the faces -- so the
overlap kernel has a frame to work in. The cells are relaxed, so the lattice constants differ by
a few per cent; the frame is therefore only approximate, and the script measures how approximate
(`cmfqsar.periodic.lattice_spread`) rather than assuming it away.

What is compared
----------------
Three kernels on the same cells, the same folds and the same ridge grid:

* **overlap** -- the periodic overlap kernel of item M4, cosine-normalised;
* **autocorrelation** -- the periodic field autocorrelation already in the module, which is what
  the item's gate names as the thing to beat;
* **open** -- the same overlap integral with the images left out, which is what the ordinary
  molecular kernel would do with these cells. The difference between it and the first row is what
  the periodic sum is worth.

Two frames, because the set is not quite one lattice:

* **as given** -- the relaxed cells as they come;
* **one lattice** -- every cell given the mean lattice constant with its fractional coordinates
  kept. Every cell of this set is cubic and axis-aligned, so this puts the series on one lattice
  exactly. It is the crystal analogue of posing a series by rule, and it separates "the kernel
  does not work" from "the lattices differ" -- at the price of scaling the bond lengths and of
  taking the lattice constant, a strong predictor in its own right, out of the kernel.

Reading the model
-----------------
The co-field of the best overlap model is drawn in the two planes of the cell that carry the
sites (z = 0 through the A site and the equatorial oxygens, z = 1/2 through the B site and the
apical oxygen), and the per-site contributions of decomposition rule 3 are averaged over the set.

Usage
-----
    python benchmarks/run_periodic_overlap.py                   # every stage, then the table
    python benchmarks/run_periodic_overlap.py --limit 800       # a quicker run
    python benchmarks/run_periodic_overlap.py --stage prepare
    python benchmarks/run_periodic_overlap.py --stage fit --kernels overlap autocorrelation
    python benchmarks/run_periodic_overlap.py --stage cofield
    python benchmarks/run_periodic_overlap.py --report
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar import periodic  # noqa: E402
from cmfqsar.fields import AtomTable  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel, channel_scale  # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels  # noqa: E402
from cmfqsar.model import CMFModel, _Learner, cv_predict, r2_score, rmse  # noqa: E402

URL = "https://ml.materialsproject.org/projects/matbench_perovskites.json.gz"
CACHE = os.path.join(HERE, "cache", "matbench")
RESULTS = os.path.join(HERE, "results", "periodic_overlap")
TARGET = "e_form"
#: the Matbench v0.1 fold seed, so that the folds are the published ones
SPLIT_SEED = 18012019
OUTER, INNER = 5, 10
LAMS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
#: attenuations of the overlap kernels.  The image radius follows from the attenuation, so a
#: small one costs more: 1.0 reaches 18 A, 4.0 reaches 9 A.
ALPHAS = (1.0, 2.0, 4.0)
#: distance grid of the autocorrelation, as `run_matbench_phonons.py` uses it
AC_SIGMAS = (0.35, 0.5, 0.75)
AC_R_MAX, AC_TAPER = 10.0, 2.0
LIMIT = 1500
KERNELS = ("overlap", "open", "autocorrelation")
FRAMES = ("as given", "one lattice")
#: cells whose sites are decomposed for the per-site table: the decomposition pairs every atom
#: of a cell against every image atom of the training set, which is about a second per cell
SITE_SAMPLE = 150
#: fields exported as a Gaussian cube over one cell, for a viewer that draws the structure too
CUBE_FIELDS = ("steric",)
#: electronegativity differences rather than pymatgen's oxidation-state guess: the guess is slow
#: on thousands of structures and fails on the mixed-valence members, and the charge field is a
#: field of this program either way (see `crystal_charges`)
CHARGES = "electronegativity"
#: the fields.  The three of :data:`cmfqsar.periodic.INORGANIC_FIELDS` and the inverse square
#: root of the mass, which is the field the phonon benchmark found worth adding.  Measured on
#: this set with the autocorrelation kernel, 1,500 cells, five folds: the three property fields
#: give R2 +0.370, the four +0.495, and adding indicator fields for the twelve commonest
#: elements +0.587.  The last would multiply the cost of the overlap kernel by four, and the
#: comparison here is between kernels on one field set, not between field sets.
FIELDS = periodic.INORGANIC_FIELDS + ("prop:inverse_sqrt_mass",)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def result(name: str) -> str:
    return os.path.join(RESULTS, name)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_default)
        fh.write("\n")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #

def fetch() -> dict:
    """The Matbench table, downloaded once into benchmarks/cache/matbench."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, os.path.basename(URL))
    if not os.path.exists(path):
        log(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (cmfqsar)"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as fh:
            fh.write(r.read())
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def reframed(atoms, cell):
    """The same structure in the reference cell, with its fractional coordinates kept.

    A frame by rule for crystals, and the one these cells allow: every cell of
    `matbench_perovskites` is cubic and axis-aligned, so giving them all one lattice constant
    puts the set on one lattice exactly, and the overlap integral then has a frame.  It is not
    free: the bond lengths are scaled with the cell, and the lattice constant -- which is itself
    a strong predictor of a formation energy -- leaves the kernel altogether.  That is why both
    frames are measured.
    """
    out = atoms.copy()
    out.set_cell(np.asarray(cell, float), scale_atoms=True)
    return out


def reference_cell(atoms_list):
    """A cubic cell of the mean lattice constant of the set."""
    a = float(np.mean([abs(np.linalg.det(np.asarray(a_.cell, float))) ** (1.0 / 3.0)
                       for a_ in atoms_list]))
    return np.eye(3) * a


def prepare(limit: int = LIMIT, force: bool = False) -> dict:
    """Read the table, take a subset, and record what the set looks like."""
    path = result("prepared.json")
    if os.path.isfile(path) and not force:
        rec = jload(path)
        if rec["limit"] == int(limit):
            log(f"prepare: {path} is there")
            return rec
    atoms, y, ids = periodic.structures_from_matbench(fetch(), target=TARGET)
    log(f"{len(atoms)} structures in the table")
    rng = np.random.RandomState(0)
    rows = np.sort(rng.choice(len(atoms), min(int(limit), len(atoms)), replace=False))
    atoms = [atoms[i] for i in rows]
    y = np.asarray(y, float)[rows]
    names = [f"{a.get_chemical_formula()}_{ids[i]}" for a, i in zip(atoms, rows)]
    fs = periodic.inorganic_fieldset(elements=[], physchem=FIELDS,
                                     charge_model=CHARGES, alpha=1.0)
    t0 = time.time()
    given = periodic.crystal_table(atoms, fs, names)
    cell = reference_cell(atoms)
    scaled = periodic.crystal_table([reframed(a, cell) for a in atoms], fs, names)
    log(f"fields {[s.name for s in fs.specs]}, {given.coords.shape[0]} atoms, "
        f"{time.time() - t0:.0f} s")
    rec = {"limit": int(limit), "n": int(len(y)), "rows": rows.tolist(), "target": TARGET,
           "fields": [s.name for s in fs.specs], "charges": CHARGES,
           "reference_cell": cell.tolist(), "lattice_constant": float(cell[0, 0]),
           "lattice_spread_as_given": periodic.lattice_spread(given),
           "lattice_spread_one_lattice": periodic.lattice_spread(scaled),
           "y_mean": float(np.mean(y)), "y_sd": float(np.std(y)),
           "atoms_per_cell": float(np.mean([m.n_atoms for m in given.molecules]))}
    jdump(rec, path)
    return rec


def load_set(limit: int = LIMIT):
    """(the tables of the two frames, the target, the field set) of the prepared subset."""
    rec = jload(result("prepared.json"))
    every, y_all, ids = periodic.structures_from_matbench(fetch(), target=TARGET)
    rows = np.asarray(rec["rows"], int)
    atoms = [every[i] for i in rows]
    y = np.asarray(y_all, float)[rows]
    names = [f"{a.get_chemical_formula()}_{ids[i]}" for a, i in zip(atoms, rows)]
    fs = periodic.inorganic_fieldset(elements=[], physchem=FIELDS,
                                     charge_model=rec["charges"], alpha=1.0)
    cell = np.asarray(rec["reference_cell"], float)
    tables = {"as given": periodic.crystal_table(atoms, fs, names),
              "one lattice": periodic.crystal_table([reframed(a, cell) for a in atoms],
                                                    fs, names)}
    return tables, y, fs, rec


def folds(n: int):
    from sklearn.model_selection import KFold

    return list(KFold(n_splits=OUTER, shuffle=True, random_state=SPLIT_SEED).split(np.arange(n)))


# --------------------------------------------------------------------------- #
# The kernels
# --------------------------------------------------------------------------- #

def set_alpha(fs, alpha: float):
    for sp in fs.specs:
        sp.alpha = float(alpha)
    return fs


def joint_overlap(table: AtomTable, fs, alpha: float, periodic_images: bool,
                  engine: str = "auto") -> np.ndarray:
    """The cosine-normalised, field-scaled joint overlap kernel of a set of cells.

    `periodic_images` chooses item M4's kernel or the open-boundary one on the same cells.
    """
    set_alpha(fs, alpha)
    specs = [s for s in fs.specs if s.enabled]
    if periodic_images:
        K = periodic.periodic_overlap_kernels(table, table, specs, engine=engine)
        d = periodic.periodic_self_overlaps(table, specs)
    else:
        from cmfqsar.kernels import self_kernels

        K = kernel_matrices(table, table, specs, symmetric=True, engine=engine)
        d = np.asarray(self_kernels(table, specs), float)
    K = normalize_kernels(K, d, d)
    scale = np.array([1.0 / max(float(np.mean(np.diag(K[f]))), 1e-300)
                      for f in range(K.shape[0])])
    return np.tensordot(scale * fs.mixing_vector()[:K.shape[0]], K, axes=(0, 0))


def joint_autocorrelation(table: AtomTable, fs, sigma: float):
    """The periodic autocorrelation kernel of the same cells, as the module builds it."""
    spec = periodic.PeriodicSpec(r_max=AC_R_MAX, step=sigma, sigma=sigma, taper=AC_TAPER,
                                 background=True)
    F = periodic.periodic_feature_matrix(table.molecules, spec, n_fields=table.weights.shape[1])
    blocks = periodic.periodic_block_sizes(spec, table.weights.shape[1])
    sc = channel_scale(F, spec.n_bins, blocks=blocks)
    return autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks)


def kernels_for(kind: str, table: AtomTable, fs, engine: str = "auto") -> dict:
    """Every setting of one kernel kind: the attenuations, or the distance widths."""
    out = {}
    if kind == "autocorrelation":
        for sigma in AC_SIGMAS:
            t0 = time.time()
            out[f"sigma {sigma:g}"] = joint_autocorrelation(table, fs, sigma)
            log(f"  autocorrelation sigma {sigma:g}: {time.time() - t0:.0f} s")
        return out
    for alpha in ALPHAS:
        t0 = time.time()
        out[f"alpha {alpha:g}"] = joint_overlap(table, fs, alpha, kind == "overlap", engine)
        log(f"  {kind} alpha {alpha:g}: {time.time() - t0:.0f} s")
    return out


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #

def evaluate(Ks: dict, y: np.ndarray) -> dict:
    """Nested cross-validation over the settings and the ridge, on the Matbench folds."""
    pred = np.full(len(y), np.nan)
    per_fold = []
    for k, (tr, te) in enumerate(folds(len(y))):
        best = None
        for setting, K in Ks.items():
            K_tr = K[np.ix_(tr, tr)]
            for lam in LAMS:
                p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, setting, lam)
        err, setting, lam = best
        K = Ks[setting]
        learner = _Learner("krr", {"lam": lam}, True).fit(K[np.ix_(tr, tr)], y[tr])
        pred[te] = learner.predict(K[np.ix_(te, tr)])
        per_fold.append({"fold": k, "setting": setting, "lam": lam, "inner_rmse": float(err),
                         "mae": float(np.mean(np.abs(pred[te] - y[te]))),
                         "rmse": float(rmse(y[te], pred[te])),
                         "r2": float(r2_score(y[te], pred[te]))})
        log(f"  fold {k}: {setting}, lam {lam:g}, R2 {per_fold[-1]['r2']:+.3f}, "
            f"MAE {per_fold[-1]['mae']:.4f}")
    return {"r2": float(r2_score(y, pred)), "rmse": float(rmse(y, pred)),
            "mae": float(np.mean(np.abs(pred - y))), "folds": per_fold,
            "predictions": pred.tolist()}


def fit(kinds=KERNELS, frames=FRAMES, limit: int = LIMIT, engine: str = "auto",
        force: bool = False) -> dict:
    path = result("fit.json")
    done = jload(path) if os.path.isfile(path) and not force else {}
    tables, y, fs, rec = load_set(limit)
    for frame in frames:
        for kind in kinds:
            key = f"{kind} | {frame}"
            if key in done:
                continue
            log(f"=== {key}")
            t0 = time.time()
            Ks = kernels_for(kind, tables[frame], fs, engine)
            out = evaluate(Ks, y)
            out.update({"kernel": kind, "frame": frame, "n": int(len(y)),
                        "seconds": time.time() - t0,
                        "settings": list(Ks), "fields": rec["fields"]})
            if kind == "overlap":
                out["asymmetry"] = periodic.overlap_asymmetry(
                    tables[frame].subset(np.arange(min(40, len(y)))),
                    [s for s in fs.specs if s.enabled])
            done[key] = out
            log(f"  {key}: R2 {out['r2']:+.4f}, MAE {out['mae']:.4f} eV/atom, "
                f"{out['seconds']:.0f} s")
            jdump(done, path)
    return done


# --------------------------------------------------------------------------- #
# Reading the model: the co-field inside the cell
# --------------------------------------------------------------------------- #

SITES = ("A", "B", "X")
#: the five sites of the perovskite cell in fractional coordinates: the A site at the corner, the
#: B site at the centre, and the three X sites at the face centres.  The set substitutes on the X
#: site as well as on A and B -- several members carry no oxygen at all -- so a site is named by
#: where it is in the cell and not by its element.
IDEAL = {"A": [(0.0, 0.0, 0.0)], "B": [(0.5, 0.5, 0.5)],
         "X": [(0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5)]}


def site_labels(mol) -> list:
    """Which site of the perovskite cell every atom sits on, by its fractional coordinates."""
    frac = np.linalg.solve(np.asarray(mol.cell, float).T,
                           np.asarray(mol.coords, float).T).T % 1.0
    targets = [(name, np.array(p)) for name, ps in IDEAL.items() for p in ps]
    out = []
    for f in frac:
        d = [float(np.min(np.linalg.norm(((f - p + 0.5) % 1.0) - 0.5, axis=-1)))
             for _name, p in targets]
        out.append(targets[int(np.argmin(d))][0])
    return out


def export_cubes(model, fs, table, fields=CUBE_FIELDS, spacing: float = 0.25) -> list:
    """The co-field over one cell as a Gaussian cube, with the atoms of a cell written into it.

    A cube file carries the grid *and* the atoms, so a viewer draws the co-field around the
    structure with no further work; the field is periodic, so one cell tiles to as much of the
    crystal as one wants to see. The atoms written are those of the first structure of the set,
    which stands at the ideal sites of the lattice like every other member.
    """
    from cmfqsar.cofields import Grid, cofield_volume, export_cube

    mol = table.molecules[0]
    a = float(np.asarray(mol.cell, float)[0, 0])
    n = int(round(a / float(spacing))) + 1
    grid = Grid(origin=np.zeros(3), spacing=a / (n - 1), shape=(n, n, n))
    names = [sp.name for sp in fs.specs if sp.enabled]
    out = []
    for name in fields:
        if name not in names:
            continue
        f = names.index(name)
        t0 = time.time()
        volume = cofield_volume(model, f, grid, include_scale=True)
        path = result(f"cofield_{name.replace(':', '_')}.cube")
        export_cube(path, grid, volume, mol,
                    title=f"CMF co-field of {TARGET}, field {name}, one cell")
        out.append({"field": name, "path": os.path.basename(path), "atoms": int(mol.n_atoms),
                    "structure": mol.name, "shape": list(grid.shape),
                    "spacing": grid.spacing, "seconds": time.time() - t0})
        log(f"  {name} as a cube over one cell: {grid.shape[0]}^3 points, "
            f"{time.time() - t0:.0f} s -> {os.path.basename(path)}")
    return out


def cofield(frame: str = "one lattice", alpha: float = 2.0, lam: float = 1e-3,
            limit: int = LIMIT, engine: str = "auto", sites: int = SITE_SAMPLE,
            cube: bool = True, cube_spacing: float = 0.25, force: bool = False) -> dict:
    """Fit one overlap model on the whole set and read it: per-site contributions and planes.

    The co-field of a periodic model carries one Gaussian per image atom of the training set --
    about 1.8 million for 1,500 cells -- so it is evaluated with the sparse reader of
    `cmfqsar.cofields`, and the per-site decomposition, which costs a pair sum against all of
    them for every cell, is averaged over a sample of `sites` cells rather than over all of them.
    """
    path = result("cofield.json")
    if os.path.isfile(path) and not force:
        log("cofield: already done")
        return jload(path)
    tables, y, fs, rec = load_set(limit)
    table = tables[frame]
    set_alpha(fs, alpha)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=True,
                     pbc_overlap=True, engine=engine)
    t0 = time.time()
    model.fit(table, y, cv_folds=0)
    pred = model.predict(table)
    log(f"the model on the whole set: R2 {r2_score(y, pred):+.4f} (fitted, not held out), "
        f"{time.time() - t0:.0f} s")

    # rule 3: one contribution per site of the cell, averaged over a sample by kind of site
    sums = {s: [] for s in SITES}
    rng = np.random.RandomState(0)
    sample = np.sort(rng.choice(table.n_mol, min(int(sites), table.n_mol), replace=False))
    t1 = time.time()
    for i in sample:
        mol = table.molecules[i]
        c = model.atom_contributions(mol)
        for site, value in zip(site_labels(mol), c):
            sums.setdefault(site, []).append(float(value))
    per_site = {s: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": int(len(v))}
                for s, v in sums.items() if len(v)}
    log(f"  the per-site decomposition of {len(sample)} cells: {time.time() - t1:.0f} s")
    for s, v in per_site.items():
        log(f"  site {s}: {v['mean']:+.4f} +- {v['sd']:.4f} eV/atom over {v['n']} sites")

    # the co-field in the two planes of the cell that carry the sites
    cell = np.asarray(table.molecules[0].cell, float)
    a = float(cell[0, 0])
    planes = {}
    n = 41
    axis = np.linspace(0.0, a, n)
    from cmfqsar.cofields import evaluate_gaussians

    basis = {}
    for f, spec in enumerate(fs.specs):
        if spec.enabled:
            basis[f] = (model.cofield_coefficients(f), float(spec.h) * float(model.scale[f]))
    for name, z in (("z = 0", 0.0), ("z = a/2", 0.5 * a)):
        pts = np.stack(np.meshgrid(axis, axis, [z], indexing="ij"), axis=-1).reshape(-1, 3)
        per_field = {}
        for f, ((centres, coeffs, widths), h) in basis.items():
            t2 = time.time()
            v = h * evaluate_gaussians(pts, centres, coeffs, widths)
            per_field[fs.specs[f].name] = v.reshape(n, n).tolist()
            log(f"  {fs.specs[f].name} on the plane {name}: {len(centres)} Gaussians, "
                f"{time.time() - t2:.0f} s")
        planes[name] = per_field
    # the cell the cube file carries, with one contribution per site: the same decomposition
    # as the table above, kept per atom so that it can be drawn on the structure
    ref = table.molecules[0]
    ref_c = model.atom_contributions(ref)
    reference = {"name": ref.name, "symbols": list(ref.symbols),
                 "coords": np.asarray(ref.coords, float).tolist(),
                 "sites": site_labels(ref), "contributions": np.asarray(ref_c, float).tolist(),
                 "bias": float(model.bias), "prediction": float(model.predict(
                     table.subset([0]))[0])}
    log(f"  the reference cell {ref.name}: "
        + ", ".join(f"{st} {v:+.2f}" for st, v in zip(reference["sites"], ref_c)))
    cubes = export_cubes(model, fs, table, spacing=cube_spacing) if cube else []
    out = {"cubes": cubes, "reference": reference,
           "frame": frame, "alpha": alpha, "lam": lam, "n": int(table.n_mol),
           "sites_sampled": int(len(sample)),
           "r2_fitted": float(r2_score(y, pred)), "per_site": per_site,
           "cell_a": a, "grid": axis.tolist(), "planes": planes,
           "fields": [s.name for s in fs.specs if s.enabled]}
    jdump(out, path)
    return out


def draw_cell(ax, plane: str, a: float) -> None:
    """The cell over the co-field: its edges, its sites and the bonds that lie in this plane.

    Drawn from the lattice rather than from one member of the set, because the co-field belongs
    to the model and every member stands at these sites; the images at the edges are drawn as
    well, since a periodic field repeats and a reader should see it repeat.
    """
    ax.plot([0, a, a, 0, 0], [0, 0, a, a, 0], "-", color="0.35", lw=0.8)
    if plane == "z = 0":
        ax.plot([0, 0, a, a], [0, a, 0, a], "o", color="k", ms=6)
        ax.plot([0.5 * a], [0.5 * a], "s", color="k", ms=5, mfc="none")
        ax.annotate("A", (0.04 * a, 0.04 * a), fontsize=8, color="k")
        ax.annotate("X", (0.54 * a, 0.52 * a), fontsize=8, color="k")
    else:
        xs = [0.0, 0.5 * a, a, 0.5 * a]
        ys = [0.5 * a, 0.0, 0.5 * a, a]
        for x, y in zip(xs, ys):                       # the equatorial bonds of the octahedron
            ax.plot([0.5 * a, x], [0.5 * a, y], "-", color="k", lw=0.9, alpha=0.65)
        ax.plot(xs, ys, "s", color="k", ms=5, mfc="none")
        ax.plot([0.5 * a], [0.5 * a], "^", color="k", ms=7)
        ax.annotate("B", (0.54 * a, 0.52 * a), fontsize=8, color="k")
        ax.annotate("X", (0.54 * a, 0.03 * a), fontsize=8, color="k")
    ax.set_xlim(-0.02 * a, 1.02 * a)
    ax.set_ylim(-0.02 * a, 1.02 * a)


#: fraction of the largest value of the volume at which the isosurfaces are drawn
ISO_LEVEL = 0.45
#: how the three-dimensional figure is viewed (elevation, azimuth), in degrees
ISO_VIEW = (20.0, -62.0)


def site_panel(ax, rec: dict, a: float) -> None:
    """The field *at* the atoms: one sphere per site, from decomposition rule 3.

    The area is the size of the contribution and the colour its sign, which is how the
    atom-contribution figure of a molecule is drawn; the means of the whole set are in the
    table, and this is one cell of it, the cell the cube file carries.
    """
    ref = rec.get("reference")
    if not ref:
        ax.set_axis_off()
        return
    xyz = np.asarray(ref["coords"], float)
    c = np.asarray(ref["contributions"], float)
    shifts = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) * a
    biggest = max(float(np.abs(c).max()), 1e-12)
    for s0, s1 in ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
                   (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)):
        p0 = np.array([(s0 >> k) & 1 for k in range(3)], float) * a
        p1 = np.array([(s1 >> k) & 1 for k in range(3)], float) * a
        ax.plot(*zip(p0, p1), color="0.25", lw=0.9)
    centre = np.array([0.5, 0.5, 0.5]) * a
    for p, value, sym, site in zip(xyz, c, ref["symbols"], ref["sites"]):
        for t in shifts:
            q = p + t
            if not (np.all(q <= 1.02 * a) and np.all(q >= -0.02 * a)):
                continue
            ax.scatter(*q, s=40 + 460 * abs(value) / biggest,
                       c="#c0392b" if value > 0 else "#2471a3",
                       edgecolors="0.2", linewidths=0.6, depthshade=False, alpha=0.9)
            d = float(np.linalg.norm(q - centre))
            if 0.1 < d < 0.6 * a:
                ax.plot(*zip(centre, q), color="0.3", lw=0.9, zorder=0)
        ax.text(p[0], p[1], p[2] + 0.3, f"{sym} ({site}) {value:+.1f}", fontsize=8)
    ax.set_xlim(0, a)
    ax.set_ylim(0, a)
    ax.set_zlim(0, a)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ISO_VIEW[0], azim=ISO_VIEW[1])
    ax.set_axis_off()
    ax.set_title("the field at the atoms: rule 3, one contribution per site\n"
                 f"{ref['name']}, in eV per atom; they sum with the bias to the prediction",
                 fontsize=10)


def isosurface_figure(cube: str = "", out: str = "", level: float = ISO_LEVEL) -> str:
    """The co-field as two isosurfaces around the cell, read back from the cube file.

    Nothing is recomputed: the cube is the volume the co-field stage already evaluated, and this
    reads it back exactly as an outside viewer would, which is also a check that the file says
    what it is meant to say.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from cmfqsar.cdft import read_cube
    from cmfqsar.cofields import Grid, isosurface

    rec = jload(result("cofield.json"))
    cube = cube or result(rec["cubes"][0]["path"])
    c = read_cube(cube)
    grid = Grid(origin=np.asarray(c.origin, float), spacing=float(c.vectors[0][0]),
                shape=c.values.shape)
    volume = np.asarray(c.values, float)
    a = float(rec["cell_a"])
    peak = float(np.abs(volume).max())
    log(f"{os.path.basename(cube)}: {volume.shape} points, |value| up to {peak:.3g}, "
        f"{len(c.numbers)} atoms")

    fig = plt.figure(figsize=(12.4, 6.2))
    ax = fig.add_subplot(121, projection="3d")
    for sign, colour in ((+1.0, "#c0392b"), (-1.0, "#2471a3")):
        got = isosurface(volume, sign * level * peak, grid)
        if got is None:
            continue
        verts, faces, _normals = got
        mesh = Poly3DCollection(verts[faces], alpha=0.33, linewidths=0)
        mesh.set_facecolor(colour)
        ax.add_collection3d(mesh)
    # the cell, drawn as twelve edges
    for s0, s1 in ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
                   (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)):
        p0 = np.array([(s0 >> k) & 1 for k in range(3)], float) * a
        p1 = np.array([(s1 >> k) & 1 for k in range(3)], float) * a
        ax.plot(*zip(p0, p1), color="0.25", lw=0.9)
    # the atoms of the cell the cube carries, with the periodic images that fall inside it:
    # a site at a corner or a face belongs to the neighbouring cells as well, and the
    # octahedron around the B site has three of its six arms in them
    from rdkit.Chem import GetPeriodicTable

    table = GetPeriodicTable()
    numbers = np.asarray(c.numbers, int)
    shifts = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) * a
    drawn = []
    for z, p in zip(numbers, np.asarray(c.positions, float)):
        for t in shifts:
            q = p + t
            if np.all(q <= 1.02 * a) and np.all(q >= -0.02 * a):
                drawn.append((int(z), q))
    heavy = max(int(z) for z in numbers)
    for z, q in drawn:
        ax.scatter(*q, s=150 if z == heavy or z > 20 else 55, c="0.15", depthshade=False)
    # one label per element, on the instance nearest the viewer's corner
    for z in sorted(set(numbers), reverse=True):
        q = max((q for zz, q in drawn if zz == z), key=lambda v: v[0] - v[1] + v[2])
        ax.text(q[0], q[1], q[2] + 0.25, table.GetElementSymbol(int(z)), fontsize=9)
    centre = np.array([0.5, 0.5, 0.5]) * a
    for _z, q in drawn:                       # the six arms of the BX6 octahedron
        d = float(np.linalg.norm(q - centre))
        if 0.1 < d < 0.6 * a:
            ax.plot(*zip(centre, q), color="0.15", lw=1.1)
    ax.set_xlim(0, a)
    ax.set_ylim(0, a)
    ax.set_zlim(0, a)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ISO_VIEW[0], azim=ISO_VIEW[1])
    ax.set_axis_off()
    ax.set_title(f"the field around the atoms: steric co-field, isosurfaces at "
                 f"±{level:.0%} of the peak", fontsize=10)
    site_panel(fig.add_subplot(122, projection="3d"), rec, a)
    fig.suptitle("The co-field of the formation energy in the perovskite cell: red raises the "
                 "predicted energy, blue lowers it", fontsize=11, y=0.99)
    # the panel titles of a three-dimensional axis sit high, so the figure keeps room for both
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out = out or result("cofield_3d.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    log(f"written to {out}")
    return out


def figure(out: str = "") -> str:
    """The co-field in the two planes of the cell, with the cell drawn over it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = jload(result("cofield.json"))
    fields = rec["fields"]
    a = rec["cell_a"]
    fig, axes = plt.subplots(2, len(fields), figsize=(3.1 * len(fields), 6.4), squeeze=False)
    for row, plane in enumerate(("z = 0", "z = a/2")):
        for col, name in enumerate(fields):
            v = np.asarray(rec["planes"][plane][name], float)
            ax = axes[row][col]
            m = float(np.max(np.abs(v))) or 1.0
            im = ax.imshow(v.T, origin="lower", extent=[0, a, 0, a], cmap="RdBu_r",
                           vmin=-m, vmax=m, aspect="equal")
            draw_cell(ax, plane, a)
            ax.set_title(f"{name}, {plane}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # the title says what the panels are; which marker is which site belongs in the caption,
    # where there is room for it, and a long title is clipped at the width of the figure
    fig.suptitle("The co-field of the formation energy inside the perovskite cell, with the "
                 "cell over it: red raises the energy, blue lowers it", fontsize=11)
    fig.tight_layout()
    out = out or result("cofield.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"written to {out}")
    return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

def report(out: str = "") -> str:
    prep = jload(result("prepared.json")) if os.path.isfile(result("prepared.json")) else {}
    fits = jload(result("fit.json")) if os.path.isfile(result("fit.json")) else {}
    cof = jload(result("cofield.json")) if os.path.isfile(result("cofield.json")) else None
    lines = ["# The periodic overlap kernel against the periodic autocorrelation (item M4)", ""]
    if prep:
        sg = prep["lattice_spread_as_given"]
        lines += [f"`matbench_perovskites`: {prep['n']} ABO3 cells of "
                  f"{prep['atoms_per_cell']:.0f} atoms, formation energy in eV/atom "
                  f"(mean {prep['y_mean']:+.3f}, spread {prep['y_sd']:.3f}). Fields: "
                  f"{', '.join('`' + f + '`' for f in prep['fields'])}, charges from "
                  f"{prep['charges']} differences.", "",
                  f"**How far the set is from one lattice.** The relaxed cells have a mean "
                  f"volume of {sg['volume_mean']:.1f} A^3, and the largest deviation from the "
                  f"mean is {100 * sg['volume_spread']:.0f} % in volume and "
                  f"{100 * sg['length_spread']:.0f} % in a lattice vector. The overlap integral "
                  f"compares the two cells point by point, so that spread is a frame error; the "
                  f"one-lattice frame removes it, and scales the bond lengths instead.", ""]
    if fits:
        lines += ["## What each kernel predicts", "",
                  "Five Matbench folds, the setting and the ridge chosen by ten-fold "
                  "cross-validation inside each training part. Lower MAE is better.", "",
                  "| kernel | frame | R2 | RMSE, eV/atom | MAE, eV/atom | settings chosen |",
                  "|---|---|---:|---:|---:|---|"]
        for key in sorted(fits):
            r = fits[key]
            chosen = sorted({f["setting"] for f in r["folds"]})
            lines.append(f"| {r['kernel']} | {r['frame']} | {r['r2']:+.3f} | {r['rmse']:.4f} | "
                         f"{r['mae']:.4f} | {', '.join(chosen)} |")
        lines.append("")
        asym = [(k, v.get("asymmetry")) for k, v in fits.items() if v.get("asymmetry") is not None]
        if asym:
            lines += ["**The two orders of the integral.** The periodic overlap integrates over "
                      "the cell of one structure and the images of the other, so with cells of "
                      "different size the two orders differ and the kernel is the average of "
                      "them. The largest disagreement before the average: "
                      + "; ".join(f"{k}, {v:.1%}" for k, v in sorted(asym)) + ".", ""]
    if cof:
        lines += ["## What the co-field says", "",
                  f"One overlap model on all {cof['n']} cells (frame: {cof['frame']}, "
                  f"attenuation {cof['alpha']:g}, ridge {cof['lam']:g}; R2 {cof['r2_fitted']:+.3f} "
                  f"on its own training set, which is not a validation). Decomposition rule 3 "
                  f"gives one contribution per site of every cell; the mean over a sample of "
                  f"{cof.get('sites_sampled', cof['n'])} cells, by kind of site:", "",
                  "| site | mean contribution, eV/atom | spread | sites |", "|---|---:|---:|---:|"]
        for s in SITES:
            v = cof["per_site"].get(s)
            if v:
                lines.append(f"| {s} | {v['mean']:+.4f} | {v['sd']:.4f} | {v['n']} |")
        lines += ["", "The co-field itself is drawn in `cofield.png`, with the cell over it, in "
                  "the plane of the A site and the X sites (z = 0) and in the plane of the B "
                  "site and the equatorial X sites (z = a/2).", ""]
        if cof.get("cubes"):
            lines += ["It is also written as a Gaussian cube over one cell, which carries the "
                      "atoms beside the values, so a viewer draws the periodic co-field around "
                      "the structure: "
                      + "; ".join(f"`{c['path']}` ({c['field']}, {c['shape'][0]}^3 points at "
                                  f"{c['spacing']:.2f} A, the atoms of {c['structure']})"
                                  for c in cof["cubes"]) + ".", "",
                      "`cofield_3d.png` draws the isosurfaces of that cube around the cell, "
                      "read back from the file as an outside viewer would read it.", ""]
    text = "\n".join(lines) + "\n"
    out = out or result("table.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)
    print(f"written to {out}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("prepare", "fit", "cofield", "figure", "isosurface"),
                    default=None)
    ap.add_argument("--kernels", nargs="+", default=list(KERNELS))
    ap.add_argument("--frames", nargs="+", default=list(FRAMES))
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument("--alpha", type=float, default=2.0, help="attenuation of the co-field model")
    ap.add_argument("--engine", default="auto", help="kernel engine: numpy, numba or auto")
    ap.add_argument("--sites", type=int, default=SITE_SAMPLE,
                    help="cells whose sites are decomposed for the per-site table")
    ap.add_argument("--no-cube", action="store_true",
                    help="skip the cube files of the co-field over one cell")
    ap.add_argument("--cube-spacing", type=float, default=0.25,
                    help="grid spacing of those cube files, in angstroms")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return 0
    stages = ((args.stage,) if args.stage
              else ("prepare", "fit", "cofield", "figure", "isosurface"))
    for stage in stages:
        log(f"=== {stage}")
        if stage == "prepare":
            prepare(limit=args.limit, force=args.force)
        elif stage == "fit":
            fit(kinds=args.kernels, frames=args.frames, limit=args.limit, engine=args.engine,
                force=args.force)
        elif stage == "cofield":
            cofield(alpha=args.alpha, limit=args.limit, engine=args.engine, sites=args.sites,
                    cube=not args.no_cube, cube_spacing=args.cube_spacing, force=args.force)
        elif stage == "figure":
            figure()
        elif stage == "isosurface":
            isosurface_figure()
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
