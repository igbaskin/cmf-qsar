# Lithium solvation shells from bulk electrolyte simulations

For plan item BT5, which asks whether an ensemble of solvation shells sampled from a liquid
describes an electrolyte better than the single optimised complex of item BT4.

## What is here

| File | What it holds |
|---|---|
| `formulations.csv` | one row per electrolyte: the salt, the solvent weight fractions, the molality, and the two measured targets |
| `shells/<id>.sdf` | the representative solvation shells of that electrolyte, each with its population weight |
| `shells.json` | what each simulation did and how well it is converged |

`formulations.csv` has one row per formulation. `log10_k_298` is the base-ten logarithm of the
ionic conductivity in mS/cm at 298.15 K and `ea_298` is the activation energy in kJ/mol at the
same temperature. Both are read off a fit of log10 k = a + b/T + c/T^2 to that formulation's own
measurements, which is an interpolation: a formulation is listed only when its measured
temperatures span at least 20 K and contain 298.15 K. `fit_rms` is the residual of that fit in
log10 k, `n_points` the number of measurements behind it, and `dois` the papers they come from.

Each file in `shells/` holds one record per representative shell, under the formulation's name.
The properties `shell_weight`, `shell_species` and `shell_sizes` are what
`cmfqsar.shells.ensemble_from_molecules` needs to rebuild the mean field, so a model is built
from these files directly:

```python
from rdkit import Chem
from cmfqsar.fields import FieldSet
from cmfqsar.shells import ensemble_from_molecules

fs = FieldSet.create(alpha=0.3, charge_model="gasteiger")
members = [m for m in Chem.SDMolSupplier("shells/LiPF6_DMC100_m1p000.sdf", removeHs=False)]
fields = ensemble_from_molecules(members, fs, name="DMC at 1 mol/kg")
```

The ion is always atom 0 of every record, and every record is in the cation frame of item BT4.

## Where the numbers come from

The conductivities are measurements, taken from CALiSol-23:

> de Blasio P. V. F., Elsborg J. T., Vegge T., Flores E., Bhowmik A. "CALiSol-23: Experimental
> electrolyte conductivity data for various Li-salts and solvent combinations", *Scientific Data*
> **11** (2024) 1-11. <https://doi.org/10.1038/s41597-024-03575-8>
> Data: <https://doi.org/10.11583/DTU.24559960>, licence CC BY 4.0.

The shells are computed here, not measured: one box per formulation, packed with packmol, the
solvents and the anion typed with GAFF-2.11 and charged with the OpenFF graph network, the
lithium ion with the Amber ion parameters, and run with OpenMM under pressure coupling. The
shells are then cut by `cmfqsar.shells`. `examples/make_electrolyte_shells.py` is the whole
recipe and `benchmarks/run_shells.py` the measurement.

## What these shells are not

A model of the liquid, not a measurement of one. Three limits are worth stating:

* **No polarisation.** A fixed-charge force field is known to bind a small cation to a carbonate
  too strongly, so the coordination numbers here are more likely too high than too low.
* **A small box and a short run.** A few hundred solvent molecules and one nanosecond.
  `shells.json` reports the `block_error`, the standard error of the mean field over blocks of
  frames as a share of the field's own size, so that the sampling can be judged rather than
  assumed.
* **The density is a check, not an input.** `density` is what the barostat settled at and
  `start_density` the ideal-mixing value the box was built from. A large difference is a result
  about the force field.

## Licence of this copy

The conductivities come from a CC BY 4.0 dataset and are redistributed here under the same
licence, with the citation above. The shell files are produced by this repository's own code.
Under the three-word scheme of `docs/public_3dqsar_datasets.md` that makes these files **free**:
the licence is explicit, so the copy may travel with the program, and the one obligation it
carries is the citation above.
