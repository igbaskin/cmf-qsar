# Point mutants of four protein domains, with their folding stability

This folder holds the labels of plan item PP1: single amino-acid substitutions in four small
natural protein domains, with the change of folding free energy each one causes.

## Where the data come from

Tsuboyama, K., Dauparas, J., Chen, J., Laine, E., Mohseni Behbahani, Y., Weinstein, J. J.,
Mangan, N. M., Ovchinnikov, S. & Rocklin, G. J. *Mega-scale experimental analysis of protein
folding stability in biology and design.* Nature **620**, 434-444 (2023).

The data are deposited on Zenodo as record 7992926 under the **Creative Commons Attribution 4.0**
licence, which allows redistribution of a derived copy with attribution and a statement of the
changes. Two files of that record were downloaded on 11 November 2026:

* `Processed_K50_dG_datasets.zip`, from which
  `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (698 MB) holds the inferred stabilities;
* `AlphaFold_model_PDBs.zip`, which holds one model of every wild-type domain.

Neither file is in this repository. `examples/make_mutant_examples.py` reads them from
`benchmarks/cache/megascale/`, which git ignores, and writes the small tables kept here.

## What is kept, and what was left out

One folder per domain, named by the identifier the study uses, which is a Protein Data Bank code:

| Folder | Positions measured | Substitutions kept | Wild-type stability, kcal/mol |
|---|---|---|---|
| `1O6X` | 72 | 1356 | 4.17 |
| `1UFM` | 72 | 1353 | 3.28 |
| `2K5N` | 67 | 1236 | 4.09 |
| `3ZGK` | 69 | 1301 | 3.56 |

Each folder holds:

* `<code>.pdb`, the wild-type model of the study, unchanged. Its residues are numbered from 1
  along the sequence that was measured, and that numbering is what a substitution label refers to.
* `mutants.csv`, one row per substitution: `label` (for example `V1Q`, meaning valine at
  position 1 replaced by glutamine), `position`, `wild`, `new`, `dG_ML`, `ddG_ML` and `dG_95CI`.
* `wild_type.txt`, the sequence, the wild-type `dG_ML` and the fold cluster of the study.

`ddG_ML` is the target: the unfolding free energy of the variant minus that of the wild type, in
kcal/mol. A negative value is a destabilising substitution.

Three filters were applied, and nothing else was changed:

1. **Single substitutions only.** A label that is not one wild-type letter, a position and one
   new letter is left out, which removes insertions, deletions and multiple substitutions.
2. **Inside the range the assay resolves.** A variant whose `dG_ML` lies outside -1 to 5 kcal/mol
   is left out. The proteolysis assay cannot separate stabilities beyond those bounds, and the
   study says so.
3. **One row per label.** The table holds repeated measurements of some variants; the first is
   kept.

The four domains were chosen before anything was measured, by three rules: a natural protein
rather than a designed one (the identifier is a PDB code), a different fold cluster for each, and
a wild-type stability above 3 kcal/mol, so that a substitution costing a few kcal/mol still falls
inside the resolved range. Among the domains that pass those rules, the four with the most
variants were taken.

## The structures of the variants are not here

A variant's structure is built from the wild-type model, not stored:

```bash
python examples/make_mutant_examples.py --build
```

This writes `benchmarks/cache/megascale/built/<code>/<label>.npz`, which git ignores. Each
structure is built by PDBFixer and relaxed with the Amber ff14SB force field through OpenMM, as
`cmfqsar/mutants.py` describes: the new side chain is started from six turns of its first
torsion, the mutated residue and the side chains of its neighbours within 6 Å are free, every
other atom is fixed, and the lowest-energy result that is a sound structure is kept. One
structure takes about twenty seconds on this machine, so a whole domain takes several hours in
one process; `--shard I --shards N` splits the work over several processes.

Two kinds of failure are reported rather than hidden:

* a starting turn whose minimisation ends with two atoms on top of each other is thrown away
  (`Structure.collapsed` counts them), because the Amber hydrogens on nitrogen and oxygen carry
  no repulsion radius and such a pair falls together with no lower bound on its energy;
* a variant whose structure cannot be read back as a chemically sound molecule is listed in
  `failed_<shard>.txt` and left out of the measurement. Four of the 1356 variants of `1O6X` were
  lost this way before the collapse check existed; none after it.
