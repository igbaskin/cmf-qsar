# Three solid electrolytes with their Li hops

The files in this folder are the hosts of the Li interface study (`work/claude/li-graded-interface`,
October 2026) in the form the hop model of plan item BT7 takes: a DFT-relaxed supercell of each
with its Bader charges as per-atom arrays, written by `cmfqsar.vasp.structure_with_arrays` from
the `CONTCAR`, `ACF.dat` and `OUTCAR` of the pristine run. They are the sample of the second part
of `benchmarks/run_hops.py` and the example of `cmf hops`.

| file | compound | atoms | cell | source run |
| --- | --- | ---: | --- | --- |
| `phosphate.extxyz` | beta-Li3PO4 (Pmn2_1) | 128 | 2 x 2 x 2 supercell | `calcs/phosphate/pristine` |
| `borophosphate.extxyz` | Li3BP2O8 (P-1) | 112 | supercell of the triclinic cell | `calcs/borophosphate/pristine` |
| `borate.extxyz` | alpha-LiBO2 (P2_1/c) | 128 | 2 x 2 x 2 supercell | `calcs/borate/pristine` |

**What the arrays are.** `bader_charges` is the net Bader charge of every atom: the valence
electrons of the PAW potential (Li 3, B 3, P 5, O 6) minus the electrons the Bader analysis gave
the atom. The same numbers are the initial charges of the structure, so that ASE keeps them
through the extended-XYZ format, and `cmf hops --charges file` uses them as the weights of the
electrostatic field. `bader_electrons` is the raw Bader column. The runs were not spin-polarised,
so there is no magnetic moment. The structures were relaxed with PBE at the Gamma point
(ENCUT 520 eV), as the study's README records.

**The labels.**

* `hops_bvse.csv` is what `cmf hops ... --bvse --out` writes: every Li hop between 1.5 and
  4.5 A in the three cells (1040 of them, 48 distinct by symmetry) with the forward and reverse
  bond-valence site-energy barrier of the softBV model (`bvlain`, landscape at 0.25 A, path box
  at 0.2 A), the site energies and the saddle behind them, the point of least free space on the
  straight path with the atoms that form it, the representative each hop was reduced to, and the
  out-of-fold prediction of the run that wrote the file.
* `dft_barriers.csv` holds the two hops the study computed with VASP as plain elastic bands
  (four interior images, no climbing image, so the barrier is read at the highest image and is a
  slight lower bound): in Li3PO4 the Li2-type ion of index 34 into the relaxed Li1 vacancy at
  index 0, and in Li3BP2O8 the Li3-type ion of index 18 into the relaxed Li2 vacancy at index 8,
  which sits in the next cell along a. These are vacancy hops with full relaxation, so they are
  not the quantity the rigid bond-valence landscape gives; `benchmarks/run_hops.py` lists the two
  side by side without comparing them in value.

```bash
cmf hops examples/hops/phosphate.extxyz examples/hops/borophosphate.extxyz examples/hops/borate.extxyz \
    --charges file --bvse --reference --invariant --profile --percolate --out hops.csv
cmf hops examples/hops/phosphate.extxyz --charges file --labels examples/hops/dft_barriers.csv --list
```
