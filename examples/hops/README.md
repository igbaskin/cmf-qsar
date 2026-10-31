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

## `disordered/`: thirteen conductors with partly occupied sites (plan item M2)

The CIFs in `disordered/` come from the Crystallography Open Database (COD,
<https://www.crystallography.net/cod/>, public domain under CC0), one file per entry, named by
the COD number. They are the structure types a solid-electrolyte study is about, with the
disorder their refinements report: Li sites partly occupied, split Li positions, and framework
sites shared by two species. `cmfqsar.periodic.read_disordered` reads them with their
occupancies, the fields see every site at its occupancy (the exact mean field of the
configurational ensemble), and `cmf hops` finds the hops between every site the Li ion can hold.

| file | compound | what is disordered |
| --- | --- | --- |
| `cod_1545083.cif` | Li6.5La3Zr1.5Ta0.5O12, cubic garnet | two Li sites at 0.38 and 0.62, Zr/Ta shared |
| `cod_1552152.cif` | Li6.2La3Zr1.2Ta0.8O12, cubic garnet | Li sites at 0.28 and 0.96, Zr/Ta shared |
| `cod_1545085.cif` | Li6.5La3Zr1.5Ta0.5O12, tetragonal garnet | ordered (the control) |
| `cod_1525568.cif` | La0.5Li0.5TiO3, rhombohedral perovskite | Li site at 0.16 |
| `cod_1000437.cif` | La0.56Li0.33TiO3, tetragonal perovskite | La/Li shared A sites |
| `cod_7215190.cif` | Li10GeP2S12 (LGPS) | four Li sites at 0.47 to 0.89, Ge/P shared |
| `cod_7222155.cif` | LiTi2(PO4)3, NASICON | ordered (the control) |
| `cod_7222156.cif` | Li3.18Ti2(PO4)3, lithiated NASICON | Li sites at 0.34 and 0.72 |
| `cod_4002499.cif` | Li2.92N | Li site at 0.96 |
| `cod_1524557.cif` | Li13.7Zn1.2Ge4O16, LISICON | Li sites at 0.16 to 0.98, Li/Zn shared |
| `cod_9011044.cif` | Li3PO4 (Pmnb) | ordered (the control; the host above as a small cell) |
| `cod_1570310.cif` | beta-Li3PS4 | Li sites at 0.3 and 0.7 |
| `cod_1529523.cif` | Li3.16Zn0.24GeO4 | Li sites at 0.06 to 0.92, Li/Zn shared |

COD entry 7206766 (undoped cubic Li7La3Zr2O12) was left out: its La and Li1 positions coincide
under the symmetry it declares, which both CIF readers reject.

`hops_bvse.csv` is what `cmf hops disordered/cod_*.cif --bvse --out` wrote: every hop with its
occupancies, the bond-valence barrier of the ordered framework (the sites the Li ion does not
hold, each with its majority species), the bottleneck, the representative and the out-of-fold
prediction of that run. The third part of `benchmarks/run_hops.py` reads it.

```bash
cmf hops examples/hops/disordered/cod_7215190.cif --list
cmf hops examples/hops/disordered/cod_*.cif --bvse --reference --percolate --out hops.csv
```
