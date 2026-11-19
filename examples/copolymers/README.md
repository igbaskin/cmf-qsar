# Copolymers: electron affinity and ionization potential

The computed electron affinity and ionization potential of 1,575 binary copolymers, each given
by its two repeat units, their fractions and its chain architecture. The table is made by
`benchmarks/prepare_copolymers.py` and read by `benchmarks/run_copolymers.py` (plan item PL4).

| file | what it holds |
| --- | --- |
| `vipea.csv` | one copolymer per row |
| `LICENSE_polymer-chemprop-data.txt` | the MIT licence of the source |
| `provenance.json` | the source file, its commit and SHA-256, the dibromides kept and refused, and the number of copolymers of each kind |

**Columns.** `id` names the copolymer: the source's pair of monomer numbers, the architecture and
the composition, for example `0_14_random_2A_6B`. `pair` is the pair of monomer numbers, `a` the
number of the diboronic acid (0 to 8) and `b` that of the dibromide. `architecture` is
`alternating`, `random` or `block`. `n_a` and `n_b` are the numbers of A and B units in the
octamers the values were computed on (4 and 4, 2 and 6, or 6 and 2), and `frac_a` and `frac_b`
the fractions. `ea_ev` and `ip_ev` are the electron affinity and the ionization potential in
electronvolts, as the source gives them before its calibration against the standard hydrogen
electrode. `unit_a` and `unit_b` are the repeat units written as SMILES with two attachment
points, `*`; `monomer_a` and `monomer_b` are the monomers as the source gives them.

**The repeat units.** Monomer A is an aryl diboronic acid and monomer B an aryl dibromide. The
octamers were made by joining these groups, so the two B(OH)2 groups of A and the two bromine
atoms on aromatic carbon of B are the attachment points. For example, the diboronic acid
`OB(O)c1cc(F)c(B(O)O)cc1F` gives the unit `*c1cc(F)c(*)cc1F`.

**How the values were made.** As the source paper and its script `make-polymers.py` describe
them: octamers ended by hydrogen; for a random copolymer every distinct order of the units, a
sequence and its reverse counted once, up to 32 orders; for a block copolymer AAAABBBB, AABBBBBB
or AAAAAABB; for the alternating copolymer ABABABAB. The ionization potential and the electron
affinity were computed with the semi-empirical tight-binding program xTB, Boltzmann-averaged over
8 conformers at 298.15 K and averaged over the sequences of the copolymer. They are computed, not
measured.

**Source.** M. Aldeghi and C. W. Coley, "A graph representation of molecular ensembles for
polymer property prediction", *Chemical Science* 13, 10486 (2022), doi:10.1039/D2SC02839E. The
file is `datasets/vipea/dataset.csv` of github.com/coleygroup/polymer-chemprop-data at commit
2af92df. Licence: MIT, reproduced in `LICENSE_polymer-chemprop-data.txt`.

**The subset.** The source has 42,966 copolymers of 9 diboronic acids and 682 dibromides. Two
dibromides carry a third bromine on aromatic carbon, so their attachment points are not defined,
and they are left out. Of the other 680, 25 were drawn with a fixed seed (0), and every copolymer
of each with the 9 diboronic acids is kept: 225 pairs of monomers, each as one alternating, three
random and three block copolymers. A kernel model on the whole set would cost days.
