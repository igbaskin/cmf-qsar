# Which atom does the enzyme attack? (plan item MC8)

Blocking a metabolic position is a question about one atom and its surroundings, which is what plan items LP1 and LP2 built the per-atom machinery for. The set is the benchmark of Zaretzki, Matlock and Swamidass (2013), the sites merged over the cytochrome P450 isoforms, prepared in three dimensions here. The measure is the one the field uses: in what share of the molecules is a true site among the two atoms the model ranks highest.

14488 heavy atoms in 636 molecules, 1301 of them sites (9.0% of the atoms), 10 folds holding out whole molecules, the ridge searched the same way for every row.

| descriptor | top-1 | top-2 | area under the curve | top-2 against the 2D baseline |
| --- | ---: | ---: | ---: | ---: |
| count over bonds (2D) | 53.6% | 70.9% | 0.859 | -- |
| count over distance (3D) | 57.5% | 73.9% | 0.864 | +3.0% [+0.3%, +5.5%] |
| local field, rho 0.15 | 54.4% | 70.1% | 0.843 | -0.8% [-4.2%, +2.7%] |
| local field, rho 0.25 | 55.0% | 71.1% | 0.853 | +0.2% [-3.3%, +3.8%] |
| local field, rho 0.5 | 55.2% | 69.7% | 0.854 | -1.3% [-4.9%, +2.5%] |
| local field, rho 1 | 55.3% | 69.3% | 0.844 | -1.6% [-5.2%, +2.2%] |
| local field, rho 2 | 53.0% | 66.5% | 0.837 | -4.4% [-8.0%, -0.8%] |
| count and field, rho 0.25 | 60.8% | 76.9% | 0.875 | +6.0% [+3.3%, +8.8%] |
| centre type and field, rho 0.25 | 59.9% | 74.8% | 0.860 | +3.9% [+0.6%, +7.2%] |

**The field adds next to the count, not instead of it.** On its own the local field is no better than the two-dimensional baseline: 71.1% against 70.9%, +0.2% [-3.3%, +3.8%]. Counting the same atom types over distance instead of over bonds is better, 73.9%, +3.0% [+0.3%, +5.5%].

The two together reach 76.9%. That is better than the count alone, +3.0% [+0.5%, +5.7%], and better than the two-dimensional baseline, +6.0% [+3.3%, +8.8%], which is what plan item MC8 asked for. The two descriptions therefore hold different things: the count says which atom types are at which distance, the field says how the charge and the shape are arranged there, and neither carries the other.

Most of what the count contributes to that mixture is the name of the atom being asked about. Replacing the whole count by its first block -- the centre's own atom type, one column per type -- reaches 74.8% where the whole count reaches 76.9%. The autocorrelation of an environment never names its centre: it is built from distances between pairs of weighted atoms, and the centre is only the atom the envelope weights most. The ease of oxidation depends first of all on what the atom is, so the field starts without the one variable that matters most and has to recover it indirectly.

The width is chosen rather than taken from the edge of the range: the field rows rise to rho 0.25 and fall on both sides of it. A wide envelope makes every atom of a molecule look alike, because the features are measured from the molecule as a whole and not from the centre; a narrow one leaves too little of the molecule inside the envelope to tell two carbons apart.

A molecule here has 23 heavy atoms on average, so two atoms taken at random would find a site in about 18 per cent of them. Every row is far above that. Most of what any of these models knows is which atom types are attacked at all, and what the rows separate is how much is left once that is known.

