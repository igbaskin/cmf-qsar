# A co-field around one atom (plan item LP3)

The sample is an aromatic carbon of a six-membered ring and its surroundings, posed by a rule: the carbon at the origin, the bond that leaves the ring on +x, the ring in the xy plane, both mirror images kept and averaged. In that frame the overlap kernel and the co-field of the program apply to an environment as they apply to a molecule. 1225 carbons in 170 molecules, shifts from 89.6 to 181.1 ppm (standard deviation 13.3); 10 folds holding out whole molecules.

## Does the pose cost anything?

| descriptor | envelope width rho | q2 | error (ppm) | ridge | atoms per sample | difference in q2 against the counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atom-centred counts (18 atom types in 6 shells) | -- | +0.768 | 6.383 | 0.01 | -- | -- |
| posed overlap | 0.05 | +0.872 | 4.745 | 0.1 | 58.8 | +0.104 [+0.070, +0.144] |
| invariant local | 0.05 | +0.116 | 12.461 | 0.1 | -- | -0.652 [-0.695, -0.594] |
| posed overlap | 0.1 | +0.887 | 4.448 | 0.01 | 56.8 | +0.119 [+0.086, +0.158] |
| invariant local | 0.1 | +0.366 | 10.554 | 0.1 | -- | -0.402 [-0.457, -0.344] |
| posed overlap | 0.25 | +0.899 | 4.217 | 0.01 | 44.9 | +0.131 [+0.098, +0.170] |
| invariant local | 0.25 | +0.803 | 5.882 | 0.001 | -- | +0.035 [+0.003, +0.068] |
| posed overlap | 0.5 | +0.889 | 4.425 | 0.0001 | 32.8 | +0.121 [+0.083, +0.162] |
| invariant local | 0.5 | +0.847 | 5.187 | 0.01 | -- | +0.079 [+0.048, +0.116] |
| posed overlap | 1 | +0.861 | 4.935 | 1e-06 | 21.3 | +0.093 [+0.046, +0.143] |
| invariant local | 1 | +0.836 | 5.361 | 0.01 | -- | +0.068 [+0.036, +0.105] |

The posed kernel against the invariant one, at the same envelope width:

* rho 0.05 (width 4.5 A): +0.756 [+0.718, +0.791]
* rho 0.1 (width 3.2 A): +0.522 [+0.477, +0.572]
* rho 0.25 (width 2.0 A): +0.096 [+0.069, +0.124]
* rho 0.5 (width 1.4 A): +0.042 [+0.022, +0.061]
* rho 1 (width 1.0 A): +0.025 [-0.015, +0.060]

And each at its own best width -- posed at rho 0.25 (+0.899) against invariant at rho 0.5 (+0.847): +0.052 [+0.033, +0.071].

## Where the co-field sits

Fitted on every sample at rho 0.1 (q2 +0.887, error 4.448 ppm out of fold, ridge 0.01), which is wide enough for the envelope to reach the ring: the factor exp(-rho d^2) is ipso 0.82, ortho 0.56, meta 0.25, para 0.17.

The co-field is read at the four positions a substituent occupies on the ring. A positive value means that a positive field there raises the shift of the carbon at the origin.

| field | ipso | ortho | meta | para |
| --- | ---: | ---: | ---: | ---: |
| electrostatic | -16.6978 | +32.9700 | -28.5087 | -9.9682 |
| steric | +15.4635 | -2.6385 | -2.6748 | -4.2055 |
| hydrophobic | +1.3637 | -2.0035 | +6.6213 | +2.2462 |
| hba | -57.4090 | +17.7586 | +29.5922 | -41.1106 |
| hbd | +7.0534 | +58.4476 | -0.1523 | +62.5847 |

**Is the difference between the positions real?** The co-field is linear in the shifts, so the noise in them propagates to it in closed form, and the positions are correlated because they are built from the same training fields: the comparisons below are weighted combinations with the whole covariance taken into account, not two error bars added. The noise level is the out-of-fold error of the model itself, 4.45 ppm.

* **|para| - |meta|** is the comparison the plan asks for. The para position is further from the measured carbon (4.19 A) than the meta one (3.70 A), so the envelope and the overlap both favour meta; a co-field larger at para says what the geometry does not.
* **conjugated - meta** is (|ortho| + |para|) / 2 - |meta|, the same question with both conjugated positions.

| field | |para| - |meta| | z | conjugated - meta | z |
| --- | ---: | ---: | ---: | ---: |
| electrostatic | -18.541 +- 24.730 | -0.7 | -7.040 +- 19.147 | -0.4 |
| steric | +1.531 +- 1.534 | +1.0 | +0.747 +- 1.256 | +0.6 |
| hydrophobic | -4.375 +- 5.894 | -0.7 | -4.496 +- 4.459 | -1.0 |
| hba | +11.518 +- 25.205 | +0.5 | -0.158 +- 18.551 | -0.0 |
| hbd | +62.432 +- 39.216 | +1.6 | +60.364 +- 32.083 | +1.9 |

**Does it follow a measured substituent?** The correlation, over the four positions, of the co-field with the measured shifts a group causes: minus them for the donor, whose oxygen is negatively charged, plus them for the acceptor, whose nitrogen is positive. The null here is 20 y-scrambles -- the same geometry, frame, kernel and ridge with the shifts shuffled between the carbons.

| field | r (donor) | scramble z | signs | r (acceptor) | signs |
| --- | ---: | ---: | ---: | ---: | ---: |
| electrostatic | +0.550 | +1.2 | 3/4 | -0.538 | 0/4 |
| steric | -0.949 | -1.7 | 1/4 | +0.874 | 2/4 |
| hydrophobic | -0.185 | -0.0 | 1/4 | +0.101 | 4/4 |
| hba | +0.607 | +1.8 | 2/4 | -0.859 | 1/4 |
| hbd | +0.712 | +1.2 | 3/4 | -0.444 | 2/4 |

The measured shifts being compared with are ipso +31.4, ortho -14.4, meta +1.0, para -7.7 ppm for methoxy and ipso +19.9, ortho -4.9, meta +0.9, para +6.1 ppm for nitro, in that order of position. Both are large at ipso, ortho and para and almost nothing at meta, and they do **not** have opposite signs although the charges of their first atoms do: that part is resonance, which fragment charges cannot carry, and it is the same wall the Hammett co-fields of this repository met for sigma_p. `cofield.png` draws the co-field in the ring plane.

**No field puts more of its co-field at the para position than at the meta one** by more than the noise in the shifts allows.

## The other site: the carboxyl carbon of an acid

Plan item LP2 found no signal on this set: on 912 acids the invariant local kernel was no better than an atom-centred count, and the difference field of the acid and its anion was far worse. A carboxyl carbon has two oxygens and one carbon of its own, which the atom types separate, so its pose is unique and nothing has to be averaged. The question here is whether the missing frame was the reason. Only the model is measured: a co-field read off a model that does not predict would be an interpretation of noise.

| descriptor | envelope width rho | q2 | error (pKa units) | ridge | difference in q2 against the counts |
| --- | ---: | ---: | ---: | ---: | ---: |
| atom-centred counts (20 atom types in 6 shells) | -- | +0.541 | 0.563 | 0.01 | -- |
| posed overlap | 0.05 | +0.419 | 0.633 | 0.1 | -0.122 [-0.193, -0.048] |
| invariant local | 0.05 | +0.570 | 0.545 | 0.001 | +0.029 [-0.030, +0.086] |
| posed overlap | 0.1 | +0.440 | 0.621 | 0.1 | -0.101 [-0.164, -0.037] |
| invariant local | 0.1 | +0.582 | 0.537 | 0.001 | +0.041 [-0.013, +0.094] |
| posed overlap | 0.25 | +0.409 | 0.638 | 0.1 | -0.131 [-0.186, -0.071] |
| invariant local | 0.25 | +0.553 | 0.555 | 0.001 | +0.013 [-0.052, +0.075] |
| posed overlap | 0.5 | +0.351 | 0.669 | 0.0001 | -0.190 [-0.438, -0.030] |
| invariant local | 0.5 | +0.526 | 0.571 | 1e-06 | -0.014 [-0.079, +0.051] |
| posed overlap | 1 | +0.330 | 0.680 | 1e-05 | -0.211 [-0.499, -0.044] |
| invariant local | 1 | +0.470 | 0.604 | 1e-05 | -0.070 [-0.126, -0.006] |

At its best width the posed kernel loses to the count: -0.101 [-0.164, -0.037]. Against the invariant kernel at the same width it is -0.142 [-0.198, -0.088].

## The same co-field, read from the indicator fields

A partial charge cannot tell a pi-donor from a pi-acceptor: methoxy and nitro both raise the carbon they sit on while their first atoms carry opposite charges. A continuous indicator field carries the atom **type** instead, so an ether oxygen at the ortho position and a carbonyl carbon at the ortho position are different features with their own co-fields. That is how a table of substituent effects is written, and this asks whether the model writes the same table. 22 fields (17 types and the five physico-chemical ones), envelope width rho 0.05, q2 +0.882, error 4.55 ppm out of fold.

The numbers are **what one atom of that type at that position does to the predicted shift, in ppm** (`cofields.atom_effect`), with the weight the envelope gives an atom at that distance, so they are what a real substituent there contributes. `n` is how many of the carbons had an atom of that type at that position; a row is shown when one position reaches 20.

| type | ipso | ortho | meta | para | measured |
| --- | ---: | ---: | ---: | ---: | --- |
| H | -1.6 +- 0.6 (n 774) | +0.2 +- 0.4 (n 1483) | +0.5 +- 0.3 (n 1448) | -0.1 +- 0.3 (n 733) | -- |
| O.3 | +5.4 +- 1.6 (n 170) | -6.4 +- 1.1 (n 342) | -1.3 +- 0.8 (n 334) | -1.5 +- 0.9 (n 157) | +31.4, -14.4, +1.0, -7.7 (methoxy (anisole)) |
| C.3 | +13.6 +- 0.8 (n 126) | +0.5 +- 0.5 (n 242) | -1.1 +- 0.4 (n 239) | -0.5 +- 0.5 (n 120) | +9.3, +0.7, -0.1, -2.9 (methyl (toluene)) |
| C.2 | +9.0 +- 1.1 (n 87) | -0.2 +- 0.7 (n 172) | -0.7 +- 0.5 (n 168) | +0.7 +- 0.7 (n 84) | +9.3, +0.2, +0.2, +4.2 (acetyl (acetophenone)) |
| C.ar | +1.0 +- 0.2 (n 0) | -0.3 +- 0.2 (n 68) | -0.1 +- 0.1 (n 108) | -0.1 +- 0.2 (n 56) | -- |
| N.pl3 | +3.1 +- 2.0 (n 25) | -9.1 +- 1.4 (n 50) | -2.2 +- 1.1 (n 50) | -5.9 +- 1.4 (n 25) | +18.0, -13.3, +0.9, -9.8 (amino (aniline)) |
| Cl | +7.4 +- 2.2 (n 16) | +0.1 +- 1.5 (n 32) | +1.7 +- 1.3 (n 32) | -2.4 +- 1.7 (n 16) | +6.2, +0.4, +1.3, -1.9 (chlorine (chlorobenzene)) |

| type | r with the measured row | signs | root mean square difference (ppm) |
| --- | ---: | ---: | ---: |
| O.3 | +0.96 | 3/4 | 14.0 |
| C.3 | +0.96 | 4/4 | 2.5 |
| C.2 | +0.94 | 2/4 | 1.8 |
| N.pl3 | +0.99 | 3/4 | 8.1 |
| Cl | +1.00 | 4/4 | 0.7 |

Over all 20 (type, position) pairs with a measured row the correlation is +0.72.

The sharpest question these fields can answer is the one the electrostatic field could not: a donor lowers the para carbon and an acceptor raises it. Here the ether oxygen gives -1.5 +- 0.9 at para and the sp2 carbon +0.7 +- 0.7, against -7.7 and +4.2 measured.
`indicator.png` draws the two co-fields in the ring plane side by side. Both are positive where a substituent sits on the measured carbon and negative at the ortho positions; what separates the donor from the acceptor is the weaker pattern further out, which the table above reads off at the points rather than by eye.

