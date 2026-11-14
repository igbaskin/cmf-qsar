# Buchwald-Hartwig couplings with a measured yield (plan item PO2)

Built by `examples/make_buchwald_examples.py --fetch` from the high-throughput screen of

> Ahneman D. T., Estrada J. G., Lin S., Dreher S. D., Doyle A. G. "Predicting reaction
> performance in C-N cross-coupling using machine learning", Science 360 (2018) 186-190.

as distributed in the Yield-BERT repository, <https://github.com/rxn4chemistry/rxn_yields>,
file `data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx`, under the MIT licence; its notice
is kept here as `LICENSE.rxn_yields`. The workbook itself is downloaded to `upstream/`, which is
not kept in the repository; the script records its SHA-256 and warns if a new download differs.

## The screen

The coupling of an aryl halide with 4-methylaniline, catalysed by palladium with a biaryl
phosphine ligand, in the presence of a base and of an isoxazole additive. Every combination of

| role | number | members |
|---|---:|---|
| aryl halide | 15 | chloride, bromide and iodide of 4-CF3-phenyl, 4-MeO-phenyl, 4-Et-phenyl, 2-pyridyl and 3-pyridyl |
| ligand | 4 | XPhos, t-BuXPhos, t-BuBrettPhos, AdBrettPhos |
| base | 3 | P2Et (a phosphazene), BTMG (a guanidine), MTBD (an amidine) |
| additive | 22 | isoxazoles, two of them benzo-fused, A01 to A22 in the order of their SMILES |

was run, 3,960 in all, and 3,955 have a yield (0 to 100 %, mean 33 %). The additives were added
to test how a reaction tolerates a heterocycle with a weak N-O bond; they are the component
that the published out-of-sample splits hold out.

## The files

| file | content |
|---|---|
| `reactions.csv` | one row per reaction: its four components by label and the yield in percent |
| `components.csv` | every component: role, label, SMILES |
| `splits.csv` | for each published split, `train` or `test` for every reaction |
| `aryl_halides.sdf` | the 15 aryl halides in the frame of the C-X bond, four images each |
| `additives.sdf` | the 22 additives in the frame of the N-O bond, two images each |
| `LICENSE.rxn_yields` | the MIT notice of the source |

**The splits** are those every later paper uses: ten random 70/30 splits (`random_01` to
`random_10`, from the sheets FullCV_01 to FullCV_10) and four out-of-sample splits that hold out
five additives each (`additive_test1` to `additive_test4`, from the sheets Test1 to Test4). The
training part of a sheet is its first rows, 2,767 for the random sheets and 3,055 to 3,058 for
the test sheets, as the Yield-BERT training script reads them.

## The frames

Each structure is the lowest of 20 ETKDG conformers minimised with MMFF94, checked for bond
lengths and contacts before it is written.

* **Aryl halide**: the ipso carbon at the origin, the halogen on +x, the ring in the xy plane.
  A planar aryl halide is the same molecule after a half turn about its C-X bond and after a
  reflection through its ring plane, so each is written as the four images these operations
  give, under one name, and the field model averages over them.
* **Additive**: the midpoint of the N-O bond at the origin, the nitrogen on +x, the centre of
  the isoxazole ring on +y, so the ring lies in the xy plane with C3 next to the nitrogen and C5
  next to the oxygen. The two faces of the ring are equivalent, so each additive is written as
  itself and its reflection through the ring plane.

The ligands and the bases have no structure here. With four ligands and three bases a model can
only tell them apart, which their labels already do.
