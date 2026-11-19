# Crown ethers and cryptands with Na+ and K+ stability constants (plan item SC5)

Built by `examples/make_crown_examples.py` from

> Izatt R. M., Bradshaw J. S., Nielsen S. A., Lamb J. D., Christensen J. J., Sen D.
> "Thermodynamic and kinetic data for cation-macrocycle interaction",
> Chem. Rev. 85 (1985) 271-339, Table I (pages 278-322) and Charts I-XII (pages 274-276).

The values are facts taken from a published table and were entered by hand; the article itself
is not in the repository. The structures were drawn from the charts of the review, and the
geometries were computed here.

## The files

| File | Content |
|---|---|
| `crowns.csv` | 79 hosts: name, abbreviation as the review prints it, family, SMILES, ring size, number of donor atoms in the ring, log K of the sodium and the potassium complex, their difference, every value the median was taken from (with its method and the review's reference number), the medium and the page |
| `hosts_Na.sdf` | every host wrapped around a sodium ion, in the frame of its cavity, without the ion; the ion's position is the property `ion_xyz` |
| `hosts_K.sdf` | the same around a potassium ion |
| `build.json` | the record of the build: per complex the energy, the starts kept, the contacts and the misfit |

## The rules of the table

* **Hosts:** monocyclic crown compounds and bicyclic cryptands with a value for both ions.
* **Medium:** methanol, pure or with at most 0.1 M of a tetraalkylammonium salt. The column
  `medium` says which.
* **Temperature:** 25 degrees C.
* **Stoichiometry:** the one-to-one complex.
* **Value:** the median of the values the table gives for a host and an ion. A range or a limit
  is not used.
* **Isomers:** a host is kept when the table names one compound. Mixtures of isomers are left
  out, and so are hosts with two or more stereo centres whose isomer is not named.

`examples/make_crown_examples.py` lists by name what these rules leave out, and says how the
table was read: the rows were found in the text layer of the scanned article and compared with
the page image. Of the 220 values, 215 were found again by a program in the text layer of the
stated page, with their method and reference number; the other five (sodium with
dibenzo-18-crown-6, page 296) stand in rows the text layer breaks, and were read on the image.

The later reviews of the same group (Chem. Rev. 91 (1991) 1721 and 95 (1995) 2529) number their
structures on drawn charts and were not merged. One conflict was noticed: for aza-15-crown-5
with potassium the 1985 table gives 2.72 and the 1991 review gives 1.60 from another paper.

## What the geometries are, and what that costs

A geometry is **the host as far around the ion as it can go**: a GFN1-xTB optimisation in
implicit methanol in which every donor atom of the ring is held at the sum of Shannon's ionic
radii from the ion (2.42 A for sodium, 2.78 A for potassium) by a harmonic restraint. It is not
a minimum of the energy. Without the restraint none of GFN1-xTB, GFN2-xTB and GFN-FF kept the
potassium ion of plain 15-crown-5 on all five of its oxygens (three with GFN1 and with GFN-FF,
four with GFN2), because each of them draws the ion-oxygen contact too short;
`examples/make_crown_examples.py` gives the numbers.

What a model can learn from these geometries is the shape a host takes when it wraps an ion of
that size: the radius of its donor ring, how far the ion stays from the plane of the donors,
where the substituents go. It cannot learn what the wrapping costs, because no energy enters.
The frame is that of `cmfqsar/cavity.py` and of `cmf align --by cavity`: the origin at the
centroid of the donor atoms of the macrocycle, z along their unique axis.

## The simple references

`benchmarks/run_crowns.py` measures every field model against numbers read from the formula
(ring size, donors of the ring by element, fused aromatic rings, donors on side arms, bicyclic
or not) and against numbers read from the complex (donors in contact, their misfit, the radius
of the donor ring, the distance of the ion from the donor centroid). Its table is
`benchmarks/results/crowns/table.md`.
