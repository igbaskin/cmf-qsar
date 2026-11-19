"""
Crown ethers and cryptands with the stability constants of their sodium and potassium
complexes in methanol, for plan item SC5.

    Izatt R. M., Bradshaw J. S., Nielsen S. A., Lamb J. D., Christensen J. J., Sen D.
    "Thermodynamic and kinetic data for cation-macrocycle interaction",
    Chem. Rev. 85 (1985) 271-339, Table I (pages 278-322) and Charts I-XII (pages 274-276).

Why this set is here
--------------------
Item SC5 asks whether a series of hosts can be aligned by its cavity and not by a scaffold, and
whether the co-field of a model built in that frame can be read.  A crown ether series is the
plainest case: every host holds the ion among the donor atoms of a ring, the property that
decides the constant (how well the ring fits the ion) is a matter of geometry, and the same
hosts were measured with two ions of different size, so the selectivity is a column of the same
table.

How the table was read
----------------------
No machine-readable open copy of the Izatt tabulations exists (the NIST stability-constant
database is licensed), so the values below were entered by hand from the review.  The article
is a scan with a text layer.  The rows were found in the text layer, and the text layer was
compared with the image of the page, digit by digit, on pages 283-287, 292, 293, 295-298, 314
and 315; no digit differed.  The owner of a row (the ligand a row belongs to) was read on the
page image wherever the text layer did not show it.  The structures were read from the charts
of the review, where every abbreviation of the table is drawn.

Rules, fixed before any model was run
-------------------------------------
* **Hosts:** monocyclic crown compounds and bicyclic cryptands with a value for both ions.
  Natural ionophores, the tricyclic cryptands and the bis(crown ether)s are left out.
* **Medium:** methanol, pure or with at most 0.1 M of a tetraalkylammonium salt, which is the
  usual medium of the cryptand measurements.  Mixed solvents and media with another alkali salt
  (0.15 M LiCl) are left out.
* **Temperature:** 25 degrees C.
* **Stoichiometry:** the constant of the one-to-one complex; a row marked (2) is the second
  step and is left out.
* **Value:** the median of the values the table gives for a host and an ion.  A value given as a
  range or as a limit ("3.3-3.6", ">7.0") is not used, and a host with no other value for one
  of the ions is left out.
* **Isomers:** a host is kept when the table names one compound.  Entries marked "mixed
  isomers" are left out, and so are hosts with two or more stereo centres whose isomer the
  table does not name.  A host with one stereo centre is a racemate in the measurement; one
  enantiomer is built.

What is left out, by name
-------------------------
Cy15C5, Cy18C6, Cy2-14C4 and THFMOM(Me)15C5 (the isomer is not named); Me5-15C5, Me6-18C6 and
Cy2-18C6 "mixture of isomers"; trans-anti-trans- and trans-syn-trans-Cy2-18C6 (measured between
19 and 24.5 degrees C, not at 25); meso-7,9-Me2B2-15C5, (MeB)2-18C6, Me(OE)2A18C6, 1,10-diaza-18-
crown-6 and B2-60C20 (one of the two ions has no value); [3.2.2]cryptand and Me4A4-[2.2.2]
(the potassium value is a limit); valinomycin, the nactins, enniatin B and monensin; the
tricyclic 1.1/1.1 cryptand.

A value the later review does not repeat
----------------------------------------
For aza-15-crown-5 with potassium the 1985 table gives 2.72 (its reference 76); the later
review of the same group (Chem. Rev. 91 (1991) 1721) gives 1.60 from another paper and does not
repeat 2.72.  The 1991 and 1995 reviews number their structures on drawn charts and were not
merged into this table; the value of the 1985 table is kept, and this is the one conflict that
was noticed.

Geometries
----------
A conformer generator does not give the conformation a binding constant belongs to: an empty
crown ether folds its ring.  ``--build`` therefore builds every host around each ion with
``cmfqsar.cavity.host_complex``: eight conformers of the host, the ion at the centre of the
donors or above and below a ring that is too small for it, and a GFN1-xTB optimisation in
implicit methanol in which every donor of the ring is pulled to its contact distance from the
ion (the sum of Shannon's radii, 2.42 A for sodium and 2.78 A for potassium) by a harmonic
restraint.  The start whose donors come closest to that distance is kept, and the host alone is
written in the frame of its cavity with the ion's position as a property.

So a geometry of this set is **the host as far around the ion as it can go**.  It is not a
minimum of the energy, and three trials are the reason:

* started from conformers of the free host without a restraint, GFN1 left the ion of plain
  15-crown-5 on three of its five oxygens, for both ions and for every start;
* released after the restrained optimisation, the potassium complex of 15-crown-5 went back to
  three contacts, two of 2.50 A and one of 2.59 A, where the sum of the radii is 2.78 A; GFN2
  draws the contacts shorter still (2.37 A on average for potassium on 12-crown-4, 2.08 A for
  sodium, and four contacts of 2.32 to 2.94 A for potassium on 15-crown-5), and GFN-FF gives
  five contacts of 2.50 to 2.71 A for potassium in 18-crown-6 and three in 15-crown-5;
* MMFF94 in RDKit gave the same geometry for sodium and for potassium.

A method that draws the contact too short prefers a few short contacts to many of the right
length, so its minimum is not the complex.  The restraint fixes the length and leaves the
shape of the ring to the method.  What a field model can learn from these geometries is
therefore the shape a host takes when it wraps an ion of that size: the radius of its donor
ring, how far the ion is from the plane, where the substituents go.  It cannot learn how much
energy the wrapping costs, because no energy enters.

GFN1 and not GFN2 for the shape as well: on the unrestrained complexes of 18-crown-6,
dibenzo-18-crown-6 and [2.2.2]cryptand the mean potassium-oxygen contact is 2.80, 2.79 and
2.78 A with GFN1 and 2.68, 2.60 and 2.71 A with GFN2.

Usage
-----
    python examples/make_crown_examples.py                 # crowns.csv (no xtb needed)
    python examples/make_crown_examples.py --build --workers 24   # the geometries: xtb, 22 min
    python examples/make_crown_examples.py --check
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "crowns")

SOURCE = "Izatt et al., Chem. Rev. 85 (1985) 271, Table I"
#: xtb method of the geometries (see the module text) and the number of conformers per complex
GFN = 1
STARTS = 8

P = "pure"            # methanol without a supporting electrolyte
ME4NBR = "0.01 M Me4NBr"
ET4NCLO4_05 = "0.05 M Et4NClO4"
ET4NCLO4_02 = "0.02 M Et4NClO4"
BU4NCLO4 = "0.1 M Bu4NClO4"

C15 = "COCCOCCOCCOCCO1"       # the rest of a 15-crown-5 ring after its substituted carbon
C18 = "COCCOCCOCCOCCOCCO1"    # the same for 18-crown-6
A15 = "N1CCOCCOCCOCCOCC1"     # aza-15-crown-5 from its nitrogen
A18 = "N1CCOCCOCCOCCOCCOCC1"  # aza-18-crown-6 from its nitrogen


def _v(value, method, ref, medium=P):
    return (float(value), method, str(ref), medium)


#: name, abbreviation in the table, family, SMILES, journal page(s), sodium values, potassium
#: values.  A value is (log K, method as the table abbreviates it, reference number(s) of the
#: review, medium).
HOSTS = [
    # ---- 12-membered rings, Chart I, page 278
    ("12-crown-4", "12C4", "12-crown-4", "C1COCCOCCOCCO1", "278",
     [_v(1.47, "Cal", 20), _v(1.41, "ISE", 21), _v(1.7, "ISE", 22)],
     [_v(1.59, "Pot", 20), _v(1.74, "ISE", 22), _v(1.58, "ISE", 21)]),
    ("hydroxymethyl-12-crown-4", "HOM12C4", "12-crown-4", "OCC1COCCOCCOCCO1", "278",
     [_v(1.37, "ISE", 21)], [_v(1.43, "ISE", 21)]),
    ("octyloxymethyl-12-crown-4", "OctOM12C4", "12-crown-4", "CCCCCCCCOCC1COCCOCCOCCO1", "278",
     [_v(1.32, "ISE", 21)], [_v(1.36, "ISE", 21)]),
    ("benzyloxymethyl-12-crown-4", "BzOM12C4", "12-crown-4", "C(OCc1ccccc1)C1COCCOCCOCCO1",
     "278", [_v(1.35, "ISE", 21)], [_v(1.42, "ISE", 21)]),
    # ---- 15-crown-5 and its carbon-substituted derivatives, Chart IV, pages 283-285
    ("15-crown-5", "15C5", "15-crown-5", "C1" + C15, "283",
     [_v(3.48, "Cal", 61), _v(3.30, "Cal", 67), _v(3.14, "Pot", 20), _v(3.25, "ISE", 55),
      _v(3.30, "ISE", 21), _v(3.24, "ISE", 22), _v(3.27, "Pot", 62), _v(3.31, "Pot", 63)],
     [_v(3.77, "Cal", 61), _v(3.35, "Cal", 67), _v(3.1, "ISE", 64), _v(3.61, "Pot", 20),
      _v(3.34, "ISE", 21), _v(3.43, "ISE", 22), _v(3.34, "Pot", 63)]),
    ("hexyl-15-crown-5", "Hex15C5", "15-crown-5", "CCCCCCC1" + C15, "284",
     [_v(3.15, "Pot", 63)], [_v(3.19, "Pot", 63)]),
    ("decyl-15-crown-5", "Dec15C5", "15-crown-5", "CCCCCCCCCCC1" + C15, "284",
     [_v(3.18, "Pot", 63)], [_v(3.15, "Pot", 63)]),
    ("phenyl-15-crown-5", "Ph15C5", "15-crown-5", "c1ccccc1C1" + C15, "284",
     [_v(3.34, "Pot", 63)], [_v(3.38, "Pot", 63)]),
    ("octyloxymethyl-15-crown-5", "OctOM15C5", "15-crown-5", "CCCCCCCCOCC1" + C15, "284",
     [_v(3.13, "Pot", 63)], [_v(2.95, "Pot", 63)]),
    ("dodecyloxymethyl-15-crown-5", "DodecOM15C5", "15-crown-5", "CCCCCCCCCCCCOCC1" + C15,
     "284", [_v(3.14, "Pot", 63)], [_v(3.09, "Pot", 63)]),
    ("(2-methoxyphenoxy)methyl-15-crown-5", "2-MeOPhOM15C5", "15-crown-5",
     "COc1ccccc1OCC1" + C15, "284", [_v(3.24, "Cal", 67)], [_v(3.32, "Cal", 67)]),
    ("(4-methoxyphenoxy)methyl-15-crown-5", "4-MeOPhOM15C5", "15-crown-5",
     "COc1ccc(cc1)OCC1" + C15, "284-285", [_v(2.90, "Cal", 67)], [_v(3.17, "Cal", 67)]),
    ("methyl-(2-methoxyethoxy)methyl-15-crown-5", "MeOEOM(Me)15C5", "15-crown-5",
     "COCCOCC1(C)" + C15, "285", [_v(3.87, "Pot", "62a")], [_v(3.42, "Pot", "62a")]),
    ("methyl-(2-pyridylmethoxy)methyl-15-crown-5", "PyMOM(Me)15C5", "15-crown-5",
     "CC1(COCc2ccccn2)" + C15, "285", [_v(3.58, "Pot", 62)], [_v(3.08, "Pot", 62)]),
    ("methyl-(2-methoxyphenoxy)methyl-15-crown-5", "2-MeOPhOM(Me)15C5", "15-crown-5",
     "COc1ccccc1OCC1(C)" + C15, "285", [_v(3.79, "Pot", 62)], [_v(3.35, "Pot", 62)]),
    ("methyl-(8-quinolyloxy)methyl-15-crown-5", "QuinOM(Me)15C5", "15-crown-5",
     "CC1(COc2cccc3cccnc23)" + C15, "285", [_v(4.87, "Pot", 62)], [_v(3.56, "Pot", 62)]),
    ("hexyl-(8-quinolyloxy)methyl-15-crown-5", "QuinOM(Hex)15C5", "15-crown-5",
     "CCCCCCC1(COc2cccc3cccnc23)" + C15, "285", [_v(4.85, "Pot", 62)], [_v(3.41, "Pot", 62)]),
    ("hexyl-(2-methoxyethoxy)methyl-15-crown-5", "MeOEOM(Hex)15C5", "15-crown-5",
     "CCCCCCC1(COCCOC)" + C15, "285", [_v(3.90, "Pot", 62)], [_v(3.29, "Pot", 62)]),
    ("bis[(2-methoxyethoxy)methyl]-15-crown-5", "(MeOEOM)2-15C5", "15-crown-5",
     "COCCOCC1(COCCOC)" + C15, "285", [_v(3.84, "Pot", 62)], [_v(3.44, "Pot", 62)]),
    ("bis[(2-(2-methoxyethoxy)ethoxy)methyl]-15-crown-5", "(MeOEOEOM)2-15C5", "15-crown-5",
     "COCCOCCOCC1(COCCOCCOC)" + C15, "285", [_v(3.86, "Pot", 62)], [_v(3.98, "Pot", 62)]),
    ("phenoxymethyl-15-crown-5", "PhOM15C5", "15-crown-5", "c1ccccc1OCC1" + C15, "285",
     [_v(3.07, "ISE", 21)], [_v(3.16, "ISE", 21)]),
    # ---- benzo-15-crown-5 compounds, Chart V, pages 285-286
    ("benzo-15-crown-5", "B15C5", "benzo-15-crown-5", "c1ccc2c(c1)OCCOCCOCCOCCO2", "285",
     [_v(3.05, "ISE", 64), _v(3.37, "Cond", 74), _v(2.87, "Pot", 340)], [_v(2.8, "", 64)]),
    ("4'-(1-hydroxy-2-methylaminoethyl)benzo-15-crown-5", "AdrB15C5", "benzo-15-crown-5",
     "CNCC(O)c1ccc2c(c1)OCCOCCOCCOCCO2", "286", [_v(2.94, "Pot", 340)], [_v(2.63, "Pot", 340)]),
    ("dibenzo-15-crown-5", "B2-15C5", "benzo-15-crown-5", "c1ccc2c(c1)OCCOCCOc1ccccc1OCCO2",
     "286", [_v(2.2, "ISE", 64)], [_v(2.0, "ISE", 64)]),
    ("rac-7,9-dimethyldibenzo-15-crown-5", "rac-7,9-Me2B2-15C5", "benzo-15-crown-5",
     "C[C@@H]1COc2ccccc2OCCOc2ccccc2OC[C@@H](C)O1", "286",
     [_v(1.9, "ISE", 64)], [_v(1.4, "ISE", 64)]),
    ("rac-6,10-dimethyldibenzo-15-crown-5", "rac-6,10-Me2B2-15C5", "benzo-15-crown-5",
     "C[C@@H]1COC[C@@H](C)Oc2ccccc2OCCOc2ccccc2O1", "286",
     [_v(1.1, "ISE", 64)], [_v(0.9, "ISE", 64)]),
    ("meso-6,10-dimethyldibenzo-15-crown-5", "meso-6,10-Me2B2-15C5", "benzo-15-crown-5",
     "C[C@@H]1COC[C@H](C)Oc2ccccc2OCCOc2ccccc2O1", "286",
     [_v(1.5, "ISE", 64)], [_v(1.4, "ISE", 64)]),
    # ---- aza-15-crown-5 compounds, Chart VI, pages 286-287
    ("aza-15-crown-5", "A15C5", "aza-15-crown-5", A15, "286",
     [_v(2.06, "ISE", 76)], [_v(2.72, "ISE", 76)]),
    ("N-butylaza-15-crown-5", "BuA15C5", "aza-15-crown-5", "CCCC" + A15, "286-287",
     [_v(3.22, "Cal", 67)], [_v(2.99, "Cal", 67)]),
    ("N-octylaza-15-crown-5", "OctA15C5", "aza-15-crown-5", "CCCCCCCC" + A15, "287",
     [_v(2.91, "ISE", 76)], [_v(3.05, "ISE", 76)]),
    ("N-(2-hydroxyethyl)aza-15-crown-5", "HOEA15C5", "aza-15-crown-5", "OCC" + A15, "287",
     [_v(3.92, "ISE", 76)], [_v(3.67, "ISE", 76)]),
    ("N-(2-methoxyethyl)aza-15-crown-5", "MeOEA15C5", "aza-15-crown-5", "COCC" + A15, "287",
     [_v(4.33, "Cal", 67)], [_v(4.20, "Cal", 67)]),
    ("N-[2-(2-hydroxyethoxy)ethyl]aza-15-crown-5", "H(OE)2A15C5", "aza-15-crown-5",
     "OCCOCC" + A15, "287", [_v(4.68, "ISE", 76)], [_v(4.42, "ISE", 76)]),
    ("N-[H(OCH2CH2)3]aza-15-crown-5", "H(OE)3A15C5", "aza-15-crown-5",
     "OCCOCCOCC" + A15, "287", [_v(4.34, "ISE", 76)], [_v(4.77, "ISE", 76)]),
    ("N-[CH3(OCH2CH2)3]aza-15-crown-5", "Me(OE)3A15C5", "aza-15-crown-5",
     "COCCOCCOCC" + A15, "287", [_v(4.32, "ISE", 76)], [_v(4.85, "ISE", 76)]),
    # ---- 18-crown-6 and its carbon-substituted derivatives, Chart VII, pages 289-294
    ("18-crown-6", "18C6", "18-crown-6", "C1" + C18, "289",
     [_v(4.36, "Cal", "61, 91, 92"), _v(4.38, "Cal", 20), _v(4.46, "Cond", 74),
      _v(4.32, "Elec", 54), _v(4.35, "ISE", 55), _v(4.35, "ISE", 22), _v(4.32, "ISE", "11, 25")],
     [_v(6.06, "Cal", "61, 92"), _v(6.08, "Cond", 93), _v(6.20, "Cond", 74),
      _v(6.1, "Elec", 54), _v(5.93, "ISE", 64), _v(6.08, "ISE", 22), _v(6.18, "Pot", 20),
      _v(6.10, "ISE", "11, 25")]),
    ("cis-anti-cis-dicyclohexano-18-crown-6", "cis-anti-cis-Cy2-18C6", "18-crown-6",
     "C1CC[C@H]2OCCOCCO[C@@H]3CCCC[C@@H]3OCCOCCO[C@H]2C1", "292",
     [_v(3.68, "ISE", "11, 25")], [_v(5.38, "ISE", "11, 25")]),
    ("cis-syn-cis-dicyclohexano-18-crown-6", "cis-syn-cis-Cy2-18C6", "18-crown-6",
     "C1CC[C@H]2OCCOCCO[C@H]3CCCC[C@H]3OCCOCCO[C@H]2C1", "292-293",
     [_v(4.08, "ISE", "11, 25")], [_v(6.01, "ISE", "11, 25")]),
    ("2,6-dioxo-18-crown-6", "K2-18C6", "18-crown-6", "O=C1COCC(=O)OCCOCCOCCOCCO1", "293",
     [_v(2.5, "Cal", "118, 119")], [_v(2.79, "Cal", "118, 119")]),
    ("octyl-18-crown-6", "Oct18C6", "18-crown-6", "CCCCCCCCC1" + C18, "294",
     [_v(3.91, "Spec", 63)], [_v(5.03, "Spec", 63)]),
    ("dodecyl-18-crown-6", "Dodec18C6", "18-crown-6", "CCCCCCCCCCCCC1" + C18, "294",
     [_v(3.93, "Spec", 63)], [_v(5.28, "Spec", 63)]),
    ("phenyl-18-crown-6", "Ph18C6", "18-crown-6", "c1ccccc1C1" + C18, "294",
     [_v(4.17, "Spec", 63)], [_v(5.56, "Spec", 63)]),
    ("octyloxymethyl-18-crown-6", "OctOM18C6", "18-crown-6", "CCCCCCCCOCC1" + C18, "294",
     [_v(3.88, "Spec", 63)], [_v(5.36, "Spec", 63)]),
    ("dodecyloxymethyl-18-crown-6", "DodecOM18C6", "18-crown-6", "CCCCCCCCCCCCOCC1" + C18,
     "294", [_v(3.83, "Spec", 63)], [_v(5.37, "Spec", 63)]),
    ("pyridino-18-crown-6", "Py18C6", "18-crown-6", "c1cc2nc(c1)COCCOCCOCCOCCOC2", "299",
     [_v(4.09, "Cal", "61, 102")], [_v(5.35, "Cal", "61, 102")]),
    ("diketopyridino-18-crown-6", "K2Py18C6", "18-crown-6",
     "O=C1OCCOCCOCCOCCOC(=O)c2cccc1n2", "299",
     [_v(4.29, "Cal", "102, 119")], [_v(4.66, "Cal", "102, 119")]),
    ("thia-18-crown-6", "T18C6", "18-crown-6", "C1COCCOCCOCCOCCOCCS1", "301",
     [_v(2.57, "Cal", 61)], [_v(3.61, "Cal", 61)]),
    # ---- benzo-18-crown-6 compounds, Charts VIII and IX, pages 295-297
    ("benzo-18-crown-6", "B18C6", "benzo-18-crown-6", "c1ccc2c(c1)OCCOCCOCCOCCOCCO2", "295",
     [_v(4.35, "Pot", 340), _v(4.03, "Spec", 122), _v(4.5, "Cond", 338)],
     [_v(5.05, "Pot", 340), _v(5.2, "ISE", 64), _v(5.27, "Spec", 122), _v(5.7, "Cond", 338)]),
    ("dibenzo-18-crown-6", "B2-18C6", "benzo-18-crown-6",
     "c1ccc2c(c1)OCCOCCOc1ccccc1OCCOCCO2", "296-297",
     [_v(4.5, "Cal", 127), _v(4.36, "Elec", 54), _v(4.36, "ISE", "11, 25"),
      _v(4.4, "Sol", "90a"), _v(4.36, "Spec", 63)],
     [_v(5.00, "ISE", "11, 25"), _v(5.1, "Cal", 127), _v(5.08, "Cond", 93),
      _v(5.00, "Elec", 54), _v(4.8, "ISE", 64), _v(5.05, "ISE", "90a"), _v(5.00, "Spec", 63)]),
    ("meso-6,10-dimethyldibenzo-18-crown-6", "meso-6,10-Me2B2-18C6", "benzo-18-crown-6",
     "C[C@@H]1COC[C@H](C)Oc2ccccc2OCCOCCOc2ccccc2O1", "297",
     [_v(3.56, "ISE", 64)], [_v(4.13, "ISE", 64)]),
    ("rac-6,10-dimethyldibenzo-18-crown-6", "rac-6,10-Me2B2-18C6", "benzo-18-crown-6",
     "C[C@@H]1COC[C@@H](C)Oc2ccccc2OCCOCCOc2ccccc2O1", "297",
     [_v(3.77, "ISE", 64)], [_v(4.37, "ISE", 64)]),
    ("meso-7,9-dimethyldibenzo-18-crown-6", "meso-7,9-Me2B2-18C6", "benzo-18-crown-6",
     "C[C@@H]1COc2ccccc2OCCOCCOc2ccccc2OC[C@H](C)O1", "297",
     [_v(2.79, "ISE", 64)], [_v(3.42, "ISE", 64)]),
    ("rac-7,9-dimethyldibenzo-18-crown-6", "rac-7,9-Me2B2-18C6", "benzo-18-crown-6",
     "C[C@@H]1COc2ccccc2OCCOCCOc2ccccc2OC[C@@H](C)O1", "297",
     [_v(3.03, "ISE", 64)], [_v(4.04, "ISE", 64)]),
    # ---- aza-18-crown-6 compounds, Charts IX and X, pages 298-300
    ("aza-18-crown-6", "A18C6", "aza-18-crown-6", A18, "298",
     [_v(2.77, "ISE", 76)], [_v(4.18, "ISE", 76), _v(3.90, "ISE", "11, 25")]),
    ("N-octylaza-18-crown-6", "OctA18C6", "aza-18-crown-6", "CCCCCCCC" + A18, "298",
     [_v(3.51, "ISE", 76)], [_v(4.65, "ISE", 76)]),
    ("N-(2-hydroxyethyl)aza-18-crown-6", "HOEA18C6", "aza-18-crown-6", "OCC" + A18, "298",
     [_v(4.75, "ISE", 76)], [_v(5.49, "ISE", 76)]),
    ("N-(2-methoxyethyl)aza-18-crown-6", "MeOEA18C6", "aza-18-crown-6", "COCC" + A18, "298",
     [_v(5.6, "Cal", 67)], [_v(5.35, "Cal", 67)]),
    ("N-[2-(2-hydroxyethoxy)ethyl]aza-18-crown-6", "H(OE)2A18C6", "aza-18-crown-6",
     "OCCOCC" + A18, "298", [_v(4.34, "ISE", 76)], [_v(5.88, "ISE", 76)]),
    ("N-[H(OCH2CH2)3]aza-18-crown-6", "H(OE)3A18C6", "aza-18-crown-6",
     "OCCOCCOCC" + A18, "299", [_v(4.26, "ISE", 76)], [_v(5.69, "ISE", 76)]),
    ("N-[CH3(OCH2CH2)3]aza-18-crown-6", "Me(OE)3A18C6", "aza-18-crown-6",
     "COCCOCCOCC" + A18, "299", [_v(4.28, "ISE", 76)], [_v(5.96, "ISE", 76)]),
    ("N,N'-dimethyl-1,10-diaza-18-crown-6", "Me2A2-18C6", "aza-18-crown-6",
     "CN1CCOCCOCCN(C)CCOCCOCC1", "300",
     [_v(3.7, "Pot", 134, ME4NBR)], [_v(5.3, "Pot", 134, ME4NBR)]),
    # ---- rings of 19 to 30 atoms, Charts III, VIII and X, pages 301-305
    ("2,4-dioxo-19-crown-6", "K2-19C6", "larger ring", "O=C1CC(=O)OCCOCCOCCOCCOCCO1", "301",
     [_v(1.8, "Cal", "118, 119")], [_v(2.55, "Cal", "118, 119")]),
    ("2,5-dioxo-20-crown-6", "K2-20C6", "larger ring", "O=C1CCC(=O)OCCOCCOCCOCCOCCO1", "301",
     [_v(1.7, "Cal", 119)], [_v(1.94, "Cal", 119)]),
    ("21-crown-7", "21C7", "larger ring", "C1COCCOCCOCCOCCOCCOCCO1", "301",
     [_v(1.73, "Cal", 61), _v(2.54, "ISE", 22)],
     [_v(4.22, "Cal", 61), _v(4.35, "ISE", 22), _v(4.41, "ISE", "11, 25")]),
    ("dibenzo-21-crown-7", "B2-21C7", "larger ring",
     "c1ccc2c(c1)OCCOCCOc1ccccc1OCCOCCOCCO2", "301",
     [_v(2.40, "ISE", "11, 25")], [_v(4.30, "ISE", "11, 25")]),
    ("24-crown-8", "24C8", "larger ring", "C1COCCOCCOCCOCCOCCOCCOCCO1", "302",
     [_v(2.35, "ISE", 22)], [_v(3.53, "ISE", 22), _v(3.48, "ISE", "11, 25")]),
    ("1,7-dibenzo-24-crown-8", "1,7-B2-24C8", "larger ring",
     "c1ccc2c(c1)OCCOc1ccccc1OCCOCCOCCOCCOCCO2", "302",
     [_v(2.55, "Pot", 138)], [_v(3.85, "Pot", 138)]),
    ("1,10-dibenzo-24-crown-8", "1,10-B2-24C8", "larger ring",
     "c1ccc2c(c1)OCCOCCOc1ccccc1OCCOCCOCCOCCO2", "302",
     [_v(2.15, "Pot", 138)], [_v(3.45, "Pot", 138)]),
    ("1,13-dibenzo-24-crown-8", "1,13-B2-24C8", "larger ring",
     "c1ccc2c(c1)OCCOCCOCCOc1ccccc1OCCOCCOCCO2", "302-303",
     [_v(2.25, "Pot", 138)], [_v(3.6, "Pot", 138), _v(3.57, "Cond", 333)]),
    ("dibenzo-30-crown-10", "B2-30C10", "larger ring",
     "c1ccc2c(c1)OCCOCCOCCOCCOc1ccccc1OCCOCCOCCOCCO2", "304-305",
     [_v(2.0, "ISE", "11, 25"), _v(2.1, "Spec", 139)], [_v(4.60, "ISE", "11, 25")]),
    # ---- cryptands, Charts XI and XII, pages 305-315
    ("[2.1.1]cryptand", "2.1.1", "cryptand", "C1COCCOCCN2CCOCCN1CCOCC2", "306",
     [_v(6.1, "Pot", 134, ME4NBR), _v(6.7, "Pot", 148, BU4NCLO4)],
     [_v(2.3, "Pot", 134, ME4NBR)]),
    ("[2.2.1]cryptand", "2.2.1", "cryptand", "C1COCCOCCN2CCOCCOCCN1CCOCC2", "308",
     [_v(9.65, "Pot", 146, ET4NCLO4_05), _v(9.3, "Pot", 148, BU4NCLO4)],
     [_v(8.54, "Pot", 146, ET4NCLO4_05), _v(8.54, "Cond", 156)]),
    ("[2.2.2]cryptand", "2.2.2", "cryptand", "C1COCCOCCN2CCOCCOCCN1CCOCCOCC2", "310-311",
     [_v(7.9, "CyVol", 162, ET4NCLO4_05), _v(7.98, "Pot", 146, ET4NCLO4_05),
      _v(7.8, "Pot", 148, BU4NCLO4)],
     [_v(10.41, "Pot", 146, ET4NCLO4_05), _v(10.8, "Pot", 162, ET4NCLO4_05)]),
    ("benzo[2.2.2]cryptand", "B2.2.2", "cryptand",
     "c1ccc2c(c1)OCCN1CCOCCOCCN(CCO2)CCOCCOCC1", "313",
     [_v(7.50, "Ag+ ISE", 173, ET4NCLO4_02)], [_v(9.21, "Ag+ ISE", 173, ET4NCLO4_02)]),
    ("dibenzo[2.2.2]cryptand", "B2-2.2.2", "cryptand",
     "c1ccc2c(c1)OCCN1CCOCCOCCN(CCO2)CCOc2ccccc2OCC1", "313",
     [_v(7.37, "Pot", 169), _v(7.60, "Ag+ ISE", 174)],
     [_v(8.74, "Ag+ ISE", 174), _v(8.60, "Pot", 169)]),
    ("decyl[2.2.2]cryptand", "Dec2.2.2", "cryptand",
     "CCCCCCCCCCC1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2", "313",
     [_v(7.0, "Ag+ ISE", 174)], [_v(9.42, "Ag+ ISE", 174)]),
    ("[2.2.C8]cryptand", "2.2.C8", "cryptand", "C1COCCOCCN2CCOCCOCCN1CCCCCCCC2", "314",
     [_v(3.5, "Pot", 134, ME4NBR)], [_v(5.2, "Pot", 134, ME4NBR)]),
    ("[3.3.2]cryptand", "3.3.2", "cryptand",
     "C1COCCOCCOCCN2CCOCCOCCOCCN1CCOCCOCC2", "315",
     [_v(3.2, "Pot", 134, ME4NBR)], [_v(6.0, "Pot", 134, ME4NBR)]),
    ("[3.3.3]cryptand", "3.3.3", "cryptand",
     "C1COCCOCCOCCN2CCOCCOCCOCCN1CCOCCOCCOCC2", "315",
     [_v(2.7, "Pot", 134, ME4NBR)], [_v(5.4, "Pot", 134, ME4NBR)]),
]

FIELDS = ("name", "abbreviation", "family", "smiles", "ring_size", "ring_donors",
          "logK_Na", "logK_K", "selectivity_K_Na", "n_Na", "n_K", "values_Na", "values_K",
          "medium", "page")


def _values_text(values):
    return "; ".join(f"{v:g} ({m or 'method not given'}, ref {r})" for v, m, r, _med in values)


def rows():
    """One dictionary per host: the medians, the values they come from, and the ring."""
    from rdkit import Chem
    from cmfqsar.cavity import donor_atoms, macrocycle_atoms
    out = []
    for name, abbr, family, smiles, page, na, k in HOSTS:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"{abbr}: RDKit cannot read {smiles}")
        rings = [r for r in mol.GetRingInfo().AtomRings() if len(r) >= 9]
        media = sorted({v[3] for v in na + k})
        ln, lk = median(v[0] for v in na), median(v[0] for v in k)
        out.append({
            "name": name, "abbreviation": abbr, "family": family,
            "smiles": Chem.MolToSmiles(mol), "ring_size": min(len(r) for r in rings),
            "ring_donors": len(donor_atoms(mol)), "logK_Na": round(ln, 3), "logK_K": round(lk, 3),
            "selectivity_K_Na": round(lk - ln, 3), "n_Na": len(na), "n_K": len(k),
            "values_Na": _values_text(na), "values_K": _values_text(k),
            "medium": "methanol" if media == [P] else "methanol, " + " or ".join(
                m if m != P else "none" for m in media),
            "page": page, "_atoms": len(macrocycle_atoms(mol)),
        })
    return out


def write_table(path=None):
    path = path or os.path.join(OUT, "crowns.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = rows()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    print(f"{len(data)} hosts -> {os.path.relpath(path, ROOT)}")
    return data


# --------------------------------------------------------------------------- #
# Geometries
# --------------------------------------------------------------------------- #
def _build_one(job):
    """One host with one ion; returns a dictionary a JSON file can hold."""
    abbr, smiles, ion, work = job
    from rdkit import Chem
    from cmfqsar import cavity as C
    t0 = time.time()
    try:
        r = C.host_complex(smiles, ion, conformers=STARTS, seed=0, solvent="methanol",
                           workdir=work, threads=1, gfn=GFN, release=False)
    except Exception as e:                                   # noqa: BLE001
        return {"abbreviation": abbr, "ion": ion, "error": f"{type(e).__name__}: {e}"}
    host, cx = r["host"], r["complex"]
    donors = C.donor_atoms(host)
    xyz = cx.GetConformer().GetPositions()
    ion_xyz = xyz[r["ion_index"]]
    origin, rot = C.cavity_axes(xyz[:-1], donors,
                                [a.GetIdx() for a in host.GetAtoms() if a.GetAtomicNum() > 1])
    framed = C.with_cavity_frame(host)
    return {"abbreviation": abbr, "ion": ion, "energy": float(r["energy"]),
            "kept": int(r["kept"]), "starts": int(r["starts"]),
            "contacts": C.ion_contacts(xyz[:-1], donors, ion_xyz, ion),
            "misfit": float(r["misfit"]),
            "ion_xyz": [float(v) for v in (ion_xyz - origin) @ rot.T],
            "molblock": Chem.MolToMolBlock(framed), "seconds": round(time.time() - t0, 1)}


def build(workers=8, only=None, work=None):
    """The hosts in the conformation of their sodium and potassium complexes."""
    from concurrent.futures import ProcessPoolExecutor
    from rdkit import Chem
    work = work or os.path.join(ROOT, "xtb_work", "crowns")
    data = {r["abbreviation"]: r for r in rows()}
    # the restart log holds every geometry and stays with the xtb scratch; the example folder
    # gets the short record without the geometries, which are in the two SDF files
    os.makedirs(work, exist_ok=True)
    log_path = os.path.join(work, "build_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    jobs = []
    for name, abbr, _fam, smiles, *_rest in HOSTS:
        for ion in ("Na", "K"):
            key = f"{abbr}|{ion}"
            if only and abbr not in only:
                continue
            if key in log and "error" not in log[key]:
                continue
            safe = "".join(c if c.isalnum() else "_" for c in abbr)
            jobs.append((abbr, smiles, ion, os.path.join(work, f"{safe}_{ion}")))
    print(f"{len(jobs)} complexes to build with {workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(_build_one, jobs), 1):
            key = f"{got['abbreviation']}|{got['ion']}"
            log[key] = got
            print(f"  {i}/{len(jobs)} {key}: " + (got.get("error") or
                  f"kept {got['kept']}/{got['starts']}, {got['contacts']} contacts, "
                  f"misfit {got['misfit']:.2f} A, "
                  f"{got['seconds']:.0f} s") + f" ({time.time() - t0:.0f} s)", flush=True)
            with open(log_path, "w", encoding="utf-8") as fh:
                json.dump(log, fh, indent=0, sort_keys=True)
    with open(os.path.join(OUT, "build.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"method": f"GFN{GFN}-xTB, ALPB methanol, every ring donor restrained to the "
                             "sum of the ionic radii from the ion",
                   "conformers": STARTS,
                   "complexes": {k: {f: v[f] for f in v if f != "molblock"}
                                 for k, v in sorted(log.items())}}, fh, indent=1)
    for ion in ("Na", "K"):
        path = os.path.join(OUT, f"hosts_{ion}.sdf")
        n = 0
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for name, abbr, *_rest in HOSTS:
                got = log.get(f"{abbr}|{ion}")
                if not got or "error" in got:
                    continue
                row = data[abbr]
                mol = Chem.MolFromMolBlock(got["molblock"], removeHs=False)
                mol.SetProp("_Name", abbr)
                for key in ("name", "family", "smiles", "logK_Na", "logK_K", "selectivity_K_Na",
                            "ring_size", "ring_donors"):
                    mol.SetProp(key, str(row[key]))
                mol.SetProp("ion", ion)
                mol.SetProp("ion_xyz", " ".join(f"{v:.4f}" for v in got["ion_xyz"]))
                mol.SetProp("ion_contacts", str(got["contacts"]))
                mol.SetProp("donor_misfit", f"{got['misfit']:.4f}")
                mol.SetProp("starts_kept", f"{got['kept']} of {got['starts']}")
                mol.SetProp("xtb_energy_hartree", f"{got['energy']:.8f}")
                fh.write(Chem.MolToMolBlock(mol) + "".join(
                    f">  <{p}>\n{mol.GetProp(p)}\n\n" for p in mol.GetPropNames()) + "$$$$\n")
                n += 1
        print(f"{n} hosts -> {os.path.relpath(path, ROOT)}")


def check():
    """The table against itself: every SMILES is read, and each named isomer is that isomer."""
    from rdkit import Chem
    data = rows()
    seen = set()
    for r in data:
        assert r["smiles"] not in seen, f"{r['abbreviation']} repeats a structure"
        seen.add(r["smiles"])
        mol = Chem.MolFromSmiles(r["smiles"])
        mirror = Chem.MolToSmiles(Chem.MolFromSmiles(
            r["smiles"].replace("@@", "!").replace("@", "@@").replace("!", "@")))
        if r["abbreviation"].startswith("meso") or "-cis-" in r["abbreviation"]:
            assert mirror == r["smiles"], f"{r['abbreviation']} is not its own mirror image"
        if r["abbreviation"].startswith("rac"):
            assert mirror != r["smiles"], f"{r['abbreviation']} is not chiral"
        centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                            useLegacyImplementation=False)
        # a bridgehead nitrogen of a cryptand inverts freely and is not a stereo centre here
        open_ = [c for c in centres if c[1] == "?" and mol.GetAtomWithIdx(c[0]).GetSymbol() == "C"]
        assert len(open_) <= 1, f"{r['abbreviation']} has {len(open_)} open stereo centres"
    fam = {}
    for r in data:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    print(f"{len(data)} hosts, families: " + ", ".join(f"{k} {v}" for k, v in fam.items()))
    for ion in ("Na", "K"):
        path = os.path.join(OUT, f"hosts_{ion}.sdf")
        if os.path.exists(path):
            n = sum(1 for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None)
            print(f"{os.path.relpath(path, ROOT)}: {n} hosts")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="build the geometries with xtb")
    ap.add_argument("--check", action="store_true", help="check the table and the files")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="abbreviations to build, for a trial")
    a = ap.parse_args()
    if a.check:
        check()
        return
    write_table()
    if a.build:
        build(a.workers, a.only)


if __name__ == "__main__":
    main()
