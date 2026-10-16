"""
Atomic increments of the original CMF implementation (Baskin & Zhokhova 2013).

The R scripts that produced the models of the 2013 paper parameterised the
hydrophobic field and the two hydrogen-bond fields with in-house one-atom
fragment increments obtained by regression of experimental logP values and of
Abraham's hydrogen-bond acidity (A) and basicity (B) solute constants (the
scripts also carry increments for Abraham's E and S constants).  The increments
are keyed by the Sybyl atom type and by the number of hydrogen atoms attached
to the atom.  Hydrogen atoms themselves carry no increments, i.e. the donor
field is centred on the heavy atom bearing the hydrogen.

This module is a faithful transcription of ``cmf-params.R``; the numbers must
not be "improved" here, because their purpose is to reproduce the published
models.  Use ``param_scheme="tripos"`` of :class:`cmfqsar.fields.FieldSet` to
select this parameterisation.
"""
from __future__ import annotations

from typing import Dict

INCREMENT_KEYS = ("hydrophobic", "hbd", "hba", "abraham_e", "abraham_s")


def normalize_sybyl_type(syb: str) -> str:
    """Canonical spelling of a Sybyl atom type (``S.o2``/``S.O2`` -> ``S.o2``)."""
    low = syb.lower()
    if low in ("s.o", "s.o2"):
        return "S." + low[2:]
    return syb


def tripos_increments(syb: str, n_h: int) -> Dict[str, float]:
    """Increments of one atom of Sybyl type `syb` carrying `n_h` hydrogen atoms.

    Returns a dict with keys ``hydrophobic`` (logP increment), ``hbd``
    (Abraham A, hydrogen-bond donor), ``hba`` (Abraham B, acceptor),
    ``abraham_e`` and ``abraham_s``.
    """
    h = a = b = e = s = 0.0
    t = normalize_sybyl_type(syb)
    if t == "C.1":
        h += 0.041270; e += -0.005210; s += -0.013906
        if n_h == 1:
            h += -0.093156; s += 0.045181
    elif t == "C.2":
        h += 0.041270; e += -0.005210; s += -0.013906
        h += 0.006410; s += 0.016508
        if n_h == 0:
            h += 0.007629; b += 0.026654; e += 0.076572
        elif n_h == 1:
            h += 0.202552; e += 0.058299
        elif n_h == 2:
            h += 0.422564; b += -0.041897; e += -0.019698; s += -0.098377
    elif t == "C.3":
        h += 0.041270; e += -0.005210; s += -0.013906; a += -0.009684
        if n_h == 0:
            h += 0.089007; b += 0.096834; e += 0.153114
        elif n_h == 1:
            b += 0.041221; e += 0.091218
        elif n_h == 2:
            h += 0.208288
        elif n_h == 3:
            h += 0.369764; b += -0.022123; e += -0.102229; s += -0.055649
    elif t == "C.ar":
        h += 0.041270; e += -0.005210; s += -0.013906
        h += 0.194585; e += 0.117666; s += 0.070704
        if n_h == 0:
            h += 0.085378; b += 0.002903; e += 0.107527; s += 0.046501
        elif n_h == 1:
            e += -0.044799; s += -0.011864
    elif t == "C.cat":
        e += -0.005210; s += -0.013906
    elif t == "N.1":
        a += 0.077667; s += 0.167948
        b += 0.061824; s += 0.232745
    elif t == "N.2":
        a += 0.077667; s += 0.167948
        if n_h == 1:
            h += -0.200643
        else:
            b += -0.763354; e += 0.682859
    elif t == "N.3":
        a += 0.077667; s += 0.167948
        h += -0.091021; b += 0.367558; e += 0.099602; s += 0.084806
        if n_h == 0:
            h += -0.495588; b += 0.150946; e += 0.088502
        elif n_h == 1:
            h += -0.452520; a += 0.163329
        elif n_h == 2:
            h += -0.763440; a += 0.120801; s += -0.069707
    elif t == "N.4":
        a += 0.077667; s += 0.167948
        h += -4.382696
    elif t == "N.am":
        a += 0.077667; s += 0.167948
        h += -0.091021
        if n_h == 0:
            h += -0.495588; a += -0.096006
        elif n_h == 1:
            h += -0.452520
        elif n_h == 2:
            h += -0.763440
    elif t == "N.ar":
        a += 0.077667; s += 0.167948
        a += -0.104421; b += 0.282666
    elif t == "N.pl3":
        a += 0.077667; s += 0.167948
        h += 0.483597; b += -0.296874; e += 0.060327; s += -0.351281
    elif t == "O.2":
        h += -0.224286
        h += -0.098496; a += 0.038390; e += -0.011749; s += 0.371065
    elif t == "O.3":
        h += -0.224286
        h += -0.102250; s += 0.111253; b += 0.147962
        if n_h == 0:
            h += -0.010867; a += -0.035354; e += -0.020649
        else:
            h += -0.279404; a += 0.445333; b += 0.014953; e += 0.010690; s += 0.027754
    elif t == "O.co2":
        h += -0.224286
    elif t == "S.2":
        h += -0.077480
        h += 0.734821; e += 0.136585
    elif t == "S.3":
        e += 0.195277; s += 0.082278
        if n_h == 0:
            h += 0.504755; b += 0.103649; e += 0.071103
    elif t == "S.o":
        h += -0.077480
        h += -1.055756
    elif t == "S.o2":
        h += -0.077480
        h += 0.027576
    elif t == "P.3":
        b += 0.686142; e += 0.103799; s += 0.215383
    elif t == "F":
        h += 0.269012; b += -0.077060; e += -0.220887; s += -0.077487
    elif t == "Cl":
        h += 0.569605; b += -0.068040; e += -0.008902; s += 0.029923
    elif t == "Br":
        h += 0.696413; b += -0.074564; e += 0.167323; s += 0.131222
    elif t == "I":
        h += 0.323219; e += 0.509975; s += 0.135703
    elif t == "Si":
        h += 0.617895
    return {"hydrophobic": h, "hbd": a, "hba": b, "abraham_e": e, "abraham_s": s}
