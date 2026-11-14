#!/usr/bin/env python
"""
Build ``examples/enantiomers/*.sdf``: series in which both enantiomers are measured (item MC3).

Plan item C7 built a handedness descriptor (`cmfqsar/stereo.py`) and could not test it on a real
series, because none of the five sets of this repository holds an enantiomeric pair.  MC3 is that
missing data task.  What it needs is strict: two molecules that are mirror images of each other,
each with its own potency, and the two potencies measured **in the same assay**, so that the
difference between them is a property of the molecules and not of two laboratories.

This script finds every such pair in ChEMBL for a fixed list of targets.

The rules it applies
--------------------
1. *A pair is a pair of mirror images.*  The test inverts every tetrahedral centre of the first
   molecule and asks whether the result is the second one.  It does not read a compound name and
   it does not use the letters R and S, so a pair is never built from a label.  Diastereomers
   (one centre inverted out of two) are not pairs and are left out.
2. *The configuration comes from the record.*  A molecule whose structure in ChEMBL has a
   stereocentre without a configuration cannot be placed in space, so it is left out rather than
   guessed at.  The configuration of the built structure is read back from its coordinates and
   has to agree with the record, or the pair is dropped.
3. *One assay for both potencies.*  The two values of a pair come from one assay identifier and
   one measurement type (Ki, Kd, IC50 or EC50).  Where a pair was measured in several assays of
   the same target, the assay with the most records is used; a tie is settled by the earlier
   publication and then by the lower assay identifier.
4. *The two structures differ in handedness and in nothing else.*  One enantiomer is embedded in
   three dimensions and the other is its reflection, atom for atom, as plan item C7's synthetic
   pairs are built.  Two structures embedded separately would differ in their torsions as much as
   in their handedness, and a descriptor that is blind to handedness could then appear to tell
   them apart.  With an exact reflection the two members of a pair have identical rotation and
   translation invariant descriptions, so whatever a model does inside a pair comes from the
   handedness channels alone.
5. *No selection on the answer.*  Pairs are not filtered by how much the two potencies differ.
   A set built from the pairs that differ would make the question easier than it is.

Written tags
------------
``pAffinity`` (the potency of this enantiomer, -log10 of the value in M), ``pair`` (the
identifier of the pair: both members carry it), ``member`` (``first`` or ``second``), ``cip``
(the configuration read from the built coordinates), ``eudismic_index`` (the difference of the
two potencies, the same number on both members), ``assay``, ``assay_type``, ``assay_description``,
``document``, ``year``, ``target``, ``target_name``, ``centres`` (how many stereocentres the
molecule has) and ``chembl_id``.

Usage
-----
    python examples/make_enantiomer_examples.py             # fetch (about 60 MB) and build
    python examples/make_enantiomer_examples.py --offline   # build from the cache only
    python examples/make_enantiomer_examples.py --min-pairs 20

The download is free and needs no key.  ChEMBL data is CC BY-SA 3.0, so the built files may be
redistributed with attribution; the attribution travels with every record as the ChEMBL
identifiers of the molecule, the assay and the publication.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from cmfqsar.orca import geometry_faults  # noqa: E402
from make_denmark_examples import sf5_stand_in  # noqa: E402
from make_hammett_examples import regular_sf5  # noqa: E402

RDLogger.DisableLog("rdApp.*")

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT = os.path.join(HERE, "enantiomers")
CACHE = os.path.join(os.path.dirname(HERE), "benchmarks", "cache", "enantiomers")
#: the measurement types that are potencies on a comparable scale
TYPES = "Ki,Kd,IC50,EC50"
ONLY = ("molecule_chembl_id,canonical_smiles,assay_chembl_id,assay_description,"
        "standard_type,standard_units,standard_relation,pchembl_value,"
        "document_chembl_id,document_year,data_validity_comment")
#: a target needs this many pairs before a file is written for it
MIN_PAIRS = 10
SEED = 0xC0FFEE
#: how many embeddings are tried, with the seeds SEED, SEED + 1, ..., before a pair is refused
#: because no structure passes the geometry check
EMBED_TRIES = 5

#: The targets, named by their UniProt accession so that no ChEMBL identifier is guessed, and
#: fixed before any count was seen.  They are the large, well measured human single-protein
#: targets of the families in which chiral ligands are common.  The list is not chosen by the
#: result: a target that gives no pair stays in it and is reported with zero.
TARGETS = [
    ("P08588", "beta-1 adrenergic receptor", "aminergic GPCR"),
    ("P07550", "beta-2 adrenergic receptor", "aminergic GPCR"),
    ("P13945", "beta-3 adrenergic receptor", "aminergic GPCR"),
    ("P35348", "alpha-1A adrenergic receptor", "aminergic GPCR"),
    ("P08913", "alpha-2A adrenergic receptor", "aminergic GPCR"),
    ("P14416", "dopamine D2 receptor", "aminergic GPCR"),
    ("P35462", "dopamine D3 receptor", "aminergic GPCR"),
    ("P21917", "dopamine D4 receptor", "aminergic GPCR"),
    ("P08908", "serotonin 5-HT1A receptor", "aminergic GPCR"),
    ("P28223", "serotonin 5-HT2A receptor", "aminergic GPCR"),
    ("P28335", "serotonin 5-HT2C receptor", "aminergic GPCR"),
    ("P50406", "serotonin 5-HT6 receptor", "aminergic GPCR"),
    ("P34969", "serotonin 5-HT7 receptor", "aminergic GPCR"),
    ("P11229", "muscarinic M1 receptor", "aminergic GPCR"),
    ("P08172", "muscarinic M2 receptor", "aminergic GPCR"),
    ("P20309", "muscarinic M3 receptor", "aminergic GPCR"),
    ("P35367", "histamine H1 receptor", "aminergic GPCR"),
    ("Q9Y5N1", "histamine H3 receptor", "aminergic GPCR"),
    ("P35372", "mu opioid receptor", "peptide GPCR"),
    ("P41143", "delta opioid receptor", "peptide GPCR"),
    ("P41145", "kappa opioid receptor", "peptide GPCR"),
    ("P25103", "neurokinin NK1 receptor", "peptide GPCR"),
    ("P21554", "cannabinoid CB1 receptor", "lipid GPCR"),
    ("P34972", "cannabinoid CB2 receptor", "lipid GPCR"),
    ("P31645", "serotonin transporter", "transporter"),
    ("Q01959", "dopamine transporter", "transporter"),
    ("P23975", "noradrenaline transporter", "transporter"),
    ("P22303", "acetylcholinesterase", "hydrolase"),
    ("P06276", "butyrylcholinesterase", "hydrolase"),
    ("O00519", "fatty acid amide hydrolase", "hydrolase"),
    ("P12821", "angiotensin-converting enzyme", "protease"),
    ("P00797", "renin", "protease"),
    ("P00734", "thrombin", "protease"),
    ("P00742", "coagulation factor Xa", "protease"),
    ("P43235", "cathepsin K", "protease"),
    ("P56817", "beta-secretase 1", "protease"),
    ("P00918", "carbonic anhydrase 2", "lyase"),
    ("P11511", "aromatase (CYP19A1)", "oxidoreductase"),
    ("P21397", "monoamine oxidase A", "oxidoreductase"),
    ("P27338", "monoamine oxidase B", "oxidoreductase"),
    ("P28845", "11-beta-hydroxysteroid dehydrogenase 1", "oxidoreductase"),
    ("P04035", "HMG-CoA reductase", "oxidoreductase"),
    ("Q08499", "phosphodiesterase 4D", "hydrolase"),
    ("O76074", "phosphodiesterase 5A", "hydrolase"),
    ("P27487", "dipeptidyl peptidase 4", "protease"),
    ("P09917", "arachidonate 5-lipoxygenase", "oxidoreductase"),
    ("P35354", "cyclooxygenase 2", "oxidoreductase"),
    ("P24941", "cyclin-dependent kinase 2", "kinase"),
    ("P00533", "epidermal growth factor receptor", "kinase"),
    ("P31749", "protein kinase B (AKT1)", "kinase"),
    ("P11309", "PIM1 kinase", "kinase"),
    ("Q13464", "rho-associated kinase 1", "kinase"),
    ("P42336", "PI3-kinase alpha", "kinase"),
    ("P04150", "glucocorticoid receptor", "nuclear receptor"),
    ("P10275", "androgen receptor", "nuclear receptor"),
    ("P03372", "oestrogen receptor alpha", "nuclear receptor"),
    ("P37231", "PPAR gamma", "nuclear receptor"),
    ("P11473", "vitamin D receptor", "nuclear receptor"),
    ("Q12809", "hERG potassium channel", "ion channel"),
    ("Q05586", "NMDA receptor subunit 1", "ion channel"),
]

TAGS = {Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW}


# --------------------------------------------------------------------------- #
# The web service
# --------------------------------------------------------------------------- #
def get(path, **params):
    params.setdefault("format", "json")
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as fh:
                return json.load(fh)
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"   retrying after {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))


def page(path, **params):
    offset = 0
    while True:
        d = get(path, limit=1000, offset=offset, **params)
        key = next(k for k in d if k != "page_meta")
        rows = d[key]
        for r in rows:
            yield r
        if not rows or not d["page_meta"].get("next"):
            return
        offset += 1000


def resolve(accession):
    """The ChEMBL identifier of the human single protein with this UniProt accession."""
    d = get("target", target_components__accession=accession, target_type="SINGLE PROTEIN",
            organism="Homo sapiens", only="target_chembl_id,pref_name", limit=5)
    hits = d.get("targets", [])
    return (hits[0]["target_chembl_id"], hits[0].get("pref_name")) if hits else (None, None)


def resolve_all(offline=False):
    """The identifier of every target of the list, kept in the cache after the first run."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "targets.json")
    known = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            known = json.load(fh)
    if offline:
        return known
    for accession, name, _family in TARGETS:
        if accession in known:
            continue
        target, pref = resolve(accession)
        if target is None:
            print(f"!! no human single protein for {accession} ({name})")
            continue
        known[accession] = [target, pref]
        print(f"{accession} {name:42s} {target}", flush=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(known, fh, indent=1)
    return known


def complete(rows):
    """Does this download carry the fields the pairs are built from?

    A server fault during one fetch returned three thousand records of one target with the assay
    identifier missing from every one of them.  Rule 3 then had nothing to group by, and the
    target reported no pair at all -- a silent wrong answer, not an error.  A download in which
    no record names its assay is treated as faulty and fetched again.
    """
    if not rows:
        return True
    return any(r.get("assay_chembl_id") for r in rows)


def activities(target, offline=False):
    """Every binding measurement of the target, from the cache when it is there."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, target + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        if complete(rows):
            return rows
        print(f"   the cached download of {target} names no assay: fetching it again",
              flush=True)
        os.remove(path)
    if offline:
        return []
    rows = list(page("activity", target_chembl_id=target, assay_type="B",
                     standard_type__in=TYPES, pchembl_value__isnull="false", only=ONLY))
    if not complete(rows):
        raise SystemExit(f"the download of {target} names no assay in {len(rows)} records")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return rows


# --------------------------------------------------------------------------- #
# Is this a pair of mirror images?
# --------------------------------------------------------------------------- #
def parse(smiles):
    """The largest fragment of the record, with its configurations assigned.

    A record may be a salt, and ChEMBL holds the free base and the hydrochloride of the same
    compound under different identifiers.  Keeping the largest fragment makes the two the same
    molecule, so a pair is not counted twice; a compound of two comparable parts is changed by
    this and is the price of it.
    """
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    parts = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: p.GetNumHeavyAtoms())
        try:
            Chem.SanitizeMol(m)
        except Exception:
            return None
    Chem.AssignStereochemistry(m, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    return m


def flat(mol):
    """The canonical SMILES with every configuration taken away: the connectivity alone."""
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def centres(mol):
    """(centres with a configuration, centres whose configuration is missing)."""
    found = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    given = sum(1 for _, code in found if code in ("R", "S"))
    return given, len(found) - given


def mirror_smiles(mol):
    """The canonical SMILES of the mirror image: every tetrahedral centre inverted."""
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        tag = atom.GetChiralTag()
        if tag in TAGS:
            atom.SetChiralTag(TAGS[tag])
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return Chem.MolToSmiles(m)


def enantiomers(a, b):
    """The pair's description when `a` and `b` are mirror images of each other, else None."""
    ma, mb = parse(a), parse(b)
    if ma is None or mb is None:
        return None
    sa, sb = Chem.MolToSmiles(ma), Chem.MolToSmiles(mb)
    if sa == sb:
        return None                                   # the same molecule, not a pair
    given_a, missing_a = centres(ma)
    given_b, missing_b = centres(mb)
    if missing_a or missing_b or given_a == 0 or given_a != given_b:
        return None                                   # a configuration that is not given
    if mirror_smiles(ma) != sb:
        return None                                   # the same skeleton, but not mirror images
    return {"first": sa, "second": sb, "centres": given_a, "flat": flat(ma)}


def usable(r):
    """A record that can carry one potency of one pair.

    A record without an assay identifier cannot take part: rule 3 asks which assay the two
    potencies come from, and that question has no answer for such a record.
    """
    return bool(r.get("standard_relation") == "=" and not r.get("data_validity_comment")
                and r.get("standard_units") == "nM" and r.get("pchembl_value")
                and r.get("assay_chembl_id") and r.get("molecule_chembl_id")
                and r.get("standard_type"))


def pairs_of_target(rows):
    """Every enantiomeric pair of the target whose two potencies come from one assay."""
    by_assay = defaultdict(lambda: defaultdict(list))
    smiles, meta = {}, {}
    for r in rows:
        if not usable(r):
            continue
        key = (r["assay_chembl_id"], r["standard_type"])
        by_assay[key][r["molecule_chembl_id"]].append(float(r["pchembl_value"]))
        smiles.setdefault(r["molecule_chembl_id"], r.get("canonical_smiles"))
        meta.setdefault(key, r)
    found = []
    for key, mols in by_assay.items():
        groups = defaultdict(list)
        for m in mols:
            mol = parse(smiles.get(m) or "")
            if mol is None:
                continue
            given, missing = centres(mol)
            if given and not missing:
                groups[flat(mol)].append(m)
        for skeleton, members in groups.items():
            if len(members) < 2:
                continue
            members.sort()
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    got = enantiomers(smiles[a], smiles[b])
                    if not got:
                        continue
                    # which member is called "first" must not be a chemical property, or the
                    # sign of the difference between the two potencies would be readable from
                    # something other than the handedness.  The two canonical SMILES in
                    # alphabetical order decide it: the same answer on every machine, and
                    # nothing a rotation-invariant description of the molecule can see.
                    one, two = (a, b) if got["first"] <= got["second"] else (b, a)
                    first_smiles = min(got["first"], got["second"])
                    second_smiles = max(got["first"], got["second"])
                    r = meta[key]
                    found.append({
                        "assay": key[0], "assay_type": key[1],
                        "description": (r.get("assay_description") or "")[:250],
                        "document": r.get("document_chembl_id"),
                        "year": int(r.get("document_year") or 0),
                        "first": one, "second": two,
                        "first_smiles": first_smiles, "second_smiles": second_smiles,
                        "first_p": float(np.mean(mols[one])),
                        "second_p": float(np.mean(mols[two])),
                        "records": len(mols[one]) + len(mols[two]),
                        "centres": got["centres"], "skeleton": skeleton})
    return one_assay_per_pair(found)


def one_assay_per_pair(found):
    """Rule 3: one assay for each pair, the one with the most records.

    The pair is recognised by its two structures and not by its identifiers, so the free base
    and the salt of the same pair are one pair here.
    """
    best = {}
    for p in found:
        key = (p["first_smiles"], p["second_smiles"])
        rank = (-p["records"], p["year"] or 9999, p["assay"])
        if key not in best or rank < best[key][0]:
            best[key] = (rank, p)
    return [p for _rank, p in sorted(best.values(), key=lambda t: t[0])]


# --------------------------------------------------------------------------- #
# The two structures: one embedding and its reflection
# --------------------------------------------------------------------------- #
def embed(smiles, seed=SEED):
    """One structure of the molecule that passes the geometry check, or None.

    The structure is an ETKDG embedding refined with MMFF94.  Two things can make it a structure
    that is not the molecule, and neither shows in a model score, so both are handled here.

    * **An SF5 group.**  MMFF94 has no parameters for hexavalent sulfur, so the refinement did
      nothing and the file held the raw embedding, with S-F lengths from 1.23 to 2.49 A where the
      group has 1.59 and with two fluorines 1.0 A apart.  The molecule is now embedded and
      refined with CF3 in place of the group (``make_denmark_examples.sf5_stand_in``) and the
      group is then set to a regular octahedron (``make_hammett_examples.regular_sf5``).
    * **A refinement that stops at a wrong stationary point.**  An embedding sometimes puts the
      two carbons of a secondary amine on one line with the nitrogen.  The force on the
      hydrogen is then zero by symmetry and MMFF94 reports convergence with a C-N-C angle of
      180 degrees and an N-H length of 1.34 A.  The structure is checked
      (:func:`cmfqsar.orca.geometry_faults`) and, if it fails, embedded again with the next
      seed, up to ``EMBED_TRIES`` times.

    A structure that needed either step says so in its ``geometry_note`` property.  The files
    written before 14 November 2026 held four SF5 pairs and one pair with a linear nitrogen.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    m = Chem.AddHs(mol)
    stand_in, kept = sf5_stand_in(m)
    first_fault = None
    for attempt in range(EMBED_TRIES):
        work = Chem.Mol(m if stand_in is None else stand_in)
        params = AllChem.ETKDGv3()
        params.randomSeed = (int(seed) + attempt) & 0x7FFFFFFF
        if AllChem.EmbedMolecule(work, params) != 0:
            if attempt == 0:
                return None
            continue
        try:
            AllChem.MMFFOptimizeMolecule(work, maxIters=500)
        except Exception:
            pass
        if stand_in is None:
            out = work
        else:
            # the atoms of the stand-in on the atoms of the molecule; the sulfur and the five
            # fluorines are placed by `regular_sf5`, which reads the ring atoms only
            out = Chem.Mol(m)
            pos = np.zeros((m.GetNumAtoms(), 3))
            pos[kept] = np.array(work.GetConformer().GetPositions(), float)
            conf = Chem.Conformer(m.GetNumAtoms())
            for atom, p in enumerate(pos):
                conf.SetAtomPosition(atom, [float(v) for v in p])
            out.RemoveAllConformers()
            out.AddConformer(conf, assignId=True)
        # MMFF set-up can change the aromaticity flags in place, so the molecule is read again
        Chem.SanitizeMol(out)
        notes = []
        if stand_in is not None:
            regular_sf5(out)
            notes.append("refined with CF3 in place of SF5, the SF5 group then set to a "
                         "regular octahedron")
        faults = geometry_faults(out)
        if not faults:
            if attempt:
                notes.append(f"embedding seed + {attempt}; the first embedding gave "
                             f"{first_fault}")
            if notes:
                out.SetProp("geometry_note", "; ".join(notes))
            return out
        first_fault = first_fault or ", ".join(faults)
    return None


def reflected(mol):
    """The exact enantiomer: the same conformation reflected, not a second embedding.

    Rule 4.  The internal geometry is untouched, so the two members of a pair differ in their
    handedness and in nothing else; every rotation and translation invariant description of the
    two is identical.
    """
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, p in enumerate(np.array(conf.GetPositions()) * np.array([1.0, 1.0, -1.0])):
        conf.SetAtomPosition(i, [float(v) for v in p])
    Chem.AssignStereochemistryFrom3D(m)
    return m


def cip_codes(mol):
    """The configuration of every centre, read from the coordinates."""
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(probe)
    return tuple(sorted((a.GetIdx(), a.GetProp("_CIPCode")) for a in probe.GetAtoms()
                        if a.HasProp("_CIPCode")))


def cip_of_smiles(smiles):
    m = parse(smiles)
    found = Chem.FindMolChiralCenters(m, includeUnassigned=False,
                                      useLegacyImplementation=False)
    return tuple(sorted(found))


def build_pair(p, seed=SEED):
    """The two structures of one pair, or None with the reason it could not be built."""
    first = embed(p["first_smiles"], seed)
    if first is None:
        return None, "the first enantiomer could not be embedded"
    second = reflected(first)
    want_first = cip_of_smiles(p["first_smiles"])
    want_second = cip_of_smiles(p["second_smiles"])
    got_first = cip_codes(first)
    got_second = cip_codes(second)
    if got_first != want_first:
        return None, "the built structure does not have the configuration of the record"
    if got_second != want_second:
        return None, "the reflection does not have the configuration of the second record"
    return (first, second), None


def write_set(path, pairs, built, target, name):
    """One file: both members of every pair, each with its own potency."""
    writer = Chem.SDWriter(path)
    n = 0
    for p, (first, second) in zip(pairs, built):
        pair_id = f"{p['first']}_{p['second']}"
        index = p["first_p"] - p["second_p"]
        for member, mol, chembl, value in (("first", first, p["first"], p["first_p"]),
                                           ("second", second, p["second"], p["second_p"])):
            m = Chem.Mol(mol)
            m.SetProp("_Name", f"{pair_id}-{member}")
            m.SetProp("pAffinity", f"{value:.3f}")
            m.SetProp("pair", pair_id)
            m.SetProp("member", member)
            m.SetProp("cip", ",".join(c for _i, c in cip_codes(mol)))
            m.SetProp("eudismic_index", f"{index:+.3f}")
            m.SetProp("assay", str(p["assay"] or ""))
            m.SetProp("assay_type", str(p["assay_type"] or ""))
            m.SetProp("assay_description", str(p["description"] or ""))
            m.SetProp("document", str(p["document"] or ""))
            m.SetProp("year", str(p["year"] or ""))
            m.SetProp("target", str(target))
            m.SetProp("target_name", str(name))
            m.SetProp("centres", str(p["centres"]))
            m.SetProp("chembl_id", str(chembl))
            writer.write(m)
            n += 1
    writer.close()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--offline", action="store_true", help="build from the cache, download nothing")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS,
                    help=f"a target with fewer pairs gets no file (default {MIN_PAIRS})")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=SEED)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    known = resolve_all(args.offline)
    summary, written = [], 0
    for accession, name, family in TARGETS:
        if accession not in known:
            print(f"{name}: no identifier")
            continue
        target, pref = known[accession]
        rows = activities(target, offline=args.offline)
        pairs = pairs_of_target(rows)
        built, kept, refused = [], [], defaultdict(int)
        for p in pairs:
            pair, why = build_pair(p, args.seed)
            if pair is None:
                refused[why] += 1
                continue
            built.append(pair)
            kept.append(p)
        row = {"accession": accession, "name": name, "family": family, "target": target,
               "chembl_name": pref, "records": len(rows), "pairs": len(pairs),
               "built": len(kept), "one_centre": sum(1 for p in kept if p["centres"] == 1),
               "refused": dict(refused)}
        path = os.path.join(OUT, name.split(" (")[0].replace(" ", "_") + ".sdf")
        if len(kept) >= args.min_pairs:
            row["file"] = os.path.basename(path)
            row["molecules"] = write_set(path, kept, built, target, name)
            written += 1
        elif os.path.exists(path):
            # a file left by an earlier run, when this target had enough pairs or when a run
            # stopped while writing: it would otherwise be read as a set of its own
            os.remove(path)
            print(f"   removed the stale {os.path.basename(path)}", flush=True)
        summary.append(row)
        print(f"{name:42s} {len(rows):7d} records {len(pairs):4d} pairs "
              f"{len(kept):4d} built {row.get('file', '')}", flush=True)

    with open(os.path.join(OUT, "pairs_per_target.json"), "w", encoding="utf-8") as fh:
        json.dump({"targets": summary, "min_pairs": args.min_pairs,
                   "source": "ChEMBL web service, CC BY-SA 3.0"}, fh, indent=1)
    total = sum(r["built"] for r in summary)
    print(f"\n{total} pairs built over {len(summary)} targets; {written} files written to {OUT}")


if __name__ == "__main__":
    main()
