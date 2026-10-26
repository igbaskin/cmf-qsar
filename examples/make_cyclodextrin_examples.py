"""
Guest sets for a fixed cyclodextrin host, from the curated association-constant database.

    Tahil G., Delorme F., Le Berre D., Monflier E., Sayede A., Tilloy S.
    "Curated dataset of association constants between a cyclodextrin and a guest for machine
    learning", Chemical Data Collections 45 (2023) 101022; data at
    https://doi.org/10.5281/zenodo.7575539, Open Data Commons Open Database Licence (ODbL 1.0).

Why this set is here
--------------------
Every other series under ``examples/`` is a congeneric set that shares a scaffold, which is how the
one hard requirement of CMF -- a common frame -- is usually met.  A cyclodextrin guest set is the
opposite case, and it is here precisely for that: the *host* is fixed and the guests are whatever
the literature happened to measure, from acetonitrile to steroids, so no scaffold is shared and no
superposition is defined.  What replaces the frame is the host cavity, which every guest enters,
and the kernel that needs no frame at all, :mod:`cmfqsar.invariant` (``--kernel autocorrelation``).
The alignment benchmark says the invariant kernel is the better choice exactly where no curated
superposition exists; here there is none available, so this is a test of that claim on a property
that is not receptor binding.

The published field-based models of this endpoint are the comparison: CoMFA with COSMOsar3D fields
on 60 alpha-CD guests, GRIND on 126 and on 233 beta-CD guests.  The tables written here are an
order of magnitude larger than those, because the curated database is.

What is kept and what is dropped
--------------------------------
The upstream ``CDEnrichedData.csv`` is the authors' curated table: 3459 measurements over 15 hosts,
each with the guest name, its PubChem CID, the isomeric SMILES of guest and host, pH, temperature,
the association constant K, DeltaG, and the literature reference.  From it this script keeps

* the rows inside one window of conditions -- ``--t-window`` around 298 K and ``--ph-window``
  around pH 7 by default -- so that a file is one host at one condition rather than a mixture of
  thermodynamic states;
* one row per guest, the **median** of log10 K over the measurements that survive, with their
  count and spread in their own columns.  The spread is almost always zero (a median of 0.00 to 0.01
  log units over the repeated guests of the three natural cyclodextrins) but not always: one
  alpha-CD guest carries 3.2 log units between two references, and the ``spread`` column is there
  to be filtered on before anything is published from these numbers;
* guests whose SMILES RDKit accepts.

Nothing is re-measured, re-derived or re-scaled: the activity is log10 of the upstream K, which the
authors had themselves reconciled from values published as K, as log K or as DeltaG.  The
stoichiometry is taken to be 1:1, as a single constant per row implies; the upstream table does not
state it row by row.

The guests at the pH they were measured at
------------------------------------------
The SMILES above are the neutral forms as PubChem deposits them, so a carboxylic acid whose
constant was measured at pH 7, where it is in fact an anion, is written protonated.  For a kernel
whose electrostatic field is one of five that is not a detail.  ``--protonate`` writes a second
copy of every table with the structures in their state at the pH of the window, the way
``examples/corrosion_hcl/`` carries the corrosion series protonated for 1 M HCl: Dimorphite-DL
decides site by site from its SMARTS pKa rules, the protonated SMILES go into the ``smiles``
column the program reads, and the neutral ones stay in ``smiles_neutral`` beside the net
``charge``.

The one setting that differs from the corrosion export is the cap on added protons.  There it is
one, because at pH 0 the rules charge every ring nitrogen in a molecule while in truth the first
protonation makes the second site far less basic.  At pH 7 only sites whose rule pKa is above 7
are charged, which is the justified case rather than the problematic one: putrescine really is
doubly protonated at pH 7 (pKa 9.0 and 10.8) and the rules say so.  So ``--max-protons`` defaults
to no cap here, and the corrosion behaviour is available by setting it.

Outputs (``examples/cyclodextrin/``)
------------------------------------
    <host>.csv      one file per host with at least ``--min-guests`` guests: an activity table
                    ``smiles,name,activity,...`` that ``cmf_cli.py`` and the GUI read directly,
                    with activity = log10 K.  The program builds the 3D conformer itself.
    hosts.csv       every host in the database, its counts before and after the window, and its
                    SMILES, so that what was left out stays visible.
    ph<pH>/<host>.csv   with ``--protonate``: the same tables and the same activities, structures
                    in their state at that pH, plus the ``smiles_neutral`` and ``charge`` columns.

Usage
-----
    python examples/make_cyclodextrin_examples.py --fetch
    python examples/make_cyclodextrin_examples.py --protonate
    python examples/make_cyclodextrin_examples.py --data-root <folder holding CDEnrichedData.csv>

``--fetch`` downloads the 489 kB archive from Zenodo into ``--data-root`` and checks its MD5
against the digest the record publishes, then extracts the two CSVs and the licence.  The copy in
the repository was made that way; the flag is here so that the folder can be rebuilt from nothing.

Then, for instance

    python cmf_cli.py build examples/cyclodextrin/beta_cyclodextrin.csv
        --kernel autocorrelation --lams 1e-2,0.1,1 --test-fraction 0.25
"""
import argparse
import csv
import hashlib
import math
import os
import statistics
import sys
import urllib.request
import zipfile

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ZENODO_DOI = "10.5281/zenodo.7575539"
ZENODO_URL = "https://zenodo.org/api/records/7575539/files/Data.zip/content"
ZENODO_MD5 = "9260a9a8a86db1da967d91a22ece8cb4"
ZENODO_BYTES = 488965

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, "cyclodextrin", "upstream")
DEFAULT_OUT = os.path.join(HERE, "cyclodextrin")

#: added after the other columns when the structures are protonated, as the corrosion export does
PROTONATION_COLUMNS = ["smiles_neutral", "charge"]

#: file name for the hosts named in the database; any other host is named by slugify() below
SLUGS = {
    "alpha-cyclodextrin": "alpha_cyclodextrin",
    "beta-cyclodextrin": "beta_cyclodextrin",
    "gamma-cyclodextrin": "gamma_cyclodextrin",
    "hp-beta-cd": "hydroxypropyl_beta_cyclodextrin",
    "sulfobutylether-beta-cd": "sulfobutylether_beta_cyclodextrin",
    "methyl beta-cyclodextrin": "methyl_beta_cyclodextrin",
    "2,6-Di-O-methyl-beta-cyclodextrin": "dimethyl_beta_cyclodextrin",
}


def slugify(name):
    s = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:48]


def fetch_upstream(root):
    """Download the Zenodo archive into `root`, verify its MD5, extract the CSVs and the licence."""
    os.makedirs(root, exist_ok=True)
    print("downloading " + ZENODO_URL)
    with urllib.request.urlopen(ZENODO_URL) as fh:
        blob = fh.read()
    got = hashlib.md5(blob).hexdigest()
    if got != ZENODO_MD5:
        raise SystemExit("MD5 mismatch: expected %s, got %s (%d bytes)"
                         % (ZENODO_MD5, got, len(blob)))
    print("  %d bytes, MD5 %s as published" % (len(blob), got))
    archive = os.path.join(root, "Data.zip")
    with open(archive, "wb") as fh:
        fh.write(blob)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if not base:
                continue
            with zf.open(member) as src, open(os.path.join(root, base), "wb") as dst:
                dst.write(src.read())
            print("  extracted " + base)
    os.remove(archive)


def in_window(row, t_centre, t_window, ph_centre, ph_window):
    try:
        t = float(row["T"])
        ph = float(row["pH"])
    except (TypeError, ValueError):
        return False
    return abs(t - t_centre) <= t_window and abs(ph - ph_centre) <= ph_window


def main():
    ap = argparse.ArgumentParser(
        description="Build cyclodextrin guest activity tables from the curated database.")
    ap.add_argument("--data-root", default=DEFAULT_ROOT,
                    help="folder holding CDEnrichedData.csv "
                         "(default: examples/cyclodextrin/upstream)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="where the activity tables are written")
    ap.add_argument("--fetch", action="store_true",
                    help="download the archive from Zenodo into --data-root first")
    ap.add_argument("--temperature", type=float, default=298.0, help="centre of the T window, K")
    ap.add_argument("--t-window", type=float, default=2.0, help="half-width of the T window, K")
    ap.add_argument("--ph", type=float, default=7.0, help="centre of the pH window")
    ap.add_argument("--ph-window", type=float, default=0.5, help="half-width of the pH window")
    ap.add_argument("--min-guests", type=int, default=20,
                    help="hosts with fewer guests than this get no file of their own")
    ap.add_argument("--protonate", nargs="?", type=float, const=None, default=False,
                    metavar="PH",
                    help="also write the tables with the structures in their protonation state "
                         "at this pH (default: the centre of the pH window) into a ph<PH> "
                         "subfolder; needs Dimorphite-DL")
    ap.add_argument("--max-protons", type=int, default=None,
                    help="cap on the protons added per molecule (default: no cap, unlike the "
                         "corrosion export, which caps at one for pH 0)")
    args = ap.parse_args()

    protonator, ph_dir = None, None
    if args.protonate is not False:
        ph = args.ph if args.protonate is None else args.protonate
        sys.path.insert(0, HERE)
        from scaffold_tree_export import Protonator      # the sibling example script
        protonator = Protonator(ph=ph, max_protons=args.max_protons)
        ph_dir = os.path.join(args.out_dir, ("ph%g" % ph).replace(".", "_"))
        os.makedirs(ph_dir, exist_ok=True)
        print("protonating at pH %g with Dimorphite-DL, %s" %
              (ph, "no cap on added protons" if args.max_protons is None
               else "at most %d proton(s) added" % args.max_protons))

    if args.fetch:
        fetch_upstream(args.data_root)

    src = os.path.join(args.data_root, "CDEnrichedData.csv")
    if not os.path.exists(src):
        raise SystemExit("%s not found; run with --fetch, or point --data-root at the extracted "
                         "Zenodo archive (%s)" % (src, ZENODO_DOI))
    with open(src, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    os.makedirs(args.out_dir, exist_ok=True)

    hosts = {}
    for r in rows:
        hosts.setdefault(r["Host"], []).append(r)

    summary, proto_totals = [], {}
    for host, hrows in sorted(hosts.items(), key=lambda kv: -len(kv[1])):
        win = [r for r in hrows
               if in_window(r, args.temperature, args.t_window, args.ph, args.ph_window)]
        by_guest, rejected = {}, 0
        for r in win:
            smi = r["IsomericSMILES"].strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                rejected += 1
                continue
            try:
                k = float(r["K"])
            except (TypeError, ValueError):
                rejected += 1
                continue
            if k <= 0:
                rejected += 1
                continue
            by_guest.setdefault(Chem.MolToSmiles(mol), []).append((math.log10(k), r))

        host_smiles = ""
        for r in hrows:
            if r["IsomericSMILES_Host"].strip():
                host_smiles = r["IsomericSMILES_Host"].strip()
                break
        entry = {"host": host, "rows": len(hrows), "rows_in_window": len(win),
                 "guests": len(by_guest), "rejected": rejected, "file": "",
                 "host_smiles": host_smiles}
        summary.append(entry)
        if len(by_guest) < args.min_guests:
            continue

        path = os.path.join(args.out_dir, (SLUGS.get(host) or slugify(host)) + ".csv")
        entry["file"] = os.path.basename(path)

        used = {}          # names must be unique within a file: the table is keyed on them
        out = []
        for smi, vals in by_guest.items():
            logks = [v for v, _ in vals]
            ref = vals[0][1]
            name = " ".join((ref["Guest"] or "").split()) or ("CID" + str(ref["CID_Guest"]))
            seen = used.get(name, 0)
            used[name] = seen + 1
            if seen:
                name = "%s [CID%s]" % (name, ref["CID_Guest"])
                if name in used:
                    name = "%s (%d)" % (name, seen + 1)
                used[name] = 1
            median = statistics.median(logks)
            out.append({
                "smiles": smi,
                "name": name,
                "activity": "%.4f" % median,
                "logK": "%.4f" % median,
                "n_values": len(logks),
                "spread": "%.4f" % (max(logks) - min(logks)),
                "K_median": "%.6g" % (10.0 ** median),
                "cid": ref["CID_Guest"],
                "pH": ref["pH"],
                "T": ref["T"],
                "host": host,
                "host_cid": ref["CID_Host"],
                "reference": " ".join((ref["Reference"] or "").split())[:300],
            })
        out.sort(key=lambda d: d["name"].lower())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        logks = [float(d["logK"]) for d in out]
        print("%-44s %5d guests  log K %.2f to %.2f" % (os.path.basename(path), len(out),
                                                        min(logks), max(logks)))

        if protonator is None:
            continue
        rows_p, changed, charges = [], 0, {}
        for d in out:
            try:
                smi_p = protonator(d["smiles"])
            except ValueError:
                smi_p = d["smiles"]
            mol_p = Chem.MolFromSmiles(smi_p)
            if mol_p is None:                      # keep the guest, in the form that parses
                smi_p, mol_p = d["smiles"], Chem.MolFromSmiles(d["smiles"])
            q = sum(a.GetFormalCharge() for a in mol_p.GetAtoms())
            charges[q] = charges.get(q, 0) + 1
            if smi_p != d["smiles"]:
                changed += 1
            row = dict(d)
            row["smiles"] = smi_p
            row["smiles_neutral"] = d["smiles"]
            row["charge"] = q
            rows_p.append(row)
        path_p = os.path.join(ph_dir, os.path.basename(path))
        with open(path_p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()) + PROTONATION_COLUMNS)
            w.writeheader()
            w.writerows(rows_p)
        spread = ", ".join("%+d: %d" % (q, n) for q, n in sorted(charges.items()))
        print("%-44s %5d changed, net charge %s"
              % ("  " + os.path.basename(ph_dir) + "/" + os.path.basename(path_p),
                 changed, spread))
        proto_totals["changed"] = proto_totals.get("changed", 0) + changed
        proto_totals["guests"] = proto_totals.get("guests", 0) + len(rows_p)
        for q, n in charges.items():
            proto_totals[q] = proto_totals.get(q, 0) + n

    hosts_path = os.path.join(args.out_dir, "hosts.csv")
    with open(hosts_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["host", "rows", "rows_in_window", "guests", "rejected",
                                          "file", "host_smiles"])
        w.writeheader()
        w.writerows(summary)
    kept = sum(1 for s in summary if s["file"])
    print("\n%s: %d hosts, %d with a file of their own; window T = %g +/- %g K, pH = %g +/- %g"
          % (hosts_path, len(summary), kept, args.temperature, args.t_window,
             args.ph, args.ph_window))
    if protonator is not None:
        qs = sorted(q for q in proto_totals if isinstance(q, int))
        print("%s: %d of %d guests changed; net charge %s"
              % (ph_dir, proto_totals.get("changed", 0), proto_totals.get("guests", 0),
                 ", ".join("%+d: %d" % (q, proto_totals[q]) for q in qs)))
        if protonator.unmapped:
            print("  %d guests whose capped state could not be mapped back to the input; "
                  "left at the first state within the cap" % len(protonator.unmapped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
