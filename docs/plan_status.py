#!/usr/bin/env python
"""
The status of every item of the development plan, read from the entries themselves.

The Sequence table of ``docs/development_plan.md`` used to be written by hand and had drifted: 29
items whose own entries recorded a completion date were still listed there as pending, so the
table anyone reads first disagreed with the entries it summarises.  This script reads the entries
and prints the table, so that it can be regenerated instead of maintained.

The convention it relies on is the one every finished entry in the file already uses: an item
records its completion in bold with a date -- "Done 27 October 2026", "Built 27 October 2026",
"measured 1 November 2026".  Two things make a plain search for that unreliable, and both are
handled here:

* the last item of a section is followed by the section's **Order.** paragraph, which names every
  other item's completion date; an item's text therefore stops at the next item *or* at that
  paragraph, whichever comes first.  Without this an untouched item inherits its neighbours' dates
  (EC10, a GUI item nobody has started, came out "finished" that way);
* five entries do not follow the convention, and they are listed in :data:`OVERRIDES` with what
  their own text says.  That list is worth as much as the table: it names the entries a reader
  cannot judge from their first sentence.

Usage::

    python docs/plan_status.py              # the Sequence table, ready to paste
    python docs/plan_status.py items        # every item, one per line
    python docs/plan_status.py open         # only what is not finished
"""
from __future__ import annotations

import io
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "development_plan.md")

ITEM = re.compile(r"\* \*\*([A-Z]{1,3}\d+[a-z]?)\. ")
MONTH = ("(?:January|February|March|April|May|June|July|August|September|October|November|"
         "December)")
#: a completion statement: a verb of finishing and a date, in the same sentence
FINISH = re.compile(r"(?:Done|done|Built|built|Measured|measured|Assessed|assessed)"
                    r"[^.\n]{0,70}?\d{1,2} " + MONTH + r" \d{4}")
#: where an item's own text ends, besides the next item
STOP = re.compile(r"^(?:\*\*Order\.\*\*|#{2,3} |\*\*(?:Small|Medium|Large) )")

#: the entries that do not follow the convention, and what their own text says
OVERRIDES: Dict[str, Tuple[str, str]] = {
    "E11": ("partly", "version (i) of four built and measured 31 October 2026"),
    "EC8": ("partly", "pilot measured 1 November 2026, without the cut step and without M5"),
    "SC5": ("partly", "the radii of EC7 are done; the frame convention and the data are left"),
    "M15": ("done", "both halves measured 7 November 2026: the row length on five series, "
                    "the weighted distance over the columns on three"),
    "NT8": ("partly", "the cell outline came with M13 and the bonds on 7 November 2026; "
                      "instanced drawing for ten thousand atoms is left"),
    "PBH1": ("done", "the title records it: done on COMPAS-1x"),
    "PBH2": ("done", "the title records it: replaced by a geometric field and calibrated"),
}

#: the section every item prefix belongs to, in the order the table lists them
SECTIONS = OrderedDict([
    ("A", "A. The platform"), ("B", "B. Scale and speed"), ("C", "C. The alignment problem"),
    ("D", "D. Fields and kernels"), ("E", "E. Learning"), ("F", "F. Interpretation"),
    ("G", "G. The application and the tool"),
    ("M", "H. Materials and periodicity"), ("NT", "H. Nanotechnology"),
    ("EC", "H. Electrochemistry"), ("BT", "H. Battery science"),
    ("LP", "H. Local properties"), ("SC", "H. Supramolecular and host-guest"),
    ("PBH", "H. Physical organic"), ("ND", "H. Deep learning"),
    ("MS", "H. Multiscale modelling"),
    ("MC", "I. Medicinal chemistry"),
])
#: prefixes that belong to a section listed under another prefix
ALIASES = {"PO": "PBH"}       # PO1, the catalyst series, is in the physical organic chemistry section


def statuses(path: str = PLAN) -> "OrderedDict[str, str]":
    """Every item of the plan, in file order, with "done", "partly" or "open"."""
    lines = io.open(path, encoding="utf-8").read().splitlines()
    starts = [(i, m.group(1)) for i, line in enumerate(lines) for m in [ITEM.match(line)] if m]
    out: "OrderedDict[str, str]" = OrderedDict()
    for k, (i, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        for j in range(i + 1, end):
            if STOP.match(lines[j]):
                end = j
                break
        body = " ".join(line.strip() for line in lines[i:end])
        out[name] = "done" if FINISH.search(body) else "open"
        if name in OVERRIDES:
            out[name] = OVERRIDES[name][0]
    return out


def _sort_key(name: str):
    m = re.match(r"[A-Z]{1,3}(\d+)([a-z]?)", name)
    return (int(m.group(1)), m.group(2))


def table(status: "OrderedDict[str, str]") -> List[str]:
    groups: "OrderedDict[str, List[str]]" = OrderedDict((k, []) for k in SECTIONS)
    for name in status:
        key = re.match(r"([A-Z]{1,3})", name).group(1)
        key = ALIASES.get(key, key)
        if key not in groups:
            raise SystemExit(f"{name} belongs to no section listed in SECTIONS")
        groups[key].append(name)
    rows = ["| Part | Items | Finished | Not finished |", "|---|---|---|---|"]
    for key, label in SECTIONS.items():
        names = sorted(groups[key], key=_sort_key)
        open_names = [n for n in names if status[n] != "done"]
        left = [n + (" (partly)" if status[n] == "partly" else "") for n in open_names]
        # a whole part with nothing finished is written as a run, which is what a reader needs:
        # "ND1 to ND13" says more than sixteen names in a row
        if open_names == names and len(names) > 3:
            left = [f"{names[0]} to {names[-1]}"]
        rows.append(f"| {label} | {len(names)} | {len(names) - len(open_names)} | "
                    f"{', '.join(left) if left else '--'} |")
    done = sum(1 for v in status.values() if v == "done")
    partly = sum(1 for v in status.values() if v == "partly")
    rows += ["", f"**{len(status)} items: {done} finished, {partly} partly, "
                 f"{len(status) - done - partly} open.**"]
    return rows


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    status = statuses()
    what = argv[0] if argv else "table"
    if what == "table":
        print("\n".join(table(status)))
        print("\noverrides used, with what the entry says:")
        for name, (state, why) in OVERRIDES.items():
            print(f"  {name}: {state} -- {why}")
        return
    for name, state in status.items():
        if what == "open" and state == "done":
            continue
        print(f"{name:<6} {state}")


if __name__ == "__main__":
    main()
