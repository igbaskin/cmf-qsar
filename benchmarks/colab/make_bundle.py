#!/usr/bin/env python
"""
Pack the files that a simulation on another machine needs into one zip file.

The zip holds the package ``cmfqsar``, ``benchmarks/run_diffusion.py``, the runner
``benchmarks/colab/colab_transport.py`` and the list of formulations
``examples/electrolyte/formulations.csv``, in the folder layout of this repository, so that the
benchmark finds its files there. It also holds ``bundle_info.json`` with the commit that the
files were taken from and the number of files that differ from it, so that a box can be traced
to the code that made it.

Nothing is uploaded. You copy the zip to Google Drive yourself.

Usage
-----
    python benchmarks/colab/make_bundle.py
    python benchmarks/colab/make_bundle.py --out my_bundle.zip

The default output is ``benchmarks/cache/colab/cmf_colab_bundle.zip``. That folder is ignored by
git.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import zipfile
from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SINGLE_FILES = ("benchmarks/run_diffusion.py", "benchmarks/colab/colab_transport.py",
                "examples/electrolyte/formulations.csv")
PACKAGE = "cmfqsar"
DEFAULT_OUT = os.path.join(ROOT, "benchmarks", "cache", "colab", "cmf_colab_bundle.zip")


def bundle_files(root: str = ROOT) -> List[Tuple[str, str]]:
    """(name inside the zip, path on disk) for every file of the bundle, sorted by name."""
    found = []
    for rel in SINGLE_FILES:
        path = os.path.join(root, *rel.split("/"))
        if not os.path.isfile(path):
            raise SystemExit(f"the file {rel} is missing in {root}")
        found.append((rel, path))
    for folder, dirs, files in os.walk(os.path.join(root, PACKAGE)):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(folder, f)
            found.append((os.path.relpath(path, root).replace(os.sep, "/"), path))
    return sorted(found)


def git_info(root: str = ROOT) -> dict:
    """The commit and the number of changed files among the bundled ones; empty values when git
    is not available."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        changed = [line for line in git("status", "--porcelain", "--", PACKAGE,
                                        *SINGLE_FILES).splitlines() if line.strip()]
        return {"commit": commit, "changed_files": len(changed)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "", "changed_files": None}


def make_bundle(out: str = DEFAULT_OUT, root: str = ROOT) -> Tuple[str, int, int]:
    """Write the zip; returns (path, number of files, size in bytes)."""
    files = bundle_files(root)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    info = dict(git_info(root), created=time.strftime("%Y-%m-%d %H:%M:%S"), files=len(files))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in files:
            z.write(path, name)
        z.writestr("bundle_info.json", json.dumps(info, indent=1))
    return out, len(files), os.path.getsize(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="the zip file to write")
    args = ap.parse_args(argv)
    path, n, size = make_bundle(args.out)
    info = git_info()
    print(f"{path}: {n} files, {size / 1e6:.1f} MB")
    if info.get("changed_files"):
        print(f"note: {info['changed_files']} of the bundled files differ from commit "
              f"{info['commit'][:8]}; the zip holds the working copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
