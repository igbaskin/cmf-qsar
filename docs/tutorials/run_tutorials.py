#!/usr/bin/env python
"""
Execute the tutorial notebooks in this folder, so that their committed outputs come from the
current code, or check that they still run.

    python docs/tutorials/run_tutorials.py            # execute every notebook in place
    python docs/tutorials/run_tutorials.py --check    # execute copies; exit 1 if one fails
    python docs/tutorials/run_tutorials.py 02         # only the notebooks whose name starts so

The notebooks are run from the repository root, which is where their paths point. Executing
needs the ``jupyter`` package (``pip install cmfqsar[tutorials]``); the 3D views need
``py3Dmol`` and are skipped with a message when it is absent.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def notebooks(prefixes):
    names = sorted(f for f in os.listdir(HERE) if f.endswith(".ipynb"))
    if prefixes:
        names = [n for n in names if any(n.startswith(p) for p in prefixes)]
    return [os.path.join(HERE, n) for n in names]


def run(path: str, write: bool, timeout: int) -> bool:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": ROOT}})
    t0 = time.time()
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001 - the notebook's own error is what matters
        print(f"FAILED  {os.path.basename(path)}  after {time.time() - t0:.0f} s")
        print(str(exc)[-3000:])
        return False
    if write:
        nbformat.write(nb, path)
    print(f"ok      {os.path.basename(path)}  {time.time() - t0:.0f} s")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prefixes", nargs="*", help="run only notebooks whose file name starts so")
    ap.add_argument("--check", action="store_true",
                    help="execute without writing the outputs back; exit 1 on any failure")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    args = ap.parse_args(argv)
    paths = notebooks(args.prefixes)
    if not paths:
        print("no notebooks match")
        return 1
    ok = all([run(p, write=not args.check, timeout=args.timeout) for p in paths])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
