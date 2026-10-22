#!/usr/bin/env python
"""The command-line tool, runnable from a checkout without installing anything.

    python cmf_cli.py build data.sdf --activity pIC50 ...

The tool itself lives in ``cmfqsar/cli.py``; an installed package also provides it as the
``cmf`` command.
"""
import sys

from cmfqsar.cli import main

if __name__ == "__main__":
    sys.exit(main())
