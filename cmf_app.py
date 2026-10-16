#!/usr/bin/env python
"""Launch the CMF 3D-QSAR desktop application.

    python cmf_app.py [dataset.sdf] [--activity PROPERTY]
"""
import sys

from cmfqsar.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
