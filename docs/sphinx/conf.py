"""Sphinx configuration for the HTML API reference of cmfqsar.

The site is built from the docstrings of the source, the same ones ``docs/make_api.py`` turns
into ``docs/api.md``; the HTML version adds working cross-references and search.  Build it with

    python -m sphinx -b html docs/sphinx docs/sphinx/_build/html

and open ``docs/sphinx/_build/html/index.html``.  The build output and the generated module
pages (``_autosummary``) are not committed.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cmfqsar import __version__  # noqa: E402

project = "cmfqsar"
author = "Igor Baskin"
copyright = "2026, Igor Baskin"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",   # the "Parameters / Returns" sections some docstrings use
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# render parameters as a plain list: the text after "name :" is a description here, not a type
# to cross-reference
napoleon_use_param = False
napoleon_use_rtype = False

# one page per module, generated from the template in _templates/autosummary
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "signature"
# optional dependencies that may be absent where the documentation is built; the packages of
# the gui, crystals and fast extras are not mocked, because a mocked base class breaks the
# modules that subclass it, so the site is built from a checkout with those extras installed
autodoc_mock_imports = ["tblite", "xtb", "dimorphite_dl"]

# a short attribute name such as "weight" exists in several classes; Sphinx links the first
# and would otherwise warn about the ambiguity at every mention
suppress_warnings = ["ref.python"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"cmfqsar {release} API reference"
html_static_path = []
