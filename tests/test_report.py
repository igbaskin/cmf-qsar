"""The model report: one self-contained HTML file for someone who will never run the program.

What the tests require is what the file promises: every section present, nothing fetched from
outside the file, a check that was not run said to be missing rather than left out, and a kernel
that has no co-fields losing the co-fields and keeping the rest.
"""
import base64
import os
import re
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.report import ReportSpec, build_html, write_report  # noqa: E402

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
          "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
          "c1ccc(Br)cc1O", "c1ccc(N)cc1C"]
Y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4])
FAST = ReportSpec(spacing=1.2, margin=2.0, cofields=False, pharmacophore=False)


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


@pytest.fixture(scope="module")
def fitted():
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(_mol(s), fs, f"m{i}") for i, s in enumerate(SMILES)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.1})
    model.fit(table, Y, cv_folds=4, cv_seed=0)
    return model


def _figures(html: str):
    return re.findall(r'src="data:image/png;base64,([^"]+)"', html)


def test_every_section_is_there(fitted):
    html = build_html(fitted, extra={"dataset": "series.sdf", "activity": "pIC50"}, spec=FAST)
    for heading in ("What this model is", "How well it fits and predicts",
                    "Whether to believe that", "What the model rests on",
                    "The extremes of the series", "Provenance"):
        assert f">{heading}</h2>" in html
    assert "krr" in html and "pIC50" in html and "series.sdf" in html


def test_nothing_in_it_comes_from_outside_the_file(fitted):
    html = build_html(fitted, spec=FAST)
    assert "http://" not in html and "https://" not in html
    srcs = re.findall(r'<img[^>]*src="([^"]{0,24})', html)
    assert srcs and all(s.startswith("data:image/png;base64,") for s in srcs)
    for b64 in _figures(html):
        assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_check_that_was_not_run_says_so(fitted):
    html = build_html(fitted, spec=FAST)
    for missing in ("y-randomisation", "Bootstrap intervals", "Applicability domain"):
        i = html.index(missing)
        assert 'class="missing"' in html[i:i + 400]


def test_the_validation_that_was_run_is_reported(fitted):
    model = fitted
    model.stats.extra["scramble"] = {"runs": 20, "statistic": "q2", "observed": 0.5,
                                     "mean": -0.1, "sd": 0.2, "max": 0.3, "z": 3.0, "p": 0.047}
    model.stats.extra["bootstrap"] = {"q2": {"value": 0.5, "lo": 0.3, "hi": 0.7, "runs": 200}}
    try:
        html = build_html(model, spec=FAST)
        assert "20 shuffles" in html and "0.047" in html
        i = html.index("Bootstrap intervals")
        assert 'class="missing"' not in html[i:i + 300]
    finally:
        model.stats.extra.pop("scramble", None)
        model.stats.extra.pop("bootstrap", None)


def test_the_co_fields_and_the_pharmacophore_are_drawn_when_asked(fitted):
    n_fields = len([s for s in fitted.fieldset.specs if s.enabled])
    whole = build_html(fitted, spec=ReportSpec(spacing=1.2, margin=2.0, cofield_z=0.0))
    assert ">The co-fields</h2>" in whole and ">The pharmacophore</h2>" in whole
    # one figure for the statistics, one for the field contributions, one per field
    assert len(_figures(whole)) >= 2 + n_fields


def test_the_co_fields_are_cut_at_their_error_bar_by_default(fitted):
    """A field with nothing above its error bar is said to have nothing, not left out."""
    html = build_html(fitted, spec=ReportSpec(spacing=1.2, margin=2.0))
    assert "standard deviations from zero is drawn" in html
    assert "survived the error bar" in html or "stands above its error bar" in html
    whole = build_html(fitted, spec=ReportSpec(spacing=1.2, margin=2.0, cofield_z=0.0))
    assert len(_figures(html)) <= len(_figures(whole))


def test_a_kernel_without_co_fields_loses_them_and_keeps_the_rest(fitted):
    model = fitted
    model.kernel_layer = "gauss"
    try:
        html = build_html(model, spec=ReportSpec(spacing=1.5, margin=2.0))
        assert ">The co-fields</h2>" in html
        i = html.index(">The co-fields</h2>")
        assert 'class="missing"' in html[i:i + 500]
        assert ">How well it fits and predicts</h2>" in html     # the rest survives
        assert ">Provenance</h2>" in html
    finally:
        model.kernel_layer = None


def test_what_a_user_writes_is_escaped(fitted):
    html = build_html(fitted, title="<script>alert(1)</script>", spec=FAST)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_it_writes_one_file_that_opens_on_its_own(fitted, tmp_path):
    out = str(tmp_path / "model.html")
    assert write_report(out, fitted, spec=FAST) == out
    text = open(out, encoding="utf-8").read()
    assert text.startswith("<!DOCTYPE html>") and text.rstrip().endswith("</html>")
    assert os.listdir(tmp_path) == ["model.html"]       # nothing beside it to carry along
