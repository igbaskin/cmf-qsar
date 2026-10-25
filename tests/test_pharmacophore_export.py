"""Exporting the co-field pharmacophore as a query (plan item F5).

A file in somebody else's format is only as good as its shape, so the tests read back what was
written and check it against the features that went in -- including the three kinds a query has
no way to say, which have to come back as a refusal rather than as a silent omission.
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.pharmacophore import (PHARMIT_TYPE, PML_TYPE, Feature, exportable,  # noqa: E402
                                   write_pharmit, write_pml, write_query)


def _feat(kind, x, r=1.5, importance=1.0):
    return Feature(kind=kind, centre=np.array([x, 2.0 * x, -x], float), radius=r, field="",
                   sign=1, importance=importance, meaning="", favourable=importance > 0)


def _features():
    return [_feat("donor", 1.0, 1.2, 5.0), _feat("acceptor", 2.0, 1.6, 4.0),
            _feat("hydrophobic", 3.0, 1.8, 3.0), _feat("positive", 4.0, 1.1, 2.0),
            _feat("negative", 5.0, 1.3, 1.5), _feat("exclusion", 6.0, 2.2, -1.0),
            _feat("bulk", 7.0, 2.5, 0.9), _feat("polar", 8.0, 2.0, -0.8),
            _feat("donor_bad", 9.0, 1.4, -0.7)]


def test_what_a_query_can_and_cannot_say():
    keep, drop = exportable(_features())
    assert {f.kind for f in keep} == {"donor", "acceptor", "hydrophobic", "positive", "negative",
                                      "exclusion"}
    assert {f.kind for f in drop} == {"bulk", "polar", "donor_bad"}


def test_the_pharmit_session_carries_every_feature_it_can(tmp_path):
    feats = _features()
    out = str(tmp_path / "q.json")
    left = write_pharmit(out, feats, enable_top=2)
    assert {f.kind for f in left} == {"bulk", "polar", "donor_bad"}
    doc = json.load(open(out, encoding="utf-8"))
    pts = doc["points"]
    assert len(pts) == 6
    assert [p["name"] for p in pts[:3]] == ["HydrogenDonor", "HydrogenAcceptor", "Hydrophobic"]
    for p, f in zip(pts, [f for f in feats if f.kind in PHARMIT_TYPE]):
        assert p["x"] == pytest.approx(f.centre[0], abs=1e-3)
        assert p["y"] == pytest.approx(f.centre[1], abs=1e-3)
        assert p["z"] == pytest.approx(f.centre[2], abs=1e-3)
        assert p["radius"] == pytest.approx(f.radius, abs=1e-3)
    assert all(set(p) >= {"enabled", "vector_on", "svector", "hasvec"} for p in pts)


def test_a_query_is_a_conjunction_so_only_a_few_start_switched_on(tmp_path):
    out = str(tmp_path / "q.json")
    write_pharmit(out, _features(), enable_top=2)
    pts = json.load(open(out, encoding="utf-8"))["points"]
    on = [p for p in pts if p["enabled"]]
    assert len(on) == 3                       # the two strongest, plus the exclusion sphere
    assert {p["name"] for p in on} == {"HydrogenDonor", "HydrogenAcceptor", "ExclusionSphere"}
    assert all(p["enabled"] for p in pts if p["name"] == "ExclusionSphere")


def test_the_strongest_features_are_the_ones_switched_on(tmp_path):
    feats = [_feat("donor", 1.0, 1.2, 0.1), _feat("acceptor", 2.0, 1.6, 9.0),
             _feat("hydrophobic", 3.0, 1.8, 5.0)]
    out = str(tmp_path / "q.json")
    write_pharmit(out, feats, enable_top=1)
    pts = json.load(open(out, encoding="utf-8"))["points"]
    assert [p["name"] for p in pts if p["enabled"]] == ["HydrogenAcceptor"]


def test_the_pml_is_xml_of_the_shape_that_format_has(tmp_path):
    feats = _features()
    out = str(tmp_path / "q.pml")
    left = write_pml(out, feats, name="a series", enable_top=3)
    assert {f.kind for f in left} == {"bulk", "polar", "donor_bad"}
    root = ET.parse(out).getroot()
    assert root.tag == "pharmacophore" and root.get("name") == "a series"
    points = root.findall("point")
    volumes = root.findall("volume")
    assert len(points) == 5 and len(volumes) == 1
    assert volumes[0].get("type") == "exclusion"
    assert [p.get("name") for p in points] == [PML_TYPE[k] for k in
                                               ("donor", "acceptor", "hydrophobic", "positive",
                                                "negative")]
    pos = points[0].find("position")
    assert float(pos.get("x3")) == pytest.approx(1.0)
    assert float(pos.get("tolerance")) == pytest.approx(1.2)
    assert sum(1 for p in points if p.get("disabled") == "false") == 3


def test_the_extension_picks_the_format(tmp_path):
    feats = _features()
    j, p = str(tmp_path / "q.json"), str(tmp_path / "q.pml")
    write_query(j, feats)
    write_query(p, feats)
    assert json.load(open(j, encoding="utf-8"))["points"]
    assert ET.parse(p).getroot().tag == "pharmacophore"
    with pytest.raises(ValueError, match="no query format"):
        write_query(str(tmp_path / "q.sdf"), feats)


def test_nothing_exportable_still_writes_a_file_a_program_can_open(tmp_path):
    feats = [_feat("bulk", 1.0), _feat("polar", 2.0)]
    j = str(tmp_path / "q.json")
    left = write_query(j, feats)
    assert len(left) == 2
    assert json.load(open(j, encoding="utf-8"))["points"] == []
