"""The drawing style of the molecular field, which depends on what else is on screen."""
import pytest

pytest.importorskip("PySide6")

from cmfqsar.gui.main_window import molfield_wireframe        # noqa: E402


def test_named_styles_ignore_the_co_field():
    """A style the user named is obeyed whatever else is drawn."""
    for drawn in (False, True):
        assert molfield_wireframe("mesh", drawn) is True
        assert molfield_wireframe("solid", drawn) is False


def test_auto_is_a_mesh_beside_a_co_field():
    # the two share a colour scale, so only the style tells them apart
    assert molfield_wireframe("auto", True) is True


def test_auto_is_solid_when_the_field_is_alone():
    # nothing to see through to, and nothing to be confused with
    assert molfield_wireframe("auto", False) is False


def test_a_ticked_but_undrawn_co_field_does_not_count():
    """The flag is whether a surface was drawn, not whether the box is ticked.

    The co-field box can be on while no co-field exists -- no model yet, or the grids not
    computed -- and in that case the molecular field is alone on screen and should be solid.
    """
    assert molfield_wireframe("auto", False) is False


def test_an_unknown_style_is_solid():
    assert molfield_wireframe("", True) is False
