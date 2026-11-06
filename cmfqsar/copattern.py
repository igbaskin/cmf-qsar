"""
The co-pattern of an invariant kernel as a curve: weight against distance (plan item NT9).

An overlap kernel has a co-field: a map C(r) that says where more field raises the property.
The invariant kernels of this program -- the molecular autocorrelation
(:mod:`cmfqsar.invariant`), the surface autocorrelation of an adsorbate, the periodic
autocorrelation of a cell (:mod:`cmfqsar.periodic`) and the reciprocal-space channels
(:mod:`cmfqsar.reciprocal`) -- have none, because a distribution of distances cannot say
*where* anything is.  What they have instead is a weight per channel, and a channel is one pair
of field types at one distance.  :func:`cmfqsar.periodic.channel_weights` computes those weights
and :func:`cmfqsar.periodic.top_channels` ranks them.

A ranked list of twenty channels out of several thousand is hard to read as a shape.  The same
numbers, drawn as a curve of weight against distance for one pair of field types, are the
reading these kernels do have: a positive part at 3 angstrom and a negative part at 6 angstrom
says that the property wants those two fields close together and not far apart.  This module
builds that curve; :class:`cmfqsar.gui.copattern_panel.CoPatternChart` draws it.

What the curve is, exactly
--------------------------
A fitted linear model on an invariant kernel predicts

    y = b + sum_c w_c phi_c(x),

with ``phi`` the scaled and unit-normalised channels of the structure and ``w`` the weights.
The channels are laid out in blocks, one block per unordered pair of field types, and inside a
block the distances run in order.  The curve of one pair is that block, plotted against the
distance grid of the kernel's own specification.

Two kernels resolve a distance further and their curve is a **sum** over what is left:

* the **surface** autocorrelation resolves the in-plane distance by the heights of both atoms,
  so each in-plane distance carries ``n_heights ** 2`` channels;
* the **periodic** autocorrelation of a layer does the same when height channels are on
  (``z_max > 0``).

For those, the point drawn at a distance is the sum of the weights of the height channels at
that distance.  The sum is the right summary because the prediction is a sum over channels: the
curve then says what the whole distance contributes, and the Reading table beside it still names
the individual height channels.  :attr:`CoPattern.grouped` says how many channels went into each
point, so a reader is never left guessing, and it is 1 whenever nothing was summed.

Two kernels have no distance axis at all.  The three-body channels of plan item M10 index a
channel by two radii and a separation, and the chirality channels of the molecular
autocorrelation by a radial shell of a signed volume; neither is a curve of weight against one
distance, so :func:`copattern` refuses them and says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import FieldSet
from .invariant import expanded_field_labels, field_pairs

#: the two things the y axis can show.  "weight" is the model's own coefficient of the channel;
#: "weight x spread" multiplies it by how much the channel varies over the training set, which is
#: how much the channel actually moves a prediction and is the ranking `top_channels` uses
VALUES = ("weight", "weight x spread")


@dataclass
class CoPattern:
    """One curve: the weights of one pair of field types against distance.

    ``x`` and ``y`` are what is drawn.  ``weight`` and ``moved`` are both carried whichever is
    drawn, so that a chart can switch between them without recomputing anything: ``weight`` is
    the coefficient and ``moved`` is the coefficient times the spread of the channel over the
    training structures.
    """
    pair: str                        # "Electrostatic x Steric", say
    pair_index: int
    x: np.ndarray                    # the distance grid, in angstrom (or 1/A in reciprocal space)
    weight: np.ndarray
    moved: np.ndarray                # weight x spread
    spread: np.ndarray               # mean spread of the channels behind each point
    x_label: str
    y_label: str
    value: str                       # which of VALUES is in `y`
    grouped: int                     # channels summed into each point; 1 when nothing was summed
    note: str                        # one line a reader needs to read the curve correctly

    @property
    def y(self) -> np.ndarray:
        return self.moved if self.value == "weight x spread" else self.weight

    def describe(self) -> str:
        k = int(np.argmax(np.abs(self.y))) if len(self.y) else -1
        if k < 0:
            return f"{self.pair}: no channels"
        return (f"{self.pair}: largest at {self.x[k]:.2f} with {self.y[k]:+.4f} "
                f"({self.y_label})")


# --------------------------------------------------------------------------- #
# The channel layout of each invariant kernel
# --------------------------------------------------------------------------- #
@dataclass
class _Layout:
    field_labels: List[str]
    axis: np.ndarray
    axis_label: str
    per_point: int
    kind: str
    note: str


def _layout(model, fieldset: FieldSet) -> _Layout:
    """Where the channels of `model` sit: the field labels, the distance grid and the grouping.

    Raises ``ValueError`` for a kernel whose channels are not indexed by one distance.
    """
    spec = model._ac_spec()
    if getattr(model, "is_threebody", False):
        raise ValueError("the three-body channels index a channel by two radii and a separation, "
                         "not by one distance, so they have no curve of weight against distance; "
                         "read them in the channel table")
    if getattr(model, "is_reciprocal", False):
        return _Layout([s.label for s in fieldset.specs], np.asarray(spec.centres, float),
                       "reciprocal length q (1/A)", 1, "reciprocal",
                       "a reciprocal length, not a distance: a peak at q means the two fields "
                       "repeat with a period of 2 pi / q")
    if getattr(model, "is_periodic", False):
        labels = [s.label for s in fieldset.specs]
        if getattr(spec, "has_heights", False):
            n_z = len(spec.heights)
            return _Layout(labels, np.asarray(spec.centres, float),
                           "in-plane distance between the two atoms (A)", n_z * n_z, "periodic",
                           f"a layer with height channels: each point is the sum of the "
                           f"{n_z * n_z} height channels at that in-plane distance")
        return _Layout(labels, np.asarray(spec.centres, float),
                       "distance between the two atoms (A)", 1, "periodic", "")
    if getattr(model, "on_surface", False):
        n_z = len(spec.heights)
        return _Layout([s.label for s in fieldset.specs], np.asarray(spec.centres, float),
                       "in-plane distance between the two atoms (A)", n_z * n_z, "surface",
                       f"the surface kernel resolves a distance by the heights of both atoms: "
                       f"each point is the sum of the {n_z * n_z} height channels at that "
                       f"in-plane distance")
    if getattr(model, "kernel_source", "") != "autocorrelation":
        raise ValueError(f"the '{getattr(model, 'kernel_source', 'fields')}' kernel is not an "
                         f"invariant kernel with distance channels; the overlap kernel has a "
                         f"co-field instead")
    note = ("the signed (chirality) channels of this kernel are indexed by a radial shell and "
            "are not in the curve" if getattr(spec, "chirality", False) else "")
    return _Layout(expanded_field_labels(fieldset), np.asarray(spec.centres, float),
                   "distance between the two atoms (A)", 1, "autocorrelation", note)


def pair_labels(model, fieldset: FieldSet) -> List[str]:
    """The label of every pair of field types, in the order the channel blocks are laid out.

    Raises ``ValueError`` for a kernel that has no distance channels, with a message that says
    what to read instead.
    """
    lay = _layout(model, fieldset)
    names = lay.field_labels
    return [names[f] if f == g else f"{names[f]} x {names[g]}"
            for f, g in field_pairs(len(names))]


def copattern(model, fieldset: FieldSet, pair_index: int = 0,
              value: str = "weight") -> CoPattern:
    """The curve of one pair of field types: weight against distance.

    `pair_index` indexes :func:`pair_labels`.  `value` is one of :data:`VALUES`.
    """
    if value not in VALUES:
        raise ValueError(f"value must be one of {VALUES}, not {value!r}")
    from .periodic import channel_weights
    lay = _layout(model, fieldset)
    pairs = pair_labels(model, fieldset)
    if not 0 <= int(pair_index) < len(pairs):
        raise ValueError(f"pair_index must be between 0 and {len(pairs) - 1}")
    w = np.asarray(channel_weights(model), float)
    sd = _channel_spread(model)
    n_bins = int(len(lay.axis)) * int(lay.per_point)
    start = int(pair_index) * n_bins
    if start + n_bins > len(w):
        raise ValueError(f"the model has {len(w)} channels, which is fewer than the "
                         f"{start + n_bins} the layout of this kernel asks for")
    block_w = w[start:start + n_bins].reshape(len(lay.axis), lay.per_point)
    block_sd = sd[start:start + n_bins].reshape(len(lay.axis), lay.per_point)
    weight = block_w.sum(axis=1)
    moved = (block_w * block_sd).sum(axis=1)
    spread = block_sd.mean(axis=1)
    y_label = ("weight x spread (how much the channel moves a prediction)"
               if value == "weight x spread" else "channel weight")
    return CoPattern(pair=pairs[int(pair_index)], pair_index=int(pair_index), x=lay.axis,
                     weight=weight, moved=moved, spread=spread, x_label=lay.axis_label,
                     y_label=y_label, value=value, grouped=int(lay.per_point), note=lay.note)


def _channel_spread(model) -> np.ndarray:
    """The standard deviation of every scaled, normalised channel over the training structures.

    This is the ``spread`` of :func:`cmfqsar.periodic.top_channels`, computed the same way, so
    the curve and the ranked table are the same numbers.
    """
    Z = np.asarray(model.ac_train_, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    return Z.std(axis=0)


def copattern_rows(model, fieldset: FieldSet, pair_index: int = 0,
                   value: str = "weight") -> Tuple[List[str], List[List[str]]]:
    """The same curve as a table: the column titles and one row per point.

    The chart is the reading; this is what a reader copies into a paper or a spreadsheet.
    """
    cp = copattern(model, fieldset, pair_index, value)
    spread_title = ("Mean spread over the training set" if cp.grouped > 1
                    else "Spread over the training set")
    cols = [cp.x_label, "Weight", spread_title, "Weight x spread"]
    rows = [[f"{x:.2f}", f"{w:+.4f}", f"{s:.4f}", f"{m:+.4f}"]
            for x, w, s, m in zip(cp.x, cp.weight, cp.spread, cp.moved)]
    return cols, rows


def has_copattern(model, fieldset: Optional[FieldSet] = None) -> bool:
    """Whether this model's channels can be drawn as weight against distance."""
    if model is None or getattr(model, "coef", None) is None:
        return False
    if not getattr(model, "is_linear", False):
        return False
    if getattr(model, "ac_train_", None) is None:
        return False
    try:
        _layout(model, fieldset if fieldset is not None else model.fieldset)
    except (ValueError, AttributeError):
        return False
    return True


__all__ = ["CoPattern", "VALUES", "copattern", "copattern_rows", "has_copattern", "pair_labels"]
