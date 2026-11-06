"""
The co-pattern chart: weight against distance for one pair of field types (plan item NT9).

The invariant kernels have no co-field, so there is no map to look at.  What they have is a
weight per channel, and a channel is one pair of field types at one distance.  A ranked list of
twenty channels out of several thousand is a table; the same numbers for one pair of field
types, drawn against distance, are a shape a reader can take in at once.
:mod:`cmfqsar.copattern` computes the curve and this widget draws it.

The chart is used in both windows, because both have kernels without a co-field: the molecular
window (the autocorrelation kernel of a set with no common frame, and the surface kernel of an
adsorbate) and the crystal window (the periodic autocorrelation of a cell).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout,
                               QWidget)

from ..copattern import VALUES, copattern, has_copattern, pair_labels

#: the colour of the curve, and of the zero line it is read against
CURVE = (40, 90, 200)
ZERO = (130, 130, 130)


class CoPatternChart(QWidget):
    """A chart of channel weight against distance, with the pair of field types to choose from.

    Call :meth:`set_model` after a fit and :meth:`clear` when there is nothing to show.  The
    widget keeps the model, so changing the pair or the quantity on the y axis costs nothing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.fieldset = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        head = QHBoxLayout()
        head.addWidget(QLabel("Pair of field types"))
        self.cmb_pair = QComboBox()
        self.cmb_pair.setMinimumWidth(240)
        self.cmb_pair.setToolTip(
            "Which two field types the curve is for. A channel of these kernels is one pair of "
            "field types at one distance, so the curve of one pair is the whole reading the "
            "kernel offers for that pair.")
        self.cmb_pair.currentIndexChanged.connect(self._redraw)
        head.addWidget(self.cmb_pair)
        head.addWidget(QLabel("   Show"))
        self.cmb_value = QComboBox()
        self.cmb_value.addItems(["weight", "weight x spread"])
        self.cmb_value.setToolTip(
            "'weight' is the model's own coefficient of the channel. 'weight x spread' "
            "multiplies it by how much the channel varies over the training structures, which "
            "is how much the channel actually moves a prediction and is the ranking the channel "
            "table uses.")
        self.cmb_value.currentIndexChanged.connect(self._redraw)
        head.addWidget(self.cmb_value)
        head.addStretch(1)
        lay.addLayout(head)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "distance between the two atoms (A)")
        self.plot.setLabel("left", "channel weight")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.zero = pg.InfiniteLine(angle=0, pos=0.0,
                                    pen=pg.mkPen(ZERO, style=Qt.DashLine))
        self.plot.addItem(self.zero)
        self.curve = self.plot.plot([], [], pen=pg.mkPen(CURVE, width=2),
                                    symbol="o", symbolSize=5,
                                    symbolBrush=pg.mkBrush(*CURVE, 200))
        lay.addWidget(self.plot, 1)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: palette(mid); font-size: 9pt")
        lay.addWidget(self.note)
        self.clear("Build a model on an invariant kernel to see its co-pattern.")

    # ------------------------------------------------------------------ state
    def clear(self, message: str = "") -> None:
        """Empty the chart and say why it is empty."""
        self.model = self.fieldset = None
        self.cmb_pair.blockSignals(True)
        self.cmb_pair.clear()
        self.cmb_pair.blockSignals(False)
        self.curve.setData([], [])
        self.note.setText(message)

    def set_model(self, model, fieldset=None) -> bool:
        """Show the co-pattern of `model`.  Returns False, and says why, when it has none."""
        fs = fieldset if fieldset is not None else getattr(model, "fieldset", None)
        if model is None or fs is None:
            self.clear("Build a model on an invariant kernel to see its co-pattern.")
            return False
        if not has_copattern(model, fs):
            self.clear(self._why_not(model))
            return False
        try:
            pairs = pair_labels(model, fs)
        except (ValueError, AttributeError) as exc:
            self.clear(str(exc))
            return False
        self.model, self.fieldset = model, fs
        self.cmb_pair.blockSignals(True)
        self.cmb_pair.clear()
        self.cmb_pair.addItems(pairs)
        self.cmb_pair.blockSignals(False)
        self._redraw()
        return True

    @staticmethod
    def _why_not(model) -> str:
        """One sentence saying why this model has no curve, and what to read instead."""
        if getattr(model, "coef", None) is None:
            return "Build a model first."
        if not getattr(model, "is_linear", False):
            return (f"The '{getattr(model, 'kernel_layer', '?')}' kernel layer is not linear in "
                    f"the channels, so the channels have no weights. Use the linear layer, or "
                    f"read the prediction with Shapley values.")
        if getattr(model, "ac_train_", None) is None:
            return ("This model uses the overlap kernel of the fields, which has a co-field: "
                    "the map in the 3D view is its reading, and this chart is for the kernels "
                    "that have no co-field.")
        if getattr(model, "is_threebody", False):
            return ("The three-body channels index a channel by two radii and a separation, not "
                    "by one distance, so there is no curve of weight against distance; the "
                    "channel table names them.")
        return "This kernel has no channels indexed by one distance."

    # ----------------------------------------------------------------- drawing
    def _redraw(self, *_) -> None:
        if self.model is None or self.cmb_pair.currentIndex() < 0:
            return
        value = VALUES[max(self.cmb_value.currentIndex(), 0)]
        try:
            cp = copattern(self.model, self.fieldset, self.cmb_pair.currentIndex(), value)
        except (ValueError, RuntimeError) as exc:
            self.clear(str(exc))
            return
        self.curve.setData(np.asarray(cp.x, float), np.asarray(cp.y, float))
        self.plot.setLabel("bottom", cp.x_label)
        self.plot.setLabel("left", cp.y_label)
        self.plot.enableAutoRange()
        lines = [cp.describe()]
        if cp.grouped > 1:
            lines.append(cp.note)
        elif cp.note:
            lines.append(cp.note)
        lines.append("Positive means that more of these two fields at this distance raises the "
                     "property; the curve is the whole reading these kernels have, because a "
                     "distribution of distances cannot say where in the structure anything is.")
        self.note.setText("  ".join(x for x in lines if x))

    def current_pair(self) -> int:
        return int(self.cmb_pair.currentIndex())

    def current_value(self) -> str:
        return VALUES[max(self.cmb_value.currentIndex(), 0)]


__all__ = ["CoPatternChart"]
