"""The viewer's controls.  Rows that appear under several panels are linked copies, made by one object."""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QSpinBox, QStyle,
                               QStyleOptionSlider, QToolButton)

SLIDER_STYLE = """
QSlider { max-height: 14px; }
QSlider::groove:horizontal { height: 4px; background: #d5d9de; border-radius: 2px; }
QSlider::sub-page:horizontal { height: 4px; background: #8ab4f8; border-radius: 2px; }
QSlider::handle:horizontal { width: 10px; margin: -4px 0; border-radius: 5px; background: #1a73e8; }
"""
STEPS = (("<<", -2), ("<", -1), (">", 1), (">>", 2))  # button, groups moved


def canvas(height):
    c = FigureCanvasQTAgg(Figure(layout="constrained"))
    c.setMinimumHeight(height)
    return c


def make_row(*items):
    """A row of widgets; a string becomes a label and None a stretch."""
    row = QHBoxLayout()
    row.setContentsMargins(6, 0, 6, 0)
    for item in items:
        if item is None:
            row.addStretch()
        else:
            row.addWidget(QLabel(item) if isinstance(item, str) else item)
    return row


def make_button(text, slot):
    b = QPushButton(text)
    b.clicked.connect(lambda: slot())
    return b


def make_edit(placeholder, entered):
    """A narrow field; ``entered`` is called on Enter."""
    edit = QLineEdit(placeholderText=placeholder)
    edit.setMaximumWidth(90)
    edit.returnPressed.connect(lambda: entered())
    return edit


def make_combo(items):
    combo = QComboBox()
    combo.addItems(items)
    return combo


def number(edit):
    """The number typed in a field, or None."""
    try:
        return float(edit.text())
    except ValueError:
        return None


class ClickSlider(QSlider):
    """Horizontal slider that jumps to the clicked point (then drags) instead of paging."""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
            handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            x = event.position().toPoint().x()
            if not handle.contains(event.position().toPoint()):
                self.setValue(QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), x - groove.x() - handle.width() // 2,
                    max(1, groove.width() - handle.width())))
        super().mousePressEvent(event)


class TimeSliders(QObject):
    """Time sliders (value = frame); moving one moves the others."""
    moved = Signal(int)    # new value
    stepped = Signal(int)  # groups to move

    def __init__(self):
        super().__init__()
        self.sliders, self.labels = [], []

    def new_row(self):
        slider, label = ClickSlider(Qt.Horizontal), QLabel()
        slider.setStyleSheet(SLIDER_STYLE)
        slider.valueChanged.connect(self.slide)
        self.sliders.append(slider)
        self.labels.append(label)
        buttons = [self.step_button(text, n) for text, n in STEPS]
        return make_row("time", *buttons[:2], slider, *buttons[2:], label)

    def step_button(self, text, n):
        b = QToolButton(text=text, autoRaise=True)
        b.setToolTip(f"{'back' if n < 0 else 'forward'} {abs(n)} group{'s' * (abs(n) > 1)} at the grouping scale")
        b.clicked.connect(lambda: self.stepped.emit(n))
        return b

    def slide(self, value):
        self.set(value)
        self.moved.emit(value)

    def set(self, value, maximum=None):
        """Move every slider without a signal."""
        for s in self.sliders:
            with QSignalBlocker(s):
                if maximum is not None:
                    s.setMaximum(maximum)
                s.setValue(value)

    def move(self, frames):
        slider = self.sliders[0]
        slider.setValue(min(max(slider.value() + frames, 0), slider.maximum()))

    def set_text(self, text):
        for label in self.labels:
            label.setText(text)


class RangeRows(QObject):
    """The d range of all panels (Enter or "Apply range"; "Show whole" clears it) and the scale and
    range of the projections' x axis (Enter or "Apply" fixes it; "Auto", or changing the scale, fits it
    again and shows the fitted range)."""
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.ranges, self.units, self.scales, self.xranges = [], [], [], []
        self.xlim = None  # None: fitted

    def new_row(self, projection):
        items = self.range_items()
        return make_row(*items, None, *self.x_items()) if projection else make_row(*items)

    def range_items(self):
        unit = make_combo(["d (nm)", "px"])
        unit.currentTextChanged.connect(lambda text: [u.setCurrentText(text) for u in self.units])
        lo = make_edit("radius from", lambda: self.apply_range(lo, hi))
        hi = make_edit("radius to", lambda: self.apply_range(lo, hi))
        self.ranges.append((lo, hi))
        self.units.append(unit)
        return ("Range", lo, "to", hi, unit, make_button("Apply range", lambda: self.apply_range(lo, hi)),
                make_button("Show whole", self.clear_range))

    def x_items(self):
        scale = make_combo(["log", "linear"])
        scale.currentTextChanged.connect(self.set_scale)
        lo = make_edit("x from", lambda: self.apply_x(lo, hi))
        hi = make_edit("x to", lambda: self.apply_x(lo, hi))
        self.scales.append(scale)
        self.xranges.append((lo, hi))
        return ("projections x", scale, "from", lo, "to", hi, make_button("Apply", lambda: self.apply_x(lo, hi)),
                make_button("Auto", lambda: self.set_xlim(None)))

    def apply_range(self, lo, hi):
        """Copy one row's d range to every row."""
        self.set_range(lo.text(), hi.text())

    def clear_range(self):
        self.set_range("", "")

    def set_range(self, a, b):
        for lo, hi in self.ranges:
            lo.setText(a)
            hi.setText(b)
        self.changed.emit()

    def typed_range(self):
        """(from, to, unit) as typed, or None if a field is not a number."""
        a, b = (number(edit) for edit in self.ranges[0])  # the rows are kept in step
        return None if a is None or b is None else (a, b, self.units[0].currentText())

    @property
    def log(self):
        return self.scales[0].currentText() == "log"

    def set_scale(self, text):
        for scale in self.scales:
            with QSignalBlocker(scale):
                scale.setCurrentText(text)
        self.set_xlim(None)

    def apply_x(self, lo, hi):
        """The typed x range, or auto if it is unusable on the current scale."""
        a, b = number(lo), number(hi)
        usable = a is not None and b is not None and a < b and (a > 0 or not self.log)
        self.set_xlim((a, b) if usable else None)

    def set_xlim(self, lim):
        self.xlim = lim
        if lim is not None:
            self.show_xlim(lim)
        self.changed.emit()

    def show_xlim(self, lim):
        for lo, hi in self.xranges:
            lo.setText(f"{lim[0]:.4g}")
            hi.setText(f"{lim[1]:.4g}")


class ContrastRow(QObject):
    """Kymograph intensity scale (log(1+I) or counts) and cutoffs in that scale, as on the
    colorbar: values at or below "min" are white, at or above "max" black.  An empty field is
    automatic (per-recording percentile)."""
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.scale = make_combo(["log(1+I)", "counts"])
        self.scale.currentTextChanged.connect(self.convert)
        self.lo, self.hi = (make_edit("auto", self.changed.emit) for _ in range(2))
        self.row = make_row("Kymograph intensity", self.scale, "min", self.lo, "to max", self.hi,
                            make_button("Apply", self.changed.emit), make_button("Auto", self.clear), None)

    @property
    def counts(self):
        return self.scale.currentText() == "counts"

    def clear(self):
        self.lo.clear()
        self.hi.clear()
        self.changed.emit()

    def convert(self, _text):
        """Convert any typed cutoffs to the new scale."""
        for edit in (self.lo, self.hi):
            v = number(edit)
            if v is not None:
                edit.setText(f"{np.expm1(v) if self.counts else np.log1p(max(v, 0.0)):.4g}")
        self.changed.emit()

    def limits(self, auto):
        """(vmin, vmax): the typed cutoffs, each falling back to its automatic value."""
        lo, hi = (a if number(edit) is None else number(edit) for edit, a in zip((self.lo, self.hi), auto))
        return (lo, hi) if lo < hi else tuple(auto)


class SmoothingRow(QObject):
    """A window (frames) that smooths one panel's lines over time; 1 = off."""
    changed = Signal()

    def __init__(self, label, note):
        super().__init__()
        self.box = QSpinBox(minimum=1, maximum=101, singleStep=2, value=1, suffix=" frames")
        self.box.setSpecialValueText("off")
        self.box.setKeyboardTracking(False)
        self.box.setToolTip("Moving mean over this many frames (odd), separately on each side of "
                            f"excluded frames.\nOnly the plotted lines change; {note}.")
        self.box.valueChanged.connect(self.set_value)
        self.row = make_row(label, self.box, None)

    def set_value(self, v):
        if v % 2:
            self.changed.emit()
        else:  # a centred window is odd: an even value moves up one, which signals again
            self.box.setValue(v + 1)

    @property
    def window(self):
        return self.box.value()

    @property
    def note(self):
        """For the panel's title."""
        return f" (mean of {self.window} frames)" if self.window > 1 else ""
