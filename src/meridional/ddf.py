"""Aurora DMC ``.ddf`` files: protocol, force and length traces, and when each kymograph column was taken.

The protocol's step times are rebuilt (each delay counts from the end of the previous step; only
Ramps have a duration).  Kymograph column 0 opens at the first non-zero ``Trigger`` step.
"""
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple

import numpy as np

DEFAULT_FRAME_TIME = 0.005  # s
FORCE_COLOR, LENGTH_COLOR = "#2b7fff", "#00a47d"


@dataclass
class Step:
    start: float
    end: float
    kind: str
    target: str
    params: list = field(default_factory=list)


@dataclass
class DDF:
    path: str
    sample_hz: float
    channels: dict   # name -> (units, scale, offset)
    protocol: list
    columns: list
    volts: np.ndarray

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def time(self):
        return self.volts[:, 0] / self.sample_hz

    def signal(self, channel):
        """(values in calibrated units, unit string)."""
        raw = self.volts[:, self.columns.index(channel)]
        if channel not in self.channels:
            return raw, "V"
        units, scale, offset = self.channels[channel]
        return (raw - offset) * scale, units or "V"

    def find_channel(self, units, fallback):
        """First measured (AI) channel whose units match, else ``fallback`` if present."""
        wanted = {u.lower() for u in units}
        for name in self.columns[1:]:
            if name.upper().startswith("AI") and name in self.channels \
                    and self.channels[name][0].lower() in wanted:
                return name
        return fallback if fallback in self.columns else None


def _cells(line):
    return [c.strip() for c in line.rstrip("\n").split("\t")]


def _float(cell, default=np.nan):
    try:
        return float(cell)
    except ValueError:
        return default


def read_protocol(lines):
    steps, clock = [], 0.0
    for line in lines:
        cells = _cells(line)
        if len(cells) < 2 or not cells[0] or np.isnan(_float(cells[0])):
            continue
        params = [p.strip() for p in cells[3].split(",")] if len(cells) > 3 and cells[3] else []
        start = clock + float(cells[0])
        duration = _float(params[-1], 0.0) if cells[1].lower().startswith("ramp") and params else 0.0
        steps.append(Step(start, start + duration, cells[1], cells[2] if len(cells) > 2 else "", params))
        clock = start + duration
    return steps


def line_index(lines, prefix, start=0):
    return next((i for i in range(start, len(lines)) if lines[i].startswith(prefix)), None)


def read_sample_hz(lines, path):
    sample_hz = None
    for line in lines[:25]:
        key, _, value = line.partition(":")
        if key.strip().lower().startswith("sample frequency"):
            sample_hz = float(value)
    if sample_hz is None:
        raise ValueError(f"{path} has no 'Sample Frequency (Hz)' line")
    return sample_hz


def read_channels(lines):
    """{name: (units, scale, offset)}; missing cells are "", 1 and 0."""
    rows = {}
    for line in lines:
        cells = _cells(line)
        if cells[0] in ("Channel", "Units", "Scale (units/V)", "Offset (volts)"):
            rows[cells[0]] = cells[1:]

    def cell(row, i, default):
        return rows[row][i] if i < len(rows.get(row, [])) else default

    return {name: (cell("Units", i, ""), _float(cell("Scale (units/V)", i, ""), 1.0) or 1.0,
                   _float(cell("Offset (volts)", i, ""), 0.0))
            for i, name in enumerate(rows.get("Channel", [])) if name}


def read_ddf(path):
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    sample_hz = read_sample_hz(lines, path)
    data_start = line_index(lines, "Test Data")
    if data_start is None:
        raise ValueError(f"{path} has no 'Test Data' block")
    protocol_start = line_index(lines, "Protocol Array:")
    protocol = read_protocol(lines[protocol_start + 1:data_start]) if protocol_start is not None else []
    head = line_index(lines, "Sample\t", data_start)
    if head is None:
        raise ValueError(f"{path} has no 'Sample' header row")
    volts = np.array([[_float(c) for c in _cells(line)] for line in lines[head + 1:] if line.strip()])
    return DDF(path, sample_hz, read_channels(lines[:data_start]), protocol, _cells(lines[head]), volts)


def load_traces(path):
    """(ddf, [force trace, length trace]) as dicts with t, y, units, key, color, label.
    A file that has not changed is parsed once (shared: not to be modified)."""
    st = os.stat(path)
    return _load_traces(path, st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=8)
def _load_traces(path, _size, _mtime):
    ddf = read_ddf(path)
    traces = []
    for key, units, fallback, color in (("force", ("mn", "n", "g"), "AI1", FORCE_COLOR),
                                        ("length", ("ref", "mm", "l0"), "AI0", LENGTH_COLOR)):
        ch = ddf.find_channel(units, fallback)
        if ch is not None:
            y, u = ddf.signal(ch)
            traces.append({"t": ddf.time, "y": y, "units": u, "key": key, "color": color,
                           "label": f"{key} ({u})\n[{ch}]"})
    if not traces or traces[0]["key"] != "force":
        raise ValueError(f"{ddf.name}: no force channel found")
    return ddf, traces


# ------------------------------------------------------------------ timing

def trigger_time(ddf):
    """When the detector was started: the first non-zero Trigger step."""
    for step in ddf.protocol:
        if step.kind.lower().startswith("trigger") and _float(step.params[0] if step.params else "", 0):
            return step.start
    return None


def detector_frame_time(qf_folder):
    """frame_time from the *_master.h5 in the qf folder's parent, or None."""
    try:
        import h5py
        parent = os.path.dirname(os.path.normpath(qf_folder))
        for name in sorted(f for f in os.listdir(parent) if f.endswith("_master.h5")):
            with h5py.File(os.path.join(parent, name), "r") as h:
                return float(h["entry/instrument/detector/frame_time"][()])
    except (ImportError, OSError, KeyError):
        pass
    return None


class ColumnTime(NamedTuple):
    trigger: float  # s on the DMC clock, when column 0 opens
    dt: float       # s per column
    dt_source: str
    how: str        # how the trigger was found


def column_time(meta, ddf=None, frame_time=None):
    """When the columns were taken: column i covers [i*dt, (i+1)*dt] after the trigger, with dt the
    detector's frame time (master .h5, else 5 ms).  Without a ddf, time counts from the first frame."""
    if frame_time:
        dt, source = frame_time, "given"
    else:
        dt = detector_frame_time(meta["source"])
        dt, source = (dt, "master .h5") if dt else (DEFAULT_FRAME_TIME, "default")
    if ddf is None:
        return ColumnTime(0.0, dt, source, "first frame")
    trigger = trigger_time(ddf)
    if trigger is None:
        return ColumnTime(float(ddf.time[0]), dt, source, "start of the trace (no Trigger in protocol)")
    return ColumnTime(trigger, dt, source, f"first non-zero Trigger ({trigger:.3f} s on the DMC clock)")


def from_trigger(trigger, traces, protocol):
    """Traces and protocol steps with time measured from the trigger."""
    return ([dict(tr, t=tr["t"] - trigger) for tr in traces],
            [Step(s.start - trigger, s.end - trigger, s.kind, s.target, s.params) for s in protocol])
