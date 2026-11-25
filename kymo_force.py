#!/usr/bin/env python3
"""Kymograph with the force and length traces of an Aurora DMC ``.ddf``.

Alignment: the protocol's step times are rebuilt (each delay counts from the
end of the previous step; only Ramps have a duration), and kymograph column 0
opens at the first non-zero ``Trigger`` step, t0.  All times (figure, CSVs)
are given from t0, so column i covers [i*dt, (i+1)*dt] with dt = detector frame_time (from the
``*_master.h5`` next to the qf folder, else 5 ms) times the frame grouping.
Beam-off columns are shaded; they should fall in a Trigger gap.

Writes, next to the kymograph:
    <base>_force.png     kymograph over force and length
    <base>_events.csv    kymograph span, beam-off spans, measured length
                         changes and protocol steps, on both clocks
    <base>_traces.csv    force/length samples with the column each falls in

Example:
    python kymo_force.py qf_results/kymo/meridian_w20_g1.npy LoActive1.ddf --pos-range 0 1200 --stretch 1.4
"""
import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DEFAULT_FRAME_TIME = 0.005  # s
DARK_FRACTION = 0.05        # of the median column sum => beam off
DPI = 150
FORCE_COLOR, LENGTH_COLOR, SPAN_COLOR, EVENT_COLOR = "#2b7fff", "#00a47d", "#d93025", "#8430ce"

# length-change detection
EVENT_SMOOTH = 0.002        # s moving average under the derivative
EVENT_SLOPE_FRACTION = 0.25 # of the fastest change => lever is moving
EVENT_MIN_AMPLITUDE = 0.10  # of the largest move
EVENT_GAP = 0.010           # s of stillness separating two moves
EVENT_PLATEAU = 0.020       # s of plateau read either side of a move
EVENT_NOISE_FLOOR = 20      # largest move must exceed this many x sample noise


# ------------------------------------------------------------------ the .ddf

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


def read_ddf(path):
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()

    sample_hz = None
    for line in lines[:25]:
        key, _, value = line.partition(":")
        if key.strip().lower().startswith("sample frequency"):
            sample_hz = float(value)
    if sample_hz is None:
        raise ValueError(f"{path} has no 'Sample Frequency (Hz)' line")

    rows = {}
    for line in lines:
        cells = _cells(line)
        if cells[0].startswith("Test Data"):
            break
        if cells[0] in ("Channel", "Units", "Scale (units/V)", "Offset (volts)"):
            rows[cells[0]] = cells[1:]
    names, units = rows.get("Channel", []), rows.get("Units", [])
    scale = [_float(c, 1.0) or 1.0 for c in rows.get("Scale (units/V)", [])]
    offset = [_float(c, 0.0) for c in rows.get("Offset (volts)", [])]
    channels = {n: (units[i] if i < len(units) else "",
                    scale[i] if i < len(scale) else 1.0,
                    offset[i] if i < len(offset) else 0.0)
                for i, n in enumerate(names) if n}

    def index_of(prefix, start=0):
        return next((i for i in range(start, len(lines)) if lines[i].startswith(prefix)), None)

    p_start, d_start = index_of("Protocol Array:"), index_of("Test Data")
    if d_start is None:
        raise ValueError(f"{path} has no 'Test Data' block")
    protocol = read_protocol(lines[p_start + 1:d_start]) if p_start is not None else []
    head = index_of("Sample\t", d_start)
    if head is None:
        raise ValueError(f"{path} has no 'Sample' header row")
    volts = np.array([[_float(c) for c in _cells(l)] for l in lines[head + 1:] if l.strip()])
    return DDF(path, sample_hz, channels, protocol, _cells(lines[head]), volts)


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


# ------------------------------------------------------------------ events

def dark_spans(kymo, t0, dt):
    """(start, stop) seconds of runs of columns with no beam."""
    totals = np.nansum(kymo, axis=0)
    median = np.nanmedian(totals)
    if not median > 0:
        return []
    dark = np.flatnonzero(totals < DARK_FRACTION * median)
    spans = []
    for run in np.split(dark, np.flatnonzero(np.diff(dark) > 1) + 1) if dark.size else []:
        spans.append((t0 + run[0] * dt, t0 + (run[-1] + 1) * dt))
    return spans


def moving_average(y, n):
    if n < 2 or y.size < n:
        return y
    padded = np.concatenate([np.full(n // 2, y[0]), y, np.full(n - 1 - n // 2, y[-1])])
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def length_changes(t, y):
    """Moves of the lever: found by speed, edges widened out to the plateaus.

    Returns [{"start", "end", "before", "after"}, ...] in time order.
    """
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    if t.size < 3:
        return []
    hz = 1.0 / float(np.median(np.diff(t)))
    smoothed = moving_average(y, max(1, int(round(EVENT_SMOOTH * hz))))
    slope = np.abs(np.gradient(smoothed, t))
    if not slope.max() > 0:
        return []
    moving = np.flatnonzero(slope > EVENT_SLOPE_FRACTION * slope.max())
    breaks = np.flatnonzero(np.diff(moving) > max(1, int(round(EVENT_GAP * hz))))
    runs = [(r[0], r[-1]) for r in np.split(moving, breaks + 1)]

    noise = float(np.median(np.abs(np.diff(smoothed)))) * 1.4826
    amplitudes = np.array([abs(smoothed[b] - smoothed[a]) for a, b in runs])
    if noise > 0 and amplitudes.max() < EVENT_NOISE_FLOOR * noise:
        return []

    window = max(2, int(round(EVENT_PLATEAU * hz)))
    out = []
    for (a, b), amp in zip(runs, amplitudes):
        if amp < EVENT_MIN_AMPLITUDE * amplitudes.max():
            continue
        before = float(np.median(smoothed[max(0, a - window):a + 1]))
        after = float(np.median(smoothed[b:b + window + 1]))
        tol = max(0.02 * abs(after - before), 3 * noise)
        lo, hi = a, b
        while lo > 0 and abs(smoothed[lo] - before) > tol:
            lo -= 1
        while hi < smoothed.size - 1 and abs(smoothed[hi] - after) > tol:
            hi += 1
        out.append({"start": float(t[lo]), "end": float(t[hi]), "before": before, "after": after})
    return out


def collect_events(moves, units, spans, protocol, t0, t_end):
    """[(name, time, source, detail), ...] sorted by time."""
    ev = [("kymograph start", t0, "kymograph", "column 0 opens"),
          ("kymograph end", t_end, "kymograph", "last column closes")]
    for i, (lo, hi) in enumerate(spans, 1):
        ev += [(f"beam off {i} start", lo, "kymograph", "shutter shut"),
               (f"beam off {i} end", hi, "kymograph", "shutter open")]
    for i, m in enumerate(moves, 1):
        change = m["after"] - m["before"]
        way = "lengthening" if change > 0 else "shortening"
        ev += [(f"length change {i} start", m["start"], "length trace",
                f"{way} of {change:+.4g} {units} begins, from {m['before']:.4g} {units}"),
               (f"length change {i} end", m["end"], "length trace",
                f"{way} of {change:+.4g} {units} done after {(m['end'] - m['start']) * 1e3:.1f} ms, "
                f"at {m['after']:.4g} {units}")]
    for s in protocol:
        name = s.kind if s.target.lower() in ("", s.kind.lower()) else f"{s.kind} {s.target}"
        detail = ", ".join(s.params)
        if s.end > s.start:
            ev += [(f"{name} start", s.start, "protocol", detail), (f"{name} end", s.end, "protocol", detail)]
        else:
            ev.append((name, s.start, "protocol", detail))
    return sorted(ev, key=lambda e: e[1])


def _column_name(trace):
    return re.sub(r"[^0-9A-Za-z]+", "_", f"{trace['key']}_{trace['units']}").strip("_")


def _frame_of(when, t0, dt, n):
    exact = (when - t0) / dt
    return (int(np.floor(exact)) if 0 <= exact < n else ""), exact


def _fmt(v):
    return "" if not np.isfinite(v) else f"{v:.6g}"


def write_traces(path, traces, t0, dt, n):
    t = traces[0]["t"]
    values = [np.interp(t, tr["t"], tr["y"], left=np.nan, right=np.nan) for tr in traces]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "frame", "frame_exact"] + [_column_name(tr) for tr in traces])
        for i, when in enumerate(t):
            frame, exact = _frame_of(when, t0, dt, n)
            w.writerow([f"{when:.6f}", frame, f"{exact:.3f}"] + [_fmt(v[i]) for v in values])


def write_events(path, events, traces, t0, dt, n):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "source", "time_s", "frame", "frame_exact"]
                   + [_column_name(tr) for tr in traces] + ["detail"])
        for name, when, source, detail in events:
            frame, exact = _frame_of(when, t0, dt, n)
            w.writerow([name, source, f"{when:.6f}", frame, f"{exact:.3f}"]
                       + [_fmt(np.interp(when, tr["t"], tr["y"], left=np.nan, right=np.nan))
                          for tr in traces] + [detail])


# ------------------------------------------------------------------ figure

def build_figure(kymo, pos, t0, dt, traces, spans, marks, inv_nm, stretch, title, subtitle):
    """Kymograph over the traces on one time axis; kymograph height = rows * stretch px."""
    n_frames = kymo.shape[1]
    t_end = t0 + n_frames * dt
    top_pad, bottom_pad, trace_h = 1.30, 0.80, 1.35
    kymo_h = max(2.0, kymo.shape[0] * stretch / DPI)
    total_h = kymo_h + trace_h * len(traces) + top_pad + bottom_pad
    left, right = 0.075, 0.845

    fig = plt.figure(figsize=(13, total_h), dpi=DPI)
    grid = fig.add_gridspec(1 + len(traces), 1, height_ratios=[kymo_h] + [trace_h] * len(traces),
                            hspace=0.12, left=left, right=right,
                            top=1 - top_pad / total_h, bottom=bottom_pad / total_h)
    ax_k = fig.add_subplot(grid[0])
    axes = [ax_k] + [fig.add_subplot(grid[i + 1], sharex=ax_k) for i in range(len(traces))]

    shown = np.log1p(np.clip(kymo, 0, None))
    lo, hi = np.percentile(shown[np.isfinite(shown)], (1.0, 99.5))
    image = ax_k.imshow(shown, origin="lower", aspect="auto", cmap="viridis", vmin=lo,
                        vmax=hi if hi > lo else lo + 1, interpolation="nearest",
                        extent=(t0, t_end, pos[0] - 0.5, pos[-1] + 0.5))
    ax_k.set_ylabel("axial position from centre (pixels)", fontsize=9)
    ax_k.tick_params(labelsize=8, labelbottom=False)
    for a, b in spans:
        ax_k.axvspan(a, b, fc="none", ec=SPAN_COLOR, lw=0.8, ls=(0, (4, 3)), zorder=3)
    ax_f = ax_k.secondary_xaxis("top", functions=(lambda x: (x - t0) / dt, lambda f: t0 + f * dt))
    ax_f.set_xlabel("column", fontsize=9)
    ax_f.tick_params(labelsize=8)
    if inv_nm:
        ax_q = ax_k.secondary_yaxis("right", functions=(lambda p: p * inv_nm, lambda q: q / inv_nm))
        ax_q.set_ylabel("q (nm$^{-1}$)", fontsize=9)
        ax_q.tick_params(labelsize=8)
    box = ax_k.get_position()
    bar = fig.colorbar(image, cax=fig.add_axes([0.895, box.y0, 0.014, box.height]))
    bar.set_label("log(1+I) (sum over strip)", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    for ax, tr in zip(axes[1:], traces):
        for a, b in spans:
            ax.axvspan(a, b, color="#9aa0a6", alpha=0.18, lw=0, zorder=0)
        ax.plot(tr["t"], tr["y"], lw=1.4, color=tr["color"])
        inside = tr["y"][(tr["t"] >= t0) & (tr["t"] <= t_end)]
        inside = inside[np.isfinite(inside)]
        if inside.size:  # scale to the visible window, not the whole trace
            ymin, ymax = inside.min(), inside.max()
            pad = 0.09 * (ymax - ymin) or max(abs(ymax) * 0.05, 1e-9)
            ax.set_ylim(ymin - pad, ymax + pad)
        for edge in (t0, t_end):
            ax.axvline(edge, lw=0.9, color=SPAN_COLOR, ls=(0, (4, 3)))
        ax.set_ylabel(tr["label"], fontsize=9)
        ax.grid(axis="y", lw=0.6, color="#e9e9e9")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, labelbottom=False)
        ax.yaxis.set_major_locator(MaxNLocator(4))

    for a, b in marks:
        for ax in axes[1:]:
            ax.axvspan(a, b, color=EVENT_COLOR, alpha=0.10, lw=0)
        for ax in axes:
            for edge in (a, b):
                ax.axvline(edge, lw=0.9, color=EVENT_COLOR, ls=(0, (2, 2)), zorder=4)

    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel("time from trigger (s)", fontsize=9)
    ax_k.set_xlim(t0, t_end)

    handles = [plt.Line2D([], [], lw=0.9, color=SPAN_COLOR, ls=(0, (4, 3)), label="kymograph span")]
    if spans:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc="#9aa0a6", alpha=0.3, label="beam off"))
    if marks:
        handles.append(plt.Line2D([], [], lw=0.9, color=EVENT_COLOR, ls=(0, (2, 2)), label="length change"))
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=len(handles),
               loc="lower right", bbox_to_anchor=(right, 0.08 / total_h))
    fig.text(left, 1 - 0.26 / total_h, title, fontsize=12, va="top")
    fig.text(left, 1 - 0.56 / total_h, subtitle, fontsize=8, color="#5f6368", va="top")
    return fig


def load_traces(ddf_path):
    """(ddf, [force trace, length trace]) as dicts with t, y, units, key, color, label."""
    ddf = read_ddf(ddf_path)
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


def from_trigger(t0, traces, protocol):
    """Traces and protocol steps with time measured from t0."""
    return ([dict(tr, t=tr["t"] - t0) for tr in traces],
            [Step(s.start - t0, s.end - t0, s.kind, s.target, s.params) for s in protocol])


def column_time(meta, ddf=None, frame_time=None):
    """(t0, dt, raw_dt, dt_source, how): column i covers [t0 + i*dt, t0 + (i+1)*dt].

    Without a ddf, t0 = 0 (time from the first frame)."""
    raw_dt = frame_time or detector_frame_time(meta["source"])
    dt_source = "master .h5" if raw_dt and not frame_time else ("given" if frame_time else "default")
    raw_dt = raw_dt or DEFAULT_FRAME_TIME
    t0 = trigger_time(ddf) if ddf is not None else None
    if t0 is not None:
        how = f"first non-zero Trigger ({t0:.3f} s on the DMC clock)"
    else:
        t0 = float(ddf.time[0]) if ddf is not None else 0.0
        how = "start of the trace (no Trigger in protocol)" if ddf is not None else "first frame"
    return t0, raw_dt * meta.get("group", 1), raw_dt, dt_source, how


def kymo_force(npy, ddf_path, pos_range=(0, 1200), stretch=1.4, frame_time=None):
    """Write <base>_force.png, _events.csv and _traces.csv; returns a summary string."""
    kymo = np.load(npy)
    with open(os.path.splitext(npy)[0] + ".json") as f:
        meta = json.load(f)
    n_frames = kymo.shape[1]

    ddf, traces = load_traces(ddf_path)
    t0, dt, raw_dt, dt_source, how = column_time(meta, ddf, frame_time)
    group = meta.get("group", 1)
    traces, protocol = from_trigger(t0, traces, ddf.protocol)
    t0, t_end = 0.0, n_frames * dt

    pos = np.arange(kymo.shape[0]) + int(meta.get("pos_start", 0))
    keep = (pos >= min(pos_range)) & (pos <= max(pos_range))
    if not keep.any():
        raise ValueError(f"pos range {pos_range} keeps no rows (kymograph covers {pos[0]}..{pos[-1]})")
    kymo_c, pos_c = kymo[keep], pos[keep]

    spans = dark_spans(kymo, t0, dt)
    length = next((tr for tr in traces if tr["key"] == "length"), None)
    moves = length_changes(length["t"], length["y"]) if length else []

    base = os.path.splitext(npy)[0]
    rec = os.path.basename(os.path.dirname(meta["source"]))
    title = f"{rec} — kymograph with {ddf.name} mechanics"
    subtitle = (f"{os.path.basename(base)}  ·  positions {pos_c[0]}–{pos_c[-1]} px  ·  "
                f"{n_frames} columns of {group} frame(s) at {raw_dt * 1e3:g} ms ({dt_source})\n"
                f"column 0 aligned to {how}  ·  {stretch:g} display px per position row")
    fig = build_figure(kymo_c, pos_c, t0, dt, traces, spans,
                       [(m["start"], m["end"]) for m in moves],
                       meta.get("inv_nm_per_pixel"), stretch, title, subtitle)
    fig.savefig(base + "_force.png", dpi=DPI, facecolor="white")
    plt.close(fig)

    events = collect_events(moves, length["units"] if length else "", spans, protocol, t0, t_end)
    write_events(base + "_events.csv", events, traces, t0, dt, n_frames)
    write_traces(base + "_traces.csv", traces, t0, dt, n_frames)

    lines = [f"{t0:.3f}–{t_end:.3f} s, {dt * 1e3:g} ms/column; {how}"]
    if spans:
        lines.append("beam off " + ", ".join(f"{a:.3f}–{b:.3f} s" for a, b in spans)
                     + " (should fall in a Trigger gap)")
    lines.append(f"wrote {base}_force.png, _events.csv, _traces.csv")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kymo", help="kymograph .npy written by kymograph.py")
    p.add_argument("ddf", help="Aurora DMC .ddf")
    p.add_argument("--pos-range", nargs=2, type=float, default=(0, 1200), metavar=("LO", "HI"))
    p.add_argument("--stretch", type=float, default=1.4, help="display px per position row")
    p.add_argument("--frame-time", type=float, help="detector frame time in s (default: master .h5)")
    args = p.parse_args()
    print(kymo_force(args.kymo, args.ddf, args.pos_range, args.stretch, args.frame_time))


if __name__ == "__main__":
    main()
