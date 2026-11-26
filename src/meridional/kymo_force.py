#!/usr/bin/env python3
"""Kymograph with the force and length traces of an Aurora DMC ``.ddf``.

Times (figure, CSVs) count from the trigger, when column 0 opens (ddf.py).  Beam-off columns
are shaded; they should fall in a Trigger gap.

Writes, next to the kymograph:
    <base>_force.png     kymograph over force and length
    <base>_force.json    what they were made from (an unchanged figure is not made again)
    <base>_events.csv    kymograph span, beam-off spans, measured length changes and protocol steps
    <base>_traces.csv    force/length samples with the column each falls in

Example:
    python kymo_force.py qf_results/kymo/meridian_w20_g1.npy LoActive1.ddf --pos-range 0 1200 --stretch 1.4
"""
import argparse
import csv
import json
import os
import re

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from . import cache
from .ddf import column_time, detector_frame_time, from_trigger, load_traces
from .frames import blank_frames
from .kymograph import load_kymograph, recording_name

CACHE_VERSION = 2           # raise when the figure or CSVs change, so saved ones are made again
DARK_FRACTION = 0.05        # of the median column sum => beam off (default; the GUI's dark_fraction)
DPI = 150
SPAN_COLOR, EVENT_COLOR = "#d93025", "#8430ce"
DASHED, DOTTED = (0, (4, 3)), (0, (2, 2))

# length-change detection
EVENT_SMOOTH = 0.002        # s moving average under the derivative
EVENT_SLOPE_FRACTION = 0.25 # of the fastest change => lever is moving
EVENT_MIN_AMPLITUDE = 0.10  # of the largest move
EVENT_GAP = 0.010           # s of stillness separating two moves
EVENT_PLATEAU = 0.020       # s of plateau read either side of a move
EVENT_NOISE_FLOOR = 20      # largest move must exceed this many x sample noise


# ------------------------------------------------------------------ events

def dark_spans(kymo, dt, dark_fraction):
    """(start, stop) seconds of runs of columns with no beam (frames.blank_frames, not padded)."""
    dark = np.flatnonzero(blank_frames(np.nansum(kymo, axis=0), dark_fraction, 0))
    runs = np.split(dark, np.flatnonzero(np.diff(dark) > 1) + 1) if dark.size else []
    return [(run[0] * dt, (run[-1] + 1) * dt) for run in runs]


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


def collect_events(moves, units, spans, protocol, t_end):
    """[(name, time, source, detail), ...] sorted by time."""
    ev = [("kymograph start", 0.0, "kymograph", "column 0 opens"),
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


def _frame_of(when, dt, n):
    """(column or "", exact column) of a time."""
    exact = when / dt
    return (int(np.floor(exact)) if 0 <= exact < n else ""), exact


def _fmt(v):
    return "" if not np.isfinite(v) else f"{v:.6g}"


def _at(trace, when):
    return np.interp(when, trace["t"], trace["y"], left=np.nan, right=np.nan)


def write_traces(path, traces, dt, n):
    times = traces[0]["t"]
    values = [_at(tr, times) for tr in traces]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "frame", "frame_exact"] + [_column_name(tr) for tr in traces])
        for i, when in enumerate(times):
            frame, exact = _frame_of(when, dt, n)
            w.writerow([f"{when:.6f}", frame, f"{exact:.3f}"] + [_fmt(v[i]) for v in values])


def write_events(path, events, traces, dt, n):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "source", "time_s", "frame", "frame_exact"]
                   + [_column_name(tr) for tr in traces] + ["detail"])
        for name, when, source, detail in events:
            frame, exact = _frame_of(when, dt, n)
            w.writerow([name, source, f"{when:.6f}", frame, f"{exact:.3f}"]
                       + [_fmt(_at(tr, when)) for tr in traces] + [detail])


# ------------------------------------------------------------------ figure

def draw_kymograph(fig, ax, kymo, pos, dt, inv_nm):
    shown = np.log1p(np.clip(kymo, 0, None))
    lo, hi = np.percentile(shown[np.isfinite(shown)], (1.0, 99.5))
    image = ax.imshow(shown, origin="lower", aspect="auto", cmap="viridis", vmin=lo,
                      vmax=hi if hi > lo else lo + 1, interpolation="nearest",
                      extent=(0.0, kymo.shape[1] * dt, pos[0] - 0.5, pos[-1] + 0.5))
    ax.set_ylabel("axial position from centre (pixels)", fontsize=9)
    ax.tick_params(labelsize=8, labelbottom=False)
    top = ax.secondary_xaxis("top", functions=(lambda t: t / dt, lambda f: f * dt))
    top.set_xlabel("column", fontsize=9)
    top.tick_params(labelsize=8)
    if inv_nm:
        right = ax.secondary_yaxis("right", functions=(lambda p: p * inv_nm, lambda q: q / inv_nm))
        right.set_ylabel("q (nm$^{-1}$)", fontsize=9)
        right.tick_params(labelsize=8)
    box = ax.get_position()
    bar = fig.colorbar(image, cax=fig.add_axes([0.895, box.y0, 0.014, box.height]))
    bar.set_label("log(1+I) (sum over strip)", fontsize=8)
    bar.ax.tick_params(labelsize=7)


def draw_trace(ax, trace, t_end):
    t, y = trace["t"], trace["y"]
    ax.plot(t, y, lw=1.4, color=trace["color"])
    inside = y[(t >= 0.0) & (t <= t_end) & np.isfinite(y)]
    if inside.size:  # scale to the visible window, not the whole trace
        ymin, ymax = inside.min(), inside.max()
        pad = 0.09 * (ymax - ymin) or max(abs(ymax) * 0.05, 1e-9)
        ax.set_ylim(ymin - pad, ymax + pad)
    for edge in (0.0, t_end):
        ax.axvline(edge, lw=0.9, color=SPAN_COLOR, ls=DASHED)
    ax.set_ylabel(trace["label"], fontsize=9)
    ax.grid(axis="y", lw=0.6, color="#e9e9e9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, labelbottom=False)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def draw_marks(ax_kymo, trace_axes, spans, marks):
    """Beam-off spans and length changes."""
    for a, b in spans:
        ax_kymo.axvspan(a, b, fc="none", ec=SPAN_COLOR, lw=0.8, ls=DASHED, zorder=3)
        for ax in trace_axes:
            ax.axvspan(a, b, color="#9aa0a6", alpha=0.18, lw=0, zorder=0)
    for a, b in marks:
        for ax in trace_axes:
            ax.axvspan(a, b, color=EVENT_COLOR, alpha=0.10, lw=0)
        for ax in [ax_kymo] + trace_axes:
            for edge in (a, b):
                ax.axvline(edge, lw=0.9, color=EVENT_COLOR, ls=DOTTED, zorder=4)


def draw_legend(fig, spans, marks, anchor):
    handles = [Line2D([], [], lw=0.9, color=SPAN_COLOR, ls=DASHED, label="kymograph span")]
    if spans:
        handles.append(Rectangle((0, 0), 1, 1, fc="#9aa0a6", alpha=0.3, label="beam off"))
    if marks:
        handles.append(Line2D([], [], lw=0.9, color=EVENT_COLOR, ls=DOTTED, label="length change"))
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=len(handles),
               loc="lower right", bbox_to_anchor=anchor)


def build_figure(kymo, pos, dt, traces, spans, marks, inv_nm, stretch, title, subtitle):
    """Kymograph over the traces on one time axis; kymograph height = rows * stretch px."""
    t_end = kymo.shape[1] * dt
    top_pad, bottom_pad, trace_h = 1.30, 0.80, 1.35
    kymo_h = max(2.0, kymo.shape[0] * stretch / DPI)
    total_h = kymo_h + trace_h * len(traces) + top_pad + bottom_pad
    left, right = 0.075, 0.845

    fig = Figure(figsize=(13, total_h), dpi=DPI)
    grid = fig.add_gridspec(1 + len(traces), 1, height_ratios=[kymo_h] + [trace_h] * len(traces),
                            hspace=0.12, left=left, right=right,
                            top=1 - top_pad / total_h, bottom=bottom_pad / total_h)
    ax_kymo = fig.add_subplot(grid[0])
    trace_axes = [fig.add_subplot(grid[i + 1], sharex=ax_kymo) for i in range(len(traces))]
    draw_kymograph(fig, ax_kymo, kymo, pos, dt, inv_nm)
    for ax, trace in zip(trace_axes, traces):
        draw_trace(ax, trace, t_end)
    draw_marks(ax_kymo, trace_axes, spans, marks)
    trace_axes[-1].tick_params(labelbottom=True)
    trace_axes[-1].set_xlabel("time from trigger (s)", fontsize=9)
    ax_kymo.set_xlim(0.0, t_end)
    draw_legend(fig, spans, marks, (right, 0.08 / total_h))
    fig.text(left, 1 - 0.26 / total_h, title, fontsize=12, va="top")
    fig.text(left, 1 - 0.56 / total_h, subtitle, fontsize=8, color="#5f6368", va="top")
    return fig


# ------------------------------------------------------------------ run

def kymo_force(npy, ddf_path, pos_range=(0, 1200), stretch=1.4, frame_time=None, dark_fraction=DARK_FRACTION):
    """Write <base>_force.png, _events.csv and _traces.csv, or keep them if <base>_force.json says
    they were made from the same data and settings; returns a summary string."""
    kymo, pos, meta = load_kymograph(npy)
    base = os.path.splitext(npy)[0]
    key = dict(version=CACHE_VERSION, kymo_sha256=cache.array_hash(kymo), meta=meta, ddf=os.path.abspath(ddf_path),
               ddf_sha256=cache.file_hash(ddf_path), pos_range=list(pos_range), stretch=stretch,
               dark_fraction=dark_fraction, frame_time=frame_time,
               detector_frame_time=detector_frame_time(meta["source"]))
    saved = cache.read_json(base + "_force.json")
    if cache.matches(saved, key) and "summary" in saved and all(
            os.path.exists(base + end) for end in ("_force.png", "_events.csv", "_traces.csv")):
        return saved["summary"] + " (cached)"

    keep = (pos >= min(pos_range)) & (pos <= max(pos_range))
    if not keep.any():
        raise ValueError(f"pos range {pos_range} keeps no rows (kymograph covers {pos[0]}..{pos[-1]})")
    summary = write_outputs(base, kymo, pos, keep, meta, ddf_path, stretch, frame_time, dark_fraction)
    with open(base + "_force.json", "w") as f:
        json.dump(dict(key, summary=summary), f, indent=2)
    return summary


def write_outputs(base, kymo, pos, keep, meta, ddf_path, stretch, frame_time, dark_fraction):
    """The figure (of the rows ``keep``) and the CSVs; returns the summary."""
    ddf, traces = load_traces(ddf_path)
    time = column_time(meta, ddf, frame_time)
    traces, protocol = from_trigger(time.trigger, traces, ddf.protocol)
    n, dt = kymo.shape[1], time.dt
    spans = dark_spans(kymo, dt, dark_fraction)
    length = next((tr for tr in traces if tr["key"] == "length"), None)
    moves = length_changes(length["t"], length["y"]) if length else []

    title = f"{recording_name(meta['source'])} — kymograph with {ddf.name} mechanics"
    subtitle = (f"{os.path.basename(base)}  ·  positions {pos[keep][0]}–{pos[keep][-1]} px  ·  "
                f"{n} frames at {dt * 1e3:g} ms ({time.dt_source})\n"
                f"column 0 aligned to {time.how}  ·  {stretch:g} display px per position row")
    fig = build_figure(kymo[keep], pos[keep], dt, traces, spans, [(m["start"], m["end"]) for m in moves],
                       meta.get("inv_nm_per_pixel"), stretch, title, subtitle)
    fig.savefig(base + "_force.png", dpi=DPI, facecolor="white")
    events = collect_events(moves, length["units"] if length else "", spans, protocol, n * dt)
    write_events(base + "_events.csv", events, traces, dt, n)
    write_traces(base + "_traces.csv", traces, dt, n)

    lines = [f"0.000–{n * dt:.3f} s, {dt * 1e3:g} ms/column; {time.how}"]
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
