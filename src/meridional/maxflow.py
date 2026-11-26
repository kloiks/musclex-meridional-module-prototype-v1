#!/usr/bin/env python3
"""Anchor-guided max-flow peak tracking on a per-frame meridional kymograph (alternative to
multiscale.py and unet.py).

Port of the ``guided_maxflow`` tracker of meridional-simulation
(src/track-detection/max-flow/flow_kymo_tracks.py with anchor_guide.py), with the settings of
its shifts-and-doublets comparison as defaults.

1. Guide: multiscale.py is run (or its saved run reused) into the sibling multiscale_w<W>/
   folder.  Anchors with a resolution scale of at most ``maxflow_guide_max_res`` are kept; each
   one's r(t) is read at the multiscale scale nearest ``maxflow_guide_scale``, interpolated over
   the frames where it was not found, so the corridor moves with the peak.
2. Detection: peaks of every single frame (baseline and noise as multiscale, smoothing
   ``maxflow_smooth``) with SNR >= ``maxflow_min_snr``, kept only within ``maxflow_corridor`` px
   of a kept anchor.  Frame detection needs a low threshold for faint peaks; the corridors
   stop that low threshold from letting noise through everywhere.
3. Linking: one min-cost flow over the whole recording (Zhang, Li & Nevatia 2008).  Every
   detection is a node; a track is a unit of flow source -> detections -> sink, costing

       c_enter + sum c_det + sum c_link + c_exit,   c_det = -(log SNR - det_half) / det_scale

   with c_link = (dr / (shift_scale sqrt(k)))^2 + log(height ratio)^2 + 0.5 log(width ratio)^2
   + c_miss per skipped frame (k - 1, at most max_gap).  Split and merge arcs (a birth out of,
   or death into, a neighbouring peak, priced by intensity conservation) make a peak that
   splits off or merges cheaper than a new track.  Tracks are added cheapest first while they
   lower the total cost, so the number of tracks is not a parameter.  Tracks with fewer than
   ``maxflow_min_length`` detections are dropped.

Excluded frames (frames.py) are cut out and the rest joined, as for the other trackers.
Tracks are named after the expected peaks as for the U-Net (tracks that share no frame may
share a name; the others are X1, X2, ... from the centre out).

Writes to ``out_dir``: run.json, maxflow.npz, tracks.csv, labels.csv, events.csv (splits/merges).

Example:
    python maxflow.py qf_results/kymo/meridian_w20_g1.npy --peaks config/expected_peaks.csv
"""
import csv
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import cache
from . import multiscale
from . import tracking
from .background import shared_background
from .frames import format_frames
from .labels import label_tracks, read_expected_peaks, unmatched_names, write_labels

# guided_maxflow in meridional-simulation/sh_scripts/create_simulations_track.sh
DEFAULTS = dict(
    maxflow_min_snr=1.5, maxflow_smooth=5, maxflow_guide_max_res=8, maxflow_guide_scale=8,
    maxflow_corridor=8.0, maxflow_det_half=0.3, maxflow_max_shift=25.0, maxflow_shift_scale=1.48,
    maxflow_max_gap=20, maxflow_c_miss=2.0, maxflow_c_enter=5.0, maxflow_c_exit=5.0,
    maxflow_min_length=10,
)
# fixed (flow_kymo_tracks.py defaults or the guided_maxflow settings)
W_SHIFT, W_AMP, W_WIDTH, W_DET, MAX_COST = 1.0, 1.0, 0.5, 1.0, 25.0
C_SPLIT, C_MERGE, W_CONSERVE = 5.25, 5.25, 8.0
MAX_TRACKS = 5000
EPS = 1e-9  # added to every reduced cost: Dijkstra on a sparse matrix drops zero-weight edges
# the keys that change the tracking; the guide counts through its multiscale.npz hash
KEYS = list(DEFAULTS) + ["rmin", "rmax", "background", "bg_method", "bg_comp2", "bg_neg_weight",
                         "noise_win", "noise_pct", "min_sep"]


# ------------------------------------------------------------------ guide

def guide_dir_for(out_dir):
    """The multiscale folder beside ``out_dir`` (maxflow_w20 -> multiscale_w20)."""
    name = os.path.basename(os.path.normpath(out_dir))
    suffix = name[len("maxflow"):] if name.startswith("maxflow") else "_guide"
    return os.path.join(os.path.dirname(os.path.normpath(out_dir)), "multiscale" + suffix)


def load_guide(guide_dir, p, n_frames):
    """(kept anchor x kept frame) r(t) of the anchors that guide, and the number of anchors."""
    run = cache.read_json(os.path.join(guide_dir, "run.json"))
    with np.load(os.path.join(guide_dir, "multiscale.npz")) as data:
        scales, r = data["scales"], data["r"]
    if r.shape[2] != n_frames:
        raise ValueError(f"the multiscale guide has {r.shape[2]} frames, the max-flow run {n_frames}")
    res = np.asarray(run["resolution_scale"])
    kept = np.flatnonzero((res > 0) & (res <= p["maxflow_guide_max_res"]))
    at = r[int(np.argmin(np.abs(scales - p["maxflow_guide_scale"])))]
    frames = np.arange(n_frames)
    ref = np.zeros((len(kept), n_frames))
    for i, a in enumerate(kept):
        seen = np.flatnonzero(np.isfinite(at[a]))
        ref[i] = np.interp(frames, seen, at[a, seen]) if len(seen) else run["anchor_r"][a]
    return ref, len(res)


# ------------------------------------------------------------------ detection

def detect_frames(kymo, pos, p, ref):
    """Detections inside the corridors, as one dict of arrays in frame order (``frame`` = column)."""
    q = dict(p, smooth=p["maxflow_smooth"], snr_floor=p["maxflow_min_snr"])
    det = multiscale.detect_level(kymo, pos, q, shared_background(kymo, pos, q))
    if len(ref):
        inside = (np.abs(det["r"] - ref[:, det["t"]]) <= p["maxflow_corridor"]).any(axis=0)
    else:
        inside = np.zeros(len(det["r"]), dtype=bool)
    order = np.argsort(det["t"], kind="stable")
    order = order[inside[order]]
    out = {k: v[order] for k, v in det.items()}
    out["frame"] = out.pop("t")
    return out, int((~inside).sum())


# ------------------------------------------------------------------ graph

def arc_costs(dr, h_i, h_j, spread, p):
    """(split cost, merge cost, gate) between one frame's peaks (i) and a later frame's (j).

    No amplitude or width terms (a blob of two peaks is brighter and wider than either); instead
    intensity conservation: h(parent, t) ~ h(parent, t+k) + h(child, t+k), the parent's own
    continuation taken as the nearest later peak other than the child (mirrored for a merge)."""
    n_i, n_j = dr.shape
    absdr = np.abs(dr)
    gate = absdr <= p["maxflow_max_shift"] * spread
    shift = W_SHIFT * (dr / (p["maxflow_shift_scale"] * spread)) ** 2
    m = np.where(gate, absdr, np.inf)

    def second_nearest(m, h, axis):
        """Per row (axis 1) or column (axis 0): height of the nearest partner other than this one."""
        order = np.argsort(m, axis=axis, kind="stable")
        n_other = m.shape[1 - axis]
        idx = np.arange(n_other)
        near1 = order[0] if axis == 0 else order[:, 0]
        d1 = m[near1, idx] if axis == 0 else m[idx, near1]
        h1 = np.where(np.isfinite(d1), h[near1], 0.0)
        if m.shape[axis] > 1:
            near2 = order[1] if axis == 0 else order[:, 1]
            d2 = m[near2, idx] if axis == 0 else m[idx, near2]
            h2 = np.where(np.isfinite(d2), h[near2], 0.0)
        else:
            h2 = np.zeros(n_other)
        if axis == 1:
            return np.where(np.arange(m.shape[1])[None, :] == near1[:, None], h2[:, None], h1[:, None])
        return np.where(np.arange(m.shape[0])[:, None] == near1[None, :], h2[None, :], h1[None, :])

    h_keep = second_nearest(m, h_j, 1)  # what the parent keeps after the split
    h_was = second_nearest(m, h_i, 0)   # what the survivor carried before the merge
    hi, hj = h_i[:, None], h_j[None, :]
    c_split = C_SPLIT + shift + W_CONSERVE * np.log(np.maximum(hi, EPS) / np.maximum(h_keep + hj, EPS)) ** 2
    c_merge = C_MERGE + shift + W_CONSERVE * np.log(np.maximum(hi + h_was, EPS) / np.maximum(hj, EPS)) ** 2
    return c_split, c_merge, gate


def build_graph(det, n_frames, p):
    """Edges (tail, head, cost) and the bookkeeping to read the flow back.

    Node 0 is the source, 1 the sink; detection d owns 2 + 2d (in) and 3 + 2d (out).  The
    detections are in frame order, so the numbering is topological."""
    n = len(det["r"])
    u = 2 + 2 * np.arange(n)
    v = u + 1
    score = np.log(np.maximum(det["snr"], EPS))
    lo, hi = np.percentile(score, [25, 75])
    c_det = -W_DET * (score - p["maxflow_det_half"]) / max((hi - lo) / 2.0, 1e-6)

    cum_miss = np.concatenate([[0.0], np.cumsum(np.full(n_frames, p["maxflow_c_miss"]))])
    by_frame = [np.flatnonzero(det["frame"] == f) for f in range(n_frames)]
    links, splits, merges = [], [], []  # (i, j, cost) arrays
    for t in range(n_frames):
        ii = by_frame[t]
        if not ii.size:
            continue
        for k in range(1, p["maxflow_max_gap"] + 2):
            if t + k >= n_frames:
                break
            jj = by_frame[t + k]
            if not jj.size:
                continue
            spread = np.sqrt(k)  # a random walk's: further after a gap, but not k times
            dr = det["r"][jj][None, :] - det["r"][ii][:, None]
            cost = W_SHIFT * (dr / (p["maxflow_shift_scale"] * spread)) ** 2
            for key, w in (("height", W_AMP), ("fwhm", W_WIDTH)):
                ratio = det[key][jj][None, :] / np.maximum(det[key][ii][:, None], EPS)
                cost = cost + w * np.log(np.maximum(ratio, EPS)) ** 2
            cost = cost + cum_miss[t + k] - cum_miss[t + 1]
            c_split, c_merge, gate = arc_costs(dr, det["height"][ii], det["height"][jj], spread, p)
            for out, c in ((links, cost), (splits, c_split), (merges, c_merge)):
                a, b = np.nonzero(gate & (c <= MAX_COST))
                if a.size:
                    out.append((ii[a], jj[b], c[a, b]))

    def stack(parts):
        if not parts:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        return tuple(np.concatenate(x) for x in zip(*parts))

    (li, lj, lc), (si, sj, sc), (mi, mj, mc) = stack(links), stack(splits), stack(merges)
    # a split arc is an entry that knows its parent, a merge arc an exit that knows its target
    tail = np.concatenate([np.zeros(n, int), u, v, v[li], np.zeros(len(sc), int), v[mi]])
    head = np.concatenate([u, v, np.ones(n, int), u[lj], u[sj], np.ones(len(mc), int)])
    cost = np.concatenate([np.full(n, p["maxflow_c_enter"]), c_det, np.full(n, p["maxflow_c_exit"]),
                           lc, sc, mc])
    bounds = np.cumsum([0, n, n, n, len(lc), len(sc), len(mc)])
    kinds = dict(zip(("enter", "det", "exit", "link", "split", "merge"), zip(bounds[:-1], bounds[1:])))
    return dict(tail=tail, head=head, cost=cost, n_nodes=2 + 2 * n, kinds=kinds,
                link=(li, lj), split=(si, sj), merge=(mi, mj))


# ------------------------------------------------------------------ min-cost flow

def dag_potentials(tail, head, cost, n_nodes):
    """Shortest distances from the source by one sweep in topological order (costs may be < 0)."""
    order = np.argsort(tail, kind="stable")
    starts = np.searchsorted(tail[order], np.arange(n_nodes), "left")
    ends = np.searchsorted(tail[order], np.arange(n_nodes), "right")
    dist = np.full(n_nodes, np.inf)
    dist[0] = 0.0
    for node in np.concatenate([[0], np.arange(2, n_nodes), [1]]):
        if np.isfinite(dist[node]) and ends[node] > starts[node]:
            out = order[starts[node]:ends[node]]
            np.minimum.at(dist, head[out], dist[node] + cost[out])
    return dist


def min_cost_flow(g):
    """0/1 flow per edge: successive shortest paths (Johnson potentials), one track per path,
    stopping at the first path that would not lower the total cost."""
    tail, head, cost, n_nodes = g["tail"], g["head"], g["cost"], g["n_nodes"]
    n_edges = len(tail)
    flow = np.zeros(n_edges, dtype=np.int8)
    pi = dag_potentials(tail, head, cost, n_nodes)
    if not np.isfinite(pi[1]):
        return flow
    pi[~np.isfinite(pi)] = pi[np.isfinite(pi)].max()
    a_tail, a_head = np.concatenate([tail, head]), np.concatenate([head, tail])
    a_cost = np.concatenate([cost, -cost])
    a_edge = np.tile(np.arange(n_edges), 2)
    a_sign = np.repeat(np.array([1, -1], np.int8), n_edges)
    for _ in range(MAX_TRACKS):
        live = np.concatenate([flow == 0, flow == 1])  # forward below capacity, backward if used
        rt, rh = a_tail[live], a_head[live]
        red = np.maximum(a_cost[live] + pi[rt] - pi[rh], 0.0) + EPS
        r_edge, r_sign = a_edge[live], a_sign[live]
        # parallel arcs between one node pair (a link and a split/merge arc): keep the cheapest
        key = rt.astype(np.int64) * n_nodes + rh
        order = np.lexsort((red, key))
        first = np.r_[True, key[order][1:] != key[order][:-1]]
        sel, ukey = order[first], key[order][first]
        dist, pred = dijkstra(csr_matrix((red[sel], (rt[sel], rh[sel])), shape=(n_nodes, n_nodes)),
                              directed=True, indices=0, return_predecessors=True)
        if not np.isfinite(dist[1]) or dist[1] + pi[1] - pi[0] >= -1e-9:
            break
        pi = pi + np.minimum(dist, dist[1])
        node = 1
        while node != 0:
            prev = pred[node]
            e = sel[np.searchsorted(ukey, np.int64(prev) * n_nodes + node)]
            flow[r_edge[e]] += r_sign[e]
            node = prev
    return flow


def extract_tracks(flow, g, n_det, min_length):
    """[(detections of the track, split parent or None, merge target or None)], longest-lived order."""
    def used(kind):
        lo, hi = g["kinds"][kind]
        return np.flatnonzero(flow[lo:hi] > 0)

    nxt = dict(zip(g["link"][0][used("link")].tolist(), g["link"][1][used("link")].tolist()))
    died_into = dict(zip(g["merge"][0][used("merge")].tolist(), g["merge"][1][used("merge")].tolist()))
    starts = [(int(d), None) for d in used("enter")]
    starts += [(int(c), int(par)) for c, par in zip(g["split"][1][used("split")], g["split"][0][used("split")])]
    out = []
    for d, parent in starts:
        chain = [d]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        if len(chain) >= min_length:
            out.append((np.array(chain), parent, died_into.get(chain[-1])))
    return out


def arc_events(tracks, det):
    """Splits and merges whose partner detection belongs to a track on the other side of it."""
    owner = {int(d): a for a, (chain, _, _) in enumerate(tracks) for d in chain}
    span = [(det["frame"][c[0]], det["frame"][c[-1]]) for c, _, _ in tracks]
    events = []
    for a, (chain, parent, target) in enumerate(tracks):
        for kind, d in (("split", parent), ("merge", target)):
            b = owner.get(d) if d is not None else None
            if b is None or b == a:
                continue
            f = int(det["frame"][d])
            if (kind == "split" and span[b][1] > f) or (kind == "merge" and span[b][0] < f):
                events.append(dict(type=kind, frame=int(det["frame"][chain[0]]) if kind == "split" else f,
                                   track=a, partner=b, r=float(det["r"][d])))
    return sorted(events, key=lambda e: (e["frame"], e["type"]))


# ------------------------------------------------------------------ run

def track(inp, p, guide_dir):
    """(r, height) as (track x kept frame) with NaN where not detected, events, facts for run.json."""
    keep = (inp.pos >= p["rmin"]) & (inp.pos <= p["rmax"])
    kymo, pos = inp.kymo[keep][:, inp.frames], inp.pos[keep]
    n = kymo.shape[1]
    ref, n_anchors = load_guide(guide_dir, p, n)
    det, n_outside = detect_frames(kymo, pos, p, ref)
    tracks = []
    if len(det["r"]):
        g = build_graph(det, n, p)
        tracks = extract_tracks(min_cost_flow(g), g, len(det["r"]), p["maxflow_min_length"])
    tracks.sort(key=lambda tr: (det["frame"][tr[0][0]], det["r"][tr[0][0]]))
    r, height = np.full((2, len(tracks), n), np.nan)
    for a, (chain, _, _) in enumerate(tracks):
        r[a, det["frame"][chain]] = det["r"][chain]
        height[a, det["frame"][chain]] = det["height"][chain]
    facts = dict(detector="maxflow", n_guide_anchors=len(ref), n_anchors=n_anchors,
                 n_detections=len(det["r"]), n_outside_corridors=n_outside)
    return r, height, arc_events(tracks, det), facts


def write_tracks(path, r, height, names, inp):
    """One row per frame from a track's first to last detection; gaps interpolated (gap = True)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "label", "frame", "r", "d_nm", "height", "gap"])
        for a in range(len(r)):
            seen = np.flatnonzero(np.isfinite(r[a]))
            span = np.arange(seen[0], seen[-1] + 1)
            for t, x in zip(span, np.interp(span, seen, r[a, seen])):
                gap = not np.isfinite(r[a, t])
                w.writerow([a, names[a], int(inp.frames[t])] + [f"{v:.5g}" for v in (x, inp.d_nm(x), height[a, t])]
                           + [gap])


def write_events(path, events, inp):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "frame", "track", "partner", "r", "d_nm"])
        for e in events:
            w.writerow([e["type"], int(inp.frames[e["frame"]]), e["track"], e["partner"],
                        f"{e['r']:.5g}", f"{inp.d_nm(e['r']):.5g}"])


def process(npy, out_dir, params=None, peaks_csv=None, exclude=None):
    """Guide (multiscale), detect, link (or reuse the saved result in out_dir) and label;
    returns (summary, cached)."""
    p = dict(multiscale.DEFAULTS, **DEFAULTS)
    p.update(params or {})
    guide_dir = guide_dir_for(out_dir)
    guide_summary, _ = multiscale.process(npy, guide_dir, p, peaks_csv, exclude)
    inp = tracking.load_input(npy, p, exclude)
    key = dict({k: p[k] for k in KEYS}, **inp.key(), detector="maxflow",
               guide_sha256=cache.file_hash(os.path.join(guide_dir, "multiscale.npz")))

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "maxflow.npz")
    run = tracking.saved_run(out_dir, key, ("maxflow.npz", "events.csv"))
    cached = run is not None
    if cached:
        with np.load(npz) as data:
            r, height = data["r"][0], data["height"][0]
    else:
        r, height, events, facts = track(inp, p, guide_dir)
        # one "scale" (per frame), in the layout the viewer reads
        np.savez_compressed(npz, scales=[1], frames=inp.frames, r=r[None], height=height[None])
        write_events(os.path.join(out_dir, "events.csv"), events, inp)
        run = dict(key, **facts, n_events=len(events))
        tracking.save_run(out_dir, run)

    rows = [dict(track=int(a), frame=int(inp.frames[t]), r=float(r[a, t])) for a, t in zip(*np.nonzero(np.isfinite(r)))]
    peaks = read_expected_peaks(peaks_csv) if peaks_csv else []
    labels = label_tracks(rows, inp.inv_nm, peaks, p["tol_pct"], p["min_frames"], share_names=True)
    write_labels(out_dir, labels)
    names = {**{a: v[0] for a, v in labels.items()}, **unmatched_names(r, labels)}
    write_tracks(os.path.join(out_dir, "tracks.csv"), r, height, names, inp)
    summary = (f"guide: {guide_summary.split(';')[0]}\n"
               f"max-flow: {run['n_guide_anchors']} of {run['n_anchors']} anchors guide (+-{p['maxflow_corridor']:g} px), "
               f"{run['n_detections']} detections kept, {run['n_outside_corridors']} outside; {len(r)} tracks, "
               f"{run['n_events']} split/merge events"
               + (f", excluded frames {format_frames(inp.dropped)}" if inp.dropped else "")
               + f"; labelled {len(labels)}, unmatched {len(r) - len(labels)}: "
               + ", ".join(sorted({v[0] for v in labels.values()})))
    return summary, cached


def main():
    tracking.run_cli(process, "maxflow", __doc__)


if __name__ == "__main__":
    main()
