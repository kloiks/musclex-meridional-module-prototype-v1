# User guide: Meridional kymographs

## What it is for

This app shows how the meridional X-ray reflections of muscle (M3, M6, Tn1, …) move and change in strength over time. It can also show the force and length recorded at the same time, and compare up to three recordings.

For each recording it:

1. finds M3 in the first image and uses it to convert pixels to nm;
2. builds a **kymograph**: the strip along the meridian of every image, placed side by side in time;
3. lines up the force and length with the images, if you give a `.ddf` file;
4. finds the peaks, follows them over time and names them.

**You need:** a folder of quadrant-folded TIFF images (`qf_results`) and, optionally, the Aurora `.ddf` force file.

**Words used here:** *d* is the spacing of a reflection in nm. *Radius* is the distance from the image centre in pixels (px). *SNR* is how far a peak stands above the noise. A *track* is one peak followed over time.

## Quick start

1. Start the app: `python kymo_gui.py`
2. Click **QF folder…** and choose the images. Optionally choose a **DDF file…** and an **Output…** folder (default: `kymo` inside the image folder).
3. Fill in row 2 to compare a second recording. **+ Add third folder** adds another.
4. Next to **Process patterns using**, click one of the detectors:
   - **Multiscale peak detection** averages frames to find weak peaks, then follows them down to shorter times.
   - **U-Net peak detection** finds peaks in single frames with a trained neural network. Faster with a graphics card, but works without one.
   - **MaxFlow peak detection** runs multiscale first, then finds peaks in single frames only near the multiscale peaks, and links them over the whole recording at once (min-cost flow). Handles peaks that shift or split better than multiscale.
5. Wait for `done` in the log. The plots appear below it.
6. Check M3 with **Calibration…**. If you change it, run again.

Steps that have not changed are skipped when you run again. **Clear second folder** empties row 2 and removes it from the plots.

## Settings windows

The four buttons are at the top right. **Changes take effect on the next run.** Settings apply to all recordings, except the calibration and the frames to exclude, which are kept per recording.

### Expected peaks…

The list used to name peaks.

| Column | Meaning |
|---|---|
| name | Name shown on the plots |
| d (nm) | Expected spacing |
| tolerance (%) | How far from d a peak may be and still get the name. Grey = the default from **Parameters…** |

A tolerance never reaches past the midpoint between two neighbouring peaks.

### Calibration…

Shows the profile used to find M3. The red dashed line is the M3 in use; the grey dotted line is the one found automatically.

| Control | Meaning |
|---|---|
| Frame | Image to show (here the first is 1) |
| Plot click / px box | Put M3 where you want it. Turn off the plot's zoom and pan before clicking |
| Auto | Use the automatic M3 of the frame shown. On frame 1 this returns to automatic calibration; on any other frame it counts as set by hand |

An M3 set by hand is kept in later runs, unless you change the strip width.

### Background…

| Setting | Default | Meaning |
|---|---|---|
| Background | Convex hull | **Convex hull**: a line stretched under the profile. **Fit**: a smooth curve fitted to the profile |
| Component 2 (Fit only) | Lorentzian | Shape of the second part of the curve: Lorentzian, power law or stretched exponential |
| Oversubtraction penalty (Fit only) | 10 | Higher = the curve is kept more strictly below the data |
| Computed from | Each grouped projection | **Each grouped projection**: the background may change with time. **The averaged profile**: one background for the whole recording |

### Normalization…

Needs a finished run (the recordings in the viewer). It changes what the viewer shows (kymographs, peak heights, background), not the saved files. The kymograph titles show what was applied, e.g. `÷ I0 × 1.172`.

**Within sequence alignment.** Each recording's beamline log (`<run>/<run>.log`, one row per frame with the I0 and I1 counts) is found next to the images; **Browse…** picks another. With **Normalize by I0 incident intensity** on, each frame is multiplied by mean(I0) / I0 of that frame. Beside each recording: how much the frame-to-frame flicker of the kymograph column totals goes down (green) or up (red). With the box off, the text shows what it would do.

**Between sequence alignment.** One factor per recording, geometric mean 1:

| Align by | What is matched |
|---|---|
| None | nothing (default) |
| Least squares | the mean log level of the mean meridional (r 50–599 px) and equatorial (r 50–329 px) profiles, each half |
| Equatorial background, r 90–600 px | the background (hull or fit, as in **Background…**) under the mean equatorial profile, summed |
| Meridional background, r 90–600 px | the same on the meridian |

**Frames used** limits the mean profiles to some frames, e.g. `2-20` for the frames before the stimulus (blank = all kept frames). The panel lists the factors and the spread of the mean levels between recordings before and after. The first time, the equatorial strip is read from the images (`equator_w<W>_g1.npy` in the output folder).

### Parameters…

**Restore defaults** resets everything in this window, including the frames to exclude. Click **Save** to keep it.

**Frames to exclude** (one list per recording, both methods)

Type frames and ranges such as `0-1, 40-60`. Frames are **numbered from 0**. The list starts with the blank frames found; **Auto** goes back to them. An empty list you typed excludes nothing.

**Kymograph**

| Setting | Default | Meaning |
|---|---|---|
| Meridional strip width (px) | 20 | Width of the strip summed in each image |
| M3 search from radius (px) | 250 | Look for M3 only beyond this radius |
| M3 spacing (nm) | 14.34 | Known d of M3 |
| Force figure: range from / to (px) | 0 / 1200 | Part of the kymograph drawn in the saved force figure |
| Force figure: display px per position row | 1.4 | Higher = taller force figure |

**Blank frames (automatic detection)**

| Setting | Default | Meaning |
|---|---|---|
| Blank if column sum < fraction of median | 0.05 | A frame weaker than this fraction of a typical frame is blank |
| Extra frames excluded either side | 1 | Also leave out this many frames before and after |

**Max-flow detection** (guided by multiscale anchors)

| Setting | Default | Meaning |
|---|---|---|
| Detection SNR in single frames | 1.5 | How strong a peak must be in one frame |
| Smoothing (px) | 5 | Smooths each profile along the radius before detection |
| Guide anchors: max resolution scale | 8 | Use multiscale peaks found at this many frames averaged or fewer |
| Guide anchors: r(t) read at scale | 8 | The corridor follows each anchor's position at this scale |
| Corridor half-width around anchors (px) | 8 | Peaks further than this from every anchor are ignored |
| Detection reward: log SNR where it changes sign | 0.3 | Weaker peaks cost, stronger ones pay. Higher = fewer, stronger peaks |
| Max step per frame (px) | 25 | Largest jump allowed between linked peaks |
| Step costing one unit (px) | 1.48 | Bigger = moving is cheaper |
| Missing frames a link may bridge | 20 | Longest gap within a track |
| Cost per missing frame | 2 | Bigger = gaps break tracks sooner |
| Cost of starting / ending a track | 5 / 5 | Bigger = fewer, longer tracks |
| Min detected frames of a track | 10 | Shorter tracks are dropped |

**U-Net detection**

| Setting | Default | Meaning |
|---|---|---|
| Score threshold (0-1) | 0.4 | How sure the network must be. Lower = more peaks, more false ones |
| Max step between linked peaks (px) | 10 | Largest jump allowed within a track |
| Missing frames a track may bridge | 10 | Longest gap within a track |
| Min detected frames of a track | 20 | Shorter tracks are dropped. Lower it for short recordings |

**Multiscale detection**

| Setting | Default | Meaning |
|---|---|---|
| Radius from / to (px) | 40 / 1400 | Where to look for peaks. *Also used by U-Net* |
| Smoothing (px) | 1 | Smooths each profile along the radius; 1 = off. *For U-Net it affects peak heights only* |
| Detection SNR floor | 1.0 | Anything weaker is never considered a peak |
| Noise window (px) | 51 | Size of the region used to measure the noise around each point |
| Noise percentile | 25 | Which part of the local variation is used to estimate the noise; a low value keeps peaks from counting as noise |
| Min peak separation (px) | 3 | Peaks closer than this count as one |

**Multiscale corridor** (how a peak is followed from long averages to single frames)

| Setting | Default | Meaning |
|---|---|---|
| SNR at the coarsest scale (anchors) | 3.0 | Strength needed in the average of the whole recording, where peaks are first found |
| SNR at the finest scale | 1.1 | Strength needed in single frames. Levels in between are graded between the two |
| Corridor half-width (px) | 4 | How far a peak may move from its starting position |
| Scale ratio | 2 | Step between averaging levels: 2 gives 1, 2, 4, 8, … frames (minimum 2) |
| Coverage that counts as resolved | 0.5 | `tracks.csv` uses the shortest averaging at which the peak is found in this fraction of frames |
| Seed anchors at expected peaks | yes | Also look at the expected peak positions, even if nothing shows in the overall average |

**Labelling** (both methods)

| Setting | Default | Meaning |
|---|---|---|
| Default tolerance (% of d) | 1.0 | Used where **Expected peaks…** has no value of its own |
| Min frames of a labelled track | 5 | Shorter tracks are not named |

## The plots

From top to bottom: force and length, kymographs, peak positions, peak heights, background. Each recording has its own colour. Beside the positions, heights and background are **projections**: the intensity profile at the selected time. The dotted line on a multiscale projection is the strength a peak needs.

Time starts at the trigger, or at the first frame without a `.ddf`. The frame time is read from the `*_master.h5` file next to the image folder; without one, **5 ms is assumed**.

| Control | Meaning |
|---|---|
| time | Moves all plots to that time. `<` `>` step one group, `<<` `>>` two |
| Peaks | Show only these, e.g. `M3, Tn`. The start of a name is enough; M1 does not include M10. Capitals matter |
| Grouping scale | Frames averaged in the kymographs, projections and background. With multiscale it also selects which tracks are shown; U-Net and max-flow tracks stay per frame |
| Show unmatched tracks | U-Net and max-flow: show the peaks without a name (X1, X2, …, dashed) |
| Kymograph intensity | Scale (log or counts) and contrast: **min** is white, **max** is black. **Auto** resets |
| Range | Part of the d (nm) or px axis to show. **Apply range** sets it, **Show whole** resets |
| projections x | Scale (log or linear) and limits of the projections. **Apply** sets, **Auto** fits |
| Smooth peak heights | Smooths the height lines over time; default 10 frames, 1 = off |
| Smooth background | Same, for the background lines |

Smoothing changes the plotted lines only, not the saved data. The **Background intensity** section has its own range (default 3–80 nm), which also sets where the background area is measured.

## Saved files

`<W>` is the strip width.

| File | Content |
|---|---|
| `calibration.json`, `calibration_projection.png` | M3 calibration. The picture is not updated when M3 is set by hand |
| `meridian_w<W>_g1.npy`, `.json`, `_log.png` | Kymograph: data, details, picture |
| `meridian_w<W>_g1_force.png`, `.json`, `_events.csv`, `_traces.csv` | With a `.ddf`: figure, its settings, event times, force and length per sample |
| `multiscale_w<W>/`, `unet_w<W>/` or `maxflow_w<W>/` | `tracks.csv` (peak positions and heights), `labels.csv` (names), `run.json` (settings used), `.npz` (data for the plots) |
| `maxflow_w<W>/events.csv` | Splits and merges the max-flow tracker found |
| `excluded_frames.json` | Frames to exclude, if you edited the list |
| `equator_w<W>_g1.npy`, `.json` | Equatorial strip per frame, for **Normalization…** |

## If something looks wrong

| Problem | Try |
|---|---|
| All d values are off, or few peaks are named | Check M3 in **Calibration…** |
| Right peak, wrong or no name | Edit d or the tolerance in **Expected peaks…** |
| Too few peaks | Lower the score threshold (U-Net), the SNR values (multiscale) or the detection SNR / reward (max-flow) |
| Too many false peaks | Raise them |
| A track breaks or is lost | Raise the max step or missing frames (U-Net or max-flow), or the corridor half-width (multiscale or max-flow) |
| Force and images do not line up | Check the `.ddf` and that the `*_master.h5` file is there |

## CLI commands

Run these from the project folder. Add `--help` to any script except `kymo_gui.py` to see its options.

**Install and start the app**

```sh
pip install -r requirements.txt
python kymo_gui.py
```

**Run the steps one by one** (replace the example paths)

```sh
# 1. Calibrate on the first image
python calibrate_m3.py qf_results/frame_00001_folded_compressed.tif --gap 20 --rmin 250 -o cal.json --debug

# 2. Build the kymograph
python kymograph.py qf_results --width 20 --calibration cal.json --out-dir qf_results/kymo

# 3. Add force and length (optional)
python kymo_force.py qf_results/kymo/meridian_w20_g1.npy LoActive1.ddf --pos-range 0 1200 --stretch 1.4

# 4. Find the peaks: use any one
python multiscale.py qf_results/kymo/meridian_w20_g1.npy --peaks config/expected_peaks.csv --params config/parameters.json
python unet.py qf_results/kymo/meridian_w20_g1.npy --peaks config/expected_peaks.csv --params config/parameters.json
python maxflow.py qf_results/kymo/meridian_w20_g1.npy --peaks config/expected_peaks.csv --params config/parameters.json
```

| Script | Option | Meaning |
|---|---|---|
| `calibrate_m3.py` | `--gap` | Strip width in px (default 20; use the same as `--width`) |
| | `--rmin` | Look for M3 beyond this radius (default 250) |
| | `-o` | Output file (default `<image>_calibration.json`) |
| | `--debug` | Also save a picture |
| `kymograph.py` | `--width` | Strip width in px (default 20) |
| | `--calibration` | File from step 1. Needed for d values and peak names |
| | `--out-dir` | Output folder (default `<qf folder>/kymo`) |
| `kymo_force.py` | `--pos-range LO HI` | Part of the kymograph to draw, in px (default 0 1200) |
| | `--stretch` | Figure height (default 1.4) |
| | `--frame-time` | Seconds per frame (default: from `*_master.h5`, else 0.005) |
| `multiscale.py`, `unet.py`, `maxflow.py` | `--peaks` | Expected peaks file. Without it nothing is named |
| | `--params` | Parameters file. Without it the defaults are used |
| | `--exclude` | Frames to leave out, e.g. `"0-1, 40-60"` (default: blank frames). A list edited in the app is not read here |
| | `--out-dir` | Output folder (default: next to the kymograph) |

**Run the tests**

```sh
python -m unittest discover tests -v
python tests/test_outputs.py --update   # accept the current results as the new reference
```
