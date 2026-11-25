# musclex-meridional-module-prototype-v1

M3 calibration, meridional kymographs (optionally with force/length from an Aurora DMC `.ddf`) and multiscale peak tracking for quadrant-folded image folders.

```sh
pip install -r requirements.txt
python kymo_gui.py
```

GUI: pick one to three `qf_results` folders, each with an optional `.ddf` and an output folder (default `<qf folder>/kymo`), then **Create kymographs and process multiscale**. **Clear second folder** empties the second row and removes its kymograph from the viewer without rerunning.
**Run Unet Detection** runs the same steps but finds and links the peaks with a U-Net (`models/new_w61.pt`, needs PyTorch; GPU if available) in place of multiscale.
**Grouping scale** in the viewer offers 1, 2, 4, 8 and 16 frames for both methods (plus multiscale's coarser scales): the kymographs, projections and background show the sliding mean of that many frames. U-Net tracks and heights stay per frame.
Steps whose images, settings and inputs are unchanged are not repeated (hashes in the `.json` files, see `cache.py`).
U-Net tracks that match no expected peak are named X1, X2, … from the centre out in each recording and drawn dashed; **Show unmatched tracks** hides them. **Smooth peak heights** and **Smooth background** (both methods, 10 frames by default, 1 = off) smooth the plotted lines only.
The **Background intensity** section at the bottom has its own range (3–80 nm by default) and projection axis controls, not linked to the others; its range also sets where the area under the background is computed. It shares the time sliders and the grouping.
**Parameters…** also lists the **frames to exclude** per selected recording (both methods), numbered from 0 and filled in with the blank frames found; an edited list is kept in `<output>/excluded_frames.json` (**Auto** goes back).
Beside the path fields: **Expected peaks…** edits `config/expected_peaks.csv` (names and d-spacings used to label tracks),
**Parameters…** edits `config/parameters.json` (strip width, M3 search radius, force-figure range and the multiscale settings), and **Calibration…** sets the
M3 radius (px → nm) per recording on a background-subtracted projection of any frame; a value set there is kept by later runs.

Per folder it writes:

| file | from |
|---|---|
| `calibration.json`, `calibration_projection.png` | M3 calibration on the first frame (strip width = kymograph width) |
| `meridian_w<W>_g1.npy/.json`, `_log.png` | kymograph, positive half, one column per frame |
| `meridian_w<W>_g1_force.png/.json`, `_events.csv`, `_traces.csv` | only with a `.ddf`; column 0 aligned to the first non-zero Trigger |
| `multiscale_w<W>/` `tracks.csv`, `labels.csv`, `multiscale.npz`, `run.json` | multiscale tracking on the per-frame kymograph, beam-off frames dropped; anchors named after the expected peaks |
| `unet_w<W>/` `tracks.csv`, `labels.csv`, `unet.npz`, `run.json` | U-Net detection (tracks of one reflection that share no frame may share a name) |

The scripts also run on their own:

```sh
python calibrate_m3.py frame_00001_folded_compressed.tif --gap 20 --rmin 250 --debug
python kymograph.py qf_results --width 20 --calibration cal.json
python multiscale.py qf_results/kymo/meridian_w20_g1.npy --peaks config/expected_peaks.csv --params config/parameters.json
python unet.py qf_results/kymo/meridian_w20_g1.npy --peaks config/expected_peaks.csv --exclude "0-1, 40-60"
python kymo_force.py qf_results/kymo/meridian_w20_g1.npy LoActive1.ddf --pos-range 0 1200 --stretch 1.4
```

Tests (two recordings against `tests/reference/`; the U-Net files are compared only on a GPU):

```sh
python -m unittest discover tests -v
python tests/test_outputs.py --update   # accept the current outputs as the new reference
```

Code:

| file | job |
|---|---|
| `kymo_gui.py` | main window and the worker thread |
| `dialogs.py`, `viewer.py` | the dialogs; the viewer, with its `controls.py`, `plots.py` and data (`recording.py`) |
| `params.py` | parameters and config paths |
| `calibrate_m3.py`, `kymograph.py` | M3 calibration; images to kymograph |
| `ddf.py`, `kymo_force.py` | reading the `.ddf`; force figure and CSVs |
| `multiscale.py`, `unet.py` | the two trackers; `tracking.py` is what they share |
| `background.py`, `labels.py`, `frames.py`, `cache.py` | backgrounds and noise; naming tracks; excluded frames; data hashes |
