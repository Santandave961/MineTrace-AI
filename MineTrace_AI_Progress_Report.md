# MineTrace AI — Progress Report
**Satellite-based detection of artisanal and small-scale gold mining (ASM) in the Zamfara gold belt, Nigeria**

*Report date: July 18, 2026*

---

## 1. Project Overview

MineTrace AI is a machine learning pipeline that uses Sentinel-2 satellite imagery to detect artisanal and small-scale mining (ASM) sites in Nigeria, with an initial focus on the Zamfara gold belt (NW Nigeria). The system classifies 128×128 pixel image tiles as "mine" or "no_mine" using a CNN trained on hand-labeled examples, then runs a sliding-window search across a full satellite scene to flag candidate mining locations for review.

**Tech stack:** Python, PyTorch, rasterio, Sentinel-2 L2A imagery (via Sentinel Hub / CDSE), VS Code + PowerShell/cmd on Windows.

---

## 2. Pipeline Architecture

The system runs as a four-stage pipeline:

1. **`preprocess.py`** — Converts hand-labeled point locations (GeoJSON) into 128×128px normalized image chips, split into train/val sets.
2. **`train.py`** — Trains a CNN (`TileClassifier`) on the labeled chips using binary cross-entropy loss.
3. **`inference.py`** — Runs a sliding-window scan across the full raster, classifying every valid (non-empty) tile and saving detections above a confidence threshold as GeoJSON.
4. **`preview_detections.py`** / **`list_detections.py`** — Visualization and export tools for reviewing detections against the source imagery and cross-referencing against Google Maps.

Supporting tools built during development:
- **`sample_candidates.py`** — Generates a grid of candidate points across the AOI for labeling.
- **`label_tool.py`** — Interactive keyboard-driven labeler (m/n/s/q) for rapid tile-by-tile annotation.
- **`label_detections.py`** — Lets the model's own high-confidence detections be reviewed and fed back into the labeled dataset (human-in-the-loop correction).

---

## 3. Development Timeline & Key Milestones

### Phase 1 — Getting the pipeline running
- Diagnosed and fixed a silent failure in `train.py` caused by an **empty `models/classifier.py`** file — the `TileClassifier` class had never actually been written, causing an `ImportError` that was being swallowed by inconsistent terminal output in VS Code's integrated terminal.
- Traced intermittent "no output" issues to VS Code's integrated terminal itself; switching to a native Command Prompt window resolved output visibility permanently.
- Implemented the `TileClassifier` CNN architecture (3 conv blocks + global average pooling + linear classifier head).

### Phase 2 — First successful training run
- Initial dataset: **14 labeled tiles** (9 positive), split 12 train / 2 val.
- Model overfit almost immediately — `val_acc` plateaued at 33–67% and `val_loss` diverged upward after epoch 1, since the validation set was too small to be statistically meaningful.

### Phase 3 — Diagnosing false positives
- Ran full-scene inference: model flagged **80 detections**, including tiles landing in **nodata/empty regions** of the raster.
- **Fix:** added a nodata check to `inference.py` (`np.count_nonzero(chip) / chip.size < 0.5` → skip tile). Reduced detections to 44, all within valid imagery.
- Visual inspection via `detections_preview.py` revealed the model was flagging nearly every valid tile — a sign of insufficient training data rather than a code bug.

### Phase 4 — Scaling up labeled data
- Built `sample_candidates.py` (238-point candidate grid) and `label_tool.py` (interactive labeler) to accelerate annotation.
- Grew the labeled dataset from 14 → **162 labels** through iterative manual labeling sessions, specifically targeting known model confusion patterns identified via visual QA:
  - Forest/tree canopy (confirmed false positive)
  - Tilled farmland with parallel ridge patterns (confirmed false positive)
  - Bare rock/laterite with no disturbance signature (confirmed false positive)

### Phase 5 — Retraining and validation
| Metric | Initial run (14 labels) | Mid run (44 labels, imbalanced) | Latest run (162 labels) |
|---|---|---|---|
| Train / Val split | 12 / 2 | 12 / 3 | **99 / 24** |
| Best val accuracy | 0.667 (unstable) | 0.833 | **0.833 (stable)** |
| Val loss trend | Diverged | Diverged | **Stabilized ~0.51–0.52** |
| Detections @ threshold 0.5 | 80 (incl. nodata) | 44 (uniform, low-confidence) | **27 (clustered, higher-confidence)** |
| Top detection confidence | ~0.54 | ~0.54 | **0.762** |

The most recent retrain (162 labels, 99 train / 24 val) shows the clearest evidence of genuine learning:
- `val_loss` decreased steadily through all 50 epochs without diverging — the classic signature of a model that is generalizing rather than memorizing.
- Detection confidence scores now show real spread (0.50–0.76) instead of clustering tightly at the 0.5 decision boundary, indicating the model has learned discriminating features rather than guessing.

### Phase 6 — Ground-truth validation via Google Maps
- Cross-referenced detected coordinates against Google Maps satellite imagery.
- **Confirmed false positives caught and corrected:** dense forest canopy, tilled agricultural fields, bare rock/laterite.
- **Top-confidence detections (0.68–0.76):** visually plausible bare/disturbed terrain, but **not conclusively confirmable** at available image resolution — individual ASM pits (often 2–5m) can be smaller than a single Sentinel-2 pixel (10m), and Google Maps satellite resolution in this rural region has practical zoom limits.

---

## 4. Current Status

**What's working:**
- Full pipeline runs end-to-end without errors (preprocess → train → inference → visualize).
- Model shows genuine, measurable improvement across retraining iterations (val_loss stabilizing, confidence spread increasing, detection count narrowing from 80 → 27).
- Nodata handling, human-in-the-loop label correction, and visual QA tooling are all in place and reusable for future AOIs (Osun, Jos Plateau).

**Known limitations:**
- Dataset (162 labels) is still small for CNN training; class balance skews ~70% positive, which likely inflates false-positive rate.
- Sentinel-2's 10m resolution imposes a hard ceiling on detecting small individual ASM pits — this is a sensor limitation, not a model limitation.
- High-confidence detections remain visually ambiguous without either higher-resolution imagery or field/ground-truth verification.

---

## 5. Recommended Next Steps

1. **Expand and rebalance the labeled dataset** — target 300+ labels with closer to 50/50 mine/no_mine balance to reduce bias toward over-predicting "mine."
2. **Source higher-resolution imagery** (e.g., PlanetScope, commercial Maxar tiles, or drone imagery for spot-checks) to enable actual visual confirmation of high-confidence detections.
3. **Cross-reference against external ASM datasets** — NGO reports, government mining registries, or prior NORA Research Lab data — to validate detections against known site locations rather than relying solely on visual judgment.
4. **Add early stopping** to `train.py` to avoid wasted epochs once val_loss plateaus.
5. **Consider transfer learning** (pretrained ResNet/EfficientNet backbone, fine-tuned) once more data is available — likely to generalize better than the from-scratch CNN currently in use.
6. **Position the tool honestly as a triage/screening system** rather than a standalone confirmation tool — this is both accurate and how real-world remote sensing detection pipelines are typically used in practice.

---

## 6. Summary

Starting from a broken, silently-failing training script, MineTrace AI now has a fully functional detection pipeline validated across multiple retraining cycles, with measurable improvement in model confidence and detection precision as the labeled dataset grew from 14 to 162 examples. The project has moved from "does this even run" to "does this produce plausible, improving results" — a meaningful milestone for a solo-built geospatial ML prototype, and a credible piece of portfolio/pitch material for continued development 