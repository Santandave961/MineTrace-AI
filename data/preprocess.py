"""
Tile downloaded Sentinel-2 imagery into fixed-size chips, normalize bands, and
build a train/val split from your label file.

Expects:
    data/raw/<region>.tif              — output of download_imagery.py
    data/labels/<region>_labels.geojson — points or polygons you hand-labeled,
                                           with a property "mine": 1 or 0

Usage:
    python data/preprocess.py
"""
import os
import sys
import json
import random

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    raise SystemExit("Missing deps. Run: pip install rasterio")


def load_raster(region_key):
    path = os.path.join(config.RAW_DATA_DIR, f"{region_key}.tif")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run data/download_imagery.py first."
        )
    return rasterio.open(path)


def load_labels(region_key):
    """
    Labels are a GeoJSON FeatureCollection of points, each with property "mine": 0/1.
    If you labeled polygons instead, use their centroid as the point.
    """
    path = os.path.join(config.LABELS_DIR, f"{region_key}_labels.geojson")
    if not os.path.exists(path):
        print(
            f"WARNING: {path} not found. Skipping labeled tiling — "
            "generating unlabeled tiles for manual review instead."
        )
        return None
    with open(path) as f:
        return json.load(f)["features"]


def normalize_band(band, low_pct=2, high_pct=98):
    """Percentile stretch — robust to outlier bright/dark pixels (clouds, shadows)."""
    lo, hi = np.percentile(band, [low_pct, high_pct])
    band = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    return band.astype(np.float32)


def tile_labeled(src, features, region_key):
    out_dir = os.path.join(config.PROCESSED_DATA_DIR, region_key)
    os.makedirs(os.path.join(out_dir, "mine"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "no_mine"), exist_ok=True)

    half = config.TILE_SIZE_PX // 2
    saved = 0
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"]
        label = int(feat["properties"]["mine"])
        row, col = src.index(lon, lat)

        window = Window(col - half, row - half, config.TILE_SIZE_PX, config.TILE_SIZE_PX)
        try:
            chip = src.read(window=window)
        except Exception:
            continue
        if chip.shape[1:] != (config.TILE_SIZE_PX, config.TILE_SIZE_PX):
            continue  # skip tiles that fall off the edge of the raster

        chip = np.stack([normalize_band(chip[b]) for b in range(chip.shape[0])])
        subfolder = "mine" if label == 1 else "no_mine"
        out_path = os.path.join(out_dir, subfolder, f"{region_key}_{saved:05d}.npy")
        np.save(out_path, chip)
        saved += 1

    print(f"Saved {saved} labeled tiles to {out_dir}")
    return out_dir


def build_split(tiles_dir, region_key):
    mine_files = [os.path.join("mine", f) for f in os.listdir(os.path.join(tiles_dir, "mine"))]
    no_mine_files = [
        os.path.join("no_mine", f) for f in os.listdir(os.path.join(tiles_dir, "no_mine"))
    ]
    all_files = [(f, 1) for f in mine_files] + [(f, 0) for f in no_mine_files]

    random.seed(config.RANDOM_SEED)
    random.shuffle(all_files)

    n_val = int(len(all_files) * config.VAL_SPLIT)
    val_files, train_files = all_files[:n_val], all_files[n_val:]

    for name, subset in [("train", train_files), ("val", val_files)]:
        with open(os.path.join(tiles_dir, f"{name}.txt"), "w") as f:
            for path, label in subset:
                f.write(f"{path},{label}\n")

    print(
        f"Split: {len(train_files)} train / {len(val_files)} val "
        f"({sum(l for _, l in all_files)} positive tiles total)"
    )


def main():
    region_key = config.ACTIVE_REGION
    src = load_raster(region_key)
    features = load_labels(region_key)

    if features is None:
        print(
            "No labels yet. Label some tiles by hand (see README 'Getting labels' "
            "section), save as data/labels/<region>_labels.geojson, then re-run this script."
        )
        return

    tiles_dir = tile_labeled(src, features, region_key)
    build_split(tiles_dir, region_key)


if __name__ == "__main__":
    main()