"""
Label the model's own detections (from zamfara_gold_detections.geojson)
as mine/no_mine based on visual inspection (e.g. via Google Maps).

Controls:
    m = mine (positive)
    n = no_mine (negative)
    s = skip
    q = quit and save

Usage:
    python data/label_detections.py
"""
import json
import os
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TIF_PATH = "data/raw/zamfara_gold.tif"
DETECTIONS_PATH = "data/processed/zamfara_gold_detections.geojson"
LABELS_PATH = os.path.join(config.LABELS_DIR, f"{config.ACTIVE_REGION}_labels.geojson")

CHIP_HALF = config.TILE_SIZE_PX // 2


def load_existing_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            data = json.load(f)
        return data["features"]
    return []


def save_labels(features):
    geojson = {"type": "FeatureCollection", "features": features}
    with open(LABELS_PATH, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved {len(features)} total labels to {LABELS_PATH}")


def already_labeled(features, lon, lat, tol=1e-6):
    for feat in features:
        clon, clat = feat["geometry"]["coordinates"]
        if abs(clon - lon) < tol and abs(clat - lat) < tol:
            return True
    return False


def main():
    with open(DETECTIONS_PATH) as f:
        detections = json.load(f)["features"]

    labels = load_existing_labels()
    print(f"Starting with {len(labels)} existing labels.")
    print(f"{len(detections)} detections to review.\n")

    result = {"action": None}

    def on_key(event):
        result["action"] = event.key
        plt.close()

    with rasterio.open(TIF_PATH) as src:
        img_full = src.read([1, 2, 3]).astype(np.float32)
        img_full = np.moveaxis(img_full, 0, -1)
        img_full = np.clip(img_full / np.percentile(img_full, 98), 0, 1)

        for i, feat in enumerate(detections):
            lon, lat = feat["geometry"]["coordinates"]
            conf = feat["properties"].get("confidence", "?")

            if already_labeled(labels, lon, lat):
                continue

            row, col = src.index(lon, lat)
            r0, r1 = max(0, row - CHIP_HALF), row + CHIP_HALF
            c0, c1 = max(0, col - CHIP_HALF), col + CHIP_HALF
            chip = img_full[r0:r1, c0:c1]

            if chip.size == 0:
                continue

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(chip)
            ax.set_title(f"Detection {i+1}/{len(detections)}  conf={conf}\n"
                         f"m=mine  n=no_mine  s=skip  q=quit\n"
                         f"lat={lat:.6f}, lon={lon:.6f}")
            ax.axis("off")
            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()

            action = result["action"]
            if action == "q":
                break
            elif action == "m":
                labels.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"mine": 1},
                })
                print(f"  -> {i+1}: labeled MINE")
            elif action == "n":
                labels.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"mine": 0},
                })
                print(f"  -> {i+1}: labeled NO_MINE")
            else:
                print(f"  -> {i+1}: skipped")

            save_labels(labels)

    print("Done.")


if __name__ == "__main__":
    main()