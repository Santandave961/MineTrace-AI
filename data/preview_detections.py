"""
Overlay model detections on the Sentinel-2 imagery for visual QA.

Usage:
    python data/preview_detections.py
"""
import json
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt

TIF_PATH = "data/raw/zamfara_gold.tif"
DETECTIONS_PATH = "data/processed/zamfara_gold_detections.geojson"
OUTPUT_PATH = "data/detections_preview.png"


def main():
    with open(DETECTIONS_PATH) as f:
        detections = json.load(f)

    features = detections["features"]
    print(f"Loaded {len(features)} detections")

    with rasterio.open(TIF_PATH) as src:
        img = src.read([1, 2, 3])  # RGB for display
        img = reshape_as_image(img).astype(np.float32)
        img = np.clip(img / np.percentile(img, 98), 0, 1)  # simple contrast stretch

        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(img)

        for i, feat in enumerate(features):
            lon, lat = feat["geometry"]["coordinates"]
            row, col = src.index(lon, lat)

            # color/size by confidence if present, else default red
            props = feat.get("properties", {})
            conf = props.get("confidence") or props.get("probability") or props.get("score")

            if conf is not None:
                color = "lime" if conf > 0.6 else ("yellow" if conf > 0.55 else "red")
            else:
                color = "red"

            ax.plot(col, row, "o", color=color, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
            ax.text(col + 5, row - 5, str(i), color="white", fontsize=6)

        ax.set_title(f"{len(features)} detections — green=high conf, yellow=med, red=low/unknown")
        plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved preview: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()