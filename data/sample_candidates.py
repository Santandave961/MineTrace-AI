"""
Generate a grid of candidate points over the AOI and save a preview
image with grid overlay so you can quickly pick mine/no_mine locations.

Usage:
    python data/sample_candidates.py
"""
import json
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TIF_PATH = "data/raw/zamfara_gold.tif"
GRID_SPACING_PX = 80  # distance between candidate points, in pixels
OUTPUT_PREVIEW = "data/candidate_grid_preview.png"
OUTPUT_POINTS = "data/candidate_points.json"


def main():
    with rasterio.open(TIF_PATH) as src:
        img = src.read([1, 2, 3])  # RGB bands for visualization
        img = reshape_as_image(img)
        img = np.clip(img / img.max(), 0, 1)  # normalize for display

        h, w = img.shape[:2]
        points = []
        for row in range(0, h, GRID_SPACING_PX):
            for col in range(0, w, GRID_SPACING_PX):
                lon, lat = src.xy(row, col)
                points.append({"row": row, "col": col, "lon": lon, "lat": lat})

    # Save preview with grid dots overlaid
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(img)
    for i, p in enumerate(points):
        ax.plot(p["col"], p["row"], "r+", markersize=6)
        ax.text(p["col"] + 3, p["row"] - 3, str(i), color="yellow", fontsize=6)
    ax.set_title(f"{len(points)} candidate points — note IDs of mine/no_mine sites")
    plt.savefig(OUTPUT_PREVIEW, dpi=150, bbox_inches="tight")
    print(f"Saved preview: {OUTPUT_PREVIEW}")

    with open(OUTPUT_POINTS, "w") as f:
        json.dump(points, f, indent=2)
    print(f"Saved {len(points)} candidate points to {OUTPUT_POINTS}")


if __name__ == "__main__":
    main()