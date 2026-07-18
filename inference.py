"""
Run a trained classifier over a full Sentinel-2 raster using a sliding window,
and output detected mine-tile locations as GeoJSON for mapping.

Usage:
    python inference.py --model classifier --threshold 0.5
"""
import os
import json
import argparse

import numpy as np
import torch
import rasterio
from rasterio.windows import Window

import config
from models.classifier import TileClassifier
from data.preprocess import normalize_band


def sliding_window_detect(model, src, device, threshold, stride=None):
    stride = stride or config.TILE_SIZE_PX
    size = config.TILE_SIZE_PX
    height, width = src.height, src.width

    detections = []
    model.eval()
    with torch.no_grad():
        for row in range(0, height - size, stride):
            for col in range(0, width - size, stride):
                window = Window(col, row, size, size)
                chip = src.read(window=window)
                if chip.shape[1:] != (size, size):
                    continue

                # Skip empty/nodata tiles: if the vast majority of pixels
                # are zero across all bands, there's no real imagery here
                nonzero_frac = np.count_nonzero(chip) / chip.size
                if nonzero_frac < 0.5:
                    continue

                chip_norm = np.stack([normalize_band(chip[b]) for b in range(chip.shape[0])])
                x = torch.from_numpy(chip_norm).unsqueeze(0).to(device)

                prob = torch.sigmoid(model(x)).item()
                if prob >= threshold:
                    center_row, center_col = row + size // 2, col + size // 2
                    lon, lat = src.xy(center_row, center_col)
                    detections.append({"lon": lon, "lat": lat, "confidence": round(prob, 3)})

    return detections


def save_geojson(detections, out_path):
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
            "properties": {"confidence": d["confidence"]},
        }
        for d in detections
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved {len(features)} detections to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classifier"], default="classifier")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TileClassifier(in_channels=len(config.BANDS)).to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_best.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"No checkpoint at {ckpt_path} — run train.py first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    raster_path = os.path.join(config.RAW_DATA_DIR, f"{config.ACTIVE_REGION}.tif")
    with rasterio.open(raster_path) as src:
        detections = sliding_window_detect(model, src, device, args.threshold)

    out_path = os.path.join(config.PROCESSED_DATA_DIR, f"{config.ACTIVE_REGION}_detections.geojson")
    save_geojson(detections, out_path)


if __name__ == "__main__":
    main()