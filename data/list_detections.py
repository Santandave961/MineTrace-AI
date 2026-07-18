"""
Print detection coordinates and confidence scores in a copy-paste-friendly
format for cross-checking against Google Maps / Sentinel Hub EO Browser.

Usage:
    python data/list_detections.py
"""
import json

DETECTIONS_PATH = "data/processed/zamfara_gold_detections.geojson"


def main():
    with open(DETECTIONS_PATH) as f:
        data = json.load(f)

    features = data["features"]
    # Sort by confidence, highest first
    features.sort(key=lambda f: f["properties"].get("confidence", 0), reverse=True)

    print(f"{len(features)} detections, sorted by confidence:\n")
    for i, feat in enumerate(features):
        lon, lat = feat["geometry"]["coordinates"]
        conf = feat["properties"].get("confidence", "?")
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
        print(f"{i+1}. conf={conf}  lat={lat:.6f}, lon={lon:.6f}")
        print(f"   {maps_url}\n")


if __name__ == "__main__":
    main()