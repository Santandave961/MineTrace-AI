"""
Quick sanity check on a downloaded Sentinel-2 composite.

Checks:
- File opens correctly and reports dimensions/band count
- Per-band stats (min/max/mean) to catch all-zero or all-nodata bands
- Saves a quick RGB preview PNG so you can visually confirm it's not blank/corrupted

Usage:
    python inspect_download.py
    python inspect_download.py --region osun_gold
"""
import os
import sys
import argparse

import numpy as np
import rasterio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def inspect_tif(path):
    print(f"Inspecting: {path}\n")

    with rasterio.open(path) as src:
        print(f"Dimensions: {src.width} x {src.height} px")
        print(f"Band count: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print()

        band_names = config.BANDS
        all_zero_bands = []

        for i in range(1, src.count + 1):
            band = src.read(i).astype(np.float32)
            name = band_names[i - 1] if i - 1 < len(band_names) else f"band_{i}"

            nan_pct = np.isnan(band).mean() * 100
            valid = band[~np.isnan(band)]

            if valid.size == 0 or np.all(valid == 0):
                all_zero_bands.append(name)
                print(f"  {name}: ALL ZERO/EMPTY -- check this band")
                continue

            print(
                f"  {name}: min={valid.min():.4f} max={valid.max():.4f} "
                f"mean={valid.mean():.4f} nan%={nan_pct:.1f}"
            )

        print()
        if all_zero_bands:
            print(f"WARNING: These bands look empty: {all_zero_bands}")
            print("This usually means a band name mismatch or a cloud-masked AOI.")
        else:
            print("All bands have valid data. Looks good.")

        return src.count, band_names


def save_preview(path, out_png="preview.png"):
    """Save a quick RGB preview using B04/B03/B02 (red/green/blue) if available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed -- skipping visual preview.")
        print("Run: pip install matplotlib")
        return

    band_names = config.BANDS
    rgb_bands = ["B04", "B03", "B02"]
    if not all(b in band_names for b in rgb_bands):
        print(f"\nCan't build RGB preview -- need {rgb_bands}, have {band_names}")
        return

    with rasterio.open(path) as src:
        indices = [band_names.index(b) + 1 for b in rgb_bands]
        rgb = np.stack([src.read(idx).astype(np.float32) for idx in indices], axis=-1)

    # Simple percentile stretch for visibility (satellite reflectance values are
    # small floats, not directly displayable without contrast stretching)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / max(hi - lo, 1e-6), 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(f"RGB preview: {os.path.basename(path)}")
    plt.axis("off")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved visual preview: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=config.ACTIVE_REGION)
    args = parser.parse_args()

    tif_path = os.path.join(config.RAW_DATA_DIR, f"{args.region}.tif")
    if not os.path.exists(tif_path):
        raise SystemExit(f"File not found: {tif_path} -- run download_imagery.py first.")

    inspect_tif(tif_path)
    save_preview(tif_path, out_png=f"preview_{args.region}.png")


if __name__ == "__main__":
    main()