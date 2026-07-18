"""
Download Sentinel-2 imagery for the configured AOI via Copernicus Data Space Ecosystem
(CDSE), using their Sentinel Hub-compatible Process API. No Google Earth Engine needed.

Setup (one-time):
    pip install sentinelhub
    1. Create a free account at https://dataspace.copernicus.eu/
    2. Go to your Dashboard -> User settings -> OAuth clients -> create a new client
    3. Set the CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables, e.g.:
       setx CDSE_CLIENT_ID "your-client-id"
       setx CDSE_CLIENT_SECRET "your-client-secret"
       (close and reopen your terminal after running setx)

Usage:
    python data/download_imagery.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from sentinelhub import (
        SHConfig,
        SentinelHubRequest,
        DataCollection,
        MimeType,
        CRS,
        BBox,
        bbox_to_dimensions,
    )
except ImportError:
    raise SystemExit("Missing deps. Run: pip install sentinelhub")


def get_sh_config():
    sh_config = SHConfig()
    sh_config.sh_client_id = os.environ.get("CDSE_CLIENT_ID")
    sh_config.sh_client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    sh_config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    sh_config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    if not sh_config.sh_client_id or not sh_config.sh_client_secret:
        raise SystemExit(
            "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set. Run:\n"
            '  setx CDSE_CLIENT_ID "your-client-id"\n'
            '  setx CDSE_CLIENT_SECRET "your-client-secret"\n'
            "then close and reopen your terminal."
        )
    return sh_config


def get_aoi_bbox(region_key):
    bounds = config.AOI_REGIONS[region_key]["bounds"]
    min_lon, min_lat, max_lon, max_lat = bounds
    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)


# Evalscript: selects the bands defined in config.BANDS and returns them as a
# multi-band float32 GeoTIFF, using the least-cloudy pixel across the date range.
EVALSCRIPT_TEMPLATE = """
//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: {bands}, units: "REFLECTANCE" }}],
    output: {{ bands: {n_bands}, sampleType: "FLOAT32" }},
    mosaicking: "ORBIT"
  }};
}}

function evaluatePixel(samples) {{
  var best = samples[0];
  for (var i = 0; i < samples.length; i++) {{
    if (samples[i].dataMask === 1) {{
      best = samples[i];
      break;
    }}
  }}
  return [{band_refs}];
}}
"""


def build_evalscript():
    bands = config.BANDS
    band_list_str = str(bands)
    band_refs = ", ".join([f"best.{b}" for b in bands])
    return EVALSCRIPT_TEMPLATE.format(
        bands=band_list_str, n_bands=len(bands), band_refs=band_refs
    )


def fetch_composite(sh_config, region_key):
    bbox = get_aoi_bbox(region_key)
    # 60m resolution keeps the request under Sentinel Hub's 2500x2500 px output cap
    # for a full-region AOI. Lower this once you switch to tiling smaller sub-areas.
    size = bbox_to_dimensions(bbox, resolution=100)
    start, end = config.DATE_RANGE

    # Bind Sentinel-2 L2A to the CDSE service endpoint explicitly — DataCollection
    # defaults to the standard Sentinel Hub servers otherwise, ignoring sh_base_url,
    # which causes 401 errors since CDSE credentials aren't valid there.
    cdse_s2l2a = DataCollection.SENTINEL2_L2A.define_from(
        "cdse_s2l2a", service_url=sh_config.sh_base_url
    )

    request = SentinelHubRequest(
        evalscript=build_evalscript(),
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=cdse_s2l2a,
                time_interval=(start, end),
                maxcc=config.MAX_CLOUD_PCT / 100.0,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=sh_config,
    )

    print(f"Requesting Sentinel-2 composite for {region_key} ({start} to {end})...")
    data = request.get_data()
    if not data:
        raise RuntimeError("No data returned — check date range, cloud threshold, or AOI bounds.")
    return data[0]


def save_geotiff(array, region_key):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    out_path = os.path.join(config.RAW_DATA_DIR, f"{region_key}.tif")

    bounds = config.AOI_REGIONS[region_key]["bounds"]
    min_lon, min_lat, max_lon, max_lat = bounds
    height, width = array.shape[0], array.shape[1]
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    n_bands = array.shape[2] if array.ndim == 3 else 1
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=n_bands,
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(n_bands):
            dst.write(array[:, :, i] if array.ndim == 3 else array, i + 1)

    print(f"Saved: {out_path}")
    return out_path


def main():
    sh_config = get_sh_config()
    region_key = config.ACTIVE_REGION
    print(f"Region: {region_key} — {config.AOI_REGIONS[region_key]['description']}")

    array = fetch_composite(sh_config, region_key)
    save_geotiff(array, region_key)


if __name__ == "__main__":
    main()