"""
Central configuration for the Nigerian ASM Detector.
Edit AOI_BOUNDS to your specific area of interest before running the pipeline.
"""

EE_PROJECT_ID = "minetrace-ai"

# --- Area of interest ---
# Known Nigerian ASM hotspots. Start with ONE region to keep the first pass small.
AOI_REGIONS = {
    "zamfara_gold": {
        # Rough bounding box [min_lon, min_lat, max_lon, max_lat]
        "bounds": [5.7, 11.6, 6.9, 12.6],
        "description": "Zamfara gold mining belt, NW Nigeria",
    },
    "osun_gold": {
        "bounds": [4.4, 7.4, 4.9, 7.9],
        "description": "Osun gold belt, SW Nigeria",
    },
    "jos_plateau_tin": {
        "bounds": [8.7, 9.6, 9.2, 10.0],
        "description": "Jos Plateau tin/columbite mining, north-central Nigeria",
    },
}

ACTIVE_REGION = "zamfara_gold"  # switch this to change AOI

# --- Imagery ---
SATELLITE = "COPERNICUS/S2_SR_HARMONIZED"  # Sentinel-2 Surface Reflectance, GEE collection ID
DATE_RANGE = ("2024-06-01", "2025-02-28")  # dry season = less cloud cover
MAX_CLOUD_PCT = 20


# Sentinel-2 bands to use. B4/B3/B2 = RGB, B8 = NIR (vegetation), B11 = SWIR (bare soil/mineral)
BANDS = ["B04", "B03", "B02", "B08", "B11"]

# --- Tiling ---
TILE_SIZE_PX = 128
TILE_OVERLAP_PX = 16

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# --- Paths ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
LABELS_DIR = "data/labels"
CHECKPOINT_DIR = "checkpoints"