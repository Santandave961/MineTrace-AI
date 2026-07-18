"""
MineTrace AI — Streamlit Dashboard

Shows detection map, key model metrics, and the project progress report.

Usage:
    streamlit run app.py

Expected files (relative to this script, matching the MineTrace AI project layout):
    data/processed/zamfara_gold_detections.geojson
    MineTrace_AI_Progress_Report.md
"""
import json
import os

import streamlit as st
import pandas as pd
import pydeck as pdk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MineTrace AI",
    page_icon="⛏️",
    layout="wide",
)

DETECTIONS_PATH = "data/processed/zamfara_gold_detections.geojson"
REPORT_PATH = "MineTrace_AI_Progress_Report.md"

# Known headline metrics from the latest validated training run.
# Update these manually after each retrain, or wire this up to a logged
# metrics file once train.py writes one (see note at bottom of file).
LATEST_METRICS = {
    "Best val accuracy": "0.833",
    "Train / Val tiles": "99 / 24",
    "Total labels": "162",
    "Val loss (final)": "0.518",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_detections(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)

    rows = []
    for feat in data.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        conf = feat.get("properties", {}).get("confidence", None)
        rows.append({"lat": lat, "lon": lon, "confidence": conf})

    if not rows:
        return pd.DataFrame(columns=["lat", "lon", "confidence"])
    return pd.DataFrame(rows)


@st.cache_data
def load_report(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def confidence_color(conf):
    """Map a confidence score to an RGB color: red (low) -> green (high)."""
    if conf is None:
        return [200, 30, 30, 160]
    if conf > 0.65:
        return [40, 180, 90, 200]   # green
    elif conf > 0.55:
        return [230, 190, 30, 190]  # yellow
    else:
        return [210, 60, 50, 160]   # red


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⛏️ MineTrace AI")
st.sidebar.caption("ASM detection — Zamfara gold belt, Nigeria")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Detection Map", "Progress Report"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with Sentinel-2 imagery + a CNN tile classifier. "
    "See the Progress Report tab for full methodology and limitations."
)

# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("MineTrace AI — Overview")
    st.write(
        "Satellite-based detection of artisanal and small-scale gold mining "
        "(ASM) sites, built on Sentinel-2 imagery."
    )

    cols = st.columns(len(LATEST_METRICS))
    for col, (label, value) in zip(cols, LATEST_METRICS.items()):
        col.metric(label, value)

    st.markdown("---")

    detections_df = load_detections(DETECTIONS_PATH)
    if detections_df is not None and not detections_df.empty:
        st.subheader("Current detections")
        c1, c2 = st.columns(2)
        c1.metric("Total detections", len(detections_df))
        if detections_df["confidence"].notna().any():
            c2.metric(
                "Top confidence",
                f"{detections_df['confidence'].max():.3f}",
            )
        st.dataframe(
            detections_df.sort_values("confidence", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info(
            f"No detections file found at `{DETECTIONS_PATH}`. "
            "Run `python inference.py --model classifier` to generate one."
        )

# ---------------------------------------------------------------------------
# Detection Map page
# ---------------------------------------------------------------------------
elif page == "Detection Map":
    st.title("Detection Map")

    detections_df = load_detections(DETECTIONS_PATH)

    if detections_df is None:
        st.error(
            f"Could not find `{DETECTIONS_PATH}`. "
            "Run inference first: `python inference.py --model classifier`"
        )
    elif detections_df.empty:
        st.warning("Detections file loaded but contains no points.")
    else:
        st.caption(
            "🟢 High confidence (>0.65)   🟡 Medium (0.55–0.65)   🔴 Low/unknown (<0.55)"
        )

        detections_df["color"] = detections_df["confidence"].apply(confidence_color)

        view_state = pdk.ViewState(
            latitude=detections_df["lat"].mean(),
            longitude=detections_df["lon"].mean(),
            zoom=9,
            pitch=0,
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=detections_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=250,
            pickable=True,
            stroked=True,
            get_line_color=[0, 0, 0],
            line_width_min_pixels=1,
        )

        tooltip = {
            "html": "<b>Confidence:</b> {confidence}<br/><b>Lat/Lon:</b> {lat}, {lon}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="mapbox://styles/mapbox/satellite-streets-v11",
            )
        )

        st.markdown("---")
        st.subheader("All detections")
        st.dataframe(
            detections_df[["lat", "lon", "confidence"]]
            .sort_values("confidence", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )

        csv = detections_df[["lat", "lon", "confidence"]].to_csv(index=False)
        st.download_button(
            "Download detections as CSV",
            data=csv,
            file_name="minetrace_detections.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Progress Report page
# ---------------------------------------------------------------------------
elif page == "Progress Report":
    report = load_report(REPORT_PATH)
    if report:
        st.markdown(report)
    else:
        st.error(
            f"Could not find `{REPORT_PATH}` in the app directory. "
            "Make sure MineTrace_AI_Progress_Report.md is alongside app.py."
        )

# ---------------------------------------------------------------------------
# NOTE for future improvement:
# To show live training curves (loss/accuracy per epoch) instead of static
# headline numbers, have train.py append each epoch's metrics to a CSV
# (e.g. checkpoints/training_log.csv), then load and chart it here with
# st.line_chart(). Not wired up yet since train.py currently only prints
# to stdout.
# ---------------------------------------------------------------------------