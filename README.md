# MineTrace AI

Satellite-based detection of artisanal and small-scale gold mining (ASM) sites in Nigeria, built on Sentinel-2 imagery and a CNN tile classifier. Initial focus: the **Zamfara gold belt**, NW Nigeria.

> Built by [Wisdom Okparaji](https://github.com/Santandave961) — Data Scientist / ML Engineer bridging accounting/finance background with applied ML. Follow the build on [X (@WOkparaji74619)](https://x.com/Santandave961).

---

## What it does

MineTrace AI takes Sentinel-2 satellite imagery of an area of interest, splits it into small image tiles, and classifies each tile as **mine** or **no_mine** using a trained CNN. It then runs this classifier across an entire scene to flag candidate ASM locations, output as GeoJSON, and visualize them on an interactive map.

It's designed as a **triage/screening tool** — narrowing a large search area down to a shortlist of candidate sites for further review — not a fully automated confirmation system. See [`MineTrace_AI_Progress_Report.md`](./MineTrace_AI_Progress_Report.md) for full methodology, current limitations, and validation notes.

---

## Pipeline

```
1. sample_candidates.py      → generates a grid of candidate points across the AOI
2. label_tool.py             → interactive keyboard labeler (m/n/s/q) for tagging tiles
3. preprocess.py              → converts labeled points into 128x128px training chips
4. train.py                   → trains the TileClassifier CNN
5. inference.py                → sliding-window scan across the full scene, outputs detections.geojson
6. preview_detections.py       → visualizes detections over the source imagery
7. list_detections.py          → exports sorted detection list w/ Google Maps links for QA
8. label_detections.py         → re-labels the model's own detections (human-in-the-loop correction)
9. app.py                      → Streamlit dashboard (map + metrics + progress report)
```

---

## Setup

```bash
git clone https://github.com/Santandave961/MineTrace-AI.git
cd MineTrace-AI
pip install -r requirements.txt
```

Edit `config.py` to set your area of interest, date range, and training hyperparameters before running the pipeline.

---

## Usage

**1. Download imagery** for your AOI (see `download_imagery.py` — requires Sentinel Hub / Copernicus Data Space Ecosystem credentials).

**2. Generate candidate points and label them:**
```bash
python data/sample_candidates.py
python data/label_tool.py
```

**3. Preprocess labeled points into training tiles:**
```bash
python data/preprocess.py
```

**4. Train the classifier:**
```bash
python train.py --model classifier
```

**5. Run inference across the full scene:**
```bash
python inference.py --model classifier --threshold 0.5
```

**6. Review detections:**
```bash
python data/preview_detections.py
python data/list_detections.py
```

**7. Launch the dashboard:**
```bash
streamlit run app.py
```

---

## Project structure

```
MineTrace AI/
├── app.py                          # Streamlit dashboard
├── train.py                        # training loop
├── inference.py                    # sliding-window detection
├── config.py                       # AOI, bands, hyperparameters
├── models/
│   └── classifier.py               # TileClassifier CNN architecture
├── data/
│   ├── raw/                        # downloaded Sentinel-2 rasters
│   ├── processed/                  # training chips + detections.geojson
│   ├── labels/                     # labeled point GeoJSON
│   ├── sample_candidates.py
│   ├── label_tool.py
│   ├── label_detections.py
│   ├── preprocess.py
│   ├── preview_detections.py
│   └── list_detections.py
├── checkpoints/                    # saved model weights
├── requirements.txt
└── MineTrace_AI_Progress_Report.md # full methodology + validation notes
```

---

## Current status

- Trained on **162 hand-labeled examples** (99 train / 24 val tiles) across the Zamfara AOI.
- Best validation accuracy: **0.833**, with a stable (non-diverging) validation loss curve.
- Detection confidence scores show real spread (0.50–0.76) rather than clustering at the decision boundary — a sign the model is learning genuine discriminating features, not guessing.
- Known limitation: Sentinel-2's 10m resolution imposes a hard ceiling on detecting small individual ASM pits (often 2–5m). High-confidence detections are visually plausible but not always conclusively confirmable without higher-resolution imagery or field verification.

Full write-up: [`MineTrace_AI_Progress_Report.md`](./MineTrace_AI_Progress_Report.md)

---

## Roadmap

- [ ] Expand and rebalance labeled dataset (target 300+, closer to 50/50 class balance)
- [ ] Source higher-resolution imagery for detection confirmation
- [ ] Cross-reference detections against external ASM datasets / mining registries
- [ ] Add early stopping to training loop
- [ ] Explore transfer learning (pretrained backbone) once more data is available
- [ ] Expand to additional AOIs: Osun gold belt, Jos Plateau tin/columbite

---

## Disclaimer

This is a research/prototype tool. Detections are model outputs based on limited training data and satellite imagery resolution constraints — they should be treated as candidates for further investigation, not confirmed findings. Not intended for enforcement or legal use without independent verification.

---

## License

MIT (or update as appropriate)
