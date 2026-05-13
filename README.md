# Sheffield Crime Trends Visualiser

End-to-end data science project that ingests Sheffield street-level crime data from the
UK Police API, runs spatial clustering and time-series forecasting, trains supervised
classifiers, and serves everything through an interactive Plotly Dash application.

Built by **Mohamed** — [github.com/](https://github.com/acb22med).

---

## Project layout

```
sheffield-crime-visualiser/
├── README.md
├── requirements.txt
├── config.py                     # All tunable parameters live here
├── src/
│   ├── data/                     # Ingestion, pre-processing, feature engineering
│   │   ├── ingest.py
│   │   ├── preprocess.py
│   │   └── features.py
│   ├── eda/explore.py            # EDA figure generation
│   ├── models/
│   │   ├── clustering.py         # K-Means + DBSCAN
│   │   ├── forecasting.py        # ARIMA + Prophet + walk-forward CV
│   │   └── classification.py     # LogReg / RF / SVM
│   └── app/
│       ├── app.py                # Dash entry point
│       ├── layout.py
│       └── callbacks.py
├── scripts/
│   ├── run_pipeline.py           # Ingest → clean → feature-engineer → SQLite
│   ├── train_models.py           # Fit all models, save artefacts, log metrics
│   └── run_app.py                # Launch the Dash app
├── tests/                        # pytest suite
├── data/                         # SQLite DB + cached API responses (gitignored)
├── figures/                      # EDA and model output figures
└── report/                       # LaTeX report (main.tex + sections/)
```

## Quick start

```bash
# 1. Clone and set up environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull Sheffield crime data and build the SQLite database
python scripts/run_pipeline.py --months 36

# 3. Train all models (clustering, forecasting, classifiers)
python scripts/train_models.py

# 4. Launch the Dash app
python scripts/run_app.py
# → open http://127.0.0.1:8050
```

A full first-time run (36 months of data + all models) takes ~8–15 minutes on a modern
laptop; the bottleneck is the UK Police API rate limit (15 req/sec, enforced client-side).

## Data source

All data comes from the **UK Police open data API** (`https://data.police.uk/api`) —
free, no authentication required, covers the last ~3 years of street-level crime reports.
Sheffield is defined as a polygon covering the South Yorkshire Police jurisdiction inside
the Sheffield city boundary (configurable in `config.py`).

## Testing

```bash
pytest tests/ -v
```

The suite covers data cleaning, feature engineering, and model wrappers with mocked API
responses, so it runs offline in <5 seconds.

## Report

The LaTeX report lives in `report/`. Compile with:

```bash
cd report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

or drop the `report/` folder into Overleaf.

## License

Private — produced for Mohamed B. under the Upwork engagement terms agreed on the
milestone schedule.
