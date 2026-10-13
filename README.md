# EstimateX — Predicting Product Development Time and Cost Using Machine Learning

EstimateX is a machine learning system trained on historical project data that predicts software and construction project effort/cost across two independent project domains. Rather than acting as a black box, it explains its predictions through feature importance, and ships as a live, interactive Flask web app, not just a notebook.

- **Software** (Software/Web, Mobile App, Enterprise/Other) — one unified Random Forest model trained on COCOMO-NASA, Desharnais, and China (PROMISE repository), predicting effort in person-months.
- **Construction** — a Random Forest model trained on the UCI Residential Building Data Set, predicting construction cost (native "10,000 IRR" units).

## Tech Stack

- **Python** — core language
- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — preprocessing, Linear Regression, Random Forest, SVR
- **XGBoost** — gradient-boosted comparison model
- **Flask** — the deployed web interface (`app/`)
- **matplotlib / seaborn** — EDA and training-time comparison charts
- **joblib** — model/preprocessing artifact persistence

## Folder Structure

```
EstimateX/
├── data/
│   ├── raw/                        # Original datasets, unmodified
│   │   ├── cocomo_nasa.csv
│   │   ├── desharnais.csv
│   │   ├── china.arff
│   │   └── construction_residential.csv
│   └── processed/                  # Cleaned/encoded data and train/test splits for all domains
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
├── src/
│   ├── preprocessing.py            # Cleaning, encoding, scaling — shared building blocks
│   ├── preprocess_construction.py  # Construction domain preprocessing
│   ├── unify_software.py           # Merges COCOMO-NASA + Desharnais + China into one software dataset
│   ├── convert_construction_raw.py # One-time Excel -> CSV conversion for the construction dataset
│   ├── split_data.py               # Train/test split (random_state=42)
│   ├── train_models.py             # Trains + cross-validates all 4 algorithms per dataset
│   ├── evaluate.py                 # Test-set MAE/RMSE/MMRE/PRED(25) + comparison charts
│   ├── feature_importance.py       # Feature importance rankings + charts for the tree models
│   ├── select_final_model.py       # Picks and re-saves the final model + preprocessing artifacts
│   └── train_new_domains.py        # Orchestrates the above for the Software-Unified + Construction domains
├── models/                         # final_model_/final_scaler_/final_encoder_ {software,construction}.pkl
│                                    # — the only artifacts the app actually loads at runtime
├── app/
│   ├── app.py                      # Flask app — real predictions from the final trained models
│   ├── requirements.txt            # Minimal deps to run just the app (no training/EDA tooling)
│   ├── templates/                  # form.html, results.html (Jinja2, extend base.html)
│   └── static/                     # style.css, app.js, kloc-presets.js, results.js, favicon.png
├── EstimateX.pptx                  # Slide deck
├── requirements.txt                # Full dependencies (EDA, training, and the app)
└── README.md
```

## Datasets

| Dataset | Domain | Records | Description |
|---|---|---|---|
| **COCOMO-NASA** | Software | 93 | NASA software projects (1971–1987) with size, effort-driver ratings, and actual development effort in person-months. |
| **Desharnais** | Software | 81 | Canadian software projects with team/manager experience and function-point-based size against actual effort in person-hours. |
| **China** | Software | 499 | PROMISE repository software projects, added for small-project (low-KLOC) coverage. |
| **UCI Residential Building** | Construction | 372 | Iranian residential construction projects with physical/financial variables and actual construction cost. |

COCOMO-NASA and Desharnais are sourced from the [PROMISE Software Engineering Repository](http://promise.site.uottawa.ca/SERepository) (ARFF, losslessly converted to CSV). China is read directly from its original ARFF file. The construction dataset is from the [UCI Residential Building Data Set](https://archive.ics.uci.edu/dataset/437/residential+building+data+set), converted from its source Excel workbook via `src/convert_construction_raw.py`.

## Team

| Name | Roll No. |
|---|---|
| Vaishnavi Dutt | 2400320101217 |
| Varnika Singh | 2400320101234 |
| Vandana Singh | 2400320101219 |

**Guide:** Ms. Divya Maheshwari

---

## Quick Start — just run the web app

This is enough to get a working, live prediction demo running; it doesn't need the EDA/training tooling.

```bash
git clone https://github.com/ExploringVD/EstimateX
cd EstimateX

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r app/requirements.txt

python app/app.py
```

Then open **http://127.0.0.1:8000/** in a browser, fill in the form, and click "Get estimate" — you'll get a real prediction from the trained Random Forest models plus a live feature-importance breakdown, not a placeholder. (macOS users: if you ever re-run model training and hit an XGBoost import error, it needs the OpenMP runtime — `brew install libomp` — but this isn't required just to run the app.)

## Full Setup — reproduce the entire pipeline

To regenerate everything from raw data through to the final deployed models (`models/final_model_software.pkl` and `models/final_model_construction.pkl`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/preprocess_construction.py   # construction raw -> data/processed/construction_*.csv
python src/unify_software.py            # cocomo_nasa + desharnais + china -> data/processed/software_unified*.csv
python src/train_new_domains.py         # trains, evaluates, and selects the final software + construction models
```

Each script is idempotent and safe to re-run — outputs are overwritten deterministically (fixed `random_state=42` throughout, defined once in `src/split_data.py`).

The original per-dataset pipeline (COCOMO-NASA and Desharnais trained and deployed separately, before they were unified into one software model) is still present and functional — `src/preprocessing.py`, `src/train_models.py`, `src/evaluate.py`, `src/feature_importance.py`, and `src/select_final_model.py` — and its generic functions are what `train_new_domains.py` reuses internally for the newer domains. Running it standalone regenerates the legacy `cocomo`/`desharnais` artifacts, which the app no longer loads:

```bash
python src/preprocessing.py
python src/split_data.py
python src/train_models.py
python src/evaluate.py
python src/feature_importance.py
python src/select_final_model.py
```
