# EstimateX — Predicting Product Development Time and Cost Using Machine Learning

## Results

Random Forest is the final, deployed model for both datasets — the strongest combined MAE + PRED(25) performer in every comparison. Full numbers: [`results/model_comparison.csv`](results/model_comparison.csv) (raw table), [`results/final_model_decision.md`](results/final_model_decision.md) (why), [`results/feature_importance_summary.md`](results/feature_importance_summary.md) (what drives each prediction).

| Dataset | Winning model | MAE | RMSE | MMRE | PRED(25) |
|---|---|---|---|---|---|
| COCOMO-NASA | Random Forest | 219.8 | 440.3 | 0.770 | 42.1% |
| Desharnais | Random Forest | 2434.1 | 3343.4 | 0.414 | 25.0% |

EstimateX is a machine learning system trained on historical project data — the COCOMO-NASA and Desharnais datasets — that predicts software/product development time and cost across two independent project domains. Rather than acting as a black box, it explains its predictions through feature importance, and ships as a live, interactive Flask web app, not just a notebook.

## Tech Stack

- **Python** — core language
- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — preprocessing, Linear Regression, Random Forest, SVR
- **XGBoost** — gradient-boosted comparison model
- **Flask** — the deployed web interface (`app/`)
- **matplotlib / seaborn** — EDA and results visualizations
- **joblib** — model/preprocessing artifact persistence

## Folder Structure

```
EstimateX/
├── data/
│   ├── raw/                  # Original datasets, unmodified (cocomo_nasa.csv, desharnais.csv)
│   └── processed/            # Cleaned + encoded data, and the train/test splits used everywhere
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis for both datasets
├── src/
│   ├── preprocessing.py      # Cleaning, encoding, scaling — shared by training AND the app
│   ├── split_data.py         # Single source of truth for the train/test split (random_state=42)
│   ├── train_models.py       # Trains + cross-validates all 4 algorithms on both datasets
│   ├── evaluate.py           # Test-set MAE/RMSE/MMRE/PRED(25) + comparison charts
│   ├── feature_importance.py # Feature importance rankings + charts for the tree models
│   └── select_final_model.py # Picks and re-saves the final deployed model + preprocessing
├── models/                   # Every trained model (.pkl) and fitted scaler/encoder (.joblib),
│                              # plus the final_*.pkl / final_*.joblib artifacts the app actually loads
├── app/
│   ├── app.py                 # Flask app — real predictions from the final trained models
│   ├── requirements.txt       # Minimal deps to run just the app (no training/EDA tooling)
│   ├── templates/              # form.html, results.html (Jinja2, extend base.html)
│   └── static/style.css        # Shared styling
├── results/
│   ├── model_comparison.csv              # Full 4-model x 2-dataset x 4-metric results table
│   ├── final_model_decision.md           # Why Random Forest was chosen
│   ├── feature_importance_summary.md     # Plain-language explainability findings
│   ├── testing_notes.md                  # End-to-end test results + one bug found & fixed
│   ├── demo_script.md                    # Literal live-demo script with real, verified numbers
│   ├── qa_prep.md                        # Answers to likely presentation questions
│   └── *.png                             # MAE comparison + feature importance charts
├── requirements.txt           # Full dependencies (EDA, training, and the app)
└── README.md
```

## Datasets

Both datasets are sourced from the [PROMISE Software Engineering Repository](http://promise.site.uottawa.ca/SERepository) (also mirrored via OpenML/Zenodo/SEACRAFT). The repository distributes them in ARFF format; they were losslessly converted to plain CSV (attribute names and data values unchanged, only the ARFF metadata/header wrapper stripped) and stored unmodified in `data/raw/`.

| Dataset | Source URL | Records | Description |
|---|---|---|---|
| **COCOMO-NASA** (`cocomo_nasa.csv`) | http://promise.site.uottawa.ca/SERepository/datasets/cocomonasa_2.arff | 93 | 93 NASA software projects (1971–1987) with size, effort-driver ratings (e.g. complexity, reliability, team capability), and actual development effort in person-months. |
| **Desharnais** (`desharnais.csv`) | http://promise.site.uottawa.ca/SERepository/datasets/desharnais.arff | 81 | 81 software projects from a Canadian software house, with team/manager experience and function-point-based size measures against actual development effort in person-hours. |

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

Then open **http://127.0.0.1:5000/** in a browser, fill in the form, and click "Get estimate" — you'll get a real prediction from the trained Random Forest models plus a live feature-importance breakdown, not a placeholder. (macOS users: if you ever re-run model training and hit an XGBoost import error, it needs the OpenMP runtime — `brew install libomp` — but this isn't required just to run the app.)

## Full Setup — reproduce the entire pipeline

To regenerate everything from raw data through to the final deployed model:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/preprocessing.py        # data/raw/*.csv -> data/processed/*_clean.csv
python src/split_data.py           # -> data/processed/*_train.csv, *_test.csv
python src/train_models.py         # trains + cross-validates all 4 models, saves to models/
python src/evaluate.py             # -> results/model_comparison.csv + MAE charts
python src/feature_importance.py   # -> results/feature_importance_*.png + printed rankings
python src/select_final_model.py   # -> models/final_*.pkl + results/final_model_decision.md
```

Each script is idempotent and safe to re-run — outputs are overwritten deterministically (fixed `random_state=42` throughout, defined once in `src/split_data.py`).

## Presenting this project

- [`results/demo_script.md`](results/demo_script.md) — exact inputs to use for a live demo, with the real results they produce.
- [`results/qa_prep.md`](results/qa_prep.md) — grounded answers to likely questions (why Random Forest, what the metrics mean, why not deep learning, known limitations, etc.).
- [`results/testing_notes.md`](results/testing_notes.md) — what was tested end-to-end, and the one bug that was found and fixed.
- `EstimateX_Research_Paper_2 (1).docx` / `EstimateX.pptx` — the full paper and slide deck, both updated with real results.
