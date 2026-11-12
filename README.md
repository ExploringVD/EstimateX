# EstimateX — Predicting Product Development Time and Cost Using Machine Learning

EstimateX predicts software project effort, duration and cost, and construction project cost, from historical project data. Every prediction comes with the factors that drove it. It runs as a Flask web app.

- **Software:** a sign-constrained log-linear model trained on COCOMO-NASA (93 projects) using all 15 Boehm-1981 cost drivers, KLOC and development mode. It predicts effort in person-months. The results also show:
  - duration and average team size, from the published COCOMO schedule formula `2.5 × effort^c`;
  - a likely range, the band half of past projects fell in, taken from the model's cross-validation errors;
  - cost as effort × a selectable regional $/person-month rate.
- **Construction:** a Random Forest trained on the UCI Residential Building dataset (372 projects). It predicts construction cost in the dataset's native units (×10,000 IRR).

## Results

| Model | Metric | Value |
|---|---|---|
| Software (10x repeated 10-fold CV) | R²(log) | 0.824 |
| | PRED(50) | 75.1% |
| | PRED(25) | 48.9% |
| | Project category accuracy (≥150 PM) | 91.7% (majority baseline 61.3%) |
| Construction (held-out test set, 75 projects) | R² | 0.932 |
| | PRED(25) | 82.7% |

PRED(k) is the share of projects predicted within k% of the actual value.

**Why the software model is constrained.** A plain regression on 93 correlated projects gave Product Complexity a negative weight, so higher complexity predicted less effort. Here each driver is encoded as log(published effort multiplier), so it already carries the right direction. The Ridge weights are forced non-negative, and complexity is a fixed offset, so its effect is always monotonic.

## Run the app

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r app/requirements.txt
python app/app.py
```

Then open http://127.0.0.1:8000/.

## Retrain and check accuracy

```bash
pip install -r requirements.txt
python src/train_construction.py    # compares 4 algorithms, deploys the best tree model
python src/train_software_cocomo.py # fits the software model and stores its likely-range percentiles
python check_accuracy.py            # read-only: prints the metrics above
```

Run these from the project root. Both training scripts overwrite the deployed files in `models/`, and both are deterministic (fixed seeds).

## Folder structure

```
EstimateX/
├── app/
│   ├── app.py               # Flask app: predictions, explanations, validation
│   ├── requirements.txt     # app-only dependencies
│   ├── templates/           # base.html, form.html, results.html
│   └── static/              # app.js, style.css, favicon.png
├── src/
│   ├── preprocessing.py           # shared encode/scale/save/load helpers
│   ├── preprocess_construction.py # construction features and preprocessing
│   ├── train_construction.py      # trains and deploys the construction model
│   └── train_software_cocomo.py   # trains and deploys the software model
├── models/                  # the 5 deployed artifacts the app loads
├── data/                    # raw and processed datasets
├── notebooks/01_eda.ipynb   # exploratory data analysis
├── check_accuracy.py        # read-only accuracy check for both models
├── requirements.txt         # full dependencies (training, notebook, app)
└── EstimateX.pptx           # slide deck
```

## Datasets

| Dataset | Used for | Records |
|---|---|---|
| [COCOMO-NASA](http://promise.site.uottawa.ca/SERepository) | Software model, EDA | 93 |
| [Desharnais](http://promise.site.uottawa.ca/SERepository) | EDA | 81 |
| [UCI Residential Building](https://archive.ics.uci.edu/dataset/437/residential+building+data+set) | Construction model | 372 |

## Team

| Name | Roll No. |
|---|---|
| Vaishnavi Dutt | 2400320101217 |
| Varnika Singh | 2400320101234 |
| Vandana Singh | 2400320101219 |

**Guide:** Ms. Divya Maheshwari
