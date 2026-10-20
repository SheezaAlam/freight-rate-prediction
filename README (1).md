# Freight Rate Prediction

Machine Learning Engineer assessment: predict the posted rate (`posted_rate`) of freight loads.

* **Labelled data:** 48,000 loads, Jan – Oct 2025 (`train_test.csv`)
* **To predict:** 12,000 loads, Nov – Dec 2025 (`validation.csv`)
* **Extra deliverable:** a fixed December 2025 chart (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb) produced by the provided `score.py`

Because the loads to predict lie entirely in the **future** of the training data, all validation is done in time order, not with a random split.

---

## Result in one paragraph

The final model is a gradient-boosted tree model for the load itself (route, equipment, weight) plus a linear component for the market signals and a time trend. Averaged over three time-ordered folds it reaches a mean absolute error of about **$45 (1.93 % MAPE)** on loads with trustworthy labels, the best of six models tested.

| Model (mean of 3 time folds) | Clean MAE ($) | Clean MAPE (%) | Raw MAE ($) |
|---|---|---|---|
| **Chosen: tree model + linear market/trend** | **44.83** | **1.93** | **98.76** |
| HistGradientBoosting, all features | 51.01 | 2.14 | 104.86 |
| Ridge, all features | 51.28 | 2.23 | 105.13 |
| Random forest, all features | 66.91 | 2.82 | 120.62 |
| HistGradientBoosting, no market/time info | 81.46 | 3.34 | 134.98 |
| Baseline: median $/mile by equipment and distance band | 105.47 | 4.61 | 158.70 |

*Clean* = test rows without corrupted labels; *raw* = all test rows (the honest number, which includes labels no model can predict).
On the final hold-out fold alone, the tree-only models are slightly better (about 1.97 % vs 2.14 % MAPE). The chosen model has the best average and is the most stable across folds. See [Limitations](#limitations).

---

## Repository structure

```
.
├── README.md
├── RUN_STEPS.md              # one-command-per-step cheat sheet
├── requirements.txt
├── score.py                  # provided scorer (unchanged)
├── validation_predictions.csv  # final predictions: load_id,predicted_rate
├── src/
│   ├── common.py             # paths, feature lists, metrics
│   ├── model.py              # the final model (FreightRateModel)
│   ├── step01_explore.py     # EDA + data-quality audit
│   ├── step02_clean.py       # cleaning + corrupted-label detection
│   ├── step03_features.py    # feature engineering
│   ├── step04_validate.py    # time-based validation, model comparison, tuning
│   ├── step05_train_final.py # final fit on all labelled data
│   ├── step06_predict.py     # writes the prediction files
│   └── step08_make_report.py # optional: builds the Word report
└── data/                     # NOT in the repo (see Setup)
```

---

## Setup

Requires Python 3.10+.

**1. Create an environment and install dependencies**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Add the data**

The assessment CSVs are not included in this repository. Place them in a `data/` folder using exactly these names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

(If your downloaded files use hyphens, e.g. `train-test.csv`, rename them with underscores.)

---

## Run the pipeline

Run every command from the project root, one after another. Each step prints its own results.

```bash
python src/step01_explore.py        # 1. explore the data, audit data quality
python src/step02_clean.py          # 2. clean inputs, flag corrupted labels
python src/step03_features.py       # 3. build model features
python src/step04_validate.py       # 4. time-based validation + model comparison (several minutes)
python src/step05_train_final.py    # 5. train the final model on all Jan-Oct data
python src/step06_predict.py        # 6. write validation_predictions.csv + fill December file
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python src/step08_make_report.py    # optional: report.docx
```

On Windows you can use `py` instead of `python`.

### Outputs

| File | Created by | Description |
|---|---|---|
| `reports/eda_overview.png` | step 1 | Exploratory plots |
| `data/processed/*.csv` | steps 2–3 | Cleaned data and features |
| `data/processed/cv_results.csv` | step 4 | Every model on every fold |
| `data/processed/best_params.json` | step 4 | Selected hyper-parameters |
| `models/freight_model.joblib` | step 5 | Trained model |
| `validation_predictions.csv` | step 6 | **Final predictions** (`load_id,predicted_rate`, 12,000 rows) |
| `data/december_chart_inputs.csv` | step 6 | December rows with `predicted_rate` filled in |
| `scorer_results/candidate_december.png` | `score.py` | **Fixed December chart** |
| `report.docx` | step 8 | Optional Word report |

All random seeds are fixed. Results may differ marginally between machines or library versions, mainly because near-tied hyper-parameter settings can be selected differently.

---

## Approach

### 1. Data split and validation

`train_test.csv` covers 1 Jan – 31 Oct 2025 and `validation.csv` covers 1 Nov – 31 Dec 2025, so the task is a two-month forecast. A random split would mix future days into training and flatter the score (a random 5-fold split gave 1.61 % MAPE against 1.93 % for the honest time-based evaluation). Instead, an expanding-window time split is used:

| Fold | Train on | Test on |
|---|---|---|
| 1 | Jan – Apr | May – Jun |
| 2 | Jan – Jun | Jul – Aug |
| 3 (hold-out) | Jan – Aug | Sep – Oct (two-month look-ahead, the same gap as Nov – Dec) |

Hyper-parameters were chosen on these folds only; the final model is refit on all of Jan – Oct.

### 2. Data-quality issues and fixes

| Issue | Train / Validation | Treatment |
|---|---|---|
| Negative weights (sign errors) | 292 / 145 rows | Converted to absolute value |
| Missing weight | 300 / 165 rows | Median by equipment type + missing flag |
| Weight at the 47,500 lb cap | 1,191 / 297 rows | Kept (real ceiling), flagged |
| Missing `market_index` | 374 / 249 rows | Median of the same date (market-wide daily signal) |
| Noisy distance on the same route; floor at 70 mi | 48 / 21 rows at the floor | Route-level median distance as a feature |
| Corrupted labels (rate about 3.4x too high or 0.28x too low) | 677 train rows (1.4 %) | Detected with a robust regression (residual above 6 MAD); removed from **training only** |

Validation rows are never dropped, because every one of them needs a prediction.

### 3. Features

* **Static (the load):** log distance, route-median distance, great-circle distance, circuity, pickup/delivery coordinates, equipment type, weight, distance-floor flag.
* **Time / market:** daily market index and quote signal (and each row's deviation from its daily value), a linear time trend, weekday dummies.
* Calendar **month** is deliberately not used, because November and December never appear in training.

### 4. Model

Exploration showed that, after accounting for route and equipment, log-rate moves linearly with the daily market index and quote signal and rises steadily over time (about 0.7 % per month). Tree models cannot extrapolate beyond values seen in training, so:

```
log(rate) = f_static(route, equipment, weight)  +  g_time(market signals, trend, weekday)
```

* `f_static`: `HistGradientBoostingRegressor` (31 leaves, learning rate 0.03, 800 iterations)
* `g_time`: ridge regression
* Both parts are fitted jointly by back-fitting. The implementation is in `src/model.py`.

### 5. December chart

`december_chart_inputs.csv` contains no `market_index` or `quote_signal`. For each December date, the day-level values observed in `validation.csv` (features only, no labels) are used, and row-level noise terms are set to zero. Predicted rates for the fixed load range from about $825 to $852.

---

## Limitations

* The upward trend is extrapolated linearly into November and December; a market shift not visible in the market index would not be captured.
* December chart predictions rely on the market signals contained in the validation file.
* On the Sep – Oct hold-out fold, tree-only models scored slightly better than the chosen model. The hold-out months had a market level below anything seen earlier in the year, which favours models that ignore the market signal. The chosen model was preferred for its best average error and stability across folds.
* The final validation metrics are computed by the assessment provider on hidden labels, so the exact final score cannot be verified here.

---

## Development notes

<!-- Edit or remove this section so it matches how you actually worked. -->
Developed with AI assistance (Claude). I ran every step myself, reviewed the results, and can explain all the code.
