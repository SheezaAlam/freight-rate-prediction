# Freight Rate Prediction Challenge

Predicts `posted_rate` for 12,000 future loads. See `RUN_STEPS.md` for the exact commands.

## Setup
Place the provided CSVs in `data/` with these names: `train_test.csv`, `validation.csv`,
`validation_predictions_template.csv`, `december_chart_inputs.csv`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/step01_explore.py
python src/step02_clean.py
python src/step03_features.py
python src/step04_validate.py
python src/step05_train_final.py
python src/step06_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python src/step08_make_report.py
```

## Approach (short)
* Time-ordered validation (validation set is Nov-Dec, train is Jan-Oct): 3 expanding folds, last fold = 2-month hold-out.
* Cleaning: negative/missing weight, missing market_index, noisy distance, ~1.4% corrupted labels removed from training only.
* Model: `log(rate) = HistGradientBoosting(route, equipment, weight) + Ridge(daily market signals, linear trend, weekday)`.
