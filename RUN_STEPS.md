# Run the project step by step (one command at a time)

Run everything from the project root. Each step prints its own results - explain them in your video.

| # | Command | What it does | What to say in the video |
|---|---------|--------------|--------------------------|
| 0 | `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`) then `pip install -r requirements.txt` | Isolated environment + dependencies | "Everything is reproducible from requirements.txt" |
| 1 | `python src/step01_explore.py` | EDA + data-quality audit (changes nothing) | Validation is the future (Nov-Dec); weight negatives/missing/cap; market_index missing; ~1.4% corrupted labels; rate/mile falls with distance and market_index drifts over time |
| 2 | `python src/step02_clean.py` | Fixes inputs, flags corrupted labels | Abs() on negative weights, median impute + flags, same-date impute for market_index, route-median distance, robust-regression label outlier flag (train only) |
| 3 | `python src/step03_features.py` | Builds model features | Static features vs time/market features; no calendar month on purpose |
| 4 | `python src/step04_validate.py` | Time-based validation, model comparison, tuning | Split design (3 expanding folds, last = 2-month hold-out), why not random split, why the chosen model wins |
| 5 | `python src/step05_train_final.py` | Fits the chosen model on all Jan-Oct data | Final fit, learned market/trend coefficients |
| 6 | `python src/step06_predict.py` | Writes `validation_predictions.csv` + fills December file | 12,000 predictions; how December inputs were completed |
| 7 | `python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv` | Official scorer, creates `scorer_results/candidate_december.png` | Show the chart |
| 8 | `python src/step08_make_report.py` | Builds `report.docx` | The report deliverable |

Code to walk through at the end: `src/step02_clean.py` (cleaning), `src/model.py` (model), `src/step04_validate.py` (split loop).
