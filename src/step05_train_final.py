"""STEP 5 - Train the final model on ALL labelled data (Jan-Oct, corrupted labels removed) and save it."""
import json
import joblib
import numpy as np
import pandas as pd
from common import PROC, MODELS, TARGET, metrics
from model import FreightRateModel

df = pd.read_csv(PROC / "features_train.csv", parse_dates=["date"])
params = json.loads((PROC / "best_params.json").read_text())[0]
print("hyper-parameters from step 4:", params)

train = df[df.label_outlier == 0]
print(f"training rows: {len(train):,} of {len(df):,} (dropped {len(df) - len(train)} corrupted labels)")
model = FreightRateModel(**params).fit(train, train[TARGET])

# in-sample sanity check only (the honest estimate is the time-based CV from step 4)
fit = metrics(train[TARGET], model.predict(train))
print("in-sample (not an estimate of future accuracy):", {k: round(v, 3) for k, v in fit.items()})

lin = model.time_
print("\nLearned linear market/trend effects on log(rate):")
for name, coef in zip(lin.feature_names_in_, lin.coef_):
    print(f"  {name:8s} {coef:+.4f}")

joblib.dump(model, MODELS / "freight_model.joblib")
print("\nSaved models/freight_model.joblib")
