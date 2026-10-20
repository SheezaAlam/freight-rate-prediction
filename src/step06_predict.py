"""STEP 6 - Predict the 12,000 validation loads + the 31 fixed December loads and fill the submission files."""
import joblib
import numpy as np
import pandas as pd
from common import DATA, PROC, MODELS, ROOT, NUM_FEATURES, TARGET, haversine_miles
from step03_features import build_features

model = joblib.load(MODELS / "freight_model.joblib")

# ---------------- 6a. validation.csv -> validation_predictions.csv ----------------
va_clean = pd.read_csv(PROC / "validation_clean.csv", parse_dates=["date"])
pred = model.predict(build_features(va_clean))
out = pd.DataFrame({"load_id": va_clean.load_id, "predicted_rate": np.round(pred, 2)})

template = pd.read_csv(DATA / "validation_predictions_template.csv")
out = template[["load_id"]].merge(out, on="load_id", how="left")          # keep the template's row order / ids
assert out.predicted_rate.notna().all() and len(out) == 12_000
out.to_csv(ROOT / "validation_predictions.csv", index=False)
print("validation_predictions.csv:", out.shape, "| predicted_rate range:", out.predicted_rate.min(), "-", out.predicted_rate.max())
print(out.predicted_rate.describe().round(1))

# sanity: implied $/mile should look like the training data (~2.1)
chk = va_clean.assign(pred=pred); print("implied median $/mile (validation):", round((chk.pred / chk.distance).median(), 3))

# ---------------- 6b. december_chart_inputs.csv ----------------
dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
tr = pd.read_csv(PROC / "train_clean.csv"); allrows = pd.concat([tr, va_clean])
city = pd.concat([allrows[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
                  allrows[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)]).groupby("city").first()
route = allrows[(allrows.pickup == "Lexington") & (allrows.delivery == "Fort Wayne")]

f = dec[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
f["pickup_lat"], f["pickup_lon"] = city.loc[f.pickup, "lat"].values, city.loc[f.pickup, "lon"].values
f["delivery_lat"], f["delivery_lon"] = city.loc[f.delivery, "lat"].values, city.loc[f.delivery, "lon"].values
f["haversine"] = haversine_miles(f.pickup_lat, f.pickup_lon, f.delivery_lat, f.delivery_lon)
f["circuity"] = f.distance / f.haversine
f["route_distance"] = route.distance.median() if len(route) else f.distance
f["distance_floor"] = 0
# December rows have no market_index / quote_signal -> use the DAY-LEVEL values observed for that date in validation.csv
# (features only, no labels). Row-level noise terms are set to 0 (= the typical load).
daily = va_clean.groupby("date")[["mi_day", "qs_day"]].first().reset_index()
f = f.merge(daily, on="date", how="left"); f["mi_dev"] = 0.0; f["qs_dev"] = 0.0
assert f[["mi_day", "qs_day"]].notna().all().all()
dec_pred = model.predict(build_features(f))
dec["predicted_rate"] = np.round(dec_pred, 2)
dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
dec.to_csv(DATA / "december_chart_inputs.csv", index=False)
print("\nDecember predictions (first 5 / last 3):\n", dec[["date", "predicted_rate"]].head(5).to_string(index=False), "\n", dec[["date", "predicted_rate"]].tail(3).to_string(index=False))
print("December min/max/mean:", dec.predicted_rate.min(), dec.predicted_rate.max(), round(dec.predicted_rate.mean(), 2))
