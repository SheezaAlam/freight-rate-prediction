"""STEP 3 - Feature engineering (pure functions, no fitting => no leakage)."""
import numpy as np
import pandas as pd
from common import PROC, NUM_FEATURES, TARGET


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_distance"] = np.log(df.distance)
    out["route_distance"] = df.route_distance
    out["haversine"] = df.haversine
    out["circuity"] = df.circuity
    for c in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        out[c] = df[c]
    out["weight"] = df.weight
    out["distance_floor"] = df.distance_floor
    out["eq_flatbed"] = (df.equipment == "Flatbed").astype(int)
    out["eq_reefer"] = (df.equipment == "Reefer").astype(int)
    # --- time / market block ---
    for c in ("mi_day", "mi_dev", "qs_day", "qs_dev"):
        out[c] = df[c]
    d = pd.to_datetime(df.date)
    out["t"] = (d - pd.Timestamp("2025-01-01")).dt.days / 100.0        # linear trend (per 100 days)
    for k in range(1, 7):                                                # weekday dummies (Monday = baseline)
        out[f"dow_{k}"] = (d.dt.dayofweek == k).astype(int)
    return out[NUM_FEATURES]   # calendar MONTH is deliberately not used: Nov/Dec never appear in training


if __name__ == "__main__":
    tr = pd.read_csv(PROC / "train_clean.csv", parse_dates=["date"])
    va = pd.read_csv(PROC / "validation_clean.csv", parse_dates=["date"])
    Xtr, Xva = build_features(tr), build_features(va)
    Xtr.assign(date=tr.date, distance=tr.distance, equipment=tr.equipment, label_outlier=tr.label_outlier, **{TARGET: tr[TARGET]}).to_csv(PROC / "features_train.csv", index=False)
    Xva.assign(load_id=va.load_id, date=va.date).to_csv(PROC / "features_validation.csv", index=False)
    print("train features:", Xtr.shape, "| validation features:", Xva.shape)
    print("feature list:", list(Xtr.columns))
    print(Xtr.describe().T[["mean", "min", "max"]].round(2))
