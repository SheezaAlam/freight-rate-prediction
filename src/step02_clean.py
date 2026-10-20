"""STEP 2 - Clean the data. Fix input issues in BOTH files, flag suspicious labels in TRAIN only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from common import DATA, PROC, TARGET, WEIGHT_CAP, haversine_miles

tr = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
va = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
tr["split"], va["split"] = "train", "validation"
df = pd.concat([tr, va], ignore_index=True)

# --- 1. weight: negative values are sign errors, missing -> median by equipment (+ flags) ----------
df["weight_missing"] = df.weight.isna().astype(int)
df["weight"] = df.weight.abs()                                   # -31,000 -> 31,000 (distribution mirrors the positives)
df["weight_capped"] = (df.weight >= WEIGHT_CAP).astype(int)      # 47,500 is a hard cap, keep but flag
df["weight"] = df.weight.fillna(df.groupby("equipment").weight.transform("median"))

# --- 2. market_index: missing -> median of the same date (it is a market-wide daily signal) ---------
df["market_index_missing"] = df.market_index.isna().astype(int)
df["market_index"] = df.market_index.fillna(df.groupby("date").market_index.transform("median"))

# --- 3a. daily market signals: median over ALL rows of that date (train + validation FEATURES, no labels).
#         market_index / quote_signal are noisy per row around a smooth daily value -> split into day-level + deviation.
df["mi_day"] = df.groupby("date").market_index.transform("median")
df["qs_day"] = df.groupby("date").quote_signal.transform("median")
df["mi_dev"] = df.market_index - df.mi_day
df["qs_dev"] = df.quote_signal - df.qs_day

# --- 3b. distance: ~2% noise per row on the same route -> route-level median (uses features only, no labels)
df["distance_floor"] = (df.distance <= 70).astype(int)           # distances are floored at 70 miles
df["route_distance"] = df.groupby(["pickup", "delivery"]).distance.transform("median")
df["haversine"] = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
df["circuity"] = df.distance / df.haversine

# --- 4. label outliers (train only): robust fit of log(rate), flag huge residuals -------------------
t = df.split == "train"
X = pd.DataFrame({
    "ld": np.log(df.distance), "ld2": np.log(df.distance) ** 2, "mi": df.market_index,
    "reefer": (df.equipment == "Reefer").astype(int), "flatbed": (df.equipment == "Flatbed").astype(int),
})
y = np.log(df.loc[t, TARGET])
huber = HuberRegressor(max_iter=500).fit(X[t], y)
resid = y - huber.predict(X[t])
mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
print(f"robust residual sigma (MAD) = {mad:.3f}")
df["label_outlier"] = 0
df.loc[t, "label_outlier"] = (np.abs(resid) > 6 * mad).astype(int)
n_out = int(df.label_outlier.sum())
print(f"flagged label outliers: {n_out} ({n_out / t.sum():.2%} of train) | "
      f"high x{np.exp(resid[resid > 6 * mad]).median():.1f} / low x{np.exp(resid[resid < -6 * mad]).median():.2f}")

# --- save -----------------------------------------------------------------------------------------
df[df.split == "train"].drop(columns="split").to_csv(PROC / "train_clean.csv", index=False)
df[df.split == "validation"].drop(columns=["split", TARGET, "label_outlier"]).to_csv(PROC / "validation_clean.csv", index=False)
print("\nAfter cleaning -> remaining NaNs:", int(df.drop(columns=[TARGET]).isna().sum().sum()),
      "| negative weights:", int((df.weight < 0).sum()))
print("Saved data/processed/train_clean.csv and validation_clean.csv")
