"""STEP 1 - Explore the data and surface data-quality issues (read-only, changes nothing)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import DATA, REPORTS, TARGET, haversine_miles

tr = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
va = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])

print("== Shapes ==", tr.shape, va.shape)
print("\n== Date ranges (note: validation is the FUTURE of train) ==")
print("train:", tr.date.min().date(), "->", tr.date.max().date())
print("valid:", va.date.min().date(), "->", va.date.max().date())

print("\n== Missing values ==")
print(pd.DataFrame({"train": tr.isna().sum(), "validation": va.isna().sum()}))

print("\n== Duplicates ==")
print("duplicate load_id:", tr.load_id.duplicated().sum(), "| duplicate rows:", tr.drop(columns="load_id").duplicated().sum())

print("\n== Weight issues ==")
for n, d in (("train", tr), ("validation", va)):
    print(f"{n}: negative={int((d.weight < 0).sum())}  at_cap_47500={int((d.weight == 47500).sum())}  missing={int(d.weight.isna().sum())}")

print("\n== Distance vs straight-line (haversine) ==")
for d in (tr, va):
    d["haversine"] = haversine_miles(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    d["circuity"] = d.distance / d.haversine
print(tr.circuity.describe().round(3))
print("rows with circuity > 1.5 (short hauls / floor at 70 mi):", int((tr.circuity > 1.5).sum()), "| distance==70:", int((tr.distance == 70).sum()))
print("distance noise within the same route (std/median):",
      round((tr.groupby(["pickup", "delivery"]).distance.transform("std") / tr.groupby(["pickup", "delivery"]).distance.transform("median")).mean(), 4))

print("\n== Label (posted_rate) sanity: rate per mile ==")
tr["rpm"] = tr[TARGET] / tr.distance
print(tr.rpm.describe(percentiles=[.005, .01, .5, .99, .995]).round(3))
print("rate/mile > 5:", int((tr.rpm > 5).sum()), "| rate/mile < 0.7:", int((tr.rpm < 0.7).sum()), "(suspicious labels)")

print("\n== Market index by month (regime shift over time) ==")
tr["month"] = tr.date.dt.to_period("M"); va["month"] = va.date.dt.to_period("M")
print(pd.concat([tr.groupby("month").agg(rate_per_mile=("rpm", "median"), market_index=("market_index", "mean")),
                 va.groupby("month").agg(market_index=("market_index", "mean"))]).round(3))

print("\n== Rate per mile by equipment / distance band ==")
print(tr.groupby("equipment").rpm.median().round(3))
print(tr.groupby(pd.cut(tr.distance, [0, 150, 300, 600, 1000, 1500, 2000, 3500]), observed=True).rpm.median().round(3))

rt = set(zip(tr.pickup, tr.delivery))
print("\n== Share of validation routes never seen in train:",
      round(1 - np.mean([r in rt for r in zip(va.pickup, va.delivery)]), 3))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].hist(tr[TARGET], bins=80); ax[0, 0].set_title("posted_rate")
ax[0, 1].hist(np.log(tr.rpm), bins=80); ax[0, 1].set_title("log(rate per mile) - note both tails")
ax[1, 0].scatter(tr.distance, tr.rpm, s=2, alpha=.3); ax[1, 0].set_ylim(0, 6); ax[1, 0].set_title("rate/mile vs distance")
tr.groupby("date").market_index.median().plot(ax=ax[1, 1], label="train")
va.groupby("date").market_index.median().plot(ax=ax[1, 1], label="validation"); ax[1, 1].legend(); ax[1, 1].set_title("market_index over time")
plt.tight_layout(); plt.savefig(REPORTS / "eda_overview.png", dpi=130)
print("\nSaved reports/eda_overview.png")
