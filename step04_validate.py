"""STEP 4 - Split + validation strategy, and model comparison.

The real validation file is the FUTURE of the training file (Nov-Dec 2025 vs Jan-Oct 2025), so we validate in time order:
  fold1: train Jan-Apr -> test May-Jun
  fold2: train Jan-Jun -> test Jul-Aug
  fold3: train Jan-Aug -> test Sep-Oct   (= the final "hold-out": a 2-month look-ahead, same gap as Nov-Dec)
Corrupted labels (step 2) are removed from TRAINING only. Every fold is scored two ways:
  raw   = on all test rows (honest, includes the noisy labels)      clean = on test rows without flagged labels
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import PROC, TARGET, NUM_FEATURES, STATIC_FEATURES, metrics
from model import FreightRateModel

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
df = pd.read_csv(PROC / "features_train.csv", parse_dates=["date"])

FOLDS = {
    "fold1 Jan-Apr->May-Jun": ("2025-05-01", "2025-07-01"),
    "fold2 Jan-Jun->Jul-Aug": ("2025-07-01", "2025-09-01"),
    "fold3 Jan-Aug->Sep-Oct (HOLD-OUT)": ("2025-09-01", "2025-11-01"),
}


class SklearnLog:                       # wraps a plain sklearn model so it predicts rates from log-rate
    def __init__(self, est, feats): self.est, self.feats = est, feats
    def fit(self, X, rate): self.est.fit(X[self.feats], np.log(rate)); return self
    def predict(self, X): return np.exp(self.est.predict(X[self.feats]))


class MedianRatePerMile:                # naive benchmark: median $/mile by equipment x distance band
    bands = [0, 150, 300, 600, 1000, 1500, 2000, 4000]
    def _key(self, X): return list(zip(X.eq_flatbed.astype(int) + 2 * X.eq_reefer.astype(int), pd.cut(np.exp(X.log_distance), self.bands).astype(str)))
    def fit(self, X, rate):
        k = pd.Series(self._key(X), index=X.index); rpm = rate / np.exp(X.log_distance)
        self.med, self.glob = rpm.groupby(k).median(), rpm.median(); return self
    def predict(self, X): return np.array([self.med.get(k, self.glob) for k in self._key(X)]) * np.exp(X.log_distance.values)


def models():
    hgb = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, l2_regularization=1.0, random_state=0)
    return {
        "0 Baseline: median $/mile by equipment x band": MedianRatePerMile(),
        "1 Ridge (all features)": SklearnLog(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), NUM_FEATURES),
        "2 RandomForest (all features)": SklearnLog(RandomForestRegressor(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=0), NUM_FEATURES),
        "3 HistGB (all features)": SklearnLog(hgb(), NUM_FEATURES),
        "4 HistGB static only (no market/time info)": FreightRateModel(time_features=[]),
        "5 CHOSEN: HistGB static + linear market/trend": FreightRateModel(),
    }


rows = []
for fname, (start, end) in FOLDS.items():
    train = df[df.date < start]; test = df[(df.date >= start) & (df.date < end)]
    train_c = train[train.label_outlier == 0]
    ok = (test.label_outlier == 0).values
    for mname, mdl in models().items():
        pred = np.asarray(mdl.fit(train_c, train_c[TARGET]).predict(test))
        raw, cln = metrics(test[TARGET], pred), metrics(test[TARGET][ok], pred[ok])
        rows.append({"fold": fname, "model": mname, "raw_MAE": raw["MAE"], "raw_RMSE": raw["RMSE"], "raw_MAPE%": raw["MAPE_%"],
                     "clean_MAE": cln["MAE"], "clean_MAPE%": cln["MAPE_%"], "clean_R2": cln["R2"]})
res = pd.DataFrame(rows)
res.to_csv(PROC / "cv_results.csv", index=False)

print("=== Mean over the 3 time folds (sorted by clean MAE) ===")
print(res.groupby("model")[["raw_MAE", "raw_MAPE%", "clean_MAE", "clean_MAPE%", "clean_R2"]].mean().sort_values("clean_MAE").round(3))
print("\n=== Clean MAPE% per fold (stability) ===")
print(res.pivot(index="model", columns="fold", values="clean_MAPE%").round(2))
print("\n=== HOLD-OUT fold (Jan-Aug -> Sep-Oct) ===")
print(res[res.fold.str.contains("HOLD")].set_index("model")[["raw_MAE", "raw_RMSE", "raw_MAPE%", "clean_MAE", "clean_MAPE%", "clean_R2"]].round(3))

# ---- why a time split? a random split leaks the future and flatters the score ------------------------------------
print("\n=== Random 5-fold vs time-based, same chosen model (clean MAPE%) ===")
rand = []
for tri, tei in KFold(5, shuffle=True, random_state=0).split(df):
    a, b = df.iloc[tri], df.iloc[tei]; a = a[a.label_outlier == 0]; b = b[b.label_outlier == 0]
    rand.append(metrics(b[TARGET], FreightRateModel().fit(a, a[TARGET]).predict(b))["MAPE_%"])
tm = res[(res.model.str.startswith("5")) ]["clean_MAPE%"].mean()
print(f"random split: {np.mean(rand):.2f}%   |   time-based (mean of 3 folds): {tm:.2f}%")

# ---- small hyper-parameter search on the time folds -------------------------------------------------------------
print("\n=== Hyper-parameter search for the static HistGB (mean clean MAE over the 3 time folds) ===")
grid = [dict(max_leaf_nodes=l, learning_rate=lr, max_iter=it) for l in (15, 31, 63) for lr, it in ((0.06, 400), (0.03, 800))]
out = []
for p in grid:
    maes = []
    for start, end in FOLDS.values():
        a = df[(df.date < start) & (df.label_outlier == 0)]; b = df[(df.date >= start) & (df.date < end) & (df.label_outlier == 0)]
        maes.append(metrics(b[TARGET], FreightRateModel(**p).fit(a, a[TARGET]).predict(b))["MAE"])
    out.append({**p, "clean_MAE": np.mean(maes)})
grid_df = pd.DataFrame(out).sort_values("clean_MAE"); print(grid_df.round(3).to_string(index=False))
grid_df.iloc[[0]].drop(columns="clean_MAE").to_json(PROC / "best_params.json", orient="records")
print("\nBest params saved to data/processed/best_params.json")
