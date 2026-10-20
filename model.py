"""The chosen model: log(rate) = f_static(route, equipment, size)  +  g_time(market, trend, weekday).

f_static : HistGradientBoosting (captures the non-linear distance / route / equipment structure)
g_time   : Ridge regression    (market signals + trend are LINEAR in log-rate, so they extrapolate to unseen months)
They are fitted jointly by back-fitting (alternate until stable)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from common import STATIC_FEATURES, TIME_FEATURES


class FreightRateModel:
    def __init__(self, n_rounds=3, static_features=None, time_features=None, **hgb_params):
        self.n_rounds = n_rounds
        self.sf = static_features if static_features is not None else STATIC_FEATURES
        self.tf = time_features if time_features is not None else TIME_FEATURES
        self.hgb_params = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        self.hgb_params.update(hgb_params)

    def fit(self, df, rate):
        y = np.log(np.asarray(rate, float))
        g = np.zeros(len(df))
        for _ in range(self.n_rounds if self.tf else 1):
            self.static_ = HistGradientBoostingRegressor(**self.hgb_params).fit(df[self.sf], y - g)
            if not self.tf:
                break
            f = self.static_.predict(df[self.sf])
            self.time_ = Ridge(alpha=1e-3).fit(df[self.tf], y - f)
            g = self.time_.predict(df[self.tf])
        return self

    def predict(self, df):
        out = self.static_.predict(df[self.sf])
        if self.tf:
            out = out + self.time_.predict(df[self.tf])
        return np.exp(out)
