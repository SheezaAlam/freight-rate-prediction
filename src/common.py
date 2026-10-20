"""Shared helpers used by every step of the pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = DATA / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for p in (PROC, REPORTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

TARGET = "posted_rate"
WEIGHT_CAP = 47_500.0

# Features fed to the model (built in step03_features.py)
# STATIC = what the load itself looks like (route, equipment, size). TIME = market / calendar signals.
STATIC_FEATURES = [
    "log_distance", "route_distance", "haversine", "circuity",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "eq_flatbed", "eq_reefer", "weight", "distance_floor",
]
TIME_FEATURES = [
    "mi_day", "mi_dev", "qs_day", "qs_dev", "t",
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
]
NUM_FEATURES = STATIC_FEATURES + TIME_FEATURES


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / y_true) * 100),
        "MedAPE_%": float(np.median(np.abs(err) / y_true) * 100),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }
