"""Demand forecasting: XGBoost regressor on hourly demand per category."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURES = ["hour", "day_of_week", "is_weekend", "category_code"]


def hourly_demand(orders: pd.DataFrame, items: pd.DataFrame):
    df = orders.merge(items[["item_id", "category"]], on="item_id")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    g = df.groupby(["date", "hour", "category"])["quantity"].sum().reset_index(name="demand")
    g["day_of_week"] = pd.to_datetime(g["date"]).dt.dayofweek
    g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
    categories = sorted(g["category"].unique())
    g["category_code"] = g["category"].map({c: i for i, c in enumerate(categories)})
    return g, categories


def train(orders, items, model_path):
    data, categories = hourly_demand(orders, items)
    X_tr, X_te, y_tr, y_te = train_test_split(data[FEATURES], data["demand"],
                                              test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        # forecast accuracy defined as 1 - WAPE (weighted absolute percentage error)
        "accuracy_1_minus_wape": float(1 - np.abs(y_te - pred).sum() / y_te.sum()),
        "test_rows": int(len(y_te)),
    }
    joblib.dump({"model": model, "categories": categories, "metrics": metrics}, model_path)
    return metrics


def load(model_path):
    return joblib.load(model_path)


def predict(bundle, category, day_of_week, hour):
    if category not in bundle["categories"]:
        raise ValueError(f"unknown category '{category}'")
    row = pd.DataFrame([{
        "hour": hour, "day_of_week": day_of_week, "is_weekend": int(day_of_week >= 5),
        "category_code": bundle["categories"].index(category)}])
    return max(0.0, float(bundle["model"].predict(row[FEATURES])[0]))
