"""Flask ML service for AI Canteen Connect (runs on port 8000 by default)."""
import os
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

import forecasting
from data.generate_sample_data import main as generate_sample_data
from recommendation import HybridRecommender

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS_CSV = os.path.join(HERE, "data", "items.csv")
ORDERS_CSV = os.path.join(HERE, "data", "orders.csv")
MODEL_PATH = os.path.join(HERE, "models", "demand_model.pkl")

app = Flask(__name__)
CORS(app)

if not (os.path.exists(ITEMS_CSV) and os.path.exists(ORDERS_CSV)):
    print("No data found - generating SYNTHETIC demo data. Replace data/*.csv with real orders.")
    generate_sample_data()

items = pd.read_csv(ITEMS_CSV)
orders = pd.read_csv(ORDERS_CSV)
recommender = HybridRecommender(items, orders)
if not os.path.exists(MODEL_PATH):
    print("Training demand model:", forecasting.train(orders, items, MODEL_PATH))
demand_bundle = forecasting.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify(status="ok", items=len(items), orders=len(orders),
                   demand_model_metrics=demand_bundle["metrics"])


@app.post("/recommend")
def recommend():
    body = request.get_json(silent=True) or {}
    history = body.get("history", [])
    top_k = int(body.get("top_k", 5))
    return jsonify(recommendations=recommender.recommend(history, top_k))


@app.get("/forecast")
def forecast():
    try:
        category = request.args["category"]
        day_of_week = int(request.args["day_of_week"])
        hour = int(request.args["hour"])
        value = forecasting.predict(demand_bundle, category, day_of_week, hour)
    except KeyError as e:
        return jsonify(error=f"missing parameter {e}"), 400
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(category=category, day_of_week=day_of_week, hour=hour,
                   predicted_items=round(value, 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
