"""
Generates a SYNTHETIC canteen dataset (items.csv, orders.csv) so the ML service
can run and be tested without real order data.

To use real data instead, export your MongoDB orders to ml-service/data/orders.csv
with the columns: user_id, item_id, quantity, timestamp (and items.csv with
item_id, name, category, description).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

ITEMS = [
    ("Masala Dosa", "Breakfast", "crispy rice crepe with spiced potato filling and chutney"),
    ("Idli Sambar", "Breakfast", "steamed rice cakes served with lentil sambar and chutney"),
    ("Poha", "Breakfast", "flattened rice with onion peanuts and mild spices"),
    ("Upma", "Breakfast", "savory semolina porridge with vegetables"),
    ("Veg Thali", "Meals", "full lunch plate with rice roti dal curry and salad"),
    ("Paneer Butter Masala Meal", "Meals", "paneer curry with rice and roti"),
    ("Chicken Biryani", "Meals", "fragrant basmati rice cooked with spiced chicken"),
    ("Veg Biryani", "Meals", "basmati rice cooked with vegetables and spices"),
    ("Curd Rice", "Meals", "cooling rice with yogurt and tempering"),
    ("Rajma Chawal", "Meals", "kidney bean curry served with rice"),
    ("Samosa", "Snacks", "fried pastry filled with spiced potato and peas"),
    ("Veg Puff", "Snacks", "baked flaky pastry with vegetable filling"),
    ("French Fries", "Snacks", "crispy fried potato sticks"),
    ("Veg Sandwich", "Snacks", "grilled sandwich with vegetables and cheese"),
    ("Vada Pav", "Snacks", "spiced potato fritter in a soft bun"),
    ("Masala Chai", "Beverages", "hot milk tea with ginger and spices"),
    ("Filter Coffee", "Beverages", "strong south indian coffee with milk"),
    ("Cold Coffee", "Beverages", "chilled coffee blended with milk and ice"),
    ("Lemon Soda", "Beverages", "fresh lime soda sweet or salted"),
    ("Mango Shake", "Beverages", "thick milkshake made with mango pulp"),
    ("Gulab Jamun", "Desserts", "fried milk dumplings soaked in sugar syrup"),
    ("Ice Cream Cup", "Desserts", "vanilla or chocolate ice cream cup"),
    ("Brownie", "Desserts", "rich chocolate brownie"),
    ("Kheer", "Desserts", "rice pudding with milk cardamom and nuts"),
]
HOUR_PEAKS = {"Breakfast": (9, 1.3), "Meals": (13, 1.4), "Snacks": (16, 1.6),
              "Beverages": (11, 3.5), "Desserts": (14, 3.0)}


def main(n_users=300, n_days=60, seed=7):
    rng = np.random.default_rng(seed)
    items = pd.DataFrame(ITEMS, columns=["name", "category", "description"])
    items.insert(0, "item_id", range(1, len(items) + 1))
    cats = items["category"].unique()
    rows = []
    start = pd.Timestamp("2026-01-05")
    for user in range(1, n_users + 1):
        fav = rng.choice(cats, size=2, replace=False)
        weights = np.array([3.0 if c in fav else 0.6 for c in items["category"]])
        weights /= weights.sum()
        for _ in range(rng.integers(8, 30)):
            item = items.iloc[rng.choice(len(items), p=weights)]
            peak, spread = HOUR_PEAKS[item["category"]]
            hour = int(np.clip(rng.normal(peak, spread), 8, 18))
            day = start + pd.Timedelta(days=int(rng.integers(0, n_days)))
            ts = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            rows.append((user, int(item["item_id"]), int(rng.integers(1, 3)), ts))
    orders = pd.DataFrame(rows, columns=["user_id", "item_id", "quantity", "timestamp"])
    items.to_csv(os.path.join(HERE, "items.csv"), index=False)
    orders.to_csv(os.path.join(HERE, "orders.csv"), index=False)
    print(f"Wrote {len(items)} items and {len(orders)} SYNTHETIC orders to {HERE}")


if __name__ == "__main__":
    main()
