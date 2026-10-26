"""Hybrid recommender: collaborative (item-item) + content-based (TF-IDF)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    def __init__(self, items: pd.DataFrame, orders: pd.DataFrame, alpha: float = 0.6):
        """alpha = weight of collaborative score; (1 - alpha) goes to content score."""
        self.items = items.reset_index(drop=True)
        self.alpha = alpha
        self.item_ids = self.items["item_id"].tolist()
        self.pos = {iid: i for i, iid in enumerate(self.item_ids)}

        text = self.items["name"] + " " + self.items["category"] + " " + self.items["description"]
        self.content_sim = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(text))

        user_item = orders.pivot_table(index="user_id", columns="item_id", values="quantity",
                                       aggfunc="sum", fill_value=0)
        user_item = user_item.reindex(columns=self.item_ids, fill_value=0)
        self.collab_sim = cosine_similarity(user_item.T.values)

        pop = orders.groupby("item_id")["quantity"].sum().reindex(self.item_ids).fillna(0).values
        self.popularity = pop / pop.max() if pop.max() > 0 else pop

    def _format(self, order, scores, k):
        out = []
        for i in order[:k]:
            row = self.items.iloc[i]
            out.append({"item_id": int(row["item_id"]), "name": row["name"],
                        "category": row["category"], "score": round(float(scores[i]), 4)})
        return out

    def recommend(self, history, top_k: int = 5):
        """history: list of item_ids the user has ordered. Falls back to popular items."""
        idx = [self.pos[i] for i in history if i in self.pos]
        if not idx:  # cold start
            return self._format(np.argsort(self.popularity)[::-1], self.popularity, top_k)
        collab = self.collab_sim[idx].mean(axis=0)
        content = self.content_sim[idx].mean(axis=0)
        scores = self.alpha * collab + (1 - self.alpha) * content
        scores[idx] = -np.inf  # do not recommend what they already ordered
        return self._format(np.argsort(scores)[::-1], scores, top_k)


def hit_rate_at_k(items, orders, k=5, min_items=4):
    """Leave-one-out: hide each user's most recent distinct item, check it is in the top-k."""
    model = HybridRecommender(items, orders)
    hits = total = 0
    for _, g in orders.sort_values("timestamp").groupby("user_id"):
        seq = list(dict.fromkeys(g["item_id"]))
        if len(seq) < min_items:
            continue
        held_out, history = seq[-1], seq[:-1]
        recs = [r["item_id"] for r in model.recommend(history, k)]
        hits += held_out in recs
        total += 1
    return hits / total if total else 0.0, total
