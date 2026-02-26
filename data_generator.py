import numpy as np
import pandas as pd
import random
from collections import defaultdict
import json

np.random.seed(42)
random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
NUM_USERS = 500
NUM_ITEMS = 150
NUM_ORDERS = 6000
NEGATIVES_PER_STEP = 10

CATEGORIES = ["Main", "Side", "Dessert", "Drink"]
CATEGORY_PROBS = [0.4, 0.25, 0.15, 0.20]

USER_TYPES = ["Budget", "Regular", "Premium"]
USER_TYPE_PROBS = [0.5, 0.3, 0.2]

TRANSITIONS = {
    "Main": {"Side": 0.5, "Drink": 0.3, "Main": 0.15, "Dessert": 0.2},
    "Side": {"Dessert": 0.3, "Drink": 0.3, "Side": 0.1},
    "Dessert": {"Drink": 0.4}
}

# Item name mapping
def get_item_name(category):
    if category == "Main":
        return random.choice(["Spicy Biryani", "Paneer Butter Masala", "Veg Burger", "Chicken Curry"])
    elif category == "Side":
        return random.choice(["Garlic Bread", "French Fries", "Raita", "Salad"])
    elif category == "Dessert":
        return random.choice(["Chocolate Cake", "Ice Cream", "Gulab Jamun"])
    else:
        return random.choice(["Cold Drink", "Lassi", "Iced Tea", "Milkshake"])

# -----------------------------
# 1️⃣ Generate Users
# -----------------------------
users = []

for user_id in range(NUM_USERS):
    user_type = np.random.choice(USER_TYPES, p=USER_TYPE_PROBS)
    users.append({
        "user_id": user_id,
        "user_type": user_type
    })

users_df = pd.DataFrame(users)

# -----------------------------
# 2️⃣ Generate Items
# -----------------------------
items = []

for item_id in range(NUM_ITEMS):
    category = np.random.choice(CATEGORIES, p=CATEGORY_PROBS)
    item_name = get_item_name(category)

    if category == "Main":
        price = np.random.randint(180, 350)
    elif category == "Side":
        price = np.random.randint(40, 120)
    elif category == "Dessert":
        price = np.random.randint(60, 150)
    else:
        price = np.random.randint(30, 100)

    popularity = np.random.zipf(2)
    is_spicy = np.random.choice([0, 1], p=[0.7, 0.3])

    items.append({
        "item_id": item_id,
        "item_name": item_name,
        "category": category,
        "price": price,
        "popularity": popularity,
        "is_spicy": is_spicy
    })

items_df = pd.DataFrame(items)

# -----------------------------
# 2.5️⃣ Generate Item Embeddings (Simulated MiniLM)
# -----------------------------
EMBEDDING_DIM = 32

category_base_vectors = {
    "Main": np.random.normal(0, 1, EMBEDDING_DIM),
    "Side": np.random.normal(0, 1, EMBEDDING_DIM),
    "Dessert": np.random.normal(0, 1, EMBEDDING_DIM),
    "Drink": np.random.normal(0, 1, EMBEDDING_DIM),
}

item_embeddings = {}

for _, row in items_df.iterrows():
    base_vector = category_base_vectors[row["category"]]
    item_vector = base_vector + np.random.normal(0, 0.1, EMBEDDING_DIM)
    item_vector = item_vector / np.linalg.norm(item_vector)
    item_embeddings[row["item_id"]] = item_vector

# Persist artifacts for inference so features match training-time generation.
embeddings_matrix = np.vstack(
    [item_embeddings[item_id] for item_id in sorted(item_embeddings.keys())]
)
embedding_item_ids = np.array(sorted(item_embeddings.keys()), dtype=np.int64)
np.savez(
    "csao_feature_artifacts.npz",
    embedding_item_ids=embedding_item_ids,
    embeddings_matrix=embeddings_matrix
)
with open("csao_transitions.json", "w", encoding="utf-8") as f:
    json.dump(TRANSITIONS, f, indent=2)

# Category lookup
category_items = defaultdict(list)
for _, row in items_df.iterrows():
    category_items[row["category"]].append(row["item_id"])

# -----------------------------
# 3️⃣ Simulate Orders
# -----------------------------
ranking_rows = []
cart_id_counter = 0

for order_id in range(NUM_ORDERS):

    user_id = random.randint(0, NUM_USERS - 1)
    user_type = users_df.loc[
        users_df["user_id"] == user_id, "user_type"
    ].values[0]

    cart_id = cart_id_counter
    cart_id_counter += 1

    # Start with Main
    main_item = random.choice(category_items["Main"])
    cart = [main_item]
    step_number = 1

    while True:

        last_item = cart[-1]
        last_row = items_df.loc[
            items_df["item_id"] == last_item
        ].iloc[0]

        last_category = last_row["category"]

        if last_category not in TRANSITIONS:
            break

        transition_dict = TRANSITIONS[last_category]

        categories = list(transition_dict.keys())
        probabilities = list(transition_dict.values())

        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()

        next_category = np.random.choice(categories, p=probabilities)
        transition_prob = transition_dict[next_category]

        # -----------------------------
        # Personalization Logic
        # -----------------------------
        if user_type == "Premium" and next_category == "Dessert":
            transition_prob += 0.15

        if user_type == "Budget" and next_category == "Dessert":
            transition_prob -= 0.15

        if user_type == "Premium" and next_category == "Main":
            transition_prob += 0.05

        if last_row["is_spicy"] == 1 and next_category == "Drink":
            transition_prob += 0.1

        transition_prob = max(0, min(1, transition_prob))

        if random.random() > transition_prob:
            break

        next_item = random.choice(category_items[next_category])

        # -----------------------------
        # CART FEATURES
        # -----------------------------
        cart_prices = items_df.loc[
            items_df["item_id"].isin(cart)
        ]["price"]

        cart_value = cart_prices.sum()

        cart_categories = items_df.loc[
            items_df["item_id"].isin(cart)
        ]["category"]

        count_main = (cart_categories == "Main").sum()
        count_side = (cart_categories == "Side").sum()
        count_dessert = (cart_categories == "Dessert").sum()
        count_drink = (cart_categories == "Drink").sum()

        next_item_row = items_df.loc[
            items_df["item_id"] == next_item
        ].iloc[0]

        # -----------------------------
        # EMBEDDING SIMILARITY
        # -----------------------------
        cart_vectors = [item_embeddings[item] for item in cart]
        cart_embedding = np.mean(cart_vectors, axis=0)
        cart_embedding = cart_embedding / np.linalg.norm(cart_embedding)

        candidate_embedding = item_embeddings[next_item]
        embedding_similarity = np.dot(cart_embedding, candidate_embedding)

        # -----------------------------
        # POSITIVE ROW
        # -----------------------------
        ranking_rows.append({
            "cart_id": cart_id,
            "step": step_number,
            "user_id": user_id,
            "user_type": user_type,
            "cart_size": len(cart),
            "cart_value": cart_value,
            "count_main": count_main,
            "count_side": count_side,
            "count_dessert": count_dessert,
            "count_drink": count_drink,
            "last_category": last_category,
            "candidate_item": next_item,
            "candidate_item_name": next_item_row["item_name"],
            "candidate_category": next_item_row["category"],
            "candidate_price": next_item_row["price"],
            "candidate_popularity": next_item_row["popularity"],
            "transition_probability": transition_prob,
            "embedding_similarity": embedding_similarity,
            "label": 1
        })

        # -----------------------------
        # NEGATIVE SAMPLING
        # -----------------------------
        negatives = []

        while len(negatives) < NEGATIVES_PER_STEP:
            neg_item = random.randint(0, NUM_ITEMS - 1)
            if neg_item not in cart and neg_item != next_item:
                negatives.append(neg_item)

        for neg_item in negatives:

            neg_row = items_df.loc[
                items_df["item_id"] == neg_item
            ].iloc[0]

            neg_transition_prob = transition_dict.get(
                neg_row["category"], 0
            )

            neg_embedding = item_embeddings[neg_item]
            embedding_similarity_neg = np.dot(cart_embedding, neg_embedding)

            ranking_rows.append({
                "cart_id": cart_id,
                "step": step_number,
                "user_id": user_id,
                "user_type": user_type,
                "cart_size": len(cart),
                "cart_value": cart_value,
                "count_main": count_main,
                "count_side": count_side,
                "count_dessert": count_dessert,
                "count_drink": count_drink,
                "last_category": last_category,
                "candidate_item": neg_item,
                "candidate_item_name": neg_row["item_name"],
                "candidate_category": neg_row["category"],
                "candidate_price": neg_row["price"],
                "candidate_popularity": neg_row["popularity"],
                "transition_probability": neg_transition_prob,
                "embedding_similarity": embedding_similarity_neg,
                "label": 0
            })

        cart.append(next_item)
        step_number += 1

        if len(cart) >= 4:
            break

# -----------------------------
# 4️⃣ Final Dataset
# -----------------------------
ranking_df = pd.DataFrame(ranking_rows)

print("Total ranking rows:", len(ranking_df))
print("Unique carts:", ranking_df["cart_id"].nunique())
print("Average cart steps:", ranking_df["step"].mean())
print("Positive ratio:", ranking_df["label"].mean())

ranking_df.to_csv(
    "csao_ranking_data_v3_personalized.csv",
    index=False
)
