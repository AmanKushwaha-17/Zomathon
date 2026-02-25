import numpy as np
import pandas as pd
import random
from collections import defaultdict

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
    "Main": {
        "Side": 0.5,
        "Drink": 0.3,
        "Main": 0.15,
        "Dessert": 0.2
    },
    "Side": {
        "Dessert": 0.3,
        "Drink": 0.3,
        "Side": 0.1
    },
    "Dessert": {
        "Drink": 0.4
    }
}

# -----------------------------
# 1️⃣ Generate Users (With Segmentation)
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
        "category": category,
        "price": price,
        "popularity": popularity,
        "is_spicy": is_spicy
    })

items_df = pd.DataFrame(items)

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
    user_type = users_df.loc[users_df["user_id"] == user_id, "user_type"].values[0]
    
    cart_id = cart_id_counter
    cart_id_counter += 1
    
    # Start with Main
    main_item = random.choice(category_items["Main"])
    cart = [main_item]
    step_number = 1
    
    while True:
        last_item = cart[-1]
        last_row = items_df.loc[items_df["item_id"] == last_item].iloc[0]
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
        
        # Spicy boost
        if last_row["is_spicy"] == 1 and next_category == "Drink":
            transition_prob += 0.1
        
        # Clip probability to valid range
        transition_prob = max(0, min(1, transition_prob))
        
        if random.random() > transition_prob:
            break
        
        next_item = random.choice(category_items[next_category])
        
        # -----------------------------
        # CART FEATURES
        # -----------------------------
        cart_prices = items_df.loc[items_df["item_id"].isin(cart)]["price"]
        cart_value = cart_prices.sum()
        
        cart_categories = items_df.loc[items_df["item_id"].isin(cart)]["category"]
        count_main = (cart_categories == "Main").sum()
        count_side = (cart_categories == "Side").sum()
        count_dessert = (cart_categories == "Dessert").sum()
        count_drink = (cart_categories == "Drink").sum()
        
        next_item_row = items_df.loc[items_df["item_id"] == next_item].iloc[0]
        
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
            "candidate_category": next_item_row["category"],
            "candidate_price": next_item_row["price"],
            "candidate_popularity": next_item_row["popularity"],
            "transition_probability": transition_prob,
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
            neg_row = items_df.loc[items_df["item_id"] == neg_item].iloc[0]
            
            neg_transition_prob = transition_dict.get(neg_row["category"], 0)
            
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
                "candidate_category": neg_row["category"],
                "candidate_price": neg_row["price"],
                "candidate_popularity": neg_row["popularity"],
                "transition_probability": neg_transition_prob,
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
print("User type distribution:")
print(ranking_df["user_type"].value_counts())

ranking_df.to_csv("csao_ranking_data_v3_personalized.csv", index=False)