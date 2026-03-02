import numpy as np
import pandas as pd
import random
import json

np.random.seed(42)
random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
NUM_USERS = 2000
NUM_ORDERS = 6000
NEGATIVES_PER_STEP = 20
NUM_RESTAURANTS = 300

CATEGORIES = ["Main", "Side", "Dessert", "Drink"]
CATEGORY_PROBS = [0.4, 0.25, 0.15, 0.20]

CITIES = ["Delhi", "Mumbai", "Bangalore"]
CUISINES = ["North Indian", "Chinese", "Italian", "South Indian", "Fast Food"]

CITY_PROBS = [0.4, 0.35, 0.25]
CUISINE_PROBS = [0.3, 0.2, 0.15, 0.2, 0.15]

USER_TYPES = ["Budget", "Regular", "Premium"]
USER_TYPE_PROBS = [0.5, 0.3, 0.2]

TRANSITIONS = {
    "Main": {"Side": 0.5, "Drink": 0.3, "Main": 0.15, "Dessert": 0.2},
    "Side": {"Dessert": 0.3, "Drink": 0.3, "Side": 0.1},
    "Dessert": {"Drink": 0.4}
}

HOUR_DISTRIBUTION = {
    0: 0.01, 1: 0.005, 2: 0.003, 3: 0.002,
    4: 0.003, 5: 0.01,
    6: 0.03, 7: 0.05, 8: 0.06, 9: 0.05, 10: 0.04,
    11: 0.07, 12: 0.10, 13: 0.12, 14: 0.08, 15: 0.05,
    16: 0.04, 17: 0.06,
    18: 0.08, 19: 0.10, 20: 0.12, 21: 0.09, 22: 0.06,
    23: 0.03
}
HOURS = np.array(list(HOUR_DISTRIBUTION.keys()))
HOUR_PROBS = np.array(list(HOUR_DISTRIBUTION.values()), dtype=float)
HOUR_PROBS = HOUR_PROBS / HOUR_PROBS.sum()

# Item name mapping
def get_item_name(category, cuisine):
    cuisine_mains = {
        "North Indian": [
            "Chicken Biryani", "Paneer Butter Masala", "Dal Makhani",
            "Tandoori Chicken", "Chole Bhature", "Butter Naan Combo"
        ],
        "Chinese": [
            "Hakka Noodles", "Chilli Chicken", "Veg Manchurian",
            "Schezwan Fried Rice", "Spring Roll Platter", "Hot Garlic Noodles"
        ],
        "Italian": [
            "Margherita Pizza", "Pasta Alfredo", "Lasagna",
            "Penne Arrabbiata", "Mushroom Risotto", "Four Cheese Pizza"
        ],
        "South Indian": [
            "Masala Dosa", "Idli Sambar", "Veg Uttapam",
            "Curd Rice Bowl", "Podi Dosa", "Mini Tiffin Combo"
        ],
        "Fast Food": [
            "Veg Burger", "Chicken Wrap", "Loaded Fries",
            "Zinger Burger", "Peri Peri Wrap", "Cheese Burst Sandwich"
        ]
    }

    sides_by_cuisine = {
        "North Indian": ["Raita", "Tandoori Roti", "Jeera Rice", "Onion Salad"],
        "Chinese": ["Veg Spring Rolls", "Chilli Potatoes", "Dimsum Basket", "Kimchi Salad"],
        "Italian": ["Garlic Bread", "Bruschetta", "Herbed Fries", "Caesar Salad"],
        "South Indian": ["Coconut Chutney", "Medu Vada", "Tomato Chutney", "Mini Idli Bowl"],
        "Fast Food": ["French Fries", "Onion Rings", "Cheese Nuggets", "Coleslaw"]
    }

    desserts_by_cuisine = {
        "North Indian": ["Gulab Jamun", "Rasmalai", "Gajar Halwa", "Kulfi"],
        "Chinese": ["Honey Noodles", "Darsaan", "Sesame Balls", "Mango Pudding"],
        "Italian": ["Tiramisu", "Panna Cotta", "Chocolate Cannoli", "Gelato"],
        "South Indian": ["Kesari", "Payasam", "Sweet Pongal", "Coconut Barfi"],
        "Fast Food": ["Brownie", "Chocolate Cake", "Ice Cream Sundae", "Choco Lava Cup"]
    }

    drinks_by_cuisine = {
        "North Indian": ["Lassi", "Masala Chaas", "Nimbu Soda", "Rose Milk"],
        "Chinese": ["Iced Tea", "Lemon Cooler", "Peach Soda", "Green Tea"],
        "Italian": ["Cold Coffee", "Lemonade", "Mocha Shake", "Sparkling Lime"],
        "South Indian": ["Filter Coffee", "Buttermilk", "Tender Coconut", "Jigarthanda"],
        "Fast Food": ["Cold Drink", "Milkshake", "Iced Cola", "Orange Fizz"]
    }

    adjectives = ["Classic", "Special", "Signature", "Chef's"]

    if category == "Main":
        base_item = random.choice(cuisine_mains.get(cuisine, ["Chef Special"]))
    elif category == "Side":
        base_item = random.choice(sides_by_cuisine.get(cuisine, ["House Side"]))
    elif category == "Dessert":
        base_item = random.choice(desserts_by_cuisine.get(cuisine, ["House Dessert"]))
    else:
        base_item = random.choice(drinks_by_cuisine.get(cuisine, ["House Drink"]))

    if random.random() < 0.35:
        return f"{random.choice(adjectives)} {base_item}"
    return base_item

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
# 1.2 Initialize User Behavioral Memory
# -----------------------------
user_memory = {}

for user_id in users_df["user_id"]:
    user_memory[user_id] = {
        "preferred_cuisine": np.random.choice(CUISINES),
        "dessert_affinity": np.random.uniform(0.0, 0.3),
        "drink_affinity": np.random.uniform(0.0, 0.3),
        "avg_spend_bias": np.random.uniform(-0.05, 0.1)
    }

# -----------------------------
# 1.5 Generate Restaurants
# -----------------------------
restaurants = []

for restaurant_id in range(NUM_RESTAURANTS):

    city = np.random.choice(CITIES, p=CITY_PROBS)
    cuisine = np.random.choice(CUISINES, p=CUISINE_PROBS)

    price_range = np.random.choice(["Budget", "Mid", "Premium"], p=[0.5, 0.35, 0.15])
    rating = np.round(np.random.uniform(3.5, 4.8), 1)
    is_chain = np.random.choice([0, 1], p=[0.7, 0.3])

    restaurants.append({
        "restaurant_id": restaurant_id,
        "city": city,
        "cuisine": cuisine,
        "price_range": price_range,
        "rating": rating,
        "is_chain": is_chain
    })

restaurants_df = pd.DataFrame(restaurants)

# -----------------------------
# 2️⃣ Generate Items (Structured Per Restaurant)
# -----------------------------
items = []
item_id = 0

for _, restaurant_row in restaurants_df.iterrows():
    restaurant_id = restaurant_row["restaurant_id"]
    cuisine = restaurant_row["cuisine"]
    city = restaurant_row["city"]
    price_range = restaurant_row["price_range"]

    # 5 items per category
    for category in ["Main", "Side", "Dessert", "Drink"]:
        for _ in range(5):
            item_name = get_item_name(category, cuisine)

            if category == "Main":
                base_price = np.random.randint(180, 350)
            elif category == "Side":
                base_price = np.random.randint(40, 120)
            elif category == "Dessert":
                base_price = np.random.randint(60, 150)
            else:
                base_price = np.random.randint(30, 100)

            if price_range == "Premium":
                price = int(base_price * 1.4)
            elif price_range == "Budget":
                price = int(base_price * 0.8)
            else:
                price = base_price

            popularity = np.random.zipf(2)
            is_spicy = np.random.choice([0, 1], p=[0.7, 0.3])

            items.append({
                "item_id": item_id,
                "restaurant_id": restaurant_id,
                "city": city,
                "cuisine": cuisine,
                "item_name": item_name,
                "category": category,
                "price": price,
                "popularity": popularity,
                "is_spicy": is_spicy
            })

            item_id += 1

items_df = pd.DataFrame(items)
item_price_lookup = items_df.set_index("item_id")["price"].to_dict()


def choose_item_with_spend_bias(item_ids, spend_bias):
    if len(item_ids) == 1:
        return item_ids[0]

    prices = np.array([item_price_lookup[item_id] for item_id in item_ids], dtype=float)
    price_min = prices.min()
    price_max = prices.max()

    if price_max == price_min:
        return random.choice(item_ids)

    price_norm = (prices - price_min) / (price_max - price_min)
    weights = 1 + 2 * spend_bias * (price_norm - 0.5)
    weights = np.clip(weights, 0.01, None)
    weights = weights / weights.sum()

    return int(np.random.choice(item_ids, p=weights))

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

# -----------------------------
# 3️⃣ Simulate Orders
# -----------------------------
ranking_rows = []
cart_id_counter = 0

for order_id in range(NUM_ORDERS):
    # -----------------------------
    # Time Context
    # -----------------------------
    hour_of_day = int(np.random.choice(HOURS, p=HOUR_PROBS))
    day_of_week = np.random.randint(0, 7)
    is_weekend = 1 if day_of_week in [5, 6] else 0

    if 6 <= hour_of_day < 11:
        meal_type = "Breakfast"
    elif 11 <= hour_of_day < 16:
        meal_type = "Lunch"
    elif 16 <= hour_of_day < 22:
        meal_type = "Dinner"
    else:
        meal_type = "LateNight"

    user_id = random.randint(0, NUM_USERS - 1)
    user_type = users_df.loc[
        users_df["user_id"] == user_id, "user_type"
    ].values[0]

    cart_id = cart_id_counter
    cart_id_counter += 1

    user_pref_cuisine = user_memory[user_id]["preferred_cuisine"]
    if random.random() < 0.7:
        preferred_restaurants = restaurants_df[
            restaurants_df["cuisine"] == user_pref_cuisine
        ]
        if not preferred_restaurants.empty:
            restaurant_row = preferred_restaurants.sample(1).iloc[0]
        else:
            restaurant_row = restaurants_df.sample(1).iloc[0]
    else:
        restaurant_row = restaurants_df.sample(1).iloc[0]
    restaurant_id = restaurant_row["restaurant_id"]
    order_city = restaurant_row["city"]
    order_cuisine = restaurant_row["cuisine"]

    restaurant_mains = items_df[
        (items_df["restaurant_id"] == restaurant_id) &
        (items_df["category"] == "Main")
    ]["item_id"].tolist()

    if not restaurant_mains:
        continue

    main_item = choose_item_with_spend_bias(
        restaurant_mains, user_memory[user_id]["avg_spend_bias"]
    )
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

        memory = user_memory[user_id]

        if next_category == "Dessert":
            transition_prob += memory["dessert_affinity"]

        if next_category == "Drink":
            transition_prob += memory["drink_affinity"]

        # -----------------------------
        # Time-Based Adjustments
        # -----------------------------
        if meal_type == "Dinner" and next_category == "Dessert":
            transition_prob += 0.10

        if meal_type == "Lunch" and next_category == "Drink":
            transition_prob += 0.08

        if meal_type == "Breakfast" and next_category == "Dessert":
            transition_prob -= 0.10

        if meal_type == "LateNight" and next_category == "Main":
            transition_prob -= 0.10

        if meal_type == "Dinner" and user_type == "Premium":
            transition_prob += 0.05

        if is_weekend and next_category in ["Dessert", "Drink"]:
            transition_prob += 0.05

        if is_weekend and user_type == "Premium":
            transition_prob += 0.05

        transition_prob = max(0, min(1, transition_prob))

        continue_prob = 0.5 + 0.5 * transition_prob
        if random.random() > continue_prob:
            break

        candidate_pool = items_df[
            (items_df["restaurant_id"] == restaurant_id) &
            (items_df["category"] == next_category)
        ]["item_id"].tolist()

        if not candidate_pool:
            break

        next_item = choose_item_with_spend_bias(
            candidate_pool, user_memory[user_id]["avg_spend_bias"]
        )

        # -----------------------------
        # CART FEATURES
        # -----------------------------
        cart_prices = items_df.loc[
            items_df["item_id"].isin(cart)
        ]["price"]

        cart_value = max(
            1,
            int(cart_prices.sum() * (1 + user_memory[user_id]["avg_spend_bias"]))
        )

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
        effective_candidate_price = max(
            1,
            int(next_item_row["price"] * (1 + user_memory[user_id]["avg_spend_bias"]))
        )

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
            "restaurant_id": restaurant_id,
            "city": order_city,
            "cuisine": order_cuisine,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "meal_type": meal_type,
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
            "candidate_price": effective_candidate_price,
            "candidate_popularity": next_item_row["popularity"],
            "transition_probability": transition_prob,
            "embedding_similarity": embedding_similarity,
            "label": 1
        })

        # -----------------------------
        # NEGATIVE SAMPLING
        # -----------------------------
        restaurant_item_pool = items_df[
            items_df["restaurant_id"] == restaurant_id
        ]["item_id"].tolist()

        valid_negative_pool = [
            item for item in restaurant_item_pool
            if item not in cart and item != next_item
        ]

        if not valid_negative_pool:
            break

        negatives = random.sample(
            valid_negative_pool,
            min(NEGATIVES_PER_STEP, len(valid_negative_pool))
        )

        for neg_item in negatives:

            neg_row = items_df.loc[
                items_df["item_id"] == neg_item
            ].iloc[0]
            effective_neg_price = max(
                1,
                int(neg_row["price"] * (1 + user_memory[user_id]["avg_spend_bias"]))
            )

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
                "restaurant_id": restaurant_id,
                "city": order_city,
                "cuisine": order_cuisine,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "meal_type": meal_type,
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
                "candidate_price": effective_neg_price,
                "candidate_popularity": neg_row["popularity"],
                "transition_probability": neg_transition_prob,
                "embedding_similarity": embedding_similarity_neg,
                "label": 0
            })

        cart.append(next_item)
        step_number += 1

        if meal_type == "Breakfast" and len(cart) >= 2:
            break
        elif meal_type == "Lunch" and len(cart) >= 3:
            break
        elif meal_type == "Dinner" and len(cart) >= 4:
            break
        elif meal_type == "LateNight" and len(cart) >= 2:
            break

    # -----------------------------
    # Update User Memory
    # -----------------------------
    final_cart_categories = items_df[
        items_df["item_id"].isin(cart)
    ]["category"]
    final_cart_prices = items_df.loc[
        items_df["item_id"].isin(cart)
    ]["price"]

    dessert_ratio = (final_cart_categories == "Dessert").mean()
    drink_ratio = (final_cart_categories == "Drink").mean()
    final_cart_value = max(
        1,
        int(final_cart_prices.sum() * (1 + user_memory[user_id]["avg_spend_bias"]))
    )

    memory = user_memory[user_id]
    memory["dessert_affinity"] = 0.7 * memory["dessert_affinity"] + 0.3 * dessert_ratio
    memory["drink_affinity"] = 0.7 * memory["drink_affinity"] + 0.3 * drink_ratio
    observed_spend_signal = np.clip((final_cart_value - 250) / 2500, -0.1, 0.15)
    memory["avg_spend_bias"] = float(
        np.clip(0.8 * memory["avg_spend_bias"] + 0.2 * observed_spend_signal, -0.1, 0.2)
    )

    if order_cuisine != memory["preferred_cuisine"] and random.random() < 0.10:
        memory["preferred_cuisine"] = order_cuisine

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
