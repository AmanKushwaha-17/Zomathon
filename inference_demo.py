import pandas as pd
import numpy as np
import lightgbm as lgb
import random
from collections import defaultdict
import time
from pathlib import Path
import threading
import json

from groq import Groq
import os

explanation_cache = {}
cache_lock = threading.Lock()

def _load_groq_api_key():
    # 1) Prefer already-exported environment variable.
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip().strip('"').strip("'")

    # 2) Fallback: parse local .env so script works without external export step.
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "GROQ_API_KEY":
            value = v.strip().strip('"').strip("'")
            return value or None
    return None

# Initialize Groq client :

api_key = _load_groq_api_key()
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    client = Groq()
else:
    print("Warning: GROQ_API_KEY not set. Groq client will not be initialized.")
    client = None

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("csao_ranking_data_v3_personalized.csv")

items_df = df[[
    "candidate_item",
    "candidate_item_name",
    "candidate_category",
    "candidate_price",
    "candidate_popularity"
]].drop_duplicates()

items_df = items_df.rename(columns={
    "candidate_item": "item_id",
    "candidate_item_name": "item_name",
    "candidate_category": "category",
    "candidate_price": "price",
    "candidate_popularity": "popularity"
})

# -----------------------------
# Load Embeddings + Transition Rules from generator artifacts
# -----------------------------
np.random.seed(42)

EMBEDDING_DIM = 32
emb_artifact_path = Path(__file__).with_name("csao_feature_artifacts.npz")
transitions_path = Path(__file__).with_name("csao_transitions.json")

item_embeddings = {}

if not emb_artifact_path.exists():
    raise FileNotFoundError(f"Missing embedding artifact: {emb_artifact_path}")
if not transitions_path.exists():
    raise FileNotFoundError(f"Missing transitions artifact: {transitions_path}")

emb_artifacts = np.load(emb_artifact_path)
embedding_item_ids = emb_artifacts["embedding_item_ids"]
embeddings_matrix = emb_artifacts["embeddings_matrix"]
item_embeddings = {
    int(item_id): embeddings_matrix[idx]
    for idx, item_id in enumerate(embedding_item_ids)
}

with open(transitions_path, "r", encoding="utf-8") as f:
    TRANSITIONS = json.load(f)

# -----------------------------
# Load Model
# -----------------------------
model = lgb.Booster(model_file="csao_lambdarank_model.txt")

# ----
# Generate Explanation Function
# ----
def generate_tooltip_llm(cart_name, recommended_name):
    prompt = (
        f"You added {cart_name}. "
        f"In one short sentence, explain why {recommended_name} complements it."
    )

    # Safe fallback if Groq client is not configured
    if client is None:
        print("Warning: Groq client not initialized - returning fallback tooltip.")
        return f"{recommended_name} pairs well with {cart_name}."

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except Exception as e:
        print(f"Warning: Groq request failed ({type(e).__name__}) - returning fallback tooltip.")
        return f"{recommended_name} pairs well with {cart_name}."

    # Extract content robustly
    try:
        content = completion.choices[0].message.content.strip()
    except Exception:
        # Best-effort fallback to string representation
        if hasattr(completion, "choices") and completion.choices:
            content = str(completion.choices[0])
        else:
            content = str(completion)
        content = content.strip()

    # Ensure single sentence ending with period
    if "." in content:
        content = content.split(".")[0] + "."
    else:
        content = content + "."

    return content

def async_generate_and_cache(key, cart_name, recommended_name):
    tooltip = generate_tooltip_llm(cart_name, recommended_name)
    with cache_lock:
        explanation_cache[key] = tooltip
    print("\nAI Generated Suggestion (Async):")
    print(tooltip)

# -----------------------------
# # Available Items
# # -----------------------------
# print("\nAvailable Items By Category:\n")
# for cat in items_df["category"].unique():
#     ids = items_df[items_df["category"] == cat]["item_id"].tolist()[:5]
#     print(f"{cat}: {ids}")



# -----------------------------
# Simulate Cart
# -----------------------------

user_type = "Regular"
cart_items = [62]

cart_df = items_df[items_df["item_id"].isin(cart_items)]

cart_size = len(cart_items)
cart_value = cart_df["price"].sum()

count_main = (cart_df["category"] == "Main").sum()
count_side = (cart_df["category"] == "Side").sum()
count_dessert = (cart_df["category"] == "Dessert").sum()
count_drink = (cart_df["category"] == "Drink").sum()

last_category = cart_df.iloc[-1]["category"]


# -----------------------------
# Compute Cart Embedding
# -----------------------------
cart_vectors = [item_embeddings[item] for item in cart_items]
cart_embedding = np.mean(cart_vectors, axis=0)
cart_embedding = cart_embedding / np.linalg.norm(cart_embedding)



# Start total pipeline timing
start_total = time.perf_counter()

# -----------------------------
# Stage 1: Candidate Generation (Balanced)
# Primary + Secondary Pool
# -----------------------------
if last_category in TRANSITIONS:
    primary_categories = list(TRANSITIONS[last_category].keys())
else:
    primary_categories = items_df["category"].unique()


# Primary pool (transition-based)
primary_candidates = items_df[
    (items_df["category"].isin(primary_categories)) &
    (~items_df["item_id"].isin(cart_items))
].copy()

# Secondary pool (exploration - top popular items from other categories)
secondary_candidates = items_df[
    (~items_df["category"].isin(primary_categories)) &
    (~items_df["item_id"].isin(cart_items))
].copy()



# Take top popular from secondary pool (exploration quota)
secondary_candidates = secondary_candidates.sort_values(
    "popularity", ascending=False
).head(10)

# Combine pools
candidates = pd.concat([primary_candidates, secondary_candidates])

# Compute transition probability
def get_transition_prob(last_cat, candidate_cat):
    if last_cat in TRANSITIONS:
        return TRANSITIONS[last_cat].get(candidate_cat, 0)
    return 0

candidates["transition_probability"] = candidates["category"].apply(
    lambda x: get_transition_prob(last_category, x)
)

# Compute embedding similarity
candidates["embedding_similarity"] = candidates["item_id"].apply(
    lambda x: np.dot(cart_embedding, item_embeddings[x])
)


# -----------------------------
# Add Context Features
# -----------------------------
candidates["user_type"] = user_type
candidates["cart_size"] = cart_size
candidates["cart_value"] = cart_value
candidates["count_main"] = count_main
candidates["count_side"] = count_side
candidates["count_dessert"] = count_dessert
candidates["count_drink"] = count_drink
candidates["last_category"] = last_category
candidates["candidate_category"] = candidates["category"]
candidates["candidate_price"] = candidates["price"]
candidates["candidate_popularity"] = candidates["popularity"]

feature_cols = [
    "user_type",
    "cart_size",
    "cart_value",
    "count_main",
    "count_side",
    "count_dessert",
    "count_drink",
    "last_category",
    "candidate_category",
    "candidate_price",
    "candidate_popularity",
    "transition_probability",
    "embedding_similarity"
]

X_infer = candidates[feature_cols].copy()

for col in ["user_type", "last_category", "candidate_category"]:
    X_infer[col] = X_infer[col].astype("category")

# Measure Ranking Time
start_ranking = time.perf_counter()
scores = model.predict(X_infer)
end_ranking = time.perf_counter()
ranking_time = (end_ranking - start_ranking) * 1000

candidates["score"] = scores

# Stage 2: Diversity (soft category limit)
# Ensure variety while respecting ranking
candidates_sorted = candidates.sort_values("score", ascending=False)

top8 = []
category_limit = 3
category_count = {}

for _, row in candidates_sorted.iterrows():
    cat = row["category"]

    if category_count.get(cat, 0) < category_limit:
        top8.append(row)
        category_count[cat] = category_count.get(cat, 0) + 1

    if len(top8) == 8:
        break

top8 = pd.DataFrame(top8)

print("\nTop 8 Recommendations:\n")
print(top8[["item_id", "item_name", "category", "price", "score"]])
print("\nUser Type:", user_type)
print("Cart Items:", cart_items)
print("Last Category:", last_category)

# Generate AI explanation for top recommendation (cache-first + async fallback)
cart_item_row = items_df[items_df["item_id"] == cart_items[0]].iloc[0]
cart_name = cart_item_row["item_name"]
recommended_name = top8.iloc[0]["item_name"]
key = (last_category, top8.iloc[0]["category"])

with cache_lock:
    cached_tooltip = explanation_cache.get(key)

if cached_tooltip is not None:
    print("\nAI Generated Suggestion (Cached):")
    start_llm = time.perf_counter()
    print(cached_tooltip)
    end_llm = time.perf_counter()
    llm_time = (end_llm - start_llm) * 1000
else:
    print("\nGenerating explanation asynchronously...")
    thread = threading.Thread(
        target=async_generate_and_cache,
        args=(key, cart_name, recommended_name)
    )
    thread.start()
    llm_time = 0.0

# Calculate total time
end_total = time.perf_counter()
total_time = (end_total - start_total) * 1000

# Print timing summary
print("\nTiming Breakdown (ms):")
print(f"Ranking Time: {ranking_time:.2f} ms")
print(f"LLM Time: {llm_time:.2f} ms")
print(f"Total Time: {total_time:.2f} ms")


