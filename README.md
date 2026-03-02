# CSAO — Cart Super Add-On Recommendation Engine

A real-time recommendation engine that suggests the **next best item** to add to a food delivery cart. Built with **LambdaRank (LightGBM)** for ranking and **Groq LLM** for natural language explanations.

---

## Architecture

```
data_generator.py          Generate synthetic training data
        |
        v
 csao_ranking_data_v3_personalized.csv   +   csao_feature_artifacts.npz   +   csao_transitions.json
        |
        v
train_ranker.py            Train LambdaRank model (NDCG@8)
        |
        v
 csao_lambdarank_model.txt
        |
        v
test_ranker.py             Evaluate model on held-out carts
        |
        v
inference_demo.py          Real-time inference with LLM tooltips
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `data_generator.py` | Synthesizes users, restaurants, items, orders with contextual signals |
| `data_generator.md` | Detailed documentation of data generation logic and decisions |
| `train_ranker.py` | Trains LambdaRank model with cart-based train/test split |
| `test_ranker.py` | Production-style evaluation with segment and meal-type breakdowns |
| `inference_demo.py` | End-to-end inference: candidate generation, ranking, diversity, LLM explanation |
| `.env` | Stores `GROQ_API_KEY` for LLM tooltip generation |

### Generated Artifacts

| File | Created By | Used By |
|------|-----------|---------|
| `csao_ranking_data_v3_personalized.csv` | `data_generator.py` | `train_ranker.py`, `test_ranker.py`, `inference_demo.py` |
| `csao_feature_artifacts.npz` | `data_generator.py` | `inference_demo.py` |
| `csao_transitions.json` | `data_generator.py` | `inference_demo.py` |
| `csao_lambdarank_model.txt` | `train_ranker.py` | `test_ranker.py`, `inference_demo.py` |

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
pip install pandas numpy lightgbm scikit-learn groq
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key from [console.groq.com](https://console.groq.com).

> The LLM tooltip is optional. If no API key is set, a fallback tooltip is generated.

---

## Complete Workflow

### Step 1: Generate Training Data

```bash
python data_generator.py
```

**What it does:**
- Simulates 2,000 users, 300 restaurants, 6,000 items, and 6,000 orders
- Generates positive (added to cart) and negative (not added) ranking pairs
- Applies contextual adjustments: user type, meal time, weekend, cuisine loyalty, spicy-to-drink boost
- Exports training CSV, item embeddings, and transition rules

**Output:**
```
Total ranking rows: ~130,000
Unique carts: ~4,200
Positive ratio: ~4.8%
```

> See `data_generator.md` for detailed documentation of all generation logic and design decisions.

---

### Step 2: Train the Model

```bash
python train_ranker.py
```

**What it does:**
- Loads the ranking dataset
- Splits by cart_id (80% train, 20% test) — no data leakage across carts
- Trains a LambdaRank model (LightGBM) optimizing NDCG@8
- Evaluates against a transition-probability baseline
- Prints feature importance

**Expected Output:**
```
Training complete.

Model Mean NDCG@8: ~0.92
Baseline Mean NDCG@8: ~0.80
Absolute Lift: ~0.12

--- Feature Importance ---
transition_probability    highest
embedding_similarity      high
candidate_category        high
...
```

**Model Config:**
- 200 estimators, learning rate 0.05, 63 leaves
- Categorical features handled natively (no one-hot encoding)

---

### Step 3: Evaluate the Model

```bash
python test_ranker.py
```

**What it does:**
- Loads the trained model and runs on the 20% test split
- Reports overall NDCG@8 and compares to baseline
- Breaks down performance by user segment (Budget/Regular/Premium)
- Breaks down performance by meal type (Breakfast/Lunch/Dinner/LateNight)

**Expected Output:**
```
Production Test NDCG@8: ~0.92
Baseline NDCG@8: ~0.80
Absolute Lift: ~0.12

--- Segment-Level Production NDCG ---
Budget:  ~0.91
Regular: ~0.93
Premium: ~0.93

--- Meal-Type Performance ---
Breakfast: ~0.91
Dinner:    ~0.92
LateNight: ~0.92
Lunch:     ~0.92
```

---

### Step 4: Run Inference

```bash
python inference_demo.py
```

**What it does:**
1. Loads model, embeddings, transition rules, and item catalog
2. Simulates a cart context (restaurant, user type, meal time)
3. Generates candidates scoped to the restaurant
4. Engineers features (transition probability, embedding similarity, cart state)
5. Ranks candidates with the trained model
6. Applies diversity layer (round-robin across categories, max 3 per category)
7. Normalizes scores to 0–100 relevance scale
8. Generates an async LLM tooltip via Groq

**Expected Output:**
```
[Cart] Current Cart:

  - Chicken Wrap (Main) -- Rs.275

  Cart Total: Rs.275

[Top 8] Recommendations:

 item_id             item_name category  price  relevance
     248 Special Dimsum Basket     Side     93      100.0
     247          Kimchi Salad     Side     90       98.1
     241   Spring Roll Platter     Main    218       71.7
     244    Hot Garlic Noodles     Main    181       71.3
     256   Chef's Lemon Cooler    Drink     68       15.0
     255              Iced Tea    Drink     70       15.0
     252  Special Sesame Balls  Dessert     86        0.2
     253          Sesame Balls  Dessert     80        0.0

Context:
User Type: Regular
Restaurant: 12
City: Mumbai
Cuisine: Chinese
Meal Type: Dinner

AI Generated Suggestion (Async):
The Special Dimsum Basket complements the Chicken Wrap by providing a refreshing contrast.

Timing Breakdown (ms):
Ranking Time: ~6 ms
Total Time: ~35 ms
```

### Configuring Inference

Edit these variables in `inference_demo.py` to test different scenarios:

```python
user_type = "Regular"        # Budget | Regular | Premium
restaurant_id = 12           # Any valid restaurant ID (0-299)
cart_items = [62]             # List of item_ids already in cart
hour_of_day = 20             # 0-23
meal_type = "Dinner"         # Breakfast | Lunch | Dinner | LateNight
```

> `city` and `cuisine` are auto-derived from the restaurant — no need to set them manually.

---

## Inference Pipeline Detail

```
Cart Items  -->  Compute Cart Embedding (mean of item embeddings)
                     |
                     v
Restaurant ID  -->  Candidate Generation
                     |  - Filter items by restaurant
                     |  - Primary pool: categories from transition rules
                     |  - Secondary pool: top-10 by popularity from remaining categories
                     |  - Deduplicate by item_id
                     v
                Feature Engineering
                     |  - transition_probability (category transition likelihood)
                     |  - embedding_similarity (cosine similarity with cart)
                     |  - contextual features (user, time, cart state)
                     v
                LambdaRank Model  -->  Raw Scores
                     |
                     v
                Diversity Layer (Round-Robin)
                     |  - Group top candidates by category (max 3 each)
                     |  - Interleave across categories
                     |  - Re-sort by score descending
                     v
                Score Normalization (0-100)
                     |
                     v
                LLM Tooltip (Async, Groq Llama 3.1 8B)
```

---

## Features Used by the Model

| Feature | Type | Description |
|---------|------|-------------|
| `user_type` | Categorical | Budget / Regular / Premium |
| `restaurant_id` | Categorical | Restaurant identifier |
| `city` | Categorical | Delhi / Mumbai / Bangalore |
| `cuisine` | Categorical | North Indian / Chinese / Italian / South Indian / Fast Food |
| `hour_of_day` | Numeric | 0–23 |
| `day_of_week` | Numeric | 0–6 |
| `is_weekend` | Binary | 0 or 1 |
| `meal_type` | Categorical | Breakfast / Lunch / Dinner / LateNight |
| `cart_size` | Numeric | Number of items already in cart |
| `cart_value` | Numeric | Total cart value (spend-adjusted) |
| `count_main` | Numeric | Main items in cart |
| `count_side` | Numeric | Side items in cart |
| `count_dessert` | Numeric | Dessert items in cart |
| `count_drink` | Numeric | Drink items in cart |
| `last_category` | Categorical | Category of last added item |
| `candidate_category` | Categorical | Category of candidate item |
| `candidate_price` | Numeric | Price of candidate |
| `candidate_popularity` | Numeric | Zipf-distributed popularity score |
| `transition_probability` | Numeric | P(candidate_category \| last_category) |
| `embedding_similarity` | Numeric | Cosine similarity between cart and candidate |

---

## Testing

### Quick Sanity Check

Run inference and verify output:

```bash
python inference_demo.py
```

Verify:
- All 8 items have unique `item_id` values
- Relevance scores are in descending order (100 → 0)
- Context (City, Cuisine) matches the restaurant
- All 4 categories are represented (Side, Main, Drink, Dessert)
- Cart items do not appear in recommendations

### Model Evaluation

```bash
python test_ranker.py
```

Verify:
- Model NDCG@8 > 0.85 (should be ~0.92)
- Model NDCG@8 > Baseline NDCG@8 (positive lift)
- All user segments have NDCG > 0.85
- All meal types have NDCG > 0.85

### Data Integrity Checks

```python
import pandas as pd

df = pd.read_csv("csao_ranking_data_v3_personalized.csv")

# No missing values
assert df.isnull().sum().sum() == 0

# Labels are only 0 or 1
assert set(df["label"].unique()) == {0, 1}

# Each (cart_id, step) group has exactly 1 positive
assert (df.groupby(["cart_id", "step"])["label"].sum() == 1).all()

# Positive ratio is reasonable
assert 0.02 < df["label"].mean() < 0.10

# Restaurant has consistent city/cuisine
for rid in df["restaurant_id"].unique()[:20]:
    rdf = df[df["restaurant_id"] == rid]
    assert rdf["city"].nunique() == 1
    assert rdf["cuisine"].nunique() == 1

print("All checks passed.")
```

### Inference Stress Test

Test across multiple restaurants to verify no crashes or duplicate items:

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
import json

df = pd.read_csv("csao_ranking_data_v3_personalized.csv")
model = lgb.Booster(model_file="csao_lambdarank_model.txt")
artifacts = np.load("csao_feature_artifacts.npz")

items_df = df[["candidate_item", "candidate_item_name", "candidate_category",
               "candidate_price", "candidate_popularity"
]].drop_duplicates(subset=["candidate_item"]).rename(columns={
    "candidate_item": "item_id", "candidate_item_name": "item_name",
    "candidate_category": "category", "candidate_price": "price",
    "candidate_popularity": "popularity"
})

# Test 50 random restaurants
for rest_id in np.random.choice(df["restaurant_id"].unique(), 50, replace=False):
    rest_items = items_df[items_df["item_id"].isin(
        df[df["restaurant_id"] == rest_id]["candidate_item"].unique()
    )]
    mains = rest_items[rest_items["category"] == "Main"]
    assert not mains.empty, f"Restaurant {rest_id} has no Main items"
    candidates = rest_items[~rest_items["item_id"].isin([mains.iloc[0]["item_id"]])]
    candidates = candidates.drop_duplicates(subset=["item_id"])
    assert candidates["item_id"].is_unique, f"Restaurant {rest_id} has duplicate candidates"

print("Stress test passed across 50 restaurants.")
```

### Relevance Validation

Key behaviors to verify manually:

| Scenario | Expected Top-1 Category |
|----------|------------------------|
| Cart = [Main] | Side |
| Cart = [Main, Side] | Dessert or Drink |
| Cart = [Main, Side, Dessert] | Drink |
| Premium user | Higher dessert scores than Budget |
| Dinner time | Higher dessert scores than Breakfast |
| Lunch time | Higher drink scores than Breakfast |

---

## Performance

| Metric | Value |
|--------|-------|
| Ranking latency | ~6 ms |
| Total inference (including I/O) | ~35 ms |
| LLM tooltip (async, non-blocking) | ~500–1500 ms |
| Model size | ~1.4 MB |
| Training data | ~15 MB |

---

## Known Limitations

1. **Synthetic data**: Training data is simulated, not from real orders. Model behavior reflects encoded rules, not true user patterns.
2. **Item names may repeat**: `data_generator.py` randomly picks from a small name pool (~4 per category per cuisine), so two different `item_id`s can share a name within the same restaurant.
3. **No real-time personalization**: User type is set at inference time but user memory (dessert/drink affinity, spend bias) is only used during data generation — not available at inference.
4. **Fixed restaurant menu**: All restaurants have exactly 20 items (5 per category). Real menus vary in size.
