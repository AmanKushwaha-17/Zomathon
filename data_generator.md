# CSAO Data Generator — Complete Documentation

## Overview

`data_generator.py` synthesizes a sequential ranking dataset for the **Cart Super Add-On (CSAO)** recommendation engine. It simulates realistic food ordering behavior end-to-end: users, restaurants, menus, order sequences, and contextual signals.

**Core Learning Task**: Given a user's current cart state and context, rank the next most likely item to be added.

### Outputs

| File | Purpose |
|------|---------|
| `csao_ranking_data_v3_personalized.csv` | Training/test dataset with positive + negative ranking rows |
| `csao_feature_artifacts.npz` | Pre-computed item embeddings (used at inference) |
| `csao_transitions.json` | Category transition probabilities (used at inference) |

---

## Step 1: Configuration Constants

### Scale Parameters

| Constant | Value | Decision Rationale |
|----------|-------|--------------------|
| `NUM_USERS` | 2,000 | Large enough for behavioral diversity, small enough for fast generation |
| `NUM_ORDERS` | 6,000 | ~3 orders per user on average, simulating repeat behavior |
| `NEGATIVES_PER_STEP` | 20 | 1:20 positive-to-negative ratio per step — enough for LambdaRank to learn meaningful rankings without overwhelming the dataset |
| `NUM_RESTAURANTS` | 300 | Gives ~7 users per restaurant on average, creating realistic sparsity |

### Category Distribution

```
Main: 40%  |  Side: 25%  |  Dessert: 15%  |  Drink: 20%
```

**Decision**: Main is weighted highest because every order starts with a Main. Side and Drink are common add-ons. Dessert is lowest because it's the least frequent real-world add-on.

### Geographic & Cuisine Distribution

- **Cities**: Delhi (40%), Mumbai (35%), Bangalore (25%) — reflects Indian metro ordering volume
- **Cuisines**: North Indian (30%), South Indian (20%), Chinese (20%), Italian (15%), Fast Food (15%)
- **User Types**: Budget (50%), Regular (30%), Premium (20%) — heavy Budget skew mirrors real-world patterns

### Category Transition Rules

```python
TRANSITIONS = {
    "Main":    {"Side": 0.5, "Drink": 0.3, "Main": 0.15, "Dessert": 0.2},
    "Side":    {"Dessert": 0.3, "Drink": 0.3, "Side": 0.1},
    "Dessert": {"Drink": 0.4}
}
```

**Decision**: These encode realistic ordering behavior:
- After a **Main**, users most likely add a **Side** (50%), then a **Drink** (30%). Adding another Main is rare (15%).
- After a **Side**, users move to **Dessert** or **Drink** equally (30% each). Rarely add another Side (10%).
- After **Dessert**, users only add a **Drink** (40%). This is usually the terminal stage.
- **Drink has no outgoing transitions** — it's always the terminal category (no one orders something after a drink).

> **Why probabilities don't sum to 1.0**: These are *relative weights*, normalized at runtime. They represent the *likelihood of continuing* in each direction, not a strict probability distribution.

### Time-of-Day Traffic Distribution

```python
HOUR_DISTRIBUTION = {
    0: 0.01, 1: 0.005, ..., 12: 0.10, 13: 0.12, ..., 20: 0.12, ..., 23: 0.03
}
```

**Decision**: Non-uniform traffic with peaks at lunch (12–13) and dinner (19–20). Late night and early morning have minimal traffic. This mirrors real Zomato/Swiggy order patterns.

---

## Step 2: Item Name Generation

```python
def get_item_name(category, cuisine):
```

**Decision**: Names are **cuisine-aware** — a Chinese restaurant gets "Hakka Noodles" while a North Indian one gets "Chicken Biryani". This ensures the generated data looks realistic and contextually coherent.

- Each cuisine has **6 Main names**, **4 names each** for Side/Dessert/Drink
- 35% chance of adding a prefix adjective ("Classic", "Special", "Signature", "Chef's") for variety
- Names are randomly selected per item, so the same restaurant can have "Hakka Noodles" and "Signature Hakka Noodles" as separate items

---

## Step 3: User Generation

```python
for user_id in range(NUM_USERS):
    user_type = np.random.choice(USER_TYPES, p=USER_TYPE_PROBS)
```

Simple user table with `user_id` and `user_type`. The type determines downstream behavior (Premium users order more desserts, Budget users skip them).

### User Behavioral Memory

Each user gets a **persistent memory state** that evolves across orders:

| Field | Initial Range | Purpose |
|-------|---------------|---------|
| `preferred_cuisine` | Random cuisine | 70% of the time, user picks a restaurant matching this |
| `dessert_affinity` | 0.0 – 0.3 | Additive boost to dessert transition probability |
| `drink_affinity` | 0.0 – 0.3 | Additive boost to drink transition probability |
| `avg_spend_bias` | -0.05 – 0.1 | Multiplier on item prices — simulates price sensitivity |

**Decision**: Memory creates **non-i.i.d. behavior** across orders. A user who orders dessert frequently will develop a higher `dessert_affinity`, making future dessert orders more likely. This creates realistic user-level patterns for the model to learn from.

---

## Step 4: Restaurant Generation

Each of the 300 restaurants gets:

| Field | Distribution | Decision |
|-------|-------------|----------|
| `city` | Weighted (Delhi 40%) | Concentrated in top metros |
| `cuisine` | Weighted (North Indian 30%) | Reflects real cuisine popularity |
| `price_range` | Budget 50%, Mid 35%, Premium 15% | Heavy budget segment |
| `rating` | Uniform 3.5–4.8 | No low-rated restaurants (survival bias) |
| `is_chain` | 30% chains | Chains tend to have standardized menus |

---

## Step 5: Item Generation

**20 items per restaurant** (5 per category: Main, Side, Dessert, Drink).

Total items: 300 restaurants x 20 items = **6,000 unique item_ids** (0–5999).

### Price Logic

| Category | Base Range | Decision |
|----------|-----------|----------|
| Main | Rs.180–350 | Highest price — primary dish |
| Side | Rs.40–120 | Low-cost add-on |
| Dessert | Rs.60–150 | Mid-range |
| Drink | Rs.30–100 | Cheapest category |

**Price Range Multiplier**:
- Premium restaurants: `base_price * 1.4`
- Budget restaurants: `base_price * 0.8`
- Mid restaurants: `base_price * 1.0`

**Decision**: This creates realistic price distributions where the same dish costs more at a premium restaurant.

### Popularity

```python
popularity = np.random.zipf(2)
```

**Decision**: Zipf distribution creates a **long-tail** — a few items are very popular (popularity > 10), most items have low popularity (1–2). This mirrors real menu item ordering frequency.

### Spend-Bias Item Selection

```python
def choose_item_with_spend_bias(item_ids, spend_bias):
```

**Decision**: When selecting which specific item a user picks from a category, we weight by their `avg_spend_bias`. A positive bias favors expensive items, negative favors cheap ones. This creates user-level price preference patterns.

---

## Step 6: Item Embeddings (Simulated)

```python
EMBEDDING_DIM = 32
category_base_vectors = {
    "Main": np.random.normal(0, 1, 32),
    "Side": np.random.normal(0, 1, 32),
    ...
}
```

Each item embedding = `category_base_vector + Gaussian noise(0, 0.1)`, then L2-normalized.

**Decision**:
- Items in the same category cluster together in embedding space (shared base vector)
- Small noise (stddev 0.1 vs base stddev 1.0) ensures within-category items are similar but not identical
- L2 normalization enables cosine similarity via dot product
- 32 dimensions is sufficient for 4-category separation without being expensive

### Persisted Artifacts

```python
np.savez("csao_feature_artifacts.npz", embedding_item_ids=..., embeddings_matrix=...)
json.dump(TRANSITIONS, "csao_transitions.json")
```

**Decision**: Embeddings and transitions are saved as artifacts so that `inference_demo.py` uses the **exact same representations** as training. Without this, inference would need to regenerate embeddings (non-deterministic) or hard-code transitions (fragile).

---

## Step 7: Order Simulation

For each of the 6,000 orders:

### 7.1 Time Context

```python
hour_of_day = np.random.choice(HOURS, p=HOUR_PROBS)
meal_type = "Breakfast" if 6 <= hour < 11 else "Lunch" if 11 <= hour < 16 else ...
```

**Decision**: `meal_type` is derived from `hour_of_day`, not sampled independently. This ensures temporal consistency (no "Breakfast" at 9 PM).

### 7.2 Restaurant Selection (Cuisine Loyalty)

```python
if random.random() < 0.7:
    # Pick restaurant matching user's preferred_cuisine
else:
    # Random restaurant
```

**Decision**: 70% cuisine loyalty creates a strong signal for the model to learn cuisine-user associations, while 30% exploration prevents overfitting to a single cuisine.

### 7.3 Cart Initialization

Cart always starts with one **Main** item from the selected restaurant, chosen via spend-bias weighting.

**Decision**: Starting with Main is realistic — almost no one starts an order with a side dish or drink.

### 7.4 Sequential Item Addition (Core Loop)

```
while True:
    1. Look at last item's category
    2. Sample next_category from TRANSITIONS[last_category]
    3. Adjust transition_prob based on context
    4. Decide whether to continue adding
    5. Select candidate item
    6. Record positive row + negative samples
    7. Add item to cart
    8. Check meal-type cart size cap
```

### 7.5 Transition Probability Adjustments

The raw transition probability is **adjusted dynamically** based on multiple contextual signals:

| Signal | Adjustment | Rationale |
|--------|-----------|-----------|
| Premium + Dessert | +0.15 | Premium users order desserts more |
| Budget + Dessert | -0.15 | Budget users skip desserts |
| Premium + Main | +0.05 | Premium users order second mains |
| Spicy last item + Drink | +0.10 | Spicy food triggers drink orders |
| User dessert_affinity | +affinity value | Personal dessert preference |
| User drink_affinity | +affinity value | Personal drink preference |
| Dinner + Dessert | +0.10 | Dinner = more likely to add dessert |
| Lunch + Drink | +0.08 | Lunch drinks are common |
| Breakfast + Dessert | -0.10 | No one orders dessert at breakfast |
| LateNight + Main | -0.10 | Late night = lighter ordering |
| Dinner + Premium user | +0.05 | Premium dinner = bigger carts |
| Weekend + Dessert/Drink | +0.05 | Weekend splurging |
| Weekend + Premium | +0.05 | Extra weekend boost for premium |

**Decision**: These 13 adjustment rules encode domain knowledge about food ordering behavior. They create **rich contextual signal** in the data that the LambdaRank model can learn from.

Final probability is clamped to [0, 1], then converted to a continue-or-stop decision:

```python
continue_prob = 0.5 + 0.5 * transition_prob
if random.random() > continue_prob:
    break
```

**Decision**: `0.5 + 0.5 * p` maps transition probability [0, 1] to continue probability [0.5, 1.0]. Even a 0% transition probability has a 50% chance of continuing (via secondary pathways). A 100% transition probability guarantees continuation.

### 7.6 Cart Size Caps

| Meal Type | Max Items | Rationale |
|-----------|-----------|-----------|
| Breakfast | 2 | Light meal |
| Lunch | 3 | Standard meal |
| Dinner | 4 | Largest meal of the day |
| LateNight | 2 | Quick/light order |

**Decision**: Prevents unrealistically large carts. Combined with the probabilistic stopping, most carts will be smaller than the cap.

---

## Step 8: Feature Engineering (Per Step)

For each positive addition, these features are computed:

### Cart-Level Features

| Feature | Computation | Purpose |
|---------|------------|---------|
| `cart_size` | `len(cart)` | How full the cart is |
| `cart_value` | `sum(prices) * (1 + spend_bias)` | Total cart spend (bias-adjusted) |
| `count_main/side/dessert/drink` | Category counts in current cart | What categories are already covered |
| `last_category` | Category of last added item | Most recent transition signal |

### Candidate Features

| Feature | Computation | Purpose |
|---------|------------|---------|
| `candidate_category` | Item's category | What type of item is being considered |
| `candidate_price` | `raw_price * (1 + spend_bias)` | Effective price for this user |
| `candidate_popularity` | Zipf-sampled value | How popular the item is |
| `transition_probability` | From TRANSITIONS dict | Base likelihood of this category following last |
| `embedding_similarity` | `dot(cart_embedding, item_embedding)` | Semantic similarity between cart context and candidate |

**Decision on `candidate_price`**: The effective price (bias-adjusted) is stored instead of raw price. This means the model sees prices from the user's perspective — a budget user sees a Rs.200 item as effectively Rs.190, while a premium user sees it as Rs.220.

> **Note**: This creates multiple rows with different `candidate_price` for the same `item_id` across different users, which was the root cause of the duplicate-items bug in inference.

---

## Step 9: Positive and Negative Row Generation

### Positive Row

One row per actual item addition with `label = 1`.

### Negative Sampling

For each positive step:

```python
valid_negative_pool = [item for item in restaurant_items if item not in cart and item != positive_item]
negatives = random.sample(pool, min(20, len(pool)))
```

**Decision**:
- Negatives come from the **same restaurant** — makes the ranking task realistic (model must learn to rank within a restaurant's menu, not across all restaurants)
- Cart items and the positive item are excluded — prevents trivial negatives
- Sample up to 20 negatives per step — enough diversity for LambdaRank without too much data bloat
- Negative transition probability is looked up from the TRANSITIONS dict for the negative item's category (not the adjusted probability) — keeps negative features simple and unbiased

---

## Step 10: User Memory Update

After each completed order:

```python
memory["dessert_affinity"] = 0.7 * old + 0.3 * observed_dessert_ratio
memory["drink_affinity"]   = 0.7 * old + 0.3 * observed_drink_ratio
memory["avg_spend_bias"]   = clip(0.8 * old + 0.2 * spend_signal, -0.1, 0.2)
```

**Decision**:
- **Exponential moving average** with 70/30 split — old behavior dominates but recent orders have meaningful influence
- Spend signal is derived from final cart value relative to a Rs.250 baseline, clipped to [-0.1, 0.15]
- **Cuisine drift**: 10% chance per order of switching `preferred_cuisine` if user ordered a different one — simulates gradual taste evolution

This ensures that user behavior is **non-stationary** — a user who starts as Budget but consistently orders expensive items will gradually shift their spend bias upward.

---

## Dataset Statistics (Typical Run)

| Metric | Approximate Value |
|--------|-------------------|
| Total ranking rows | ~120,000–140,000 |
| Unique carts | ~5,500–5,800 |
| Positive ratio | ~4.5–5.5% |
| Average steps per cart | ~1.2–1.5 |

---

## Design Decisions Summary

| Decision | Why |
|----------|-----|
| Items scoped to restaurants | Realistic — each restaurant has its own menu |
| Spend-bias-adjusted prices | Captures user price sensitivity at feature level |
| Zipf popularity | Realistic long-tail distribution |
| 13 contextual transition adjustments | Rich signal for model to learn non-trivial patterns |
| Same-restaurant negatives | Harder, more realistic ranking task |
| Exponential moving average memory | Non-stationary user behavior without catastrophic forgetting |
| Persisted embeddings + transitions | Inference uses identical features as training |
| Category transition as terminal graph | Drink is always last, Main always first — mirrors real behavior |

---

## Run

```bash
python data_generator.py
```

### Sanity Checks

```python
# Category distribution by meal type
print(ranking_df.groupby("meal_type")["candidate_category"].value_counts(normalize=True))

# Average cart size by meal type
print(ranking_df.groupby("meal_type")["cart_size"].mean())

# Traffic distribution by hour
print(ranking_df.groupby("hour_of_day").size())

# Check for item_id uniqueness per restaurant
print(items_df.groupby("restaurant_id")["item_id"].nunique())  # Should be 20
```
