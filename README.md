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
| `app.py` | Streamlit interactive frontend for real-time demo and testing |
| `.env` | Stores `GROQ_API_KEY` for LLM tooltip generation |



## Setup

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
pip install pandas numpy lightgbm scikit-learn groq streamlit
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

### Step 5: Run the Streamlit Frontend

```bash
streamlit run app.py
```

**What it does:**
1. Interactive sidebar to select restaurant, user type, time, and cart items
2. Real-time inference with 4x2 recommendation grid
3. Cart summary with category badges and total
4. AI-powered suggestion (Groq Llama 3.1 8B) displayed below cart
5. Context metrics bar (city, cuisine, meal type, latency)

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


