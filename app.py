"""
CSAO Recommendation Engine - Streamlit Frontend
================================================
Interactive demo for Cart Super Add-On recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import time
from pathlib import Path
import threading

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="CSAO Recommendation Engine",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #E23744 0%, #FF6B6B 50%, #FF8E53 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(226, 55, 68, 0.3);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card .metric-value {
        color: #FF6B6B;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-card .metric-label {
        color: rgba(255,255,255,0.7);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .cart-item {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 4px solid #E23744;
    }
    .cart-item .item-name {
        color: white;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .cart-item .item-meta {
        color: rgba(255,255,255,0.6);
        font-size: 0.8rem;
    }
    .cart-item .item-price {
        color: #4CAF50;
        font-weight: 600;
        font-size: 1rem;
    }

    .rec-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        transition: transform 0.2s;
    }
    .rec-card:hover {
        transform: translateX(4px);
    }
    .rec-card .rec-rank {
        color: rgba(255,255,255,0.4);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .rec-card .rec-name {
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 0.2rem 0;
    }
    .rec-card .rec-meta {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
    }

    .relevance-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(255,255,255,0.1);
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .relevance-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    .tooltip-card {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .tooltip-card .tooltip-label {
        color: #BB86FC;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .tooltip-card .tooltip-text {
        color: white;
        font-size: 1rem;
        font-style: italic;
        margin-top: 0.5rem;
        line-height: 1.6;
    }

    .timing-badge {
        display: inline-block;
        background: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }

    .category-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown label {
        color: rgba(255,255,255,0.85);
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Assets (Cached) ─────────────────────────────────
@st.cache_resource
def load_assets():
    df = pd.read_csv("csao_ranking_data_v3_personalized.csv")
    artifacts = np.load("csao_feature_artifacts.npz")
    with open("csao_transitions.json", "r", encoding="utf-8") as f:
        transitions = json.load(f)
    model = lgb.Booster(model_file="csao_lambdarank_model.txt")

    embedding_ids = artifacts["embedding_item_ids"]
    embeddings_matrix = artifacts["embeddings_matrix"]
    item_emb = {int(eid): embeddings_matrix[i] for i, eid in enumerate(embedding_ids)}

    items_df = df[[
        "candidate_item", "candidate_item_name", "candidate_category",
        "candidate_price", "candidate_popularity"
    ]].drop_duplicates(subset=["candidate_item"]).rename(columns={
        "candidate_item": "item_id",
        "candidate_item_name": "item_name",
        "candidate_category": "category",
        "candidate_price": "price",
        "candidate_popularity": "popularity"
    })

    return df, items_df, item_emb, transitions, model


df, items_df, item_emb, transitions, model = load_assets()

FEATURE_COLS = [
    "user_type", "restaurant_id", "city", "cuisine",
    "hour_of_day", "day_of_week", "is_weekend", "meal_type",
    "cart_size", "cart_value", "count_main", "count_side",
    "count_dessert", "count_drink", "last_category",
    "candidate_category", "candidate_price", "candidate_popularity",
    "transition_probability", "embedding_similarity"
]
CAT_COLS = ["user_type", "restaurant_id", "city", "cuisine",
            "meal_type", "last_category", "candidate_category"]

CATEGORY_COLORS = {
    "Side": "#FF6B6B",
    "Main": "#4ECDC4",
    "Drink": "#45B7D1",
    "Dessert": "#DDA0DD"
}


# ─── Inference Function ───────────────────────────────────
def run_inference(cart_item_ids, restaurant_id, user_type, hour, day, meal_type):
    start_total = time.perf_counter()

    rest_rows = df[df["restaurant_id"] == restaurant_id]
    city = rest_rows["city"].iloc[0]
    cuisine = rest_rows["cuisine"].iloc[0]

    rest_item_ids = rest_rows["candidate_item"].unique()
    rest_items = items_df[items_df["item_id"].isin(rest_item_ids)]

    cart_df = rest_items[rest_items["item_id"].isin(cart_item_ids)]
    last_category = cart_df.iloc[-1]["category"]

    candidates = rest_items[~rest_items["item_id"].isin(cart_item_ids)].copy()
    candidates = candidates.drop_duplicates(subset=["item_id"])

    if candidates.empty:
        return pd.DataFrame(), 0, 0, city, cuisine

    # Cart embedding
    cart_vectors = [item_emb[i] for i in cart_item_ids if i in item_emb]
    cart_embedding = np.mean(cart_vectors, axis=0)
    cart_embedding = cart_embedding / np.linalg.norm(cart_embedding)

    is_weekend = 1 if day in [5, 6] else 0

    candidates["transition_probability"] = candidates["category"].apply(
        lambda x: transitions.get(last_category, {}).get(x, 0)
    )
    candidates["embedding_similarity"] = candidates["item_id"].apply(
        lambda x: np.dot(cart_embedding, item_emb[x]) if x in item_emb else 0
    )
    candidates["user_type"] = user_type
    candidates["restaurant_id"] = restaurant_id
    candidates["city"] = city
    candidates["cuisine"] = cuisine
    candidates["hour_of_day"] = hour
    candidates["day_of_week"] = day
    candidates["is_weekend"] = is_weekend
    candidates["meal_type"] = meal_type
    candidates["cart_size"] = len(cart_item_ids)
    candidates["cart_value"] = int(cart_df["price"].sum())
    candidates["count_main"] = (cart_df["category"] == "Main").sum()
    candidates["count_side"] = (cart_df["category"] == "Side").sum()
    candidates["count_dessert"] = (cart_df["category"] == "Dessert").sum()
    candidates["count_drink"] = (cart_df["category"] == "Drink").sum()
    candidates["last_category"] = last_category
    candidates["candidate_category"] = candidates["category"]
    candidates["candidate_price"] = candidates["price"]
    candidates["candidate_popularity"] = candidates["popularity"]

    X = candidates[FEATURE_COLS].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")

    start_rank = time.perf_counter()
    scores = model.predict(X)
    rank_time = (time.perf_counter() - start_rank) * 1000

    candidates["score"] = scores

    # Diversity layer
    candidates_sorted = candidates.sort_values("score", ascending=False)
    buckets = {}
    for _, row in candidates_sorted.iterrows():
        cat = row["category"]
        if cat not in buckets:
            buckets[cat] = []
        if len(buckets[cat]) < 3:
            buckets[cat].append(row)

    top8 = []
    cat_order = list(buckets.keys())
    idx = {c: 0 for c in cat_order}
    while len(top8) < 8:
        added = False
        for c in cat_order:
            if len(top8) >= 8:
                break
            if idx[c] < len(buckets[c]):
                top8.append(buckets[c][idx[c]])
                idx[c] += 1
                added = True
        if not added:
            break

    result = pd.DataFrame(top8).sort_values("score", ascending=False)

    raw = result["score"].values
    smin, smax = raw.min(), raw.max()
    if smax != smin:
        result["relevance"] = ((raw - smin) / (smax - smin) * 100).round(1)
    else:
        result["relevance"] = 100.0

    total_time = (time.perf_counter() - start_total) * 1000
    return result, rank_time, total_time, city, cuisine


# ─── LLM Tooltip ──────────────────────────────────────────
def get_llm_tooltip(cart_name, rec_name):
    try:
        import os
        from groq import Groq

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            env_path = Path(__file__).with_name(".env")
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "GROQ_API_KEY":
                            key = v.strip().strip('"').strip("'")
            if not key:
                return f"{rec_name} pairs well with {cart_name}."

        client = Groq(api_key=key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"You added {cart_name}. In one short sentence, explain why {rec_name} complements it."
            }]
        )
        content = completion.choices[0].message.content.strip()
        if "." in content:
            content = content.split(".")[0] + "."
        return content
    except Exception:
        return f"{rec_name} pairs well with {cart_name}."


# ─── Build Restaurant/Item Lookups ────────────────────────
restaurant_info = df.groupby("restaurant_id").agg(
    city=("city", "first"),
    cuisine=("cuisine", "first")
).reset_index()


def get_restaurant_items(rest_id):
    rest_item_ids = df[df["restaurant_id"] == rest_id]["candidate_item"].unique()
    return items_df[items_df["item_id"].isin(rest_item_ids)]


# ─── Header ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>CSAO Recommendation Engine</h1>
    <p>Context-Aware Sequential Cart Optimization &mdash; Powered by LambdaRank + LLM</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    # Restaurant selection
    rest_options = restaurant_info.apply(
        lambda r: f"#{int(r['restaurant_id'])} - {r['cuisine']} ({r['city']})", axis=1
    ).tolist()
    selected_rest_idx = st.selectbox("Restaurant", range(len(rest_options)),
                                      format_func=lambda i: rest_options[i])
    restaurant_id = int(restaurant_info.iloc[selected_rest_idx]["restaurant_id"])

    st.markdown("---")

    # User type
    user_type = st.selectbox("User Type", ["Budget", "Regular", "Premium"], index=1)

    # Time
    st.markdown("---")
    hour = st.slider("Hour of Day", 0, 23, 20)
    day = st.selectbox("Day of Week",
                        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                        index=5)
    day_num = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(day)

    if 6 <= hour < 11:
        meal_type = "Breakfast"
    elif 11 <= hour < 16:
        meal_type = "Lunch"
    elif 16 <= hour < 22:
        meal_type = "Dinner"
    else:
        meal_type = "LateNight"

    st.markdown(f"**Meal Type**: {meal_type}")
    st.markdown(f"**Weekend**: {'Yes' if day_num in [5,6] else 'No'}")

    # Cart items
    st.markdown("---")
    st.markdown("### Cart Items")
    rest_menu = get_restaurant_items(restaurant_id)

    cart_options = rest_menu.apply(
        lambda r: f"{r['item_name']} ({r['category']}) - Rs.{r['price']}", axis=1
    ).tolist()
    cart_ids_list = rest_menu["item_id"].tolist()

    # Default: pick first Main
    default_indices = []
    mains = rest_menu[rest_menu["category"] == "Main"]
    if not mains.empty:
        first_main_id = mains.iloc[0]["item_id"]
        if first_main_id in cart_ids_list:
            default_indices = [cart_ids_list.index(first_main_id)]

    selected_cart_indices = st.multiselect(
        "Select items in cart",
        range(len(cart_options)),
        default=default_indices,
        format_func=lambda i: cart_options[i]
    )

    cart_item_ids = [cart_ids_list[i] for i in selected_cart_indices]

    st.markdown("---")
    run_btn = st.button("Get Recommendations", type="primary", use_container_width=True)


# ─── Main Area ────────────────────────────────────────────
if not cart_item_ids:
    st.info("Select at least one item in the cart from the sidebar to get recommendations.")
    st.stop()

if run_btn or "last_result" not in st.session_state:
    with st.spinner("Running inference..."):
        top8, rank_time, total_time, city, cuisine = run_inference(
            cart_item_ids, restaurant_id, user_type, hour, day_num, meal_type
        )
        st.session_state["last_result"] = (top8, rank_time, total_time, city, cuisine)
else:
    top8, rank_time, total_time, city, cuisine = st.session_state["last_result"]

# ─── Context Bar ──────────────────────────────────────────
cols = st.columns(5)
with cols[0]:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{city}</div>
        <div class="metric-label">City</div>
    </div>""", unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{cuisine}</div>
        <div class="metric-label">Cuisine</div>
    </div>""", unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{meal_type}</div>
        <div class="metric-label">Meal Type</div>
    </div>""", unsafe_allow_html=True)
with cols[3]:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{user_type}</div>
        <div class="metric-label">User Type</div>
    </div>""", unsafe_allow_html=True)
with cols[4]:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color: #4CAF50">{total_time:.0f}ms</div>
        <div class="metric-label">Total Latency</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Cart + Recommendations Side by Side ──────────────────
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("### 🛒 Current Cart")

    cart_df = items_df[items_df["item_id"].isin(cart_item_ids)]
    cart_total = cart_df["price"].sum()

    for _, item in cart_df.iterrows():
        color = CATEGORY_COLORS.get(item["category"], "#888")
        st.markdown(f"""
        <div class="cart-item">
            <div>
                <div class="item-name">{item["item_name"]}</div>
                <div class="item-meta">
                    <span class="category-badge" style="background: {color}22; color: {color};">{item["category"]}</span>
                </div>
            </div>
            <div class="item-price">Rs.{int(item["price"])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align: right; padding: 0.8rem 1rem; margin-top: 0.5rem;
                border-top: 1px solid rgba(255,255,255,0.1);">
        <span style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Cart Total</span>
        <span style="color: #4CAF50; font-weight: 700; font-size: 1.2rem; margin-left: 0.5rem;">Rs.{int(cart_total)}</span>
    </div>
    """, unsafe_allow_html=True)

    # Timing
    st.markdown(f"""
    <div style="margin-top: 1rem;">
        <span class="timing-badge">Ranking: {rank_time:.1f}ms</span>
        <span class="timing-badge" style="margin-left: 0.3rem;">Total: {total_time:.1f}ms</span>
    </div>
    """, unsafe_allow_html=True)


with right_col:
    st.markdown("### 🎯 Top 8 Recommendations")

    if top8.empty:
        st.warning("No candidates found for this restaurant.")
    else:
        for rank_idx, (_, row) in enumerate(top8.iterrows()):
            color = CATEGORY_COLORS.get(row["category"], "#888")
            relevance = row["relevance"]

            st.markdown(f"""
            <div class="rec-card" style="border-left-color: {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="rec-rank">#{rank_idx + 1} Recommendation</div>
                        <div class="rec-name">{row["item_name"]}</div>
                        <div class="rec-meta">
                            <span class="category-badge" style="background: {color}22; color: {color};">{row["category"]}</span>
                            <span style="margin-left: 0.5rem; color: rgba(255,255,255,0.5);">Rs.{int(row["price"])}</span>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 80px;">
                        <div style="color: {color}; font-size: 1.4rem; font-weight: 700;">{relevance:.0f}%</div>
                        <div style="color: rgba(255,255,255,0.4); font-size: 0.7rem;">relevance</div>
                    </div>
                </div>
                <div class="relevance-bar">
                    <div class="relevance-fill" style="width: {relevance}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── LLM Tooltip ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if not top8.empty:
    cart_name = cart_df.iloc[-1]["item_name"]
    rec_name = top8.iloc[0]["item_name"]

    with st.expander("✨ AI-Powered Suggestion (Groq Llama 3.1 8B)", expanded=True):
        with st.spinner("Generating explanation..."):
            tooltip = get_llm_tooltip(cart_name, rec_name)

        st.markdown(f"""
        <div class="tooltip-card">
            <div class="tooltip-label">AI Generated Insight</div>
            <div class="tooltip-text">"{tooltip}"</div>
        </div>
        """, unsafe_allow_html=True)
