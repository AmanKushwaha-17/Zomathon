import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
df = pd.read_csv("csao_ranking_data_v3_personalized.csv")
df = df.sort_values(["cart_id", "step"]).reset_index(drop=True)

# -----------------------------
# 2️⃣ Define Features
# -----------------------------
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

categorical_cols = ["user_type", "last_category", "candidate_category"]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# -----------------------------
# 3️⃣ Simulate Production Split (Same Logic as Training)
# -----------------------------
unique_carts = df["cart_id"].unique()
split_idx = int(len(unique_carts) * 0.8)

test_carts = unique_carts[split_idx:]
test_df = df[df["cart_id"].isin(test_carts)].reset_index(drop=True)

X_test = test_df[feature_cols]
y_test = test_df["label"]

test_groups = test_df.groupby(["cart_id", "step"]).size().values

# -----------------------------
# 4️⃣ Load Trained Model
# -----------------------------
model = lgb.Booster(model_file="csao_lambdarank_model.txt")

print("✅ Model loaded successfully.\n")

# -----------------------------
# 5️⃣ Generate Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6️⃣ Evaluate Model NDCG@8
# -----------------------------
ndcg_scores = []
start = 0

for g in test_groups:
    end = start + g

    true_labels = y_test.iloc[start:end].values.reshape(1, -1)
    pred_scores = y_pred[start:end].reshape(1, -1)

    ndcg = ndcg_score(true_labels, pred_scores, k=8)
    ndcg_scores.append(ndcg)

    start = end

model_ndcg = np.mean(ndcg_scores)

print("🔥 Production Test NDCG@8:", round(model_ndcg, 4))

# -----------------------------
# 7️⃣ Baseline Evaluation
# -----------------------------
baseline_scores = X_test["transition_probability"].values

ndcg_scores_baseline = []
start = 0

for g in test_groups:
    end = start + g

    true_labels = y_test.iloc[start:end].values.reshape(1, -1)
    pred_scores = baseline_scores[start:end].reshape(1, -1)

    ndcg = ndcg_score(true_labels, pred_scores, k=8)
    ndcg_scores_baseline.append(ndcg)

    start = end

baseline_ndcg = np.mean(ndcg_scores_baseline)

print("Baseline NDCG@8:", round(baseline_ndcg, 4))
print("📈 Absolute Lift:", round(model_ndcg - baseline_ndcg, 4))

# -----------------------------
# 8️⃣ Segment-Level Evaluation
# -----------------------------
segment_results = {
    "Budget": [],
    "Regular": [],
    "Premium": []
}

start = 0

for g in test_groups:
    end = start + g

    true_labels = y_test.iloc[start:end].values.reshape(1, -1)
    pred_scores = y_pred[start:end].reshape(1, -1)

    segment = test_df.iloc[start]["user_type"]

    ndcg = ndcg_score(true_labels, pred_scores, k=8)
    segment_results[segment].append(ndcg)

    start = end

print("\n--- 📊 Segment-Level Production NDCG ---")

for segment in segment_results:
    print(segment, ":", round(np.mean(segment_results[segment]), 4))
