import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
df = pd.read_csv("csao_ranking_data_v3_personalized.csv")

# Ensure proper ordering
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
    "embedding_similarity"  # 🔥 NEW FEATURE
]

target_col = "label"

categorical_cols = ["user_type", "last_category", "candidate_category"]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# -----------------------------
# 3️⃣ Proper Cart-Based Train/Test Split
# -----------------------------
unique_carts = df["cart_id"].unique()

train_cart_count = int(len(unique_carts) * 0.8)
train_carts = unique_carts[:train_cart_count]
test_carts = unique_carts[train_cart_count:]

train_df = df[df["cart_id"].isin(train_carts)]
test_df = df[df["cart_id"].isin(test_carts)]

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

# -----------------------------
# 4️⃣ Create Group Sizes
# -----------------------------
train_groups = train_df.groupby(["cart_id", "step"]).size().values
test_groups = test_df.groupby(["cart_id", "step"]).size().values

# -----------------------------
# 5️⃣ Train LambdaRank Model
# -----------------------------
model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=31
)

model.fit(
    X_train,
    y_train,
    group=train_groups,
    eval_set=[(X_test, y_test)],
    eval_group=[test_groups],
    eval_at=[8],
    categorical_feature=categorical_cols
)

print("✅ Training complete.\n")

# -----------------------------
# 6️⃣ Evaluate Model NDCG@8
# -----------------------------
y_pred = model.predict(X_test)

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
print(f"🔥 Model Mean NDCG@8: {model_ndcg:.4f}")

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
print(f"Baseline Mean NDCG@8: {baseline_ndcg:.4f}")

print(f"📈 Absolute Lift: {model_ndcg - baseline_ndcg:.4f}\n")

# -----------------------------
# 8️⃣ Segment-Level Evaluation
# -----------------------------
segment_results = {
    "Budget": {"model": [], "baseline": []},
    "Regular": {"model": [], "baseline": []},
    "Premium": {"model": [], "baseline": []}
}

start = 0
test_df = test_df.reset_index(drop=True)

for g in test_groups:
    end = start + g

    true_labels = y_test.iloc[start:end].values.reshape(1, -1)
    pred_scores_model = y_pred[start:end].reshape(1, -1)
    pred_scores_baseline = baseline_scores[start:end].reshape(1, -1)

    segment = test_df.iloc[start]["user_type"]

    ndcg_model = ndcg_score(true_labels, pred_scores_model, k=8)
    ndcg_baseline = ndcg_score(true_labels, pred_scores_baseline, k=8)

    segment_results[segment]["model"].append(ndcg_model)
    segment_results[segment]["baseline"].append(ndcg_baseline)

    start = end

print("\n--- 📊 Segment-Level Results ---")

for segment in segment_results:
    model_mean = np.mean(segment_results[segment]["model"])
    baseline_mean = np.mean(segment_results[segment]["baseline"])
    lift = model_mean - baseline_mean

    print(f"\n{segment}:")
    print(f"  Model NDCG@8: {model_mean:.4f}")
    print(f"  Baseline NDCG@8: {baseline_mean:.4f}")
    print(f"  Absolute Lift: {lift:.4f}")

# -----------------------------
# 9️⃣ Feature Importance
# -----------------------------
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n--- 🔍 Feature Importance ---")
print(importance_df)

# Save Model
model.booster_.save_model("csao_lambdarank_model.txt")
print("\n✅ Model saved as csao_lambdarank_model.txt")