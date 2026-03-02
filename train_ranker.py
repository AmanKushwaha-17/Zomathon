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
# 2️⃣ Define Features (UPDATED)
# -----------------------------
feature_cols = [

    # User
    "user_type",

    # Restaurant Context
    "restaurant_id",
    "city",
    "cuisine",

    # Time Context
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "meal_type",

    # Cart Context
    "cart_size",
    "cart_value",
    "count_main",
    "count_side",
    "count_dessert",
    "count_drink",
    "last_category",

    # Candidate Features
    "candidate_category",
    "candidate_price",
    "candidate_popularity",
    "transition_probability",
    "embedding_similarity"
]

target_col = "label"

categorical_cols = [
    "user_type",
    "restaurant_id",
    "city",
    "cuisine",
    "meal_type",
    "last_category",
    "candidate_category"
]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# -----------------------------
# 3️⃣ Cart-Based Train/Test Split
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
train_groups = train_df.groupby(["cart_id", "step"], sort=False).size().values
test_groups = test_df.groupby(["cart_id", "step"], sort=False).size().values

# -----------------------------
# 5️⃣ Train LambdaRank Model
# -----------------------------
model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20
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

print("Training complete.\n")

# -----------------------------
# 6️⃣ Evaluate Model NDCG@8
# -----------------------------
def mean_ndcg_by_group(y_true, y_scores, groups, k=8):
    ndcg_scores = []
    skipped_groups = 0
    start_idx = 0

    for group_size in groups:
        end_idx = start_idx + group_size
        if group_size < 2:
            skipped_groups += 1
            start_idx = end_idx
            continue

        true_labels = y_true.iloc[start_idx:end_idx].values.reshape(1, -1)
        pred_scores = y_scores[start_idx:end_idx].reshape(1, -1)
        ndcg_scores.append(ndcg_score(true_labels, pred_scores, k=k))
        start_idx = end_idx

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return mean_ndcg, skipped_groups, len(groups)

y_pred = model.predict(X_test)

model_ndcg, skipped_model_groups, total_groups = mean_ndcg_by_group(
    y_test, y_pred, test_groups, k=8
)
print(f"Model Mean NDCG@8: {model_ndcg:.4f}")
print(f"Skipped groups (<2 docs): {skipped_model_groups}/{total_groups}")

# -----------------------------
# 7️⃣ Baseline Evaluation
# -----------------------------
baseline_scores = X_test["transition_probability"].values

baseline_ndcg, skipped_baseline_groups, _ = mean_ndcg_by_group(
    y_test, baseline_scores, test_groups, k=8
)

print(f"Baseline Mean NDCG@8: {baseline_ndcg:.4f}")
print(f"Baseline skipped groups (<2 docs): {skipped_baseline_groups}/{total_groups}")
print(f"Absolute Lift: {model_ndcg - baseline_ndcg:.4f}\n")

# -----------------------------
# 8️⃣ Feature Importance
# -----------------------------
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n--- Feature Importance ---")
print(importance_df)

# -----------------------------
# 9️⃣ Save Model
# -----------------------------
model.booster_.save_model("csao_lambdarank_model.txt")
print("\nModel saved as csao_lambdarank_model.txt")


