import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score

# -----------------------------
# 1) Load Dataset
# -----------------------------
df = pd.read_csv("csao_ranking_data_v3_personalized.csv")
df = df.sort_values(["cart_id", "step"]).reset_index(drop=True)

# -----------------------------
# 2) Define Features
# -----------------------------
feature_cols = [
    "user_type",
    "restaurant_id",
    "city",
    "cuisine",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "meal_type",
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
    "embedding_similarity",
]

categorical_cols = [
    "user_type",
    "restaurant_id",
    "city",
    "cuisine",
    "meal_type",
    "last_category",
    "candidate_category",
]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# -----------------------------
# 3) Simulate Production Split
# -----------------------------
unique_carts = df["cart_id"].unique()
split_idx = int(len(unique_carts) * 0.8)

test_carts = unique_carts[split_idx:]
test_df = df[df["cart_id"].isin(test_carts)].reset_index(drop=True)

X_test = test_df[feature_cols]
y_test = test_df["label"]

group_frame = test_df.groupby(["cart_id", "step"], sort=False)
test_groups = group_frame.size().values
group_meta = group_frame.agg(
    user_type=("user_type", "first"),
    meal_type=("meal_type", "first"),
).reset_index(drop=True)


def iter_group_slices(groups):
    start = 0
    for group_size in groups:
        end = start + group_size
        yield start, end, group_size
        start = end


def safe_group_ndcg(y_true, y_scores, groups, k=8):
    scores = []
    skipped = 0
    for start, end, group_size in iter_group_slices(groups):
        if group_size < 2:
            skipped += 1
            continue
        true_labels = y_true.iloc[start:end].values.reshape(1, -1)
        pred_scores = y_scores[start:end].reshape(1, -1)
        scores.append(ndcg_score(true_labels, pred_scores, k=k))

    return (np.mean(scores) if scores else 0.0), skipped, len(groups)


# -----------------------------
# 4) Load Trained Model
# -----------------------------
model = lgb.Booster(model_file="csao_lambdarank_model.txt")
print("Model loaded successfully.\n")

# -----------------------------
# 5) Generate Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6) Evaluate Model NDCG@8
# -----------------------------
model_ndcg, skipped_model_groups, total_groups = safe_group_ndcg(
    y_test, y_pred, test_groups, k=8
)
print("Production Test NDCG@8:", round(model_ndcg, 4))
print(f"Skipped groups (<2 docs): {skipped_model_groups}/{total_groups}")

# -----------------------------
# 7) Baseline Evaluation
# -----------------------------
baseline_scores = X_test["transition_probability"].values
baseline_ndcg, skipped_baseline_groups, _ = safe_group_ndcg(
    y_test, baseline_scores, test_groups, k=8
)
print("Baseline NDCG@8:", round(baseline_ndcg, 4))
print(f"Baseline skipped groups (<2 docs): {skipped_baseline_groups}/{total_groups}")
print("Absolute Lift:", round(model_ndcg - baseline_ndcg, 4))

# -----------------------------
# 8) Segment-Level Evaluation
# -----------------------------
segment_results = {
    "Budget": [],
    "Regular": [],
    "Premium": [],
}
meal_results = {}

for idx, (start, end, group_size) in enumerate(iter_group_slices(test_groups)):
    if group_size < 2:
        continue

    true_labels = y_test.iloc[start:end].values.reshape(1, -1)
    pred_scores = y_pred[start:end].reshape(1, -1)
    ndcg = ndcg_score(true_labels, pred_scores, k=8)

    segment = group_meta.iloc[idx]["user_type"]
    meal = group_meta.iloc[idx]["meal_type"]

    if segment in segment_results:
        segment_results[segment].append(ndcg)
    meal_results.setdefault(meal, []).append(ndcg)

print("\n--- Segment-Level Production NDCG ---")
for segment in segment_results:
    segment_mean = np.mean(segment_results[segment]) if segment_results[segment] else 0.0
    print(segment, ":", round(segment_mean, 4))

# -----------------------------
# 9) Time-Based Breakdown
# -----------------------------
print("\n--- Meal-Type Performance ---")
for meal in sorted(meal_results.keys()):
    print(meal, ":", round(np.mean(meal_results[meal]), 4))
