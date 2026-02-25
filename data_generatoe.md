# 📌 CSAO Synthetic Data Generator – Team Summary

## 🎯 Objective

We are building a **sequential cart recommendation training dataset** for the CSAO (Cart Super Add-On) system.

The goal is to simulate:

> “Given the current cart state, which item will the user add next?”

This dataset will be used to train a **ranking model (LightGBM Ranker)**.

---

# 🧠 What We Have Built

We created a **synthetic behavioral simulator** that generates:

1. Users (with segmentation)
2. Items (with features)
3. Orders (step-by-step cart growth)
4. Ranking groups (1 positive + 10 negatives per step)

---

# 🏗 System Architecture Overview

## 1️⃣ User Simulation

We generate users with segments:

* 50% Budget
* 30% Regular
* 20% Premium

Each user has:

```
user_id
user_type
```

User type influences behavior (e.g., Premium adds desserts more often).

---

## 2️⃣ Item Simulation

Each item has:

* category (Main, Side, Dessert, Drink)
* price
* popularity (Zipf distribution for long tail)
* is_spicy (binary)

This gives realistic product diversity.

---

## 3️⃣ Sequential Cart Growth

Every order:

1. Starts with a Main
2. Then evolves based on transition rules

Example transitions:

Main → Side (0.5)
Main → Drink (0.3)
Main → Main (0.15)
Main → Dessert (0.2)

These probabilities can be modified by:

* user_type
* spicy boost

So behavior becomes personalized + contextual.

---

## 4️⃣ Ranking Dataset Creation

At each cart step:

We generate:

* 1 positive item (what user actually added)
* 10 negative items (items not chosen)

Each (cart_id + step) forms one ranking group.

This structure allows training a learning-to-rank model.

---

# 📊 Features Available Per Candidate

For each candidate item, model sees:

### Cart Context

* cart_size
* cart_value
* count_main
* count_side
* count_dessert
* count_drink
* last_category

### Candidate Features

* candidate_category
* candidate_price
* candidate_popularity
* transition_probability

### User Feature

* user_type

### Target

* label (1 or 0)

---


# 🚀 What We Can Improve Next

Here are areas teammates can enhance to make dataset more realistic:

---

##  1. Add Cuisine Type

Add:

```
cuisine_type
```

Examples:
North Indian
Chinese
Italian

Then:
Increase transition probability when cuisines match.

---

##  2. Add Time Context

Add:

* hour_of_day
* meal_type (Breakfast/Lunch/Dinner)

Modify transitions based on meal time.

Example:
Dessert more likely at dinner.

---

##  3. Add Location / Zone

Simulate:

* City
* Delivery zone

Make popularity vary by zone.

---

##  4. Add User Behavioral Memory

Instead of pure random orders:

Track user historical preference:

* Dessert ratio
* Drink ratio
* Avg cart value

Use that to modify probabilities.

---

##  5. Improve Negative Sampling

Currently:
Negatives are random.

Upgrade to:
Hard negatives (same category but not chosen).

This improves model quality.


