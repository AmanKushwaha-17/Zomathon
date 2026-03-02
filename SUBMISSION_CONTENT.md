# 7. System Design & Production Readiness

## 7.1 Inference Pipeline Architecture

```
    User Request (cart state)
            |
            v
    +----- Cart Embedding -----+
    | Mean of item embeddings  |    ~5 ms
    | L2 normalized            |
    +--------------------------+
            |
            v
    +-- Candidate Generation --+
    | Restaurant-scoped        |    ~10 ms
    | Primary + Secondary pool |
    | Deduplicated by item_id  |
    +--------------------------+
            |
            v
    +-- Feature Engineering ---+
    | 20 features computed     |    ~10 ms
    | Transition probs         |
    | Embedding similarity     |
    +--------------------------+
            |
            v
    +---- LambdaRank Model ----+
    | LightGBM predict()       |    ~6 ms
    +--------------------------+
            |
            v
    +---- Diversity Layer -----+
    | Round-robin interleave   |    ~3 ms
    | Max 3 per category       |
    | Re-sort by score         |
    | Normalize to 0-100       |
    +--------------------------+
            |
            v
       Top 8 CSAO Response         Total: ~35 ms
            |
     (async) LLM Tooltip via Groq   Non-blocking
```

## 7.2 Latency Budget

| Stage | Time | Notes |
|-------|------|-------|
| Cart Embedding | ~5 ms | Pre-computed embeddings, fast lookup |
| Candidate Generation | ~10 ms | Restaurant-scoped: ~20 items, not millions |
| Feature Engineering | ~10 ms | All computed in-memory, no external calls |
| Ranking | **~6 ms** | LightGBM single-batch prediction |
| Diversity + Normalization | ~3 ms | Simple iteration over 8 items |
| **Total** | **~35 ms** | **Well within 200-300 ms constraint** |
| LLM Tooltip | 500-1500 ms | **Async — does not block response** |

## 7.3 Scalability Architecture

| Design Decision | Benefit |
|----------------|---------|
| **Stateless inference** | Each request is independent — horizontally scalable |
| **Restaurant-scoped candidates** | ~20 candidates per request, not millions |
| **Pre-computed embeddings** | Loaded once, served from memory |
| **CPU-friendly model** | LightGBM runs on CPU — no GPU dependency |
| **Cached transitions** | JSON loaded at startup, no database calls |
| **Model size: 1.4 MB** | Fits in memory across all instances |

At peak traffic (lunch/dinner), this architecture handles **millions of requests** by adding stateless inference pods behind a load balancer.

## 7.4 Benchmarking Strategy

| Test Type | What We Validated |
|-----------|------------------|
| Data Integrity (53 tests) | No NaN, correct labels, valid categories, group structure |
| Inference Stress Test | 50 random restaurants — 0 failures, 0 duplicates |
| Score Distribution | Positives consistently score higher than negatives |
| Determinism | Same input produces identical output every time |
| Relevance Validation | Side ranks #1 after Main in 83-90% of restaurants |
| Cart Exclusion | Cart items never appear in recommendations |
| Latency Profiling | Ranking consistently <10ms, total <40ms |

---

# 8. Cold Start Strategy

## 8.1 New Users (No Order History)

| Strategy | Implementation |
|----------|---------------|
| Default user_type | "Regular" (median behavior) |
| Ranking signal | Popularity + transition probability + embedding similarity |
| No memory dependency | All 20 features are computable without user history |

The model does not require user-specific features like purchase history. User type, restaurant context, and cart state are sufficient for high-quality ranking.

## 8.2 New Restaurants (No Training Data)

| Strategy | Implementation |
|----------|---------------|
| Menu embeddings | Generated from category base vectors (available instantly) |
| Transition priors | Global transition matrix applies universally |
| Popularity fallback | Default popularity=1 for new items |
| Restaurant ID | Model treats unseen IDs via LightGBM's categorical handling |

Both cold start scenarios are handled without any separate fallback system — the same model serves warm and cold users/restaurants.

---

# 9. Business Impact & A/B Testing

## 9.1 Offline-to-Business Metric Translation

| Offline Metric | Business Impact |
|---------------|-----------------|
| NDCG@8: 0.82 (vs baseline 0.64) | More relevant items appear in top positions |
| +28.5% ranking improvement | Higher probability of users accepting top-3 suggestions |
| Side ranks #1 after Main (83%+) | Natural meal completion drives add-on acceptance |

**Projected Business Impact:**

| Metric | Projected Change |
|--------|-----------------|
| **AOV Lift** | **8-15%** |
| Add-on acceptance rate | 18-25% (up from ~10% baseline) |
| Cart-to-Order ratio | +3-5% improvement |
| Incomplete meal rate | Reduced by 20-30% |

## 9.2 A/B Testing Framework

**Design:**

| Parameter | Value |
|-----------|-------|
| Traffic Split | 50% Control / 50% Treatment |
| Control | Current CSAO rail (rule-based or no ML) |
| Treatment | LambdaRank model |
| Duration | 2-4 weeks for statistical significance |

**Metrics Hierarchy:**

| Type | Metric | Success Criteria |
|------|--------|-----------------|
| **Primary** | Average Order Value (AOV) | +5% lift |
| Secondary | Add-to-cart CTR on CSAO rail | +15% lift |
| Secondary | Items per order | +0.3 items |
| Guardrail | Order completion rate | Must NOT drop >1% |
| Guardrail | API latency p99 | Must stay <300ms |
| Guardrail | Category starvation | No category with 0% representation |

**Rollout Plan:**

| Phase | Traffic | Duration | Gate |
|-------|---------|----------|------|
| Phase 1 | 10% | 1 week | No guardrail violations |
| Phase 2 | 50% | 2 weeks | Primary metric positive |
| Phase 3 | 100% | Ongoing | Full deployment with monitoring |

---

# 10. Conclusion

We built an end-to-end Cart Super Add-On recommendation engine that is:

- **Sequential-aware**: Transition modeling captures natural meal-building flow
- **Personalized**: User type, behavioral memory, and spend bias drive individual recommendations
- **Context-rich**: 20 features spanning user, restaurant, time, cart, candidate, and semantic dimensions
- **Low latency**: ~35ms total inference, well within 200-300ms production constraint
- **Production scalable**: Stateless, CPU-friendly, horizontally scalable architecture
- **AI-augmented**: LLM-powered natural language tooltips enhance user experience
- **Cold-start robust**: No dependency on user history for baseline ranking
- **Business-aligned**: Clear A/B testing framework with projected 8-15% AOV lift

The system demonstrates that a well-engineered learning-to-rank approach — combining domain-specific feature engineering, realistic data simulation, and modern AI augmentation — can deliver a production-ready recommendation rail that directly impacts business revenue.

---

# Links

| Resource | URL |
|----------|-----|
| GitHub Repository | [Set to Public Access] |
| Key Files | `data_generator.py`, `train_ranker.py`, `test_ranker.py`, `inference_demo.py` |
| Documentation | `data_generator.md`, `README.md` |
