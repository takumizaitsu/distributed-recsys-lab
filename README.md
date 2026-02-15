# distributed-recsys-lab

![Status](https://img.shields.io/badge/status-active_research-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A reproducible laboratory for recommender systems: **Retrieval → Ranking → Experimentation**.  
Designed with scale, reliability, and causal reasoning in mind.

## Core Principles

- **Reproducible by default**: No "it works on my machine". Everything is defined by seed, env, and runbooks.
- **Measure > Vibe**: Decisions are based on latency, cost, and offline/online metrics, not intuition.
- **Offline ↔ Online Consistency**: Minimizing training-serving skew is a priority.
- **SLO-driven Design**: Architecture respects defined SLOs (latency/availability/cost).
- **Causal when needed**: Beyond correlation; aiming for counterfactual evaluation and bias reduction.

## Metrics (first-class citizens)
We track ranking quality (NDCG/Recall), system performance (p95/p99 latency, QPS, cost), and reliability (error rate, freshness).

## Data policy
Only public or synthetic datasets are used. No proprietary or client data will be included.

## Quickstart
Coming soon. The first baseline will be MF + NDCG/Recall evaluation with a reproducible seed and environment.

## Structure

- `docs/` : Design docs (RFCs), architectural decisions, and runbooks.
- `src/`  : Implementation of algorithms and infrastructure.
- `experiments/` : Reproducible experiment logs and notebooks.

---
*Note: This is an ongoing research project for building distributed recommender systems with reproducible experiments.*
