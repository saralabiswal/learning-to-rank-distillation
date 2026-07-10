# Project Requirements — `learning-to-rank-distillation`

**v1.0 Milestone Spec.** This document covers Tier 1 only — the working foundation. Tiers 2–5 (methodological depth, generalization, production-shaping, visibility) live in `ROADMAP.md` and build on top of this, not instead of it.

**Purpose:** A reusable, dataset-agnostic toolkit for (1) teacher→student knowledge distillation of ranking models, and (2) marketplace-aware multi-objective reranking. Built to close two specific gaps for the Expedia Senior Director, ML/AI interview loop — but architected as general infrastructure. Amazon ESCI is the primary public real-data path; Expedia RecTour remains the secondary travel-marketplace target.

**Repo:** `github.com/saralabiswal/learning-to-rank-distillation`
**License:** Apache 2.0 (code) — matches the license Expedia Group uses on its own challenge repos, which is a small deliberate signal of familiarity with their conventions.

---

## 1. Design Principle: Library First, Dataset Second

The single biggest change from the original spec: **no component should import dataset-specific column names.** Every dataset — Amazon ESCI, Expedia RecTour, a synthetic generator, or future adapters — implements the same interface and everything downstream (teacher, student, distillation, reranking, governance) is written against that interface only.

```python
# The only contract any dataset adapter must satisfy
@dataclass
class RankingExample:
    query_id: str          # one ranked list per query_id
    item_id: str           # the entity being ranked
    features: dict          # query-side + item-side features, adapter-defined
    label: float             # relevance label (click=1/booking=2/none=0, or adapter-defined scale)
    group_id: str            # the "supply" unit for fairness (e.g. property_id) — usually == item_id
    is_unbiased: bool        # True if this row's position was randomized (see DR-1)
    position: int | None     # observed position, only meaningful when is_unbiased=True
```

Everything in Sections 3–6 is written against `RankingExample`, never against raw dataset fields directly. Raw fields are mapped to this schema inside `adapters/` only.

---

## 2. Datasets — Amazon ESCI Primary, Expedia RecTour Secondary

### 2.1 Amazon ESCI Shopping Queries Dataset

Amazon ESCI is the primary public-data flow because it is available, reviewable, and has the same
query-candidate-relevance shape needed for learning-to-rank.

| ESCI field | Maps to |
|---|---|
| `query_id` | `RankingExample.query_id` |
| `product_id` | `RankingExample.item_id` |
| `esci_label` | `RankingExample.label`, mapped as `E=3`, `S=2`, `C=1`, `I=0` |
| `product_brand` | `RankingExample.group_id` when available; otherwise `product_id` |
| `query`, `product_title`, `product_description`, `product_bullet_point` | Text-derived features such as token counts and query/product overlap |
| `product_locale`, `source`, `product_color` | Categorical features |
| `split`, `small_version`, locale fields | Adapter filters, not model features |

**Required files:** `shopping_queries_dataset_examples.csv` or `.parquet` under `data/esci/`.
Product and source files are optional but used when present.

### 2.2 Expedia Group RecTour Research Dataset (2021)

**Source:** Released by Expedia Group's ML team for the RecSys 2021 RecTour workshop, explicitly motivated by multi-objective/multi-stakeholder marketplace research — which is why this replaces the 2013 ICDM dataset used in the earlier draft of this spec.

**License:** CC BY 4.0 (attribution only, no non-commercial restriction) — confirm current terms directly from Expedia Group before publishing, since dataset terms can be updated.

#### Known schema fields to map into `RankingExample`

| RecTour field | Maps to |
|---|---|
| `checkin_date`, `checkout_date`, `adult_count`, `destination_id`, `point_of_sale` | Query-side `features` |
| `applied_filters`, `sort_type`, `is_travel_ad` | Query-side `features` — also useful later for understanding pre-existing ranking/exposure bias in the logged data |
| `review_rating`, `review_count`, `price_bucket` | Item-side `features` |
| `is_drr` | Flag worth investigating early — confirm its exact definition from Expedia's dataset documentation before deciding whether it belongs in `features` or is a leakage risk (treat similarly to `position`, see DR-1) |
| property identifier field | `item_id` and `group_id` |
| click/booking outcome field | `label` |

**Action item before coding starts:** pull the current RecTour dataset documentation directly (schema has evolved since the original 2021 workshop PDF) and finalize the exact column-to-schema mapping — do not hardcode from memory of the 2021 slide deck, confirm against the actual data files.

### 2.3 Data requirements

- **DR-1 (position bias):** Exactly as in the original spec — any row where result order was influenced by a prior ranking model (i.e. not randomized) must not have its position used as a training feature, and offline evaluation should preferentially use the randomized-order subset if the dataset provides one. Confirm whether RecTour includes an equivalent to the ICDM dataset's `random_bool` flag; if not, document that offline evaluation is subject to logged-policy bias and note it explicitly in `README.md` limitations.
- **DR-2 (split integrity):** Split by `query_id`, never by row.
- **DR-3 (supply exposure):** Compute each `group_id`'s historical exposure (impression count) and outcome count from the training split only — this feeds the marketplace-fairness component (Section 5).
- **DR-4 (documentation):** Record exact dataset version/download date and any subsampling in `README.md`.

---

## 3. Functional Requirements — Teacher (FR-1 series)

- **FR-1.1:** `models/teacher.py` implements a `Teacher` interface (`fit(examples)`, `predict(examples) -> scores`) with a LightGBM LambdaMART implementation as the v1.0 default.
- **FR-1.2:** Listwise/pairwise ranking loss only — no pointwise classification objective.
- **FR-1.3:** Report NDCG@5 and NDCG@10 on held-out data as the quality ceiling.
- **FR-1.4:** Persist the trained teacher with a content hash of its training data — this is the reference the student and the promotion gate are measured against.

## 4. Functional Requirements — Student + Distillation (FR-2 series)

- **FR-2.1:** Two-tower architecture, implemented in **PyTorch**: query tower + item tower, combined via dot product/cosine similarity. Item-side embeddings precomputable and servable via ANN retrieval (**faiss-cpu**).
- **FR-2.2:** `distillation/response_based.py` — KL-divergence between softened student/teacher output distributions, blended with a standard ranking loss against ground-truth labels. Temperature is a documented, tunable hyperparameter.
- **FR-2.3:** `distillation/no_kd_baseline.py` — the same student architecture trained on labels only, no distillation loss. **Not optional.** This is the control that proves distillation contributed anything.
- **FR-2.4:** Both response-based and no-KD variants must be runnable through the same benchmark suite (Section 6) so results are directly comparable.

*(Feature-based and relation-based distillation are Tier 2 — see `ROADMAP.md`. v1.0 ships response-based only, deliberately, so the control experiment in FR-2.3 stays clean and interpretable before adding more moving parts.)*

## 5. Functional Requirements — Marketplace Reranking (FR-4 series)

- **FR-4.1:** `fairness/exposure.py` computes a per-`group_id` exposure-fairness signal from DR-3.
- **FR-4.2:** `fairness/constrained_rerank.py` implements a greedy constrained rerank: maximize relevance subject to a minimum exposure floor for low-exposure `group_id`s within each ranked list.
- **FR-4.3:** `benchmark/fairness_tradeoff.py` sweeps the exposure floor and reports NDCG@5 vs. an exposure-fairness metric (e.g. Gini coefficient across `group_id`s, or share of impressions to low-exposure groups) as a Pareto frontier, plotted with **matplotlib** (static plot — interactive exploration is Tier 5, not v1.0).
- **FR-4.4:** `README.md` states the fairness metric's definition and limitations explicitly — it's a proxy for supply-side exposure, not a model of actual partner revenue or negotiated terms.

*(Full multi-objective optimization via Pareto-frontier search, rather than a single greedy heuristic, is Tier 2 — see `ROADMAP.md`.)*

## 6. Functional Requirements — Benchmark Suite (FR-3 series)

- **FR-3.1:** `benchmark/run_all.py` produces one comparison table: Teacher vs. Student (KD) vs. Student (no-KD) across NDCG@5, NDCG@10, model size, and inference latency (p50/p99) at simulated batch query volume.
- **FR-3.2:** Quality-vs-latency Pareto plot (**matplotlib**, static) across at least 3 student configurations (e.g. varying embedding dimension or temperature).

## 7. Functional Requirements — Governance Gate (FR-5 series)

- **FR-5.1:** `governance/promotion_gate.py` — reuses the eight-stage governed pipeline pattern from `agentic-cdp-mlops`, implemented here as executable code, not narrative.
- **FR-5.2:** Default promotion rule (adjust after seeing real numbers): promote if NDCG@5 drop ≤ 2% and p99 latency improves ≥ 3x vs. teacher.
- **FR-5.3:** Every promotion decision logged (model version, metrics, gate result, timestamp) to a local SQLite registry.

*(CI-enforced promotion gating — where a GitHub Actions run automatically fails a PR that regresses past these thresholds — is Tier 2, and is flagged in `ROADMAP.md` as the highest-priority Tier 2 item.)*

## 8. Documentation Requirements (FR-6 series)

- **FR-6.1:** `README.md` — motivation (stated honestly as originating from closing a named interview gap, generalized into reusable infrastructure), architecture diagram, reproduction steps, benchmark table.
- **FR-6.2:** "Design Decisions" section mapping choices back to specific JD language.
- **FR-6.3:** "What This Doesn't Cover" section — no online A/B test, exposure metric is a proxy, and no hosted production deployment.
- **FR-6.4:** `ROADMAP.md` linked prominently from the top of `README.md` — this is what signals an ongoing project rather than a one-off, and should be treated as part of the v1.0 deliverable, not an afterthought.

---

## 9. Non-Functional Requirements

- **NFR-1 (Reproducibility):** Fixed seeds, documented dataset version, and pinned dependencies exactly as follows (final — do not substitute or upgrade without flagging it):

  | Package | Version | Role |
  |---|---|---|
  | Python | 3.11+ | |
  | pandas | ≥2.2 | Data handling |
  | pyarrow | ≥15 | Parquet support for ESCI files |
  | numpy | ≥1.26 | Data handling |
  | lightgbm | ≥4.3 | Teacher (LambdaMART) |
  | torch | ≥2.2 | Student (two-tower) |
  | faiss-cpu | ≥1.8 | ANN retrieval — CPU-only, deliberate (see NFR-2) |
  | scikit-learn | ≥1.4 | Query-grouped splitting, general metrics |
  | matplotlib | ≥3.8 | Pareto plots (static — Tier 5 owns interactive) |
  | fastapi | ≥0.115 | Local serving endpoint |
  | uvicorn | ≥0.30 | ASGI server |
  | prometheus-client | ≥0.20 | Runtime metrics |
  | pytest | ≥8.0 | Testing |
  | ruff | ≥0.4 | Lint/format |

  SQLite is stdlib (governance registry, FR-5.3) — no separate dependency. Streamlit is an optional
  dashboard extra, not required for the core reproducible pipeline.
- **NFR-2 (Runtime):** Full v1.0 pipeline runs end-to-end in under ~2 hours on a single machine.
- **NFR-3 (Code quality):** Adapter interface (Section 1) enforced — no dataset-specific logic outside `adapters/`. This is the requirement that makes Tier 3 (a second dataset) cheap later instead of a rewrite.
- **NFR-4 (Honesty in framing):** Every README claim verifiable by running the code.
- **NFR-5 (Packaging):** Installable via `pip install -e .`; a minimal CLI exposed under the short alias `ltrd` (e.g. `ltrd train-teacher --dataset rectour`, `ltrd benchmark`) — the repo and package keep the full descriptive name for searchability, but the CLI entry point stays short enough to actually type. Full CLI surface can grow in Tier 5, but the package structure should support it from v1.0 rather than needing a restructure.

---

## 10. Suggested Repo Structure

```
learning-to-rank-distillation/
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── src/learning_to_rank_distillation/
│   ├── schema.py                    # RankingExample (Section 1)
│   ├── datasets.py                  # Dataset selection shared by CLI/benchmark
│   ├── cli.py                       # ltrd command entrypoint
│   ├── adapters/
│   │   ├── esci.py                  # Amazon ESCI → RankingExample
│   │   ├── rectour.py               # Expedia RecTour → RankingExample
│   │   └── synthetic.py             # Deterministic fallback data
│   ├── models/
│   │   └── teacher.py               # FR-1
│   ├── distillation/
│   │   ├── response_based.py        # FR-2.2
│   │   └── no_kd_baseline.py        # FR-2.3
│   ├── fairness/
│   │   ├── exposure.py              # FR-4.1
│   │   └── constrained_rerank.py    # FR-4.2
│   ├── benchmark/
│   │   ├── run_all.py               # FR-3.1
│   │   └── fairness_tradeoff.py     # FR-4.3
│   └── governance/
│       └── promotion_gate.py        # FR-5
├── notebooks/
│   └── exploration.ipynb
├── artifacts/
│   └── promotion_registry.sqlite
└── docs/
    └── architecture_diagram.png
```

---

## 11. v1.0 Milestones

| Phase | Exit Criteria |
|---|---|
| M1 — Schema + RecTour adapter | `RankingExample` defined; RecTour data mapped and validated against DR-1–DR-4 |
| M2 — Teacher | Trained; NDCG@5/@10 baseline reported |
| M3 — Student + distillation | Response-based and no-KD student both trained; benchmark table produced |
| M4 — Marketplace layer | Constrained rerank implemented; relevance-vs-fairness Pareto frontier plotted |
| M5 — Governance + docs | Promotion gate coded and logged; README + ROADMAP complete |

## 12. Definition of Done (v1.0)

- [x] `RankingExample` schema defined; zero dataset-specific mapping logic outside `adapters/`
- [x] Amazon ESCI adapter implemented, tested, and selectable from CLI/benchmark paths
- [x] Guarded RecTour adapter implemented; it fails clearly until real files and exact mapping are supplied
- [ ] RecTour 2021 field mapping confirmed against current real files/documentation
- [x] Teacher, student (KD), and student (no-KD) all train and run through the benchmark suite
- [x] Benchmark table + quality-vs-latency Pareto plot
- [x] Marketplace exposure-fairness metric implemented and plotted as a second Pareto frontier
- [x] Promotion gate implemented as code, with a logged decision
- [x] `README.md` complete (motivation, architecture, reproduction, Design Decisions, What This Doesn't Cover, link to ROADMAP)
- [x] `ROADMAP.md` published (see companion document)
- [ ] Repo public, installable, and the link ready to share
