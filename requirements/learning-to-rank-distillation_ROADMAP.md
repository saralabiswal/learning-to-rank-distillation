# Roadmap — `learning-to-rank-distillation`

`README.md` covers what v1.0 (Tier 1) does today. This document is the public statement of where the project goes next — it's part of the deliverable, not a someday-list, because a roadmap is what signals an ongoing tool rather than a one-off interview artifact.

Tiers are ordered by dependency, not strictly by priority — see **"Suggested Build Order"** at the bottom for where to actually spend time first.

---

## Tier 2 — Methodological Depth

Goal: move from "distillation works" to "distillation is well-understood here," with evidence, not just a working pipeline.

- [ ] **Transformer-based teacher.** Add a small multi-task transformer ranker as a second teacher option alongside LightGBM — this is what earns the term "Ranking Foundation Model" rather than borrowing it from the job description.
- [ ] **Feature-based distillation.** Auxiliary loss matching the student's item-tower embedding to a projection of the teacher's intermediate representation. Only justified if the response-based student (Tier 1) underperforms enough on an ablation to need it.
- [ ] **Relation-based distillation.** Preserve the teacher's pairwise/listwise ranking relations directly in the student's loss, rather than only matching output distributions.
- [ ] **Ablation table.** All three distillation methods compared head-to-head on the same benchmark suite, with a written explanation of which mattered most and why — this table is more valuable than any single method's result.
- [ ] **Real multi-objective optimization for fairness.** Replace the Tier 1 greedy constrained rerank with proper Pareto-frontier search (scalarization sweep or NSGA-II) over relevance vs. exposure-fairness.
- [ ] **Counterfactual / off-policy evaluation.** Inverse propensity scoring to correct offline evaluation for position bias in the logged data — directly relevant if RecTour's position field turns out to reflect a prior ranking policy rather than randomized order (see `REQUIREMENTS.md` DR-1).
- [ ] **CI-enforced promotion gate.** GitHub Actions re-runs the benchmark suite on every PR and fails the build if a change regresses NDCG or latency past the Tier 1 promotion thresholds. Turns "governance discipline" from a README claim into something verifiable by opening the Actions tab.

---

## Tier 3 — Generalization

Goal: prove the pipeline is a tool, not a script that happens to work on one dataset.

- [ ] **Second dataset adapter — Amazon ESCI (Shopping Queries Dataset).** Real e-commerce search relevance data with genuinely different item semantics from travel — the strongest possible proof the `RankingExample` interface actually generalizes.
- [ ] **Third dataset adapter — MovieLens.** Fastest to add given how standard it is; useful as the "quickstart" example for anyone trying the toolkit for the first time.
- [ ] **Synthetic marketplace generator.** A configurable generator (supply concentration level, cold-start rate, exposure skew) so the fairness reranker can be stress-tested without needing any of the three real datasets — also the fastest way for someone else to try the fairness component without a data-access step at all.
- [ ] **Cross-dataset benchmark comparison.** One table showing teacher/student quality-vs-latency trade-offs are consistent in shape across all adapters, not just favorable on RecTour.

---

## Tier 4 — Production Shaping

Goal: move from research-shaped code to something that resembles what would actually get deployed.

- [ ] **Real serving endpoint.** Student model behind FastAPI, item embeddings in a FAISS or ScaNN index, actual ANN retrieval rather than in-memory brute-force scoring.
- [ ] **Load testing.** Locust or k6 load test producing real p50/p99 latency numbers under simulated concurrent traffic, replacing the Tier 1 batch-scoring latency estimate.
- [ ] **Experiment tracking.** MLflow or Weights & Biases wired into training runs, replacing ad hoc script output.
- [ ] **Model registry integration.** Promotion gate (Tier 1/2) writes to a real model registry (MLflow Model Registry or equivalent) instead of a local SQLite log.
- [ ] **Containerization.** Dockerfile + docker-compose for the serving endpoint, so the whole thing runs with one command on someone else's machine.

---

## Tier 5 — Visibility

Goal: make the work legible to people who will never clone the repo.

- [ ] **Interactive Pareto dashboard.** Streamlit app with a live exposure-floor slider updating the relevance-vs-fairness trade-off in real time — the artifact to actually screen-share, far more persuasive live than a static plot.
- [ ] **Technical write-up.** A 4–6 page piece in the style of Expedia's own RecTour/MORS workshop papers — publishable as a blog post, shareable on LinkedIn, potentially shapeable toward an actual workshop submission.
- [ ] **Polished packaging.** PyPI publish, README badges (build status, license, PyPI version), a proper `CONTRIBUTING.md` if the project starts attracting outside interest.
- [ ] **Architecture diagram, professionally rendered** — not a hand-sketch, something that could sit in a slide deck.

---

## Suggested Build Order (Highest Leverage First)

If time is unconstrained but attention isn't, this is the order I'd actually work in, independent of tier number:

1. **Feature-based + relation-based distillation** (Tier 2) — the most direct answer to "have you really done this," since response-based alone is the entry-level version of the technique.
2. **CI-enforced promotion gate** (Tier 2) — cheap to build, and converts a governance claim into something a stranger can verify without asking you.
3. **Interactive Pareto dashboard** (Tier 5) — highest visibility-per-hour of anything on this list; it's what actually gets looked at.
4. **Second dataset adapter** (Tier 3) — proves generality, but lower urgency than the three above.
5. Everything else — genuinely valuable, but "ongoing project you keep improving" rather than "blocking anything."

---

## Non-Goals (For Now)

Stated explicitly so scope stays honest:

- Not attempting real-time online A/B testing against live traffic — this is offline/simulated throughout.
- Not modeling actual partner revenue, commission structures, or negotiated exposure agreements — the fairness metric is a stated proxy (see `REQUIREMENTS.md` FR-4.4).
- Not targeting distributed/multi-GPU training — everything is scoped to run on a single machine by design (see `REQUIREMENTS.md` NFR-2).
