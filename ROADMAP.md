# Roadmap — `learning-to-rank-distillation`

`README.md` covers what v1.0 (Tier 1) does today. This document is the public statement of where the project goes next — it's part of the deliverable, not a someday-list, because a roadmap is what signals an ongoing tool rather than a one-off interview artifact.

Tiers are ordered by dependency, not strictly by priority — see **"Suggested Build Order"** at the bottom for where to actually spend time first.

---

## Current Baseline

Tier 1 is implemented against deterministic synthetic ranking data while the primary real-data flow
is being moved to Amazon ESCI:

- Shared `RankingExample` schema and guarded RecTour adapter.
- RecTour-like synthetic generator enriched with 2013 ICDM Expedia feature-family ideas.
- LightGBM LambdaMART teacher.
- PyTorch two-tower student with response-based KD and no-KD control.
- Benchmark table, quality-vs-latency plot, exposure-fairness plot, and SQLite promotion log.

Amazon ESCI should become the primary real-data benchmark because it is publicly available and has
the same query-candidate-relevance shape needed for learning-to-rank. Expedia RecTour remains the
secondary/travel-marketplace target domain; its adapter stays guarded until real files are available.
The official RecTour workshop Dropbox link appears unavailable, so real RecTour schema/version,
real RecTour benchmark claims, and dataset download date remain intentionally unclaimed.

---

## Tier 1.5 — Repo Readiness

Goal: make the existing foundation easy to clone, test, review, and run on Amazon ESCI before adding
new research scope.

- [x] **Amazon ESCI primary adapter.** Add `adapters/esci.py`, `data/esci/.gitkeep`, tests, and field mapping from ESCI query/product rows into `RankingExample`.
- [x] **Configurable dataset switching.** Make `synthetic`, `esci`, and `rectour` selectable in CLI and benchmark paths, with no dataset-specific logic outside `adapters/`.
- [ ] **Primary benchmark refresh.** Produce a real-data benchmark table on Amazon ESCI and keep synthetic as a fast smoke fixture; keep RecTour benchmark claims disabled until real files are present.
- [ ] **Initialize and publish the repository.** Local git is initialized and the first commit exists; pushing to GitHub and verifying from a fresh clone are still pending.
- [x] **CI smoke checks.** Add GitHub Actions for lint, tests, and a small benchmark smoke run.
- [x] **Makefile.** Add `make install`, `make lint`, `make test`, `make benchmark`, and `make clean` so local usage is one-command.
- [x] **Artifact policy.** Decide which generated artifacts are committed (`benchmark_table.json`, plots) and which are ignored (`models/`, SQLite registry, raw data).
- [x] **Requirements checklist update.** Mark completed v1.0 items in `REQUIREMENTS.md` and leave only real-data-dependent items open.

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

- [ ] **Expedia RecTour as secondary adapter.** Keep the guarded RecTour adapter ready for real files and use it to validate the travel-marketplace target when data access is restored.
- [ ] **Third dataset adapter — MovieLens.** Fastest to add given how standard it is; useful as the "quickstart" example for anyone trying the toolkit for the first time.
- [ ] **Synthetic marketplace generator.** A configurable generator (supply concentration level, cold-start rate, exposure skew) so the fairness reranker can be stress-tested without needing any of the three real datasets — also the fastest way for someone else to try the fairness component without a data-access step at all.
- [ ] **Cross-dataset benchmark comparison.** One table showing teacher/student quality-vs-latency trade-offs across ESCI, synthetic, and RecTour when real RecTour files are available.

---

## Tier 4 — Production Shaping

Goal: move from research-shaped code to something that resembles what would actually get deployed.

- [ ] **Artifact bundle format.** Save/load a coherent bundle containing student model weights, vectorizer, feature schema, item embeddings/index metadata, training config, metrics, and data hash.
- [ ] **Real serving endpoint.** Student model behind FastAPI, item embeddings in a FAISS or ScaNN index, actual ANN retrieval rather than in-memory brute-force scoring.
- [ ] **ANN index lifecycle.** Separate job to build, validate, version, and atomically publish a compatible `model + vectorizer + index` bundle.
- [ ] **Load testing.** Locust or k6 load test producing real p50/p95/p99 latency numbers under simulated concurrent traffic, replacing the Tier 1 batch-scoring latency estimate.
- [ ] **Experiment tracking.** MLflow or Weights & Biases wired into training runs, replacing ad hoc script output.
- [ ] **Model registry integration.** Promotion gate (Tier 1/2) writes to a real model registry (MLflow Model Registry or equivalent) instead of a local SQLite log.
- [ ] **Runtime monitoring.** Emit OpenTelemetry/Prometheus-friendly metrics for latency, errors, empty results, feature missingness, score distribution, drift, and exposure fairness.
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

1. **Amazon ESCI primary flow** (Tier 1.5) — real-data adapter, CLI/benchmark dataset switch, and ESCI benchmark artifacts. This unblocks honest real-data claims.
2. **Repo readiness** (Tier 1.5) — first commit, CI smoke checks, Makefile, and artifact policy. This makes the current work reviewable.
3. **Feature-based + relation-based distillation** (Tier 2) — the most direct answer to "have you really done this," since response-based alone is the entry-level version of the technique.
4. **CI-enforced promotion gate** (Tier 2) — converts a governance claim into something a stranger can verify without asking you.
5. **Artifact bundle + serving endpoint** (Tier 4) — the highest-value production-grade step after the research baseline.
6. **Interactive Pareto dashboard** (Tier 5) — highest visibility-per-hour of anything on this list; it's what actually gets looked at.
7. **MovieLens adapter** (Tier 3) — useful for a tiny quickstart dataset after ESCI is working.
8. Everything else — genuinely valuable, but "ongoing project you keep improving" rather than "blocking anything."

---

## Non-Goals (For Now)

Stated explicitly so scope stays honest:

- Not attempting real-time online A/B testing against live traffic — this is offline/simulated throughout.
- Not modeling actual partner revenue, commission structures, or negotiated exposure agreements — the fairness metric is a stated proxy (see `REQUIREMENTS.md` FR-4.4).
- Not targeting distributed/multi-GPU training — everything is scoped to run on a single machine by design (see `REQUIREMENTS.md` NFR-2).
