# AGENTS.md

Operational instructions for Codex working in this repository. `REQUIREMENTS.md` and `ROADMAP.md` are the specs — this file is *how* to execute against them. Read this file first, every session.

## Project

`learning-to-rank-distillation` — a dataset-agnostic toolkit for ranking-model distillation (teacher → student) and marketplace-aware fairness reranking. Full spec: `REQUIREMENTS.md` (v1.0 scope). Future scope: `ROADMAP.md` — do not build anything from `ROADMAP.md` unless explicitly asked; it exists to show direction, not as pending work.

## Environment & Setup

- Python 3.11+. Dependency manager: `pip` with `pyproject.toml`.
- Install: `pip install -e ".[dev]"`
- Run tests: `pytest tests/ -v`
- Run the full v1.0 benchmark pipeline: `python -m learning_to_rank_distillation.benchmark.run_all`
- Lint/format before considering any task done: `ruff check . && ruff format --check .`

### Finalized Tech Stack — Do Not Substitute

This stack is locked. If a task seems to call for a different library (a different DL framework, a different ANN library, a different plotting library), stop and flag it rather than substituting — these were deliberate choices, not defaults to override for convenience.

```
python  = ">=3.11"
pandas  = ">=2.2"
numpy   = ">=1.26"
lightgbm     = ">=4.3"   # Teacher (LambdaMART) — FR-1
torch        = ">=2.2"   # Student two-tower — FR-2.1
faiss-cpu    = ">=1.8"   # ANN retrieval — FR-2.1 (CPU-only, see REQUIREMENTS.md NFR-2)
scikit-learn = ">=1.4"   # Query-grouped splitting, metrics
matplotlib   = ">=3.8"   # Pareto plots — FR-3.2, FR-4.3 (static; Tier 5 owns interactive)
pytest  = ">=8.0"
ruff    = ">=0.4"
```

SQLite (governance registry, FR-5.3) is stdlib — no dependency needed. Full rationale for each pin is in `REQUIREMENTS.md` NFR-1.

## Network Access — Scoped On, For Data Acquisition Only

By default this environment has no network access. For this project, network access is deliberately enabled but restricted to a small domain allowlist, so Codex can attempt to fetch the RecTour dataset itself without being able to reach anything else.

Add to `config.toml` (local CLI) before starting a session where you want Codex to attempt the download:

```toml
[sandbox_workspace_write]
network_access = true

# If using the newer permission-profile syntax instead:
[permissions.project-edit.network]
enabled = true
mode = "limited"

[permissions.project-edit.network.domains]
"web.ec.tuwien.ac.at" = "allow"
"ceur-ws.org" = "allow"
"medium.com" = "allow"
"expediagroup.com" = "allow"
"github.com" = "allow"
"raw.githubusercontent.com" = "allow"
```

Adjust the allowlist once the actual hosting domain is confirmed (see Data Acquisition below) — start narrow, widen only as needed, and never set `mode = "full"` for this project.

## Data Acquisition — Let Codex Attempt This First, With a Hard Stop

If `data/rectour/` is empty, before falling back to the synthetic fixture, Codex should attempt the following, **in order, stopping immediately at whichever step resolves it:**

1. **Check for an existing local copy first.** Search common locations (`~/Downloads`, `~/Desktop`, any path the user mentioned) for files that look like the RecTour dataset before assuming network access is needed at all. If found, copy (don't move) into `data/rectour/` and proceed.
2. **Research the current access path.** Fetch the known reference pages — the RecTour 2021 workshop page (`web.ec.tuwien.ac.at`), the dataset documentation PDF (`ceur-ws.org`), and Expedia Group's technology blog (`medium.com`, `expediagroup.com`) — to find the current, authoritative download or access-request instructions. Do not assume a URL structure or guess a direct file link that hasn't been confirmed on one of these pages.
3. **If a direct, unauthenticated download link is found:** download it into `data/rectour/`, verify the file isn't an HTML error/login page (check content-type and file size sanity before treating it as data), and report what was downloaded and from where.
4. **If access requires a request form, registration, login, or any personal/identifying information (name, email, affiliation, "intended use" description):** **stop immediately.** Do not fill out or submit the form. Do not invent placeholder identity details to get past it. Report back exactly what the form requires and the URL to it, so the user can complete that step themselves. Then proceed with the synthetic fixture (see below) until real files appear in `data/rectour/`.
5. **If nothing conclusive is found after steps 1–4:** report that clearly rather than guessing, and fall back to the synthetic fixture.

This mirrors the same principle as Working Rule #2 below (ask, don't invent) — applied specifically to data acquisition, where inventing a wrong answer means silently training on the wrong data or, worse, submitting a form with fabricated personal information.

## Data Access — Read Before Touching Anything in `adapters/`

**Network access for anything outside the allowlist above is still off.** Do not attempt to reach any Kaggle URL or any domain not explicitly listed. Once `data/rectour/` is populated (by Codex per the steps above, or manually by the user), two paths apply:

1. **If `data/rectour/` already contains files** (the user placed them there manually before starting the session): read the actual files present and infer the schema from them directly — column names, dtypes, sample values. Do not rely on the field mapping table in `REQUIREMENTS.md` Section 2.1 as authoritative; that table is best-effort from documentation review, not a confirmed schema. If a column referenced in `REQUIREMENTS.md` doesn't exist in the actual file, or an unlisted column looks relevant, **stop and report the discrepancy** rather than silently adapting the mapping.
2. **If `data/rectour/` is empty or missing:** do not fabricate a schema from memory. Build and test against `tests/fixtures/synthetic_ranking_data.py` instead (create it if it doesn't exist — a small synthetic generator matching the `RankingExample` schema in `REQUIREMENTS.md` Section 1, with made-up but structurally valid query/item/feature/label data). Every adapter, model, and pipeline component should be developed and unit-tested against this synthetic fixture first. Wire in the real RecTour adapter last, and only once real data files are actually present in `data/rectour/`.

## Working Rules

1. **One milestone per session, in order.** Work through `REQUIREMENTS.md` Section 11 (M1 → M5) sequentially. Do not start M2 until M1's exit criteria are met. State which milestone you're starting and which you completed at the end of each session.
2. **Ask, don't invent.** If a requirement is ambiguous, a data field is undocumented, or a design decision in `REQUIREMENTS.md` conflicts with what the actual data looks like — stop and ask, in a single clear question, rather than picking a plausible-sounding default and continuing. This applies especially to: the RecTour field mapping (Section 2.1), the `is_drr` flag's meaning, and whether an equivalent to the `random_bool` position-bias flag exists in the actual dataset (DR-1).
3. **The no-KD baseline (FR-2.3) is not optional and not decorative.** Do not skip it or stub it out even under time pressure — it's the control that makes every other distillation result meaningful. If forced to cut scope, cut something else first.
4. **No dataset-specific logic outside `adapters/`.** If you find yourself writing an `if dataset == "rectour"` branch anywhere in `models/`, `distillation/`, `fairness/`, or `benchmark/`, stop — that logic belongs in the adapter, per `REQUIREMENTS.md` Section 1 and NFR-3. This rule exists specifically so a second dataset adapter is cheap to add later; don't quietly violate it for short-term convenience.
5. **Every FR/DR/NFR ID in `REQUIREMENTS.md` should be traceable to a specific file or function.** When you implement FR-2.2, put `# Implements FR-2.2` (or equivalent) near the relevant code. This makes the Definition of Done checklist (Section 12) actually verifiable later, by a human or another agent.
6. **Write tests as you go, not at the end.** Each module in the repo structure (`REQUIREMENTS.md` Section 10) should have a corresponding test file. Do not defer testing to a final "add tests" pass.
7. **Commit messages reference the requirement ID** they implement, e.g. `feat: implement response-based distillation loss (FR-2.2)`.

## What "Done" Means for a Session

Before ending a session, confirm:
- [ ] Code runs (`pytest` passes, benchmark script doesn't error)
- [ ] Lint/format clean
- [ ] The specific `REQUIREMENTS.md` milestone's exit criteria are met, or you've stated explicitly what's incomplete and why
- [ ] Any open questions from Working Rule #2 are surfaced clearly in your final summary, not buried mid-session

## Explicitly Out of Scope for Codex to Decide Alone

- Adding `ROADMAP.md` (Tier 2+) features without being asked
- Changing the repo name, package name, or public API surface
- Choosing a different dataset or adding a second adapter before v1.0 (M1–M5) is complete
- Modifying the promotion gate thresholds in FR-5.2 without flagging the change explicitly (these are business-meaningful numbers, not just config)
- **Submitting any data-access request form, registering an account, or providing identity/contact information on the user's behalf** — this is a hard stop per the Data Acquisition section above, not a judgment call to make case-by-case
- Widening the network domain allowlist beyond what's listed in this file without flagging it first
