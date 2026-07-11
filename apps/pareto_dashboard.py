"""Decision-oriented dashboard for ranking quality and exposure fairness artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ARCHITECTURE_DIAGRAM_PATH = Path("docs/architecture_diagram.png")


@dataclass(frozen=True, slots=True)
class ArtifactSet:
    label: str
    path: Path
    evidence_level: str
    description: str
    caveat: str
    command: str


ARTIFACT_SETS = (
    ArtifactSet(
        label="Amazon ESCI local benchmark",
        path=Path("artifacts/esci"),
        evidence_level="Primary real-data flow",
        description=(
            "Uses the Amazon Shopping Queries ESCI relevance data. This is the project primary "
            "public-data benchmark path."
        ),
        caveat=(
            "The local ESCI run can merge product metadata when the ignored products parquet is "
            "present. Without that file, the adapter still runs with examples, sources, locale, "
            "query fields, and stable product-id hash features."
        ),
        command="make benchmark-esci",
    ),
    ArtifactSet(
        label="Synthetic smoke benchmark",
        path=Path("artifacts"),
        evidence_level="Fast deterministic smoke run",
        description=(
            "Uses generated marketplace-like data so tests, plots, and fairness logic can run "
            "without external data access."
        ),
        caveat=(
            "Use this for workflow verification and demos. Do not treat these numbers as real "
            "marketplace performance claims."
        ),
        command="make benchmark",
    ),
)

FAILURE_MODES = (
    (
        "The serving trap",
        "The strongest offline ranker may be too slow or too expensive to serve at p99 latency.",
    ),
    (
        "The silent exposure shift",
        "A relevance gain can quietly move top-rank visibility away from low-exposure suppliers.",
    ),
    (
        "Promotion by vibes",
        "A model should be promoted by executable evidence, not by a subjective "
        "release discussion.",
    ),
)

ARCHITECTURE_STAGES = (
    ("1", "Adapter", "Raw data becomes RankingExample rows."),
    ("2", "Teacher", "LightGBM LambdaMART optimizes ranking quality."),
    ("3", "Student", "Two-tower PyTorch model is shaped for serving."),
    ("4", "Benchmark", "NDCG, size, and latency are measured together."),
    ("5", "Fairness", "Exposure trade-offs are swept and plotted."),
    ("6", "Promotion", "A gate records whether the candidate is defensible."),
)

METRIC_GLOSSARY = (
    ("NDCG@5", "Top-5 ranking quality. Higher means more relevant items near the top."),
    ("p99 latency", "Worst-case-ish local scoring latency. Lower is better for serving."),
    ("Model size", "Approximate artifact footprint. Smaller is easier to package and ship."),
    (
        "Low-exposure share",
        "Share of top-5 impressions assigned to groups that historically had lower exposure.",
    ),
    (
        "Exposure Gini@5",
        "Top-5 exposure inequality. Lower means impressions are less concentrated.",
    ),
    (
        "Pareto-efficient",
        "A tested setting that is not dominated by another setting on both relevance and fairness.",
    ),
)


def main() -> None:
    st.set_page_config(
        page_title="LTRD Ranking Dashboard",
        page_icon="LTRD",
        layout="wide",
    )
    _configure_altair()
    _inject_css()

    config = _select_artifact_set()
    benchmark = _prepare_benchmark(_load_rows(config.path / "benchmark_table.json"))
    pareto = _prepare_pareto(_load_rows(config.path / "fairness_pareto_frontier.json"))
    tradeoff = _prepare_tradeoff(_load_rows(config.path / "fairness_tradeoff.json"))

    _render_header(config)
    _render_artifact_status(config, benchmark, pareto, tradeoff)
    _render_what_you_are_viewing(config)

    if benchmark.empty and pareto.empty and tradeoff.empty:
        st.warning(
            f"No dashboard artifacts were found under `{config.path}`. "
            f"Run `{config.command}` first."
        )
        return

    _render_decision_summary(config, benchmark, pareto, tradeoff)

    tabs = st.tabs(
        [
            "Start Here",
            "Executive View",
            "Quality vs Latency",
            "Fairness Trade-off",
            "Artifact Tables",
            "How To Explain It",
        ]
    )
    with tabs[0]:
        _render_start_here(config)
    with tabs[1]:
        _render_executive_view(config, benchmark, pareto, tradeoff)
    with tabs[2]:
        _render_quality_latency(benchmark)
    with tabs[3]:
        _render_fairness_tradeoff(pareto, tradeoff)
    with tabs[4]:
        _render_tables(benchmark, pareto, tradeoff)
    with tabs[5]:
        _render_explanation_guide(config)


def _select_artifact_set() -> ArtifactSet:
    available = [artifact_set for artifact_set in ARTIFACT_SETS if artifact_set.path.exists()]
    if not available:
        available = list(ARTIFACT_SETS)

    labels = [artifact_set.label for artifact_set in available]
    with st.sidebar:
        st.header("Artifact Run")
        selected_label = st.selectbox("Dataset / run", labels, index=0)
        selected = next(
            artifact_set for artifact_set in available if artifact_set.label == selected_label
        )

        st.markdown(f"**Evidence level:** {selected.evidence_level}")
        st.caption(selected.description)
        st.markdown("**Artifact directory**")
        st.code(str(selected.path), language="text")
        st.markdown("**Regenerate this run**")
        st.code(selected.command, language="bash")
        st.info(selected.caveat)

    return selected


def _render_header(config: ArtifactSet) -> None:
    st.markdown(
        f"""
        <section class="hero">
          <p class="eyebrow">Learning-to-rank distillation</p>
          <h1>Ranking Quality and Exposure Fairness Dashboard</h1>
          <p>
            This dashboard turns benchmark artifacts into the same production question used in the
            README: is this model good enough, fast enough, and fair enough to serve, and can that
            decision be defended?
          </p>
          <div class="dataset-pill">{config.label}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_what_you_are_viewing(config: ArtifactSet) -> None:
    st.markdown(
        f"""
        <div class="context-card">
          <h3>What you are looking at</h3>
          <p>
            This is an <strong>offline model-decision dashboard</strong>, not a product search page.
            It reads benchmark JSON files from <code>{config.path}</code> and explains whether the
            ranking lifecycle produced a candidate that is relevant, servable, and exposure-aware.
          </p>
          <p class="muted">
            Current evidence level: <strong>{config.evidence_level}</strong>. Use the sidebar to
            switch between the Amazon ESCI local benchmark and the synthetic smoke benchmark.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_artifact_status(
    config: ArtifactSet,
    benchmark: pd.DataFrame,
    pareto: pd.DataFrame,
    tradeoff: pd.DataFrame,
) -> None:
    loaded = []
    missing = []
    for filename, frame in (
        ("benchmark_table.json", benchmark),
        ("fairness_pareto_frontier.json", pareto),
        ("fairness_tradeoff.json", tradeoff),
    ):
        if frame.empty:
            missing.append(filename)
        else:
            loaded.append(filename)

    if loaded:
        st.caption(f"Loaded from `{config.path}`: {', '.join(loaded)}")
    if missing:
        st.warning(f"Missing or empty artifact files: {', '.join(missing)}")


def _render_start_here(config: ArtifactSet) -> None:
    st.subheader("Why This Dashboard Exists")
    st.markdown(
        """
        The README frames the project as a governed ranking lifecycle. The UI follows that same
        logic: it is designed to help a reviewer understand the ranking decision, not only inspect
        raw charts.
        """
    )

    st.markdown(
        """
        <div class="question-card">
          <p class="eyebrow">Production question</p>
          <h3>Is this model good enough, fast enough, and fair enough to serve?</h3>
          <p>
            A useful ranking system has to defend all three dimensions at the same time. Strong
            offline relevance is not enough if the candidate is slow, shifts exposure silently, or
            cannot pass a promotion policy.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Failure Modes The UI Is Meant To Catch")
    _render_failure_modes()

    st.subheader("Architecture Flow Behind The Numbers")
    _render_architecture_diagram()
    _render_architecture_flow()

    st.subheader("Metric Glossary")
    _render_metric_glossary()

    st.info(
        f"You are currently viewing `{config.label}`. Regenerate this run with `{config.command}`."
    )


def _render_failure_modes() -> None:
    cols = st.columns(3)
    for column, (title, description) in zip(cols, FAILURE_MODES, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="info-card">
                  <h4>{title}</h4>
                  <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_architecture_flow() -> None:
    cards = "".join(
        (
            '<div class="stage-card">'
            f'<span class="stage-num">{number}</span>'
            f"<h4>{title}</h4>"
            f"<p>{description}</p>"
            "</div>"
        )
        for number, title, description in ARCHITECTURE_STAGES
    )
    st.markdown(f'<div class="stage-grid">{cards}</div>', unsafe_allow_html=True)


def _render_architecture_diagram() -> None:
    if not ARCHITECTURE_DIAGRAM_PATH.exists():
        st.info(f"Architecture diagram not found at `{ARCHITECTURE_DIAGRAM_PATH}`.")
        return
    st.image(
        str(ARCHITECTURE_DIAGRAM_PATH),
        caption=(
            "System architecture: adapters normalize datasets into RankingExample rows; teacher "
            "and student models feed benchmark, fairness, governance, and serving layers."
        ),
        width="stretch",
    )


def _render_metric_glossary() -> None:
    rows = pd.DataFrame(METRIC_GLOSSARY, columns=["metric", "plain_english_meaning"])
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_decision_summary(
    config: ArtifactSet,
    benchmark: pd.DataFrame,
    pareto: pd.DataFrame,
    tradeoff: pd.DataFrame,
) -> None:
    best_model = _best_model(benchmark)
    fastest_model = _fastest_model(benchmark)
    recommendation = _recommended_pareto_point(pareto)
    selected_floor = _selected_floor_row(tradeoff)

    cols = st.columns(4)
    with cols[0]:
        if best_model is None:
            st.metric("Best model NDCG@5", "n/a")
        else:
            st.metric(
                "Best model NDCG@5",
                _fmt(best_model["ndcg_at_5"]),
                help="Highest model quality in the benchmark table. Higher is better.",
            )
            st.caption(str(best_model["model"]))
    with cols[1]:
        if fastest_model is None:
            st.metric("Fastest p99 latency", "n/a")
        else:
            st.metric(
                "Fastest p99 latency",
                f"{float(fastest_model['latency_p99_ms']):.1f} ms",
                help="Lowest p99 scoring latency in the benchmark table. Lower is better.",
            )
            st.caption(str(fastest_model["model"]))
    with cols[2]:
        if recommendation is None:
            st.metric("Recommended fairness weight", "n/a")
        else:
            st.metric(
                "Recommended fairness weight",
                _fmt_weight(recommendation["fairness_weight"]),
                help="Pareto-efficient fairness setting selected from the scalarized search.",
            )
            st.caption(
                f"share {_fmt(recommendation['low_exposure_impression_share'])}, "
                f"NDCG {_fmt(recommendation['ndcg_at_5'])}"
            )
    with cols[3]:
        if selected_floor is None:
            st.metric("Selected exposure floor", "n/a")
        else:
            st.metric(
                "Selected exposure floor",
                _fmt(selected_floor["exposure_floor"]),
                help="Middle tested exposure-floor setting from the constrained rerank sweep.",
            )
            st.caption(
                f"Gini {_fmt(selected_floor['exposure_gini_at_5'])}, "
                f"share {_fmt(selected_floor['low_exposure_impression_share'])}"
            )

    summary = _fairness_delta_summary(pareto)
    if summary:
        st.markdown(
            f"""
            <div class="decision-card">
              <h3>Decision readout</h3>
              <p>{summary}</p>
              <p class="muted">
                Evidence context: <strong>{config.evidence_level}</strong>. The dashboard is an
                offline evaluation surface, not an online A/B-test result.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_executive_view(
    config: ArtifactSet,
    benchmark: pd.DataFrame,
    pareto: pd.DataFrame,
    tradeoff: pd.DataFrame,
) -> None:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("What This Run Says")
        st.markdown(
            """
            The ranking system is evaluated on three axes:

            - **Relevance:** does the model rank useful items near the top?
            - **Serving cost:** can the model score within a reasonable p99 latency budget?
            - **Exposure fairness:** do low-exposure groups receive meaningful top-k visibility?
            """
        )
        if benchmark.empty:
            st.info("No benchmark table is available for this artifact set.")
        else:
            best_student = _best_student(benchmark)
            teacher = _teacher_model(benchmark)
            if best_student is not None and teacher is not None:
                st.markdown(
                    f"""
                    <div class="explain-box">
                      <strong>Teacher vs. student:</strong>
                      the strongest student in this run is <code>{best_student["model"]}</code>
                      with NDCG@5 <strong>{_fmt(best_student["ndcg_at_5"])}</strong> and p99
                      latency <strong>{float(best_student["latency_p99_ms"]):.1f} ms</strong>.
                      The teacher has NDCG@5 <strong>{_fmt(teacher["ndcg_at_5"])}</strong> and p99
                      latency <strong>{float(teacher["latency_p99_ms"]):.1f} ms</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("How To Use The Screen")
        st.markdown(
            """
            1. Pick an artifact run from the sidebar.
            2. Check whether a student approaches teacher quality.
            3. Inspect Pareto-efficient fairness points.
            4. Prefer settings that improve exposure with small NDCG loss.
            5. Treat synthetic runs as workflow evidence, not business evidence.
            """
        )
        st.info(config.caveat)

    if not pareto.empty:
        st.subheader("Recommended Operating Point")
        _render_recommendation_table(pareto)
    if not tradeoff.empty and _is_flat_tradeoff(tradeoff):
        st.info(
            "The exposure-floor sweep is flat for this run. That means all tested floors produced "
            "the same top-k allocation, either because the baseline already satisfies those floors "
            "or because the tested candidate lists do not allow more movement."
        )


def _render_quality_latency(benchmark: pd.DataFrame) -> None:
    st.subheader("Quality vs. Serving Cost")
    st.markdown(
        """
        Higher on the chart means better relevance. Further left means lower p99 scoring latency.
        A serving candidate should ideally move toward the upper-left: strong quality with lower
        latency and smaller model size.
        """
    )
    if benchmark.empty:
        st.warning("No benchmark table is available.")
        return

    chart = _quality_latency_chart(benchmark)
    st.altair_chart(chart, width="stretch")

    best_student = _best_student(benchmark)
    if best_student is not None:
        st.success(
            "Best student candidate: "
            f"`{best_student['model']}` with NDCG@5 {_fmt(best_student['ndcg_at_5'])}, "
            f"NDCG@10 {_fmt(best_student['ndcg_at_10'])}, and p99 latency "
            f"{float(best_student['latency_p99_ms']):.1f} ms."
        )


def _render_fairness_tradeoff(pareto: pd.DataFrame, tradeoff: pd.DataFrame) -> None:
    st.subheader("Fairness Trade-off")
    st.markdown(
        """
        The Pareto chart compares ranking quality with exposure share for low-exposure groups.
        Points marked Pareto-efficient are the settings where no other tested setting is better on
        both dimensions.
        """
    )

    if pareto.empty and tradeoff.empty:
        st.warning("No fairness artifacts are available.")
        return

    if not pareto.empty:
        st.altair_chart(_pareto_chart(pareto), width="stretch")

    if not tradeoff.empty:
        floors = sorted(float(value) for value in tradeoff["exposure_floor"].unique())
        selected_floor = st.select_slider(
            "Inspect exposure floor",
            options=floors,
            value=floors[len(floors) // 2],
            format_func=lambda value: f"{value:.2f}",
        )
        selected = _row_nearest(tradeoff, "exposure_floor", selected_floor)

        cols = st.columns(3)
        cols[0].metric("NDCG@5 at floor", _fmt(selected["ndcg_at_5"]))
        cols[1].metric(
            "Low-exposure top-5 share",
            _fmt(selected["low_exposure_impression_share"]),
        )
        cols[2].metric("Exposure Gini@5", _fmt(selected["exposure_gini_at_5"]))

        left, right = st.columns(2)
        with left:
            st.altair_chart(
                _tradeoff_line(
                    tradeoff,
                    y_field="ndcg_at_5",
                    y_title="Ranking quality (NDCG@5)",
                    color="#315a9a",
                ),
                width="stretch",
            )
        with right:
            st.altair_chart(
                _tradeoff_line(
                    tradeoff,
                    y_field="low_exposure_impression_share",
                    y_title="Low-exposure top-5 share",
                    color="#0b6f6a",
                ),
                width="stretch",
            )


def _render_tables(
    benchmark: pd.DataFrame,
    pareto: pd.DataFrame,
    tradeoff: pd.DataFrame,
) -> None:
    st.subheader("Benchmark Table")
    if benchmark.empty:
        st.info("No benchmark rows loaded.")
    else:
        st.dataframe(
            benchmark[
                [
                    "model",
                    "model_family",
                    "ndcg_at_5",
                    "ndcg_at_10",
                    "latency_p50_ms",
                    "latency_p99_ms",
                    "model_size_kb",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

    st.subheader("Pareto Search Rows")
    if pareto.empty:
        st.info("No Pareto rows loaded.")
    else:
        st.dataframe(
            pareto[
                [
                    "fairness_weight",
                    "ndcg_at_5",
                    "low_exposure_impression_share",
                    "exposure_gini_at_5",
                    "is_pareto_efficient",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

    st.subheader("Exposure-Floor Sweep Rows")
    if tradeoff.empty:
        st.info("No trade-off rows loaded.")
    else:
        st.dataframe(tradeoff, hide_index=True, width="stretch")


def _render_explanation_guide(config: ArtifactSet) -> None:
    st.subheader("Plain-English Explanation")
    st.markdown(
        f"""
        This dashboard is for explaining the project as a production-shaped ranking decision system.
        It is currently showing **{config.label}**.

        **Business problem:** a marketplace wants relevant search results, but also needs to avoid
        silently shifting all top-rank exposure toward already-dominant suppliers or sellers.

        **Architecture answer:** train a strong teacher ranker, distill a smaller student for
        serving, then evaluate the student with relevance, latency, size, and exposure fairness
        together.

        **How to read a good result:**

        - A model with high `NDCG@5` ranks relevant items near the top.
        - A model with low `p99 latency` is more plausible to serve.
        - A fairness setting with higher low-exposure share gives more top-k visibility to groups
          that historically received less exposure.
        - A lower exposure Gini means impressions are less concentrated.
        - A Pareto-efficient point is a serious candidate because another tested point does not
          dominate it on both relevance and fairness.

        **What this does not prove:** it is not a live traffic result, not an A/B test, and not a
        full business fairness policy. It is an offline decision surface that makes the model trade-
        offs explicit and reviewable.
        """
    )


def _render_recommendation_table(pareto: pd.DataFrame) -> None:
    baseline = _baseline_pareto_point(pareto)
    recommended = _recommended_pareto_point(pareto)
    if baseline is None or recommended is None:
        st.info("No recommendation is available.")
        return

    rows = pd.DataFrame(
        [
            {
                "setting": "No fairness bonus",
                "fairness_weight": baseline["fairness_weight"],
                "ndcg_at_5": baseline["ndcg_at_5"],
                "low_exposure_share": baseline["low_exposure_impression_share"],
                "exposure_gini_at_5": baseline["exposure_gini_at_5"],
            },
            {
                "setting": "Recommended Pareto point",
                "fairness_weight": recommended["fairness_weight"],
                "ndcg_at_5": recommended["ndcg_at_5"],
                "low_exposure_share": recommended["low_exposure_impression_share"],
                "exposure_gini_at_5": recommended["exposure_gini_at_5"],
            },
        ]
    )
    st.dataframe(rows, hide_index=True, width="stretch")


def _quality_latency_chart(frame: pd.DataFrame) -> alt.Chart:
    points = (
        alt.Chart(frame)
        .mark_circle(size=150, opacity=0.9)
        .encode(
            x=alt.X(
                "latency_p99_ms:Q",
                title="p99 scoring latency (ms, lower is better)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "ndcg_at_5:Q",
                title="Ranking quality (NDCG@5, higher is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("model_family:N", title="Model type"),
            size=alt.Size("model_size_kb:Q", title="Model size (KB)", legend=None),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("model_family:N", title="Type"),
                alt.Tooltip("ndcg_at_5:Q", title="NDCG@5", format=".4f"),
                alt.Tooltip("ndcg_at_10:Q", title="NDCG@10", format=".4f"),
                alt.Tooltip("latency_p99_ms:Q", title="p99 latency", format=".2f"),
                alt.Tooltip("model_size_kb:Q", title="Model size KB", format=".1f"),
            ],
        )
    )
    labels = points.mark_text(align="left", dx=9, dy=-7, fontSize=12).encode(
        text=alt.Text("short_model:N"),
        size=alt.value(12),
    )
    return (points + labels).properties(height=420).interactive()


def _pareto_chart(frame: pd.DataFrame) -> alt.Chart:
    points = (
        alt.Chart(frame)
        .mark_circle(size=145, opacity=0.92)
        .encode(
            x=alt.X(
                "low_exposure_impression_share:Q",
                title="Low-exposure top-5 impression share (higher is fairer)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "ndcg_at_5:Q",
                title="Ranking quality (NDCG@5, higher is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "pareto_status:N",
                title="Candidate status",
                scale=alt.Scale(
                    domain=["Pareto-efficient", "Dominated"],
                    range=["#0b6f6a", "#8da2c0"],
                ),
            ),
            tooltip=[
                alt.Tooltip("fairness_weight:Q", title="Fairness weight", format=".2f"),
                alt.Tooltip("ndcg_at_5:Q", title="NDCG@5", format=".4f"),
                alt.Tooltip(
                    "low_exposure_impression_share:Q",
                    title="Low-exposure share",
                    format=".4f",
                ),
                alt.Tooltip("exposure_gini_at_5:Q", title="Exposure Gini@5", format=".4f"),
                alt.Tooltip("pareto_status:N", title="Status"),
            ],
        )
    )
    labels = points.mark_text(align="left", dx=9, dy=-7, fontSize=12).encode(
        text=alt.Text("fairness_weight_label:N"),
        size=alt.value(12),
    )
    return (points + labels).properties(height=430).interactive()


def _tradeoff_line(frame: pd.DataFrame, *, y_field: str, y_title: str, color: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_line(point=True, color=color, strokeWidth=3)
        .encode(
            x=alt.X("exposure_floor:Q", title="Exposure floor"),
            y=alt.Y(f"{y_field}:Q", title=y_title, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("exposure_floor:Q", title="Exposure floor", format=".2f"),
                alt.Tooltip("ndcg_at_5:Q", title="NDCG@5", format=".4f"),
                alt.Tooltip(
                    "low_exposure_impression_share:Q",
                    title="Low-exposure share",
                    format=".4f",
                ),
                alt.Tooltip("exposure_gini_at_5:Q", title="Exposure Gini@5", format=".4f"),
            ],
        )
        .properties(height=300)
    )


def _load_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def _prepare_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["model_family"] = frame["model"].map(_model_family)
    frame["short_model"] = frame["model"].map(_short_model)
    frame["model_size_kb"] = frame["model_size_bytes"] / 1024.0
    return frame.sort_values(["model_family", "model"]).reset_index(drop=True)


def _prepare_pareto(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["is_pareto_efficient"] = frame["is_pareto_efficient"].astype(bool)
    frame["pareto_status"] = frame["is_pareto_efficient"].map(
        {True: "Pareto-efficient", False: "Dominated"}
    )
    frame["fairness_weight_label"] = frame["fairness_weight"].map(
        lambda value: f"w={_fmt_weight(value)}"
    )
    return frame.sort_values("fairness_weight").reset_index(drop=True)


def _prepare_tradeoff(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values("exposure_floor").reset_index(drop=True)


def _best_model(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.loc[frame["ndcg_at_5"].idxmax()]


def _fastest_model(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.loc[frame["latency_p99_ms"].idxmin()]


def _teacher_model(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    teacher = frame[frame["model"].str.startswith("teacher")]
    if teacher.empty:
        return None
    return teacher.iloc[0]


def _best_student(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    students = frame[frame["model"].str.startswith("student")]
    if students.empty:
        return None
    return students.loc[students["ndcg_at_5"].idxmax()]


def _baseline_pareto_point(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.sort_values("fairness_weight").iloc[0]


def _recommended_pareto_point(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    baseline = _baseline_pareto_point(frame)
    if baseline is None:
        return None

    efficient = frame[frame["is_pareto_efficient"]]
    candidates = efficient if not efficient.empty else frame

    quality_floor = float(baseline["ndcg_at_5"]) * 0.98
    close_quality = candidates[candidates["ndcg_at_5"] >= quality_floor]
    candidates = close_quality if not close_quality.empty else candidates

    return candidates.sort_values(
        ["low_exposure_impression_share", "ndcg_at_5", "fairness_weight"],
        ascending=[False, False, False],
    ).iloc[0]


def _selected_floor_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    floors = sorted(float(value) for value in frame["exposure_floor"].unique())
    return _row_nearest(frame, "exposure_floor", floors[len(floors) // 2])


def _row_nearest(frame: pd.DataFrame, column: str, value: float) -> pd.Series:
    index = (frame[column] - value).abs().idxmin()
    return frame.loc[index]


def _fairness_delta_summary(frame: pd.DataFrame) -> str:
    baseline = _baseline_pareto_point(frame)
    recommended = _recommended_pareto_point(frame)
    if baseline is None or recommended is None:
        return ""

    ndcg_delta = float(recommended["ndcg_at_5"] - baseline["ndcg_at_5"])
    share_delta = float(
        recommended["low_exposure_impression_share"] - baseline["low_exposure_impression_share"]
    )
    gini_delta = float(recommended["exposure_gini_at_5"] - baseline["exposure_gini_at_5"])

    if share_delta > 0 and ndcg_delta >= -0.01:
        decision = "This is a credible fairness operating point for review."
    elif share_delta > 0:
        decision = "This improves exposure, but the relevance cost needs explicit review."
    else:
        decision = (
            "This run does not show a meaningful exposure improvement from the tested search."
        )

    return (
        f"Compared with no fairness bonus, the recommended Pareto point uses "
        f"<strong>fairness_weight={_fmt_weight(recommended['fairness_weight'])}</strong>, changes "
        f"NDCG@5 by <strong>{_fmt_signed(ndcg_delta)}</strong>, changes low-exposure top-5 share "
        f"by <strong>{_fmt_signed(share_delta)}</strong>, and changes exposure Gini@5 by "
        f"<strong>{_fmt_signed(gini_delta)}</strong>. {decision}"
    )


def _is_flat_tradeoff(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    columns = ["ndcg_at_5", "low_exposure_impression_share", "exposure_gini_at_5"]
    return all(frame[column].round(8).nunique() <= 1 for column in columns)


def _model_family(model: str) -> str:
    if model.startswith("teacher"):
        return "Teacher"
    if "no-kd" in model:
        return "Student baseline"
    if model.startswith("student-kd"):
        return "Distilled student"
    return "Model"


def _short_model(model: str) -> str:
    return (
        model.replace("teacher-", "teacher ")
        .replace("student-", "")
        .replace("lightgbm", "LightGBM")
        .replace("no-kd", "no KD")
        .replace("kd-", "KD ")
    )


def _fmt(value: object) -> str:
    return f"{float(value):.4f}"


def _fmt_weight(value: object) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:.2f}"


def _fmt_signed(value: float) -> str:
    return f"{value:+.4f}"


def _configure_altair() -> None:
    alt.data_transformers.disable_max_rows()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
          max-width: 1800px;
          padding-top: 2rem;
        }
        .hero {
          border: 1px solid #d5dde6;
          border-radius: 10px;
          background: linear-gradient(135deg, #111827 0%, #172033 55%, #0b514e 100%);
          color: #f8fafc;
          padding: 28px 30px;
          margin-bottom: 18px;
        }
        .hero h1 {
          margin: 0;
          font-size: 2.35rem;
          line-height: 1.08;
          letter-spacing: 0;
        }
        .hero p {
          max-width: 900px;
          margin: 12px 0 0;
          color: #dbe6ef;
          font-size: 1.02rem;
        }
        .eyebrow {
          margin: 0 0 10px !important;
          color: #9bd7d1 !important;
          font-size: 0.78rem !important;
          font-weight: 800 !important;
          letter-spacing: 0.08em !important;
          text-transform: uppercase !important;
        }
        .dataset-pill {
          display: inline-flex;
          margin-top: 18px;
          min-height: 30px;
          align-items: center;
          border: 1px solid rgba(255,255,255,0.24);
          border-radius: 999px;
          padding: 4px 12px;
          background: rgba(255,255,255,0.08);
          color: #ffffff;
          font-weight: 750;
        }
        .decision-card,
        .explain-box,
        .context-card,
        .question-card,
        .info-card,
        .stage-card {
          border: 1px solid #d5dde6;
          border-radius: 8px;
          background: #ffffff;
        }
        .decision-card,
        .explain-box,
        .context-card,
        .question-card {
          border-left: 4px solid #0b6f6a;
          padding: 16px 18px;
          margin: 16px 0;
        }
        .decision-card h3 {
          margin: 0 0 8px;
          font-size: 1.05rem;
        }
        .context-card h3,
        .question-card h3 {
          margin: 0 0 8px;
          font-size: 1.12rem;
        }
        .decision-card p,
        .explain-box p,
        .context-card p,
        .question-card p,
        .info-card p,
        .stage-card p {
          margin: 0;
        }
        .question-card {
          background: #eef7f5;
        }
        .info-card {
          min-height: 150px;
          padding: 16px;
          border-top: 4px solid #315a9a;
        }
        .info-card h4,
        .stage-card h4 {
          margin: 0 0 8px;
          font-size: 1rem;
        }
        .info-card p,
        .stage-card p {
          color: #5c6875;
          font-size: 0.93rem;
        }
        .stage-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(220px, 1fr));
          gap: 14px;
          margin-top: 14px;
        }
        .stage-card {
          min-height: 150px;
          padding: 14px;
          position: relative;
        }
        .stage-card .stage-num {
          display: inline-flex;
          width: 28px;
          height: 28px;
          align-items: center;
          justify-content: center;
          margin-bottom: 10px;
          border-radius: 6px;
          background: #dcefeb;
          color: #0b514e;
          font-weight: 800;
        }
        @media (max-width: 980px) {
          .stage-grid {
            grid-template-columns: repeat(2, minmax(220px, 1fr));
          }
        }
        @media (max-width: 640px) {
          .stage-grid {
            grid-template-columns: 1fr;
          }
        }
        .muted {
          color: #5c6875;
          margin-top: 8px !important;
        }
        code {
          border-radius: 5px;
          background: #edf1f5;
          padding: 2px 5px;
          color: #243247;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
