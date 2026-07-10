"""Interactive Pareto dashboard for generated benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path("artifacts")


def main() -> None:
    st.set_page_config(page_title="LTRD Pareto Dashboard", layout="wide")
    st.title("Relevance vs. Exposure Fairness")

    pareto = _load_rows(ARTIFACT_DIR / "fairness_pareto_frontier.json")
    tradeoff = _load_rows(ARTIFACT_DIR / "fairness_tradeoff.json")
    if pareto.empty and tradeoff.empty:
        st.warning("Run `make benchmark` first to generate fairness artifacts.")
        return

    left, right = st.columns([2, 1])
    with left:
        if not pareto.empty:
            st.subheader("Scalarized Pareto Search")
            st.scatter_chart(
                pareto,
                x="low_exposure_impression_share",
                y="ndcg_at_5",
                color="is_pareto_efficient",
            )
        if not tradeoff.empty:
            st.subheader("Exposure-Floor Sweep")
            st.line_chart(
                tradeoff,
                x="low_exposure_impression_share",
                y="ndcg_at_5",
            )

    with right:
        if not tradeoff.empty:
            selected_floor = st.slider(
                "Exposure floor",
                min_value=float(tradeoff["exposure_floor"].min()),
                max_value=float(tradeoff["exposure_floor"].max()),
                value=float(tradeoff["exposure_floor"].median()),
                step=0.1,
            )
            selected = tradeoff.iloc[
                (tradeoff["exposure_floor"] - selected_floor).abs().argsort()[:1]
            ]
            st.metric("NDCG@5", f"{float(selected['ndcg_at_5'].iloc[0]):.4f}")
            st.metric(
                "Low-exposure share",
                f"{float(selected['low_exposure_impression_share'].iloc[0]):.4f}",
            )
            st.metric("Exposure Gini@5", f"{float(selected['exposure_gini_at_5'].iloc[0]):.4f}")
        if not pareto.empty:
            st.subheader("Non-dominated points")
            st.dataframe(pareto[pareto["is_pareto_efficient"]], hide_index=True)


def _load_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
