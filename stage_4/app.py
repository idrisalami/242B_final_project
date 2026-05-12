"""Streamlit app for Stage 4."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from stage_4.analysis import (
    lambda_sweep_df,
    parameter_variations_df,
    stage_comparison_df,
    train_history_df,
)
from stage_4.pipeline import RecommendationPipeline, detect_artifacts, load_analysis_data
from stage_4.spotify import (
    get_spotify_client,
    parse_track_uris,
    resolve_track_aliases,
)


st.set_page_config(page_title="Spotify Recommender", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RecommendationPipeline()


@st.cache_resource(show_spinner=False)
def load_spotify():
    return get_spotify_client()


@st.cache_data(show_spinner=False)
def cached_track_aliases(uris: tuple[str, ...]):
    return resolve_track_aliases(load_spotify(), list(uris))


def show_status(status):
    cols = st.columns(3)
    cols[0].metric("Stage 1 ALS", "ready" if status.stage1_ready else "missing")
    cols[1].metric("Stage 2 SASRec", "ready" if status.stage2_ready else "fallback")
    cols[2].metric("Stage 3 MMR", "ready" if status.stage3_ready else "fallback")
    for msg in status.messages:
        st.caption(msg)


def show_input_preview(tracks):
    if not tracks:
        return
    st.subheader(f"Input playlist ({len(tracks)} tracks)")
    preview = tracks[:8]
    for row in preview:
        st.write(f"**{row['uri']}**")
    if len(tracks) > len(preview):
        st.caption(f"... and {len(tracks) - len(preview)} more")


def show_recommendations(result):
    st.subheader("Recommendations")
    if not result.recommendations:
        st.warning("No recommendations could be generated from the known input tracks.")
        return
    for rec in result.recommendations:
        track_id = rec.uri.split(":")[-1]
        cols = st.columns([4.6, 1.1, 1])
        cols[0].write(f"**{rec.rank}. {rec.uri}**")
        cols[1].caption(f"ranking score {rec.score:.3f}")
        cols[2].link_button("Open", f"https://open.spotify.com/track/{track_id}")


def show_analysis(status):
    st.header("Experiment Analysis")
    analysis = load_analysis_data(status)

    st.subheader("Model comparison")
    comparison = stage_comparison_df(analysis)
    comparison_long = comparison.melt(
        id_vars="model",
        value_vars=["R@10", "R@100", "NDCG@10"],
        var_name="metric",
        value_name="value",
    )
    model_chart = (
        alt.Chart(comparison_long)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("metric:N", title=None, sort=["R@10", "R@100", "NDCG@10"]),
            xOffset=alt.XOffset("model:N"),
            y=alt.Y("value:Q", title="score", scale=alt.Scale(domain=[0, 0.26])),
            color=alt.Color("model:N", title=None),
            tooltip=["model:N", "metric:N", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=260)
    )
    st.altair_chart(model_chart, width="stretch")

    stage2 = analysis["stage2_metrics"]
    cols = st.columns(3)
    cols[0].metric("R@10", f"{stage2['R@10']:.3f}")
    cols[1].metric("R@100", f"{stage2['R@100']:.3f}")
    cols[2].metric("NDCG@10", f"{stage2['NDCG@10']:.3f}")
    st.caption(f"Source: {analysis['source']['stage2']}")

    train_history = train_history_df(analysis)
    if not train_history.empty:
        st.subheader("Stage 2 training curves")
        loss_chart = (
            alt.Chart(train_history)
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:O", title="epoch"),
                y=alt.Y("train_loss:Q", title="training loss"),
                tooltip=["epoch:O", alt.Tooltip("train_loss:Q", format=".3f")],
            )
            .properties(height=220)
        )
        metric_cols = [c for c in ["val_R@10", "val_R@100", "val_NDCG@10"] if c in train_history.columns]
        if metric_cols:
            val_long = train_history.melt(
                id_vars="epoch",
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            val_chart = (
                alt.Chart(val_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("epoch:O", title="epoch"),
                    y=alt.Y("value:Q", title="validation metric"),
                    color=alt.Color("metric:N", title=None),
                    tooltip=["epoch:O", "metric:N", alt.Tooltip("value:Q", format=".3f")],
                )
                .properties(height=220)
            )
            st.altair_chart(loss_chart | val_chart, width="stretch")
        else:
            st.altair_chart(loss_chart, width="stretch")
        st.caption(f"Source: {analysis['source']['stage2_train_history']}")
    else:
        st.info(
            "Stage 2 training-curve JSON is not available locally. "
            "Pull `stage_2/checkpoints/train_history.json` from Modal to show loss/validation curves."
        )

    st.subheader("MMR lambda sweep")
    sweep = lambda_sweep_df(analysis)
    baseline = analysis["stage3_baseline"]
    metric_long = sweep.melt(
        id_vars="lambda",
        value_vars=["Recall@20", "ILD"],
        var_name="metric",
        value_name="value",
    )
    tradeoff_chart = (
        alt.Chart(metric_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("lambda:Q", title="MMR lambda", scale=alt.Scale(domain=[0.25, 0.75])),
            y=alt.Y("value:Q", title="score"),
            color=alt.Color("metric:N", title=None),
            tooltip=[
                alt.Tooltip("lambda:Q", format=".1f"),
                "metric:N",
                alt.Tooltip("value:Q", format=".4f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(tradeoff_chart, width="stretch")

    delta_long = sweep.melt(
        id_vars="lambda",
        value_vars=["Recall change vs raw top-20", "ILD change vs raw top-20"],
        var_name="metric",
        value_name="percent_change",
    )
    delta_chart = (
        alt.Chart(delta_long)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("lambda:N", title="MMR lambda"),
            y=alt.Y("percent_change:Q", title="% vs Stage 2 raw top-20"),
            color=alt.Color("metric:N", title=None),
            tooltip=[
                "metric:N",
                alt.Tooltip("lambda:N", title="lambda"),
                alt.Tooltip("percent_change:Q", format=".1f"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(delta_chart, width="stretch")

    st.dataframe(sweep, width="stretch", hide_index=True)
    st.caption(
        f"Baseline: {baseline['label']} "
        f"(Recall@20={baseline['Recall@20']:.4f}, ILD={baseline['ILD']:.4f}). "
        f"Source: {analysis['source']['stage3']}"
    )
    st.write(
        "Lower lambda increases diversity but usually sacrifices recall. "
        "The reported lambda=0.5 setting is the balanced point: it keeps most "
        "of the Stage 2 top-20 recall while giving a clear diversity gain."
    )

    st.subheader("Parameter variations covered")
    st.dataframe(parameter_variations_df(), width="stretch", hide_index=True)


def main():
    st.title("Spotify Playlist Recommender")
    st.caption("ALS candidate generation -> SASRec ranking -> MMR diversity re-ranking")

    status = detect_artifacts()
    show_status(status)

    tab_demo, tab_analysis = st.tabs(["Demo", "Analysis"])

    with tab_demo:
        n_recs = st.slider("Number of recommendations", min_value=5, max_value=25, value=20, step=5)
        mmr_lambda = st.slider("MMR lambda", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

        if "confirmed_playlist_uris" not in st.session_state:
            st.session_state.confirmed_playlist_uris = []

        pasted = st.text_area(
            "Spotify track URIs or track URLs",
            placeholder="spotify:track:...\nspotify:track:...",
            height=140,
        )
        parsed_uris = parse_track_uris(pasted)

        if st.button("Confirm tracks"):
            st.session_state.confirmed_playlist_uris = parsed_uris

        playlist_uris = st.session_state.confirmed_playlist_uris
        if playlist_uris:
            st.success(f"Confirmed {len(playlist_uris)} track(s).")
            tracks = [{"uri": uri} for uri in playlist_uris]
            show_input_preview(tracks)
        elif pasted:
            st.caption(f"Parsed {len(parsed_uris)} valid track URI(s). Confirm tracks to use them.")

        if playlist_uris and st.button("Generate recommendations", type="primary"):
            try:
                with st.spinner("Loading pipeline and generating recommendations..."):
                    pipeline = load_pipeline()
                    aliases = cached_track_aliases(tuple(playlist_uris))
                    result = pipeline.recommend(
                        playlist_uris,
                        n_recommendations=n_recs,
                        mmr_lambda=mmr_lambda,
                        alias_uris=aliases,
                    )
                cov = result.coverage
                st.info(f"MPD coverage: {cov['known']} / {cov['total']} input tracks found.")
                alias_hits = {
                    uri: values
                    for uri, values in aliases.items()
                    if uri in result.known_input_uris and uri not in pipeline.uri_to_id
                }
                if alias_hits:
                    st.caption(
                        f"Resolved {len(alias_hits)} input track(s) through Spotify relink aliases."
                    )
                status_df = pd.DataFrame(
                    [{"stage": k, "status": v} for k, v in result.stage_status.items()]
                )
                st.dataframe(status_df, width="stretch", hide_index=True)
                show_recommendations(result)
                if result.unknown_input_uris:
                    with st.expander("Input tracks not found in MPD mapping"):
                        st.write(result.unknown_input_uris)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.code(
                    "stage_1/checkpoints/\n"
                    "  als_item_factors.npy\n"
                    "  uri_to_id.json",
                    language="text",
                )
            except Exception as exc:
                st.exception(exc)

    with tab_analysis:
        show_analysis(status)


if __name__ == "__main__":
    main()
