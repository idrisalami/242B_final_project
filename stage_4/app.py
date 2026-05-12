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
    fetch_playlist_tracks,
    fetch_track_metadata,
    get_spotify_client,
    parse_track_uris,
)


st.set_page_config(page_title="Spotify Recommender", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RecommendationPipeline()


@st.cache_resource(show_spinner=False)
def load_spotify():
    return get_spotify_client()


@st.cache_data(show_spinner=False)
def cached_playlist_tracks(playlist_url_or_id: str):
    sp = load_spotify()
    if sp is None:
        raise RuntimeError("Spotify credentials are not configured.")
    return fetch_playlist_tracks(sp, playlist_url_or_id)


@st.cache_data(show_spinner=False)
def cached_track_metadata(uris: tuple[str, ...]):
    return fetch_track_metadata(load_spotify(), uris)


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
        cols = st.columns([0.6, 5])
        if row.get("album_art"):
            cols[0].image(row["album_art"], width=52)
        else:
            cols[0].write("")
        cols[1].write(f"**{row.get('name', row['uri'])}**")
        cols[1].caption(row.get("artist", row["uri"]))
    if len(tracks) > len(preview):
        st.caption(f"... and {len(tracks) - len(preview)} more")


def show_recommendations(result, metadata):
    st.subheader("Recommendations")
    if not result.recommendations:
        st.warning("No recommendations could be generated from the known input tracks.")
        return
    for rec in result.recommendations:
        meta = metadata.get(rec.uri, {})
        cols = st.columns([0.55, 4.5, 1.2])
        if meta.get("album_art"):
            cols[0].image(meta["album_art"], width=56)
        cols[1].write(f"**{rec.rank}. {meta.get('name', rec.uri)}**")
        cols[1].caption(meta.get("artist", rec.uri))
        if meta.get("spotify_url"):
            cols[2].link_button("Open", meta["spotify_url"])
        cols[2].caption(f"score {rec.score:.3f}")


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
    st.altair_chart(model_chart, use_container_width=True)

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
            st.altair_chart(loss_chart | val_chart, use_container_width=True)
        else:
            st.altair_chart(loss_chart, use_container_width=True)
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
    st.altair_chart(tradeoff_chart, use_container_width=True)

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
    st.altair_chart(delta_chart, use_container_width=True)

    st.dataframe(sweep, use_container_width=True, hide_index=True)
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
    st.dataframe(parameter_variations_df(), use_container_width=True, hide_index=True)


def main():
    st.title("Spotify Playlist Recommender")
    st.caption("ALS candidate generation -> SASRec ranking -> MMR diversity re-ranking")

    status = detect_artifacts()
    show_status(status)

    tab_demo, tab_analysis = st.tabs(["Demo", "Analysis"])

    with tab_demo:
        input_mode = st.radio(
            "Input mode",
            ["Spotify playlist URL", "Paste track URIs"],
            horizontal=True,
        )
        n_recs = st.slider("Number of recommendations", min_value=5, max_value=25, value=20, step=5)
        mmr_lambda = st.slider("MMR lambda", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

        tracks = []
        playlist_uris = []

        if input_mode == "Spotify playlist URL":
            playlist_url = st.text_input(
                "Spotify playlist URL",
                placeholder="https://open.spotify.com/playlist/...",
            )
            if playlist_url:
                try:
                    with st.spinner("Fetching playlist from Spotify..."):
                        tracks = cached_playlist_tracks(playlist_url)
                    playlist_uris = [t["uri"] for t in tracks]
                except Exception as exc:
                    st.error(str(exc))
        else:
            pasted = st.text_area(
                "Spotify track URIs or track URLs",
                placeholder="spotify:track:...\nspotify:track:...",
                height=140,
            )
            playlist_uris = parse_track_uris(pasted)
            tracks = [{"uri": uri, "name": uri, "artist": "Pasted URI", "album_art": None} for uri in playlist_uris]

        show_input_preview(tracks)

        if playlist_uris and st.button("Generate recommendations", type="primary"):
            try:
                with st.spinner("Loading pipeline and generating recommendations..."):
                    pipeline = load_pipeline()
                    result = pipeline.recommend(
                        playlist_uris,
                        n_recommendations=n_recs,
                        mmr_lambda=mmr_lambda,
                    )
                cov = result.coverage
                st.info(f"MPD coverage: {cov['known']} / {cov['total']} input tracks found.")
                status_df = pd.DataFrame(
                    [{"stage": k, "status": v} for k, v in result.stage_status.items()]
                )
                st.dataframe(status_df, use_container_width=True, hide_index=True)
                rec_uris = tuple(rec.uri for rec in result.recommendations)
                metadata = cached_track_metadata(rec_uris)
                show_recommendations(result, metadata)
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
