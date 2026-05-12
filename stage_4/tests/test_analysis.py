from pathlib import Path

from stage_4.analysis import (
    export_analysis_tables,
    lambda_sweep_df,
    parameter_variations_df,
    stage_comparison_df,
    train_history_df,
)


def test_stage_comparison_has_expected_models():
    df = stage_comparison_df()
    assert df["model"].tolist() == ["Stage 1 ALS", "Stage 2 SASRec"]
    assert {"R@10", "R@100", "NDCG@10"}.issubset(df.columns)


def test_lambda_sweep_has_relative_changes():
    df = lambda_sweep_df()
    assert df["lambda"].tolist() == [0.3, 0.5, 0.7]
    assert "Recall change vs raw top-20" in df.columns
    assert "ILD change vs raw top-20" in df.columns
    assert df.loc[df["lambda"] == 0.5, "ILD change vs raw top-20"].iloc[0] > 0


def test_parameter_variations_documents_offline_and_demo_controls():
    df = parameter_variations_df()
    assert "MMR lambda" in df["parameter"].tolist()
    assert "Number of recommendations" in df["parameter"].tolist()


def test_export_analysis_tables(tmp_path: Path):
    paths = export_analysis_tables(tmp_path)
    assert {"stage_comparison", "lambda_sweep", "parameter_variations"}.issubset(paths)
    for path in paths.values():
        assert path.exists()
        assert path.read_text().strip()


def test_train_history_empty_when_missing():
    assert train_history_df({"stage2_train_history": None}).empty
