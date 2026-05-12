"""Analysis tables and exports for Stage 4.

The app uses these helpers for charts, and the module can be run directly to
export concise CSV tables for the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from stage_4.pipeline import load_analysis_data

STAGE1_DOCUMENTED = {
    "model": "Stage 1 ALS",
    "R@10": 0.035,
    "R@100": 0.151,
    "NDCG@10": 0.018,
}

DEFAULT_STAGE3_BASELINE = {
    "label": "Stage 2 raw top-20",
    "Recall@20": 0.1186,
    "ILD": 0.3682,
}


def stage_comparison_df(analysis: Dict[str, object] | None = None) -> pd.DataFrame:
    """Model-level comparison for the main report table/chart."""
    analysis = analysis or load_analysis_data()
    stage2 = analysis["stage2_metrics"]
    rows = [
        STAGE1_DOCUMENTED,
        {
            "model": "Stage 2 SASRec",
            "R@10": stage2["R@10"],
            "R@100": stage2["R@100"],
            "NDCG@10": stage2["NDCG@10"],
        },
    ]
    return pd.DataFrame(rows)


def lambda_sweep_df(analysis: Dict[str, object] | None = None) -> pd.DataFrame:
    """MMR parameter sweep with absolute metrics and relative changes."""
    analysis = analysis or load_analysis_data()
    baseline = analysis.get("stage3_baseline") or DEFAULT_STAGE3_BASELINE
    rows = []
    for row in analysis["stage3_sweep"]:
        recall = float(row["recall_at_20"])
        ild = float(row["ild"])
        rows.append(
            {
                "lambda": float(row["lambda"]),
                "Recall@20": recall,
                "ILD": ild,
                "Recall change vs raw top-20": (recall / baseline["Recall@20"] - 1.0) * 100.0,
                "ILD change vs raw top-20": (ild / baseline["ILD"] - 1.0) * 100.0,
            }
        )
    return pd.DataFrame(rows).sort_values("lambda")


def parameter_variations_df() -> pd.DataFrame:
    """Document which parameters are actually varied and how to interpret them."""
    return pd.DataFrame(
        [
            {
                "parameter": "MMR lambda",
                "values": "0.3, 0.5, 0.7",
                "measured metrics": "Recall@20, ILD",
                "purpose": "Measure relevance/diversity tradeoff.",
                "status": "Evaluated on 150K test playlists.",
            },
            {
                "parameter": "Number of recommendations",
                "values": "5, 10, 15, 20, 25",
                "measured metrics": "Live demo output only",
                "purpose": "Change how many final songs the app displays.",
                "status": "Interactive control; not an offline metric sweep.",
            },
            {
                "parameter": "Evaluation sample size",
                "values": "150K test playlists",
                "measured metrics": "All reported metrics",
                "purpose": "Use stable full-test-set estimates.",
                "status": "Fixed in current artifacts; no subsample sweep saved.",
            },
        ]
    )


def train_history_df(analysis: Dict[str, object] | None = None) -> pd.DataFrame:
    """Return Stage 2 training history if the checkpoint JSON is available."""
    analysis = analysis or load_analysis_data()
    history = analysis.get("stage2_train_history")
    if not history:
        return pd.DataFrame()

    rows = []
    for item in history:
        val = item.get("val", {})
        rows.append(
            {
                "epoch": item.get("epoch"),
                "train_loss": item.get("train_loss"),
                "val_R@10": val.get("R@10"),
                "val_R@100": val.get("R@100"),
                "val_NDCG@10": val.get("NDCG@10"),
                "wall_min": item.get("wall_min"),
            }
        )
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def export_analysis_tables(out_dir: Path) -> Dict[str, Path]:
    """Write analysis CSVs and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = load_analysis_data()
    tables = {
        "stage_comparison": stage_comparison_df(analysis),
        "lambda_sweep": lambda_sweep_df(analysis),
        "parameter_variations": parameter_variations_df(),
    }
    train_history = train_history_df(analysis)
    if not train_history.empty:
        tables["train_history"] = train_history
    paths: Dict[str, Path] = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage 4 analysis tables.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stage_4/analysis_exports"),
        help="Directory where CSV tables will be written.",
    )
    args = parser.parse_args()
    paths = export_analysis_tables(args.out_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
