"""Stage 4 integration layer for the Spotify recommendation pipeline.

This module keeps the app honest about what is actually available locally:
Stage 1 is required, while Stage 2 and Stage 3 are enabled only when their
checkpoint artifacts exist. All Stage 2/3 calls use the +1-shifted ID space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CKPT = PROJECT_ROOT / "stage_1" / "checkpoints"
STAGE2_CKPT = PROJECT_ROOT / "stage_2" / "checkpoints"
STAGE3_CKPT = PROJECT_ROOT / "stage_3" / "checkpoints"


@dataclass(frozen=True)
class ArtifactStatus:
    """Resolved local artifact paths and availability flags."""

    stage1_ready: bool
    stage2_ready: bool
    stage3_ready: bool
    als_factors: Path
    uri_to_id: Path
    stage2_model: Optional[Path]
    stage2_embeddings: Optional[Path]
    stage2_metrics: Optional[Path]
    stage2_train_history: Optional[Path]
    stage3_metrics: Optional[Path]
    stage3_lambda_sweep: Optional[Path]
    messages: Tuple[str, ...]


@dataclass(frozen=True)
class Recommendation:
    """One final recommendation returned by the Stage 4 pipeline."""

    rank: int
    uri: str
    raw_id: int
    shifted_id: int
    score: float


@dataclass(frozen=True)
class PipelineResult:
    """Full output from a recommendation request."""

    recommendations: List[Recommendation]
    known_input_uris: List[str]
    unknown_input_uris: List[str]
    stage_status: Dict[str, str]
    coverage: Dict[str, int]


def raw_to_shifted(raw_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """Convert Stage 1 raw track IDs to Stage 2/3 shifted IDs."""
    return np.asarray(raw_ids, dtype=np.int64) + 1


def shifted_to_raw(shifted_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """Convert Stage 2/3 shifted IDs back to Stage 1 raw track IDs."""
    raw = np.asarray(shifted_ids, dtype=np.int64) - 1
    if np.any(raw < 0):
        raise ValueError("Shifted IDs must be >= 1 for real tracks.")
    return raw


def detect_artifacts(
    stage1_ckpt: Path = STAGE1_CKPT,
    stage2_ckpt: Path = STAGE2_CKPT,
    stage3_ckpt: Path = STAGE3_CKPT,
) -> ArtifactStatus:
    """Inspect local checkpoint directories without loading large arrays."""
    als_factors = stage1_ckpt / "als_item_factors.npy"
    uri_to_id = stage1_ckpt / "uri_to_id.json"
    stage1_ready = als_factors.exists() and uri_to_id.exists()

    nested_model = stage2_ckpt / "best" / "model.pt"
    nested_embeddings = stage2_ckpt / "best" / "item_embeddings.npy"
    flat_model = stage2_ckpt / "best_model.pt"
    flat_embeddings = stage2_ckpt / "best_item_embeddings.npy"

    if nested_model.exists() and nested_embeddings.exists():
        stage2_model = nested_model
        stage2_embeddings = nested_embeddings
    elif flat_model.exists() and flat_embeddings.exists():
        stage2_model = flat_model
        stage2_embeddings = flat_embeddings
    else:
        stage2_model = None
        stage2_embeddings = flat_embeddings if flat_embeddings.exists() else None

    stage2_ready = stage2_model is not None and stage2_embeddings is not None
    stage3_ready = stage2_embeddings is not None and stage2_embeddings.exists()

    stage2_metrics = stage2_ckpt / "test_metrics.json"
    if not stage2_metrics.exists():
        stage2_metrics = None

    stage2_train_history = stage2_ckpt / "train_history.json"
    if not stage2_train_history.exists():
        stage2_train_history = None

    stage3_metrics = stage3_ckpt / "test_metrics.json"
    if not stage3_metrics.exists():
        stage3_metrics = None

    stage3_lambda_sweep = stage3_ckpt / "lambda_sweep.json"
    if not stage3_lambda_sweep.exists():
        stage3_lambda_sweep = None

    messages: List[str] = []
    if not stage1_ready:
        messages.append("Stage 1 artifacts are missing; recommendations cannot run.")
    if not stage2_ready:
        messages.append("Stage 2 checkpoint is missing; app will use ALS ranking fallback.")
    if not stage3_ready:
        messages.append("Stage 3 MMR embeddings are missing; diversity re-ranking is unavailable.")

    return ArtifactStatus(
        stage1_ready=stage1_ready,
        stage2_ready=stage2_ready,
        stage3_ready=stage3_ready,
        als_factors=als_factors,
        uri_to_id=uri_to_id,
        stage2_model=stage2_model,
        stage2_embeddings=stage2_embeddings,
        stage2_metrics=stage2_metrics,
        stage2_train_history=stage2_train_history,
        stage3_metrics=stage3_metrics,
        stage3_lambda_sweep=stage3_lambda_sweep,
        messages=tuple(messages),
    )


class RecommendationPipeline:
    """Load available artifacts and run Stage 1 -> Stage 2 -> Stage 3."""

    def __init__(
        self,
        status: ArtifactStatus | None = None,
        device: str | None = None,
        mmap_stage1: bool = True,
    ):
        self.status = status or detect_artifacts()
        if not self.status.stage1_ready:
            missing = ", ".join(str(p) for p in (self.status.als_factors, self.status.uri_to_id))
            raise FileNotFoundError(f"Missing required Stage 1 artifacts: {missing}")

        self.item_factors = np.load(
            self.status.als_factors,
            mmap_mode="r" if mmap_stage1 else None,
        )
        with open(self.status.uri_to_id) as f:
            self.uri_to_id: Dict[str, int] = {k: int(v) for k, v in json.load(f).items()}
        self.id_to_uri: Dict[int, str] = {v: k for k, v in self.uri_to_id.items()}

        self.device = device or self._default_device()
        self.stage2_model = None
        self.stage3_embeddings_normed = None

        if self.status.stage2_ready:
            self.stage2_model = self._load_stage2_model()
        if self.status.stage3_ready:
            self.stage3_embeddings_normed = self._load_stage3_embeddings()

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_stage2_model(self):
        import torch

        from stage_2.models.sasrec import SASRec

        assert self.status.stage2_model is not None
        assert self.status.stage2_embeddings is not None

        embeddings = np.load(self.status.stage2_embeddings, mmap_mode=None).astype(np.float32)
        model = SASRec(
            vocab_size=int(embeddings.shape[0]),
            d_model=int(embeddings.shape[1]),
            n_layers=2,
            n_heads=2,
            ffn_dim=256,
            max_seq_len=50,
            dropout=0.2,
        ).to(self.device)
        with torch.no_grad():
            model.item_emb.weight.copy_(torch.from_numpy(embeddings).to(self.device))
        state = torch.load(self.status.stage2_model, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def _load_stage3_embeddings(self) -> np.ndarray:
        from stage_3.mmr import l2_normalize_rows

        assert self.status.stage2_embeddings is not None
        embeddings = np.load(self.status.stage2_embeddings, mmap_mode=None).astype(np.float32)
        return l2_normalize_rows(embeddings)

    def known_raw_ids(self, playlist_uris: Sequence[str]) -> Tuple[List[int], List[str], List[str]]:
        """Map Spotify URIs to raw Stage 1 IDs and return known/unknown URI lists."""
        ids: List[int] = []
        known: List[str] = []
        unknown: List[str] = []
        for uri in playlist_uris:
            if uri in self.uri_to_id:
                ids.append(self.uri_to_id[uri])
                known.append(uri)
            else:
                unknown.append(uri)
        return ids, known, unknown

    def stage1_candidates(
        self,
        raw_ids: Sequence[int],
        k: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1 ALS retrieval in raw Stage 1 ID space."""
        if not raw_ids:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        k = min(int(k), int(self.item_factors.shape[0]))
        context = np.asarray(raw_ids, dtype=np.int64)
        user_emb = np.asarray(self.item_factors[context]).mean(axis=0)
        scores = user_emb @ self.item_factors.T
        scores = np.asarray(scores, dtype=np.float32)
        scores[context] = -np.inf

        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return top.astype(np.int32), scores[top].astype(np.float32)

    def _stage2_or_fallback(
        self,
        raw_ids: Sequence[int],
        candidate_raw: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Return top-100 shifted IDs and scores."""
        if self.stage2_model is None:
            top_raw = candidate_raw[:100]
            scores = candidate_scores[:100]
            return raw_to_shifted(top_raw).astype(np.int32), scores.astype(np.float32), "fallback: ALS top-100"

        from stage_2.inference.predict import predict_top_100

        playlist_shifted = raw_to_shifted(raw_ids).astype(np.int32).tolist()
        candidates_shifted = raw_to_shifted(candidate_raw).astype(np.int32)
        top_ids, top_scores = predict_top_100(
            model=self.stage2_model,
            playlist_shifted_ids=playlist_shifted,
            stage1_candidates=candidates_shifted,
            device=self.device,
            max_seq_len=50,
        )
        return top_ids.astype(np.int32), top_scores.astype(np.float32), "real: SASRec top-100"

    def _stage3_or_fallback(
        self,
        ranked_shifted: np.ndarray,
        ranked_scores: np.ndarray,
        n_recommendations: int,
        mmr_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Return final shifted IDs and scores."""
        n = min(int(n_recommendations), int(ranked_shifted.shape[0]))
        if self.stage3_embeddings_normed is None:
            return (
                ranked_shifted[:n].astype(np.int32),
                ranked_scores[:n].astype(np.float32),
                "fallback: top-N ranking, no MMR embeddings",
            )

        from stage_3.mmr import mmr_batch

        final_ids, final_scores = mmr_batch(
            candidates=ranked_shifted[None, :].astype(np.int32),
            rel_scores=ranked_scores[None, :].astype(np.float32),
            item_emb_normed=self.stage3_embeddings_normed,
            K=n,
            lam=float(mmr_lambda),
        )
        return final_ids[0], final_scores[0], f"real: MMR lambda={mmr_lambda:.2f}"

    def recommend(
        self,
        playlist_uris: Sequence[str],
        n_recommendations: int = 20,
        mmr_lambda: float = 0.5,
    ) -> PipelineResult:
        """Run the best available local pipeline for a Spotify URI playlist."""
        raw_ids, known, unknown = self.known_raw_ids(playlist_uris)
        if not raw_ids:
            return PipelineResult(
                recommendations=[],
                known_input_uris=known,
                unknown_input_uris=unknown,
                stage_status={
                    "stage1": "blocked: no input tracks found in MPD mapping",
                    "stage2": "not run",
                    "stage3": "not run",
                },
                coverage={"total": len(playlist_uris), "known": 0, "unknown": len(playlist_uris)},
            )

        candidate_raw, candidate_scores = self.stage1_candidates(raw_ids, k=1000)
        ranked_shifted, ranked_scores, stage2_status = self._stage2_or_fallback(
            raw_ids=raw_ids,
            candidate_raw=candidate_raw,
            candidate_scores=candidate_scores,
        )
        final_shifted, final_scores, stage3_status = self._stage3_or_fallback(
            ranked_shifted=ranked_shifted,
            ranked_scores=ranked_scores,
            n_recommendations=n_recommendations,
            mmr_lambda=mmr_lambda,
        )

        final_raw = shifted_to_raw(final_shifted)
        recs: List[Recommendation] = []
        for idx, (raw_id, shifted_id, score) in enumerate(zip(final_raw, final_shifted, final_scores), start=1):
            uri = self.id_to_uri.get(int(raw_id))
            if uri is None:
                continue
            recs.append(
                Recommendation(
                    rank=idx,
                    uri=uri,
                    raw_id=int(raw_id),
                    shifted_id=int(shifted_id),
                    score=float(score),
                )
            )

        return PipelineResult(
            recommendations=recs,
            known_input_uris=known,
            unknown_input_uris=unknown,
            stage_status={
                "stage1": f"real: ALS top-{len(candidate_raw)}",
                "stage2": stage2_status,
                "stage3": stage3_status,
            },
            coverage={"total": len(playlist_uris), "known": len(known), "unknown": len(unknown)},
        )


DEFAULT_STAGE2_METRICS = {
    "R@10": 0.082,
    "R@100": 0.236,
    "NDCG@10": 0.046,
}

DEFAULT_STAGE3_SWEEP = [
    {"lambda": 0.3, "recall_at_20": 0.0975, "ild": 0.4958},
    {"lambda": 0.5, "recall_at_20": 0.1135, "ild": 0.4327},
    {"lambda": 0.7, "recall_at_20": 0.1173, "ild": 0.3925},
]

DEFAULT_STAGE3_BASELINE = {
    "label": "Stage 2 raw top-20",
    "Recall@20": 0.1186,
    "ILD": 0.3682,
}


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_analysis_data(status: ArtifactStatus | None = None) -> Dict[str, object]:
    """Load report metrics if present, otherwise use documented README values."""
    status = status or detect_artifacts()
    stage2_json = load_json(status.stage2_metrics)
    stage2_train_history = load_json(status.stage2_train_history)
    stage3_metrics = load_json(status.stage3_metrics)
    stage3_sweep_json = load_json(status.stage3_lambda_sweep)

    stage2_metrics = DEFAULT_STAGE2_METRICS
    if stage2_json:
        pipeline = stage2_json.get("pipeline", stage2_json)
        stage2_metrics = {
            "R@10": float(pipeline.get("R@10", DEFAULT_STAGE2_METRICS["R@10"])),
            "R@100": float(pipeline.get("R@100", DEFAULT_STAGE2_METRICS["R@100"])),
            "NDCG@10": float(pipeline.get("NDCG@10", DEFAULT_STAGE2_METRICS["NDCG@10"])),
        }

    stage3_sweep = DEFAULT_STAGE3_SWEEP
    stage3_baseline = DEFAULT_STAGE3_BASELINE
    if stage3_sweep_json and "sweep" in stage3_sweep_json:
        parsed = []
        for key, value in stage3_sweep_json["sweep"].items():
            lam = float(key.replace("lambda_", ""))
            parsed.append(
                {
                    "lambda": lam,
                    "recall_at_20": float(value["recall_at_K"]),
                    "ild": float(value["intra_list_diversity"]),
                }
            )
        stage3_sweep = sorted(parsed, key=lambda row: row["lambda"])
        baseline = stage3_sweep_json.get("baseline_stage2_top20")
        if baseline:
            stage3_baseline = {
                "label": "Stage 2 raw top-20",
                "Recall@20": float(baseline["recall_at_K"]),
                "ILD": float(baseline["intra_list_diversity"]),
            }

    return {
        "stage2_metrics": stage2_metrics,
        "stage2_train_history": stage2_train_history,
        "stage3_metrics": stage3_metrics,
        "stage3_baseline": stage3_baseline,
        "stage3_sweep": stage3_sweep,
        "source": {
            "stage2": str(status.stage2_metrics) if status.stage2_metrics else "README defaults",
            "stage2_train_history": (
                str(status.stage2_train_history) if status.stage2_train_history else "not available"
            ),
            "stage3": str(status.stage3_lambda_sweep) if status.stage3_lambda_sweep else "README defaults",
        },
    }
