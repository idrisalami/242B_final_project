import json

import numpy as np

from stage_4.pipeline import (
    RecommendationPipeline,
    detect_artifacts,
    raw_to_shifted,
    shifted_to_raw,
)


def test_id_shift_round_trip():
    raw = np.array([0, 41, 2262291], dtype=np.int64)
    shifted = raw_to_shifted(raw)
    assert shifted.tolist() == [1, 42, 2262292]
    assert shifted_to_raw(shifted).tolist() == raw.tolist()


def test_detect_artifacts_supports_flat_and_nested_stage2(tmp_path):
    stage1 = tmp_path / "stage_1"
    stage2 = tmp_path / "stage_2"
    stage3 = tmp_path / "stage_3"
    stage1.mkdir()
    stage2.mkdir()
    stage3.mkdir()
    (stage1 / "als_item_factors.npy").write_bytes(b"x")
    (stage1 / "uri_to_id.json").write_text("{}")
    (stage2 / "best_model.pt").write_bytes(b"x")
    (stage2 / "best_item_embeddings.npy").write_bytes(b"x")

    status = detect_artifacts(stage1_ckpt=stage1, stage2_ckpt=stage2, stage3_ckpt=stage3)

    assert status.stage1_ready
    assert status.stage2_ready
    assert status.stage3_ready
    assert status.stage2_model == stage2 / "best_model.pt"
    assert status.stage2_embeddings == stage2 / "best_item_embeddings.npy"


def test_detect_artifacts_supports_nested_stage2(tmp_path):
    stage1 = tmp_path / "stage_1"
    stage2 = tmp_path / "stage_2"
    stage3 = tmp_path / "stage_3"
    nested = stage2 / "best"
    stage1.mkdir()
    nested.mkdir(parents=True)
    stage3.mkdir()
    (stage1 / "als_item_factors.npy").write_bytes(b"x")
    (stage1 / "uri_to_id.json").write_text("{}")
    (nested / "model.pt").write_bytes(b"x")
    (nested / "item_embeddings.npy").write_bytes(b"x")

    status = detect_artifacts(stage1_ckpt=stage1, stage2_ckpt=stage2, stage3_ckpt=stage3)

    assert status.stage2_ready
    assert status.stage2_model == nested / "model.pt"
    assert status.stage2_embeddings == nested / "item_embeddings.npy"


def test_stage1_masks_seen_tracks(tmp_path):
    stage1 = tmp_path / "stage_1"
    stage2 = tmp_path / "stage_2"
    stage3 = tmp_path / "stage_3"
    stage1.mkdir()
    stage2.mkdir()
    stage3.mkdir()

    factors = np.array(
        [
            [10.0, 0.0],
            [8.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.5],
        ],
        dtype=np.float32,
    )
    np.save(stage1 / "als_item_factors.npy", factors)
    (stage1 / "uri_to_id.json").write_text(
        json.dumps(
            {
                "spotify:track:a": 0,
                "spotify:track:b": 1,
                "spotify:track:c": 2,
                "spotify:track:d": 3,
            }
        )
    )
    status = detect_artifacts(stage1_ckpt=stage1, stage2_ckpt=stage2, stage3_ckpt=stage3)
    pipeline = RecommendationPipeline(status=status, device="cpu", mmap_stage1=False)

    ids, _ = pipeline.stage1_candidates([0], k=3)

    assert 0 not in ids.tolist()
