"""Modal entry point for Stage 2 SASRec.

Local usage:
    modal run stage_2/modal_app.py --cmd cache
    modal run stage_2/modal_app.py --cmd train --run-id smoke --smoke
    modal run stage_2/modal_app.py --cmd train --run-id main
    modal run stage_2/modal_app.py --cmd eval --run-id main
    modal run stage_2/modal_app.py --cmd infer --run-id main

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §7.
"""

import json
from pathlib import Path

import modal
import yaml

CONFIG_PATH = "/repo/stage_2/config.yaml"


def _find_config_yaml() -> Path:
    """Locate config.yaml at module-import time.

    Locally: `Path(__file__).parent / "config.yaml"` works. Inside the Modal
    container, Modal places the entry file at /root/modal_app.py (so
    __file__.parent == /root, which has no config.yaml), but
    `add_local_dir(".", "/repo")` mounts the full repo, so the config is at
    /repo/stage_2/config.yaml.
    """
    local = Path(__file__).parent / "config.yaml"
    if local.exists():
        return local
    in_container = Path(CONFIG_PATH)
    if in_container.exists():
        return in_container
    raise FileNotFoundError(
        f"config.yaml not found at {local} or {in_container}"
    )


with open(_find_config_yaml()) as f:
    CFG_LOCAL = yaml.safe_load(f)

stub = modal.App(CFG_LOCAL["modal"]["app_name"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy>=1.24",
        "tqdm>=4.67",
        "pyyaml>=6.0",
    )
    .add_local_dir(".", "/repo")
)

volume = modal.Volume.from_name(CFG_LOCAL["modal"]["volume_name"], create_if_missing=True)

VOL_PATH = {"vol": volume}
GPU = CFG_LOCAL["modal"]["gpu"]
TIMEOUT = CFG_LOCAL["modal"]["function_timeout_sec"]


def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@stub.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=TIMEOUT)
def cache_stage1_candidates():
    """Pass 1 (pad playlists) + Pass 2 (Stage 1 candidate top-1000)."""
    import sys
    sys.path.insert(0, "/repo")
    from stage_2.training.preprocess import run_preprocessing
    cfg = _load_cfg()
    run_preprocessing(
        cfg=cfg,
        raw_playlists_path=cfg["paths"]["playlists_raw"],
        als_factors_path=cfg["paths"]["als_factors"],
        output_dir="/vol/derived",
    )
    volume.commit()


@stub.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=TIMEOUT)
def train(run_id: str = "main", smoke: bool = False):
    import sys
    sys.path.insert(0, "/repo")
    from stage_2.training.train import train_main
    cfg = _load_cfg()
    out = train_main(cfg, run_id=run_id, smoke=smoke)
    print(json.dumps({"best_ndcg": out["best_ndcg"], "run_dir": out["run_dir"]}, indent=2))
    volume.commit()


@stub.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=TIMEOUT)
def evaluate(run_id: str = "main", mode: str = "both"):
    """mode in {'pipeline', 'unconstrained', 'both'}."""
    import sys
    sys.path.insert(0, "/repo")
    import numpy as np
    import torch
    from stage_2.models.sasrec import SASRec
    from stage_2.training.train import load_checkpoint
    from stage_2.training.evaluate import (
        evaluate_pipeline_mode, evaluate_unconstrained, derived_metrics,
    )

    cfg = _load_cfg()
    run_dir = Path(cfg["paths"]["runs_dir"]) / run_id

    padded = np.load(cfg["paths"]["playlists_padded"])
    lengths = np.load(cfg["paths"]["playlists_lengths"])
    candidates = np.load(cfg["paths"]["candidates"])

    n_train = cfg["data"]["train_size"]
    n_val = cfg["data"]["val_size"]
    test_padded = padded[n_train + n_val:]
    test_lengths = lengths[n_train + n_val:]
    test_candidates = candidates[n_train + n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SASRec(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        ffn_dim=cfg["model"]["ffn_dim"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(run_dir, "best", model)

    results = {}
    if mode in ("pipeline", "both"):
        results["pipeline"] = evaluate_pipeline_mode(
            model, test_padded, test_lengths, test_candidates,
            batch_size=cfg["training"]["batch_size"],
            device=device, use_bf16=(device == "cuda"),
        )
    if mode in ("unconstrained", "both"):
        results["unconstrained"] = evaluate_unconstrained(
            model, test_padded, test_lengths,
            vocab_size=cfg["model"]["vocab_size"],
            batch_size=cfg["training"]["batch_size"],
            device=device, use_bf16=(device == "cuda"),
        )
    if "pipeline" in results and "unconstrained" in results:
        results["derived"] = derived_metrics(results["pipeline"], results["unconstrained"])

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    volume.commit()


@stub.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=TIMEOUT)
def precompute(run_id: str = "main"):
    import sys
    sys.path.insert(0, "/repo")
    import numpy as np
    import torch
    from stage_2.models.sasrec import SASRec
    from stage_2.training.train import load_checkpoint
    from stage_2.inference.predict import precompute_test_top100

    cfg = _load_cfg()
    run_dir = Path(cfg["paths"]["runs_dir"]) / run_id

    padded = np.load(cfg["paths"]["playlists_padded"])
    lengths = np.load(cfg["paths"]["playlists_lengths"])
    candidates = np.load(cfg["paths"]["candidates"])

    n_train = cfg["data"]["train_size"]
    n_val = cfg["data"]["val_size"]
    test_padded = padded[n_train + n_val:]
    test_lengths = lengths[n_train + n_val:]
    test_candidates = candidates[n_train + n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SASRec(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        ffn_dim=cfg["model"]["ffn_dim"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(run_dir, "best", model)

    precompute_test_top100(
        model=model,
        padded=test_padded,
        lengths=test_lengths,
        candidates=test_candidates,
        out_dir=run_dir,
        batch_size=cfg["training"]["batch_size"],
        device=device,
        use_bf16=(device == "cuda"),
    )
    volume.commit()


@stub.local_entrypoint()
def main(
    cmd: str,
    run_id: str = "main",
    smoke: bool = False,
    mode: str = "both",
):
    if cmd == "cache":
        cache_stage1_candidates.remote()
    elif cmd == "train":
        train.remote(run_id=run_id, smoke=smoke)
    elif cmd == "eval":
        evaluate.remote(run_id=run_id, mode=mode)
    elif cmd == "infer":
        precompute.remote(run_id=run_id)
    else:
        raise ValueError(f"Unknown cmd: {cmd}. Use cache|train|eval|infer.")
