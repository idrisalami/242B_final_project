"""
Utility functions for Two-Tower model
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import torch


def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device(device_str: str = "cuda") -> str:
    """Get available device"""
    if device_str == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: torch.nn.Module, save_path: str):
    """Save model weights"""
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")


def load_model(model: torch.nn.Module, load_path: str):
    """Load model weights"""
    model.load_state_dict(torch.load(load_path))
    logging.info(f"Model loaded from {load_path}")
    return model
