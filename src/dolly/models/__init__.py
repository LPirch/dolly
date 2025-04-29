import json
from pathlib import Path

from transformers import PretrainedConfig

from dolly.models.llm import UniXcoder, UniXcoderConfig
from dolly.models.gnn import CloneGNN, GNNConfig


def load_model(model_name: str, config: PretrainedConfig | None = None, pos_weight: float = None):
    if model_name == "llm":
        if config is None:
            config = UniXcoderConfig(pos_weight=pos_weight)
        return UniXcoder(config)
    elif model_name == "gnn":
        if config is None:
            config = GNNConfig(pos_weight=pos_weight)
        return CloneGNN(config)
    else:
        raise ValueError(f"Model {model_name} not found")


def load_trained(model_path: Path):
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
    model_name = config["_name_or_path"]
    if model_name == "llm":
        return UniXcoder.load_model(model_path)
    elif model_name == "gnn":
        return CloneGNN.load_model(model_path)
    else:
        raise ValueError(f"Model {model_name} not found")


def init_hf_cache():
    UniXcoder(UniXcoderConfig())  # load the model to cache the hf cache
