from pathlib import Path
from shutil import rmtree, move
import json
import hashlib

import typer
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
from transformers import AutoTokenizer
from loguru import logger

from dolly.graphs.parse import parse_cpg
from dolly.graphs.export import export_cpg
from dolly.graphs.pyg import graph_to_pyg, embed_graph_batch

app = typer.Typer()


def read_split(split_file: Path, idx_map: dict):
    rows = []
    with open(split_file, "r") as f:
        for line in f:
            if line.strip():
                a, b, label = line.strip().split()
                _a = idx_map[a]
                _b = idx_map[b]
                label = int(label)
                rows.append(dict(a=_a, b=_b, a_orig=int(a), b_orig=int(b), label=label))
    return rows


def tokenize_ds(ds: Dataset, base_model: str, max_length: int = 1024):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    def tokenize(sample):
        return tokenizer(sample["func"], max_length=max_length, truncation=True)
    ds = ds.map(tokenize, batched=True)
    return ds


def load_hf(hf_dir: Path):
    hf_funcs = hf_dir / "funcs.hf"
    hf_splits = hf_dir / "splits.hf"
    if not hf_funcs.exists():
        raise ValueError(f"{hf_funcs} not found")
    if not hf_splits.exists():
        raise ValueError(f"{hf_splits} not found")
    return load_from_disk(hf_funcs), load_from_disk(hf_splits)


@app.command("init")
def init_dataset(
    raw_dir: Path = typer.Option(default=Path("data/big-clone-bench/dataset")),
    hf_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf")),
    base_model: str = typer.Option(default="microsoft/unixcoder-base-nine"),
):
    ds_funcs = load_dataset("json", data_files=[str(raw_dir/"data.jsonl")], split="train")
    ds_funcs = tokenize_ds(ds_funcs, base_model)
    idx_map = {row["idx"]: i for i, row in enumerate(ds_funcs)}
    ds_splits = DatasetDict({
        "train": Dataset.from_list(read_split(raw_dir / "train.txt", idx_map)),
        "valid": Dataset.from_list(read_split(raw_dir / "valid.txt", idx_map)),
        "test": Dataset.from_list(read_split(raw_dir / "test.txt", idx_map)),
    })
    hf_dir.mkdir(parents=True, exist_ok=True)
    ds_funcs.save_to_disk(hf_dir / "funcs.hf")
    ds_splits.save_to_disk(hf_dir / "splits.hf")


class CloneDataset(TorchDataset):
    def __init__(self, hf_dir: Path, split: str, subsample: float = 1.0, seed: int = 42):
        funcs, splits = load_hf(hf_dir)
        self.funcs = funcs.with_format("torch")
        self.idx = splits[split].with_format("torch")
        if subsample < 1.0:
            self.idx = self.idx.shuffle(seed=seed).select(range(int(len(self.idx) * subsample)))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        item = self.idx[i]
        a = self.funcs[item["a"].item()]
        b = self.funcs[item["b"].item()]
        del a["idx"], b["idx"]
        label = int(item["label"])
        return {"a": a, "b": b, "label": label}

    def get_positive_ratio(self):
        return self.idx["label"].float().mean()

    def get_pos_weight(self) -> float:
        pos_ratio = self.get_positive_ratio().item()
        return 1.0 / pos_ratio


class PairCollator:
    def __init__(self, collate_fn, tokenizer):
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

    def __call__(self, batch):
        a = self.collate_fn([sample["a"] for sample in batch], self.tokenizer)
        b = self.collate_fn([sample["b"] for sample in batch], self.tokenizer)
        label = torch.tensor([sample["label"] for sample in batch])
        return {"a": a, "b": b, "label": label}


@app.command("parse-cpgs")
def parse_cpgs(
    hf_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf")),
    cpg_dir: Path = typer.Option(default=Path("data/big-clone-bench/cpg")),
    num_proc: int = typer.Option(default=4),
):
    cpg_dir.mkdir(parents=True, exist_ok=True)
    funcs, _ = load_hf(hf_dir)
    funcs = funcs.map(parse_cpg, batched=False, cache_file_name=None, fn_kwargs={
        "cpg_dir": cpg_dir,
        "joern_memory": "2g",
        "joern_cores": 4,
    }, num_proc=num_proc)
    funcs.save_to_disk(hf_dir / "funcs-parsed.hf")
    rmtree(hf_dir / "funcs.hf")
    move(hf_dir / "funcs-parsed.hf", hf_dir / "funcs.hf")


@app.command("export-cpgs")
def export_cpgs(
    hf_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf")),
    pyg_dir: Path = typer.Option(default=Path("data/big-clone-bench/pyg")),
    num_proc: int = typer.Option(default=4),
):
    pyg_dir.mkdir(parents=True, exist_ok=True)
    funcs, _ = load_hf(hf_dir)
    funcs = funcs.map(export_cpg, batched=False, cache_file_name=None, fn_kwargs={
        "pyg_dir": pyg_dir,
        "joern_memory": "2g",
        "joern_cores": 4,
    }, num_proc=num_proc)
    funcs.save_to_disk(hf_dir / "funcs-exported.hf")
    rmtree(hf_dir / "funcs.hf")
    move(hf_dir / "funcs-exported.hf", hf_dir / "funcs.hf")


@app.command("to-pyg")
def to_pyg(
    hf_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf")),
    hf_pyg_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf-pyg")),
    base_model: str = typer.Option(default="microsoft/unixcoder-base-nine"),
    num_proc: int = typer.Option(default=4),
):
    hf_pyg_dir.mkdir(parents=True, exist_ok=True)
    funcs, splits = load_hf(hf_dir)
    # only keep pairs with valid pyg files
    num_funcs = len(funcs)

    def is_valid(sample):
        a = int(sample["a"])
        b = int(sample["b"])
        return a < num_funcs and b < num_funcs \
            and Path(funcs[a]["pyg_path"]).exists() \
            and Path(funcs[b]["pyg_path"]).exists()
    train_len, val_len, test_len = len(splits["train"]), len(splits["valid"]), len(splits["test"])
    splits = splits.filter(is_valid)
    logger.info(f"Removed {train_len-len(splits['train'])}/{val_len-len(splits['valid'])}/{test_len-len(splits['test'])} invalid train/val/test samples.")
    funcs = funcs.map(graph_to_pyg, batched=False, load_from_cache_file=False, fn_kwargs={
        "base_model": base_model,
    }, num_proc=num_proc)

    # build node type mapping
    all_node_types = set()
    for node_types in funcs["x_types"]:
        for node_type in node_types:
            all_node_types.add(node_type)
    node_types = sorted(all_node_types)
    node_type_map = {node_type: i for i, node_type in enumerate(node_types)}
    with open(hf_pyg_dir / "node_type_map.json", "w") as f:
        json.dump(node_type_map, f, indent=4)

    def node_type_map_fn(sample):
        if sample is None:
            return None
        return {
            **sample,
            "x_types": [node_type_map[node_type] for node_type in sample["x_types"]],
        }
    funcs = funcs.map(node_type_map_fn, batched=False, cache_file_name=None)

    funcs.save_to_disk(hf_pyg_dir / "funcs.hf")
    splits.save_to_disk(hf_pyg_dir / "splits.hf")


@app.command("embed-graphs")
def embed_graphs(
    hf_pyg_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf-pyg")),
    hf_embed_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf-pyg-embed")),
    base_model: str = typer.Option(default="microsoft/unixcoder-base-nine"),
    num_proc: int = typer.Option(default=4),
):
    funcs, splits = load_hf(hf_pyg_dir)
    funcs = funcs.map(embed_graph_batch, batched=True, batch_size=1000, cache_file_name=None, fn_kwargs={
        "base_model": base_model,
    }, num_proc=num_proc)
    funcs.save_to_disk(hf_embed_dir / "funcs.hf")
    splits.save_to_disk(hf_embed_dir / "splits.hf")


def code_to_graph(code: str, inference_root: Path, node_type_map):
    """ Full conversion pipeline for inference. """
    sample = {
        "idx": hashlib.sha256(code.encode()).hexdigest(),
        "func": code,
    }
    sample = parse_cpg(sample, inference_root / "cpg", "2g", 4)
    sample = export_cpg(sample, inference_root / "pyg", "2g", 4)
    if not Path(sample["pyg_path"]).exists():
        raise ValueError(f"Couldn't extract graph from sample {sample['idx']}")
    sample = graph_to_pyg(sample, "microsoft/unixcoder-base-nine")
    sample["x_types"] = [node_type_map[node_type] for node_type in sample["x_types"]]
    return sample
