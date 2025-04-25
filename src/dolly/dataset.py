from pathlib import Path

from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
from transformers import AutoTokenizer


def read_split(split_file: Path, idx_map: dict):
    rows = []
    with open(split_file, "r") as f:
        for line in f:
            if line.strip():
                a, b, label = line.strip().split()
                a = idx_map[a]
                b = idx_map[b]
                rows.append(dict(a=a, b=b, label=label))
    return rows


def tokenize_ds(ds: Dataset, base_model: str, max_length: int = 1024):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    def tokenize(sample):
        return tokenizer(sample["func"], max_length=max_length, truncation=True)
    ds = ds.map(tokenize, batched=True, remove_columns=["func"])
    return ds


def load_hf(raw_dir: Path | None, cache_dir: Path, base_model: str):
    cache_funcs = cache_dir / "funcs.hf"
    cache_splits = cache_dir / "splits.hf"
    if cache_funcs.exists() and cache_splits.exists():
        return load_from_disk(cache_funcs), load_from_disk(cache_splits)

    if raw_dir is None:
        raise ValueError("raw_dir is required")

    ds_funcs = load_dataset("json", data_files=[str(raw_dir/"data.jsonl")], split="train")
    ds_funcs = tokenize_ds(ds_funcs, base_model)
    idx_map = {row["idx"]: i for i, row in enumerate(ds_funcs)}
    ds_splits = DatasetDict({
        "train": Dataset.from_list(read_split(raw_dir / "train.txt", idx_map)),
        "valid": Dataset.from_list(read_split(raw_dir / "valid.txt", idx_map)),
        "test": Dataset.from_list(read_split(raw_dir / "test.txt", idx_map)),
    })
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_funcs.save_to_disk(cache_dir / "funcs.hf")
    ds_splits.save_to_disk(cache_dir / "splits.hf")
    return ds_funcs, ds_splits


class CloneDataset(TorchDataset):
    def __init__(self, hf_dir: Path, base_model: str, split: str):
        funcs, splits = load_hf(None, hf_dir, base_model)
        self.funcs = funcs.with_format("torch")
        self.idx = splits[split].with_format("torch")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        item = self.idx[i]
        a = self.funcs[item["a"].item()]
        b = self.funcs[item["b"].item()]
        del a["idx"], b["idx"]
        label = int(item["label"])
        return {"a": a, "b": b, "label": label}
