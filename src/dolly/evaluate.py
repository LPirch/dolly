import json
from pathlib import Path

import torch
import typer
from loguru import logger
from transformers import TrainingArguments, Trainer, AutoTokenizer

from dolly.models import load_trained
from dolly.dataset import CloneDataset, PairCollator
from dolly.metrics import compute_metrics


app = typer.Typer()


@app.callback(invoke_without_command=True)
def evaluate(
    model_dir: Path = typer.Argument(default="data/models/llm"),
    hf_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf-pyg")),
):
    test_dataset = CloneDataset(hf_dir, "test")
    best_model_ckpt = model_dir / "best_model.ckpt"
    model = load_trained(best_model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(best_model_ckpt)

    dataloader_workers = 0
    batch_size = 1000
    train_args = TrainingArguments(
        output_dir=None,
        eval_strategy="epoch",
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=dataloader_workers,
        dataloader_persistent_workers=dataloader_workers > 0,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=PairCollator(model.collate_fn, tokenizer),
    )
    trainer._signature_columns = model.signature_columns
    logger.info("Evaluating on test set...")
    metrics = trainer.evaluate(test_dataset)
    with open(model_dir / "eval_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Test finished. Results:\n{json.dumps(metrics, indent=4)}")
