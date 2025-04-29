import json
import atexit
import signal
from pathlib import Path
from math import ceil

import torch
import numpy as np
from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback
import typer
from loguru import logger

from dolly.models import load_model
from dolly.dataset import CloneDataset, PairCollator
from dolly.metrics import compute_metrics


app = typer.Typer()


class TorchProfilerCallback(TrainerCallback):
    def __init__(self, profile_dir="tb_profile", wait=1, warmup=1, active=3, repeat=1):
        self.profile_dir = profile_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.profiler = None

        # Ensure profiler cleanup on exit
        atexit.register(self._safe_exit)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print(f"\n[Profiler] Caught signal {sig}, saving trace...")
        self._safe_exit()
        exit(0)

    def _safe_exit(self):
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            print(f"[Profiler] Trace written to: {self.profile_dir}")
            self.profiler = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_dir),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            profile_memory=True
        )
        self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        self._safe_exit()


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_predictions = []
        self.epoch_labels = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        self.epoch_predictions.append(outputs.logits.detach().cpu().numpy())
        self.epoch_labels.append(inputs["label"].cpu().numpy())
        return (loss, outputs) if return_outputs else loss


class TrainingMetricsCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_predictions = np.concatenate(self._trainer.epoch_predictions)
        epoch_labels = np.concatenate(self._trainer.epoch_labels)

        pred_wrapper = lambda *args: None  # create dummy object
        pred_wrapper.predictions = epoch_predictions
        pred_wrapper.label_ids = epoch_labels
        metrics = compute_metrics(pred_wrapper)

        logger.info(
            f"Epoch {state.epoch} accuracy: {metrics['Acc']:.3f}, loss: {metrics['loss']:.5f}, "
            f"positive_rate: {metrics['positive_rate']:.3f}, precision: {metrics['Prec']:.3f}, "
            f"recall: {metrics['Rec']:.3f}, f1: {metrics['F1']:.3f}"
        )

        # Clear stored predictions, labels, and loss for the next epoch
        self._trainer.epoch_predictions = []
        self._trainer.epoch_labels = []
        return None


@app.callback(invoke_without_command=True)
def train(
    model_name: str = typer.Option(default="llm"),
    hf_dir: Path = typer.Option(default=Path("data/big-clone-bench/hf-pyg")),
    model_root: Path = typer.Option(default=Path("data/models")),
    subsample: float = typer.Option(default=1.0),
    seed: int = typer.Option(default=42),
):
    train_dataset = CloneDataset(hf_dir, "train", subsample, seed)
    pos_weight = train_dataset.get_pos_weight()
    eval_dataset = CloneDataset(hf_dir, "valid", subsample, seed)
    test_dataset = CloneDataset(hf_dir, "test", subsample, seed)

    model_dir = model_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_ckpt = model_dir / "best_model.ckpt"
    model = load_model(model_name, pos_weight=pos_weight)
    tokenizer = AutoTokenizer.from_pretrained(model.base_encoder)

    dataloader_workers = 0
    batch_size = 1000
    log_unit = ceil(subsample*10_000/batch_size)  # reference unit for logging/saving steps
    logger.info(f"logging every {log_unit} steps")
    train_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=dataloader_workers,
        dataloader_persistent_workers=dataloader_workers > 0,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
        logging_steps=log_unit,
        eval_steps=2*log_unit,
        save_steps=2*log_unit,
        save_total_limit=2,
        warmup_steps=ceil(50*subsample),
        weight_decay=0.01,
        seed=seed,
        num_train_epochs=2,
    )

    trainer = CustomTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=PairCollator(model.collate_fn, tokenizer=tokenizer),
        # callbacks=[TorchProfilerCallback(profile_dir=str(model_dir / "profiler"))],
    )
    trainer.add_callback(TrainingMetricsCallback(trainer))
    trainer._signature_columns = model.signature_columns

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Starting training. Number of trainable parameters: {num_params:.2e}")
    train_output = trainer.train()
    trainer.model.save_model(best_model_ckpt)
    tokenizer.save_pretrained(best_model_ckpt)
    with open(model_dir / "train_output.json", "w") as f:
        json.dump(train_output, f, indent=4)

    logger.info("Evaluating on test set...")
    metrics = trainer.evaluate(test_dataset)
    with open(model_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
