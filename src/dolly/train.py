import os
import atexit
import signal
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, TrainerCallback


from dolly.models.llm import UniXcoder
from dolly.dataset import CloneDataset



class PairCollator:
    def __init__(self, tokenizer: AutoTokenizer, **collator_kwargs):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad = DataCollatorWithPadding(tokenizer, **collator_kwargs)

    def __call__(self, batch):
        a = self.pad([sample["a"] for sample in batch])
        b = self.pad([sample["b"] for sample in batch])
        return {"a": a, "b": b, "label": torch.tensor([sample["label"] for sample in batch])}


def load_model(model_name: str):
    if model_name == "unixcoder":
        return UniXcoder("microsoft/unixcoder-base-nine", freeze_encoder=True)
    else:
        raise ValueError(f"Model {model_name} not found")


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


def train(model_name: str, hf_dir: Path, model_root: Path):
    model_dir = model_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model.base_model)
    # tokenizer.model_input_names = ["a", "b", "label"]
    train_dataset = CloneDataset(hf_dir, model_name, "train")
    eval_dataset = CloneDataset(hf_dir, model_name, "valid")

    dataloader_workers = 0
    train_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=100,
        # gradient_accumulation_steps=1,
        per_device_eval_batch_size=100,
        dataloader_num_workers=dataloader_workers,
        dataloader_persistent_workers=dataloader_workers > 0,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
        logging_steps=500,
        eval_steps=1000,
        save_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PairCollator(
            tokenizer,
            padding="longest",
            max_length=1024,
            return_tensors="pt",
        ),
        callbacks=[TorchProfilerCallback(profile_dir=str(model_dir / "profiler"))],
    )
    trainer._signature_columns = ["a", "b", "label"]

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    trainer.evaluate()


if __name__ == "__main__":
    train("unixcoder", Path("data/big-clone-bench/hf"), Path("data/models"))
