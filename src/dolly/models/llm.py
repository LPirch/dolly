from pathlib import Path

from transformers import RobertaModel, PretrainedConfig, PreTrainedModel, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
import torch


from dolly.models.base import CloneDetector


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def _init_encoder(base_encoder: str, use_peft: bool, freeze_encoder: bool):
    encoder = RobertaModel.from_pretrained(base_encoder, add_pooling_layer=False)
    if use_peft:
        encoder = get_peft_model(encoder, get_peft_config())
        encoder.print_trainable_parameters()
    if freeze_encoder:
        encoder = freeze(encoder)
    return encoder


class UniXcoderConfig(PretrainedConfig):
    model_type = "unixcoder_clone"

    def __init__(
        self,
        freeze_encoder: bool = False,
        use_peft: bool = True,
        pos_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.freeze_encoder = freeze_encoder
        self.use_peft = use_peft
        self.pos_weight = pos_weight


class UniXcoder(CloneDetector, PreTrainedModel):
    name = "unixcoder"
    signature_columns = ["a", "b", "label"]
    encoder_columns = ["input_ids", "attention_mask"]
    base_encoder = "microsoft/unixcoder-base-nine"
    config_class = UniXcoderConfig

    def __init__(self, config: UniXcoderConfig):
        PreTrainedModel.__init__(self, config)
        pos_weight = torch.tensor([config.pos_weight])
        encoder = _init_encoder(self.base_encoder, config.use_peft, config.freeze_encoder)
        CloneDetector.__init__(self, encoder, pos_weight)

    def save_pretrained(self, save_directory: str, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, str(Path(save_directory) / "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, model_name, *model_args, **kwargs):
        config = cls.config_class.from_pretrained(model_name)
        model = cls(config)
        state_dict = torch.load(
            str(Path(model_name) / "pytorch_model.bin")
        )
        model.load_state_dict(state_dict)
        return model

    def embed(self, sample):
        return self.encoder(**sample).last_hidden_state[:, 0, :]

    @staticmethod
    def collate_fn(batch, tokenizer):
        pad = DataCollatorWithPadding(tokenizer, padding="longest", max_length=1024, return_tensors="pt")
        batch = pad([_filter_columns(sample, UniXcoder.encoder_columns) for sample in batch])
        return batch


def _filter_columns(sample, columns):
    return {k: v for k, v in sample.items() if k in columns}


def get_peft_config():
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    return peft_config
