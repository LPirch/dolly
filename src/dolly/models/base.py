from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, SequenceClassifierOutput
from transformers import PretrainedConfig


class CloneDetector(nn.Module):
    name = None
    signature_columns = None

    def __init__(self, encoder, pos_weight: float = 1.0):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.config = self.encoder.config
        self.config.name_or_path = self.name
        self.config.num_labels = 1
        _hidden_size = self.config.hidden_size
        self.config.hidden_size *= 2  # initialize classification head with double the size
        self.classifier = RobertaClassificationHead(self.config)
        self.config.hidden_size = _hidden_size
        self.pos_weight = torch.tensor(pos_weight) if isinstance(pos_weight, float) else pos_weight
        self.pos_weight = torch.nn.Parameter(self.pos_weight, requires_grad=False)

    def _init_encoder(self):
        raise NotImplementedError("Must be implemented by subclass")

    def embed(self, sample):
        raise NotImplementedError("Must be implemented by subclass")

    def forward(self, a, b, label):
        emb_a = self.embed(a)
        emb_b = self.embed(b)
        emb = torch.cat([emb_a, emb_b], dim=1)
        logits = self.classifier(emb.view(emb.shape[0], 1, -1))
        return self._roberta_sequence_output(logits, emb, labels=label)

    def _roberta_sequence_output(self, logits, embeddings, labels=None):
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).float(), pos_weight=self.pos_weight)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=embeddings,
            attentions=None,
        )

    def save_model(self, save_directory: str, **kwargs):
        # Save the entire wrapper model state
        state_dict = self.state_dict()
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(state_dict, str(Path(save_directory) / "pytorch_model.bin"))

    @classmethod
    def load_model(cls, model_dir: str):
        config = cls.config_class.from_pretrained(model_dir)
        model = cls(config)
        model.load_state_dict(torch.load(str(Path(model_dir) / "pytorch_model.bin")))
        return model
