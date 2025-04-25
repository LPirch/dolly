from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


class UniXcoder(nn.Module):
    def __init__(self, base_model: str, freeze_encoder: bool = False, use_peft: bool = True):
        super().__init__()
        self.base_model = base_model
        self.encoder = self._init_encoder(use_peft, freeze_encoder)
        self.encoder.config.num_labels = 1
        self.encoder.config.hidden_size *= 2
        self.classifier = RobertaClassificationHead(self.encoder.config)

    def _init_encoder(self, use_peft: bool, freeze_encoder: bool):
        encoder = RobertaModel.from_pretrained(self.base_model, add_pooling_layer=False)
        if use_peft:
            encoder = get_peft_model(encoder, get_peft_config())
            encoder.print_trainable_parameters()
        if freeze_encoder:
            encoder = freeze(encoder)
        return encoder

    def forward(self, a, b, label):
        emb_a = self.encoder(**a).last_hidden_state[:, 0, :]
        emb_b = self.encoder(**b).last_hidden_state[:, 0, :]
        emb = torch.cat([emb_a, emb_b], dim=1)
        logits = self.classifier(emb.view(emb.shape[0], 1, -1))
        return self._roberta_sequence_output(logits, emb, labels=label)

    def _roberta_sequence_output(self, logits, embeddings, labels=None):
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).float())
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=embeddings,
            attentions=None,
        )

def get_peft_config():
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    return peft_config
