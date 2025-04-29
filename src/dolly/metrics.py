import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


def compute_metrics(pred):
    """
    Calculates a few helpful metrics
    :param pred: list
    """
    true = pred.label_ids
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds[np.isnan(preds)] = 0
    predicted = preds > 0
    loss = F.binary_cross_entropy_with_logits(torch.tensor(preds).view(-1), torch.tensor(true).float()).item()
    positive_rate = (torch.sum(torch.tensor(predicted) == 1) / len(predicted)).item()
    return {
        "MCC": matthews_corrcoef(true, predicted),
        "F1": f1_score(true, predicted),
        "Acc": accuracy_score(true, predicted),
        "BAcc": balanced_accuracy_score(true, predicted),
        "Prec": precision_score(true, predicted),
        "Rec": recall_score(true, predicted),
        "auprec": average_precision_score(true, preds),
        "aucroc": roc_auc_score(true, preds),
        "positive_rate": positive_rate,
        "loss": loss,
    }
