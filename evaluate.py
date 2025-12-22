# evaluate.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score


def _onset_precision_recall(pred_skel, true_skel):
    """
    pred_skel, true_skel: tensors  [B, T, 3]  (0-1 thresholded)
    """
    pred = pred_skel.reshape(-1).cpu().numpy().round()
    true = true_skel.reshape(-1).cpu().numpy().round()
    p = precision_score(true, pred, zero_division=0)
    r = recall_score(true, pred, zero_division=0)
    return p, r


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Return all metrics in a dictionary"""
    model.eval()

    cls_preds, cls_labels = [], []
    dev_preds, dev_labels = [], []
    skel_preds, skel_labels = [], []

    for batch in loader:
        x   = batch["input"].to(device)
        y_skel = batch["skeleton"].to(device)
        y_cls  = batch["rhythm_class"].cpu().numpy()
        y_dev  = batch["rhythm_deviation"].cpu().numpy()

        pred_skel, pred_cls_logits, pred_dev = model(x)

        # Classification
        cls_preds.extend(pred_cls_logits.argmax(dim=1).cpu().numpy())
        cls_labels.extend(y_cls)

        # Regression
        dev_preds.extend(pred_dev.cpu().numpy())
        dev_labels.extend(y_dev)

        # Structure
        skel_preds.append(pred_skel.cpu())
        skel_labels.append(y_skel.cpu())

    # === Metrics ===
    acc = accuracy_score(cls_labels, cls_preds)
    f1  = f1_score(cls_labels, cls_preds, average="macro")
    mae = mean_absolute_error(dev_labels, dev_preds)

    p, r = _onset_precision_recall(
        torch.cat(skel_preds, 0), torch.cat(skel_labels, 0)
    )

    return {
        "acc": acc,
        "f1":  f1,
        "mae": mae,
        "onset_precision": p,
        "onset_recall":    r,
    }
