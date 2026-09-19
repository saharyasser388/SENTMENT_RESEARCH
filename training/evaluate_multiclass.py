"""Evaluation shared by both Study 2 classifiers."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_multiclass(model, data_loader, device, criterion, num_classes=7):
    model.eval()
    total_loss = 0.0
    example_count = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in data_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            example_count += labels.size(0)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    if not example_count:
        raise ValueError("Cannot evaluate an empty data loader")
    labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="weighted", zero_division=0
    )
    return {
        "loss": total_loss / example_count,
        "accuracy": accuracy_score(targets, predictions),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_f1": weighted[2],
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }
