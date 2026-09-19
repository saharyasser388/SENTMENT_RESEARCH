import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    model,
    data_loader,
    device,
):
    """
    Evaluate a sentiment classifier.

    Returns:
        {
            "loss": ...,
            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1": ...,
            "confusion_matrix": ...
        }
    """

    # -------------------------------------------------
    # 1. Evaluation mode
    # -------------------------------------------------

    model.eval()

    # Same loss function used during training.
    criterion = nn.CrossEntropyLoss()


    # -------------------------------------------------
    # 2. Storage
    # -------------------------------------------------

    total_loss = 0.0

    all_labels = []
    all_predictions = []


    # -------------------------------------------------
    # 3. No gradients during evaluation
    # -------------------------------------------------

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)


            # -----------------------------------------
            # Forward pass
            # -----------------------------------------

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


            # -----------------------------------------
            # Loss
            # -----------------------------------------

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()


            # -----------------------------------------
            # Convert logits → predicted class
            # -----------------------------------------

            predictions = torch.argmax(
                logits,
                dim=-1
            )


            # -----------------------------------------
            # Move results back to CPU
            # -----------------------------------------

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )


    # -------------------------------------------------
    # 4. Average loss
    # -------------------------------------------------

    average_loss = (
        total_loss / len(data_loader)
    )


    # -------------------------------------------------
    # 5. Classification metrics
    # -------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )


    # -------------------------------------------------
    # 6. Return results
    # -------------------------------------------------

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }