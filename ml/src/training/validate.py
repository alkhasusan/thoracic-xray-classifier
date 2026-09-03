"""
validate.py
Validation loop.

Computes validation loss (for monitoring/early-stopping feel) AND
per-class + mean AUC-ROC -- AUC-ROC is the project's actual evaluation
metric, used to compare CBAM-DenseNet against the DenseNet-121 baseline.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def validate(model, dataloader, criterion, device, disease_names=None):
    """
    Returns:
        epoch_loss:    float, mean loss over the validation set.
        mean_auc:      float, macro-average AUC-ROC across all classes
                        that had both positive and negative examples in
                        this validation set. NaN if no class qualified.
        per_class_auc: dict[str, float], AUC-ROC per disease. A class
                        with only one label value present (e.g. all-negative
                        in a small val split) gets NaN rather than crashing
                        -- roc_auc_score is undefined in that case.
    """
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)  # raw logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    num_classes = all_labels.shape[1]

    per_class_auc = {}
    valid_aucs = []

    for i in range(num_classes):
        name = disease_names[i] if disease_names is not None else f"class_{i}"
        col_labels = all_labels[:, i]

        if len(np.unique(col_labels)) < 2:
            # Can't compute AUC with only one class present in this split.
            per_class_auc[name] = float("nan")
            continue

        auc = roc_auc_score(col_labels, all_probs[:, i])
        per_class_auc[name] = auc
        valid_aucs.append(auc)

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    return epoch_loss, mean_auc, per_class_auc