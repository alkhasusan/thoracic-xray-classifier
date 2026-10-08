"""
confusion_matrix.py

Generates per-class confusion matrices for multi-label chest X-ray
classification. Since each image can have multiple disease labels,
a single confusion matrix doesn't apply the way it would for
single-label classification — instead, each of the 14 diseases gets
its own 2x2 (TN/FP/FN/TP) matrix, treating that disease as an
independent binary "present vs absent" prediction.

Usage:
    from confusion_matrix import get_predictions, compute_confusion_matrices, print_confusion_matrices

    y_true, y_probs = get_predictions(model, test_loader, device)
    matrices = compute_confusion_matrices(y_true, y_probs, DISEASES, threshold=0.5)
    print_confusion_matrices(matrices)
"""

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """
    Runs the model over a full dataloader and collects true labels
    and predicted probabilities (after sigmoid).

    Returns:
        y_true:  numpy array, shape [n_samples, n_classes]
        y_probs: numpy array, shape [n_samples, n_classes]
    """
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_labels.append(labels.numpy())
        all_probs.append(probs)

    y_true = np.concatenate(all_labels, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)
    return y_true, y_probs


def compute_confusion_matrices(y_true, y_probs, class_names, threshold=0.5):
    """
    Thresholds predicted probabilities and computes one 2x2 confusion
    matrix per class.

    Returns:
        dict mapping disease name -> {"tn":, "fp":, "fn":, "tp":}
    """
    y_pred = (y_probs >= threshold).astype(int)

    # shape: [n_classes, 2, 2], each [[tn, fp], [fn, tp]]
    cms = multilabel_confusion_matrix(y_true, y_pred)

    results = {}
    for i, name in enumerate(class_names):
        tn, fp, fn, tp = cms[i].ravel()
        results[name] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    return results


def print_confusion_matrices(matrices):
    """Pretty-prints the per-class confusion matrices and basic derived metrics."""
    print(f"{'Disease':<20} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6} {'Precision':>10} {'Recall':>8}")
    print("-" * 70)
    for name, m in matrices.items():
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"{name:<20} {tn:>6} {fp:>6} {fn:>6} {tp:>6} {precision:>10.4f} {recall:>8.4f}")


def plot_confusion_matrices(matrices, class_names, save_path=None):
    """
    Plots all 14 confusion matrices as a grid of heatmaps.
    Requires matplotlib and seaborn.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(class_names)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, name in enumerate(class_names):
        m = matrices[name]
        matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"],
            ax=axes[i],
        )
        axes[i].set_title(name, fontsize=10)

    # hide any unused subplot slots
    for j in range(len(class_names), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()