"""
train.py
Main training script.
"""
from pathlib import Path
import json

import torch
from src.datasets.dataloader import get_dataloaders
from src.datasets.dataset import ChestXrayDataset
from src.evaluation.confusion_matrix import (
    get_predictions, best_thresholds, compute_confusion_matrices,
    print_confusion_matrices,
)
from src.models.cbam_densenet import FocalLoss
from src.models.model import get_model_by_name
from src.training.trainer import train_one_epoch
from src.training.validate import validate
from src.utils.config import (
    BATCH_SIZE, CBAM_REDUCTION_RATIO, CBAM_SPATIAL_KERNEL_SIZE,
    CHECKPOINT_DIR, DEVICE, EPOCHS, IMAGE_ROOT, LEARNING_RATE,
    LOSS_ALPHA, LOSS_GAMMA, MODEL_NAME, NUM_CLASSES, PRETRAINED, RESUME,
    TEST_CSV, TRAIN_CSV, WEIGHT_DECAY,
)


def main():
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    # No val split: train.csv includes what used to be val.csv.
    # test_loader is held out purely for final evaluation.
    train_loader, test_loader = get_dataloaders(
        train_csv=TRAIN_CSV, test_csv=TEST_CSV,
        batch_size=BATCH_SIZE, image_root=IMAGE_ROOT,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    model = get_model_by_name(
        MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED,
        reduction_ratio=CBAM_REDUCTION_RATIO, spatial_kernel_size=CBAM_SPATIAL_KERNEL_SIZE,
    )
    model = model.to(DEVICE)

    criterion = FocalLoss(alpha=LOSS_ALPHA, gamma=LOSS_GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    final_model_path = Path(CHECKPOINT_DIR) / f"final_{MODEL_NAME}.pth"
    resume_checkpoint_path = Path(CHECKPOINT_DIR) / f"last_{MODEL_NAME}_checkpoint.pth"

    disease_names = ChestXrayDataset.DISEASES

    # --------------------------
    # Resume, if enabled and a checkpoint exists
    # --------------------------
    start_epoch = 0

    if RESUME and resume_checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {resume_checkpoint_path}")
        ckpt = torch.load(resume_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch + 1}")
    else:
        print("\nNo checkpoint to resume from (or resume disabled) - starting fresh.")

    # --------------------------
    # Training loop: fixed epochs, no validation-based early stopping
    # --------------------------
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print("Training finished")
        print(f"Train Loss: {train_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, resume_checkpoint_path)

    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete. Saved final model to {final_model_path}")

    # --------------------------
    # Final evaluation on the held-out test set
    # --------------------------
    print("\nEvaluating on test set...")
    test_loss, test_auc, per_class_auc = validate(
        model, test_loader, criterion, DEVICE, disease_names=disease_names,
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Mean AUC-ROC: {test_auc:.4f}")
    for name, auc in per_class_auc.items():
        if auc == auc:
            print(f"   {name}: {auc:.4f}")
        else:
            print(f"   {name}: N/A (only one class present in test split)")

    # --------------------------
    # Confusion matrices on the test set
    # --------------------------
    y_true, y_probs = get_predictions(model, test_loader, DEVICE)

    print("\nConfusion matrices, fixed threshold = 0.5")
    matrices_fixed = compute_confusion_matrices(y_true, y_probs, disease_names, threshold=0.5)
    print_confusion_matrices(matrices_fixed)

    thr = best_thresholds(y_true, y_probs)
    print("\nConfusion matrices, per-class tuned thresholds (Youden's J)")
    for name, t in zip(disease_names, thr):
        print(f"   {name}: threshold = {t:.4f}")
    matrices_tuned = compute_confusion_matrices(y_true, y_probs, disease_names, threshold=thr)
    print_confusion_matrices(matrices_tuned)

    results = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_loss": float(test_loss),
        "test_mean_auc": float(test_auc),
        "per_class_auc": {k: (None if v != v else float(v)) for k, v in per_class_auc.items()},
        "tuned_thresholds": {n: float(t) for n, t in zip(disease_names, thr)},
        "confusion_fixed_0.5": matrices_fixed,
        "confusion_tuned": matrices_tuned,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"results_{MODEL_NAME}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {OUTPUT_DIR / f'results_{MODEL_NAME}.json'}")


if __name__ == "__main__":
    main()