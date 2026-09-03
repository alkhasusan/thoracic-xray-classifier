"""
train.py
Main training script.
"""
from pathlib import Path

import torch

from src.datasets.dataloader import get_dataloaders
from src.datasets.dataset import ChestXrayDataset
from src.models.cbam_densenet import FocalLoss
from src.models.model import get_model_by_name
from src.training.trainer import train_one_epoch
from src.training.validate import validate
from src.utils.config import (
    BATCH_SIZE,
    CBAM_REDUCTION_RATIO,
    CBAM_SPATIAL_KERNEL_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LOSS_ALPHA,
    LOSS_GAMMA,
    MODEL_NAME,
    NUM_CLASSES,
    PRETRAINED,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    WEIGHT_DECAY,
)


def main():
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    # --------------------------
    # Data
    # --------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # --------------------------
    # Model
    # --------------------------
    # Picks baseline vs. CBAM-DenseNet based on model.yaml's model_name --
    # change that one line in the config to switch, no code edits needed.
    model = get_model_by_name(
        MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        reduction_ratio=CBAM_REDUCTION_RATIO,
        spatial_kernel_size=CBAM_SPATIAL_KERNEL_SIZE,
    )
    model = model.to(DEVICE)

    # --------------------------
    # Loss
    # --------------------------
    # FocalLoss (not plain BCE) per the project plan, to handle class
    # imbalance across the 14 disease labels. Used for both the baseline
    # and CBAM runs so the AUC-ROC comparison isn't confounded by a
    # loss-function difference between the two.
    criterion = FocalLoss(alpha=LOSS_ALPHA, gamma=LOSS_GAMMA)

    # --------------------------
    # Optimizer
    # --------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_auc = float("-inf")
    epochs_without_improvement = 0
    disease_names = ChestXrayDataset.DISEASES

    # --------------------------
    # Training Loop
    # --------------------------
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
        )
        print("✅ Training finished")

        print("➡️ Starting validation...")
        val_loss, val_auc, per_class_auc = validate(
            model, val_loader, criterion, DEVICE, disease_names=disease_names,
        )
        print("✅ Validation finished")

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Mean AUC-ROC: {val_auc:.4f}")
        for name, auc in per_class_auc.items():
            if auc == auc:  # not NaN
                print(f"   {name}: {auc:.4f}")
            else:
                print(f"   {name}: N/A (only one class present in val split)")

        # Save best model based on mean AUC-ROC (per train.yaml's
        # checkpoints.monitor_metric: val_auc_roc), not loss -- AUC-ROC is
        # the metric the project plan actually compares against the baseline.
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0

            save_path = Path(CHECKPOINT_DIR) / f"best_{MODEL_NAME}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path} (val_auc={val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if (
            EARLY_STOPPING_PATIENCE is not None
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print(f"\nBest validation AUC-ROC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()