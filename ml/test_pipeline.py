import torch

from src.datasets.dataloader import get_dataloaders
from src.models.model import get_model_by_name
from src.models.cbam_densenet import FocalLoss
from src.utils.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    BATCH_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
    PRETRAINED,
    DEVICE,
    LOSS_ALPHA,
    LOSS_GAMMA,
)


def main():
    print("=== Testing DataLoader ===")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
    )

    images, labels = next(iter(train_loader))

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Device:", DEVICE)

    assert images.ndim == 4
    assert images.shape[1:] == (3, 224, 224)
    assert labels.ndim == 2
    assert labels.shape[1] == NUM_CLASSES

    print("✓ DataLoader test passed")

    print("\n=== Testing Model ===")

    # IMPORTANT:
    # Use pretrained=False for this local pipeline test.
    model = get_model_by_name(
        MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    model = model.to(DEVICE)
    images = images.to(DEVICE)
    labels = labels.float().to(DEVICE)

    outputs = model(images)

    print("Model output:", outputs.shape)

    assert outputs.shape == (images.shape[0], NUM_CLASSES)

    print("✓ Model forward pass passed")

    print("\n=== Testing Focal Loss ===")

    criterion = FocalLoss(
        alpha=LOSS_ALPHA,
        gamma=LOSS_GAMMA,
    )

    loss = criterion(outputs, labels)

    print("Loss:", loss.item())

    assert torch.isfinite(loss)
    assert loss.item() > 0

    print("✓ Focal Loss test passed")

    print("\n=== ALL PIPELINE TESTS PASSED ===")


if __name__ == "__main__":
    main()