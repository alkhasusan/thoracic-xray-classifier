"""
dataloader.py

Creates PyTorch DataLoaders for training,
validation and testing.
"""

from torch.utils.data import DataLoader

from .dataset import ChestXrayDataset
from .transforms import train_transform, val_transform


def get_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    batch_size=16,
):
    train_dataset = ChestXrayDataset(
        train_csv,
        transform=train_transform,
    )

    val_dataset = ChestXrayDataset(
        val_csv,
        transform=val_transform,
    )

    test_dataset = ChestXrayDataset(
        test_csv,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader, test_loader