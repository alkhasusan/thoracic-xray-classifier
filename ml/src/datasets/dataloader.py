"""
dataloader.py

Creates PyTorch DataLoaders for training and testing
(no validation split — merged into train per updated requirements).
"""

from torch.utils.data import DataLoader

from .dataset import ChestXrayDataset
from .transforms import train_transform, val_transform


def get_dataloaders(
    train_csv,
    test_csv,
    batch_size=16,
    image_root=None,  # root folder prepended to each CSV's "Image Path"
):
    train_dataset = ChestXrayDataset(
        train_csv,
        transform=train_transform,
        image_root=image_root,
    )

    test_dataset = ChestXrayDataset(
        test_csv,
        transform=val_transform,  # reuse val_transform (no augmentation) for eval
        image_root=image_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, test_loader