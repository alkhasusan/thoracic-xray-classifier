"""
dataset.py
PyTorch Dataset for the NIH ChestX-ray14 dataset.
"""

from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    """
    Custom Dataset for NIH ChestX-ray14
    """

    DISEASES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
        "Pleural_Thickening", "Pneumonia", "Pneumothorax",
    ]

    def __init__(self, csv_file, transform=None, image_root=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # NEW: if provided, joined onto each row's "Image Path".
        # If "Image Path" already holds a full absolute path, Path's
        # join operator ignores image_root automatically (pathlib
        # behavior: joining an absolute path onto anything discards
        # the left side) -- so this is safe with your current CSVs
        # even before you switch them to relative paths.
        self.image_root = Path(image_root) if image_root else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = row["Image Path"]
        if self.image_root is not None:
            img_path = self.image_root / img_path

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[self.DISEASES].values.astype("float32"),
            dtype=torch.float32,
        )

        return image, labels