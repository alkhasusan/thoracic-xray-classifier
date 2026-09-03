"""
dataset.py

PyTorch Dataset for the NIH ChestX-ray14 dataset.
"""

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    """
    Custom Dataset for NIH ChestX-ray14
    """

    DISEASES = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image = Image.open(row["Image Path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[self.DISEASES].values.astype("float32"),
            dtype=torch.float32,
            )

        return image, labels