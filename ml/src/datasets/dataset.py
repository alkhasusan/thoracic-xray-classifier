"""
dataset.py
PyTorch Dataset for the NIH ChestX-ray14 dataset.

Optional image cache: if cache_dir is given, images are read from a pre-resized
224x224 uint8 array (images_224.npy + names.txt, built by cache_images.py)
instead of decoding the original 1024x1024 PNGs. The model sees identical
pixels, because transforms.Resize((224, 224)) is the first transform anyway
(and is a no-op on an image that is already 224x224).
"""

from pathlib import Path

import numpy as np
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

    def __init__(self, csv_file, transform=None, image_root=None, cache_dir=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # If provided, joined onto each row's "Image Path". If "Image Path"
        # already holds a full absolute path, pathlib ignores image_root.
        self.image_root = Path(image_root) if image_root else None

        # Optional 224x224 cache (see cache_images.py)
        self._cache = None          # opened lazily, once per DataLoader worker
        self._cache_path = None
        self.rows = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            self._cache_path = cache_dir / "images_224.npy"
            names = (cache_dir / "names.txt").read_text().split("\n")
            name_to_row = {n: i for i, n in enumerate(names)}
            self.rows = [name_to_row[Path(p).name] for p in self.data["Image Path"]]

    def __len__(self):
        return len(self.data)

    def _load_cached(self, idx):
        if self._cache is None:
            self._cache = np.load(self._cache_path, mmap_mode="r")
        arr = np.array(self._cache[self.rows[idx]])   # copy out of the memmap
        return Image.fromarray(arr).convert("RGB")    # 2D uint8 -> "L" -> RGB

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.rows is not None:
            image = self._load_cached(idx)
        else:
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