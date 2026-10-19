"""
scse_densenet.py

DenseNet-121 (ImageNet-pretrained, the winner of the backbone comparison)
with an scSE attention block inserted after each of its 4 dense blocks.

Design choice: instead of re-implementing DenseNet's forward pass, each dense
block inside torchvision's own model is wrapped as
    denseblockN  ->  Sequential(denseblockN, SCSE(channels))
so the rest of the model (transition layers, final norm, ReLU, pooling,
classifier) is exactly torchvision's. The ONLY difference from the plain
DenseNet-121 baseline is the four scSE blocks, which makes the
"before vs after attention" comparison clean. The model keeps the same
`features` / `classifier` structure, so Grad-CAM code that hooks into
model.features keeps working.

Output: raw logits [batch, num_classes] (apply sigmoid for probabilities).
"""

import torch.nn as nn
from torchvision import models

from .attention import SCSE

# Output channels of each dense block in DenseNet-121
# (growth rate 32; layers per block 6 / 12 / 24 / 16):
#   64 + 6*32 = 256,  128 + 12*32 = 512,  256 + 24*32 = 1024,  512 + 16*32 = 1024
DENSE_BLOCK_CHANNELS = [
    ("denseblock1", 256),
    ("denseblock2", 512),
    ("denseblock3", 1024),
    ("denseblock4", 1024),
]


def get_scse_densenet121(num_classes=14, pretrained=True, reduction_ratio=16, **kwargs):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Wrap each dense block AFTER the pretrained weights are loaded.
    # scSE blocks are new layers, trained from random initialization.
    features = model.features
    for name, channels in DENSE_BLOCK_CHANNELS:
        block = getattr(features, name)
        setattr(features, name, nn.Sequential(block, SCSE(channels, reduction_ratio)))

    return model


if __name__ == "__main__":
    import torch

    model = get_scse_densenet121(num_classes=14, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    print("Output shape (expect [2, 14]):", tuple(out.shape))
    print("Parameters (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))