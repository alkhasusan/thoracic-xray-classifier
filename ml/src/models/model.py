"""
model.py

Model factory for multi-label Chest X-ray classification.

This file contains ONLY the plain ImageNet-pretrained backbones used in the
backbone comparison:

    densenet121_baseline, resnet50, efficientnet_b0, vgg16_bn, mobilenet_v3_large

No attention module appears here on purpose. The workflow is:
  1. train and compare the plain backbones (this registry),
  2. pick the best one,
  3. add an attention module (see attention.py) to that winner only,
     in its own file.

get_model_by_name() lets configs/model.yaml select a backbone by name.
"""

import torch.nn as nn
from torchvision import models
from .scse_densenet import get_scse_densenet121

from .backbones import (
    build_efficientnet_b0,
    build_mobilenet_v3_large,
    build_resnet50,
    build_vgg16_bn,
)


def get_model(num_classes=14, pretrained=True, **kwargs):
    """
    DenseNet-121 with a num_classes-way linear head (multi-label logits).
    Unchanged from the original baseline. Accepts **kwargs so every factory
    shares one call signature.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


# Name -> factory function. Keys are what configs/model.yaml's `model_name` uses.
MODEL_REGISTRY = {
    "densenet121_baseline": get_model,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "vgg16_bn": build_vgg16_bn,
    "mobilenet_v3_large": build_mobilenet_v3_large,
    "scse_densenet121": get_scse_densenet121
}


def get_model_by_name(model_name, num_classes=14, pretrained=True, **kwargs):
    """Build a model by name; raises a clear error for unknown names."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](
        num_classes=num_classes, pretrained=pretrained, **kwargs
    )
