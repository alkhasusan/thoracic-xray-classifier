"""
model.py

Model factory for multi-label Chest X-ray classification.

Contains two model families:
  1. get_model()      -- the original DenseNet-121 baseline (unchanged).
                          Kept because the project plan requires comparing
                          CBAM-DenseNet's AUC-ROC directly against this
                          baseline -- both models need to remain runnable.
  2. get_cbam_model()  -- the new CBAM-DenseNet121 architecture (backbone +
                          4 CBAM attention blocks + classifier head).

get_model_by_name() is a thin dispatcher so configs/model.yaml can select
which one to build by name, without training/train.py needing to know
the difference.

⚠️ ASSUMPTION FLAGGED:
get_cbam_model() below assumes CBAMDenseNet121's constructor signature is
    CBAMDenseNet121(num_classes: int = 14, pretrained: bool = True)
mirroring get_model()'s existing signature. This has NOT been confirmed
against the actual cbam_densenet.py file (not shared in this chat yet).
If the real constructor differs -- e.g. it takes a `backbone` argument,
or pulls pretrained weights a different way -- update get_cbam_model()
accordingly before running this.
"""

import torch.nn as nn
from torchvision import models

from .cbam_densenet import CBAMDenseNet121


def get_model(num_classes=14, pretrained=True, **kwargs):
    """
    Returns a DenseNet-121 model for multi-label classification.
    This is the baseline model -- unchanged from the original repo.

    Accepts **kwargs so it has the same call signature as get_cbam_model()
    and can be dispatched to interchangeably by get_model_by_name() --
    e.g. CBAM-specific kwargs like reduction_ratio are simply ignored here.
    """
    # Load pretrained DenseNet121
    weights = (
        models.DenseNet121_Weights.DEFAULT
        if pretrained
        else None
    )
    model = models.densenet121(weights=weights)

    # Number of features coming into classifier
    in_features = model.classifier.in_features

    # Replace classifier
    model.classifier = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def get_cbam_model(num_classes=14, pretrained=True, reduction_ratio=16, spatial_kernel_size=7, **kwargs):
    """
    Returns the CBAM-DenseNet121 model: a DenseNet-121 backbone with a
    CBAM (Convolutional Block Attention Module) inserted after each of
    the 4 dense blocks, followed by a GAP -> FC(num_classes) -> Sigmoid
    head for multi-label classification.

    reduction_ratio and spatial_kernel_size are forwarded to every CBAM
    block -- these now actually flow from model.yaml, through config.py
    and train.py, into the real model. (Previously this function silently
    dropped them and CBAMDenseNet121 always fell back to its own hardcoded
    defaults, which happened to equal model.yaml's defaults -- so the bug
    was invisible until someone changed the YAML values and nothing
    happened.)
    """
    model = CBAMDenseNet121(
        num_classes=num_classes,
        pretrained=pretrained,
        reduction_ratio=reduction_ratio,
        spatial_kernel_size=spatial_kernel_size,
    )
    return model


# Name -> factory function, used by configs/model.yaml's `model_name` field
# so training code can build either model from a single string, e.g.:
#   model = get_model_by_name(cfg["model_name"], num_classes=14, pretrained=True)
MODEL_REGISTRY = {
    "densenet121_baseline": get_model,
    "cbam_densenet121": get_cbam_model,
}


def get_model_by_name(model_name, num_classes=14, pretrained=True, **kwargs):
    """
    Dispatch to the correct factory function by name. Raises a clear
    error if the name isn't recognized, instead of silently defaulting
    to the baseline.

    **kwargs (e.g. reduction_ratio, spatial_kernel_size) are forwarded
    to whichever factory is selected. get_model() ignores ones it doesn't
    use; get_cbam_model() uses them.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )
    factory_fn = MODEL_REGISTRY[model_name]
    return factory_fn(num_classes=num_classes, pretrained=pretrained, **kwargs)