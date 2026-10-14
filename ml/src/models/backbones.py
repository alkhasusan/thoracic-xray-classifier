"""
backbones.py

Plain (no CBAM) ImageNet-pretrained backbones for the backbone comparison.
Each builder swaps the final layer for a num_classes-way linear head and
returns a model whose forward() gives RAW LOGITS [batch, num_classes]
(sigmoid is applied later, exactly like CBAMDenseNet121).

All builders accept **kwargs so train.py can keep passing
reduction_ratio / spatial_kernel_size to every model without errors.
"""

import torch.nn as nn
import torchvision.models as tvm


def _w(pretrained, weights_enum):
    return weights_enum.DEFAULT if pretrained else None


def build_resnet50(num_classes=14, pretrained=True, **kwargs):
    m = tvm.resnet50(weights=_w(pretrained, tvm.ResNet50_Weights))
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_vgg16_bn(num_classes=14, pretrained=True, **kwargs):
    m = tvm.vgg16_bn(weights=_w(pretrained, tvm.VGG16_BN_Weights))
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m


def build_efficientnet_b0(num_classes=14, pretrained=True, **kwargs):
    m = tvm.efficientnet_b0(weights=_w(pretrained, tvm.EfficientNet_B0_Weights))
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def build_vit_b_16(num_classes=14, pretrained=True, **kwargs):
    # ViT requires exactly 224x224 input (your transforms already resize to 224)
    m = tvm.vit_b_16(weights=_w(pretrained, tvm.ViT_B_16_Weights))
    m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    return m


def build_mobilenet_v3_large(num_classes=14, pretrained=True, **kwargs):
    m = tvm.mobilenet_v3_large(weights=_w(pretrained, tvm.MobileNet_V3_Large_Weights))
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


BACKBONE_BUILDERS = {
    "resnet50": build_resnet50,
    "vgg16_bn": build_vgg16_bn,
    "efficientnet_b0": build_efficientnet_b0,
    "vit_b_16": build_vit_b_16,
    "mobilenet_v3_large": build_mobilenet_v3_large,
}


if __name__ == "__main__":
    # Quick shape check (random weights, no download): every model should print [2, 14]
    import torch

    x = torch.randn(2, 3, 224, 224)
    for name, builder in BACKBONE_BUILDERS.items():
        out = builder(num_classes=14, pretrained=False)(x)
        print(f"{name}: {tuple(out.shape)}")