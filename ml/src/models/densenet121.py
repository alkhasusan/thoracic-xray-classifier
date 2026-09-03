"""
densenet121.py

From-scratch PyTorch implementation of DenseNet-121, written to match
torchvision's internal module names exactly (features.conv0, features.norm0,
features.denseblock1, features.transition1, ... features.norm5).

Why match torchvision's names instead of inventing our own?
--------------------------------------------------------------
Because it lets us load the official ImageNet-pretrained weights with a
plain `load_state_dict()` call -- no manual key remapping, no risk of
silently loading garbage into the wrong layer. The *code* is fully our
own (every nn.Module here is written from scratch, not imported from
torchvision.models), but the resulting state_dict is 1:1 compatible with
torchvision's pretrained checkpoint.

Architecture recap (DenseNet-121):
  - Stem: 7x7 conv (stride 2) -> BN -> ReLU -> 3x3 maxpool (stride 2)
  - 4 dense blocks with block_config = (6, 12, 24, 16) layers
  - growth_rate = 32, bn_size = 4 (bottleneck width multiplier)
  - Transition layers (1x1 conv + 2x2 avgpool, stride 2) between dense blocks
  - Final BN (norm5) -> ReLU -> global avg pool -> classifier

This file intentionally exposes the *feature maps after each dense block*
(before its transition layer) via `return_intermediate=True`, since that's
exactly what the CBAM-DenseNet model needs to hook into.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _DenseLayer(nn.Module):
    """
    One dense layer: BN-ReLU-Conv1x1 (bottleneck) -> BN-ReLU-Conv3x3 (growth),
    with the input concatenated onto the output (dense connectivity).
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features, bn_size * growth_rate,
            kernel_size=1, stride=1, bias=False
        )

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.drop_rate = float(drop_rate)

    def forward(self, prev_features):
        # prev_features: list of feature tensors from all preceding layers
        concated = torch.cat(prev_features, 1)
        bottleneck = self.conv1(self.relu1(self.norm1(concated)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _DenseBlock(nn.ModuleDict):
    """
    A dense block: a stack of _DenseLayer modules where each layer receives
    the concatenation of ALL previous layers' outputs (including the block
    input) as its input.
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.0):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """
    Transition layer between dense blocks: BN-ReLU-Conv1x1 (halves channels)
    followed by 2x2 average pooling (halves spatial resolution).
    """
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(num_input_features, num_output_features,
                      kernel_size=1, stride=1, bias=False),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


# ---------------------------------------------------------------------------
# Full DenseNet-121
# ---------------------------------------------------------------------------

class DenseNet121(nn.Module):
    """
    From-scratch DenseNet-121. Module names mirror torchvision.models.densenet121
    exactly, so state_dict keys match and pretrained ImageNet weights load
    with a plain `load_state_dict()`.

    Args:
        growth_rate: channels added per dense layer (default 32).
        block_config: number of layers in each of the 4 dense blocks
            (default (6, 12, 24, 16) -> this is what makes it "121").
        num_init_features: channels after the stem conv (default 64).
        bn_size: bottleneck width multiplier (default 4).
        drop_rate: dropout rate inside dense layers (default 0).
        num_classes: size of the final FC layer. Set to None to drop the
            classifier entirely (useful when this backbone feeds into a
            downstream head, e.g. CBAM-DenseNet).
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.0, num_classes=1000):
        super().__init__()

        # ---- Stem ----
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7,
                                 stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # ---- Dense blocks + transitions ----
        num_features = num_init_features
        # Output channel counts after each dense block (before its transition),
        # i.e. exactly what CBAM needs to know to size its attention modules.
        self.block_output_channels = []

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            self.block_output_channels.append(num_features)

            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        # ---- Final BN (torchvision calls this norm5) ----
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        self.num_features = num_features  # 1024 for DenseNet-121

        # ---- Classifier ----
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            self.classifier = None

        self._init_weights()

    def _init_weights(self):
        # Matches torchvision's init scheme; irrelevant once pretrained
        # weights are loaded, but keeps the model sane if used from scratch.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_intermediate=False):
        """
        Standard forward pass.

        If return_intermediate=True, also returns a list of the 4 feature
        maps captured right after each dense block (before its transition
        layer / before norm5+pool for the last block) -- this is the hook
        point CBAM attaches to.
        """
        intermediates = [] if return_intermediate else None

        out = x
        for name, module in self.features.named_children():
            out = module(out)
            if return_intermediate and name.startswith("denseblock"):
                intermediates.append(out)

        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        logits = self.classifier(out) if self.classifier is not None else out

        if return_intermediate:
            return logits, intermediates
        return logits


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

def load_pretrained_densenet121(num_classes=None, drop_rate=0.0):
    """
    Builds our from-scratch DenseNet121 and loads ImageNet-pretrained
    weights from torchvision directly (no remapping needed, since our
    module names match torchvision's exactly).

    Args:
        num_classes: if None, the classifier is dropped (backbone-only mode,
            what CBAM-DenseNet wants). If an int, a fresh randomly-initialized
            FC layer of that size is created (pretrained fc is for 1000-way
            ImageNet classification and isn't reused).
        drop_rate: dropout rate inside dense layers.

    Returns:
        (model, missing_keys, unexpected_keys) -- the key lists should both
        be empty except for the classifier when num_classes != 1000, which
        confirms the backbone loaded cleanly.
    """
    # 1. Build our from-scratch model with a 1000-way head (matches the
    #    pretrained checkpoint's classifier shape) so ALL keys, including
    #    the classifier, load without mismatch.
    model = DenseNet121(num_classes=1000, drop_rate=drop_rate)

    # 2. Pull torchvision's official pretrained weights.
    tv_model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
    pretrained_state = tv_model.state_dict()

    # 3. Load directly -- strict=True will raise loudly if any name mismatches,
    #    which is exactly the safety check we want.
    missing, unexpected = model.load_state_dict(pretrained_state, strict=True)

    # 4. Now swap in the head we actually want.
    if num_classes is None:
        model.classifier = None
        model.num_classes = None
    elif num_classes != 1000:
        model.classifier = nn.Linear(model.num_features, num_classes)
        model.num_classes = num_classes

    return model, missing, unexpected


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building from-scratch DenseNet-121 and loading pretrained weights...")
    model, missing, unexpected = load_pretrained_densenet121(num_classes=None)
    print(f"  missing_keys:    {missing}")
    print(f"  unexpected_keys: {unexpected}")
    assert len(missing) == 0 and len(unexpected) == 0, "Pretrained weights did not load cleanly!"
    print("  Pretrained weights loaded cleanly. ✔")

    model.eval()
    dummy = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        pooled_features = model(dummy)  # classifier=None -> returns pooled 1024-d features
        print(f"\nBackbone-only forward pass:")
        print(f"  input shape:  {tuple(dummy.shape)}")
        print(f"  output shape: {tuple(pooled_features.shape)}  (expected (2, 1024))")

        logits, intermediates = model(dummy, return_intermediate=True)
        print(f"\nreturn_intermediate=True forward pass:")
        print(f"  final output shape: {tuple(logits.shape)}")
        print(f"  block_output_channels: {model.block_output_channels}  (expected [256, 512, 1024, 1024])")
        for i, feat in enumerate(intermediates):
            print(f"  denseblock{i + 1} output shape: {tuple(feat.shape)}")

    print("\nAll shape checks passed. Ready to plug into CBAM-DenseNet.")