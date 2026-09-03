"""
cbam_densenet.py

CBAM-DenseNet121: our custom architecture for multi-label Chest X-ray
classification.

Structure:
  - Backbone: DenseNet-121 (from densenet121.py, ImageNet-pretrained),
    all 4 dense blocks kept as-is.
  - New addition: a CBAM (Convolutional Block Attention Module) inserted
    after EACH of the 4 dense blocks, before that block's transition layer
    (or before the final norm5 for the last block).
  - Head: Global Average Pooling -> FC(num_classes) -> Sigmoid.
  - Loss: FocalLoss (below), used instead of plain BCE to handle class
    imbalance across the 14 disease labels.

CBAM channel counts are read directly from the backbone's
`block_output_channels` (256 -> 512 -> 1024 -> 1024 for DenseNet-121),
so this stays correct even if block_config ever changes.

Depends on: densenet121.py (must be in the same package / models/ folder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet121 import DenseNet121, load_pretrained_densenet121


# ---------------------------------------------------------------------------
# CBAM: Channel Attention + Spatial Attention
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Learns which feature CHANNELS matter.
    avg-pool and max-pool the spatial dims down to 1x1, run both through a
    shared MLP, sum, sigmoid -> per-channel weight, broadcast-multiplied
    back onto the input.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        avg_pooled = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pooled = F.adaptive_max_pool2d(x, 1).view(b, c)

        avg_out = self.mlp(avg_pooled)
        max_out = self.mlp(max_pooled)

        channel_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * channel_weights


class SpatialAttention(nn.Module):
    """
    Learns which image REGIONS matter.
    Channel-pool (avg + max across the channel dim) down to 2 maps,
    concat, run through a 7x7 conv, sigmoid -> per-pixel weight,
    broadcast-multiplied back onto the input.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd to preserve spatial dims"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)

        spatial_weights = self.sigmoid(self.conv(concat))  # (B, 1, H, W)
        return x * spatial_weights


class CBAM(nn.Module):
    """
    Full CBAM block: Channel Attention applied first, then Spatial
    Attention applied to the channel-refined output (sequential, per the
    original CBAM paper).
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ---------------------------------------------------------------------------
# CBAM-DenseNet121
# ---------------------------------------------------------------------------

class CBAMDenseNet121(nn.Module):
    """
    DenseNet-121 backbone + a CBAM block after each of the 4 dense blocks
    + GAP -> FC(num_classes) head for multi-label classification.

    Args:
        num_classes: number of disease labels (14 for ChestX-ray14).
        pretrained: if True, loads ImageNet-pretrained weights into the
            backbone via load_pretrained_densenet121(). If False, the
            backbone is randomly initialized (useful for quick local
            shape-testing without network access).
        reduction_ratio: CBAM channel-attention MLP reduction ratio.
        spatial_kernel_size: CBAM spatial-attention conv kernel size.

    Forward:
        Returns raw logits by default (num_classes-dim), NOT sigmoid
        probabilities -- this is standard practice for numerical
        stability: FocalLoss (below) expects logits and applies sigmoid
        internally via BCEWithLogitsLoss under the hood. Pass
        apply_sigmoid=True to get probabilities directly (e.g. at
        inference time, or to match the "...-> Sigmoid" head description
        literally).
    """
    def __init__(self, num_classes=14, pretrained=True, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()

        if pretrained:
            backbone, missing, unexpected = load_pretrained_densenet121(num_classes=None)
            assert len(missing) == 0 and len(unexpected) == 0, (
                "Pretrained DenseNet-121 backbone did not load cleanly -- "
                f"missing={missing}, unexpected={unexpected}"
            )
        else:
            backbone = DenseNet121(num_classes=None)

        self.backbone = backbone

        # [256, 512, 1024, 1024] for standard DenseNet-121
        channel_counts = self.backbone.block_output_channels

        self.cbam_blocks = nn.ModuleList([
            CBAM(c, reduction_ratio=reduction_ratio, spatial_kernel_size=spatial_kernel_size)
            for c in channel_counts
        ])

        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x, apply_sigmoid=False):
        out = x
        cbam_idx = 0

        # Walk the backbone's stem -> denseblock1 -> transition1 -> ... -> norm5
        # in order, inserting the matching CBAM block right after each
        # denseblockN, before its transitionN (or before norm5 for the
        # last block).
        for name, module in self.backbone.features.named_children():
            out = module(out)
            if name.startswith("denseblock"):
                out = self.cbam_blocks[cbam_idx](out)
                cbam_idx += 1

        assert cbam_idx == len(self.cbam_blocks), (
            f"Expected to apply {len(self.cbam_blocks)} CBAM blocks, "
            f"only applied {cbam_idx}. Backbone structure may have changed."
        )

        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        logits = self.classifier(out)

        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


# ---------------------------------------------------------------------------
# Focal Loss (for multi-label class imbalance)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Multi-label Focal Loss, built on top of BCEWithLogitsLoss.

    Down-weights easy (already well-classified) examples so the loss
    focuses on hard/rare examples -- important here since most of the
    14 disease labels are heavily imbalanced (most images are negative
    for most diseases).

    Args:
        alpha: balances positive vs. negative examples (0.25 is the
            common default from the original Focal Loss paper; positive
            examples get weighted by alpha, negatives by 1-alpha).
        gamma: focusing parameter. Higher gamma -> more down-weighting
            of easy examples. gamma=0 reduces to plain weighted BCE.
        reduction: 'mean', 'sum', or 'none'.

    Expects RAW LOGITS as input (not sigmoid probabilities) -- matches
    CBAMDenseNet121's default forward() output.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building CBAM-DenseNet121 (pretrained=False for a fast, offline shape check)...")
    model = CBAMDenseNet121(num_classes=14, pretrained=False)
    model.eval()

    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (4, 14)).float()

    with torch.no_grad():
        logits = model(dummy_images)
        probs = model(dummy_images, apply_sigmoid=True)

    print(f"  input shape:  {tuple(dummy_images.shape)}")
    print(f"  logits shape: {tuple(logits.shape)}  (expected (4, 14))")
    print(f"  probs shape:  {tuple(probs.shape)}  (expected (4, 14))")
    print(f"  probs range:  [{probs.min().item():.4f}, {probs.max().item():.4f}]  (expected within [0, 1])")

    print("\nTesting FocalLoss...")
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, dummy_labels)
    print(f"  loss value: {loss.item():.4f}  (should be a single positive scalar)")
    assert loss.item() > 0, "Loss should be positive"

    print("\nAll shape and loss checks passed.")
    print("Note: this run used pretrained=False (random backbone weights) so it")
    print("works offline. Run with pretrained=True on Kaggle/local (with network")
    print("access) before actual training, to load ImageNet weights.")