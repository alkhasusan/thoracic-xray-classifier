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


class FocalLoss(nn.Module):
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