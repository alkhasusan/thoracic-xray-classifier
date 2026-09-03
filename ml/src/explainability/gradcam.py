"""
gradcam.py

Grad-CAM for multi-label Chest X-ray classification.

Rewritten to work with BOTH model architectures (no hardcoded checkpoint
path, no hardcoded architecture):
  - "densenet121_baseline" -- plain DenseNet-121 (torchvision structure).
  - "cbam_densenet121"     -- CBAM-DenseNet121 (custom forward pass that
    manually walks backbone.features and inserts CBAM after each dense
    block -- see cbam_densenet.py).

Since this is multi-label (not single-label ImageNet-style) classification,
Grad-CAM here is computed PER DISEASE, not per single predicted class: you
give it an image, it tells you which diseases were predicted above a
threshold (or top-K), and generates one heatmap per disease -- exactly
what a "Grad-CAM for each highly predicted disease" UI needs.

Usage (typical FastAPI inference-endpoint flow):

    from explainability.gradcam import (
        load_model_for_gradcam,
        run_gradcam_inference,
    )

    model = load_model_for_gradcam(
        checkpoint_path="ml/checkpoints/best_cbam_densenet121.pth",
        model_name="cbam_densenet121",
        num_classes=14,
        device=device,
    )

    result = run_gradcam_inference(
        model=model,
        image=pil_image,          # a PIL.Image, already RGB
        transform=val_transform,  # your existing transforms.py val_transform
        disease_names=ChestXrayDataset.DISEASES,
        device=device,
        threshold=0.5,            # or use top_k=3 instead
    )

    # result["predictions"] -> [{"disease": ..., "probability": ..., "heatmap_png_base64": ...}, ...]
"""

import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Target-layer auto-detection
# ---------------------------------------------------------------------------

def _get_target_layer(model):
    """
    Returns the nn.Module Grad-CAM should hook into, auto-detected from
    the model's structure. This is the piece that breaks if you hardcode
    a layer path for one architecture and then swap in the other.

    - CBAM-DenseNet121: hooks the LAST CBAM block's output. This is
      deliberate, not arbitrary -- it shows what the model attended to
      AFTER attention was applied, which is the actual novel component
      of this architecture. (Hooking backbone.features.norm5 instead
      would only show post-BN features, one step removed from what CBAM
      actually did.)
    - Plain DenseNet-121 baseline (torchvision structure): hooks
      features.norm5, the standard Grad-CAM target for DenseNet
      (last conv-derived feature map before global pooling).
    """
    if hasattr(model, "cbam_blocks"):
        return model.cbam_blocks[-1]

    if hasattr(model, "features") and hasattr(model.features, "norm5"):
        return model.features.norm5

    raise ValueError(
        "Could not auto-detect a Grad-CAM target layer for this model. "
        "Pass target_layer explicitly to GradCAM(...) instead."
    )


# ---------------------------------------------------------------------------
# Core Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Standard Grad-CAM, computed per-class, supporting multi-label models.

    Hooks a target layer's forward activations and backward gradients.
    One forward pass is cached; generate() can then be called once per
    class of interest without recomputing the forward pass (retain_graph
    handles this) -- important for multi-label, where you want several
    class heatmaps from the SAME input image.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or _get_target_layer(model)
        self.activations = None
        self.gradients = None
        # NOTE: deliberately NOT using register_full_backward_hook here.
        # torchvision's DenseNet forward does `F.relu(features, inplace=True)`
        # immediately after norm5 -- a module-level backward hook on norm5
        # conflicts with that in-place op ("Output 0 of BackwardHookFunction
        # is a view and is being modified inplace", a known PyTorch gotcha).
        # Hooking the gradient directly on the output TENSOR (inside the
        # forward hook, via output.register_hook) sidesteps this entirely
        # and works identically for both architectures.
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def remove_hooks(self):
        self._fwd_handle.remove()

    def forward(self, input_tensor):
        """
        Runs the forward pass once and caches it. Call this ONCE per
        image, then call generate_for_class() as many times as needed
        (once per disease of interest).

        Returns raw logits, shape (1, num_classes).
        """
        self.model.zero_grad(set_to_none=True)
        self._logits = self.model(input_tensor)
        return self._logits

    def generate_for_class(self, class_idx, input_hw):
        """
        Backprops from a single class's logit and turns the cached
        activations + fresh gradients into a normalized [0, 1] heatmap,
        resized to (H, W) = input_hw (the original image size).

        Must be called after forward(). Safe to call multiple times for
        different class_idx values on the same forward pass.
        """
        self.model.zero_grad(set_to_none=True)
        score = self._logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients          # (1, C, h, w)
        activations = self.activations      # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)     # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]. Guard against an all-zero map (can happen
        # if a class had zero gradient contribution, e.g. saturated sigmoid).
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        cam_resized = cv2.resize(cam, (input_hw[1], input_hw[0]))  # cv2 wants (W, H)
        return cam_resized


# ---------------------------------------------------------------------------
# Heatmap -> overlay image -> base64 PNG (frontend-ready)
# ---------------------------------------------------------------------------

def overlay_heatmap_on_image(original_image, cam, alpha=0.4):
    """
    Blends a [0,1] Grad-CAM heatmap onto the original RGB image using a
    jet colormap, matching the conventional Grad-CAM look.

    Args:
        original_image: PIL.Image (RGB), any size.
        cam: 2D numpy array in [0, 1], same H, W as original_image.
        alpha: heatmap opacity in the blend (0=invisible, 1=heatmap only).

    Returns:
        PIL.Image (RGB) -- the overlay, same size as original_image.
    """
    original_np = np.array(original_image.convert("RGB"))

    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color + (1 - alpha) * original_np).astype(np.uint8)
    return Image.fromarray(overlay)


def image_to_base64_png(pil_image):
    """Encodes a PIL.Image as a base64 PNG string, ready to drop into a JSON API response."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Model loading (no hardcoded checkpoint / architecture)
# ---------------------------------------------------------------------------

def load_model_for_gradcam(checkpoint_path, model_name, num_classes, device):
    """
    Builds the correct architecture via model.py's get_model_by_name()
    (same registry train.py uses), loads a checkpoint's state_dict, and
    puts it in eval mode. pretrained=False here because we're about to
    overwrite ALL weights from the checkpoint anyway -- no need to also
    download ImageNet weights first.
    """
    from src.models.model import get_model_by_name

    model = get_model_by_name(model_name, num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# End-to-end inference: predict + Grad-CAM for each highly-predicted disease
# ---------------------------------------------------------------------------

def run_gradcam_inference(
    model,
    image,
    transform,
    disease_names,
    device,
    threshold=0.5,
    top_k=None,
    alpha=0.4,
):
    """
    Runs prediction on a single image, then generates a Grad-CAM overlay
    for each disease that qualifies as "highly predicted" -- either every
    disease above `threshold`, or the top `top_k` by probability if you'd
    rather always show a fixed number regardless of threshold.

    Args:
        model: a model returned by load_model_for_gradcam().
        image: PIL.Image, RGB, ORIGINAL resolution (not yet transformed --
            this function keeps a copy of it to overlay heatmaps onto at
            full resolution, and applies `transform` separately for the
            model's input tensor).
        transform: your existing val_transform from transforms.py (resize
            + ToTensor + whatever normalization your training used).
        disease_names: ordered list of 14 disease names, e.g.
            ChestXrayDataset.DISEASES -- order MUST match the model's
            output indices.
        threshold: probability cutoff for "highly predicted." Ignored if
            top_k is set.
        top_k: if set, always return exactly this many diseases (the
            highest-probability ones), regardless of threshold.
        alpha: heatmap overlay opacity, passed to overlay_heatmap_on_image.

    Returns:
        {
          "all_probabilities": {disease_name: probability, ...},   # all 14
          "predictions": [
              {"disease": ..., "probability": ..., "heatmap_png_base64": ...},
              ...   # sorted by probability, descending
          ]
        }
    """
    image_rgb = image.convert("RGB")
    original_hw = (image_rgb.height, image_rgb.width)

    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    cam_extractor = GradCAM(model)
    try:
        logits = cam_extractor.forward(input_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]  # (num_classes,)

        all_probabilities = {
            disease_names[i]: float(probs[i]) for i in range(len(disease_names))
        }

        if top_k is not None:
            selected_indices = np.argsort(-probs)[:top_k]
        else:
            selected_indices = np.where(probs >= threshold)[0]
            selected_indices = selected_indices[np.argsort(-probs[selected_indices])]

        predictions = []
        for idx in selected_indices:
            idx = int(idx)
            cam = cam_extractor.generate_for_class(idx, input_hw=original_hw)
            overlay = overlay_heatmap_on_image(image_rgb, cam, alpha=alpha)
            predictions.append({
                "disease": disease_names[idx],
                "probability": float(probs[idx]),
                "heatmap_png_base64": image_to_base64_png(overlay),
            })

    finally:
        # Always remove hooks, even if something above raised -- otherwise
        # they silently keep firing on every future forward pass of this
        # model instance, which is a nasty memory/behavior leak in a
        # long-running FastAPI process serving many requests.
        cam_extractor.remove_hooks()

    return {
        "all_probabilities": all_probabilities,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/src
    from models.cbam_densenet import CBAMDenseNet121
    from models.model import get_model

    device = torch.device("cpu")

    def dummy_transform(pil_img):
        arr = np.array(pil_img.resize((224, 224))).astype("float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    disease_names = [f"disease_{i}" for i in range(14)]
    dummy_image = Image.fromarray(
        (np.random.rand(300, 400, 3) * 255).astype("uint8")
    )

    for label, model in [
        ("CBAM-DenseNet121", CBAMDenseNet121(num_classes=14, pretrained=False)),
        ("Baseline DenseNet-121", get_model(num_classes=14, pretrained=False)),
    ]:
        print(f"\nTesting Grad-CAM on {label} (random weights, offline test)...")
        model.eval()

        result = run_gradcam_inference(
            model=model,
            image=dummy_image,
            transform=dummy_transform,
            disease_names=disease_names,
            device=device,
            top_k=3,
        )

        assert len(result["all_probabilities"]) == 14
        assert len(result["predictions"]) == 3
        for pred in result["predictions"]:
            assert 0.0 <= pred["probability"] <= 1.0
            assert isinstance(pred["heatmap_png_base64"], str)
            assert len(pred["heatmap_png_base64"]) > 0
        print(f"  Top-3 predictions: {[(p['disease'], round(p['probability'], 3)) for p in result['predictions']]}")
        print(f"  Each has a valid base64 PNG heatmap: ✔")

    print("\nAll Grad-CAM checks passed for both architectures.")