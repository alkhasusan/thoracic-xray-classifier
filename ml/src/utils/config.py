"""
config.py

Loads model.yaml and train.yaml (from ml/configs/) and exposes their values
as plain importable constants, so train.py / trainer.py / validate.py never
touch YAML directly.

Expected location: ml/src/utils/config.py
  -> parents[2] resolves to ml/           (where configs/ lives)
  -> parents[3] resolves to the repo root (where data paths in train.yaml
     are written relative to, e.g. "ml/data/splits/train.csv")
"""

from pathlib import Path

import torch
import yaml

_THIS_FILE = Path(__file__).resolve()
ML_ROOT = _THIS_FILE.parents[2]        # .../ml
PROJECT_ROOT = _THIS_FILE.parents[3]   # repo root
CONFIG_DIR = ML_ROOT / "configs"

with open(CONFIG_DIR / "model.yaml", encoding="utf-8") as f:
    _model_cfg = yaml.safe_load(f)

with open(CONFIG_DIR / "train.yaml", encoding="utf-8") as f:
    _train_cfg = yaml.safe_load(f)


# --- Model selection (from model.yaml) ---
MODEL_NAME = _model_cfg["model_name"]
NUM_CLASSES = _model_cfg["num_classes"]
PRETRAINED = _model_cfg["pretrained"]
CBAM_REDUCTION_RATIO = _model_cfg.get("cbam", {}).get("reduction_ratio", 16)
CBAM_SPATIAL_KERNEL_SIZE = _model_cfg.get("cbam", {}).get("spatial_kernel_size", 7)

# --- Data (from train.yaml) ---
# Paths in train.yaml are written relative to the repo root, e.g.
# "ml/data/splits/train.csv" -- on Kaggle, override these three env vars
# (or edit train.yaml directly) to point at /kaggle/input/... instead.
TRAIN_CSV = PROJECT_ROOT / _train_cfg["data"]["train_csv"]
VAL_CSV = PROJECT_ROOT / _train_cfg["data"]["val_csv"]
TEST_CSV = PROJECT_ROOT / _train_cfg["data"]["test_csv"]
BATCH_SIZE = _train_cfg["data"]["batch_size"]

# --- Optimizer ---
LEARNING_RATE = _train_cfg["optimizer"]["lr"]
WEIGHT_DECAY = _train_cfg["optimizer"].get("weight_decay", 0.0)

# --- Loss (FocalLoss params) ---
LOSS_ALPHA = _train_cfg["loss"]["alpha"]
LOSS_GAMMA = _train_cfg["loss"]["gamma"]

# --- Training schedule ---
EPOCHS = _train_cfg["training"]["epochs"]
EARLY_STOPPING_PATIENCE = _train_cfg["training"].get("early_stopping_patience")

_requested_device = _train_cfg["training"].get("device", "cuda")
DEVICE = torch.device(
    "cuda" if (_requested_device == "cuda" and torch.cuda.is_available()) else "cpu"
)

# --- Checkpoints ---
CHECKPOINT_DIR = PROJECT_ROOT / _train_cfg["checkpoints"]["save_dir"]
SAVE_BEST_ONLY = _train_cfg["checkpoints"].get("save_best_only", True)
MONITOR_METRIC = _train_cfg["checkpoints"].get("monitor_metric", "val_loss")

# --- Logging ---
LOG_EVERY_N_STEPS = _train_cfg["logging"].get("log_every_n_steps", 50)
OUTPUT_DIR = PROJECT_ROOT / _train_cfg["logging"].get("output_dir", "ml/outputs")