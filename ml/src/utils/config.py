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

_THIS_FILE = Path(__file__).resolve()  # finds the exact location of the current Python file on your computer
ML_ROOT = _THIS_FILE.parents[2]        # .../ml folder
PROJECT_ROOT = _THIS_FILE.parents[3]   # repo root (thoracic-xray-classifier)
CONFIG_DIR = ML_ROOT / "configs"       # this folder ml/configs have model.yaml and train.yaml

with open(CONFIG_DIR / "model.yaml", encoding="utf-8") as f:    #encoding is how text is decoded into Python strings
    _model_cfg = yaml.safe_load(f)

with open(CONFIG_DIR / "train.yaml", encoding="utf-8") as f:
    _train_cfg = yaml.safe_load(f)


# --- Model selection (from model.yaml) ---
MODEL_NAME = _model_cfg["model_name"]
NUM_CLASSES = _model_cfg["num_classes"]
PRETRAINED = _model_cfg["pretrained"]

# --- Data (from train.yaml) ---
# Paths in train.yaml are written relative to the repo root, e.g.
# "ml/data/splits/train.csv" -- on Kaggle, override these (or edit
# train.yaml directly) to point at /kaggle/input/... instead.
# No VAL_CSV anymore -- val.csv was merged into train.csv upstream.
TRAIN_CSV = PROJECT_ROOT / _train_cfg["data"]["train_csv"]
TEST_CSV = PROJECT_ROOT / _train_cfg["data"]["test_csv"]
BATCH_SIZE = _train_cfg["data"]["batch_size"]
# Root folder for images. None if CSVs already contain full paths.
IMAGE_ROOT = _train_cfg["data"].get("image_root")

# --- Optimizer ---
LEARNING_RATE = _train_cfg["optimizer"]["lr"]
WEIGHT_DECAY = _train_cfg["optimizer"].get("weight_decay", 0.0)

# --- Loss (FocalLoss params) ---
LOSS_ALPHA = _train_cfg["loss"]["alpha"]
LOSS_GAMMA = _train_cfg["loss"]["gamma"]

# --- Training schedule ---
EPOCHS = _train_cfg["training"]["epochs"]
# Whether to resume from a previous checkpoint if one exists
RESUME = _train_cfg["training"].get("resume", False)

_requested_device = _train_cfg["training"].get("device", "cuda")
DEVICE = torch.device(
    "cuda" if (_requested_device == "cuda" and torch.cuda.is_available()) else "cpu"
)

# --- Checkpoints ---
CHECKPOINT_DIR = PROJECT_ROOT / _train_cfg["checkpoints"]["save_dir"]

# --- Logging ---
LOG_EVERY_N_STEPS = _train_cfg["logging"].get("log_every_n_steps", 50)
OUTPUT_DIR = PROJECT_ROOT / _train_cfg["logging"].get("output_dir", "ml/outputs")