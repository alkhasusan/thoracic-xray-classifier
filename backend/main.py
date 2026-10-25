
from pathlib import Path
import sys

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Resolve the project root regardless of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.models.scse_densenet import get_scse_densenet121
from ml.src.datasets.labels import DISEASE_LABELS

CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "checkpoints" / "final_scse_densenet121.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the same preprocessing as the validation pipeline.
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

app = FastAPI(title="Thoracic X-ray Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load the trained model once when the server starts.
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

model = get_scse_densenet121(num_classes=len(DISEASE_LABELS), pretrained=False)
state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)
model.load_state_dict(state_dict, strict=True)
model.to(DEVICE)
model.eval()


@app.get("/")
def home():
    return {
        "message": "Thoracic X-ray Classifier API",
        "model": "SCSE-DenseNet121",
        "status": "ready",
    }


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model": "SCSE-DenseNet121",
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        from io import BytesIO

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        )

    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().tolist()

    predictions = [
        {"disease": label, "score": round(float(score), 6)}
        for label, score in zip(DISEASE_LABELS, probabilities)
    ]

    predictions.sort(key=lambda item: item["score"], reverse=True)

    return {
        "model": "SCSE-DenseNet121",
        "note": (
            "Scores are model outputs, not confirmed diagnoses. "
            "This research model is not a substitute for radiologist review."
        ),
        "predictions": predictions,
    }
