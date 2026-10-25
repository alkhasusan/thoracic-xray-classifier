
from pathlib import Path
import inspect
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Resolve the project root regardless of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.datasets.labels import DISEASE_LABELS

CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "checkpoints" / "final_scse_densenet121.pth"

MODEL_CHECKPOINTS = [
    {
        "name": "DenseNet121 Baseline",
        "checkpoint": "final_densenet121_baseline.pth",
    },
    {"name": "ResNet50", "checkpoint": "final_resnet50.pth"},
    {"name": "EfficientNet-B0", "checkpoint": "final_efficientnet_b0.pth"},
    {"name": "VGG16-BN", "checkpoint": "final_vgg16_bn.pth"},
    {
        "name": "MobileNetV3-Large",
        "checkpoint": "final_mobilenet_v3_large.pth",
    },
    {
        "name": "SCSE-DenseNet121",
        "checkpoint": "final_scse_densenet121.pth",
    },
]

DEVICE = "unavailable"
inference_transform = None
model = None

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

# Load the trained model once when the server starts. The API can still start
# without a checkpoint so its health endpoint can explain why inference is
# unavailable instead of failing during application import.
if CHECKPOINT_PATH.is_file():
    import torch
    from torchvision import transforms

    from ml.src.models.scse_densenet import get_scse_densenet121

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    model = get_scse_densenet121(
        num_classes=len(DISEASE_LABELS),
        pretrained=False,
    )
    load_options = {}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, **load_options)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()


@app.get("/")
def home():
    return {
        "message": "Thoracic X-ray Classifier API",
        "model": "SCSE-DenseNet121",
        "status": "ready" if model is not None else "unavailable",
    }


@app.get("/health")
def health():
    health_status = {
        "status": "ready" if model is not None else "unavailable",
        "model": "SCSE-DenseNet121",
        "device": str(DEVICE),
    }
    if model is None:
        health_status["detail"] = (
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )
    return health_status


@app.get("/models")
def list_models():
    checkpoint_dir = PROJECT_ROOT / "ml" / "checkpoints"
    return {
        "models": [
            {
                **model_info,
                "available": (checkpoint_dir / model_info["checkpoint"]).is_file(),
            }
            for model_info in MODEL_CHECKPOINTS
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The SCSE-DenseNet121 checkpoint is not installed. "
                "Add the trained checkpoint before requesting predictions."
            ),
        )

    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        )

    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    import torch

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
