# Multi-label Thoracic Disease Classifier by Chest X-ray Scanning
To build a website that automatically detects 14 possible thoracic diseases from chest X-ray images and visually explains its predictions using Grad-CAM heatmaps, so that the output is interpretable.

## Run locally

Install the Python dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Predictions require the trained SCSE-DenseNet121 checkpoint at
`ml/checkpoints/final_scse_densenet121.pth`. Without it, the API starts but
reports that the model is unavailable and returns HTTP 503 for predictions.
The model comparison page shows metrics only when actual evaluation results
are available; it does not substitute estimated or placeholder scores.
