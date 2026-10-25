import { useEffect, useState } from "react";
import "./ModelComparison.css";

const API_URL = "http://127.0.0.1:8000";

const modelNames = [
  { name: "DenseNet121 Baseline", checkpoint: "final_densenet121_baseline.pth" },
  { name: "ResNet50", checkpoint: "final_resnet50.pth" },
  { name: "EfficientNet-B0", checkpoint: "final_efficientnet_b0.pth" },
  { name: "VGG16-BN", checkpoint: "final_vgg16_bn.pth" },
  { name: "MobileNetV3-Large", checkpoint: "final_mobilenet_v3_large.pth" },
  { name: "SCSE-DenseNet121", checkpoint: "final_scse_densenet121.pth" },
];

const diseases = [
  "Atelectasis",
  "Cardiomegaly",
  "Consolidation",
  "Edema",
  "Effusion",
  "Emphysema",
  "Fibrosis",
  "Hernia",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pleural Thickening",
  "Pneumonia",
  "Pneumothorax",
];

export default function ModelComparison({ onBack }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadModels() {
      try {
        const response = await fetch(`${API_URL}/models`);
        if (!response.ok) {
          throw new Error(`Could not load model status (HTTP ${response.status}).`);
        }
        const data = await response.json();
        if (active) setModels(data.models);
      } catch (error) {
        if (active) {
          setModels(modelNames.map((model) => ({ ...model, available: false })));
          setLoadError(error.message);
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    loadModels();
    return () => {
      active = false;
    };
  }, []);

  const availableCount = models.filter((model) => model.available).length;

  return (
    <div className="comparison-page">
      <header className="comparison-topbar">
        <div className="comparison-brand">
          <span className="comparison-mark">T</span>
          <div>
            <strong>Thoracic AI</strong>
            <small>Model evaluation</small>
          </div>
        </div>
        <button className="comparison-back" type="button" onClick={onBack}>
          ← X-ray prediction
        </button>
      </header>

      <main className="comparison-main">
        <section className="comparison-hero">
          <p className="comparison-eyebrow">RESEARCH · MODEL EVALUATION</p>
          <h1>Comparing six chest X-ray models</h1>
          <p>
            Compare test loss, mean ROC-AUC, and per-disease performance for
            the six trained architectures using their actual evaluation
            results.
          </p>
        </section>

        <section className="comparison-notice" role="status">
          <strong>
            {loading
              ? "Checking model checkpoints..."
              : `${availableCount} of 6 model checkpoints available`}
          </strong>
          <p>
            Checkpoint availability is read from the running backend. Test
            loss and ROC-AUC require the test X-ray images as well as trained
            weights. The configured test CSV refers to Kaggle paths that are
            not present on this machine, so those metrics cannot be computed
            yet. No performance results are fabricated.
          </p>
          {loadError && <p role="alert">{loadError}</p>}
        </section>

        <section className="comparison-stats" aria-label="Comparison summary">
          <article className="comparison-stat">
            <span>ARCHITECTURES</span>
            <strong>06</strong>
            <small>Architectures tracked</small>
          </article>
          <article className="comparison-stat">
            <span>CHECKPOINTS AVAILABLE</span>
            <strong>{loading ? "…" : `${availableCount}/6`}</strong>
            <small>Loaded from local checkpoint folder</small>
          </article>
          <article className="comparison-stat">
            <span>TEST METRICS</span>
            <strong className="comparison-stat-unavailable">Need test images</strong>
            <small>Test split image files are not configured</small>
          </article>
        </section>

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>01 / OVERALL PERFORMANCE</span>
              <h2>Performance charts</h2>
            </div>
            <small>Mean ROC-AUC ↑ · Test loss ↓</small>
          </div>
          <div className="comparison-chart-grid">
            {["Mean test ROC-AUC", "Test loss"].map((metric) => (
              <div className="comparison-chart" key={metric}>
                <h3>{metric}</h3>
                <div className="comparison-chart-empty">
                  Evaluation needs the test images listed in the test split CSV.
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>02 / MODEL METRICS</span>
              <h2>Test-set comparison</h2>
            </div>
            <small>Actual metrics require the test split images</small>
          </div>
          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Checkpoint</th>
                  <th scope="col">Test loss</th>
                  <th scope="col">Mean test ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.name}>
                    <td><strong>{model.name}</strong></td>
                    <td className={model.available ? "checkpoint-available" : "metric-unavailable"}>
                      {loading
                        ? "Checking..."
                        : model.available
                          ? "Available"
                          : "Missing"}
                    </td>
                    <td className="metric-unavailable">Not available</td>
                    <td className="metric-unavailable">Not available</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>03 / BASELINE VS ATTENTION</span>
              <h2>SCSE-DenseNet121 vs DenseNet121 Baseline</h2>
            </div>
          </div>
          <div className="comparison-delta">
            {[
              "DenseNet121 Baseline",
              "SCSE-DenseNet121",
              "Mean ROC-AUC difference",
            ].map((label) => (
              <div key={label}>
                <span>{label}</span>
                <strong className="comparison-unavailable-value">
                  Not available
                </strong>
                <small>Actual test evaluation results required</small>
              </div>
            ))}
          </div>
          <p className="comparison-footnote">
            The impact of adding SCSE attention cannot be evaluated until both
            models have results from the same test split and evaluation
            procedure.
          </p>
        </section>

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>04 / DISEASE-LEVEL PERFORMANCE</span>
              <h2>ROC-AUC across 14 labels</h2>
            </div>
          </div>
          <div className="comparison-table-wrap">
            <table className="comparison-table disease-table">
              <thead>
                <tr>
                  <th scope="col">Disease label</th>
                  {models.map((model) => (
                    <th scope="col" key={model.name}>{model.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {diseases.map((disease) => (
                  <tr key={disease}>
                    <td><strong>{disease}</strong></td>
                    {models.map((model) => (
                      <td className="metric-unavailable" key={model.name}>
                        Not available
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="comparison-footnote">
            Per-disease ROC-AUC values require each model&apos;s predictions on
            the same test images and labels; missing values do not mean zero
            performance.
          </p>
        </section>

        <footer className="comparison-footer">
          Research Project · Thoracic AI · Not for clinical diagnosis
        </footer>
      </main>
    </div>
  );
}
