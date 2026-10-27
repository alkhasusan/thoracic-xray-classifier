
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

// Values transcribed from the supplied Kaggle screenshots.
// These have not been independently recalculated locally.
// Training loss is kept separate from test loss.
const evaluationResults = {
  "DenseNet121 Baseline": {
    source: "Kaggle evaluation screenshot",
    testLoss: 0.0148,
    meanTestRocAuc: 0.8104,
    finalTrainingLoss: 0.0144,
    diseaseRocAuc: {
      Atelectasis: 0.7891,
      Cardiomegaly: 0.8962,
      Consolidation: 0.7962,
      Edema: 0.8897,
      Effusion: 0.8698,
      Emphysema: 0.8948,
      Fibrosis: 0.7701,
      Hernia: 0.8491,
      Infiltration: 0.6917,
      Mass: 0.8094,
      Nodule: 0.7325,
      "Pleural Thickening": 0.7672,
      Pneumonia: 0.7395,
      Pneumothorax: 0.8509,
    },
    tunedThresholds: {
      Atelectasis: 0.2265,
      Cardiomegaly: 0.1443,
      Consolidation: 0.1835,
      Edema: 0.1260,
      Effusion: 0.2201,
      Emphysema: 0.1227,
      Fibrosis: 0.1029,
      Hernia: 0.0694,
      Infiltration: 0.2771,
      Mass: 0.1973,
      Nodule: 0.1694,
      "Pleural Thickening": 0.1420,
      Pneumonia: 0.0942,
      Pneumothorax: 0.1690,
    },
  },

  ResNet50: {
    source: "Kaggle evaluation screenshot",
    testLoss: 0.0154,
    meanTestRocAuc: 0.7903,
    finalTrainingLoss: 0.0146,
    diseaseRocAuc: {
      Atelectasis: 0.7828,
      Cardiomegaly: 0.8501,
      Consolidation: 0.7797,
      Edema: 0.8739,
      Effusion: 0.8657,
      Emphysema: 0.8672,
      Fibrosis: 0.7704,
      Hernia: 0.8349,
      Infiltration: 0.6786,
      Mass: 0.7685,
      Nodule: 0.7169,
      "Pleural Thickening": 0.7389,
      Pneumonia: 0.7025,
      Pneumothorax: 0.8338,
    },
    tunedThresholds: {
      Atelectasis: 0.2421,
      Cardiomegaly: 0.1308,
      Consolidation: 0.1683,
      Edema: 0.1065,
      Effusion: 0.2422,
      Emphysema: 0.1535,
      Fibrosis: 0.1258,
      Hernia: 0.0716,
      Infiltration: 0.2588,
      Mass: 0.2017,
      Nodule: 0.2336,
      "Pleural Thickening": 0.1548,
      Pneumonia: 0.0986,
      Pneumothorax: 0.2280,
    },
  },

  "EfficientNet-B0": {
    source: "Kaggle evaluation screenshot",
    testLoss: 0.0150,
    meanTestRocAuc: 0.7982,
    finalTrainingLoss: 0.0148,
    diseaseRocAuc: {
      Atelectasis: 0.7797,
      Cardiomegaly: 0.8814,
      Consolidation: 0.7705,
      Edema: 0.8858,
      Effusion: 0.8663,
      Emphysema: 0.8822,
      Fibrosis: 0.7678,
      Hernia: 0.8072,
      Infiltration: 0.6885,
      Mass: 0.7897,
      Nodule: 0.7247,
      "Pleural Thickening": 0.7671,
      Pneumonia: 0.7197,
      Pneumothorax: 0.8440,
    },
    tunedThresholds: {
      Atelectasis: 0.2237,
      Cardiomegaly: 0.1546,
      Consolidation: 0.1816,
      Edema: 0.0984,
      Effusion: 0.2449,
      Emphysema: 0.1443,
      Fibrosis: 0.1314,
      Hernia: 0.0663,
      Infiltration: 0.2986,
      Mass: 0.1593,
      Nodule: 0.2026,
      "Pleural Thickening": 0.1550,
      Pneumonia: 0.1173,
      Pneumothorax: 0.1809,
    },
  },

  "VGG16-BN": {
    source: "Kaggle evaluation screenshot",
    testLoss: 0.0153,
    meanTestRocAuc: 0.7793,
    finalTrainingLoss: 0.0154,
    diseaseRocAuc: {
      Atelectasis: 0.7495,
      Cardiomegaly: 0.7441,
      Consolidation: 0.7697,
      Edema: 0.8775,
      Effusion: 0.8622,
      Emphysema: 0.8701,
      Fibrosis: 0.7579,
      Hernia: 0.8014,
      Infiltration: 0.6705,
      Mass: 0.7889,
      Nodule: 0.7075,
      "Pleural Thickening": 0.7579,
      Pneumonia: 0.7124,
      Pneumothorax: 0.8404,
    },
    tunedThresholds: {
      Atelectasis: 0.2402,
      Cardiomegaly: 0.1251,
      Consolidation: null,
      Edema: 0.1415,
      Effusion: 0.2629,
      Emphysema: 0.1283,
      Fibrosis: 0.1229,
      Hernia: 0.0620,
      Infiltration: 0.2948,
      Mass: 0.1723,
      Nodule: 0.1873,
      "Pleural Thickening": 0.1534,
      Pneumonia: 0.1183,
      Pneumothorax: 0.1708,
    },
  },

  "MobileNetV3-Large": {
    source: "Kaggle evaluation screenshot",
    testLoss: 0.0154,
    meanTestRocAuc: 0.8010,
    finalTrainingLoss: 0.0146,
    diseaseRocAuc: {
      Atelectasis: 0.7681,
      Cardiomegaly: 0.8919,
      Consolidation: 0.7839,
      Edema: 0.8837,
      Effusion: 0.8635,
      Emphysema: 0.8838,
      Fibrosis: 0.7899,
      Hernia: 0.8353,
      Infiltration: 0.6883,
      Mass: 0.7814,
      Nodule: 0.6910,
      "Pleural Thickening": 0.7723,
      Pneumonia: 0.7213,
      Pneumothorax: 0.8599,
    },
    tunedThresholds: {
      Atelectasis: 0.2729,
      Cardiomegaly: 0.2015,
      Consolidation: 0.1642,
      Edema: 0.1054,
      Effusion: 0.2429,
      Emphysema: 0.1283,
      Fibrosis: 0.1331,
      Hernia: 0.0848,
      Infiltration: 0.2727,
      Mass: 0.1274,
      Nodule: 0.1654,
      "Pleural Thickening": 0.1520,
      Pneumonia: 0.1048,
      Pneumothorax: 0.1411,
    },
  },

  "SCSE-DenseNet121": {
    source: "Kaggle evaluation screenshot",
    testLoss: 0.0147,
    meanTestRocAuc: 0.8114,
    finalTrainingLoss: 0.0144,
    diseaseRocAuc: {
      Atelectasis: 0.7952,
      Cardiomegaly: 0.8895,
      Consolidation: 0.7903,
      Edema: 0.8878,
      Effusion: 0.8749,
      Emphysema: 0.9023,
      Fibrosis: 0.7871,
      Hernia: 0.8304,
      Infiltration: 0.6927,
      Mass: 0.7995,
      Nodule: 0.7411,
      "Pleural Thickening": 0.7731,
      Pneumonia: 0.7351,
      Pneumothorax: 0.8606,
    },
    tunedThresholds: {
      Atelectasis: 0.2390,
      Cardiomegaly: 0.1528,
      Consolidation: 0.1658,
      Edema: 0.1319,
      Effusion: 0.2336,
      Emphysema: 0.1379,
      Fibrosis: 0.1373,
      Hernia: 0.0670,
      Infiltration: 0.2712,
      Mass: 0.1575,
      Nodule: 0.1638,
      "Pleural Thickening": 0.1529,
      Pneumonia: 0.1172,
      Pneumothorax: 0.1819,
    },
  },
};

function getResults(modelName) {
  return evaluationResults[modelName] ?? {
    testLoss: null,
    meanTestRocAuc: null,
    finalTrainingLoss: null,
    diseaseRocAuc: {},
    tunedThresholds: {},
  };
}

function formatMetric(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "Not available";
}

function MetricChart({ title, models, metric, better }) {
  const values = models
    .map((model) => ({
      name: model.name,
      value: getResults(model.name)[metric],
    }))
    .filter(
      ({ value }) =>
        typeof value === "number" && Number.isFinite(value)
    );
  values.sort((left, right) =>
    better === "higher"
      ? right.value - left.value
      : left.value - right.value
  );

  if (!values.length) {
    return (
      <div className="comparison-chart">
        <h3>{title}</h3>
        <div className="comparison-chart-empty">
          No reported evaluation results are available.
        </div>
      </div>
    );
  }

  const maxValue =
    metric === "meanTestRocAuc"
      ? 1
      : Math.max(...values.map(({ value }) => value)) * 1.1;

  return (
    <div className="comparison-chart">
      <h3>{title}</h3>
      <p className="comparison-chart-description">
        {metric === "meanTestRocAuc"
          ? "ROC-AUC ranges from 0 to 1; higher means better ranking of positive cases."
          : "Lower test loss is better. Bars use a zero-based scale."}
      </p>
      <div style={{ display: "grid", gap: 15, marginTop: 18 }}>
        {values.map(({ name, value }) => (
          <div key={name}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 8,
                marginBottom: 6,
                fontSize: 11,
                color: "#63788a",
              }}
            >
              <span>{name}</span>
              <strong>{formatMetric(value)}</strong>
            </div>
            <div
              className="comparison-bar"
              style={{ width: "100%" }}
              role="img"
              aria-label={`${name}: ${formatMetric(value)}. ${
                better === "higher" ? "Higher is better." : "Lower is better."
              }`}
            >
              <span
                style={{
                  width: `${Math.max(
                    0,
                    Math.min(100, (value / maxValue) * 100)
                  )}%`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="comparison-chart-scale" aria-hidden="true">
        <span>0</span>
        <span>{formatMetric(maxValue / 2)}</span>
        <span>{formatMetric(maxValue)}</span>
      </div>
      <p className="comparison-footnote">
        {better === "higher" ? "Higher is better." : "Lower is better."}
        {" "}Scale: 0 to {formatMetric(maxValue)}. Values are screenshot-derived.
      </p>
    </div>
  );
}

function DiseasePerformanceChart({ models }) {
  const [selectedModel, setSelectedModel] = useState("SCSE-DenseNet121");
  const results = getResults(selectedModel);
  const values = diseases
    .map((disease) => ({
      disease,
      value: results.diseaseRocAuc[disease],
    }))
    .filter(
      ({ value }) => typeof value === "number" && Number.isFinite(value)
    )
    .sort((left, right) => right.value - left.value);

  const strongest = values[0];
  const weakest = values[values.length - 1];

  return (
    <section className="comparison-card">
      <div className="comparison-section-heading">
        <div>
          <span>04 / DISEASE-LEVEL PERFORMANCE</span>
          <h2>Explore ROC-AUC by disease</h2>
        </div>
        <label className="comparison-model-select">
          <span>Model</span>
          <select
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
          >
            {models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <p className="comparison-intro">
        ROC-AUC shows how well a model ranks positive cases above negative
        cases for each label. It is not a disease probability or diagnostic
        confidence score; 0.5 is chance-level ranking and 1.0 is perfect
        ranking.
      </p>

      {strongest && weakest ? (
        <>
          <div className="comparison-disease-summary" aria-live="polite">
            <span>
              Highest reported: <strong>{strongest.disease}</strong>{" "}
              ({formatMetric(strongest.value)})
            </span>
            <span>
              Lowest reported: <strong>{weakest.disease}</strong>{" "}
              ({formatMetric(weakest.value)})
            </span>
          </div>
          <div
            className="comparison-disease-chart"
            role="list"
            aria-label={`Disease-level ROC-AUC for ${selectedModel}, sorted highest to lowest`}
          >
            {values.map(({ disease, value }) => (
              <div
                className="comparison-disease-row"
                key={disease}
                role="listitem"
              >
                <span className="comparison-disease-name">{disease}</span>
                <div
                  className="comparison-disease-track"
                  aria-hidden="true"
                >
                  <span style={{ width: `${value * 100}%` }} />
                </div>
                <strong>{formatMetric(value)}</strong>
              </div>
            ))}
          </div>
          <div className="comparison-disease-axis" aria-hidden="true">
            <span>0</span>
            <span>0.5 · chance-level ranking</span>
            <span>1.0 · perfect ranking</span>
          </div>
        </>
      ) : (
        <div className="comparison-chart-empty">
          No disease-level evaluation results are available for this model.
        </div>
      )}
      <p className="comparison-footnote">
        Values are transcribed from supplied evaluation screenshots and have
        not been independently recalculated.
      </p>
    </section>
  );
}

export default function ModelComparison({ onBack }) {
  const [models, setModels] = useState(
    modelNames.map((model) => ({ ...model, available: false }))
  );
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadModels() {
      try {
        const response = await fetch(`${API_URL}/models`);

        if (!response.ok) {
          throw new Error(
            `Could not load model status (HTTP ${response.status}).`
          );
        }

        const data = await response.json();
        const backendModels = Array.isArray(data.models)
          ? data.models
          : [];

        const mergedModels = modelNames.map((knownModel) => {
          const backendModel = backendModels.find(
            (item) =>
              item.name === knownModel.name ||
              item.checkpoint === knownModel.checkpoint
          );

          return {
            ...knownModel,
            ...(backendModel ?? {}),
            name: knownModel.name,
            checkpoint: knownModel.checkpoint,
            available: backendModel?.available === true,
          };
        });

        if (active) {
          setModels(mergedModels);
          setLoadError("");
        }
      } catch (error) {
        if (active) {
          setLoadError(error.message);
          setModels(
            modelNames.map((model) => ({
              ...model,
              available: false,
            }))
          );
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

  const availableCount = models.filter(
    (model) => model.available
  ).length;

  const baseline = getResults("DenseNet121 Baseline");
  const scse = getResults("SCSE-DenseNet121");

  const aucDifference =
    typeof baseline.meanTestRocAuc === "number" &&
    typeof scse.meanTestRocAuc === "number"
      ? scse.meanTestRocAuc - baseline.meanTestRocAuc
      : null;

  const modelsWithResults = modelNames.filter(
    (model) =>
      typeof getResults(model.name).meanTestRocAuc === "number"
  ).length;

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

        <button
          className="comparison-back"
          type="button"
          onClick={onBack}
        >
          ← X-ray prediction
        </button>
      </header>

      <main className="comparison-main">
        <section className="comparison-hero">
          <p className="comparison-eyebrow">
            RESEARCH · MODEL EVALUATION
          </p>
          <h1>Comparing six chest X-ray models</h1>
          <p>
            Compare reported test loss, mean ROC-AUC, and per-disease
            performance for six trained architectures using the
            supplied Kaggle evaluation screenshots.
          </p>
        </section>

        <section className="comparison-notice" role="status">
          <strong>
            {loading
              ? "Checking model checkpoints..."
              : `${availableCount} of 6 model checkpoints available`}
          </strong>
          <p>
            Performance metrics below are transcribed from Kaggle
            screenshots. Checkpoint availability is obtained separately
            from the running backend. Results have not been independently
            recalculated on this machine. The same test split and
            evaluation procedure have not yet been independently verified
            for all six models.
          </p>
          {loadError && (
            <p role="alert">
              Could not load checkpoint status: {loadError}
            </p>
          )}
        </section>

        <section
          className="comparison-stats"
          aria-label="Comparison summary"
        >
          <article className="comparison-stat">
            <span>ARCHITECTURES</span>
            <strong>06</strong>
            <small>Architectures tracked</small>
          </article>

          <article className="comparison-stat">
            <span>CHECKPOINTS AVAILABLE</span>
            <strong>{loading ? "…" : `${availableCount}/6`}</strong>
            <small>Reported by the local backend</small>
          </article>

          <article className="comparison-stat">
            <span>MODELS WITH REPORTED RESULTS</span>
            <strong>{modelsWithResults}/6</strong>
            <small>Screenshot-derived evaluation results</small>
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
            <MetricChart
              title="Mean test ROC-AUC"
              models={models}
              metric="meanTestRocAuc"
              better="higher"
            />
            <MetricChart
              title="Test loss"
              models={models}
              metric="testLoss"
              better="lower"
            />
          </div>

          <p className="comparison-footnote">
            All six models have reported values. The bars visualize
            the reported metrics; they do not establish that the
            evaluation conditions were identical.
          </p>
        </section>

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>02 / MODEL METRICS</span>
              <h2>Test-set comparison</h2>
            </div>
            <small>Values reported in Kaggle screenshots</small>
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
                {models.map((model) => {
                  const result = getResults(model.name);

                  return (
                    <tr key={model.name}>
                      <td><strong>{model.name}</strong></td>
                      <td
                        className={
                          loading
                            ? "metric-unavailable"
                            : model.available
                              ? "checkpoint-available"
                              : "metric-unavailable"
                        }
                      >
                        {loading
                          ? "Checking..."
                          : model.available
                            ? "Available"
                            : "Missing"}
                      </td>
                      <td>{formatMetric(result.testLoss)}</td>
                      <td>{formatMetric(result.meanTestRocAuc)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="comparison-footnote">
            Checkpoint availability does not itself validate the
            evaluation metrics. Values are transcribed from screenshots.
          </p>
        </section>

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>03 / BASELINE VS ATTENTION</span>
              <h2>
                SCSE-DenseNet121 vs DenseNet121 Baseline
              </h2>
            </div>
          </div>

          <div className="comparison-delta">
            <div>
              <span>DenseNet121 Baseline</span>
              <strong>
                {formatMetric(baseline.meanTestRocAuc)}
              </strong>
              <small>Mean test ROC-AUC</small>
            </div>

            <div>
              <span>SCSE-DenseNet121</span>
              <strong>{formatMetric(scse.meanTestRocAuc)}</strong>
              <small>Mean test ROC-AUC</small>
            </div>

            <div>
              <span>Mean ROC-AUC difference</span>
              <strong className="comparison-unavailable-value">
                {aucDifference === null
                  ? "Not available"
                  : `${aucDifference > 0 ? "+" : ""}${aucDifference.toFixed(4)}`}
              </strong>
              <small>SCSE minus baseline; reported difference</small>
            </div>
          </div>

          <p className="comparison-footnote">
            Reported test loss — baseline: {formatMetric(baseline.testLoss)};
            {" "}SCSE: {formatMetric(scse.testLoss)}.
            {" "}The reported mean ROC-AUC difference is +0.0010.
            This is a descriptive difference only; a fair comparison
            requires confirmation of the same test split and evaluation
            procedure.
          </p>
        </section>

        <DiseasePerformanceChart models={models} />

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>05 / DISEASE-LEVEL DETAILS</span>
              <h2>ROC-AUC across 14 labels</h2>
            </div>
          </div>

          <div className="comparison-table-wrap">
            <table className="comparison-table disease-table">
              <thead>
                <tr>
                  <th scope="col">Disease label</th>
                  {models.map((model) => (
                    <th scope="col" key={model.name}>
                      {model.name}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {diseases.map((disease) => (
                  <tr key={disease}>
                    <td><strong>{disease}</strong></td>
                    {models.map((model) => {
                      const value =
                        getResults(model.name).diseaseRocAuc[disease];

                      return (
                        <td key={model.name}>
                          {formatMetric(value)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="comparison-footnote">
            Values are screenshot-derived. Each disease-level ROC-AUC
            is reported by the corresponding model evaluation output.
            Missing values, if any, are not treated as zero.
          </p>
        </section>

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>06 / TRAINING INFORMATION</span>
              <h2>Final training loss vs test loss</h2>
            </div>
          </div>

          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Final training loss</th>
                  <th scope="col">Test loss</th>
                  <th scope="col">Mean test ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => {
                  const result = getResults(model.name);

                  return (
                    <tr key={model.name}>
                      <td><strong>{model.name}</strong></td>
                      <td>{formatMetric(result.finalTrainingLoss)}</td>
                      <td>{formatMetric(result.testLoss)}</td>
                      <td>{formatMetric(result.meanTestRocAuc)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="comparison-footnote">
            Training loss measures the model's loss on training data.
            Test loss is the reported loss on the test set. They are
            different metrics and should not be substituted for one another.
          </p>
        </section>

        <section className="comparison-card">
          <div className="comparison-section-heading">
            <div>
              <span>07 / DATA PROVENANCE</span>
              <h2>Results source and limitations</h2>
            </div>
          </div>

          <ul className="comparison-guidance">
            <li>
              <strong>Source:</strong> Kaggle training and evaluation
              screenshots supplied for this project.
            </li>
            <li>
              <strong>Reported evaluation:</strong> Test loss, mean
              test ROC-AUC, and disease-level ROC-AUC are transcribed
              from each model's evaluation output.
            </li>
            <li>
              <strong>Training metrics:</strong> Final training loss
              is displayed separately from test loss.
            </li>
            <li>
              <strong>Thresholds:</strong> The supplied confusion-matrix
              outputs include a fixed threshold of 0.5 and separately
              tuned per-disease thresholds. These must not be mixed.
            </li>
            <li>
              <strong>Comparison limitation:</strong> The common test
              split and evaluation procedure have not yet been independently
              verified across all six models.
            </li>
            <li>
              <strong>Local evaluation:</strong> Results have not been
              recomputed locally because the configured Kaggle test-image
              paths are unavailable on this machine.
            </li>
          </ul>
        </section>

        <footer className="comparison-footer">
          Research Project · Thoracic AI · Not for clinical diagnosis
        </footer>
      </main>
    </div>
  );
}
