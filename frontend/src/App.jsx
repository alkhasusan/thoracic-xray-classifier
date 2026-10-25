
import { useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  function selectFile(selectedFile) {
    setError("");
    setResults(null);

    if (!selectedFile) return;

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!allowedTypes.includes(selectedFile.type)) {
      setError("Please select a JPEG, PNG, or WebP image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Please select an image smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  }

  function handleFileChange(event) {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function handlePredict() {
    if (!file) {
      setError("Upload a chest X-ray image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "The prediction request failed."
        );
      }

      setResults(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Check that FastAPI is running at http://127.0.0.1:8000."
          : err.message || "Something went wrong while processing the image."
      );
    } finally {
      setLoading(false);
    }
  }

  function clearImage() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setResults(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const topPrediction = results?.predictions?.[0];

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Thoracic AI home">
          <span className="brand-mark">T</span>
          <span>
            <strong>Thoracic AI</strong>
            <small>Research classifier</small>
          </span>
        </a>

        <div className="status-pill">
          <span className="status-dot" />
          SCSE-DenseNet121
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <p className="eyebrow">CHEST X-RAY ANALYSIS</p>
          <h1>Understand your X-ray model's predictions.</h1>
          <p className="hero-copy">
            Upload a chest X-ray to explore the output scores of our
            SCSE-DenseNet121 research model across 14 labels.
          </p>
        </section>

        <div className="workspace">
          <section className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <span className="step-label">STEP 01</span>
                <h2>Upload an image</h2>
              </div>
              <span className="panel-icon" aria-hidden="true">↑</span>
            </div>

            <input
              ref={fileInputRef}
              className="visually-hidden"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
            />

            {preview ? (
              <div className="preview-container">
                <img
                  className="xray-preview"
                  src={preview}
                  alt="Preview of uploaded chest X-ray"
                />
                <div className="preview-caption">
                  <div className="file-info">
                    <span className="file-symbol">IMG</span>
                    <div>
                      <strong>{file?.name}</strong>
                      <small>
                        {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : ""}
                      </small>
                    </div>
                  </div>
                  <button
                    className="text-button"
                    type="button"
                    onClick={clearImage}
                    disabled={loading}
                  >
                    Remove
                  </button>
                </div>
              </div>
            ) : (
              <button
                type="button"
                className={`dropzone ${dragging ? "dropzone-active" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
              >
                <span className="upload-icon" aria-hidden="true">↑</span>
                <strong>Drop your X-ray here</strong>
                <span>or click to browse your files</span>
                <span className="file-types">JPEG, PNG or WebP · Max 10 MB</span>
              </button>
            )}

            {preview && (
              <button
                type="button"
                className="secondary-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
              >
                Choose another image
              </button>
            )}

            <div className="upload-note">
              <span aria-hidden="true">ⓘ</span>
              Use a chest X-ray image for research testing. Avoid uploading
              images containing identifying information.
            </div>

            <button
              className="primary-button"
              type="button"
              onClick={handlePredict}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analysing image...
                </>
              ) : (
                <>Run prediction <span aria-hidden="true">→</span></>
              )}
            </button>

            {error && (
              <div className="error-message" role="alert">
                <strong>Something went wrong</strong>
                <p>{error}</p>
              </div>
            )}
          </section>

          <section className="panel results-panel">
            <div className="panel-heading">
              <div>
                <span className="step-label">STEP 02</span>
                <h2>Prediction results</h2>
              </div>
              <span className="panel-icon" aria-hidden="true">↗</span>
            </div>

            {!results ? (
              <div className="empty-results">
                <div className="empty-illustration" aria-hidden="true">
                  <div className="empty-line line-one" />
                  <div className="empty-line line-two" />
                  <div className="empty-line line-three" />
                  <span>14</span>
                </div>
                <h3>{loading ? "Processing your image" : "Your results will appear here"}</h3>
                <p>
                  {loading
                    ? "The model is generating its output scores. This may take a moment."
                    : "Upload an X-ray and run a prediction to see the scores for all 14 disease labels."}
                </p>
                {loading && <div className="loading-track"><span /></div>}
              </div>
            ) : (
              <div className="results-content">
                <div className="result-summary">
                  <span className="summary-label">HIGHEST MODEL SCORE</span>
                  <div className="summary-main">
                    <h3>{topPrediction?.disease ?? "—"}</h3>
                    <strong>
                      {topPrediction
                        ? `${(topPrediction.score * 100).toFixed(1)}%`
                        : "—"}
                    </strong>
                  </div>
                  <p>Highest score among the model's 14 outputs.</p>
                </div>

                <div className="score-heading">
                  <h3>All disease labels</h3>
                  <span>{results.predictions.length} labels</span>
                </div>

                <div className="score-list">
                  {results.predictions.map((item, index) => (
                    <div className="score-item" key={item.disease}>
                      <div className="score-meta">
                        <span className="score-name">
                          <span className="score-rank">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          {item.disease.replaceAll("_", " ")}
                        </span>
                        <span className="score-value">
                          {(item.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div
                        className="score-track"
                        role="progressbar"
                        aria-label={`${item.disease} model score`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={Math.max(
                          0,
                          Math.min(100, item.score * 100)
                        )}
                      >
                        <div
                          className={`score-fill ${index === 0 ? "score-fill-top" : ""}`}
                          style={{
                            width: `${Math.max(0, Math.min(100, item.score * 100))}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  className="secondary-button"
                  type="button"
                  onClick={clearImage}
                >
                  Analyse another image
                </button>

                <p className="result-disclaimer">
                  {results.note}
                </p>
              </div>
            )}
          </section>
        </div>

        <section className="notice">
          <span className="notice-symbol" aria-hidden="true">!</span>
          <div>
            <strong>Research use only — not for clinical diagnosis</strong>
            <p>
              Scores are outputs from a machine-learning model. They do not
              establish whether a disease is present or absent and should not
              guide medical decisions. Interpretation requires qualified
              clinical review.
            </p>
          </div>
        </section>
      </main>

      <footer className="footer">
        <span>Research Project</span>
      </footer>
    </div>
  );
}

export default App;
