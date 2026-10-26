import { useRef, useState } from "react";
import Dropzone from "./Dropzone.jsx";
import ResultPanel from "./ResultPanel.jsx";

/** One self-contained upload -> analyze -> result workflow. Image and video cards each get
 * their own instance, so their state never mixes. */
export default function DetectionCard({ title, icon, accept, hint, previewKind, submit }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | uploading | error | done
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const objectUrlRef = useRef(null);

  const reset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setStatus("idle");
    setProgress(0);
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    setPreviewUrl(null);
  };

  const onFile = (f) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(f);
    objectUrlRef.current = url;
    setFile(f);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
    setStatus("idle");
    setProgress(0);
  };

  const analyze = async () => {
    if (!file) return;
    setStatus("uploading");
    setError(null);
    setProgress(0);
    try {
      const data = await submit(file, setProgress);
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(e.message || "Something went wrong.");
      setStatus("error");
    }
  };

  const busy = status === "uploading";

  return (
    <section className="card" aria-labelledby={`${title}-heading`}>
      <header className="card-header">
        <span className="card-icon" aria-hidden="true">{icon}</span>
        <div>
          <h2 id={`${title}-heading`}>{title}</h2>
          <p className="card-sub">{hint}</p>
        </div>
      </header>

      {!file && <Dropzone accept={accept} hint={hint} onFile={onFile} disabled={busy} />}

      {file && (
        <div className="preview-area">
          {previewKind === "image" ? (
            <img src={previewUrl} alt={`Preview of ${file.name}`} className="preview-image" />
          ) : (
            <video src={previewUrl} controls className="preview-video" aria-label={`Preview of ${file.name}`} />
          )}
          <div className="file-row">
            <span className="file-name" title={file.name}>{file.name}</span>
            <button type="button" className="btn btn-text" onClick={reset} disabled={busy}>
              Remove
            </button>
          </div>
        </div>
      )}

      <div className="actions">
        <button type="button" className="btn btn-primary" onClick={analyze} disabled={!file || busy}>
          {busy ? "Analyzing…" : "Analyze"}
        </button>
      </div>

      {busy && (
        <div className="upload-progress">
          <div
            className="upload-progress-track"
            role="progressbar"
            aria-valuenow={progress}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Upload progress"
          >
            <div className="upload-progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <span>{progress < 100 ? `Uploading… ${progress}%` : "Analyzing…"}</span>
        </div>
      )}

      {status === "error" && (
        <div className="error-box" role="alert">
          <strong>Couldn't analyze this file.</strong>
          <p>{error}</p>
          <button type="button" className="btn btn-text" onClick={analyze}>
            Try again
          </button>
        </div>
      )}

      {status === "done" && result && <ResultPanel result={result} />}
    </section>
  );
}
