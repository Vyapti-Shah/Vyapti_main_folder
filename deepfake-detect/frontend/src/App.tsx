import { useCallback, useRef, useState } from "react";
import { detectFile, DetectionResult } from "./api";

type Status = "idle" | "uploading" | "processing" | "done" | "error";

export default function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setStatus("uploading");
    setProgress(0);
    setError(null);
    setResult(null);

    try {
      const res = await detectFile(file, (percent) => {
        setProgress(percent);
        if (percent >= 100) setStatus("processing");
      });
      setResult(res);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setStatus("error");
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const reset = () => {
    setStatus("idle");
    setResult(null);
    setError(null);
    setProgress(0);
    if (inputRef.current) inputRef.current.value = "";
  };

  const busy = status === "uploading" || status === "processing";

  return (
    <div className="app">
      <div className="header">
        <h1>DeepFake Detector</h1>
        <p>Upload an image or video to check if it's real or AI-generated.</p>
      </div>

      {status !== "done" && (
        <div
          className={`dropzone${dragging ? " dragging" : ""}`}
          onClick={() => !busy && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*,video/*"
            onChange={onSelect}
            disabled={busy}
          />
          {busy ? (
            <div>
              <div>{status === "uploading" ? "Uploading..." : "Analyzing frames..."}</div>
              <div className="progress-bar">
                <div
                  className="fill"
                  style={{ width: status === "processing" ? "100%" : `${progress}%` }}
                />
              </div>
            </div>
          ) : (
            <div>
              <div>Drop an image or video here, or click to browse</div>
              <div className="hint">Supports JPG, PNG, MP4, MOV, AVI, WEBM</div>
            </div>
          )}
        </div>
      )}

      {status === "processing" && (
        <div className="status-text">
          Videos are analyzed frame-by-frame — this may take a moment.
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      {result && status === "done" && (
        <div className="result-card">
          <div className="result-header">
            <strong>{result.file_name}</strong>
            <span className={`verdict-badge ${result.is_fake ? "fake" : "real"}`}>
              {result.is_fake ? "AI GENERATED" : "REAL"}
            </span>
          </div>
          <div className="result-media">
            {result.file_type === "image" ? (
              <img src={result.output_path} alt="Detection result" />
            ) : (
              <video src={result.output_path} controls autoPlay loop muted />
            )}
          </div>
          <div className="result-meta">
            <div>
              <div className="label">Confidence</div>
              <div>{(result.confidence * 100).toFixed(1)}%</div>
            </div>
            <div>
              <div className="label">Type</div>
              <div>{result.file_type}</div>
            </div>
            {result.frames_analyzed != null && (
              <div>
                <div className="label">Frames Analyzed</div>
                <div>{result.frames_analyzed}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {(status === "done" || status === "error") && (
        <button className="reset-btn" onClick={reset}>
          Analyze another file
        </button>
      )}
    </div>
  );
}
