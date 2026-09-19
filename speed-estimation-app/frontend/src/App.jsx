import React, { useState, useRef } from "react";

const API = "/api";

export default function App() {
  const [video, setVideo] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(null);
  const [charts, setCharts] = useState([]);
  const pollRef = useRef(null);

  const reset = () => {
    setJobId(null);
    setStatus(null);
    setReport(null);
    setCharts([]);
    setError(null);
    if (pollRef.current) clearInterval(pollRef.current);
  };

  const startPolling = (id) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      const res = await fetch(`${API}/status/${id}`);
      const data = await res.json();
      setStatus(data);
      if (data.status === "done") {
        clearInterval(pollRef.current);
        const r = await fetch(`${API}/result/${id}`);
        setReport(await r.json());
        const c = await fetch(`${API}/charts/${id}`);
        setCharts(await c.json());
      } else if (data.status === "error") {
        clearInterval(pollRef.current);
      }
    }, 1500);
  };

  const onFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    reset();
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const uploadRes = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (!uploadRes.ok) throw new Error(`upload failed: ${uploadRes.status}`);
      const uploaded = await uploadRes.json();
      setVideo(uploaded);

      const processRes = await fetch(`${API}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: uploaded.video_id,
          calibration_mode: "auto",
        }),
      });
      if (!processRes.ok) throw new Error(`processing failed to start: ${processRes.status}`);
      const job = await processRes.json();
      setJobId(job.job_id);
      setStatus(job);
      startPolling(job.job_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  const movingTracks = report ? [...report.tracks].sort((a, b) => b.mean_speed_kmh - a.mean_speed_kmh) : [];

  return (
    <div className="app">
      <header>
        <h1>Forensic Speed Estimation</h1>
        <p className="subtitle">
          Upload a video — moving objects are detected, tracked, and their speed
          calculated automatically. Stationary objects are excluded.
        </p>
      </header>

      <section className="card">
        <h2>Upload video</h2>
        <input type="file" accept="video/*" onChange={onFileChange} />
        {uploading && <p>Uploading…</p>}
        {error && <p className="error">{error}</p>}
        {video && (
          <p className="meta">
            {video.width}×{video.height}, {video.fps.toFixed(2)} fps, {video.n_frames} frames
          </p>
        )}
      </section>

      {status && (
        <section className="card">
          <h2>Progress</h2>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${(status.progress || 0) * 100}%` }} />
          </div>
          <p>{status.status} — {status.message}</p>
        </section>
      )}

      {report && (
        <section className="card">
          <h2>Results</h2>
          <video controls width="100%" src={`${API}/video/${jobId}`} />

          <h3>Moving objects — {movingTracks.length} detected</h3>
          {movingTracks.length === 0 && (
            <p className="hint">No genuinely moving objects were detected in this video.</p>
          )}
          {movingTracks.length > 0 && (
            <table className="results-table">
              <thead>
                <tr>
                  <th>ID</th><th>Object</th><th>Speed</th><th>95% confidence range</th>
                  <th>Duration</th><th>Distance</th><th>Detection confidence</th>
                </tr>
              </thead>
              <tbody>
                {movingTracks.map((t) => (
                  <tr key={t.track_id}>
                    <td>#{t.track_id}</td>
                    <td>{t.class_name}</td>
                    <td><strong>{t.mean_speed_kmh.toFixed(1)} km/h</strong></td>
                    <td>{t.ci95_low_kmh.toFixed(1)} – {t.ci95_high_kmh.toFixed(1)} km/h</td>
                    <td>{t.elapsed_s.toFixed(1)} s</td>
                    <td>{t.distance_m.toFixed(1)} m</td>
                    <td>{(t.tracking_confidence * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {movingTracks.some((t) => t.known_class_size === false) && (
            <p className="hint">
              Some detected objects have no reliable typical-size prior — their
              confidence range is wider accordingly.
            </p>
          )}

          {report.evaluation && (
            <div>
              <h3>Evaluation against known speeds</h3>
              <ul>
                <li>MAE: {report.evaluation.MAE_kmh.toFixed(2)} km/h</li>
                <li>RMSE: {report.evaluation.RMSE_kmh.toFixed(2)} km/h</li>
                <li>MAPE: {report.evaluation.MAPE_pct.toFixed(1)}%</li>
                <li>Bias: {report.evaluation.bias_kmh.toFixed(2)} km/h</li>
                <li>95% CI coverage: {report.evaluation.ci95_coverage_pct.toFixed(0)}%</li>
              </ul>
            </div>
          )}

          {charts.length > 0 && (
            <div>
              <h3>Charts</h3>
              <div className="chart-grid">
                {charts.map((name) => (
                  <img key={name} src={`${API}/chart/${jobId}/${name}`} alt={name} />
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}