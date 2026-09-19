import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelRender, clipStreamUrl, deleteClip, deleteRender, fmtDuration, getHealth,
  getRenderStatus, listClips, listRenders, renderStreamUrl, startRender, timeAgo,
  uploadClip,
  type Clip, type Health, type JobStatus, type Render,
} from "./api";
import { BoxSelect, type Box } from "./BoxSelect";

const POLL_MS = 1000;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [clip, setClip] = useState<Clip | null>(null);
  const [renders, setRenders] = useState<Render[]>([]);
  const [selected, setSelected] = useState<Render | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [canceling, setCanceling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [paused, setPaused] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => { getHealth().then(setHealth); }, []);
  const refreshClips = useCallback(() => { listClips().then(setClips); }, []);
  useEffect(() => { refreshClips(); }, [refreshClips]);

  const refreshRenders = useCallback((id: string) => {
    listRenders(id).then(setRenders).catch(() => setRenders([]));
  }, []);

  /* Selecting a clip resets everything downstream of it. Leaving the previous
   * clip's render selected would leave the player showing a video that has
   * nothing to do with the clip now listed as current. */
  const chooseClip = useCallback((c: Clip | null) => {
    setClip(c);
    setSelected(null);
    setBox(null);
    setJob(null);
    setRenders([]);
    setError(null);
    if (c) {
      refreshRenders(c.clip_id);
      getRenderStatus(c.clip_id).then((s) => s && setJob(s));
    }
  }, [refreshRenders]);

  const stopPolling = () => {
    if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
  };
  useEffect(() => stopPolling, []);

  function poll(clipId: string) {
    stopPolling();
    pollRef.current = window.setInterval(async () => {
      const st = await getRenderStatus(clipId);
      if (!st) return;
      setJob(st);
      if (st.status !== "running") {
        stopPolling();
        setCanceling(false);
        setBusy(false);
        if (st.status === "done") {
          refreshRenders(clipId);
          // Show it immediately — waiting for a render and then having to hunt
          // for it in a list is the wrong end of the interaction.
          const rows = await listRenders(clipId);
          const fresh = rows.find((r) => r.render_id === st.render_id) ?? rows[0] ?? null;
          setSelected(fresh);
        }
      }
    }, POLL_MS);
  }

  async function onUpload(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setBusy(true); setError(null);
    try {
      const c = await uploadClip(file);
      refreshClips();
      chooseClip(c);
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally { setBusy(false); }
  }

  async function render() {
    if (!clip || !box) return;
    setBusy(true); setError(null); setSelected(null);
    try {
      const t = videoRef.current?.currentTime ?? 0;
      const st = await startRender(
        clip.clip_id, [box.x1, box.y1, box.x2, box.y2], t,
        `${Math.round((box.x2 - box.x1) * clip.width)}×${Math.round((box.y2 - box.y1) * clip.height)} @ ${fmtDuration(t)}`);
      setJob(st);
      if (st.status === "running") poll(clip.clip_id);
    } catch (e) {
      setError(String((e as Error).message || e));
      setBusy(false);
    }
  }

  async function stop() {
    if (!clip) return;
    setCanceling(true);
    try { const st = await cancelRender(clip.clip_id); if (st) setJob(st); }
    catch { setCanceling(false); }
  }

  async function removeRender(id: string) {
    if (!clip) return;
    try {
      await deleteRender(clip.clip_id, id);
      if (selected?.render_id === id) setSelected(null);
      refreshRenders(clip.clip_id);
    } catch (e) { setError(String((e as Error).message || e)); }
  }

  async function removeClip(id: string) {
    try {
      await deleteClip(id);
      if (clip?.clip_id === id) chooseClip(null);
      refreshClips();
    } catch (e) { setError(String((e as Error).message || e)); }
  }

  const running = job?.status === "running";
  const src = clip
    ? (selected ? renderStreamUrl(clip.clip_id, selected.render_id) : clipStreamUrl(clip.clip_id))
    : "";

  // Assigning a new `src` does not reliably re-fetch in every browser, so
  // switching between the original and a render needs an explicit load().
  useEffect(() => { if (src) videoRef.current?.load(); }, [src]);

  const canDraw = !!clip && !selected && paused && !running;

  return (
    <div className="app">
      <header className="top">
        <h1>Track &amp; Zoom</h1>
        <span className="grow" />
        {health && (
          <span className={`chip ${health.sam2.weights_present ? "ok" : "warn"}`}
                title={health.sam2.weights_path}>
            SAM2 {health.sam2.importable ? health.sam2.device : "not installed"}
            {!health.sam2.weights_present && " · no weights"}
          </span>
        )}
      </header>

      <div className="body">
        <aside className="side">
          <label className={`upload ${busy ? "is-busy" : ""}`}>
            <input type="file" accept="video/*" disabled={busy}
                   onChange={(e) => onUpload(e.target.files)} />
            {busy && !running ? "Uploading…" : "Upload a video"}
          </label>

          <div className="side__head">Clips · {clips.length}</div>
          <ul className="list">
            {clips.map((c) => (
              <li key={c.clip_id} className={c.clip_id === clip?.clip_id ? "is-selected" : ""}>
                <button className="pick" onClick={() => chooseClip(c)} disabled={running}>
                  <span className="name">{c.display_name}</span>
                  <span className="meta">{c.width}×{c.height} · {fmtDuration(c.duration)}</span>
                </button>
                <button className="del" title="Delete clip and its renders"
                        onClick={() => removeClip(c.clip_id)} disabled={running}>✕</button>
              </li>
            ))}
            {!clips.length && <li className="empty">Nothing uploaded yet.</li>}
          </ul>
        </aside>

        <main className="main">
          {!clip && <div className="placeholder">Upload or pick a clip to begin.</div>}

          {clip && (
            <>
              <div className="stage">
                {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                <video ref={videoRef} src={src} controls className="video"
                       onPlay={() => setPaused(false)} onPause={() => setPaused(true)} />
                <BoxSelect active={canDraw} box={box} onChange={setBox}
                           getVideo={() => videoRef.current} />
                {selected && (
                  <button className="stage__close" onClick={() => setSelected(null)}
                          title="Back to the original video" aria-label="Back to the original video">✕</button>
                )}
              </div>

              <div className="controls">
                {selected ? (
                  <span className="hint">
                    Showing render <b>{selected.label || timeAgo(selected.created_at)}</b>
                    {" · "}{selected.zoom_w}×{selected.zoom_h} zoom window
                  </span>
                ) : running ? (
                  <span className="render-live">
                    <span className="progress">
                      <span className="progress__fill"
                            style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }} />
                    </span>
                    <span className="pct">{Math.round((job?.progress ?? 0) * 100)}%</span>
                    <button className="cancel" onClick={stop} disabled={canceling}
                            title={canceling ? "Stopping…" : "Stop this render"}>
                      {canceling ? "…" : "✕"}
                    </button>
                  </span>
                ) : (
                  <>
                    <span className="hint">
                      {paused
                        ? (box
                            ? `Selected ${Math.round((box.x2 - box.x1) * clip.width)}×${Math.round((box.y2 - box.y1) * clip.height)}px — that exact box is the zoom.`
                            : "Pause where you want to start, then drag a box around the object.")
                        : "Pause the video to draw a selection."}
                    </span>
                    <button className="primary" onClick={render} disabled={!box || busy}>
                      Track &amp; zoom
                    </button>
                    {box && <button className="ghost" onClick={() => setBox(null)}>Clear</button>}
                  </>
                )}
              </div>

              {job?.status === "failed" && <p className="error">{job.error || "render failed"}</p>}
              {job?.status === "canceled" && (
                <p className="hint">Render stopped — nothing was saved.</p>
              )}
              {error && <p className="error">{error}</p>}

              {renders.length > 0 && (
                <div className="renders">
                  <div className="renders__head">Saved renders · {renders.length}</div>
                  <ul className="list">
                    {renders.map((r) => (
                      <li key={r.render_id}
                          className={r.render_id === selected?.render_id ? "is-selected" : ""}>
                        <button className="pick" onClick={() => setSelected(r)}
                                title="Play this render">
                          <span className="name">{r.label || timeAgo(r.created_at)}</span>
                          <span className="meta">
                            {r.frames_tracked}/{r.frame_count} tracked · {r.zoom_w}×{r.zoom_h}
                            {r.frames_occluded > 0 && ` · ${r.frames_occluded} occluded`}
                          </span>
                        </button>
                        <button className="del" title="Delete this render"
                                onClick={() => removeRender(r.render_id)}>✕</button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
