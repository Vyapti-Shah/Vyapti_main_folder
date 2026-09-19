/* The Track & Zoom API client. Relative URLs throughout: vite proxies /api in
 * dev and nginx proxies it in the container, so one client works in both. */

export interface Clip {
  clip_id: string;
  filename: string;
  display_name: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration: number;
  created_at: number;
}

export interface Render {
  render_id: string;
  clip_id: string;
  filename: string;
  label: string;
  seed_frame_idx: number;
  seed_box_norm: number[];
  frame_count: number;
  fps: number;
  frames_tracked: number;
  frames_occluded: number;
  zoom_w: number;
  zoom_h: number;
  created_at: number;
}

export interface JobStatus {
  /** "canceled" is a normal terminal state, not a failure, so it carries no error. */
  status: "running" | "done" | "failed" | "canceled";
  progress: number;
  error?: string | null;
  render_id?: string;
  cancel_requested?: boolean;
  updated_at?: number;
}

export interface Health {
  status: string;
  sam2: {
    importable: boolean;
    model: string;
    weights_path: string;
    weights_present: boolean;
    device: string;
  };
  zoom: Record<string, unknown>;
  data_dir: string;
}

async function json<T>(r: Response, what: string): Promise<T> {
  if (!r.ok) {
    const detail = await r.json().catch(() => ({} as { detail?: string }));
    throw new Error(detail.detail || `${what} failed (${r.status})`);
  }
  return r.json() as Promise<T>;
}

export const getHealth = async (): Promise<Health | null> => {
  const r = await fetch("/api/health");
  return r.ok ? r.json() : null;
};

export const listClips = async (): Promise<Clip[]> => {
  const r = await fetch("/api/clips");
  return r.ok ? r.json() : [];
};

export async function uploadClip(file: File): Promise<Clip> {
  const fd = new FormData();
  fd.append("file", file);
  return json<Clip>(await fetch("/api/clips", { method: "POST", body: fd }), "upload");
}

export async function deleteClip(clipId: string): Promise<void> {
  await json(await fetch(`/api/clips/${clipId}`, { method: "DELETE" }), "delete clip");
}

export const clipStreamUrl = (clipId: string) => `/api/clips/${clipId}/stream`;
export const renderStreamUrl = (clipId: string, renderId: string) =>
  `/api/clips/${clipId}/renders/${renderId}/stream`;

export async function startRender(
  clipId: string, boxNorm: number[], timeS: number, label: string,
): Promise<JobStatus> {
  return json<JobStatus>(await fetch(`/api/clips/${clipId}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ box_norm: boxNorm, time_s: timeS, label }),
  }), "start render");
}

export const getRenderStatus = async (clipId: string): Promise<JobStatus | null> => {
  const r = await fetch(`/api/clips/${clipId}/render/status`);
  return r.ok ? r.json() : null;
};

/** Cooperative: the worker notices within a frame, so the job stays `running`
 *  for a moment after this resolves. */
export const cancelRender = async (clipId: string): Promise<JobStatus | null> => {
  const r = await fetch(`/api/clips/${clipId}/render/cancel`, { method: "POST" });
  return r.ok ? r.json() : null;
};

export const listRenders = async (clipId: string): Promise<Render[]> => {
  const r = await fetch(`/api/clips/${clipId}/renders`);
  return r.ok ? r.json() : [];
};

export async function deleteRender(clipId: string, renderId: string): Promise<void> {
  await json(await fetch(`/api/clips/${clipId}/renders/${renderId}`, { method: "DELETE" }),
             "delete render");
}

export function fmtDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  return `${m}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
}

export function timeAgo(epochSeconds: number): string {
  const s = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
