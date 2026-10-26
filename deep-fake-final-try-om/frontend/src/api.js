// Configurable at build time via VITE_API_BASE_URL. Defaults to the relative "/api" path,
// which nginx proxies to the backend container — the browser never needs to know a
// container-only hostname like "backend".
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

/** POST a file to `path` with upload-progress reporting. Rejects with a human-readable message. */
function uploadFile(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}${path}`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
    };

    xhr.onload = () => {
      let body;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300 && body) {
        onProgress?.(100);
        resolve(body);
      } else {
        reject(new Error(body?.detail || `Request failed (HTTP ${xhr.status}).`));
      }
    };

    xhr.onerror = () => reject(new Error("Could not reach the server. Check your connection and try again."));
    xhr.ontimeout = () => reject(new Error("The request timed out."));
    xhr.timeout = 10 * 60 * 1000;

    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export const detectImage = (file, onProgress) => uploadFile("/detect/image", file, onProgress);
export const detectVideo = (file, onProgress) => uploadFile("/detect/video", file, onProgress);

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
