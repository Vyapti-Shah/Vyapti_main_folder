export interface DetectionResult {
  file_id: string;
  file_name: string;
  file_type: "image" | "video";
  is_fake: boolean;
  confidence: number;
  output_path: string;
  created_at: string;
  frames_analyzed: number | null;
}

export async function detectFile(
  file: File,
  onProgress?: (percent: number) => void
): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/detect");

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || "Detection failed"));
        } catch {
          reject(new Error("Detection failed"));
        }
      }
    };

    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(formData);
  });
}

export function resultUrl(path: string): string {
  return path;
}
