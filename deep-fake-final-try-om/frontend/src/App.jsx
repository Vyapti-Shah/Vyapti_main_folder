import DetectionCard from "./components/DetectionCard.jsx";
import { detectImage, detectVideo } from "./api.js";

export default function App() {
  return (
    <main>
      <header className="page-header">
        <h1>Deepfake Detection Studio</h1>
        <p className="sub">
          AI-generated image detection, powered by CommunityForensics ViT. Estimates are not
          guarantees, and the model is not a detector of every kind of manipulation.
        </p>
      </header>

      <div className="card-grid">
        <DetectionCard
          title="Image Detection"
          icon="🖼"
          accept="image/jpeg,image/png,image/webp"
          hint="JPG, PNG or WEBP"
          previewKind="image"
          submit={detectImage}
        />
        <DetectionCard
          title="Video Detection"
          icon="🎬"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
          hint="MP4, MOV, AVI, MKV or WEBM — analyzed frame by frame, not a true motion analysis"
          previewKind="video"
          submit={detectVideo}
        />
      </div>
    </main>
  );
}
