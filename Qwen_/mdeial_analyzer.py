"""
Unified Video/Image Analyzer using Qwen3-VL
=============================================

Give it a path (image OR video). It figures out which one it is,
extracts frames if it's a video, runs Qwen3-VL on the frame(s), and
returns/prints a structured analysis covering:
    - Object detection
    - People / celebrity / entity recognition
    - OCR (on-screen / in-scene text)
    - General scene description

MODEL DOWNLOAD
--------------
Model weights are NOT downloaded manually. The first time you run this
script, `transformers` will pull the weights from the Hugging Face Hub
and cache them locally (default cache: ~/.cache/huggingface/hub).

Model page (browse / manual clone if you want):
    https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
Full model family / bigger or smaller variants:
    https://huggingface.co/collections/Qwen/qwen3-vl

If you want to pre-download instead of lazy-loading on first run:
    pip install -U "huggingface_hub[cli]"
    huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir ./models/Qwen3-VL-30B-A3B-Instruct

INSTALL
-------
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U "transformers>=4.57.0" accelerate opencv-python pillow qwen-vl-utils
# flash-attn is optional but recommended given "no hardware limits":
pip install flash-attn --no-build-isolation

USAGE
-----
python media_analyzer.py --input /path/to/video_or_image.mp4
python media_analyzer.py --input /path/to/photo.jpg --prompt "Custom prompt here"
python media_analyzer.py --input clip.mp4 --frame-interval 30 --max-frames 40
"""

import argparse
import json
import mimetypes
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# Other options (pass via --model):
#   "Qwen/Qwen3-VL-8B-Instruct"        -> smaller dense model, faster, lighter VRAM
#   "Qwen/Qwen3-VL-235B-A22B-Instruct" -> flagship, needs multi-GPU / huge VRAM

DEFAULT_ANALYSIS_PROMPT = """You are a meticulous visual analysis engine. Analyze this image thoroughly and return ONLY valid JSON (no markdown fences, no commentary) with this exact schema:

{
  "scene_description": "one or two sentence overview of the scene",
  "objects": [{"label": "string", "description": "string", "approx_location": "e.g. top-left, center, foreground"}],
  "people": [{"description": "physical description / role in scene", "identity_guess": "named person or celebrity if recognizable with reasonable confidence, else null", "confidence": "high/medium/low", "action": "what they are doing"}],
  "entities_or_brands": ["any recognizable brands, logos, landmarks, or named entities"],
  "ocr_text": ["exact text strings visible in the image, e.g. signage, captions, labels"],
  "notable_details": "anything else relevant (emotions, setting, lighting, event type, etc.)"
}

Rules:
- If a category has nothing to report, use an empty list / null, don't omit the key.
- For people identification, only name someone if you are reasonably confident; otherwise describe them generically and set identity_guess to null.
- Transcribe OCR text exactly as seen, preserving case.
- Output must be parseable JSON and nothing else.
"""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}


# ---------------------------------------------------------------------------
# Input type detection
# ---------------------------------------------------------------------------

def detect_media_type(path: str) -> str:
    """Return 'image' or 'video' based on extension, falling back to mimetype sniffing."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"

    mime, _ = mimetypes.guess_type(path)
    if mime:
        if mime.startswith("image"):
            return "image"
        if mime.startswith("video"):
            return "video"

    # Last resort: try opening with OpenCV's VideoCapture; if it reports
    # more than 1 frame, treat as video.
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count > 1:
            return "video"
        if frame_count == 1:
            return "image"

    raise ValueError(f"Could not determine media type for: {path}")


# ---------------------------------------------------------------------------
# Frame extraction (OpenCV)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedFrame:
    index: int
    timestamp_sec: float
    image: Image.Image


def extract_frames(
    video_path: str,
    frame_interval: int = 30,
    max_frames: int = 40,
    scene_change_mode: bool = False,
    scene_change_threshold: float = 30.0,
) -> List[ExtractedFrame]:
    """
    Extract frames from a video using OpenCV.

    frame_interval: sample every Nth frame (simple uniform sampling).
    max_frames: hard cap so extremely long videos don't blow up memory/compute.
    scene_change_mode: if True, instead of uniform sampling, only grab frames
        where the visual content changed significantly (good for catching new
        shots/scenes rather than redundant near-duplicate frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: List[ExtractedFrame] = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        take_frame = False
        if scene_change_mode:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if prev_gray is None:
                take_frame = True
            else:
                diff = cv2.absdiff(gray, prev_gray)
                score = diff.mean()
                if score > scene_change_threshold:
                    take_frame = True
            prev_gray = gray
        else:
            take_frame = (frame_idx % frame_interval == 0)

        if take_frame:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            timestamp = frame_idx / fps
            frames.append(ExtractedFrame(index=frame_idx, timestamp_sec=timestamp, image=pil_img))

            if len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted from video — check the file / interval settings.")

    print(f"[frames] video: {total_frames} total frames @ {fps:.2f} fps -> extracted {len(frames)} frames")
    return frames


# ---------------------------------------------------------------------------
# Qwen3-VL wrapper
# ---------------------------------------------------------------------------

class QwenVLAnalyzer:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"[model] loading {model_id} (first run will download + cache the weights)...")

        attn_impl = "flash_attention_2" if self._flash_attn_available() else "sdpa"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"[model] loaded. attn_implementation={attn_impl}, device_map={device_map}")

    @staticmethod
    def _flash_attn_available() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    def analyze_image(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_ANALYSIS_PROMPT,
        max_new_tokens: int = 1024,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text


# ---------------------------------------------------------------------------
# JSON parsing helper (models sometimes wrap output in ```json fences anyway)
# ---------------------------------------------------------------------------

def parse_model_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json"):]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_output": raw_text, "parse_error": True}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_media(
    input_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    prompt: str = DEFAULT_ANALYSIS_PROMPT,
    frame_interval: int = 30,
    max_frames: int = 40,
    scene_change_mode: bool = False,
) -> dict:
    media_type = detect_media_type(input_path)
    print(f"[input] detected media type: {media_type}")

    analyzer = QwenVLAnalyzer(model_id=model_id)

    result = {"input_path": input_path, "media_type": media_type}

    if media_type == "image":
        img = Image.open(input_path).convert("RGB")
        raw_output = analyzer.analyze_image(img, prompt=prompt)
        result["analysis"] = parse_model_json(raw_output)

    else:  # video
        frames = extract_frames(
            input_path,
            frame_interval=frame_interval,
            max_frames=max_frames,
            scene_change_mode=scene_change_mode,
        )
        per_frame_results = []
        for f in frames:
            print(f"[analyze] frame idx={f.index} t={f.timestamp_sec:.2f}s ...")
            raw_output = analyzer.analyze_image(f.image, prompt=prompt)
            per_frame_results.append({
                "frame_index": f.index,
                "timestamp_sec": round(f.timestamp_sec, 2),
                "analysis": parse_model_json(raw_output),
            })
        result["frame_count_analyzed"] = len(per_frame_results)
        result["frames"] = per_frame_results

    return result


def main():
    parser = argparse.ArgumentParser(description="Unified image/video analyzer using Qwen3-VL")
    parser.add_argument("--input", required=True, help="Path to an image or video file")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--prompt", default=None, help="Custom analysis prompt (overrides default JSON schema prompt)")
    parser.add_argument("--frame-interval", type=int, default=30, help="Sample every Nth frame (uniform sampling mode)")
    parser.add_argument("--max-frames", type=int, default=40, help="Max frames to analyze from a video")
    parser.add_argument("--scene-change", action="store_true", help="Use scene-change detection instead of uniform interval sampling")
    parser.add_argument("--output", default=None, help="Path to save JSON results (default: <input_name>_analysis.json)")
    args = parser.parse_args()

    prompt = args.prompt if args.prompt else DEFAULT_ANALYSIS_PROMPT

    result = analyze_media(
        input_path=args.input,
        model_id=args.model,
        prompt=prompt,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        scene_change_mode=args.scene_change,
    )

    out_path = args.output or (str(Path(args.input).with_suffix("")) + "_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[done] results saved to: {out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()