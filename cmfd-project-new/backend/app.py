import dataclasses
import traceback

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detector import detect

app = FastAPI(title="Copy-Move Forgery Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIZE = 15 * 1024 * 1024  # 15 MB


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect_endpoint(image: UploadFile = File(...)):
    data = await image.read()

    if not data:
        return JSONResponse(status_code=400, content={"error": "Empty file"})
    if len(data) > MAX_SIZE:
        return JSONResponse(status_code=400, content={"error": "File too large (max 15MB)"})

    try:
        result = detect(data)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {str(e)}"})

    return {
        "filename": image.filename,
        "total_keypoints": result.total_keypoints,
        "raw_matches": result.raw_matches,
        "matches_after_ratio_test": result.matches_after_ratio_test,
        "matches_after_distance_filter": result.matches_after_distance_filter,
        "matches_after_geometric_verification": result.matches_after_geometric_verification,
        "verdict": result.verdict,
        "overall_confidence": result.overall_confidence,
        "region_pairs": [dataclasses.asdict(r) for r in result.region_pairs],
        "annotated_image_b64": result.annotated_image_b64,
        "mask_image_b64": result.mask_image_b64,
    }
