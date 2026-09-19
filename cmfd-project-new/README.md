# Copy-Move Forgery (Cloning) Detection

Detects copy-paste ("clone") manipulation within a single image: finds the
**source** region and the **destination (cloned)** region, and returns both
a JSON forensic report and an annotated image / mask.

## Run it

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/api/detect (POST, multipart field `image`)

No GPU or model download required — everything runs on CPU inside the containers.

## How detection works (accuracy-focused pipeline)

This uses a **classical, verifiable forensic pipeline** rather than a black-box
deep model, because for copy-move forgery this approach is generally more
reliable and auditable at inference time:

1. **Preprocess** — grayscale + denoise.
2. **Feature extraction** — SIFT (falls back to ORB if SIFT unavailable) with
   up to 8,000 keypoints, tuned for weak-contrast/textured regions so subtle
   clones aren't missed.
3. **Self-matching** — every descriptor is matched against *every other*
   descriptor in the *same* image (FLANN for SIFT / brute-force Hamming for ORB),
   k=6 neighbors so the trivial "self" match doesn't hide the real second-best.
4. **Lowe's ratio test** (0.75) to drop ambiguous matches.
5. **Minimum-distance filter** — matches whose two points are closer than 40px
   are dropped (these are almost always the same real object/texture, not a clone).
6. **Displacement clustering** — real copy-move forgery produces many matches
   that all share nearly the same (dx, dy) shift (the patch was moved as a
   whole). Matches are grouped by displacement vector; small/inconsistent
   groups (typical of repetitive textures like tiles, bricks, windows,
   foliage) are discarded.
7. **RANSAC geometric verification** — an affine transform is fit per cluster
   and only inlier matches are kept, rejecting anything that isn't
   geometrically consistent with a genuine copy-paste.
8. **Confidence scoring** per region pair combines: inlier match count,
   displacement consistency (low std-dev = more suspicious), descriptor
   similarity, and region area.
9. **Output** — annotated image (source = green outline/dots, cloned =
   red outline/dots + confidence %), a binary/gray mask (source=gray,
   clone=white), and a structured report.

This directly targets the false-positive sources called out for this kind of
system (repeated windows, tiles, bricks, text, foliage) via steps 5–7, which
is where naive "just match features" approaches lose accuracy.

### About the MGCFDN reference model

The requested repo (`tuhanglsWorld/MGCFDN`) is a research codebase without a
packaged pip install or a guaranteed public pretrained-weights release, so it
isn't something that can be reliably wired into a one-command
`docker compose up` deployment. The pipeline above is the accuracy-oriented,
dependency-free alternative. If you obtain MGCFDN's checkpoint file yourself,
it can be dropped into `backend/models/` and `detector.py` extended with a
second scoring path (deep-model heatmap) that's fused with the geometric
score above — ask and this can be scaffolded.

## API

`POST /api/detect` — multipart form field `image`. Response JSON:

```json
{
  "filename": "suspect_image.jpg",
  "total_keypoints": 4832,
  "raw_matches": 187,
  "matches_after_ratio_test": 187,
  "matches_after_distance_filter": 61,
  "matches_after_geometric_verification": 42,
  "verdict": "⚠ Potential Copy-Move Manipulation Detected",
  "overall_confidence": 91.4,
  "region_pairs": [
    {
      "source_bbox": [120, 180, 130, 160],
      "destination_bbox": [430, 175, 130, 160],
      "num_matches": 42,
      "displacement": [310.2, -4.1],
      "confidence": 91.4
    }
  ],
  "annotated_image_b64": "...",
  "mask_image_b64": "..."
}
```

## Tuning for your dataset

In `backend/detector.py`:
- `MIN_DISTANCE_PX` — raise if you get false positives on fine repeated texture.
- `MIN_CLUSTER_SIZE` — raise for stricter detection (fewer false positives),
  lower to catch smaller clones.
- `RATIO_TEST_THRESH` — lower (e.g. 0.65) for stricter/more precise matches.
- `RANSAC_REPROJ_THRESH` — lower for stricter geometric agreement.
