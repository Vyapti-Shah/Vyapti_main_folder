"""
Copy-Move Forgery Detection (CMFD) engine.

Pipeline:
  1. Preprocess (grayscale, denoise)
  2. Dense keypoint + descriptor extraction (SIFT primary, ORB fallback)
  3. Self-matching (image against itself) via FLANN, k-NN with ratio test
  4. Remove trivial matches (near-duplicate / near-distance points)
  5. RANSAC-based geometric verification (estimate affine transform,
     keep only matches that agree with a consistent transform)
  6. Cluster matches by spatial displacement vector (agglomerative) to find
     groups of matches that share a near-identical (dx, dy) shift ->
     strongest signal of real copy-move forgery vs. natural repetition
  7. Build source / destination convex hulls + bounding boxes per cluster
  8. Score confidence per cluster from: number of inlier matches,
     displacement consistency (std-dev of dx,dy), descriptor distance,
     and region area
  9. Produce annotated image (source in green, cloned/destination in red)
     and a structured JSON report
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import base64


@dataclass
class RegionPair:
    source_bbox: Tuple[int, int, int, int]       # x, y, w, h
    destination_bbox: Tuple[int, int, int, int]
    source_hull: List[Tuple[int, int]]
    destination_hull: List[Tuple[int, int]]
    num_matches: int
    displacement: Tuple[float, float]
    displacement_std: Tuple[float, float]
    mean_descriptor_distance: float
    confidence: float


@dataclass
class DetectionResult:
    total_keypoints: int
    raw_matches: int
    matches_after_ratio_test: int
    matches_after_distance_filter: int
    matches_after_geometric_verification: int
    region_pairs: List[RegionPair] = field(default_factory=list)
    annotated_image_b64: str = ""
    mask_image_b64: str = ""
    verdict: str = "No Manipulation Detected"
    overall_confidence: float = 0.0


MIN_DISTANCE_PX = 40          # reject matches whose points are this close (self-similar texture)
RATIO_TEST_THRESH = 0.75      # Lowe's ratio test
RANSAC_REPROJ_THRESH = 5.0
MIN_CLUSTER_SIZE = 4          # minimum inlier matches to call a region a clone
DISPLACEMENT_CLUSTER_TOL = 15 # px tolerance when grouping matches by (dx, dy)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    return gray


def _extract_features(gray: np.ndarray):
    """Try SIFT (higher accuracy for subtle/rotated/scaled clones), fall back to ORB."""
    try:
        sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=15)
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is not None and len(kp) > 20:
            return kp, desc, "SIFT", cv2.NORM_L2
    except Exception:
        pass
    orb = cv2.ORB_create(nfeatures=8000)
    kp, desc = orb.detectAndCompute(gray, None)
    return kp, desc, "ORB", cv2.NORM_HAMMING


def _self_match(desc, norm_type):
    if norm_type == cv2.NORM_L2:
        index_params = dict(algorithm=1, trees=5)  # FLANN KD-tree
        search_params = dict(checks=64)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc, desc, k=6)  # k>2 because best match = itself
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc, desc, k=6)
    return matches


def _filter_matches(matches, kp):
    """Drop self-match (distance 0, same index), apply ratio test + min distance."""
    good = []
    for group in matches:
        # first entry is always the point matched to itself (distance ~0); skip it
        candidates = [m for m in group if m.trainIdx != m.queryIdx]
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda m: m.distance)
        m, n = candidates[0], candidates[1]
        if m.distance >= RATIO_TEST_THRESH * n.distance:
            continue
        p1 = np.array(kp[m.queryIdx].pt)
        p2 = np.array(kp[m.trainIdx].pt)
        if np.linalg.norm(p1 - p2) < MIN_DISTANCE_PX:
            continue
        # canonical ordering to avoid duplicate (a,b)/(b,a) pairs
        if m.queryIdx > m.trainIdx:
            continue
        good.append((m.queryIdx, m.trainIdx, m.distance))
    return good


def _cluster_by_displacement(good_matches, kp):
    """Group matches whose (dx, dy) displacement vectors agree -> a real copy-move
    typically comes from one translation/affine transform applied to a whole patch."""
    if not good_matches:
        return []

    pts1 = np.array([kp[a].pt for a, b, d in good_matches])
    pts2 = np.array([kp[b].pt for a, b, d in good_matches])
    disp = pts2 - pts1

    clusters = []
    used = np.zeros(len(good_matches), dtype=bool)

    for i in range(len(good_matches)):
        if used[i]:
            continue
        base = disp[i]
        dists = np.linalg.norm(disp - base, axis=1)
        idxs = np.where((dists < DISPLACEMENT_CLUSTER_TOL) & (~used))[0]
        if len(idxs) < MIN_CLUSTER_SIZE:
            continue
        used[idxs] = True
        clusters.append(idxs)

    return clusters, pts1, pts2, disp, [d for _, _, d in good_matches]


def _verify_cluster_geometry(pts1_c, pts2_c):
    """RANSAC affine fit; returns inlier mask or None if not enough consensus."""
    if len(pts1_c) < 3:
        return None
    try:
        M, inliers = cv2.estimateAffinePartial2D(
            pts1_c.astype(np.float32), pts2_c.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESH, maxIters=3000
        )
    except Exception:
        return None
    if M is None or inliers is None:
        return None
    return inliers.ravel().astype(bool)


def _bbox_and_hull(points):
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    hull = cv2.convexHull(points.astype(np.int32)).reshape(-1, 2).tolist()
    return (int(x), int(y), int(w), int(h)), [(int(px), int(py)) for px, py in hull]


def detect(image_bytes: bytes) -> DetectionResult:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    h_img, w_img = image.shape[:2]
    gray = _to_gray(image)
    kp, desc, method, norm_type = _extract_features(gray)

    result = DetectionResult(
        total_keypoints=len(kp) if kp else 0,
        raw_matches=0,
        matches_after_ratio_test=0,
        matches_after_distance_filter=0,
        matches_after_geometric_verification=0,
    )

    if desc is None or len(kp) < 20:
        result.annotated_image_b64 = _encode(image)
        result.mask_image_b64 = _encode(np.zeros_like(image))
        return result

    raw = _self_match(desc, norm_type)
    result.raw_matches = sum(len(g) for g in raw)

    good = _filter_matches(raw, kp)
    result.matches_after_ratio_test = len(raw)
    result.matches_after_distance_filter = len(good)

    if len(good) < MIN_CLUSTER_SIZE:
        result.annotated_image_b64 = _encode(image)
        result.mask_image_b64 = _encode(np.zeros_like(image))
        return result

    clusters, pts1, pts2, disp, dists = _cluster_by_displacement(good, kp)

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    annotated = image.copy()
    region_pairs = []
    total_inliers = 0

    for idxs in clusters:
        p1c, p2c = pts1[idxs], pts2[idxs]
        inlier_mask = _verify_cluster_geometry(p1c, p2c)
        if inlier_mask is None or inlier_mask.sum() < MIN_CLUSTER_SIZE:
            continue
        p1_in, p2_in = p1c[inlier_mask], p2c[inlier_mask]
        d_in = np.array(dists)[idxs][inlier_mask]
        disp_in = p2_in - p1_in

        total_inliers += len(p1_in)
        src_bbox, src_hull = _bbox_and_hull(p1_in)
        dst_bbox, dst_hull = _bbox_and_hull(p2_in)

        disp_mean = disp_in.mean(axis=0)
        disp_std = disp_in.std(axis=0)

        area_frac = (src_bbox[2] * src_bbox[3]) / float(w_img * h_img)
        match_score = min(1.0, len(p1_in) / 40.0)
        consistency_score = 1.0 / (1.0 + disp_std.mean())
        distance_score = max(0.0, 1.0 - (d_in.mean() / (256.0 if norm_type == cv2.NORM_HAMMING else 400.0)))
        area_score = min(1.0, area_frac * 20)
        confidence = float(np.clip(
            0.35 * match_score + 0.30 * consistency_score + 0.20 * distance_score + 0.15 * area_score,
            0, 1
        )) * 100

        region_pairs.append(RegionPair(
            source_bbox=src_bbox,
            destination_bbox=dst_bbox,
            source_hull=src_hull,
            destination_hull=dst_hull,
            num_matches=int(len(p1_in)),
            displacement=(float(disp_mean[0]), float(disp_mean[1])),
            displacement_std=(float(disp_std[0]), float(disp_std[1])),
            mean_descriptor_distance=float(d_in.mean()),
            confidence=round(confidence, 1),
        ))

        # draw on annotated image: source = GREEN, destination(clone) = RED
        cv2.polylines(annotated, [np.array(src_hull, dtype=np.int32)], True, (0, 200, 0), 3)
        cv2.polylines(annotated, [np.array(dst_hull, dtype=np.int32)], True, (0, 0, 255), 3)
        cv2.rectangle(annotated, (src_bbox[0], src_bbox[1]),
                      (src_bbox[0]+src_bbox[2], src_bbox[1]+src_bbox[3]), (0, 200, 0), 1)
        cv2.rectangle(annotated, (dst_bbox[0], dst_bbox[1]),
                      (dst_bbox[0]+dst_bbox[2], dst_bbox[1]+dst_bbox[3]), (0, 0, 255), 1)
        cv2.putText(annotated, "SOURCE", (src_bbox[0], max(20, src_bbox[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(annotated, f"CLONED {confidence:.0f}%", (dst_bbox[0], max(20, dst_bbox[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (x, y) in p1_in.astype(int):
            cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in p2_in.astype(int):
            cv2.circle(annotated, (x, y), 2, (0, 0, 255), -1)
        cv2.fillConvexPoly(mask, np.array(src_hull, dtype=np.int32), 128)
        cv2.fillConvexPoly(mask, np.array(dst_hull, dtype=np.int32), 255)

    result.matches_after_geometric_verification = total_inliers
    region_pairs.sort(key=lambda r: -r.confidence)
    result.region_pairs = region_pairs

    if region_pairs:
        result.overall_confidence = round(max(r.confidence for r in region_pairs), 1)
        result.verdict = "⚠ Potential Copy-Move Manipulation Detected"
    else:
        result.overall_confidence = 0.0
        result.verdict = "No Manipulation Detected"

    result.annotated_image_b64 = _encode(annotated)
    result.mask_image_b64 = _encode(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    return result


def _encode(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")
