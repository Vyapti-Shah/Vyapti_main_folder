from click import group
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DigitShapeClassifier:
    """
    Classifies digits in the revealed (stripe-free) binary mask using
    pure shape analysis, for a custom pictographic digit style where:
      - a single ring/circle with a solid bar directly beneath it = "2"
      - two stacked rings/circles (same horizontal position)        = "8"
      - a single ring/circle alone (no bar beneath, no second ring) = "0"
      - a single solid bar with a ring/circle directly beneath it = "5"
    Any character group with no ring shapes at all is not covered by
    these rules and falls back to an external OCR reader supplied by
    the caller.
    """

    def __init__(self, min_component_area: int = 15,
                 min_overlap_ratio: float = 0.3,
                 min_hole_area_ratio: float = 0.12,
                 enable_ring_rules: bool = True):
        self.min_component_area = min_component_area
        # Two components are grouped as one character only when they are
        # STACKED (a ring with its bar directly beneath it), never based
        # on how close they sit horizontally. "Stacked" means: their y
        # (vertical) ranges barely overlap, AND their x (horizontal)
        # ranges overlap by at least this ratio (i.e. same column).
        # Side-by-side characters on the same baseline have the opposite
        # signature (high y-overlap, low/no x-overlap) and are always
        # kept as separate groups regardless of how close together they
        # are drawn — this works whether spacing is tight (e.g. "OPTIC")
        # or loose (e.g. digit strings), unlike a fixed pixel/ratio margin.
        self.min_overlap_ratio = min_overlap_ratio
        # a contour's hole only counts as a genuine "ring" if the hole
        # is at least this fraction of the outer contour's area — this
        # filters out small enclosed pockets (e.g. the numeral "4" can
        # form a tiny closed triangle after cleanup, which is NOT a
        # circle and must not be misread as a ring)
        self.min_hole_area_ratio = min_hole_area_ratio
        # The ring/bar pictographic rules (ring+bar="2", 2 rings="8",
        # lone ring="0") are only meaningful for images that are actually
        # using that custom digit style. Plain text images can contain
        # ring-shaped letters (e.g. "O", "Q", "D") that would otherwise
        # get misread as digits. Set this to False for text-only images
        # so every group is routed straight to ocr_fallback instead.
        self.enable_ring_rules = enable_ring_rules

    def _extract_components(self, mask: np.ndarray):
        """Returns a list of dicts: {bbox, has_hole, area} for every
        sufficiently large white blob in the mask."""
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []
        hierarchy = hierarchy[0]

        components = []
        for i, contour in enumerate(contours):
            parent = hierarchy[i][3]
            if parent != -1:
                continue  # this is a hole contour itself, not an outer blob

            area = cv2.contourArea(contour)
            if area < self.min_component_area:
                continue

            has_hole = False
            first_child = hierarchy[i][2]
            child = first_child
            while child != -1:
                hole_area = cv2.contourArea(contours[child])
                if area > 0 and (hole_area / area) >= self.min_hole_area_ratio:
                    has_hole = True
                    break
                child = hierarchy[child][0]  # next sibling hole

            x, y, w, h = cv2.boundingRect(contour)
            components.append({
                "bbox": (x, y, w, h),
                "has_hole": has_hole,
                "area": area,
            })
        return components

    def _is_stacked(self, a, b) -> bool:
        """True if components a and b look like a ring-over-bar pair:
        same column (x-ranges overlap), but one sits above the other
        (y-ranges do NOT overlap). This is a geometric relationship, not
        a distance/margin — so it works regardless of how tightly or
        loosely characters happen to be spaced in a given image."""
        ax, ay, aw, ah = a["bbox"]
        bx, by, bw, bh = b["bbox"]

        x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        min_w = min(aw, bw)
        x_overlap_ratio = x_overlap / min_w if min_w > 0 else 0

        y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
        min_h = min(ah, bh)
        y_overlap_ratio = y_overlap / min_h if min_h > 0 else 0

        return x_overlap_ratio >= self.min_overlap_ratio and y_overlap_ratio < 0.3

    def _group_components(self, components):
        """Groups components into characters using union-find: two
        components merge only if _is_stacked() says one sits directly
        above the other (ring + bar). Side-by-side components — even
        ones drawn only a few pixels apart, like tightly-kerned letters —
        never merge, since they share a baseline (high y-overlap) rather
        than a column."""
        if not components:
            return []

        n = len(components)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                if self._is_stacked(components[i], components[j]):
                    union(i, j)

        clusters = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(components[i])

        groups = []
        for comps in clusters.values():
            min_x = min(c["bbox"][0] for c in comps)
            groups.append({"components": comps, "x": min_x})

        groups.sort(key=lambda g: g["x"])
        return groups

    def _classify_group(self, group, mask: np.ndarray, ocr_fallback=None):
        comps = group["components"]
        ring_comps = [c for c in comps if c["has_hole"]]
        solid_comps = [c for c in comps if not c["has_hole"]]

        if self.enable_ring_rules:
            if len(ring_comps) == 2 and len(solid_comps) == 0:
                return "8"
            if len(ring_comps) == 1 and len(solid_comps) == 1:
                return "2"
            if len(ring_comps) == 1 and len(solid_comps) == 0:
                return "0"

            if ring_comps:
                logger.warning(f"Unrecognized ring pattern: {len(ring_comps)} rings, {len(solid_comps)} solids")
                return "?"

        if ocr_fallback is not None:
            xs = [c["bbox"][0] for c in comps]
            ys = [c["bbox"][1] for c in comps]
            xe = [c["bbox"][0] + c["bbox"][2] for c in comps]
            ye = [c["bbox"][1] + c["bbox"][3] for c in comps]
            pad = 10
            x0 = max(0, min(xs) - pad)
            y0 = max(0, min(ys) - pad)
            x1 = min(mask.shape[1], max(xe) + pad)
            y1 = min(mask.shape[0], max(ye) + pad)
            crop = mask[y0:y1, x0:x1]
            try:
                text = ocr_fallback(crop).strip()
                return text if text else "?"
            except Exception:
                logger.exception("OCR fallback failed on non-ring character group")
                return "?"

        return "?"

    def classify(self, mask: np.ndarray, ocr_fallback=None) -> str:
        """
        mask: revealed binary shape (white foreground on black background).
        ocr_fallback: optional callable(crop_image) -> str, used only for
        character groups that contain no ring shapes.
        """
        components = self._extract_components(mask)
        groups = self._group_components(components)

        chars = [self._classify_group(g, mask, ocr_fallback) for g in groups]
        return "".join(chars)