import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StripePreprocessor:
    def __init__(self, config: dict):
        self.peak_radius = config["fft"]["peak_radius"]
        self.threshold_ratio = config["fft"]["threshold_ratio"]
        self.gabor_cfg = config["gabor"]
        self.orientation_cfg = config.get("orientation", {
            "blur_ksize": 9, "blur_sigma": 2, "min_mask_ratio": 0.005
        })

    # ---------- PRIMARY: structure-tensor orientation discontinuity ----------
    def orientation_discontinuity_mask(self, gray: np.ndarray) -> np.ndarray:
        """
        Finds regions where local line orientation deviates from the
        dominant (background) orientation. Works when hidden content
        shares the same stripe frequency as the background but is
        phase/direction-shifted (confirmed on real test image).
        """
        g = gray.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

        k = self.orientation_cfg["blur_ksize"]
        sigma = self.orientation_cfg["blur_sigma"]
        Jxx = cv2.GaussianBlur(gx * gx, (k, k), sigma)
        Jyy = cv2.GaussianBlur(gy * gy, (k, k), sigma)
        Jxy = cv2.GaussianBlur(gx * gy, (k, k), sigma)

        theta = 0.5 * np.arctan2(2 * Jxy, (Jxx - Jyy))
        theta_deg = np.degrees(theta)

        hist, edges = np.histogram(theta_deg, bins=180, range=(-90, 90))
        dominant = edges[np.argmax(hist)]

        diff = np.abs(((theta_deg - dominant + 90) % 180) - 90)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # light cleanup: remove speckle noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, diff_norm

    def _mask_is_usable(self, mask: np.ndarray) -> bool:
        """Sanity check: mask shouldn't be empty (nothing found) or ~all-white
        (dominant-orientation estimate failed, e.g. non-stripe backgrounds)."""
        ratio = np.count_nonzero(mask) / mask.size
        min_ratio = self.orientation_cfg["min_mask_ratio"]
        return min_ratio < ratio < 0.6

    # ---------- FALLBACK: FFT notch + Gabor (original spec) ----------
    def fft_notch_filter(self, img_gray: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2

        mag_search = magnitude.copy()
        mag_search[crow-5:crow+5, ccol-5:ccol+5] = 0

        thresh = self.threshold_ratio * mag_search.max()
        peak_coords = np.argwhere(mag_search > thresh)

        mask = np.ones((rows, cols), np.uint8)
        for (y, x) in peak_coords:
            cv2.circle(mask, (x, y), self.peak_radius, 0, -1)

        fshift_filtered = fshift * mask
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def gabor_bank_response(self, img_gray: np.ndarray) -> np.ndarray:
        c = self.gabor_cfg
        responses = []
        for theta in np.arange(0, np.pi, np.pi / 8):
            kernel = cv2.getGaborKernel(
                (c["ksize"], c["ksize"]), c["sigma"], theta,
                c["lambd"], c["gamma"], 0, ktype=cv2.CV_32F
            )
            responses.append(cv2.filter2D(img_gray, cv2.CV_8UC3, kernel))
        return np.max(np.stack(responses, axis=0), axis=0)

    def _fallback_process(self, gray: np.ndarray) -> np.ndarray:
        logger.warning("Orientation-discontinuity mask unusable — falling back to FFT/Gabor")
        fft_cleaned = self.fft_notch_filter(gray)
        gabor_resp = self.gabor_bank_response(fft_cleaned)
        combined = cv2.addWeighted(fft_cleaned, 0.5, gabor_resp, 0.5, 0)
        return cv2.equalizeHist(combined)

    # ---------- "Squint simulation": reveal low-frequency shape ----------
    def reveal_shape(self, mask: np.ndarray) -> np.ndarray:
        """
        The raw discontinuity mask is noisy at high frequency (the stripe
        pattern itself). Squinting / stepping back / shrinking the image
        works because it acts as a low-pass filter — it kills the
        high-frequency stripe noise while keeping the low-frequency shape
        of the hidden digit intact. This reproduces that effect
        numerically: blur, then shrink and grow the image back up
        (which forces additional low-pass smoothing via interpolation),
        then re-threshold and clean up small speckles.
        """
        h, w = mask.shape[:2]

        blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)

        small = cv2.resize(blurred, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

        _, revealed = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        revealed = cv2.morphologyEx(revealed, cv2.MORPH_OPEN, kernel)
        revealed = cv2.morphologyEx(revealed, cv2.MORPH_CLOSE, kernel)

        # drop tiny speckle noise, but keep every sufficiently large
        # component — a multi-digit number has multiple separate blobs,
        # so we must NOT collapse this down to a single "largest" one
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(revealed, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = areas.max()
            min_area_ratio = 0.08
            keep_labels = [
                i + 1 for i, a in enumerate(areas)
                if a >= max_area * min_area_ratio
            ]
            revealed = np.where(np.isin(labels, keep_labels), 255, 0).astype(np.uint8)

        return revealed
        

    # ---------- Public entry point ----------
    def process(self, image: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            mask, _ = self.orientation_discontinuity_mask(gray)

            if self._mask_is_usable(mask):
                return self.reveal_shape(mask)
            else:
                fallback = self._fallback_process(gray)
                return self.reveal_shape(fallback)

        except Exception:
            logger.exception("Stage1 preprocessing failed")
            raise