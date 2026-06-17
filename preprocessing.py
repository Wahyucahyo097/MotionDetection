from __future__ import annotations

import cv2
import numpy as np

from config import Config


class Preprocessor:
    """Preprocessing pipeline: resize -> grayscale -> GaussianBlur -> optional threshold."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process frame.

        Returns:
            (gray_blur, gray_for_display)
        """
        # Resize
        frame = frame_bgr
        if self.cfg.resize_width > 0:
            h, w = frame.shape[:2]
            scale = self.cfg.resize_width / float(w)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (self.cfg.resize_width, new_h), interpolation=cv2.INTER_AREA)

        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur
        blur = cv2.GaussianBlur(gray, self.cfg.gaussian_kernel, 0)

        if self.cfg.thresholding_enabled:
            _th = self._apply_threshold(blur)
            # For downstream background subtractors we still return blur image
            return _th, blur

        return blur, blur

    def _apply_threshold(self, gray_blur: np.ndarray) -> np.ndarray:
        """Apply binary thresholding."""
        _, th = cv2.threshold(
            gray_blur,
            int(self.cfg.threshold),
            255,
            cv2.THRESH_BINARY,
        )
        return th

