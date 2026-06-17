from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from config import Config


@dataclass
class DetectedObject:
    label: str
    bbox_xywh: tuple[int, int, int, int]  # x, y, w, h
    centroid_xy: tuple[int, int]
    area: float


class MotionDetector:
    """Detect moving objects from a foreground mask using contours."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphology to reduce noise."""
        k = int(self.cfg.morph_kernel_size)
        if k < 1:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = mask
        it = int(max(1, self.cfg.morph_iterations))

        # Use opening + closing to remove small noise and fill holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=it)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=it)
        return cleaned

    def detect_from_mask(
        self,
        frame_bgr: np.ndarray,
        fg_mask: np.ndarray,
    ) -> tuple[List[DetectedObject], np.ndarray]:
        """Run contour detection and draw bounding boxes.

        Returns:
            (objects, annotated_frame)
        """
        mask = fg_mask.copy()
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cleaned = self.clean_mask(mask)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects: List[DetectedObject] = []
        annotated = frame_bgr

        for idx, cnt in enumerate(contours, start=1):
            area = float(cv2.contourArea(cnt))
            if area < float(self.cfg.min_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            centroid = self._centroid(cnt)
            label = f"Object {len(objects) + 1}"

            obj = DetectedObject(
                label=label,
                bbox_xywh=(x, y, w, h),
                centroid_xy=centroid,
                area=area,
            )
            objects.append(obj)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cx, cy = centroid
            cv2.circle(annotated, (cx, cy), 3, (0, 200, 255), -1)
            cv2.putText(
                annotated,
                label,
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        return objects, annotated

    def _centroid(self, cnt: np.ndarray) -> tuple[int, int]:
        m = cv2.moments(cnt)
        if abs(m["m00"]) < 1e-6:
            # fallback: use bounding rect center
            x, y, w, h = cv2.boundingRect(cnt)
            return (x + w // 2, y + h // 2)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return (cx, cy)

