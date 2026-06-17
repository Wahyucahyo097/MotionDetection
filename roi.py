from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ROIState:
    drawing: bool = False
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0
    roi_rect: Optional[tuple[int, int, int, int]] = None  # x,y,w,h


class ROIManager:
    """Mouse-driven ROI rectangle.

    - User drags rectangle on display
    - We create a mask for the selected rectangle
    """

    def __init__(self) -> None:
        self.state = ROIState()

    def set_from_drag_start(self, x: int, y: int) -> None:
        self.state.drawing = True
        self.state.x0, self.state.y0 = int(x), int(y)
        self.state.x1, self.state.y1 = int(x), int(y)
        self.state.roi_rect = None

    def set_from_drag_move(self, x: int, y: int) -> None:
        if not self.state.drawing:
            return
        self.state.x1, self.state.y1 = int(x), int(y)

    def set_from_drag_end(self, x: int, y: int) -> None:
        if not self.state.drawing:
            return
        self.state.drawing = False
        self.state.x1, self.state.y1 = int(x), int(y)
        self._finalize_rect()

    def clear(self) -> None:
        self.state = ROIState()

    def _finalize_rect(self) -> None:
        x0, y0, x1, y1 = self.state.x0, self.state.y0, self.state.x1, self.state.y1
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        w, h = (x_max - x_min), (y_max - y_min)
        if w <= 2 or h <= 2:
            self.state.roi_rect = None
            return
        self.state.roi_rect = (x_min, y_min, w, h)

    @property
    def roi_rect(self) -> Optional[tuple[int, int, int, int]]:
        return self.state.roi_rect

    def apply_mask(self, frame_gray: np.ndarray) -> np.ndarray:
        """Return a masked frame_gray (zeros outside ROI)."""
        if self.state.roi_rect is None:
            return frame_gray
        x, y, w, h = self.state.roi_rect
        out = np.zeros_like(frame_gray)
        out[y : y + h, x : x + w] = frame_gray[y : y + h, x : x + w]
        return out

    def apply_mask_to_mask(self, mask: np.ndarray) -> np.ndarray:
        """Mask a foreground mask with ROI rectangle."""
        if self.state.roi_rect is None:
            return mask
        x, y, w, h = self.state.roi_rect
        out = np.zeros_like(mask)
        out[y : y + h, x : x + w] = mask[y : y + h, x : x + w]
        return out

    def draw_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw ROI rectangle overlay (for preview)."""
        out = frame_bgr
        if self.state.drawing:
            cv2.rectangle(
                out,
                (self.state.x0, self.state.y0),
                (self.state.x1, self.state.y1),
                (0, 255, 255),
                2,
            )
        if self.state.roi_rect is not None:
            x, y, w, h = self.state.roi_rect
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return out

