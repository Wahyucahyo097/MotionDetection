from __future__ import annotations

import cv2
import numpy as np

from config import Config


class BackgroundSubtractor:
    """Unified interface for MOG2 and KNN background subtraction."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.history),
            varThreshold=float(cfg.varThreshold),
            detectShadows=bool(cfg.detectShadows),
        )
        self.knn = cv2.createBackgroundSubtractorKNN(
            history=int(cfg.history),
            dist2Threshold=float(cfg.varThreshold),
            detectShadows=bool(cfg.detectShadows),
        )

    def apply(self, method: str, frame_gray_blur: np.ndarray, learning_rate: float) -> np.ndarray:
        """Apply selected background subtraction.

        Args:
            method: 'MOG2' or 'KNN'
            frame_gray_blur: single-channel image (gray/blur)
            learning_rate: learning rate to update the background model

        Returns:
            foreground mask (uint8)
        """
        lr = float(learning_rate)
        if method.upper() == "MOG2":
            mask = self.mog2.apply(frame_gray_blur, learningRate=lr)
        elif method.upper() == "KNN":
            mask = self.knn.apply(frame_gray_blur, learningRate=lr)
        else:
            raise ValueError("method must be 'MOG2' or 'KNN'")
        return mask

    def apply_both(self, frame_gray_blur: np.ndarray, learning_rate: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute both masks for comparison."""
        lr = float(learning_rate)
        return (
            self.mog2.apply(frame_gray_blur, learningRate=lr),
            self.knn.apply(frame_gray_blur, learningRate=lr),
        )

    def reset(self) -> None:
        """Re-create subtractors (reset background model)."""
        self.__init__(self.cfg)

