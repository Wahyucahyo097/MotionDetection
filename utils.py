from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


@dataclass
class FPSCounter:
    """Simple FPS counter based on processing loop timestamps."""

    prev_t: float | None = None

    def update(self) -> float:
        """Update and return current FPS."""
        now = time.perf_counter()
        if self.prev_t is None:
            self.prev_t = now
            return 0.0
        dt = now - self.prev_t
        self.prev_t = now
        if dt <= 0:
            return 0.0
        return 1.0 / dt


def put_text_multiline(
    frame: np.ndarray, lines: Iterable[str], origin: tuple[int, int] = (10, 20)
) -> np.ndarray:
    """Draw multiline text on frame."""
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            str(line),
            (x, y + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def save_screenshot(frame_bgr: np.ndarray, out_path: str) -> None:
    """Save a BGR frame as PNG."""
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, frame_bgr)


class VideoRecorder:
    """Record BGR frames into an MP4 using cv2.VideoWriter."""

    def __init__(
        self,
        out_path: str,
        fps: float = 20.0,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        self.out_path = out_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer: cv2.VideoWriter | None = None
        self.started = False

    def start(self, frame: np.ndarray) -> None:
        if self.started:
            return
        h, w = frame.shape[:2]
        size = self.frame_size or (w, h)
        ensure_dir(os.path.dirname(self.out_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.out_path, fourcc, float(self.fps), size)
        self.started = True

    def write(self, frame: np.ndarray) -> None:
        if not self.started:
            self.start(frame)
        assert self.writer is not None
        if self.writer is None:
            return
        self.writer.write(frame)

    def stop(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.started = False

