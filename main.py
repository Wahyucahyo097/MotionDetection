from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from background import BackgroundSubtractor
from config import Config
from detector import MotionDetector
from gui import MotionAppGUI
from preprocessing import Preprocessor
from roi import ROIManager
from utils import FPSCounter, VideoRecorder


@dataclass
class CaptureState:
    running: bool = False
    source: str | None = None
    cap: cv2.VideoCapture | None = None
    recorder: Optional[VideoRecorder] = None
    last_frame_bgr: Optional[np.ndarray] = None
    last_annotated_bgr: Optional[np.ndarray] = None


class MotionApp:
    def __init__(self) -> None:
        self.cfg = Config()
        self.state = CaptureState()

        self.preprocessor = Preprocessor(self.cfg)
        self.bg = BackgroundSubtractor(self.cfg)
        self.detector = MotionDetector(self.cfg)
        self.roi = ROIManager()
        self.fps_counter = FPSCounter()

        self.gui_root = __import__("tkinter").Tk()
        self.gui = MotionAppGUI(self.gui_root)

        self.gui.on_open_video = self._on_open_video
        self.gui.on_start = self.start
        self.gui.on_stop = self.stop
        self.gui.on_save_screenshot = self.save_screenshot

        self.gui.bind_roi_mouse(
            self._roi_drag_start,
            self._roi_drag_move,
            self._roi_drag_end,
        )

        self.gui_root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------- ROI mouse mapping --------
    def _roi_drag_start(self, x: int, y: int) -> None:
        if self.state.cap is None:
            return
        # Canvas coordinates need mapping to frame coordinates; backend will handle using ratio.
        self._roi_canvas_start = (x, y)
        self.roi.state.drawing = True
        self.roi.set_from_drag_start(x, y)

    def _roi_drag_move(self, x: int, y: int) -> None:
        if not self.roi.state.drawing:
            return
        self.roi.set_from_drag_move(x, y)

    def _roi_drag_end(self, x: int, y: int) -> None:
        self.roi.set_from_drag_end(x, y)

    # -------- GUI callbacks --------
    def _on_open_video(self, path_token: str) -> None:
        # Stop existing
        if self.state.running:
            self.stop()

        if path_token == "__WEBCAM__":
            self.state.source = "__WEBCAM__"
            self.state.cap = cv2.VideoCapture(0)
        else:
            self.state.source = path_token
            self.state.cap = cv2.VideoCapture(path_token)

        if not self.state.cap or not self.state.cap.isOpened():
            self.gui.set_status("Failed to open capture source.")
            self.state.cap = None
            return

        self.gui.set_status("Source loaded. Set parameters and press Start Detection.")

    def start(self) -> None:
        if self.state.cap is None or not self.state.cap.isOpened():
            self.gui.set_status("Open Video/Webcam first.")
            return

        if self.state.running:
            return

        # Reset background models so params history changes take effect.
        self.bg = BackgroundSubtractor(self.cfg)
        self.fps_counter = FPSCounter()

        self.state.running = True
        self.state.recorder = None

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.gui.set_status("Detection running...")

    def stop(self) -> None:
        self.state.running = False
        if self.state.recorder is not None:
            self.state.recorder.stop()
            self.state.recorder = None
        if self.state.cap is not None:
            # Don't release for GUI responsiveness; but safe to release.
            pass
        self.gui.set_status("Detection stopped.")

    def _on_close(self) -> None:
        self.stop()
        try:
            if self.state.cap is not None:
                self.state.cap.release()
        except Exception:
            pass
        self.gui_root.destroy()

    # -------- Output features --------
    def save_screenshot(self) -> None:
        if self.state.last_annotated_bgr is None:
            return
        import os
        from utils import ensure_dir

        ts = time.strftime("%Y%m%d-%H%M%S")
        ensure_dir(self.cfg.outputs_dir)
        out_path = os.path.join(self.cfg.outputs_dir, f"screenshot-{ts}.png")
        cv2.imwrite(out_path, self.state.last_annotated_bgr)
        self.gui.set_status(f"Screenshot saved: {out_path}")

    # -------- Main processing loop --------
    def _sync_cfg_from_gui(self) -> None:
        self.cfg.threshold = self.gui.get_threshold()
        self.cfg.min_area = self.gui.get_min_area()
        self.cfg.learningRate = self.gui.get_learning_rate()
        self.cfg.history = self.gui.get_history()
        self.cfg.varThreshold = self.gui.get_var_threshold()

    def _loop(self) -> None:
        # Delay start recorder creation until we have a first frame.
        method = "MOG2"

        while self.state.running and self.state.cap is not None:
            ret, frame_bgr = self.state.cap.read()
            if not ret:
                self.gui.set_status("End of stream or read error.")
                break

            self._sync_cfg_from_gui()
            method = self.gui.get_selected_method()

            # Preprocess
            processed, gray_for_display = self.preprocessor.process(frame_bgr)
            # processed: either thresholded or blur; but background subtraction expects grayscale-like.

            # Background subtraction
            if self.cfg.roi_enabled and self.roi.roi_rect is not None:
                # ROI mask applied before subtraction
                gray_in = self.roi.apply_mask(gray_for_display)
            else:
                gray_in = gray_for_display

            if self.cfg.save_outputs:
                # recorder init on first frame
                if self.state.recorder is None:
                    from utils import VideoRecorder

                    ts = time.strftime("%Y%m%d-%H%M%S")
                    import os

                    out_path = os.path.join(self.cfg.outputs_dir, f"detection-{ts}.mp4")
                    self.state.recorder = VideoRecorder(out_path=out_path, fps=self.state.cap.get(cv2.CAP_PROP_FPS) or 20.0)

            lr = float(self.cfg.learningRate)

            if method.upper() == "MOG2":
                fg = self.bg.apply("MOG2", processed, lr)
            else:
                fg = self.bg.apply("KNN", processed, lr)

            # Also compute both for comparison panels
            fg_mog2, fg_knn = self.bg.apply_both(processed, lr)

            # Apply ROI to masks for counting
            if self.cfg.roi_enabled and self.roi.roi_rect is not None:
                fg_mog2 = self.roi.apply_mask_to_mask(fg_mog2)
                fg_knn = self.roi.apply_mask_to_mask(fg_knn)
                fg = self.roi.apply_mask_to_mask(fg)

            # Detect
            objects, annotated = self.detector.detect_from_mask(frame_bgr.copy(), fg)

            # Metrics / status
            fps = self.fps_counter.update()
            motion_status = "Motion Detected" if len(objects) > 0 else "No Motion"

            # Overlay info
            info_lines = [
                f"Method: {method.upper()} | {motion_status}",
                f"FPS: {fps:.1f}",
                f"Objects: {len(objects)}",
            ]
            # Add up to first 4 objects info for readability
            for obj in objects[:4]:
                x, y, w, h = obj.bbox_xywh
                cx, cy = obj.centroid_xy
                info_lines.append(f"{obj.label}: area={obj.area:.0f} bbox=({x},{y},{w},{h}) centroid=({cx},{cy})")

            cv2.putText(
                annotated,
                " | ".join([info_lines[0], info_lines[1]]),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # multiline smaller
            y0 = 50
            for i, line in enumerate(info_lines[2:]):
                cv2.putText(
                    annotated,
                    line,
                    (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 220),
                    2,
                    cv2.LINE_AA,
                )

            # Draw ROI on annotated frame
            annotated = self.roi.draw_roi(annotated)

            # Save outputs
            self.state.last_frame_bgr = frame_bgr
            self.state.last_annotated_bgr = annotated
            if self.state.recorder is not None:
                self.state.recorder.write(annotated)

            # Prepare side-by-side panels: Original, Mask MOG2, Mask KNN, Result
            panels = []
            panels.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
            panels.append(Image.fromarray(fg_mog2))
            panels.append(Image.fromarray(fg_knn))
            panels.append(Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))

            # Convert masks to RGB for display
            panels[1] = panels[1].convert("RGB")
            panels[2] = panels[2].convert("RGB")

            def _render() -> None:
                self.gui.set_video_panels(panels)
                self.gui.set_status(
                    f"{motion_status} | Method={method.upper()} | Objects={len(objects)} | FPS={fps:.1f}"
                )

            # Tkinter thread safety
            self.gui_root.after(0, _render)

            # Let UI update
            time.sleep(0.001)

        # Cleanup
        if self.state.recorder is not None:
            self.state.recorder.stop()
            self.state.recorder = None


def main() -> None:
    app = MotionApp()
    app.gui_root.mainloop()


if __name__ == "__main__":
    main()

