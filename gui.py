from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from typing import Callable, Optional


class MotionAppGUI:
    """Tkinter GUI wrapper.

    Provides callbacks for:
      - open video
      - start/stop
      - method selection
      - ROI drawing area (mouse drag)

    The actual frame rendering is delegated via `set_video_panels`.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Motion Detection - MOG2 vs KNN")
        self.root.geometry("1200x720")

        self.on_open_video: Optional[Callable[[str], None]] = None
        self.on_start: Optional[Callable[[], None]] = None
        self.on_stop: Optional[Callable[[], None]] = None
        self.on_save_screenshot: Optional[Callable[[], None]] = None

        self.selected_method = tk.StringVar(value="MOG2")
        self.threshold_var = tk.IntVar(value=25)
        self.min_area_var = tk.IntVar(value=500)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        self.history_var = tk.IntVar(value=500)
        self.var_threshold_var = tk.DoubleVar(value=16.0)

        self._build_layout()

    def _build_layout(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Controls
        controls = ttk.LabelFrame(main, text="Controls")
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        controls.columnconfigure(0, weight=1)

        # Method
        ttk.Label(controls, text="Background Subtraction Method").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        rb1 = ttk.Radiobutton(controls, text="MOG2", value="MOG2", variable=self.selected_method)
        rb2 = ttk.Radiobutton(controls, text="KNN", value="KNN", variable=self.selected_method)
        rb1.grid(row=1, column=0, sticky="w")
        rb2.grid(row=2, column=0, sticky="w")

        # Sliders
        self._add_slider(controls, "Threshold", self.threshold_var, 0, 255, 25, 3)
        self._add_slider(controls, "Min Area", self.min_area_var, 50, 5000, 500, 4, integer=True)
        self._add_slider(controls, "Learning Rate", self.learning_rate_var, 0.0001, 0.5, 0.001, 5)
        self._add_slider(controls, "History", self.history_var, 10, 2000, 500, 6, integer=True)
        self._add_slider(
            controls,
            "VarThreshold",
            self.var_threshold_var,
            1.0,
            100.0,
            16.0,
            7,
        )

        # Buttons
        btn_row = ttk.Frame(controls)
        btn_row.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        ttk.Button(btn_row, text="Open Video", command=self._open_video).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(btn_row, text="Webcam", command=self._open_webcam).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )

        btn_row2 = ttk.Frame(controls)
        btn_row2.grid(row=9, column=0, sticky="ew", pady=(8, 0))
        btn_row2.columnconfigure(0, weight=1)
        btn_row2.columnconfigure(1, weight=1)

        ttk.Button(btn_row2, text="Start Detection", command=self._start).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(btn_row2, text="Stop Detection", command=self._stop).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )

        ttk.Button(controls, text="Save Screenshot", command=self._save_screenshot).grid(
            row=10, column=0, sticky="ew", pady=(10, 0)
        )

        # Panels
        panels = ttk.LabelFrame(main, text="Detection Output")
        panels.grid(row=0, column=1, sticky="nsew")
        panels.rowconfigure(0, weight=1)
        panels.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="No detection running.")
        ttk.Label(panels, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        self.video_canvas = tk.Canvas(panels, bg="#0b1220", height=480)
        self.video_canvas.grid(row=1, column=0, sticky="nsew")

    def _add_slider(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.Variable,
        min_v: float,
        max_v: float,
        init_v: float,
        row: int,
        integer: bool = False,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        value_lbl = ttk.Label(frame, textvariable=var)
        value_lbl.grid(row=0, column=1, sticky="e", padx=(10, 0))

        res = 1 if integer else 0.001
        slider = ttk.Scale(
            frame,
            from_=min_v,
            to=max_v,
            value=init_v,
            orient="horizontal",
            command=lambda _v: self._update_var_from_slider(var, slider, integer),
        )
        slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        if integer:
            var.set(int(init_v))
        else:
            var.set(float(init_v))

    def _update_var_from_slider(self, var: tk.Variable, slider: ttk.Scale, integer: bool) -> None:
        v = slider.get()
        if integer:
            var.set(int(float(v)))
        else:
            var.set(float(v))

    # -------- Callbacks bindings --------
    def _open_video(self) -> None:
        if self.on_open_video is None:
            return
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self.on_open_video(path)

    def _open_webcam(self) -> None:
        # We use special path token; the backend will interpret it.
        if self.on_open_video is None:
            return
        self.on_open_video("__WEBCAM__")

    def _start(self) -> None:
        if self.on_start is not None:
            self.on_start()

    def _stop(self) -> None:
        if self.on_stop is not None:
            self.on_stop()

    def _save_screenshot(self) -> None:
        if self.on_save_screenshot is not None:
            self.on_save_screenshot()

    # -------- Frame rendering --------
    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def bind_roi_mouse(self, on_drag_start, on_drag_move, on_drag_end) -> None:
        """Bind ROI mouse drag on canvas."""

        def _start(event):
            on_drag_start(event.x, event.y)

        def _move(event):
            on_drag_move(event.x, event.y)

        def _end(event):
            on_drag_end(event.x, event.y)

        self.video_canvas.bind("<ButtonPress-1>", _start)
        self.video_canvas.bind("<B1-Motion>", _move)
        self.video_canvas.bind("<ButtonRelease-1>", _end)

    def set_video_panels(self, pil_images) -> None:
        """Render 1-4 images on canvas using PhotoImage.

        pil_images: list of PIL.Image objects.
        """
        from PIL import ImageTk

        self.video_canvas.delete("all")

        if not pil_images:
            return

        # Layout: up to 2 columns, dynamic rows
        n = len(pil_images)
        cols = 2 if n > 1 else 1
        rows = (n + cols - 1) // cols

        canvas_w = self.video_canvas.winfo_width() or 1000
        canvas_h = self.video_canvas.winfo_height() or 480
        cell_w = canvas_w // cols
        cell_h = canvas_h // max(1, rows)

        # Keep refs to avoid GC
        self._tk_refs = []

        for i, img in enumerate(pil_images):
            r = i // cols
            c = i % cols
            resized = img.resize((cell_w, cell_h))
            tk_img = ImageTk.PhotoImage(resized)
            self._tk_refs.append(tk_img)
            self.video_canvas.create_image(
                c * cell_w,
                r * cell_h,
                anchor="nw",
                image=tk_img,
            )

    # -------- Parameter getters --------
    def get_selected_method(self) -> str:
        return self.selected_method.get()

    def get_threshold(self) -> int:
        return int(self.threshold_var.get())

    def get_min_area(self) -> int:
        return int(self.min_area_var.get())

    def get_learning_rate(self) -> float:
        return float(self.learning_rate_var.get())

    def get_history(self) -> int:
        return int(self.history_var.get())

    def get_var_threshold(self) -> float:
        return float(self.var_threshold_var.get())

