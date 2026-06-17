from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """App-wide configuration.

    All numeric parameters are adjustable via GUI sliders.
    """

    # Preprocessing
    resize_width: int = 640
    gaussian_kernel: tuple[int, int] = (5, 5)

    # Thresholding (optional)
    threshold: int = 25
    thresholding_enabled: bool = False

    # Morphology
    morph_kernel_size: int = 3
    morph_iterations: int = 1

    # Detection filtering
    min_area: int = 500

    # Background subtraction - MOG2 / KNN shared parameters
    learningRate: float = 0.001
    history: int = 500
    varThreshold: float = 16.0
    detectShadows: bool = False

    # Runtime / output
    save_outputs: bool = True
    outputs_dir: str = "outputs"

    # For ROI
    roi_enabled: bool = True


