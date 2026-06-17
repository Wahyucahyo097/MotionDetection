# TODO - MotionDetection App

- [ ] Create initial project structure (python modules + outputs folder via runtime)
- [ ] Implement config.py (dataclasses for parameters)
- [ ] Implement utils.py (fps, safe folder creation, screenshot/video writing)
- [ ] Implement preprocessing.py (resize, grayscale, Gaussian blur, optional thresholding)
- [ ] Implement background.py (MOG2 + KNN background subtraction with shared API)
- [ ] Implement roi.py (mouse-drawn ROI polygon/rectangle, point-in-ROI mask)
- [ ] Implement detector.py (contours, boundingRect, centroid, filtering by min_area, metrics)
- [ ] Implement gui.py (Tkinter modern UI: open video/webcam, start/stop, method selection, sliders, parameter live update, ROI drawing)
- [ ] Implement main.py (wiring: instantiate GUI, background methods, preprocessing, detector, and rendering loop)
- [ ] Add requirements.txt (opencv-python, pillow, numpy, typing-extensions)
- [ ] Add README.md (install, run, explanation of algorithms, structure, ROI usage, outputs)
- [ ] Smoke test via running python main.py (if GUI can't be tested here, ensure no syntax errors)


