# Human-Behavior-Recognition

## description
- Real-time video streaming for automated human subtle behavior recognition. The framework uses an open source library to track the facial feature points and the rigid body face transformations on each frame. Unlike activity recognition tasks, where the algorithm can only globally recognize  a certain activity (i.e soccer, basketball, running, etc),  the proposed behavioral recognition framework buffers local spatio-temporal vectors to detect and track subtle movements to perceive human behavior. 


## code
- All source code is in `src`.
- Human behavior recognition pipeline code resides at the `behavioral_detector.py` file.
- Human head-pose estimator can be found at the `headpose.py` file.
- Utilities including image processing (filtering, resizing, denoising, angle-measurements, etc) can be found at `utils.py` file.


## documentation
- Code is the documentation of itself.

## usage
- Use `./runDetector.sh` to generate a confusion matrix.
- A summary of the pipeline is given in `report.pdf`.

## demonstration
The pipeline is demonstrated below.

- Human-Behavior-Detector.

 ![](./video/behavior-detector.gif)
