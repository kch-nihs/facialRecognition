# YOLO Facial Expression Detection
A real-time facial expression detection system using YOLOv8 and a custom-trained dataset. The project provides a web interface for live webcam detection.

## The Algorithm
This project uses the YOLOv8 object detection algorithm, fine-tuned on a facial expression dataset. The training is performed using the Ultralytics YOLO Python package. The inference pipeline captures webcam frames, processes them with the trained YOLO model, and streams annotated results via a Flask web server.

**Key Features:**
- Real-time detection with webcam capture.
- Flask-based web server for live video streaming.
- Easy model training and inference using Ultralytics YOLO.

**Dependencies:**
- Python 3.8+
- OpenCV
- Flask
- Ultralytics YOLO
