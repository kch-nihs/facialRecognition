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

[View a video explanation here](https://youtu.be/Mef8XdRZyS4)

## Running this project

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/kch-nihs/facialRecognition.git
    cd facialRecognition

2. **Install YOLO on Jetson:**
   Install Torch and Torchvision
   ```bash
   pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
   pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
   ```

   Install cuSPARSELt
   ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install libcusparselt0 libcusparselt-dev


3. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python flask torch numpy
4. **Run inference.py file:**
    ```bash
    python inference.py

5. **Access to webstie**  
    http://\<YOUR-IP-ADDRESS\>:5000
