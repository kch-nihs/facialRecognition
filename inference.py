#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template_string, Response
import threading
import time
import queue
import argparse


class FrameProcessor:
    def __init__(self, model_path='best.pt', conf_threshold=0.25):
        """Initialize YOLO model"""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to prevent delay
        self.result_queue = queue.Queue(maxsize=2)
        self.processing = False
        self.latest_frame = None
        self.latest_result = None

    def process_frames(self):
        """Background thread for processing frames"""
        while self.processing:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()

                    # Process with YOLO
                    results = self.model.predict(
                        source=frame,
                        imgsz=640,
                        conf=self.conf_threshold,
                        verbose=False,
                        device='0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
                    )

                    # Store latest result
                    self.latest_result = results[0]

                    # Clear old results
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            break

                    self.result_queue.put(results[0])

                time.sleep(0.001)  # Small delay to prevent CPU overload

            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.01)

    def start_processing(self):
        """Start background processing thread"""
        self.processing = True
        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

    def stop_processing(self):
        """Stop background processing"""
        self.processing = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()


class LowLatencyYOLOWebcam:
    def __init__(self, model_path='best.pt', conf_threshold=0.25):
        """Initialize with optimized settings"""
        self.processor = FrameProcessor(model_path, conf_threshold)
        self.cap = None
        self.is_running = False

    def open_camera(self, camera_id=0):
        """Open camera with ultra-low latency settings"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {camera_id}")

        # Ultra-low latency settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer

        # Try to use MJPEG for better performance
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        print(f"✓ Camera {camera_id} opened with low-latency settings")

    def run_web_server(self, host='0.0.0.0', port=5000):
        """Run optimized web server"""
        app = Flask(__name__)

        def generate_frames():
            self.open_camera()
            self.processor.start_processing()

            frame_count = 0
            last_detection_frame = None

            try:
                while True:
                    # Clear camera buffer to get latest frame
                    for _ in range(2):  # Clear multiple frames
                        ret, frame = self.cap.read()
                        if not ret:
                            break

                    if not ret:
                        break

                    frame_count += 1

                    # Send frame for processing every 3rd frame
                    if frame_count % 3 == 0:
                        if not self.processor.frame_queue.full():
                            self.processor.frame_queue.put(frame.copy())

                    # Get latest detection result
                    current_frame = frame.copy()
                    try:
                        if not self.processor.result_queue.empty():
                            result = self.processor.result_queue.get_nowait()
                            last_detection_frame = result.plot()
                    except queue.Empty:
                        pass

                    # Use detection frame if available, otherwise raw frame
                    if last_detection_frame is not None:
                        display_frame = last_detection_frame

                        # Add info overlay
                        info_text = f"Conf: {self.processor.conf_threshold:.2f}"
                        if self.processor.latest_result and self.processor.latest_result.boxes is not None:
                            info_text += f" | Objects: {len(self.processor.latest_result.boxes)}"

                        cv2.putText(display_frame, info_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        display_frame = current_frame
                        cv2.putText(display_frame, "Processing...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Encode with optimized settings
                    ret, buffer = cv2.imencode('.jpg', display_frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, 60,
                                                cv2.IMWRITE_JPEG_OPTIMIZE, 1])

                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"Stream error: {e}")
            finally:
                self.processor.stop_processing()
                self.cleanup()

        @app.route('/')
        def index():
            return render_template_string(MINIMAL_TEMPLATE)

        @app.route('/video_feed')
        def video_feed():
            return Response(generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/set_confidence/<float:conf>')
        def set_confidence(conf):
            self.processor.conf_threshold = max(0.0, min(1.0, conf))
            return f"Confidence set to {self.processor.conf_threshold}"

        print(f"✓ Ultra-low latency server starting on http://{host}:{port}")

        try:
            app.run(host=host, port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n✓ Server stopped by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        self.processor.stop_processing()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup completed")


# Minimal HTML template for maximum performance
MINIMAL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            margin: 0;
            padding: 10px;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: Arial, sans-serif;
        }
        .container {
            text-align: center;
        }
        .video-stream {
            max-width: 100%;
            max-height: 90vh;
            border: 2px solid #333;
            border-radius: 8px;
        }
        .title {
            color: #fff;
            margin-bottom: 10px;
            font-size: 24px;
        }
        .status {
            color: #0f0;
            margin-top: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">YOLO Facial Express Detection</h1>
        <img src="{{ url_for('video_feed') }}" class="video-stream" alt="Live Stream">
    </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='Ultra-Low Latency YOLO Detection')
    parser.add_argument('--model', default='best.pt', help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')

    args = parser.parse_args()

    try:
        yolo_cam = LowLatencyYOLOWebcam(model_path=args.model, conf_threshold=args.conf)
        yolo_cam.run_web_server(host=args.host, port=args.port)
    except Exception as e:
        print(f"Failed to start: {e}")


if __name__ == "__main__":
    main()