from ultralytics import YOLO

def main():
    # Path to your dataset yaml file
    data_yaml = 'C:/Users/Student/Desktop/Facial_express/Facial Expressions/data.yaml'  # Change this path as needed

    # Load pre-trained YOLOv8n model (nano version - fast, small)
    model = YOLO('yolov8n.pt')

    # Train model on your dataset
    model.train(
        data=data_yaml,
        epochs=40,        # You can reduce epochs to 20 if time is short
        imgsz=640,        # Image size for training
        batch=16,         # Batch size, reduce if GPU memory is limited
        device=0          # Use GPU 0; use 'cpu' if no GPU available
    )

    # After training, save the model weights automatically in 'runs/train/exp/weights'
    print("Training complete!")

if __name__ == '__main__':
    main()
