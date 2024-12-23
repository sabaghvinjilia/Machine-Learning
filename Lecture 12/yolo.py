from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8s.pt")

# Perform prediction on the webcam feed
results = model.predict(source=0, show=True)

# Print results
print(results)
