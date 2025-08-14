import gradio as gr
import torch
from ultralytics import YOLO
import cv2

# Load TorchScript model
MODEL_PATH = "runs/detect/yolov8n-custom3/weights/best.torchscript"  # Make sure this matches your exported file name
model = YOLO(MODEL_PATH)

def detect_objects(image):
    # Run YOLOv8 inference
    results = model(image)[0]  # first batch
    annotated_frame = results.plot()  # draw boxes on image
    return annotated_frame

# Gradio UI
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload or capture image"),
    outputs=gr.Image(type="numpy", label="Detection result"),
    title="YOLOv8 Object Detection",
    description="Upload or take a photo to run YOLOv8 detection using a custom TorchScript model."
)

if __name__ == "__main__":
    demo.launch()
