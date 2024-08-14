import os
import torch
from ultralytics import YOLO
from PIL import Image


image_dir = 'path of images you want to label'
label_dir = 'output path to your created labels'
model_path = 'yolov8s.pt'


classes = ['traffic light', 'traffic sign', 'person', 'car', 'bus', 'truck', 'rider', 'train', 'motorcycle', 'bicycle']
num_classes = len(classes)


def download_model(url, path):
    import requests
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded successfully and saved to {path}.")
    else:
        print(f"Failed to download model from {url}. Status code: {response.status_code}")


if not os.path.exists(model_path):
    print("Downloading YOLOv8 model...")
    download_model('https://github.com/ultralytics/assets/releases/download/v0.0/yolov8s.pt', model_path)
else:
    print("Model file already exists.")


print("Loading YOLOv8 model...")
model = YOLO(model_path)


os.makedirs(label_dir, exist_ok=True)


def save_yolo_labels(image_path, results):
    base_name = os.path.basename(image_path)
    label_path = os.path.join(label_dir, os.path.splitext(base_name)[0] + '.txt')
    with open(label_path, 'w') as f:
        for result in results:
            cls = int(result.cls)
            conf = result.conf
            x1, y1, x2, y2 = result.xyxy[0].tolist()  # Ensure we get the coordinates as a list of values

            img = Image.open(image_path)
            width, height = img.size
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            if cls >= num_classes:
                print(f"Skipping invalid class {cls} for image {image_path}")
                continue
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


print("Labeling images...")
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        results = model(image_path)[0].boxes

        save_yolo_labels(image_path, results)
        print(f"Processed {filename}")

print("Labeling completed.")
