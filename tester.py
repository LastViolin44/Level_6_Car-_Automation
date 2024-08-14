import io  
import cv2
import os
import ipywidgets as widgets
from IPython.display import display
from ultralytics import YOLO
from PIL import Image
import numpy as np


model = YOLO('path of your model')


upload_widget = widgets.FileUpload(accept='.jpg,.jpeg,.png,.mp4', multiple=False)
display(upload_widget)


def process_upload(change):
    for name, file_info in upload_widget.value.items():
        content = file_info['content']

        
        if name.endswith(('.jpg', '.jpeg', '.png')):
            # Process the image
            image = Image.open(io.BytesIO(content)).convert("RGB")
            results = model(image)
            result_img = results[0].plot()

            # Convert numpy array to PIL Image
            result_img_pil = Image.fromarray(result_img)

            # Save the processed image to Google Drive
            save_image_path = os.path.join('path of your model', f'processed_{name}')
            result_img_pil.save(save_image_path)

            # Display the image
            display(result_img_pil)
            display(f"Processed image saved to: {save_image_path}")

        elif name.endswith('.mp4'):
            # Process the video
            video_path = 'your test video path'
            with open(video_path, 'wb') as f:
                f.write(content)

            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            save_video_path = os.path.join('path your model', 'result_video6.mp4')
            out = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated_frame = results[0].plot()

                if out is None:
                    out = cv2.VideoWriter(save_video_path, fourcc, 20.0, (annotated_frame.shape[1], annotated_frame.shape[0]))

                out.write(annotated_frame)

            cap.release()
            out.release()
            display(f"Processed video saved to: {save_video_path}")

        else:
            display("Unsupported file type!")


upload_widget.observe(process_upload, names='value')