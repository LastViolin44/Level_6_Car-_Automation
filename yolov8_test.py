import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO
import math

# Set CUDA_LAUNCH_BLOCKING for debugging CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Custom dataset class for BDD100K images
class BDD10KDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls = int(parts[0])
                    x1, y1, x2, y2 = map(float, parts[1:])
                    boxes.append([cls, x1, y1, x2, y2])
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(boxes, dtype=torch.float32)

# Custom collate function to handle varying sizes of bounding boxes
def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    # Find the maximum number of bounding boxes in any image
    max_num_boxes = max([len(t) for t in targets])

    # Pad all bounding box arrays to the same length
    padded_targets = []
    for t in targets:
        num_boxes = len(t)
        padded = torch.cat([t, torch.zeros((max_num_boxes - num_boxes, 5))], dim=0)
        padded_targets.append(padded)
    
    padded_targets = torch.stack(padded_targets, dim=0)
    return images, padded_targets

# Function to validate the model
def validate_model(model, val_loader):
    model.eval()
    metrics = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.cuda()
            labels = labels.cuda()
            results = model(images)
            # Append results and labels for further metric computation
            metrics.append((results, labels))
    return metrics

# Main function
def main():
    # Paths
    train_image_dir = 'path of your training images'
    val_image_dir = 'path of your validation images'
    train_label_dir = 'path of your training data s labels'
    val_label_dir = 'path of your validation data s labels'
    yaml_path = 'path of your YAML file'
    
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8s.pt')
    print("Model loaded.")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    # Prepare datasets and dataloaders
    print("Preparing datasets and dataloaders...")
    train_dataset = BDD10KDataset(train_image_dir, train_label_dir, transform=transform)
    val_dataset = BDD10KDataset(val_image_dir, val_label_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    print("Datasets and dataloaders prepared.")

    # Check a sample from the dataset to ensure labels are correct
    sample_image, sample_labels = train_dataset[0]
    print(f"Sample image size: {sample_image.size()}")
    print(f"Sample labels: {sample_labels}")

    # Train the model
    print("Starting training...")
    model.train(data=yaml_path, epochs=10, batch=16, imgsz=640, device='cuda')  # Ensure CUDA is used
    print("Training completed.")

    # Validate the model
    print("Validating the model...")
    validate_model(model, val_loader)
    print("Validation completed.")

    # Save the trained model
    model.save('yolov8_custom_trained.pt')

if __name__ == '__main__':
    main()
