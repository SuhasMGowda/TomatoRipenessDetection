import os
from ultralytics import YOLO
import torch

def setup_and_train():
# Set memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check if GPU is available
    if torch.cuda.is_available():
        device = '0'  # Use the first GPU
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("GPU not detected. Running on CPU.")

# Ensure you have the necessary folders
    os.makedirs("train", exist_ok=True)
    os.makedirs("val", exist_ok=True)

# Define and save data configuration
    data_config = """
names:
  - b_fully_ripened
  - b_half_ripened
  - b_green
  - l_fully_ripened
  - l_half_ripened
  - l_green
nc: 6
path: C:/Users/ACER/Desktop/Suhas/College/Main Project/tomato_dataset
train: C:/Users/ACER/Desktop/Suhas/College/Main Project/tomato_dataset/train/images
val: C:/Users/ACER/Desktop/Suhas/College/Main Project/tomato_dataset/val/images
"""

# Write the dataset configuration to a YAML file
    with open("dataset.yaml", "w") as file:
        file.write(data_config)
    print("Dataset configuration ready.")

# Clear GPU memory
    torch.cuda.empty_cache()

# Load the YOLO model
    model = YOLO("yolov8n.pt")

# Train the model with optimized parameters
    results = model.train(
        data="dataset.yaml", 
        epochs=100, 
        imgsz=512,  # Reduced image size
        batch=4,    # Reduced batch size
        device=device, 
        amp=True    # Enable mixed precision training
    )

# Save the best model weights
    print("Training complete. Best model saved at:", results.best_model_path)

if __name__ == '__main__':
    setup_and_train()
