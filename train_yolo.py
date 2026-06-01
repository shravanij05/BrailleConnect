import torch
from ultralytics import YOLO

def main():
    # Force GPU acceleration to process features at high speed
    device_target = 0 if torch.cuda.is_available() else "cpu"
    print(f"🛰️ Compute Target Locked to: {device_target}")

    # Load the base YOLOv8 nano skeleton architecture
    model = YOLO("yolov8n.pt") 

    # Launch the training loop on your new Roboflow dataset
    model.train(
        data="data.yaml",
        epochs=30,          # 30 passes maps the features deeply without overfitting
        imgsz=416,          # 416px resolution optimizes processing speed on laptop hardware
        batch=16,           # Ideal memory batch allocation size for laptop VRAM
        device=device_target,
        workers=0,          # CRITICAL: Prevents Windows multi-threading file lag
        cache=False,
        val=False           # Skips midway validations to accelerate the core training run
    )
    print("🔥 YOLOv8 Training Complete! Weights saved.")

if __name__ == "__main__":
    main()
