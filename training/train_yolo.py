import torch
from ultralytics import YOLO

def main():
    device_target = 0 if torch.cuda.is_available() else "cpu"
    print(f"🛰️ Compute Target Locked to: {device_target}")
    model = YOLO("yolov8n.pt") 

    model.train(
        data="data.yaml",
        epochs=30,          
        imgsz=416,          
        batch=16,           
        device=device_target,
        workers=0,          
        cache=False,
        val=False         
    )
    print("🔥 YOLOv8 Training Complete! Weights saved.")

if __name__ == "__main__":
    main()
