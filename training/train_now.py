import os
import torch
from ultralytics import YOLO

def main():
    if torch.cuda.is_available():
        device_target = 0
        print(f"🚀 GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        device_target = "cpu"
        print("⚠️ Warning: Running on CPU.")

    model = YOLO("yolo11n.pt")

    print("🏋️ Starting Accelerated Hackathon Training Pass...")
    model.train(
        data="data.yaml",       
        epochs=25,              
        imgsz=416,             
        batch=16,               
        device=device_target,   
        workers=0,             
        cache=False,            
        val=False               
    )
    print("🎉 Done! Grab your weights at: C:/braille_project/runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
