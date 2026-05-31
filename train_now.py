import os
import torch
from ultralytics import YOLO

def main():
    # 1. Double check GPU availability 
    if torch.cuda.is_available():
        device_target = 0
        print(f"🚀 GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        device_target = "cpu"
        print("⚠️ Warning: Running on CPU.")

    # 2. Load the base architecture skeleton
    model = YOLO("yolo11n.pt")

    # 3. Fire the lightning-fast training matrix
    print("🏋️ Starting Accelerated Hackathon Training Pass...")
    model.train(
        data="data.yaml",       
        epochs=25,              # 25 epochs is the perfect sweet spot for a sharp hackathon demo model
        imgsz=416,              # 416px cuts the math workload in half compared to 640px
        batch=16,               # Perfect for the 4GB VRAM on your RTX 3050
        device=device_target,   
        workers=0,              # CRITICAL: Forces Windows to handle files on the main thread (No lag!)
        cache=False,            # Prevents Windows from freezing memory blocks
        val=False               # 🌟 FIXED: This is the correct way to turn off midway scanning logs!
    )
    print("🎉 Done! Grab your weights at: C:/braille_project/runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
