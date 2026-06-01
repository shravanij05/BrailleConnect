import os
from ultralytics import YOLO

def main():
    weights_path = "weights/best.pt"
    if not os.path.exists(weights_path):
        print(f"❌ Error: Missing '{weights_path}'")
        return

    # Load model and silence standard print logs
    model = YOLO(weights_path)
    
    print("📊 Calculating metrics profile... Please wait a moment.")
    metrics = model.val(data="data.yaml", imgsz=416, split="val", workers=0, verbose=False)
    
    # Extract the core metrics from the results array
    mp = metrics.box.mp * 100        # Mean Precision
    mr = metrics.box.mr * 100        # Mean Recall
    map50 = metrics.box.map50 * 100  # mAP @ 50% (Overall Accuracy)
    f1 = 2 * (mp * mr) / (mp + mr)   # Calculated F1-Score Balance

    print("\n" + "═"*45)
    print(" 🚀 FINAL BRAILLE MODEL METRICS PROFILE")
    print("═"*45)
    print(f" 🎯 Accuracy (mAP50):  {map50:.2f}%")
    print(f" 🔍 Precision:         {mp:.2f}%")
    print(f" 🔄 Recall:            {mr:.2f}%")
    print(f" ⚖️  F1-Score:          {f1:.2f}%")
    print("═"*45)

if __name__ == "__main__":
    main()
