import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Index {i} — WORKING ✅ (shape: {frame.shape})")
        else:
            print(f"Index {i} — opens but no frame ⚠️")
        cap.release()
    else:
        print(f"Index {i} — not available ❌")

print("Done. Use the WORKING index in your sidebar.")