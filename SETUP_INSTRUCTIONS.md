# BrailleConnect Setup Instructions

## Prerequisites

- Python 3.10+
- Webcam or DroidCam
- Internet connection (for first-time dependency installation)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shravanij05/BrailleConnect.git
cd BrailleConnect
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Option 1: Using Laptop Webcam

Run the application:

```bash
streamlit run dashboard.py
```

In the dashboard:

1. Set Camera Index to `0`
2. Click Start Camera

Most systems use index `0` for the default webcam.

---

## Option 2: Using DroidCam (Recommended)

Using a mobile phone camera generally provides:

- Better image quality
- Better focus
- Higher Braille dot visibility
- Improved detection performance

### Step 1: Install DroidCam

**Mobile App:**
- [DroidCam on Google Play](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)

**Desktop Client:**
- [DroidCam Official Website](https://www.dev47apps.com/)

---

### Step 2: Connect Devices

1. Connect both laptop and phone to the same Wi-Fi network.

   If Wi-Fi is unavailable, enable a mobile hotspot and connect both devices to the hotspot.

2. Open DroidCam on the phone.

3. Note the IP address shown in the app.

4. Open DroidCam Client on the laptop.

5. Select "Connect over WiFi".

6. Enter the IP address displayed on the phone.

7. Click Start.

---

### Step 3: Identify Camera Index

Run:

```bash
python find_camera.py
```

Example output:

```text
Index 0 — WORKING ✅
Index 1 — not available ❌
Index 2 — WORKING ✅
Index 3 — not available ❌
Index 4 — not available ❌
Index 5 — not available ❌
```

Any index marked **WORKING ✅** represents an available camera.

After connecting DroidCam, run `find_camera.py` and note the newly available camera index.

For example:

- Laptop Webcam → Index 0
- DroidCam → Index 2

In the dashboard sidebar, set **Camera Index** to the corresponding WORKING index and click **Start Camera**.

> Note: Camera indices may vary between systems. Always run `find_camera.py` after connecting DroidCam to determine the correct index.
### Why Use DroidCam?

Braille detection works best when the Braille dots are clearly visible and in focus.

Compared to many laptop webcams, smartphone cameras generally provide:

- Better image quality
- Better autofocus
- Higher resolution
- Improved visibility of Braille dots

For best results, DroidCam is recommended during testing and demonstrations.

### Step 4: Run BrailleConnect

```bash
streamlit run dashboard.py
```

In the dashboard:

1. Set Camera Index to the DroidCam index obtained from `find_camera.py`
2. Click Start Camera

---

## Model Evaluation

Run:

```bash
python accuracy.py
```

This calculates:

- Accuracy (mAP50)
- Precision
- Recall
- F1 Score

---

## Troubleshooting

### Camera Not Opening

Run:

```bash
python find_camera.py
```

and verify the correct camera index.

### No Detections

- Increase lighting
- Hold camera directly above the Braille sheet
- Keep paper flat
- Use DroidCam for higher image quality

### Missing Dependencies

Reinstall:

```bash
pip install -r requirements.txt
```
