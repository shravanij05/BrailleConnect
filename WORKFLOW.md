# BrailleConnect Project Workflow

This document describes the complete development, training, evaluation, and deployment workflow of BrailleConnect.

---

# Project Overview

BrailleConnect is a real-time Braille recognition system that converts Braille characters captured through a webcam into readable text and speech.

The system combines:

- YOLOv8 for Braille character detection
- SVM for prediction verification
- OpenCV for image processing
- Streamlit for the user interface
- Pyttsx3 for speech generation

---

# Complete Development Workflow

```text
Dataset Collection
        ↓
Dataset Preparation
        ↓
YOLOv8 Training
        ↓
Model Evaluation
        ↓
SVM Training
        ↓
Real-Time Detection Pipeline
        ↓
Braille Translation
        ↓
Text-To-Speech Output
```

---

# Dataset Information

Dataset Source:

https://universe.roboflow.com/braille-eitpg/braille-detection-f0rb5

Dataset Format:

- YOLO Object Detection Format

Dataset Statistics:

- Total Images: 8,054
- Training Images: 5,610 (69.7%)
- Validation Images: 1,217 (15.1%)
- Testing Images: 1,227 (15.2%)

Dataset Usage:

### Training Set

The training dataset is used for:

- YOLOv8 model training
- Feature extraction
- SVM training

This dataset teaches the models how Braille characters appear under different conditions.

### Validation Set

The validation dataset is used during model development to:

- Monitor training progress
- Evaluate model generalization
- Calculate performance metrics

### Testing Set

The testing dataset contains unseen images and is used to verify that the trained model performs correctly on new Braille samples.

---

# Step 1: Dataset Preparation

After downloading the dataset:

1. Extract the dataset.
2. Verify image and label folders.
3. Configure paths inside:

```text
training/data.yaml
```

---

# Step 2: YOLOv8 Training

YOLOv8 is responsible for Braille character detection.

Training Script:

```bash
python training/train_yolo.py
```

Training Configuration:

- Model: YOLOv8 Nano
- Epochs: 30
- Image Size: 416×416
- Batch Size: 16

Training Process:

1. Load dataset configuration.
2. Initialize YOLOv8 model.
3. Train using Braille images.
4. Learn character localization patterns.
5. Save trained weights.

Generated Files:

```text
weights/best.pt
weights/last.pt
```

---

# Step 3: Model Evaluation

Evaluate the trained model:

```bash
python accuracy.py
```

Metrics Generated:

- Precision
- Recall
- F1 Score
- mAP50

Evaluation Resources:

```text
model_metrics/
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxF1_curve.png
├── BoxP_curve.png
├── BoxPR_curve.png
├── BoxR_curve.png
├── training_sample.jpeg
└── prediction_sample.jpeg
```

---

# Step 4: SVM Training

After YOLO training, detected Braille character regions are used to train an SVM verification model.

Training Script:

```bash
python training/train_svm.py
```

Workflow:

1. Load trained YOLO model.
2. Detect Braille cells from training images.
3. Crop detected character regions.
4. Convert crops to grayscale.
5. Resize to 32×32 pixels.
6. Extract pixel features.
7. Train SVM using an RBF kernel.
8. Save trained model.

Generated File:

```text
svm_braille.pkl
```

Purpose:

- Reduce false positives.
- Improve prediction confidence.
- Verify YOLO detections.

---

# Step 5: Camera Configuration

BrailleConnect supports:

- Laptop Webcam
- USB Camera
- DroidCam

To identify available cameras:

```bash
python find_camera.py
```

The detected index is selected in the dashboard.

---

# Step 6: Running BrailleConnect

Launch the application:

```bash
streamlit run dashboard.py
```

The Streamlit dashboard provides:

- Live camera feed
- Detection visualization
- Real-time translation
- Speech generation controls

---

# Step 7: Real-Time Detection Pipeline

```text
Camera Input
      ↓
Frame Capture
      ↓
Image Enhancement
      ↓
YOLOv8 Detection
      ↓
Non-Maximum Suppression
      ↓
Character Clustering
      ↓
SVM Verification
      ↓
Temporal Stabilization
      ↓
Majority Voting
      ↓
Braille Translation
      ↓
Readable Text
      ↓
Text-To-Speech
```

---

# Detection Logic

The detection system follows these steps:

1. Capture live frame.
2. Improve image quality using preprocessing.
3. Detect Braille characters using YOLOv8.
4. Remove overlapping detections.
5. Group neighboring detections.
6. Verify predictions using SVM.
7. Collect detections across multiple frames.
8. Perform majority voting.
9. Lock stable predictions.
10. Generate final text output.

---

# Output Generation

BrailleConnect produces:

### Text Output

Detected Braille characters are translated into readable text and displayed on the dashboard.

### Speech Output

Using Pyttsx3, translated text can be converted into speech to improve accessibility.

---

# Technical Architecture

```text
Camera / DroidCam
        │
        ▼
OpenCV Preprocessing
        │
        ▼
YOLOv8 Detection
        │
        ▼
SVM Verification
        │
        ▼
Temporal Stabilization
        │
        ▼
Braille Translation
        │
 ┌──────┴──────┐
 ▼             ▼
Text Output   Speech Output
        │
        ▼
Streamlit Dashboard
```

---

# Generated Assets

Training produces:

```text
weights/best.pt
weights/last.pt
svm_braille.pkl
```

Evaluation produces:

```text
model_metrics/*
```

Deployment uses:

```text
dashboard.py
find_camera.py
accuracy.py
```

---

# Conclusion

BrailleConnect uses a hybrid YOLOv8 + SVM architecture to provide reliable real-time Braille recognition. The workflow covers dataset preparation, model training, evaluation, verification, deployment, translation, and speech generation to create an accessible Braille reading solution.