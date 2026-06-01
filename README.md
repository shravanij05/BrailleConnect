# ⠿ BrailleConnect

BrailleConnect is a real-time Braille recognition system that converts Braille characters captured through a webcam into readable text and speech. The project combines YOLOv8 object detection, SVM-based verification, OpenCV image processing, and Streamlit to provide an accessible and interactive Braille reading experience.

## Features
- Real-time Braille recognition through webcam input
- Conversion of Braille characters into readable text
- Text-to-Speech output for detected content
- Fast and interactive web-based interface
- Braille character detection using Computer Vision
- Accessible solution for improving Braille readability
- Different camera-based input supported

---

## Tech Stack

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- OpenCV
- Scikit-learn
- NumPy
- Joblib
- Pyttsx3

---

## Project Structure


---

### Updated Project Structure

```text
BrailleConnect/
│
├── dashboard.py
├── accuracy.py
├── find_camera.py
├── svm_braille.pkl
├── requirements.txt
├── README.md
├── SETUP_INSTRUCTIONS.md
│
├── model_metrics/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── F1_curve.png
│   ├── Prediction_Curve.png
│   ├── Prediction-Recall_Curve.png
│   ├── Recall-Confidence_Curve.png
│   ├── prediction_sample.jpeg
│   └── training_sample.jpeg
│
├── samples/
│   ├── sample_input/
│   │   ├── 001.png
│   │   ├── 002.png
│   │   ├── 003.png
│   │   └── ...
│   │
│   └── sample_output/
│       ├── 001_op.jpeg
│       ├── 002_op.jpeg
│       ├── 003_op.jpeg
│       └── ...
│
├── training/
│   ├── train_yolo.py
│   ├── train_svm.py
│   ├── train_now.py
│   └── data.yaml
│
└── weights/
    ├── best.pt
    └── last.pt
``` 

---

## How It Works

1. The webcam captures a Braille document in real time.
2. Image preprocessing improves visibility and contrast.
3. YOLOv8 detects Braille cells from the frame.
4. Detected characters are stabilized.
5. SVM verification validates predictions.
6. Recognized characters are converted into readable text.
7. The translated text can be spoken aloud using Text-to-Speech.

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

For detailed setup instructions, DroidCam configuration, troubleshooting, and camera setup, refer to:

📄 [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)

---
## Running the Application

Start the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

Open the local URL displayed in the terminal.

### Webcam Setup

After launching the dashboard:

1. In the **Camera Settings** section in the sidebar.
2. Set **Camera Index** to:

```text
0
```

3. Click **Start Camera** to begin Braille detection.

> Note: For most systems, index `0` corresponds to the default webcam. If the camera does not open, use `find_camera.py` to identify the correct camera index.

---

## Model Evaluation

To evaluate the trained YOLO model:

```bash
python accuracy.py
```

This displays:

- Accuracy (mAP50)
- Precision
- Recall
- F1 Score

---

## Camera Utility

To identify available camera indices:

```bash
python find_camera.py
```

Use the detected camera index in the dashboard settings if required.

## Dataset

The Braille detection model was trained using a custom Braille dataset annotated for object detection.

Dataset Source:
https://universe.roboflow.com/braille-eitpg/braille-detection-f0rb5

**Annotation Format:** YOLO

The dataset consists of annotated Braille character images used for training and evaluating the detection model.

---

## Project Documentation

The repository contains additional files and folders that support training, evaluation, deployment, and testing of the BrailleConnect system.

### Core Application Files

| File | Description |
|--------|--------|
| `dashboard.py` | Main Streamlit application used for real-time Braille recognition and translation. |
| `find_camera.py` | Utility script used to identify available camera indices on the system. |
| `accuracy.py` | Evaluation script used to calculate model performance metrics such as Accuracy (mAP50), Precision, Recall, and F1 Score. |
| `svm_braille.pkl` | Trained SVM verification model used to validate Braille character predictions. |

---

### Documentation

| File | Description |
|--------|--------|
| `README.md` | Project overview, features, architecture, and usage instructions. |
| `SETUP_INSTRUCTIONS.md` | Detailed installation guide, DroidCam setup, camera configuration, troubleshooting, and deployment instructions. |

---

### Training Resources

The `training/` folder contains files used during model development and training.

| File | Description |
|--------|--------|
| `train_yolo.py` | YOLOv8 training script for Braille character detection. |
| `train_svm.py` | SVM training script used for prediction verification. |
| `train_now.py` | Utility script for initiating model training. |
| `data.yaml` | YOLO dataset configuration file containing dataset paths, classes, and training configuration. |

---

### Model Weights

The `weights/` folder contains trained model checkpoints.

| File | Description |
|--------|--------|
| `best.pt` | Best-performing YOLOv8 model checkpoint based on validation performance. Used by the application for inference. |
| `last.pt` | Final checkpoint saved at the end of training. Useful for resuming training or further experimentation. |

---

### Model Evaluation Resources

The `model_metrics/` folder contains evaluation results and training artifacts.

Typical contents include:

- Confusion Matrix
- Precision Curve
- Recall Curve
- F1 Score Curve
- Precision-Recall Curve
- Training Results Graphs
- Validation Predictions
- Sample Detection Outputs

These files provide quantitative and visual evidence of model performance.

---

### Sample Inputs and Outputs

The `samples/` folder contains example Braille images along with their corresponding expected recognition outputs. These samples can be used for testing, demonstrations, and validating the performance of the system without requiring a live camera feed.

#### Sample Structure

```text
samples/
├── sample_input/
│   ├── 001.png
│   ├── 002.png
│   ├── ...
│
└── sample_output/
    ├── 001_op.jpeg
    ├── 002_op.jpeg
    ├── ...
```

#### Contents

| Folder | Description |
|----------|----------|
| `sample_input/` | Contains sample Braille images that can be used as test inputs for the model. |
| `sample_output/` | Contains the corresponding expected detection and recognition outputs generated by BrailleConnect. |

Each file in `sample_input/` has a corresponding expected output in `sample_output/`.

For example:

- `001.png` → `001_op.jpeg`
- `002.png` → `002_op.jpeg`
- `003.png` → `003_op.jpeg`

This naming convention makes it easy to compare the original Braille image with the output generated by BrailleConnect and verify the recognition results.

---

## Future Improvements

- Support for Grade 2 Braille
- Multi-line Braille paragraph recognition
- Mobile application integration
- OCR-assisted document alignment
- Cloud-based translation and accessibility services
- Multi Language Support

---

## Accessibility Impact

BrailleConnect aims to bridge the communication gap between Braille users and non-Braille readers by providing instant translation and speech output, making Braille content more accessible in educational and everyday environments.

---
## Contributors

- Shravani Joshi
- Sanika Mane
- Trisha Deshmukh
- Bliss Gonsalves

---

## License

This project is developed for educational, research, and accessibility-focused purposes.

---