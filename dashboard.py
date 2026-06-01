import streamlit as st
import cv2
import numpy as np
import joblib
import os
from collections import deque
from ultralytics import YOLO

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="BrailleVision Engine",
    page_icon="👁️",
    layout="wide"
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

.main { background-color: #0f1117; }

.big-box {
    background: #1a1d27;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2a2d3a;
    font-family: 'Space Mono', monospace;
    color: #e0e0e0;
    font-size: 18px;
    letter-spacing: 2px;
    word-spacing: 8px;
}

.translation-box {
    background: linear-gradient(135deg, #1a1d27, #12151f);
    padding: 30px 25px;
    border-radius: 14px;
    text-align: center;
    font-size: 42px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    color: #4fc3f7;
    border: 1px solid #2a2d3a;
    letter-spacing: 4px;
    word-spacing: 16px;
    min-height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
}

.engine-box {
    background: #12151f;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #8b9ab0;
    font-family: 'Space Mono', monospace;
    margin-top: 8px;
}

.engine-active   { color: #66bb6a; }
.engine-fallback { color: #ffa726; }
.status-stable    { color: #66bb6a; font-weight: 600; }
.status-detecting { color: #ffa726; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Control Panel")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.30, 0.05,
    help="Lower = detects more, Higher = only confident detections"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Stability Settings")

SMOOTH_FRAMES = st.sidebar.slider(
    "Smoothing Window (frames)",
    min_value=2, max_value=20, value=10,
    help="Higher = more stable but slower to update"
)

VOTE_THRESHOLD = st.sidebar.slider(
    "Vote Threshold (%)",
    min_value=30, max_value=90, value=60,
    help="Character must appear this % of frames to be accepted"
)

IOU_THRESHOLD = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.1, max_value=0.9, value=0.4,
    help="Lower = more aggressive duplicate removal"
)

CLUSTER_DISTANCE = st.sidebar.slider(
    "X-Cluster Distance (px)",
    min_value=5, max_value=100, value=40,
    help="Pixels within which two detections count as the same cell"
)

WORD_GAP_MULTIPLIER = st.sidebar.slider(
    "Word Gap Multiplier",
    min_value=1.2, max_value=4.0, value=2.0, step=0.1,
    help="Gap between cells > (avg cell width × this) = word space. Lower = more spaces detected."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Image Enhancement")

use_clahe   = st.sidebar.checkbox("CLAHE Contrast Boost", value=True)
use_sharpen = st.sidebar.checkbox("Sharpen Frame",         value=True)
use_denoise = st.sidebar.checkbox("Denoise Frame",         value=False,
    help="Turn on if feed looks grainy")

st.sidebar.markdown("---")
st.sidebar.subheader("📱 DroidCam Settings")

camera_index = st.sidebar.number_input(
    "Camera Index", min_value=0, max_value=5, value=1, step=1,
    help="DroidCam is usually index 1. Run find_camera.py to confirm."
)

use_dshow = st.sidebar.checkbox("Use DirectShow (Windows only)", value=True)

target_width  = st.sidebar.selectbox("Capture Resolution", options=[640, 1280], index=0)
target_height = {640: 480, 1280: 720}[target_width]

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 SVM Settings")

use_svm = st.sidebar.checkbox(
    "Enable SVM Verification", value=False,
    help="Disable if you see ? marks — SVM may not match live lighting"
)

st.sidebar.markdown("---")
st.sidebar.info("""
### Tips for Best Results
- Hold phone **directly above** paper
- Keep paper **flat and well-lit**
- All characters should **fit in frame**
- Tap phone screen to **lock focus**
- Adjust **Word Gap Multiplier** to tune spacing
""")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("👁️ BrailleVision Hybrid Engine")
st.caption("YOLOv8 · Word-Space Detection · Temporal Stabilization")
st.divider()

# --------------------------------------------------
# MODEL LOAD — YOLO
# --------------------------------------------------
YOLO_PATH = "weights/best.pt" if os.path.exists("weights/best.pt") else "yolov8n.pt"

@st.cache_resource
def load_yolo(path):
    return YOLO(path)

try:
    model = load_yolo(YOLO_PATH)
except Exception as e:
    st.error(f"Unable to load YOLO model: {e}")
    st.stop()

# --------------------------------------------------
# MODEL LOAD — SVM
# --------------------------------------------------
@st.cache_resource
def load_svm():
    if os.path.exists("svm_braille.pkl"):
        return joblib.load("svm_braille.pkl")
    return None

svm_model = load_svm() if use_svm else None

if svm_model and use_svm:
    engine_html = """
    <div class='engine-box'>
        🟢 <span class='engine-active'>Stage 1: YOLOv8 — Active</span><br>
        🧬 <span class='engine-active'>Stage 2: SVM RBF Verifier — Active (Dual-Engine Mode)</span>
    </div>"""
else:
    engine_html = """
    <div class='engine-box'>
        🟢 <span class='engine-active'>Stage 1: YOLOv8 — Active (YOLO-Only Mode)</span><br>
        🟠 <span class='engine-fallback'>Stage 2: SVM — Disabled · Enable in sidebar if needed</span>
    </div>"""

st.markdown(engine_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# HELPER: Preprocess frame
# --------------------------------------------------
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    result = frame.copy()
    if use_denoise:
        result = cv2.fastNlMeansDenoisingColored(result, None, 6, 6, 7, 21)
    if use_clahe:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if use_sharpen:
        gaussian = cv2.GaussianBlur(result, (0, 0), sigmaX=2)
        result   = cv2.addWeighted(result, 1.6, gaussian, -0.6, 0)
    return result


# --------------------------------------------------
# HELPER: Manual NMS
# --------------------------------------------------
def apply_nms(boxes_xyxy, confidences, labels, iou_thresh):
    if not boxes_xyxy:
        return [], [], []
    boxes  = np.array(boxes_xyxy, dtype=np.float32)
    scores = np.array(confidences, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2-x1+1) * (y2-y1+1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0.0, xx2-xx1+1)
        h   = np.maximum(0.0, yy2-yy1+1)
        iou = (w*h) / (areas[i] + areas[order[1:]] - w*h)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return [boxes_xyxy[k] for k in keep], [confidences[k] for k in keep], [labels[k] for k in keep]


# --------------------------------------------------
# HELPER: X-position clustering
# --------------------------------------------------
def cluster_x_positions(detections, cluster_dist):
    if not detections:
        return []
    sorted_det = sorted(detections, key=lambda d: d[0])
    clusters   = []
    for x, label, conf in sorted_det:
        placed = False
        for cluster in clusters:
            if abs(x - np.mean(cluster[0])) < cluster_dist:
                cluster[0].append(x); cluster[1].append(label); cluster[2].append(conf)
                placed = True; break
        if not placed:
            clusters.append([[x], [label], [conf]])
    result = []
    for cluster in clusters:
        centre_x = np.mean(cluster[0])
        best_idx = int(np.argmax(cluster[2]))
        result.append((centre_x, cluster[1][best_idx], cluster[2][best_idx]))
    return sorted(result, key=lambda d: d[0])


# --------------------------------------------------
# HELPER: Word-space aware text builder
# --------------------------------------------------
def build_text_with_spaces(voted_chars, boxes_by_label, gap_multiplier):
    """
    voted_chars : list of (centre_x, label) sorted by x
    boxes_by_label : dict of centre_x -> (x1, x2) for width estimation
    gap_multiplier : word gap threshold = avg_cell_width * gap_multiplier

    Logic:
      - Compute average cell width from bounding boxes
      - Walk left→right; if gap between consecutive cells >
        avg_cell_width * gap_multiplier, insert a space
    """
    if not voted_chars:
        return ""

    if len(voted_chars) == 1:
        return voted_chars[0][1]

    # Estimate avg cell width
    widths = []
    for cx, _ in voted_chars:
        if cx in boxes_by_label:
            x1, x2 = boxes_by_label[cx]
            widths.append(x2 - x1)

    # Fallback: estimate from cluster spacing if no box widths
    if widths:
        avg_width = np.mean(widths)
    else:
        gaps = [voted_chars[i+1][0] - voted_chars[i][0]
                for i in range(len(voted_chars)-1)]
        avg_width = np.mean(gaps) * 0.6  # rough cell width estimate

    word_gap_threshold = avg_width * gap_multiplier

    # Build text with spaces
    text = voted_chars[0][1]
    for i in range(1, len(voted_chars)):
        gap = voted_chars[i][0] - voted_chars[i-1][0]
        if gap > word_gap_threshold:
            text += " "   # word boundary detected
        text += voted_chars[i][1]

    return text


# --------------------------------------------------
# HELPER: Temporal voting (returns chars + box widths)
# --------------------------------------------------
def get_stable_text(buffer, vote_thresh_pct, cluster_dist, gap_multiplier):
    if len(buffer) < 2:
        return "", {}, False

    # buffer entries are (centre_x, label, conf, x1, x2)
    all_dets = [det for frame in buffer for det in frame]
    if not all_dets:
        return "", {}, False

    # cluster using only first 3 fields
    clustered  = cluster_x_positions(
        [(d[0], d[1], d[2]) for d in all_dets], cluster_dist
    )
    min_votes  = max(1, int(len(buffer) * (vote_thresh_pct / 100)))
    voted      = []
    box_widths = {}   # centre_x -> (avg_x1, avg_x2) across frames

    for centre_x, label, _ in clustered:
        votes  = 0
        x1s, x2s = [], []
        for frame_dets in buffer:
            for d in frame_dets:
                if abs(d[0] - centre_x) < cluster_dist:
                    votes += 1
                    if len(d) >= 5:
                        x1s.append(d[3]); x2s.append(d[4])
                    break
        if votes >= min_votes:
            voted.append((centre_x, label))
            if x1s:
                box_widths[centre_x] = (np.mean(x1s), np.mean(x2s))

    if not voted:
        return "", {}, False

    voted.sort(key=lambda d: d[0])
    text       = build_text_with_spaces(voted, box_widths, gap_multiplier)
    is_stable  = len(buffer) >= SMOOTH_FRAMES
    return text, box_widths, is_stable


# --------------------------------------------------
# HELPER: SVM verify
# --------------------------------------------------
def svm_verify(crop, yolo_label, svm):
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return yolo_label, False
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    feat    = resized.flatten().reshape(1, -1) / 255.0
    pred    = svm.predict(feat)[0]
    return yolo_label, str(pred) == str(yolo_label)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "detection_buffer" not in st.session_state:
    st.session_state.detection_buffer = deque(maxlen=SMOOTH_FRAMES)

if "running" not in st.session_state:
    st.session_state.running = False

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("🎥 Live Camera Feed")
    frame_placeholder  = st.empty()
    status_placeholder = st.empty()

with right:
    st.subheader("📖 Translation")
    translation_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔤 Detected Characters")
    chars_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    with colA: detections_metric = st.empty()
    with colB: confidence_metric = st.empty()
    with colC: stability_metric  = st.empty()

# --------------------------------------------------
# CAMERA CONTROL
# --------------------------------------------------
col_start, col_stop = st.columns(2)
with col_start:
    start = st.button("▶️ Start Camera", use_container_width=True)
with col_stop:
    stop  = st.button("⏹️ Stop Camera",  use_container_width=True)

if start:
    st.session_state.running = True
    st.session_state.detection_buffer.clear()

if stop:
    st.session_state.running = False

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
if st.session_state.running:

    if use_dshow:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  target_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,     0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    if not cap.isOpened():
        st.error(f"Could not open camera index {camera_index}. Try a different index.")
        st.stop()

    for _ in range(10):
        cap.read()

    while st.session_state.running:

        success, frame = cap.read()
        if not success:
            st.error("❌ Camera read failed. Check DroidCam connection.")
            break

        processed = preprocess_frame(frame)

        results = model(
            processed,
            conf=confidence_threshold,
            iou=IOU_THRESHOLD,
            verbose=False,
            augment=True,
        )
        result = results[0]

        raw_boxes, raw_confs, raw_labels = [], [], []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_labels.append(model.names[cls_id])
            raw_confs.append(float(box.conf[0]))
            raw_boxes.append(box.xyxy[0].tolist())

        boxes_nms, confs_nms, labels_nms = apply_nms(
            raw_boxes, raw_confs, raw_labels, IOU_THRESHOLD
        )

        # ── Build detections — store x1,x2 for word gap estimation ──
        frame_detections = []   # (centre_x, label, conf, x1, x2)
        annotated = frame.copy()
        svm_verified_count = 0

        for xyxy, label, conf in zip(boxes_nms, labels_nms, confs_nms):
            x1, y1, x2, y2 = map(int, xyxy)
            x_centre = (x1 + x2) / 2

            if svm_model and use_svm:
                crop = frame[y1:y2, x1:x2]
                final_label, agreed = svm_verify(crop, label, svm_model)
                if agreed:
                    box_color = (52, 152, 219); tag_text = f"{final_label} ✓"
                    svm_verified_count += 1
                else:
                    box_color = (0, 165, 255);  tag_text = f"{final_label} ?"
            else:
                final_label = label
                box_color   = (46, 204, 113)
                tag_text    = final_label

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated, tag_text, (x1, max(y1-8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            # Draw word-gap indicator lines between cells
            frame_detections.append((x_centre, final_label, conf, float(x1), float(x2)))

        # Cluster (use first 3 fields)
        clustered_frame = cluster_x_positions(
            [(d[0], d[1], d[2]) for d in frame_detections], CLUSTER_DISTANCE
        )

        # Re-attach x1/x2 to clustered results
        def find_box(cx):
            for d in frame_detections:
                if abs(d[0] - cx) < CLUSTER_DISTANCE:
                    return d[3], d[4]
            return None, None

        frame_dets_full = []
        for cx, lbl, conf in clustered_frame:
            bx1, bx2 = find_box(cx)
            frame_dets_full.append((cx, lbl, conf,
                                    bx1 if bx1 else cx-20,
                                    bx2 if bx2 else cx+20))

        # Draw word-gap markers on annotated frame
        if len(frame_dets_full) > 1:
            avg_w = np.mean([d[4]-d[3] for d in frame_dets_full])
            thresh = avg_w * WORD_GAP_MULTIPLIER
            for i in range(1, len(frame_dets_full)):
                gap = frame_dets_full[i][0] - frame_dets_full[i-1][0]
                if gap > thresh:
                    # Draw a vertical dashed line to mark word boundary
                    mid_x = int((frame_dets_full[i][0] + frame_dets_full[i-1][0]) / 2)
                    for y_dash in range(0, frame.shape[0], 12):
                        cv2.line(annotated,
                                 (mid_x, y_dash),
                                 (mid_x, min(y_dash+6, frame.shape[0])),
                                 (255, 255, 0), 1)

        st.session_state.detection_buffer.append(frame_dets_full)

        stable_text, box_widths, is_stable = get_stable_text(
            st.session_state.detection_buffer,
            VOTE_THRESHOLD,
            CLUSTER_DISTANCE,
            WORD_GAP_MULTIPLIER
        )

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        buf_len  = len(st.session_state.detection_buffer)
        svm_info = f" · SVM ✓ {svm_verified_count}/{len(clustered_frame)}" if (svm_model and use_svm) else ""

        if is_stable:
            status_placeholder.markdown(
                f"<p class='status-stable'>🔒 Stable lock · Buffer {buf_len}/{SMOOTH_FRAMES}{svm_info}</p>",
                unsafe_allow_html=True)
        else:
            status_placeholder.markdown(
                f"<p class='status-detecting'>⏳ Building buffer… {buf_len}/{SMOOTH_FRAMES}</p>",
                unsafe_allow_html=True)

        display_text = stable_text if stable_text else "Waiting…"
        translation_placeholder.markdown(
            f"<div class='translation-box'>{display_text}</div>",
            unsafe_allow_html=True)

        # Detected chars with | between letters and · between words
        char_display = ""
        if clustered_frame:
            avg_w  = np.mean([d[4]-d[3] for d in frame_dets_full]) if frame_dets_full else 40
            thresh = avg_w * WORD_GAP_MULTIPLIER
            parts  = [clustered_frame[0][1]]
            for i in range(1, len(clustered_frame)):
                gap = clustered_frame[i][0] - clustered_frame[i-1][0]
                if gap > thresh:
                    parts.append("·")   # word gap marker
                parts.append(clustered_frame[i][1])
            char_display = " | ".join(parts)

        chars_placeholder.markdown(
            f"<div class='big-box'>{char_display if char_display else '—'}</div>",
            unsafe_allow_html=True)

        avg_conf = (
            sum(d[2] for d in frame_dets_full) / len(frame_dets_full)
            if frame_dets_full else 0
        )
        detections_metric.metric("Cells",      len(frame_dets_full))
        confidence_metric.metric("Confidence", f"{avg_conf*100:.1f}%")
        stability_metric.metric("Buffer",      f"{buf_len}/{SMOOTH_FRAMES}")

    cap.release()