import streamlit as st
import cv2
import numpy as np
import joblib
import os
import threading
from collections import deque, Counter
from ultralytics import YOLO

# ── TTS ───────────────────────────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

_tts_lock   = threading.Lock()
_tts_thread = None

def speak_text(text: str, rate: int = 150, volume: float = 1.0):
    if not TTS_AVAILABLE or not text.strip():
        return
    def _run():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate",   rate)
                engine.setProperty("volume", volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
    global _tts_thread
    if _tts_thread and _tts_thread.is_alive():
        return
    _tts_thread = threading.Thread(target=_run, daemon=True)
    _tts_thread.start()


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="BrailleVision", page_icon="⠿", layout="wide")

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne+Mono&family=Syne:wght@400;600;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #080b10 !important;
}
[data-testid="stSidebar"] {
    background: #070a0f !important;
    border-right: 1px solid #1a2035 !important;
}
[data-testid="stSidebar"] * { color: #8a9ab8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stSubheader { color: #c8d4e8 !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stCheckbox > label { color: #8a9ab8 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Main content text ── */
h1, h2, h3, p, label, .stMarkdown { color: #c8d4e8; }

/* ── Top header bar ── */
.bv-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid #1a2035;
    margin-bottom: 24px;
}
.bv-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 26px;
    color: #f0f4ff;
    letter-spacing: -0.5px;
}
.bv-logo span { color: #e8a020; }
.bv-tagline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #3a4a6a;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Engine status strip ── */
.engine-strip {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 3px;
    letter-spacing: 0.5px;
}
.pill-green  { background: #0d2318; color: #3ddc84; border: 1px solid #1a4030; }
.pill-amber  { background: #231a08; color: #e8a020; border: 1px solid #40300a; }
.pill-blue   { background: #0a1828; color: #5ab4ff; border: 1px solid #0a2848; }
.pill-red    { background: #280a0a; color: #ff6060; border: 1px solid #481010; }

/* ── Camera panel ── */
.cam-wrapper {
    background: #0a0d14;
    border: 1px solid #1a2035;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
}
.cam-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #3a4a6a;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 14px 0 14px;
    margin-bottom: -4px;
}

/* ── Translation output ── */
.translation-panel {
    background: #0a0d14;
    border: 1px solid #1a2035;
    border-radius: 10px;
    padding: 0;
    overflow: hidden;
    margin-bottom: 12px;
}
.translation-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #3a4a6a;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 12px 16px 8px 16px;
    border-bottom: 1px solid #1a2035;
}
.translation-text {
    font-family: 'Syne Mono', monospace;
    font-size: 44px;
    font-weight: 400;
    color: #e8a020;
    padding: 24px 20px 28px 20px;
    min-height: 110px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    letter-spacing: 6px;
    word-spacing: 18px;
    text-shadow: 0 0 30px rgba(232,160,32,0.25);
    line-height: 1.3;
}
.translation-text.waiting {
    font-size: 18px;
    color: #2a3550;
    letter-spacing: 3px;
}

/* ── Char ticker ── */
.chars-panel {
    background: #0a0d14;
    border: 1px solid #1a2035;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 12px;
}
.chars-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #3a4a6a;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 12px 16px 8px 16px;
    border-bottom: 1px solid #1a2035;
}
.chars-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px;
    color: #5ab4ff;
    padding: 12px 16px 14px 16px;
    min-height: 42px;
    letter-spacing: 2px;
}
.char-sep { color: #1e2d48; margin: 0 4px; }
.word-sep { color: #e8a020; margin: 0 6px; opacity: 0.7; }

/* ── Metrics row ── */
.metrics-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
}
.metric-card {
    flex: 1;
    background: #0a0d14;
    border: 1px solid #1a2035;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
}
.metric-value {
    font-family: 'Syne Mono', monospace;
    font-size: 26px;
    color: #f0f4ff;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #3a4a6a;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── Action buttons ── */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 10px 0 !important;
    transition: all 0.15s ease !important;
}

/* Start button */
.start-btn > button {
    background: #0d2318 !important;
    color: #3ddc84 !important;
    border: 1px solid #1a4030 !important;
}
.start-btn > button:hover {
    background: #133020 !important;
    border-color: #3ddc84 !important;
    color: #3ddc84 !important;
}

/* Stop button */
.stop-btn > button {
    background: #280a0a !important;
    color: #ff6060 !important;
    border: 1px solid #481010 !important;
}
.stop-btn > button:hover {
    background: #3a0f0f !important;
    border-color: #ff6060 !important;
    color: #ff6060 !important;
}

/* Speak button */
.speak-btn > button {
    background: #231a08 !important;
    color: #e8a020 !important;
    border: 1px solid #40300a !important;
}
.speak-btn > button:hover {
    background: #332408 !important;
    border-color: #e8a020 !important;
    color: #e8a020 !important;
}

/* Clear button */
.clear-btn > button {
    background: #0a0d14 !important;
    color: #3a4a6a !important;
    border: 1px solid #1a2035 !important;
}
.clear-btn > button:hover {
    background: #10141e !important;
    border-color: #3a4a6a !important;
    color: #8a9ab8 !important;
}

/* ── Status badge ── */
.status-bar {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    padding: 6px 12px;
    border-radius: 4px;
    display: inline-block;
    margin-top: 6px;
}
.status-locked    { background: #0d2318; color: #3ddc84; border: 1px solid #1a4030; }
.status-detecting { background: #231a08; color: #e8a020; border: 1px solid #40300a; }
.status-idle      { background: #0a0d14; color: #2a3550; border: 1px solid #1a2035; }

/* ── Section divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #1a2035;
    margin: 16px 0;
}

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
    background: #0a0d14 !important;
    border: 1px solid #1a2035 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne Mono', monospace !important;
    color: #f0f4ff !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #3a4a6a !important;
}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙ Controls")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.20, 0.05,
    help="Lower if many cells are missed (try 0.10–0.20 for dim/angled shots)")

st.sidebar.markdown("---")
st.sidebar.markdown("**Stability**")
SMOOTH_FRAMES       = st.sidebar.slider("Smoothing Window",    4,  30, 15)
VOTE_THRESHOLD      = st.sidebar.slider("Vote Threshold %",   40,  90, 65)
LOCK_FRAMES         = st.sidebar.slider("Lock-in Frames",      3,  15,  6)
IOU_THRESHOLD       = st.sidebar.slider("NMS IoU Threshold", 0.1, 0.9, 0.35, 0.05)
CLUSTER_DISTANCE    = st.sidebar.slider("X-Cluster Dist px",   5, 100,  25,
    help="Lower = better for closely spaced letters. Try 20-30 for standard Braille.")
WORD_GAP_MULTIPLIER = st.sidebar.slider("Word Gap ×",        1.2, 4.0, 2.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Image Enhancement**")
use_clahe         = st.sidebar.checkbox("CLAHE Contrast",        value=True)
use_sharpen       = st.sidebar.checkbox("Sharpen",               value=True)
use_denoise       = st.sidebar.checkbox("Denoise",               value=False)
use_gamma         = st.sidebar.checkbox("Gamma Brighten",        value=True)
gamma_value       = st.sidebar.slider("Gamma", 0.5, 3.0, 1.8, 0.1) if use_gamma else 1.0
use_adaptive      = st.sidebar.checkbox("Adaptive Threshold",    value=False)
show_preprocessed = st.sidebar.checkbox("Show Preprocessed Feed",value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Camera**")
camera_index  = st.sidebar.number_input("Camera Index", 0, 5, 1, 1)
use_dshow     = st.sidebar.checkbox("DirectShow (Windows)", value=True)
target_width  = st.sidebar.selectbox("Resolution", [640, 1280], index=0)
target_height = {640: 480, 1280: 720}[target_width]

st.sidebar.markdown("---")
st.sidebar.markdown("**Text-to-Speech**")
tts_enabled = st.sidebar.checkbox("Enable TTS", value=TTS_AVAILABLE, disabled=not TTS_AVAILABLE)
tts_auto    = st.sidebar.checkbox("Auto-speak on lock", value=False)
tts_rate    = st.sidebar.slider("Rate WPM", 80, 250, 150)
tts_volume  = st.sidebar.slider("Volume",  0.1, 1.0, 1.0, 0.05)
if not TTS_AVAILABLE:
    st.sidebar.caption("pip install pyttsx3")

st.sidebar.markdown("---")
st.sidebar.markdown("**SVM**")
use_svm = st.sidebar.checkbox("SVM Verification", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("""
📌 **Tips**  
Hold phone directly above paper  
Keep paper flat & well-lit  
Confidence 0.10–0.15 for faint dots  
Enable Adaptive Threshold for low contrast
""")


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bv-header">
  <div>
    <div class="bv-logo">⠿ Braille<span>Vision</span></div>
    <div class="bv-tagline">Real-time Braille Recognition Engine</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── LOAD YOLO ─────────────────────────────────────────────────────────────────
YOLO_PATH = "weights/best.pt" if os.path.exists("weights/best.pt") else "yolov8n.pt"

@st.cache_resource
def load_yolo(path):
    return YOLO(path)

try:
    model = load_yolo(YOLO_PATH)
except Exception as e:
    st.error(f"Cannot load YOLO: {e}")
    st.stop()

# ── LOAD SVM ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_svm():
    return joblib.load("svm_braille.pkl") if os.path.exists("svm_braille.pkl") else None

svm_model = load_svm() if use_svm else None

# Engine pills
yolo_pill = '<span class="pill pill-green">● YOLO v8 ACTIVE</span>'
svm_pill  = f'<span class="pill {"pill-green" if (svm_model and use_svm) else "pill-amber"}">{"● SVM ACTIVE" if (svm_model and use_svm) else "○ SVM OFF"}</span>'
tts_pill  = f'<span class="pill {"pill-blue" if (TTS_AVAILABLE and tts_enabled) else "pill-amber"}">{"♪ TTS READY" if (TTS_AVAILABLE and tts_enabled) else "♪ TTS OFF"}</span>'
st.markdown(f'<div class="engine-strip">{yolo_pill}{svm_pill}{tts_pill}</div>', unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    r = frame.copy()
    if use_denoise:
        r = cv2.fastNlMeansDenoisingColored(r, None, 6, 6, 7, 21)
    if use_gamma and gamma_value != 1.0:
        inv_g = 1.0 / gamma_value
        table = np.array([((i / 255.0) ** inv_g) * 255 for i in range(256)], dtype=np.uint8)
        r = cv2.LUT(r, table)
    if use_clahe:
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(l)
        r = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if use_sharpen:
        g = cv2.GaussianBlur(r, (0, 0), sigmaX=1.5)
        r = cv2.addWeighted(r, 1.8, g, -0.8, 0)
    if use_adaptive:
        gray  = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        gray  = cv2.bitwise_not(gray)
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        adapt  = cv2.dilate(adapt, kernel, iterations=1)
        r = cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)
    return r


def apply_nms(boxes, confs, labels, iou_thresh):
    if not boxes:
        return [], [], []
    boxes_a  = np.array(boxes,  dtype=np.float32)
    scores_a = np.array(confs,  dtype=np.float32)
    x1, y1, x2, y2 = boxes_a[:,0], boxes_a[:,1], boxes_a[:,2], boxes_a[:,3]
    areas  = (x2-x1+1) * (y2-y1+1)
    order  = scores_a.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1); h = np.maximum(0.0, yy2-yy1+1)
        iou = (w*h) / (areas[i] + areas[order[1:]] - w*h)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return [boxes[k] for k in keep], [confs[k] for k in keep], [labels[k] for k in keep]


def cluster_x_positions(detections, cluster_dist):
    """
    Cluster by X using FIXED ANCHOR (first point), not running mean.
    Running mean drifts and merges adjacent distinct letters (e.g. L+L in HELLO).
    """
    if not detections:
        return []
    clusters = []   # each entry: [anchor_x, [xs], [labels], [confs]]
    for x, label, conf in sorted(detections, key=lambda d: d[0]):
        placed = False
        for c in clusters:
            if abs(x - c[0]) < cluster_dist:   # fixed anchor comparison
                c[1].append(x); c[2].append(label); c[3].append(conf)
                placed = True; break
        if not placed:
            clusters.append([x, [x], [label], [conf]])
    result = []
    for c in clusters:
        best = int(np.argmax(c[3]))
        result.append((np.mean(c[1]), c[2][best], c[3][best]))
    return sorted(result, key=lambda d: d[0])


def build_text_with_spaces(voted_chars, box_widths, gap_multiplier):
    if not voted_chars: return ""
    if len(voted_chars) == 1: return voted_chars[0][1]
    widths = [box_widths[cx][1]-box_widths[cx][0] for cx,_ in voted_chars if cx in box_widths]
    avg_w  = np.mean(widths) if widths else np.mean([voted_chars[i+1][0]-voted_chars[i][0] for i in range(len(voted_chars)-1)])*0.6
    thresh = avg_w * gap_multiplier
    text   = voted_chars[0][1]
    for i in range(1, len(voted_chars)):
        if voted_chars[i][0] - voted_chars[i-1][0] > thresh: text += " "
        text += voted_chars[i][1]
    return text


def get_stable_text(buffer, vote_thresh_pct, cluster_dist, gap_multiplier):
    if len(buffer) < 2: return "", {}, False
    all_dets  = [det for frame in buffer for det in frame]
    if not all_dets: return "", {}, False
    clustered = cluster_x_positions([(d[0],d[1],d[2]) for d in all_dets], cluster_dist)
    min_votes = max(1, int(len(buffer)*(vote_thresh_pct/100)))
    voted, box_widths = [], {}
    for centre_x, _, _ in clustered:
        frame_labels, x1s, x2s = [], [], []
        for frame_dets in buffer:
            # Use half cluster_dist for vote matching to avoid cross-cluster bleeding
            best_match = None
            best_dist  = cluster_dist * 0.6
            for d in frame_dets:
                dist = abs(d[0] - centre_x)
                if dist < best_dist:
                    best_dist  = dist
                    best_match = d
            if best_match:
                frame_labels.append(best_match[1])
                if len(best_match) >= 5:
                    x1s.append(best_match[3]); x2s.append(best_match[4])
        if len(frame_labels) >= min_votes:
            majority = Counter(frame_labels).most_common(1)[0][0]
            voted.append((centre_x, majority))
            if x1s: box_widths[centre_x] = (np.mean(x1s), np.mean(x2s))
    if not voted: return "", {}, False
    voted.sort(key=lambda d: d[0])
    return build_text_with_spaces(voted, box_widths, gap_multiplier), box_widths, len(buffer) >= SMOOTH_FRAMES


def svm_verify(crop, yolo_label, svm):
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8: return yolo_label, False
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    feat    = resized.flatten().reshape(1, -1) / 255.0
    pred    = svm.predict(feat)[0]
    return yolo_label, str(pred) == str(yolo_label)


# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [
    ("detection_buffer", deque(maxlen=SMOOTH_FRAMES)),
    ("running",          False),
    ("last_stable_text", ""),
    ("stable_run",       0),
    ("locked_text",      ""),
    ("last_spoken",      ""),
    ("pending_speak",    False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.detection_buffer.maxlen != SMOOTH_FRAMES:
    st.session_state.detection_buffer = deque(st.session_state.detection_buffer, maxlen=SMOOTH_FRAMES)


# ── LAYOUT ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="medium")

with left:
    st.markdown('<div class="cam-label">● LIVE FEED</div>', unsafe_allow_html=True)
    frame_ph  = st.empty()
    status_ph = st.empty()
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # Camera controls
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        start_btn = st.button("▶  START CAMERA", use_container_width=True, key="start")
        st.markdown('</div>', unsafe_allow_html=True)
    with cc2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_btn = st.button("■  STOP CAMERA", use_container_width=True, key="stop")
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    # Translation output
    translation_ph = st.empty()

    # Action buttons
    ab1, ab2 = st.columns(2)
    with ab1:
        st.markdown('<div class="speak-btn">', unsafe_allow_html=True)
        speak_btn = st.button("♪  SPEAK", use_container_width=True, key="speak")
        st.markdown('</div>', unsafe_allow_html=True)
    with ab2:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        clear_btn = st.button("✕  CLEAR", use_container_width=True, key="clear")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    # Chars ticker
    chars_ph = st.empty()

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    # Metrics
    mA, mB, mC = st.columns(3)
    with mA: m_dets = st.empty()
    with mB: m_conf = st.empty()
    with mC: m_stab = st.empty()


# ── BUTTON HANDLERS ───────────────────────────────────────────────────────────
if start_btn:
    st.session_state.running          = True
    st.session_state.detection_buffer.clear()
    st.session_state.locked_text      = ""
    st.session_state.last_stable_text = ""
    st.session_state.stable_run       = 0

if stop_btn:
    st.session_state.running = False

if speak_btn:
    st.session_state.pending_speak = True

if clear_btn:
    st.session_state.locked_text      = ""
    st.session_state.last_stable_text = ""
    st.session_state.stable_run       = 0
    st.session_state.last_spoken      = ""
    st.session_state.detection_buffer.clear()

# Speak when camera is NOT running
if not st.session_state.running and st.session_state.get("pending_speak"):
    st.session_state.pending_speak = False
    txt = st.session_state.locked_text
    if not TTS_AVAILABLE:
        st.warning("pyttsx3 not installed — run `pip install pyttsx3`")
    elif not tts_enabled:
        st.warning("TTS is disabled in the sidebar.")
    elif not txt:
        st.info("No locked text yet.")
    else:
        speak_text(txt, tts_rate, tts_volume)
        st.toast(f"Speaking: {txt}")


# ── IDLE STATE RENDER ─────────────────────────────────────────────────────────
def render_idle():
    frame_ph.markdown("""
    <div style="background:#0a0d14;border:1px solid #1a2035;border-radius:10px;
                height:320px;display:flex;flex-direction:column;align-items:center;
                justify-content:center;gap:12px;">
      <div style="font-size:48px;opacity:0.15">⠿</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                  color:#2a3550;letter-spacing:2px;">CAMERA OFFLINE</div>
    </div>
    """, unsafe_allow_html=True)
    status_ph.markdown(
        '<span class="status-bar status-idle">○ IDLE — press START CAMERA</span>',
        unsafe_allow_html=True)
    locked = st.session_state.locked_text
    if locked:
        translation_ph.markdown(f"""
        <div class="translation-panel">
          <div class="translation-label">TRANSLATION</div>
          <div class="translation-text">{locked}</div>
        </div>""", unsafe_allow_html=True)
    else:
        translation_ph.markdown("""
        <div class="translation-panel">
          <div class="translation-label">TRANSLATION</div>
          <div class="translation-text waiting">AWAITING INPUT</div>
        </div>""", unsafe_allow_html=True)
    chars_ph.markdown("""
    <div class="chars-panel">
      <div class="chars-label">DETECTED CHARACTERS</div>
      <div class="chars-text" style="color:#2a3550">—</div>
    </div>""", unsafe_allow_html=True)
    m_dets.metric("Cells",      "—")
    m_conf.metric("Confidence", "—")
    m_stab.metric("Stable Run", "—")

if not st.session_state.running:
    render_idle()


# ── MAIN CAMERA LOOP ──────────────────────────────────────────────────────────
if st.session_state.running:

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if use_dshow else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  target_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,     0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    if not cap.isOpened():
        st.error(f"Cannot open camera index {camera_index}.")
        st.session_state.running = False
        st.stop()

    for _ in range(10):
        cap.read()

    while st.session_state.running:

        ok, frame = cap.read()
        if not ok:
            st.error("❌ Camera read failed — check connection.")
            st.session_state.running = False
            break

        processed = preprocess_frame(frame)
        results   = model(processed, conf=confidence_threshold, iou=IOU_THRESHOLD,
                          verbose=False, augment=True)
        result    = results[0]

        raw_boxes, raw_confs, raw_labels = [], [], []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_labels.append(model.names[cls_id])
            raw_confs.append(float(box.conf[0]))
            raw_boxes.append(box.xyxy[0].tolist())

        b_nms, c_nms, l_nms = apply_nms(raw_boxes, raw_confs, raw_labels, IOU_THRESHOLD)

        frame_dets = []
        annotated  = frame.copy()

        for xyxy, label, conf in zip(b_nms, l_nms, c_nms):
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) / 2
            if svm_model and use_svm:
                final_label, agreed = svm_verify(frame[y1:y2, x1:x2], label, svm_model)
                col = (52, 198, 100) if agreed else (255, 160, 32)
                tag = f"{final_label}{'✓' if agreed else '?'}"
            else:
                final_label = label
                col = (255, 180, 40)   # amber boxes to match theme
                tag = final_label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(annotated, tag, (x1, max(y1-7, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            frame_dets.append((cx, final_label, conf, float(x1), float(x2)))

        clustered_frame = cluster_x_positions(
            [(d[0], d[1], d[2]) for d in frame_dets], CLUSTER_DISTANCE)

        def find_box(cx):
            # Use tighter match to avoid wrong box being assigned to adjacent letter
            best, best_dist = None, CLUSTER_DISTANCE * 0.6
            for d in frame_dets:
                dist = abs(d[0] - cx)
                if dist < best_dist:
                    best_dist = dist; best = d
            if best: return best[3], best[4]
            return cx-20, cx+20

        frame_dets_full = [(cx, lbl, conf, *find_box(cx)) for cx, lbl, conf in clustered_frame]

        # Word-gap markers (amber dashed lines)
        if len(frame_dets_full) > 1:
            avg_w  = np.mean([d[4]-d[3] for d in frame_dets_full])
            thresh = avg_w * WORD_GAP_MULTIPLIER
            for i in range(1, len(frame_dets_full)):
                if frame_dets_full[i][0] - frame_dets_full[i-1][0] > thresh:
                    mid_x = int((frame_dets_full[i][0] + frame_dets_full[i-1][0]) / 2)
                    for yd in range(0, frame.shape[0], 12):
                        cv2.line(annotated, (mid_x, yd),
                                 (mid_x, min(yd+6, frame.shape[0])), (255, 180, 40), 1)

        # Detection count overlay
        cv2.putText(annotated,
                    f"{len(frame_dets_full)} cells  conf>{confidence_threshold:.2f}",
                    (8, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 100, 140), 1)

        st.session_state.detection_buffer.append(frame_dets_full)

        stable_text, _, is_stable = get_stable_text(
            st.session_state.detection_buffer, VOTE_THRESHOLD, CLUSTER_DISTANCE, WORD_GAP_MULTIPLIER)

        # Lock-in logic
        if stable_text == st.session_state.last_stable_text:
            st.session_state.stable_run += 1
        else:
            st.session_state.stable_run   = 1
            st.session_state.last_stable_text = stable_text

        if st.session_state.stable_run >= LOCK_FRAMES and stable_text:
            if stable_text != st.session_state.locked_text:
                st.session_state.locked_text = stable_text
                if tts_auto and tts_enabled and TTS_AVAILABLE:
                    if stable_text != st.session_state.last_spoken:
                        speak_text(stable_text, tts_rate, tts_volume)
                        st.session_state.last_spoken = stable_text

        # Consume pending speak (button clicked while camera running)
        if st.session_state.get("pending_speak"):
            st.session_state.pending_speak = False
            txt = st.session_state.locked_text
            if TTS_AVAILABLE and tts_enabled and txt:
                speak_text(txt, tts_rate, tts_volume)

        # ── Render camera feed ────────────────────────────────────────────────
        if show_preprocessed:
            feed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        else:
            feed = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_ph.image(feed, channels="RGB", use_container_width=True)

        # Status badge
        buf_len    = len(st.session_state.detection_buffer)
        lock_ratio = min(st.session_state.stable_run, LOCK_FRAMES)
        if is_stable and st.session_state.stable_run >= LOCK_FRAMES:
            status_ph.markdown(
                f'<span class="status-bar status-locked">🔒 LOCKED  {lock_ratio}/{LOCK_FRAMES}  BUF {buf_len}/{SMOOTH_FRAMES}</span>',
                unsafe_allow_html=True)
        else:
            status_ph.markdown(
                f'<span class="status-bar status-detecting">⏳ STABILISING  {lock_ratio}/{LOCK_FRAMES}  BUF {buf_len}/{SMOOTH_FRAMES}</span>',
                unsafe_allow_html=True)

        # Translation box
        display_text = st.session_state.locked_text
        if display_text:
            translation_ph.markdown(f"""
            <div class="translation-panel">
              <div class="translation-label">TRANSLATION</div>
              <div class="translation-text">{display_text}</div>
            </div>""", unsafe_allow_html=True)
        else:
            translation_ph.markdown("""
            <div class="translation-panel">
              <div class="translation-label">TRANSLATION</div>
              <div class="translation-text waiting">AWAITING INPUT</div>
            </div>""", unsafe_allow_html=True)

        # Char ticker
        char_html = ""
        if frame_dets_full:
            avg_w  = np.mean([d[4]-d[3] for d in frame_dets_full])
            thresh = avg_w * WORD_GAP_MULTIPLIER
            parts  = []
            for i, d in enumerate(frame_dets_full):
                if i > 0:
                    gap = frame_dets_full[i][0] - frame_dets_full[i-1][0]
                    if gap > thresh:
                        parts.append('<span class="word-sep">◆</span>')
                    else:
                        parts.append('<span class="char-sep">·</span>')
                parts.append(f'<span style="color:#5ab4ff">{d[1]}</span>')
            char_html = "".join(parts)
        chars_ph.markdown(f"""
        <div class="chars-panel">
          <div class="chars-label">DETECTED CHARACTERS</div>
          <div class="chars-text">{char_html if char_html else '<span style="color:#2a3550">—</span>'}</div>
        </div>""", unsafe_allow_html=True)

        # Metrics
        avg_conf = sum(d[2] for d in frame_dets_full)/len(frame_dets_full) if frame_dets_full else 0
        m_dets.metric("Cells",       len(frame_dets_full))
        m_conf.metric("Confidence",  f"{avg_conf*100:.1f}%")
        m_stab.metric("Stable Run",  st.session_state.stable_run)

    cap.release()