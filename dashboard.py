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
st.set_page_config(page_title="BrailleVision", page_icon="⠿", layout="wide", initial_sidebar_state="expanded")

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=JetBrains+Mono:wght@400;500;700&family=Spectral:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap');

:root {
    --bg:        #ede8de;
    --card:      #faf8f5;
    --navy:      #0e1a2e;
    --navy2:     #1a2d4a;
    --amber:     #a97010;
    --amber-lt:  #fdf2da;
    --amber-mid: #d4920e;
    --text:      #000000;
    --text2:     #000000;
    --muted:     #000000;
    --border:    #ddd7cc;
    --border2:   #e8e3d9;
    --green:     #145e30;
    --green-lt:  #edf7f1;
    --red:       #a82020;
    --red-lt:    #fdf0f0;
    --blue:      #1a3d80;
    --blue-lt:   #eef3fc;
    --shadow:    0 1px 3px rgba(28,24,20,0.07), 0 4px 12px rgba(28,24,20,0.05);
    --shadow-sm: 0 1px 2px rgba(28,24,20,0.06);
}

html, body { background: var(--bg) !important; }

/* ── Braille-dot paper texture ── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(circle at 7px 9px,  rgba(169,112,16,0.22) 1.8px, transparent 1.8px),
        radial-gradient(circle at 7px 23px, rgba(169,112,16,0.15) 1.8px, transparent 1.8px),
        radial-gradient(circle at 7px 37px, rgba(169,112,16,0.11) 1.8px, transparent 1.8px),
        radial-gradient(circle at 21px 9px,  rgba(169,112,16,0.18) 1.8px, transparent 1.8px),
        radial-gradient(circle at 21px 23px, rgba(169,112,16,0.10) 1.8px, transparent 1.8px),
        radial-gradient(circle at 21px 37px, rgba(169,112,16,0.20) 1.8px, transparent 1.8px) !important;
    background-size: 32px 50px !important;
}

/* ── Hide Streamlit chrome & sidebar collapse elements ── */
#MainMenu, footer, header, [data-testid="stHeader"], 
[data-testid="stDecoration"],
[data-testid="stSidebarCollapseButton"],
div[class*="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
div[class*="stSidebarCollapsedControl"] {
    display: none !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    background-image: none !important;
    border-right: 2px solid #000000 !important;
}
[data-testid="stSidebar"] * {
    color: #000000 !important;
    font-weight: 500;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #000000 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    color: #000000 !important;
    padding-bottom: 4px !important;
    display: inline-block !important;
}
[data-testid="stSidebar"] input[type="number"] {
    background-color: #FFFFFF !important;
    border: 2px solid #000000 !important;
    color: #000000 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
    background: #000000 !important;
    border-color: #000000 !important;
}
[data-testid="stSidebar"] .stCaption {
    color: #000000 !important;
    font-weight: 500 !important;
}

/* Style stepper buttons (+ and -) to be visible and match input styling */
[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"] button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: none !important;
    border-left: 1px solid #000000 !important;
}
[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"] button:hover {
    background-color: #EEEEEE !important;
}
[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"] button svg {
    fill: #000000 !important;
    stroke: #000000 !important;
}

/* ── Sidebar Sections & Controls Spacing ── */
.sb-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #000000 !important;
    margin-top: 16px !important;
    margin-bottom: 8px !important;
    padding-bottom: 6px;
    border-bottom: 3px solid #000000 !important;
}
.sb-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #000000 !important;
    line-height: 1.6;
    margin: 12px 0 !important;
    padding: 12px !important;
    background: #F0F4FF !important;
    border-radius: 5px;
    border: 2px solid #000000 !important;
    border-left: 6px solid #d4920e !important;
}
.sb-note strong {
    font-weight: 700 !important;
}
.sb-tip {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    margin-bottom: 6px;
    font-family: 'Spectral', serif;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #000000 !important;
    line-height: 1.3;
    padding: 4px 0 !important;
}
.sb-tip-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #000000 !important;
    margin-top: 5px;
    flex-shrink: 0;
}

/* Padding and spacing inside Sidebar widgets */
[data-testid="stSidebar"] .stNumberInput, 
[data-testid="stSidebar"] .stSlider {
    padding: 12px 0 !important;
}
/* Style status/spinner widgets to be transparent with no icon, showing only black text */
div[data-testid="stStatusWidget"],
div[data-testid="stSpinner"],
div.stSpinner,
div[class*="stSpinner"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Hide the circular loading spinner animation/icons */
div[data-testid="stStatusWidget"] svg,
div[data-testid="stSpinner"] svg,
div.stSpinner svg,
div[class*="stSpinner"] svg {
    display: none !important;
}

/* Set text font to black and remove any nested background styling */
div[data-testid="stStatusWidget"] *,
div[data-testid="stSpinner"] *,
div.stSpinner *,
div[class*="stSpinner"] * {
    color: #000000 !important;
    background: transparent !important;
    background-color: transparent !important;
    font-weight: 500 !important;
}



/* ── Main content ── */
[data-testid="stMain"] .block-container {
    padding-top: 1.8rem !important;
    max-width: 1360px !important;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding-bottom: 18px;
    border-bottom: 1.5px solid var(--border);
    margin-bottom: 18px;
}
.page-wordmark {
    display: flex;
    align-items: baseline;
    gap: 12px;
}
.page-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 34px;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.5px;
    line-height: 1;
}
.page-title em { color: var(--amber-mid); font-style: italic; }
.page-ver {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--amber);
    background: var(--amber-lt);
    border: 1px solid rgba(169,112,16,0.22);
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.page-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: bold !important;
    color: #000000 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 5px;
}
.engine-strip {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
    padding-bottom: 2px;
}
.pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 0.5px;
    font-weight: 500;
}
.pill-green { background: var(--green-lt); color: var(--green); border: 1px solid rgba(20,94,48,0.18); }
.pill-amber { background: var(--amber-lt); color: var(--amber); border: 1px solid rgba(169,112,16,0.20); }
.pill-blue  { background: var(--blue-lt);  color: var(--blue);  border: 1px solid rgba(26,61,128,0.18); }
.pill-muted { background: var(--border2);  color: var(--muted); border: 1px solid var(--border); }

/* ── Cards ── */
.bv-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.bv-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 16px;
    border-bottom: 1px solid var(--border2);
    background: var(--card);
}
.bv-card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    color: #111111 !important;
    font-weight: bold !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    background: #22c55e;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.20);
    animation: blink 2s ease-in-out infinite;
}
.offline-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    background: var(--muted);
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

.cam-idle-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 340px;
    background: #f7f5f2;
    gap: 16px;
}
.cam-idle-glyph {
    font-family: monospace;
    font-size: 72px;
    color: var(--border);
    line-height: 1;
    user-select: none;
}
.cam-idle-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

/* ── Translation card ── */
.trans-body {
    padding: 26px 24px 30px;
    min-height: 115px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.trans-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 54px;
    font-weight: 600;
    color: var(--navy);
    letter-spacing: 10px;
    word-spacing: 22px;
    text-align: center;
    line-height: 1.28;
}
.trans-empty {
    font-family: 'Spectral', serif;
    font-style: italic;
    font-size: 16px;
    color: #000000 !important;
    font-weight: bold !important;
    letter-spacing: 0.5px;
}

/* ── Chars card ── */
.chars-body {
    padding: 10px 16px 13px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13.5px;
    color: var(--blue);
    min-height: 42px;
    letter-spacing: 1.5px;
    line-height: 1.6;
}
.char-sep  { color: var(--border); margin: 0 3px; }
.word-mark { color: var(--amber); margin: 0 6px; font-size: 10px; }

/* ── Metrics ── */
.metric-row { display: flex; gap: 8px; margin-bottom: 10px; }
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 10px 12px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.metric-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 30px;
    font-weight: 700;
    color: var(--navy);
    line-height: 1;
    margin-bottom: 5px;
}
.metric-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px !important;
    color: #000000 !important;
    font-weight: bold !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.stMarkdown p {
    color: #000000 !important;
    font-weight: bold !important;
}

/* ── Status badge ── */
.status-bar {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px !important;
    font-weight: bold !important;
    letter-spacing: 1px;
    padding: 6px 13px;
    border-radius: 6px;
    display: inline-block;
    margin-top: 9px;
}
.status-locked    { background: var(--green-lt); color: #145e30 !important; border: 2px solid #145e30 !important; font-weight: bold !important; }
.status-detecting { background: var(--amber-lt); color: #a97010 !important; border: 2px solid #a97010 !important; font-weight: bold !important; }
.status-idle      { background: var(--border2);  color: #000000 !important; border: 2px solid #000000 !important; font-weight: bold !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: bold !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 10px 0 !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    width: 100% !important;
}
.start-btn button {
    background: var(--green-lt) !important;
    color: var(--green) !important;
    border: 1.5px solid rgba(20,94,48,0.28) !important;
}
.start-btn button:hover {
    background: #d6f0e0 !important;
    border-color: var(--green) !important;
}
.stop-btn button {
    background: var(--red-lt) !important;
    color: var(--red) !important;
    border: 1.5px solid rgba(168,32,32,0.28) !important;
}
.stop-btn button:hover {
    background: #fce0e0 !important;
    border-color: var(--red) !important;
}
.speak-btn button {
    background: var(--amber-lt) !important;
    color: var(--amber) !important;
    border: 1.5px solid rgba(169,112,16,0.28) !important;
}
.speak-btn button:hover {
    background: #f9e7b5 !important;
    border-color: var(--amber-mid) !important;
}
.clear-btn button {
    background: var(--border2) !important;
    color: var(--text2) !important;
    border: 1.5px solid var(--border) !important;
}
.clear-btn button:hover {
    background: var(--border) !important;
    color: var(--text) !important;
}

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 28px !important;
    color: #000000 !important;
    font-weight: bold !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #000000 !important;
    font-weight: bold !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* Globally make sure any secondary text, captions, and slider labels are solid black and less bold */
.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
[data-testid="stWidgetLabel"] p, span[data-testid="stWidgetLabel"],
[data-testid="stSlider"] div, [data-testid="stSlider"] span, .stSlider p,
[class*="StyledThumbValue"], [class*="StyledTickBar"],
.sb-tip span, .sb-note {
    color: #000000 !important;
    font-weight: 500 !important;
}

/* Ensure strong text inside the note box remains bold */
.sb-note strong {
    font-weight: 700 !important;
}

/* Catch-all to make all sidebar text elements solid black and less bold */
[data-testid="stSidebar"] *, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] div {
    color: #000000 !important;
    font-weight: 500;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #000000 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="sb-section-label">Camera</div>', unsafe_allow_html=True)
camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
st.sidebar.markdown("""
<div class="sb-note">
  <strong style="color:#d4920e !important">0</strong> — built-in / default webcam<br>
  1, 2 … — external or additional cameras
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sb-section-label">Audio</div>', unsafe_allow_html=True)
tts_volume = st.sidebar.slider("Volume", 0.1, 1.0, 1.0, 0.05)

st.sidebar.markdown('<div class="sb-section-label">Usage Tips</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sb-tip"><span class="sb-tip-dot"></span><span>Hold device directly above Braille paper for best results</span></div>
<div class="sb-tip"><span class="sb-tip-dot"></span><span>Keep paper flat and well-lit; avoid shadows</span></div>
<div class="sb-tip"><span class="sb-tip-dot"></span><span>Lower confidence to 0.10–0.15 for faint or worn dots</span></div>
<div class="sb-tip"><span class="sb-tip-dot"></span><span>Enable Adaptive Threshold for low-contrast environments</span></div>
<div class="sb-tip"><span class="sb-tip-dot"></span><span>CLAHE + Sharpen gives best results for most lighting</span></div>
""", unsafe_allow_html=True)


# ── LOAD YOLO ─────────────────────────────────────────────────────────────────
YOLO_PATH = "weights/best.pt" if os.path.exists("weights/best.pt") else "yolov8n.pt"

@st.cache_resource(show_spinner="Loading models...")
def load_yolo(path):
    return YOLO(path)

try:
    model = load_yolo(YOLO_PATH)
except Exception as e:
    st.error(f"Cannot load YOLO: {e}")
    st.stop()

@st.cache_resource(show_spinner="Loading models...")
def load_svm():
    return joblib.load("svm_braille.pkl") if os.path.exists("svm_braille.pkl") else None


# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div>
    <div class="page-wordmark">
      <div class="page-title">Braille<em>Vision</em></div>
      <div class="page-ver">v2</div>
    </div>
    <div class="page-tagline">Real-time Braille Character Recognition</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── DETECTION & ENGINE SETTINGS ───────────────────────────────────────────────
confidence_threshold = 0.20
IOU_THRESHOLD        = 0.35
CLUSTER_DISTANCE     = 25
WORD_GAP_MULTIPLIER  = 2.0

SMOOTH_FRAMES  = 15
VOTE_THRESHOLD = 65
LOCK_FRAMES    = 6

use_clahe    = True
use_sharpen  = True
use_denoise  = False
use_gamma    = True
gamma_value  = 1.8
use_adaptive = False

use_dshow        = True
target_width     = 640
show_preprocessed= False

tts_enabled = TTS_AVAILABLE
tts_auto    = False
tts_rate    = 150
use_svm     = True

target_height = {640: 480, 1280: 720}[target_width]

svm_model = load_svm() if use_svm else None

# ── ENGINE STATUS ─────────────────────────────────────────────────────────────
yolo_pill = '<span class="pill pill-green">● YOLO v8</span>'
svm_pill  = f'<span class="pill {"pill-green" if (svm_model and use_svm) else "pill-muted"}">{"● SVM" if (svm_model and use_svm) else "○ SVM off"}</span>'
tts_pill  = f'<span class="pill {"pill-blue" if (TTS_AVAILABLE and tts_enabled) else "pill-muted"}">{"♪ TTS" if (TTS_AVAILABLE and tts_enabled) else "♪ TTS off"}</span>'
st.markdown(f'<div class="engine-strip" style="margin-bottom:14px">{yolo_pill}{svm_pill}{tts_pill}</div>',
            unsafe_allow_html=True)


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
    if not detections:
        return []
    clusters = []
    for x, label, conf in sorted(detections, key=lambda d: d[0]):
        placed = False
        for c in clusters:
            if abs(x - c[0]) < cluster_dist:
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
    avg_w  = np.mean(widths) if widths else np.mean(
        [voted_chars[i+1][0]-voted_chars[i][0] for i in range(len(voted_chars)-1)])*0.6
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
    st.session_state.detection_buffer = deque(
        st.session_state.detection_buffer, maxlen=SMOOTH_FRAMES)


# ── LAYOUT ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="medium")

with left:
    st.markdown('<div class="bv-card">', unsafe_allow_html=True)
    cam_dot = '<span class="live-dot"></span>' if st.session_state.running else '<span class="offline-dot"></span>'
    cam_status = "Live Feed" if st.session_state.running else "Camera Offline"
    st.markdown(f"""
    <div class="bv-card-header">
      <div class="bv-card-label">{cam_dot}{cam_status}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    frame_ph  = st.empty()
    status_ph = st.empty()
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        start_btn = st.button("▶  Start Camera", use_container_width=True, key="start")
        st.markdown('</div>', unsafe_allow_html=True)
    with cc2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_btn = st.button("■  Stop Camera", use_container_width=True, key="stop")
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    translation_ph = st.empty()
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    ab1, ab2 = st.columns(2)
    with ab1:
        st.markdown('<div class="speak-btn">', unsafe_allow_html=True)
        speak_btn = st.button("♪  Speak", use_container_width=True, key="speak")
        st.markdown('</div>', unsafe_allow_html=True)
    with ab2:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        clear_btn = st.button("✕  Clear", use_container_width=True, key="clear")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    chars_ph = st.empty()
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

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

if not st.session_state.running and st.session_state.get("pending_speak"):
    st.session_state.pending_speak = False
    txt = st.session_state.locked_text
    if not TTS_AVAILABLE:
        st.warning("pyttsx3 not installed — run `pip install pyttsx3`")
    elif not tts_enabled:
        st.warning("TTS is disabled.")
    elif not txt:
        st.info("No locked text to speak yet.")
    else:
        speak_text(txt, tts_rate, tts_volume)
        st.toast(f"Speaking: {txt}")


# ── IDLE STATE RENDER ─────────────────────────────────────────────────────────
def render_idle():
    frame_ph.markdown("""
    <div class="bv-card">
      <div class="cam-idle-area">
        <div class="cam-idle-glyph">⠿</div>
        <div class="cam-idle-text">Camera Offline — Press Start</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    status_ph.markdown(
        '<span class="status-bar status-idle">○  Idle</span>',
        unsafe_allow_html=True)

    locked = st.session_state.locked_text
    if locked:
        translation_ph.markdown(f"""
        <div class="bv-card" style="margin-bottom:0">
          <div class="bv-card-header">
            <div class="bv-card-label">Translation</div>
          </div>
          <div class="trans-body">
            <div class="trans-text">{locked}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        translation_ph.markdown("""
        <div class="bv-card" style="margin-bottom:0">
          <div class="bv-card-header">
            <div class="bv-card-label">Translation</div>
          </div>
          <div class="trans-body">
            <div class="trans-empty">Awaiting Braille input…</div>
          </div>
        </div>""", unsafe_allow_html=True)

    chars_ph.markdown("""
    <div class="bv-card">
      <div class="bv-card-header"><div class="bv-card-label">Detected Characters</div></div>
      <div class="chars-body" style="color:#000000 !important; font-weight: bold !important;">—</div>
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
            st.error("Camera read failed — check connection.")
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
                col = (20, 94, 48) if agreed else (169, 112, 16)
                tag = f"{final_label}{'✓' if agreed else '?'}"
            else:
                final_label = label
                col = (26, 61, 128)
                tag = final_label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(annotated, tag, (x1, max(y1-7, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            frame_dets.append((cx, final_label, conf, float(x1), float(x2)))

        clustered_frame = cluster_x_positions(
            [(d[0], d[1], d[2]) for d in frame_dets], CLUSTER_DISTANCE)

        def find_box(cx):
            best, best_dist = None, CLUSTER_DISTANCE * 0.6
            for d in frame_dets:
                dist = abs(d[0] - cx)
                if dist < best_dist:
                    best_dist = dist; best = d
            if best: return best[3], best[4]
            return cx-20, cx+20

        frame_dets_full = [(cx, lbl, conf, *find_box(cx)) for cx, lbl, conf in clustered_frame]

        if len(frame_dets_full) > 1:
            avg_w  = np.mean([d[4]-d[3] for d in frame_dets_full])
            thresh = avg_w * WORD_GAP_MULTIPLIER
            for i in range(1, len(frame_dets_full)):
                if frame_dets_full[i][0] - frame_dets_full[i-1][0] > thresh:
                    mid_x = int((frame_dets_full[i][0] + frame_dets_full[i-1][0]) / 2)
                    for yd in range(0, frame.shape[0], 12):
                        cv2.line(annotated, (mid_x, yd),
                                 (mid_x, min(yd+6, frame.shape[0])), (169, 112, 16), 1)

        cv2.putText(annotated,
                    f"{len(frame_dets_full)} cells  conf>{confidence_threshold:.2f}",
                    (8, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 80, 70), 1)

        st.session_state.detection_buffer.append(frame_dets_full)

        stable_text, _, is_stable = get_stable_text(
            st.session_state.detection_buffer, VOTE_THRESHOLD, CLUSTER_DISTANCE, WORD_GAP_MULTIPLIER)

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

        if st.session_state.get("pending_speak"):
            st.session_state.pending_speak = False
            txt = st.session_state.locked_text
            if TTS_AVAILABLE and tts_enabled and txt:
                speak_text(txt, tts_rate, tts_volume)

        # Camera feed
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
                f'<span class="status-bar status-locked">🔒 Locked  {lock_ratio}/{LOCK_FRAMES}  ·  Buffer {buf_len}/{SMOOTH_FRAMES}</span>',
                unsafe_allow_html=True)
        else:
            status_ph.markdown(
                f'<span class="status-bar status-detecting">⏳ Stabilising  {lock_ratio}/{LOCK_FRAMES}  ·  Buffer {buf_len}/{SMOOTH_FRAMES}</span>',
                unsafe_allow_html=True)

        # Translation card
        display_text = st.session_state.locked_text
        if display_text:
            translation_ph.markdown(f"""
            <div class="bv-card" style="margin-bottom:0">
              <div class="bv-card-header">
                <div class="bv-card-label">Translation</div>
              </div>
              <div class="trans-body">
                <div class="trans-text">{display_text}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            translation_ph.markdown("""
            <div class="bv-card" style="margin-bottom:0">
              <div class="bv-card-header">
                <div class="bv-card-label">Translation</div>
              </div>
              <div class="trans-body">
                <div class="trans-empty">Awaiting Braille input…</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Chars ticker
        char_html = ""
        if frame_dets_full:
            avg_w  = np.mean([d[4]-d[3] for d in frame_dets_full])
            thresh = avg_w * WORD_GAP_MULTIPLIER
            parts  = []
            for i, d in enumerate(frame_dets_full):
                if i > 0:
                    gap = frame_dets_full[i][0] - frame_dets_full[i-1][0]
                    if gap > thresh:
                        parts.append('<span class="word-mark">◆</span>')
                    else:
                        parts.append('<span class="char-sep">·</span>')
                parts.append(f'<span style="color:var(--blue)">{d[1]}</span>')
            char_html = "".join(parts)

        chars_ph.markdown(f"""
        <div class="bv-card">
          <div class="bv-card-header"><div class="bv-card-label">Detected Characters</div></div>
          <div class="chars-body">{char_html if char_html else '<span style="color:#000000 !important; font-weight: bold !important;">—</span>'}</div>
        </div>""", unsafe_allow_html=True)
