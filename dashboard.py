import streamlit as st
import cv2
import os
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

.main {
    background-color: #f5f7fa;
}

.big-box {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.translation-box {
    background: #ffffff;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1565C0;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.metric-card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Control Panel")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.45,
    0.05
)

st.sidebar.markdown("---")

st.sidebar.info("""
### Instructions

1. Start Camera
2. Show Braille Sheet
3. Watch Translation
4. Use Read Aloud Feature
""")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("👁️ BrailleVision Translation Engine")
st.caption("Real-Time AI Powered Braille to English Translation System")

st.divider()

# --------------------------------------------------
# MODEL
# --------------------------------------------------
MODEL_PATH = "weights/best.pt"

try:
    model = YOLO(MODEL_PATH)

except Exception:
    st.error("Unable to load model.")
    st.stop()

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
left, right = st.columns([2,1])

with left:
    st.subheader("🎥 Live Camera Feed")
    frame_placeholder = st.empty()

with right:
    st.subheader("📖 Translation")

    translation_placeholder = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🔤 Detected Characters")

    chars_placeholder = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        detections_metric = st.empty()

    with colB:
        confidence_metric = st.empty()

# --------------------------------------------------
# CAMERA CONTROL
# --------------------------------------------------
start = st.button("▶️ Start Camera")
stop = st.button("⏹️ Stop Camera")

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True

if stop:
    st.session_state.running = False

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
if st.session_state.running:

    cap = cv2.VideoCapture(0)

    while st.session_state.running:

        success, frame = cap.read()

        if not success:
            st.error("Camera not detected")
            break

        results = model(
            frame,
            conf=confidence_threshold,
            verbose=False
        )

        result = results[0]

        annotated_frame = result.plot()

        frame_rgb = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        detected_chars = []
        confidence_values = []

        for box in result.boxes:

            cls_id = int(box.cls[0])

            label = model.names[cls_id]

            conf = float(box.conf[0])

            x_position = float(box.xyxy[0][0])

            detected_chars.append(
                (x_position, label)
            )

            confidence_values.append(conf)

        detected_chars.sort(key=lambda x: x[0])

        final_text = "".join(
            [char for _, char in detected_chars]
        )

        avg_conf = (
            sum(confidence_values) /
            len(confidence_values)
            if confidence_values else 0
        )

        # Translation Display
        translation_placeholder.markdown(
            f"""
            <div class='translation-box'>
                {final_text if final_text else "Waiting..."}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Detected chars
        chars_placeholder.markdown(
            f"""
            <div class='big-box'>
                {' | '.join([c for _, c in detected_chars])}
            </div>
            """,
            unsafe_allow_html=True
        )

        detections_metric.metric(
            "Cells",
            len(detected_chars)
        )

        confidence_metric.metric(
            "Confidence",
            f"{avg_conf*100:.1f}%"
        )

    cap.release()