

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Leukemia Detection System", layout="centered")

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:          #f4f2ee;
    --surface:     #ffffff;
    --surface-2:   #ededea;
    --border:      #dddbd5;
    --navy:        #0f1f3d;
    --navy-light:  #1a3160;
    --text:        #1a1a1a;
    --muted:       #7a7870;
    --accent:      #c8a96e;
    --danger:      #9b1c1c;
    --danger-bg:   #fff1f0;
    --danger-brd:  #f5c2c2;
    --safe:        #0d4f2e;
    --safe-bg:     #f0faf4;
    --safe-brd:    #a7d7bc;
    --font-serif:  'DM Serif Display', Georgia, serif;
    --font-mono:   'DM Mono', monospace;
    --font-sans:   'DM Sans', sans-serif;
    --radius:      12px;
    --shadow:      0 2px 24px rgba(15,31,61,0.07);
}

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: var(--font-sans);
    color: var(--text);
}

[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 680px;
    padding: 0 1.5rem 5rem;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Top bar ── */
.topbar {
    background: var(--navy);
    margin: 0 -1.5rem 0;
    padding: 0.55rem 2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.topbar-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
}
.topbar-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: rgba(255,255,255,0.45);
    letter-spacing: 0.22em;
    text-transform: uppercase;
}

/* ── Hero block ── */
.hero {
    padding: 3rem 0 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.2rem;
}
.hero-eyebrow {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.hero-title {
    font-family: var(--font-serif);
    font-size: 2.9rem;
    color: var(--navy);
    line-height: 1.1;
    margin: 0 0 0.8rem;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-family: var(--font-sans);
    font-size: 0.9rem;
    color: var(--muted);
    font-weight: 300;
    line-height: 1.6;
    max-width: 480px;
}

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

/* ── File uploader override ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-family: var(--font-sans) !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
}

/* ── Image grid ── */
.img-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1.5rem 0;
}
.img-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.img-panel-header {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.img-panel-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
}
.img-panel-title {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
}
.img-panel-body {
    padding: 0.75rem;
}

/* ── Predict button ── */
[data-testid="stButton"] button {
    width: 100%;
    background: var(--navy) !important;
    border: none !important;
    color: #ffffff !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.18em !important;
    padding: 0.9rem 0 !important;
    border-radius: var(--radius) !important;
    text-transform: uppercase;
    transition: background 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 16px rgba(15,31,61,0.18) !important;
}
[data-testid="stButton"] button:hover {
    background: var(--navy-light) !important;
    box-shadow: 0 6px 24px rgba(15,31,61,0.28) !important;
}

/* ── Result card ── */
.result-wrap {
    margin-top: 1.8rem;
}
.result-card {
    border-radius: var(--radius);
    padding: 2rem 2rem 1.8rem;
    border: 1.5px solid;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.result-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 120px; height: 120px;
    border-radius: 50%;
    opacity: 0.06;
    transform: translate(30px, -30px);
}
.result-card.danger {
    background: var(--danger-bg);
    border-color: var(--danger-brd);
}
.result-card.danger::after { background: var(--danger); }
.result-card.safe {
    background: var(--safe-bg);
    border-color: var(--safe-brd);
}
.result-card.safe::after { background: var(--safe); }

.result-badge {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.result-badge.danger { color: var(--danger); }
.result-badge.safe   { color: var(--safe); }

.result-label {
    font-family: var(--font-serif);
    font-size: 2.6rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-label.danger { color: var(--danger); }
.result-label.safe   { color: var(--safe); }

.result-conf {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    opacity: 0.6;
}
.result-conf.danger { color: var(--danger); }
.result-conf.safe   { color: var(--safe); }

.result-note {
    margin-top: 1.2rem;
    padding-top: 1.2rem;
    border-top: 1px solid;
    font-size: 0.78rem;
    font-family: var(--font-sans);
    font-weight: 300;
    line-height: 1.5;
    opacity: 0.65;
}
.result-note.danger { border-color: var(--danger-brd); color: var(--danger); }
.result-note.safe   { border-color: var(--safe-brd);   color: var(--safe); }

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    color: var(--navy) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em;
}

/* ── Streamlit image ── */
[data-testid="stImage"] img {
    border-radius: 6px;
    width: 100% !important;
    height: auto;
}

/* ── Error / warning ── */
[data-testid="stAlertContainer"] {
    border-radius: var(--radius) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
}

/* ── Footer strip ── */
.footer {
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists("leukemia_best_model.h5"):
        return None
    return tf.keras.models.load_model("leukemia_best_model.h5")


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img)

    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.abs(sobel_x) + np.abs(sobel_y)
    edges = edges / 255.0
    img_input = edges[np.newaxis, :, :, np.newaxis]

    return img_input, edges


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

def main():

    # Top bar
    st.markdown("""
    <div class="topbar">
        <div class="topbar-dot"></div>
        <div class="topbar-label">Clinical AI · Haematology</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Blood Cell Analysis</div>
        <div class="hero-title">Leukemia<br><em>Detection</em></div>
        <div class="hero-sub">
            Upload a microscopic blood cell image. The model applies Sobel edge 
            detection and classifies the sample as leukemic or healthy using a 
            trained convolutional neural network.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model = load_model()
    if model is None:
        st.error("Model file not found — place leukemia_best_model.h5 in this directory.")
        st.stop()

    # Upload
    st.markdown('<div class="section-label">Upload Sample</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "upload",
        label_visibility="collapsed",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_input, edge_img = preprocess_image(image)

        # Image panels
        st.markdown('<div class="img-grid">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="img-panel">
                <div class="img-panel-header">
                    <div class="img-panel-dot"></div>
                    <div class="img-panel-title">Original Sample</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("""
            <div class="img-panel">
                <div class="img-panel-header">
                    <div class="img-panel-dot"></div>
                    <div class="img-panel-title">Edge Detection</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(edge_img, use_container_width=True, clamp=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Predict button
        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Analysing sample..."):
                prediction = model.predict(img_input, verbose=0)[0][0]
                confidence = float(prediction)

                if confidence >= 0.5:
                    label       = "Leukemia Detected"
                    conf_val    = confidence * 100
                    cls         = "danger"
                    badge       = "⚕  Positive Finding"
                    note        = (
                        "This result indicates potential leukemic markers. "
                        "Please consult a qualified haematologist for clinical confirmation."
                    )
                else:
                    label       = "No Leukemia"
                    conf_val    = (1 - confidence) * 100
                    cls         = "safe"
                    badge       = "✓  Negative Finding"
                    note        = (
                        "No leukemic markers detected in this sample. "
                        "Regular screening is still recommended by a medical professional."
                    )

            st.markdown(f"""
            <div class="result-wrap">
                <div class="result-card {cls}">
                    <div class="result-badge {cls}">{badge}</div>
                    <div class="result-label {cls}">{label}</div>
                    <div class="result-conf {cls}">Confidence — {conf_val:.1f}%</div>
                    <div class="result-note {cls}">{note}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
