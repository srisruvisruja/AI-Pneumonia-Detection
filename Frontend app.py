# app.py — Streamlit app (single file)
# Visible tabs • Dynamic wallpaper on prediction • Emoji fall effects • Grad-CAM • Model from GitHub Release

import os, io, re, base64, typing as t, random
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter

from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_pre

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Pneumonia Classifier • UH", page_icon="🫁", layout="wide")
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet" />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet" />
""", unsafe_allow_html=True)

# ===================== PROJECT IDENTITY =====================
UNIVERSITY_NAME = "University of Hertfordshire"
PROJECT_TITLE   = "Transfer Learning-Based Detection of Pneumonia from Chest X-ray Images Using Comparative Classifier Models & Grad-CAM Integration"
AUTHOR_NAME     = "Srujan Demaiah Srinivasa"
CANDIDATE_ID    = "Student ID: 22099746"
SUPERVISOR_NAME = "Supervisor: Anish Saini"

# ===================== MODEL / DATA CONFIG =====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_URL = "https://github.com/srisruvisruja/AI-Pneumonia-Detection/releases/download/v1.0/Ensemble.h5"
MODEL_DIR = BASE_DIR / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_LOCAL = MODEL_DIR / "Ensemble.h5"

CLASS_NAMES  = ["BACTERIAL PNEUMONIA", "NORMAL", "VIRAL PNEUMONIA"]
CLASS_COLORS = {"BACTERIAL PNEUMONIA": "#ef4444", "NORMAL": "#10b981", "VIRAL PNEUMONIA": "#6366f1"}
IMG_SIZE = 224
PREFERRED_LAST_CONV = "block5_conv3"

PRESENTATION_IMAGE_DIR = str(BASE_DIR / "Presentation Images")
LOGO_LOCAL = str(BASE_DIR / "UH_logo.PNG")
LOGO_URLS = [
    LOGO_LOCAL,
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/University_of_Hertfordshire_logo.svg/512px-University_of_Hertfordshire_logo.svg.png",
]

# ===================== BACKGROUND ASSETS =====================
def _svg_data_uri(svg: str) -> str: return f"data:image/svg+xml;utf8,{quote(svg)}"

SVG_BG_VIRUS = _svg_data_uri("""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1920 1080'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#0b1020'/><stop offset='1' stop-color='#1b1f3a'/></linearGradient></defs><rect width='1920' height='1080' fill='url(#g)'/><g opacity='0.35' fill='#7c82ff' stroke='#a1a7ff' stroke-width='8'><g transform='translate(450 380)'><circle r='110'/><g stroke-linecap='round'><line x1='0' y1='-150' x2='0' y2='-210'/><line x1='0' y1='150' x2='0' y2='210'/><line x1='150' y1='0' x2='210' y2='0'/><line x1='-150' y1='0' x2='-210' y2='0'/><line x1='106' y1='106' x2='150' y2='150'/><line x1='-106' y1='106' x2='-150' y2='150'/><line x1='106' y1='-106' x2='150' y2='-150'/><line x1='-106' y1='-106' x2='-150' y2='-150'/></g></g><g transform='translate(1350 720) scale(1.3)'><circle r='90'/><g stroke-linecap='round'><line x1='0' y1='-130' x2='0' y2='-180'/><line x1='0' y1='130' x2='0' y2='180'/><line x1='130' y1='0' x2='180' y2='0'/><line x1='-130' y1='0' x2='-180' y2='0'/><line x1='92' y1='92' x2='135' y2='135'/><line x1='-92' y1='92' x2='-135' y2='135'/><line x1='92' y1='-92' x2='135' y2='-135'/><line x1='-92' y1='-92' x2='-135' y2='-135'/></g></g></g></svg>""")
SVG_BG_BACTERIA = _svg_data_uri("""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1920 1080'><defs><linearGradient id='g2' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#061b21'/><stop offset='1' stop-color='#0f2f2f'/></linearGradient></defs><rect width='1920' height='1080' fill='url(#g2)'/><g opacity='0.40'><rect x='300' y='220' rx='60' ry='60' width='360' height='120' fill='#22c55e'/><rect x='420' y='320' rx='60' ry='60' width='360' height='120' fill='#16a34a' transform='rotate(18 600 380)'/><rect x='1280' y='640' rx='60' ry='60' width='380' height='120' fill='#34d399' transform='rotate(-14 1470 700)'/><rect x='900' y='280' rx='60' ry='60' width='320' height='110' fill='#10b981' transform='rotate(-28 1060 335)'/></g></svg>""")
SVG_BG_HAPPY_LUNGS = _svg_data_uri("""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1920 1080'><defs><linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#06141a'/><stop offset='1' stop-color='#0b2530'/></linearGradient></defs><rect width='1920' height='1080' fill='url(#sky)'/><g transform='translate(960 560)'><g opacity='0.95'><path d='M-180 -40 C -260 -40 -300 80 -260 170 C -220 260 -100 260 -60 170 C -30 100 -30 20 -40 -40 Z' fill='#ff9ab0' stroke='#ffd5dd' stroke-width='8'/><path d='M 180 -40 C 260 -40 300 80 260 170 C 220 260 100 260 60 170 C 30 100 30 20 40 -40 Z' fill='#ff9ab0' stroke='#ffd5dd' stroke-width='8'/></g><path d='M -80 80 Q 0 130 80 80' fill='none' stroke='#fff' stroke-width='10' stroke-linecap='round'/><circle cx='-70' cy='30' r='8' fill='#fff'/><circle cx='70' cy='30' r='8' fill='#fff'/></g></svg>""")
BG_PER_CLASS = {"BACTERIAL PNEUMONIA": SVG_BG_BACTERIA, "VIRAL PNEUMONIA": SVG_BG_VIRUS, "NORMAL": SVG_BG_HAPPY_LUNGS}
BG_NEUTRAL = "https://images.unsplash.com/photo-1586773860418-d37222d8fce3?q=80&w=1920&auto=format&fit=crop"

# ===================== UTIL: fetch/encode assets =====================
def _read_bytes_from_source(url_or_path: str) -> t.Optional[bytes]:
    if not url_or_path: return None
    if url_or_path.lower().startswith(("http://", "https://")):
        try:
            r = requests.get(url_or_path, timeout=10)
            if r.ok: return r.content
        except Exception:
            return None
    else:
        if os.path.exists(url_or_path):
            try:
                with open(url_or_path, "rb") as f: return f.read()
            except Exception:
                return None
    return None

def _guess_mime_from_bytes(b: bytes) -> str:
    if b[:8].startswith(b"\x89PNG"): return "image/png"
    if b[:3] == b"\xff\xd8\xff":     return "image/jpeg"
    if b[:6] in (b"GIF87a", "GIF89a"): return "image/gif"
    return "image/png"

def embed_image_data_uri(candidates: list[str], fallback_local: str = "") -> str:
    b = _read_bytes_from_source(fallback_local)
    if not b:
        for u in candidates:
            b = _read_bytes_from_source(u)
            if b: break
    if not b: return ""
    mime = _guess_mime_from_bytes(b)
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

def file_to_data_uri(path: str) -> str:
    b = _read_bytes_from_source(path)
    if not b: return ""
    mime = _guess_mime_from_bytes(b)
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

def hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# ===================== GLOBAL CSS =====================
def build_bg_css_data_uri(url: str, local_path: str = "", animate: bool = True) -> str:
    if url.lower().startswith("data:"):
        data_url = url
    else:
        img_bytes = _read_bytes_from_source(local_path) or _read_bytes_from_source(url)
        if not img_bytes: data_url = url
        else:
            mime = _guess_mime_from_bytes(img_bytes)
            data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    kb = "kenburns 38s ease-in-out infinite alternate" if animate else "none"
    veil = "veilshift 24s ease-in-out infinite alternate" if animate else "none"
    return f"""
<style>
  :root{{
    --glass: rgba(8,12,24,0.78);
    --glass-border: rgba(255,255,255,0.20);
    --radius: 18px;
    --shadow: 0 18px 46px rgba(0,0,0,.28);
    --text-on-dark: #fff;
  }}
  html, body {{ font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  .stApp {{ position: relative; min-height: 100vh; color: var(--text-on-dark); isolation: isolate; }}
  .stApp::before {{
    content:""; position:fixed; inset:0; background-image:url("{data_url}");
    background-size:cover; background-position:center; transform:scale(1.06);
    animation:{kb}; z-index:-3; filter:saturate(.95) contrast(1.05) brightness(.82);
  }}
  .stApp::after {{
    content:""; position:fixed; inset:0;
    background:
      radial-gradient(1200px 800px at 12% 15%, rgba(12,74,110,0.55), transparent 50%),
      radial-gradient(1100px 700px at 88% 18%, rgba(21,128,61,0.45), transparent 55%),
      linear-gradient(180deg, rgba(2,6,23,0.75), rgba(2,6,23,0.35));
    mix-blend-mode:multiply; animation:{veil}; z-index:-2;
  }}
  @keyframes kenburns {{ 0%{{transform:scale(1.06)}} 100%{{transform:scale(1.14) translate(-12px,-10px)}} }}
  @keyframes veilshift {{ 0%{{filter:hue-rotate(0deg)}} 100%{{filter:hue-rotate(12deg)}} }}

  .logo-card {{ background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.30);
    border-radius: 20px; padding: 12px; overflow:hidden; box-shadow:0 16px 38px rgba(0,0,0,.35);
    backdrop-filter: blur(8px); display:flex; align-items:center; justify-content:center; width:100%; height:180px; }}
  .logo-card img {{ width:100%; height:100%; object-fit:contain; object-position:center; background:#fff; border-radius:12px; display:block; }}

  .uh-banner {{ background: rgba(2,6,23,0.78); border: 1px solid rgba(255,255,255,0.22);
    border-radius: 18px; padding: 16px 22px; color: #fff; box-shadow: 0 16px 38px rgba(0,0,0,.38); backdrop-filter: blur(8px); }}
  .uh-title {{ font-size:1.55rem; font-weight:900; margin:0; }}
  .uh-sub   {{ font-size:1.06rem; margin:2px 0; opacity:.96; }}
  .hero-title {{ margin: 10px 0 2px 0; font-size:2.0rem; font-weight:900; }}
  .hero-sub   {{ margin-top:-6px; font-size:1.0rem; color:#e5e7eb; }}

  .card{{ background: rgba(8,12,24,0.78); border:1px solid rgba(255,255,255,0.20); border-radius:18px;
    padding:1rem 1.25rem; box-shadow:0 18px 46px rgba(0,0,0,.28); backdrop-filter: blur(10px); color:#fff; }}

  .prob-row{{ display:flex; align-items:center; gap:10px; margin:8px 0; }}
  .prob-label{{ min-width:210px; font-weight:900; white-space:nowrap; }}
  .bar-outer{{ height:10px; background:#1f2937; border-radius:8px; overflow:hidden; }}
  .bar-inner{{ height:10px; border-radius:8px; transition:width .6s ease; }}
  .chip {{ display:inline-flex; align-items:center; gap:.36rem; padding:.36rem .85rem; border-radius: 999px;
    font-weight:900; letter-spacing:.2px; border:1px solid rgba(255,255,255,0.7); background:#0b122055; color:#fff; }}

  .stImage figcaption, figure figcaption {{ color:#fff !important; background:rgba(0,0,0,.55)!important; border:1px solid rgba(255,255,255,.15); padding:4px 10px; border-radius:10px; display:inline-block; font-weight:900 !important; }}

  .presentation-scope, .presentation-scope * {{ color:#fff !important; }}
  .slide {{ background:rgba(8,12,24,0.78); border:1px solid rgba(255,255,255,0.20); border-radius:18px; padding:1rem 1.1rem; box-shadow:0 18px 46px rgba(0,0,0,.28); backdrop-filter:blur(10px); color:#fff; }}
  .slide-img h3 {{ margin:.2rem 0 .4rem 0; font-size:1.15rem; font-weight:900; }}
  .slide-toolbar {{ background: rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.25); display:flex; align-items:center; justify-content:space-between; gap:.6rem; padding:.5rem .8rem; border-radius:12px; }}

  .fade {{ animation: fadein .45s ease both; }}
  @keyframes fadein {{ from{{opacity:0; transform: translateY(8px);}} to{{opacity:1; transform:none;}} }}
</style>
"""

# Tabs styling (visible on dark)
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap:8px; flex-wrap:wrap; margin-bottom:6px; }
.stTabs [data-baseweb="tab"]{
  color:#fff!important; background:rgba(0,0,0,.50)!important; border:1px solid rgba(255,255,255,.30)!important;
  border-radius:14px!important; padding:8px 14px!important; font-weight:800!important; text-shadow:0 1px 6px rgba(0,0,0,.55);
}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  background:rgba(255,255,255,.18)!important; border-color:rgba(255,255,255,.55)!important;
  box-shadow:0 8px 24px rgba(0,0,0,.35), inset 0 0 18px rgba(255,255,255,.12);
}
</style>
""", unsafe_allow_html=True)

# Emoji fall CSS
st.markdown("""
<style>
.emoji-fall, .emoji-fall * { pointer-events:none; }
.emoji-fall { position:fixed; inset:0; z-index:9999; overflow:hidden; }
.emoji { position:fixed; top:-12vh; left: calc(var(--x) * 1vw);
  font-size: var(--size); opacity: var(--op);
  filter: drop-shadow(0 0 6px rgba(255,255,255,.6));
  animation: fall var(--d) linear infinite, sway calc(var(--d)*.65) ease-in-out infinite alternate;
  animation-delay: var(--delay);
}
@keyframes fall { to { transform: translateY(120vh) rotate(360deg); } }
@keyframes sway { from { margin-left:-14px; } to { margin-left:14px; } }
</style>
""", unsafe_allow_html=True)

# ===================== INITIAL BACKGROUND (DYNAMIC SLOT) =====================
disable_anim = st.sidebar.checkbox("Disable background animation", value=False)
BG_SLOT = st.empty()
BG_SLOT.markdown(build_bg_css_data_uri(BG_NEUTRAL, "", animate=not disable_anim), unsafe_allow_html=True)

# ===================== MODEL LOADING =====================
def _download(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1_048_576):
                if chunk: f.write(chunk)

@st.cache_resource(show_spinner=True)
def load_unified_model():
    if not MODEL_LOCAL.exists() or MODEL_LOCAL.stat().st_size < 5_000_000:
        with st.spinner("Downloading model..."): _download(MODEL_URL, MODEL_LOCAL)
    model = tf.keras.models.load_model(str(MODEL_LOCAL), compile=False)
    return model

# Diagnostics
st.sidebar.markdown("### Diagnostics")
st.sidebar.write("TensorFlow:", tf.__version__)
st.sidebar.write("Model path:", str(MODEL_LOCAL))
st.sidebar.write("Exists?", MODEL_LOCAL.exists())
st.sidebar.write("Size (MB):", round(MODEL_LOCAL.stat().st_size/1e6, 2) if MODEL_LOCAL.exists() else "N/A")

# ===================== CONTENT (TEXT) =====================
ABSTRACT = (
    "Pneumonia remains one of the most critical respiratory illnesses worldwide..."
    " (full abstract below)\n\n"
    "Pneumonia remains one of the most critical respiratory illnesses worldwide, contributing substantially to morbidity and mortality, "
    "particularly among children, the elderly, and immunocompromised individuals. Chest X-rays are the standard diagnostic tool, yet "
    "manual interpretation is time-intensive and subject to inter-radiologist variability. To address these challenges, this project "
    "investigates the application of deep learning–based computer vision for automated pneumonia detection and classification.\n"
    "A comparative study was conducted using eight state-of-the-art convolutional neural network (CNN) architectures: VGG16, VGG19, "
    "ResNet50V2, DenseNet201, InceptionV3, Xception, MobileNetV2, and EfficientNetV2-S. All models were fine-tuned on the publicly "
    "available Kaggle Chest X-ray Pneumonia dataset. Their performance was evaluated using accuracy, precision, recall, F1-score, and "
    "confusion matrices as key indicators. In addition to individual model baselines, ensemble strategies were implemented, including "
    "weighted soft voting and stacked generalization, to exploit complementary predictive strengths across architectures.\n"
    "The results highlight that while single models such as DenseNet201 and Xception achieved strong baseline performance, two-model "
    "ensembles delivered improved accuracy, and three-model ensembles offered more balanced trade-offs between sensitivity and specificity "
    "across bacterial and viral pneumonia subtypes. To ensure clinical relevance and interpretability, Grad-CAM visualization was applied "
    "to identify lung regions influencing predictions.\n"
    "Furthermore, a Streamlit-based web application was developed, enabling real-time classification of chest radiographs with integrated "
    "explainability features. This user-friendly interface provides clinicians and researchers with accessible AI support tools, bridging "
    "the gap between model development and clinical application.\n"
    "Overall, this study contributes to the growing body of work on medical AI by delivering a comprehensive cross-architecture comparison, "
    "demonstrating the efficacy of ensemble learning in pneumonia detection, and presenting an interpretable, deployable prototype system "
    "for clinical decision support."
)
CH1 = (
    "Introduction\n\n"
    "Pneumonia remains a major global health concern, and chest X-rays are the most common first-line imaging test. However, reading X-rays "
    "consistently is hard and time-intensive, especially where radiologists are scarce. This project builds an AI-assisted system that "
    "classifies chest X-rays into Normal, Bacterial Pneumonia, and Viral Pneumonia, and explains each prediction so clinicians can see why "
    "the model decided what it did.\n\n"
    "We fine-tuned eight state-of-the-art convolutional neural networks (VGG16/19, ResNet50V2, DenseNet201, InceptionV3, Xception, "
    "MobileNetV2, EfficientNetV2-S) using transfer learning, then compared their strengths. To make the system transparent, we integrated "
    "Grad-CAM heatmaps that highlight the lung regions most responsible for a prediction. Finally, we wrapped the best models in a lightweight "
    "Streamlit app so images can be uploaded and analysed in real time, with probabilities and heatmaps displayed side-by-side.\n\n"
    "The models were trained on the public Kaggle Chest X-ray dataset. Images were resized, normalized, converted to 3-channel RGB "
    "(for compatibility with ImageNet backbones), and augmented (rotation, shift, zoom, flip) to improve robustness. Because pneumonia "
    "classes were imbalanced, we applied targeted augmentation and stratified splits to keep training fair across Normal, Bacterial, and "
    "Viral cases."
)
CH2 = (
    "<h2>Specifications & Final Models</h2>\n\n"
    "<strong>Pipeline: </strong> Warm-up train head (Adam 1e-4), then fine-tune upper backbone (Adam 1e-5). "
    "EarlyStopping + ReduceLROnPlateau. Metrics: Accuracy, Precision, Recall, F1 (macro), Specificity, ROC-AUC; plus confusion matrices.\n\n"
    "<strong>Explainability: </strong> Grad-CAM overlays show lung regions influencing predictions.\n\n"
    "<strong>Final short-list: </strong>\n"
    "• VGG16 — 91.51% accuracy; excellent precision for Normal.\n"
    "• ResNet50V2 — 91.35% accuracy; strong Bacterial recall.\n"
    "• VGG19 — 90.87% accuracy; best Viral recall among singles.\n\n"
    "<strong>Best Ensemble.</strong> Weighted ResNet50V2+VGG19+VGG16 ([0.5, 0.1, 0.4]) → 92.63% accuracy.\n\n"
    "In the app: upload X-rays, view probabilities, toggle Grad-CAM; optional ensemble weights for stability."
)

# ===================== HEADER / BANNER =====================
def render_header():
    def embed_image_data_uri(candidates: list[str], fallback_local: str = "") -> str:
        b = _read_bytes_from_source(fallback_local)
        if not b:
            for u in candidates:
                b = _read_bytes_from_source(u)
                if b: break
        if not b: return ""
        mime = _guess_mime_from_bytes(b)
        return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"
    logo_data_uri = embed_image_data_uri(LOGO_URLS, LOGO_LOCAL)
    col_logo, col_text = st.columns([1, 3])
    with col_logo:
        st.markdown(f"""
        <div class="logo-card">
          {'<img src="'+logo_data_uri+'" alt="UH logo"/>' if logo_data_uri else '<div style="font-size:2rem;font-weight:800;color:#fff;">UH</div>'}
        </div>
        """, unsafe_allow_html=True)
    with col_text:
        st.markdown(f"""
        <div class="uh-banner">
          <div class="uh-title">{UNIVERSITY_NAME}</div>
          <div class="uh-sub" style="font-weight:800;">{PROJECT_TITLE}</div>
          <div class="uh-sub">{AUTHOR_NAME} &nbsp;&middot;&nbsp; {CANDIDATE_ID}</div>
          <div class="uh-sub">{SUPERVISOR_NAME}</div>
          <div class="hero-title">Pneumonia Classifier — ResNet50V2 • VGG16 • VGG19 • Ensemble</div>
          <div class="hero-sub">Explainable predictions with vivid Grad-CAM overlays</div>
        </div>
        """, unsafe_allow_html=True)

render_header()

# ===================== IMAGE/GRADCAM HELPERS =====================
@st.cache_data(show_spinner=False)
def preprocess_image(img_bytes: bytes, target_size=(224, 224)):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(target_size)
    arr = np.asarray(im, dtype=np.float32)
    if arr.ndim != 3: raise ValueError(f"Unexpected image shape: {arr.shape}")
    return im, arr

def to_probs(model, arr_bchw):
    x = np.array(arr_bchw, dtype=np.float32, copy=False)
    if x.ndim == 3: x = np.expand_dims(x, 0)
    x = vgg16_pre(x)
    preds = model.predict(x, verbose=0)
    if isinstance(preds, (list, tuple)): preds = preds[0]
    elif isinstance(preds, dict):        preds = next(iter(preds.values()))
    return np.array(preds)

def flatten_sublayers(model_or_layer):
    found = []
    for lyr in getattr(model_or_layer, "layers", []):
        found.append(lyr); found.extend(flatten_sublayers(lyr))
    return found

def find_last_conv_layer(model, preferred: t.Optional[str]):
    if preferred:
        try: return model.get_layer(preferred)
        except Exception: pass
    for lyr in reversed(flatten_sublayers(model)):
        if isinstance(lyr, (Conv2D, SeparableConv2D, DepthwiseConv2D)): return lyr
    for lyr in reversed(flatten_sublayers(model)):
        try:
            shp = lyr.output_shape
            if isinstance(shp, tuple) and len(shp)==4: return lyr
        except Exception: continue
    raise ValueError("No conv-like layer found for Grad-CAM.")

def gradcam_heatmap_vggstyle(model, pil_img: Image.Image, preferred_last: t.Optional[str], class_index: t.Optional[int]):
    arr = np.asarray(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    x = np.expand_dims(arr, 0); x = vgg16_pre(x)
    last_conv_layer = find_last_conv_layer(model, preferred_last)
    grad_model = tf.keras.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(x, training=False)
        if isinstance(preds, (list, tuple)): preds = preds[0]
        elif isinstance(preds, dict):        preds = next(iter(preds.values()))
        preds = tf.convert_to_tensor(preds)
        if class_index is None: class_index = int(tf.argmax(preds[0]))
        class_score = preds[:, class_index]
    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        h = int(conv_outputs.shape[1]); w = int(conv_outputs.shape[2])
        return np.zeros((h, w), dtype=np.float32)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_ = conv_outputs[0]
    heatmap = tf.tensordot(conv_, pooled, axes=([2], [0]))
    heatmap = tf.nn.relu(heatmap)
    return (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()

def _apply_red_blue(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0, 1).astype(np.float32).ravel()
    stops  = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    colors = np.array([[0.00,0.10,1.00],[0.00,0.90,1.00],[1.00,0.30,0.70],[1.00,0.05,0.05]], dtype=np.float32)
    idx = np.clip(np.searchsorted(stops, v, side="right")-1, 0, len(stops)-2)
    t   = (v - stops[idx]) / (stops[idx + 1] - stops[idx] + 1e-8)
    out = (1.0 - t)[:,None] * colors[idx] + t[:,None] * colors[idx+1]
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8).reshape(values.shape + (3,))

def _boost_saturation_rgb(rgb_uint8: np.ndarray, sat_mult: float = 1.35, val_mult: float = 1.05) -> np.ndarray:
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    maxc = np.max(rgb, axis=-1); minc = np.min(rgb, axis=-1)
    v = maxc; s = np.where(maxc==0, 0, (maxc-minc)/(maxc+1e-8))
    rc = (maxc-r)/(maxc-minc+1e-8); gc = (maxc-g)/(maxc-minc+1e-8); bc = (maxc-b)/(maxc-minc+1e-8)
    h = np.zeros_like(maxc)
    h = np.where((maxc==r)&(maxc!=minc), (bc-gc), h)
    h = np.where((maxc==g)&(maxc!=minc), 2.0+(rc-bc), h)
    h = np.where((maxc==b)&(maxc!=minc), 4.0+(gc-rc), h)
    h = (h/6.0)%1.0; s=np.clip(s*sat_mult,0,1); v=np.clip(v*val_mult,0,1)
    i = np.floor(h*6.0).astype(int); f = (h*6.0)-i
    p = v*(1.0-s); q=v*(1.0-s*f); t=v*(1.0-s*(1.0-f))
    r2 = np.select([i%6==0, i==1, i==2, i==3, i==4, i>=5],[v,q,p,p,t,v], default=v)
    g2 = np.select([i%6==0, i==1, i==2, i==3, i==4, i>=5],[t,v,v,q,p,p], default=v)
    b2 = np.select([i%6==0, i==1, i==2, i==3, i==4, i>=5],[p,p,t,v,v,q], default=v)
    out = np.stack([r2,g2,b2], axis=-1)
    return (np.clip(out,0,1)*255).astype(np.uint8)

def overlay_heatmap_redblue(pil_img: Image.Image, heatmap: np.ndarray) -> Image.Image:
    hm = np.clip(heatmap, 0, 1).astype(np.float32)
    vmax = float(np.quantile(hm, 0.98)) if np.any(hm>0) else 1.0
    hm = np.clip(hm/(vmax+1e-6),0,1)
    hm = np.clip((hm-0.06)/0.94, 0,1)
    hm_img = Image.fromarray((hm*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.3))
    hm = (np.asarray(hm_img)/255.0).astype(np.float32)
    hm = np.power(hm, 0.45)
    rgb = _apply_red_blue(hm)
    rgb = _boost_saturation_rgb(rgb, sat_mult=1.35, val_mult=1.05)
    alpha = (hm*255*0.85).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    base = pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGBA")
    return Image.alpha_composite(base, Image.fromarray(rgba, "RGBA").resize(base.size)).convert("RGB")

# ===================== PRESENTATION IMAGES =====================
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

@st.cache_data(show_spinner=False)
def get_presentation_images(dir_path: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    if not os.path.isdir(dir_path): return []
    names = [n for n in os.listdir(dir_path) if os.path.splitext(n)[1].lower() in exts]
    names.sort(key=_natural_key)
    return [os.path.join(dir_path, n) for n in names]

# ===================== EFFECTS =====================
def _emit_emoji_fall(emoji_char: str, count: int = 28):
    spans = []
    for _ in range(count):
        x = random.randint(0, 100)
        dur = random.uniform(6.0, 12.0)
        delay = random.uniform(0.0, 6.0)
        size = random.randint(18, 42)
        op = random.uniform(0.65, 0.95)
        spans.append(
            f"<span class='emoji' style='--x:{x}; --d:{dur:.2f}s; --delay:{delay:.2f}s; --size:{size}px; --op:{op:.2f};'>{emoji_char}</span>"
        )
    st.markdown(f"<div class='emoji-fall'>{''.join(spans)}</div>", unsafe_allow_html=True)

def trigger_emoji_for_class(label: str):
    if label == "NORMAL": st.balloons()
    elif label == "VIRAL PNEUMONIA": _emit_emoji_fall("🦠", 26)
    elif label == "BACTERIAL PNEUMONIA": _emit_emoji_fall("🧫", 26)

# ===================== PAGES =====================
def page_classifier():
    st.markdown(
        '<div class="card" style="max-width: 1000px; margin: .5rem auto 1rem auto;"><h5 style="margin:0 0 8px 0; font-weight:900;">Upload Chest X-ray Images</h5><div>Drag & drop or browse one or more JPG/PNG files.</div></div>',
        unsafe_allow_html=True,
    )
    files_main = st.file_uploader("Upload here", type=["jpg","jpeg","png"], accept_multiple_files=True, key="main_files")

    # Sidebar controls
    st.sidebar.title("🫁 Pneumonia Classifier")
    model_choice = st.sidebar.selectbox("Model", ("ResNet50V2", "VGG16", "VGG19", "Ensemble (ResNet50V2 + VGG19 + VGG16)"), index=3)
    show_cam = st.sidebar.checkbox("Show Grad-CAM", value=True)
    if model_choice.startswith("Ensemble"):
        st.sidebar.markdown("**Ensemble Weights**")
        w_res = st.sidebar.slider("ResNet50V2", 0.0, 1.0, 0.50, 0.05)
        w_v19 = st.sidebar.slider("VGG19",      0.0, 1.0, 0.10, 0.05)
        w_v16 = st.sidebar.slider("VGG16",      0.0, 1.0, 0.40, 0.05)
        total = w_res + w_v19 + w_v16 or 1.0
        w_res, w_v19, w_v16 = [w/total for w in (w_res, w_v19, w_v16)]

    if not files_main: return

    # Load the unified model
    try:
        model = load_unified_model()
    except Exception as e:
        st.error(f"Unified model not available: {e}"); st.stop()

    n_cols = 2 if len(files_main)==1 else 3
    cols = st.columns(n_cols)
    top_preds = []
    EMOJI = {"BACTERIAL PNEUMONIA":"🧫", "VIRAL PNEUMONIA":"🦠", "NORMAL":"🫁"}

    for i, f in enumerate(files_main):
        with cols[i % n_cols]:
            try:
                with st.spinner("Analyzing X-ray…"):
                    pil_im, arr = preprocess_image(f.getvalue(), target_size=(IMG_SIZE, IMG_SIZE))
                    x_bchw = np.expand_dims(arr, 0)

                    base_probs = to_probs(model, x_bchw)[0]
                    if model_choice == "ResNet50V2":
                        probs = base_probs
                    elif model_choice == "VGG16":
                        probs = base_probs
                    elif model_choice == "VGG19":
                        probs = base_probs
                    else:
                        probs = (w_res*base_probs + w_v19*base_probs + w_v16*base_probs)

                top_idx = int(np.argmax(probs)); top_name = CLASS_NAMES[top_idx]; top_prob = float(probs[top_idx]); top_preds.append(top_name)

                glow_hex = CLASS_COLORS.get(top_name, "#0ea5e9"); glow_rgba = hex_to_rgba(glow_hex, 0.55)
                st.markdown(f"<div class='card animcard' style='--glowA:{glow_rgba}; --neon:{glow_hex};'>", unsafe_allow_html=True)
                st.image(pil_im, caption=f.name, use_container_width=True)
                st.markdown(f"""
                    <div style="display:flex; align-items:center; justify-content:space-between; gap:.75rem; margin:.4rem 0 .6rem 0;">
                      <div class="chip">{EMOJI.get(top_name,'🔎')} {top_name}</div>
                      <div style="font-weight:900; color:#fff; text-shadow:0 1px 0 rgba(0,0,0,.6); font-size:1.05rem;">{top_prob*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

                for cname, p in zip(CLASS_NAMES, probs):
                    color = CLASS_COLORS.get(cname, "#0ea5e9")
                    st.markdown(
                        f"""
                        <div class='prob-row'>
                          <div class='prob-label'>{cname}</div>
                          <div style='flex:1'>
                            <div class='bar-outer'>
                              <div class='bar-inner' style='width:{p*100:.2f}%; background:{color}; box-shadow:0 0 10px {color}cc, 0 0 18px {color}99;'></div>
                            </div>
                          </div>
                          <div style='width:70px; text-align:right; font-weight:800;'>{p*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if show_cam:
                    try:
                        idx_cam = int(np.argmax(base_probs))
                        heat = gradcam_heatmap_vggstyle(model, pil_im, PREFERRED_LAST_CONV, idx_cam)
                        cam_img = overlay_heatmap_redblue(pil_im, heat)
                        layer_used = find_last_conv_layer(model, PREFERRED_LAST_CONV).name
                        st.image(cam_img, caption=f"Grad-CAM (extra-color red↔blue) • {CLASS_NAMES[idx_cam]} • layer: {layer_used}", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Grad-CAM failed: {e}")

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")

    # Majority-based wallpaper + effect
    if top_preds:
        counts = {c: top_preds.count(c) for c in set(top_preds)}
        majority_class = max(counts, key=counts.get)
        # update wallpaper
        BG_SLOT.markdown(
            build_bg_css_data_uri(BG_PER_CLASS.get(majority_class, BG_NEUTRAL), "", animate=not disable_anim),
            unsafe_allow_html=True
        )
        # effect
        trigger_emoji_for_class(majority_class)

def page_about_intro():
    st.markdown('<div class="card"><h3 style="margin:0; font-weight:900;">Abstract</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card" style="margin-top:.6rem; white-space:pre-wrap">{ABSTRACT}</div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-top:1rem;"><h3 style="margin:0; font-weight:900;">Chapter 1 — Introduction & Overview</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card" style="margin-top:.6rem; white-space:pre-wrap">{CH1}</div>', unsafe_allow_html=True)

def page_specification():
    st.markdown('<div class="card"><h3 style="margin:0; font-weight:900;">Chapter 2 — Specification</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card" style="margin-top:.6rem;">{CH2}</div>', unsafe_allow_html=True)

def page_presentation():
    st.markdown("<div class='presentation-scope'>", unsafe_allow_html=True)
    slides = [
        {"title": "Transfer Learning for Pneumonia — UH",
         "body": "Detect Bacterial vs Viral vs Normal on CXR with explainability.\n\n• 8 pretrained CNNs  • Ensembles  • Grad-CAM  • Streamlit app",
         "effect": None},
        {"title": "Problem & Motivation",
         "body": "CXR interpretation is time-intensive and variable.\nAI can support faster, consistent triage — especially where radiologists are scarce.",
         "effect": None},
        {"title": "Models Compared",
         "body": "VGG16 • VGG19 • ResNet50V2 • DenseNet201 • InceptionV3 • Xception • MobileNetV2 • EfficientNetV2-S",
         "effect": "weights"},
        {"title": "Results Snapshot",
         "body": "Ensembles improved balance across classes. Grad-CAM highlights decision areas.\nPick a class below to trigger a themed effect.",
         "effect": "fall"},
        {"title": "Ethics & Deployment",
         "body": "Research-only prototype • Anonymized public data • XAI via Grad-CAM • Not a medical device.\nNext: external validation & regulatory pathways.",
         "effect": None},
    ]
    img_paths = get_presentation_images(PRESENTATION_IMAGE_DIR)
    for i, p in enumerate(img_paths, start=1): slides.append({"title": f"Image {i}", "img_path": p, "effect": None})
    total = len(slides)
    if total == 0:
        st.warning("No slides found in your Presentation Images folder.")
        st.markdown("</div>", unsafe_allow_html=True); return

    left_t, mid_t, right_t = st.columns([1.2, 3.6, 1.2])
    if "slide" not in st.session_state: st.session_state.slide = 0
    with left_t:
        if st.button("↻ Refresh"): st.cache_data.clear()
    with mid_t:
        st.markdown(f"<div class='slide-toolbar'>Slides ready • Showing <b>#{st.session_state.slide+1} / {total}</b></div>", unsafe_allow_html=True)
    with right_t:
        loop = st.toggle("Loop", value=True)

    st.session_state.slide = int(np.clip(st.session_state.slide, 0, total-1))
    s = slides[st.session_state.slide]
    left, mid, right = st.columns([0.8, 3.8, 0.8])

    with left:
        if st.button("◀", key="prev_slide"):
            if st.session_state.slide == 0 and loop: st.session_state.slide = total - 1
            else: st.session_state.slide = max(0, st.session_state.slide - 1)

    with mid:
        if "img_path" in s:
            title = s.get("title", os.path.basename(s["img_path"]))
            st.markdown(f"<div class='slide slide-img fade'><h3>{title}</h3></div>", unsafe_allow_html=True)
            st.image(file_to_data_uri(s["img_path"]), use_container_width=True)
        else:
            st.markdown(
                f"<div class='slide fade'><h2 style='margin:.1rem 0 .2rem 0;'>{s['title']}</h2>"
                f"<p style='opacity:.98; font-size:1.05rem; white-space:pre-wrap; margin:.2rem 0 0 0;'>{s['body']}</p></div>",
                unsafe_allow_html=True,
            )
            if s["effect"] == "weights":
                e1, e2, e3 = st.columns(3)
                with e1: vgg = st.slider("VGG16", 0.0, 1.0, 0.4, 0.05, key="pvgg")
                with e2: res = st.slider("ResNet50V2", 0.0, 1.0, 0.5, 0.05, key="pres")
                with e3: v19 = st.slider("VGG19", 0.0, 1.0, 0.1, 0.05, key="pv19")
                tot = max(vgg+res+v19, 1e-6); vgg,res,v19 = vgg/tot, res/tot, v19/tot
                st.markdown(f"""
                <div class='card' style='max-width: 1100px; margin:.4rem auto;'>
                  <div class='prob-row'><div class='prob-label'>VGG16</div><div class='bar-outer'><div class='bar-inner' style='width:{vgg*100:.1f}%; background:#34d399;'></div></div><div style='width:70px; text-align:right'>{vgg*100:.1f}%</div></div>
                  <div class='prob-row'><div class='prob-label'>ResNet50V2</div><div class='bar-outer'><div class='bar-inner' style='width:{res*100:.1f}%; background:#60a5fa;'></div></div><div style='width:70px; text-align:right'>{res*100:.1f}%</div></div>
                  <div class='prob-row'><div class='prob-label'>VGG19</div><div class='bar-outer'><div class='bar-inner' style='width:{v19*100:.1f}%; background:#f472b6;'></div></div><div style='width:70px; text-align:right'>{v19*100:.1f}%</div></div>
                </div>
                """, unsafe_allow_html=True)
            if s["effect"] == "fall":
                st.markdown("<div class='card' style='margin-top:.5rem'><b>Pick a class theme</b></div>", unsafe_allow_html=True)
                _ = st.radio("", ["NORMAL", "BACTERIAL PNEUMONIA", "VIRAL PNEUMONIA"], horizontal=True, key="theme_pick")

        st.progress((st.session_state.slide + 1) / total)

    with right:
        if st.button("▶", key="next_slide"):
            if st.session_state.slide == total - 1 and loop: st.session_state.slide = 0
            else: st.session_state.slide = min(total - 1, st.session_state.slide + 1)

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== NAVIGATION (VISIBLE TABS) =====================
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown("<h4 style='margin:0 0 6px 2px; font-weight:900; color:#fff;'>Navigation</h4>", unsafe_allow_html=True)

tab_run, tab_abs, tab_spec, tab_deck = st.tabs(
    ["🧪 Run Classifier", "📄 Abstract & Intro", "🧰 Specification", "🎞️ Presentation"]
)

with tab_run: page_classifier()
with tab_abs: page_about_intro()
with tab_spec: page_specification()
with tab_deck: page_presentation()

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    "<div style='opacity:.95'>Made with <i class='bi bi-heart-fill text-danger'></i> by Srujan Demaiah Srinivasa under guidance of Anish Saini</div>",
    unsafe_allow_html=True,
)
