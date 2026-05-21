"""
WasteVision AI — app.py
Redesigned UI:  Modern AI SaaS dashboard  |  Light / Dark mode
All inference, pre-processing, model, and classification logic preserved unchanged.
"""

import os, re, json, pickle
import numpy as np
import joblib
import streamlit as st
from PIL import Image
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
#  PAGE CONFIG
# ============================================================


st.set_page_config(
    page_title="WasteVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  PATHS
# ============================================================
_DIR           = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH   = os.path.join(_DIR, "weights", "best_accuracy.weights.h5")
ELM_PATH       = os.path.join(_DIR, "weights", "elm_ensemble.pkl")
SCALER_PATH    = os.path.join(_DIR, "weights", "feature_scaler.pkl")
CLASS_IDX_PATH = os.path.join(_DIR, "weights", "class_indices.json")

# ============================================================
#  CONSTANTS
# ============================================================
NUM_CLASSES   = 36
IMG_SIZE      = (224, 224)
OOD_THRESHOLD = 0.30

MODEL_LABELS = {
    "cnn": "CNN Only  (EfficientNetB3)",
    "tta": "CNN + TTA  (5-Pass Augmentation)",
    "elm": "CNN + ELM Ensemble  (7 × Voting)",
}

# ============================================================
#  STAGE MAPPINGS  (unchanged)
# ============================================================
STAGE_3_TO_STAGE_2 = {
    "A_Foods": "A_Green Waste",
    "B_Animal Dead Body": "A_Green Waste",
    "C_Cardboard": "B_Recyclable Waste",
    "D_Newspaper": "B_Recyclable Waste",
    "E_Paper Cups": "B_Recyclable Waste",
    "F_Papers": "B_Recyclable Waste",
    "G_Brown Glass": "C_Glass",
    "H_Porcelin": "C_Glass",
    "I_Green Glass": "C_Glass",
    "J_White Glass": "C_Glass",
    "K_Beverage Cans": "D_Metal",
    "L_Construction Scrap": "D_Metal",
    "M_Metal Containers": "D_Metal",
    "N_Plastic Bag": "E_Polymer (Petrolium Based)",
    "O_Plastic Bottle": "E_Polymer (Petrolium Based)",
    "Q_Plastic Containers": "E_Polymer (Petrolium Based)",
    "R_Plastic Cups": "E_Polymer (Petrolium Based)",
    "S_Tetra Pak": "E_Polymer (Petrolium Based)",
    "T_Clothes": "F_Leather and Fabric",
    "U_Shoes": "F_Leather and Fabric",
    "V_Gloves": "F_Leather and Fabric",
    "W_Masks": "G_Medical Waste",
    "Z_G_Battery": "G_Medical Waste",
    "Z_H_Thermometer": "G_Medical Waste",
    "X_Bandai": "H_E Waste",
    "Z_B_Electrical Cables": "H_E Waste",
    "Z_C_Electronic Chips": "H_E Waste",
    "Z_D_Laptops": "H_E Waste",
    "Z_E_Small Appliances": "H_E Waste",
    "Z_F_Smartphones": "H_E Waste",
    "Y_Medicine and Medicine Strip": "I_Hazardous Waste",
    "Z_A_A_Syringe": "I_Hazardous Waste",
    "Z_A_Diaper": "I_Hazardous Waste",
    "Z_I_Cigarette Butt": "I_Hazardous Waste",
    "Z_J_Pesticidebottle": "I_Hazardous Waste",
    "Z_K_Spray cans": "I_Hazardous Waste",
}

STAGE_2_TO_STAGE_1 = {
    "A_Green Waste":               "A-Biodegradable",
    "B_Recyclable Waste":          "B-Non Biodegradable",
    "C_Glass":                     "B-Non Biodegradable",
    "D_Metal":                     "B-Non Biodegradable",
    "E_Polymer (Petrolium Based)": "B-Non Biodegradable",
    "F_Leather and Fabric":        "B-Non Biodegradable",
    "G_Medical Waste":             "B-Non Biodegradable",
    "H_E Waste":                   "B-Non Biodegradable",
    "I_Hazardous Waste":           "B-Non Biodegradable",
}

DISPOSAL_GUIDE = {
    "A_Green Waste":               "Compost or send to organic waste collection. Can be used for biogas generation.",
    "B_Recyclable Waste":          "Flatten cardboard, bundle papers. Drop at recycling centre or kerbside pickup.",
    "C_Glass":                     "Rinse and sort by colour. Never mix with general waste — glass is 100% recyclable.",
    "D_Metal":                     "Clean and compact. Scrap yards accept most metals. Avoid landfill at all costs.",
    "E_Polymer (Petrolium Based)": "Check resin code (#1-#7). Rigid plastics go to recycling; soft film to plastic bag drop-off.",
    "F_Leather and Fabric":        "Donate wearable items. Non-wearable textiles go to textile banks — not regular bins.",
    "G_Medical Waste":             "Seal in a leak-proof container. Return to pharmacy or authorised medical waste collector.",
    "H_E Waste":                   "Take to certified e-waste facility. Never put in regular trash — contains toxic heavy metals.",
    "I_Hazardous Waste":           "Store in original container. Take to hazardous waste drop-off event. NEVER pour down drain.",
}

STAGE2_EMOJI = {
    "A_Green Waste": "🌿",
    "B_Recyclable Waste": "♻️",
    "C_Glass": "🫙",
    "D_Metal": "🔩",
    "E_Polymer (Petrolium Based)": "🧴",
    "F_Leather and Fabric": "👕",
    "G_Medical Waste": "🏥",
    "H_E Waste": "💻",
    "I_Hazardous Waste": "☣️",
}

ENV_IMPACT = {
    "A_Green Waste":               "Composting diverts organic waste from landfill, reducing methane emissions.",
    "B_Recyclable Waste":          "Recycling paper saves trees and cuts CO₂ vs. virgin paper production.",
    "C_Glass":                     "Recycled glass melts at lower temps, saving energy in manufacturing.",
    "D_Metal":                     "Recycling aluminium uses 95% less energy than primary production.",
    "E_Polymer (Petrolium Based)": "Recycling plastic bottles saves energy and reduces ocean plastic pollution.",
    "F_Leather and Fabric":        "Textile recycling saves water and reduces fast-fashion landfill pressure.",
    "G_Medical Waste":             "Proper disposal prevents pathogen spread and groundwater contamination.",
    "H_E Waste":                   "Recovering rare metals from e-waste reduces toxic mining operations.",
    "I_Hazardous Waste":           "Safe disposal prevents soil/water contamination and protects ecosystems.",
}

STAGE1_META = {
    "A-Biodegradable":     {"card": "card-bio",    "icon": "🌱", "color": "#16a34a"},
    "B-Non Biodegradable": {"card": "card-nonbio", "icon": "⚠️", "color": "#ea580c"},
}

# ============================================================
#  HELPERS
# ============================================================
_OVERRIDES = {
    "H_E Waste":           "E-Waste",
    "A-Biodegradable":     "Biodegradable",
    "B-Non Biodegradable": "Non-Biodegradable",
}

def clean_name(name: str) -> str:
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    parts  = re.split(r"[_\-]", name)
    result = []
    found  = False
    for p in parts:
        if not found and len(p) == 1 and p.isalpha():
            continue
        found = True
        result.append(p)
    return " ".join(result) if result else name


# ============================================================
#  GRAD-CAM MODEL BUILDER
# ============================================================
def _build_gradcam_model(cnn_model, backbone):
    last_conv = None
    try:
        last_conv = backbone.get_layer("top_activation")
    except ValueError:
        for lyr in reversed(backbone.layers):
            if (hasattr(lyr, "output_shape")
                    and len(lyr.output_shape) == 4
                    and not isinstance(lyr, tf.keras.layers.InputLayer)):
                last_conv = lyr
                break
    if last_conv is None:
        return None

    try:
        backbone_dual = tf.keras.Model(
            inputs  = backbone.inputs,
            outputs = [last_conv.output, backbone.output],
            name    = "backbone_dual",
        )
        dense_layers = [l for l in cnn_model.layers
                        if isinstance(l, tf.keras.layers.Dense)]
        if len(dense_layers) < 3:
            return None

        gc_inp   = tf.keras.Input(shape=(224, 224, 3), name="gc_input")
        x_prep   = tf.keras.layers.Lambda(efficientnet_preprocess, name="gc_prep")(gc_inp)
        conv_out, bb_out = backbone_dual(x_prep, training=False)

        x        = tf.keras.layers.GlobalAveragePooling2D()(bb_out)
        x        = dense_layers[0](x)
        x        = tf.keras.layers.Dropout(0.5)(x, training=False)
        x        = dense_layers[1](x)
        x        = tf.keras.layers.Dropout(0.4)(x, training=False)
        preds    = dense_layers[2](x)

        return tf.keras.Model(inputs=gc_inp, outputs=[conv_out, preds],
                              name="gradcam_model")
    except Exception:
        return None


# ============================================================
#  ARTEFACT LOADING  (cached once per session)
# ============================================================
@st.cache_resource(show_spinner="Loading WasteVision model…")
def load_all_artefacts():
    backbone = EfficientNetB3(input_shape=(224, 224, 3),
                              include_top=False, weights=None)
    backbone.trainable = True

    inp  = layers.Input(shape=(224, 224, 3), name="input_images")
    x    = layers.Lambda(efficientnet_preprocess, name="preprocessing")(inp)
    x    = backbone(x, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(512, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4))(x)
    x    = layers.Dropout(0.5)(x)
    feat = layers.Dense(256, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4),
                         name="feature_vector")(x)
    x    = layers.Dropout(0.4)(feat)
    out  = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    cnn_model         = models.Model(inp, out,  name="EfficientNetB3_waste")
    feature_extractor = models.Model(inp, feat, name="feature_extractor")

    cnn_model.load_weights(WEIGHTS_PATH)

    with open(ELM_PATH, "rb") as fh:
        elm_payload = pickle.load(fh)

    scaler = joblib.load(SCALER_PATH)
    with open(CLASS_IDX_PATH) as fh:
        raw = json.load(fh)
    class_indices = {int(k): v for k, v in raw.items()}

    gradcam_model = _build_gradcam_model(cnn_model, backbone)

    return cnn_model, feature_extractor, elm_payload, scaler, class_indices, gradcam_model


# ============================================================
#  INFERENCE  (all logic unchanged)
# ============================================================
def preprocess_image(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS), dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def check_ood(probs: np.ndarray) -> tuple:
    max_conf = float(np.max(probs))
    entropy  = -np.sum(probs * np.log(probs + 1e-10))
    norm_ent = float(entropy / np.log(len(probs)))
    ood      = (max_conf < OOD_THRESHOLD) or (norm_ent > 0.92)
    return ood, max_conf, norm_ent


_TTA_DATAGEN = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10,
    brightness_range=[0.85, 1.15],
)


def infer_cnn(cnn_model, img_array: np.ndarray) -> tuple:
    probs = cnn_model.predict(img_array, verbose=0)[0]
    return probs, int(np.argmax(probs))


def infer_tta(cnn_model, img_array: np.ndarray, n_tta: int = 5) -> tuple:
    img_np    = img_array[0]
    all_probs = [cnn_model.predict(img_array, verbose=0)[0]]
    for _ in range(n_tta):
        aug   = _TTA_DATAGEN.random_transform(img_np.copy())
        probs = cnn_model.predict(np.expand_dims(aug, 0), verbose=0)[0]
        all_probs.append(probs)
    avg = np.mean(all_probs, axis=0)
    return avg, int(np.argmax(avg))


def infer_elm(cnn_model, feat_ext, elm_payload, scaler,
              img_array: np.ndarray) -> tuple:
    probs       = cnn_model.predict(img_array, verbose=0)[0]
    feat_raw    = feat_ext.predict(img_array, verbose=0)
    feat_scaled = scaler.transform(feat_raw)
    votes = []
    for e in elm_payload:
        H = np.maximum(0.0, feat_scaled @ e["W"] + e["b"])
        votes.append(int(np.argmax(H @ e["beta"], axis=1)[0]))
    elm_idx = int(np.bincount(votes).argmax())
    return probs, elm_idx


def run_inference(img: Image.Image, model_mode: str,
                  cnn_model, feat_ext, elm_payload, scaler,
                  class_indices: dict) -> dict:
    img_array = preprocess_image(img)

    if model_mode == "cnn":
        probs, class_idx = infer_cnn(cnn_model, img_array)
    elif model_mode == "tta":
        probs, class_idx = infer_tta(cnn_model, img_array, n_tta=5)
    else:
        probs, class_idx = infer_elm(cnn_model, feat_ext, elm_payload, scaler, img_array)

    ood, max_conf, norm_ent = check_ood(probs)
    if ood:
        return {"ood": True, "max_conf": max_conf,
                "norm_ent": norm_ent, "model_mode": model_mode}

    stage3  = class_indices.get(class_idx, "Unknown")
    stage2  = STAGE_3_TO_STAGE_2.get(stage3, "Unknown")
    stage1  = STAGE_2_TO_STAGE_1.get(stage2, "Unknown")
    top5_i  = np.argsort(probs)[::-1][:5]
    top5    = [(class_indices.get(int(i), "?"), float(probs[i])) for i in top5_i]

    return {
        "ood":        False,
        "stage3":     stage3,
        "stage2":     stage2,
        "stage1":     stage1,
        "top5":       top5,
        "conf":       float(probs[class_idx]),
        "class_idx":  class_idx,
        "model_mode": model_mode,
    }


# ============================================================
#  GRAD-CAM  (unchanged)
# ============================================================
def compute_gradcam(gradcam_model, img_array: np.ndarray,
                    class_idx: int) -> np.ndarray:
    if gradcam_model is None:
        return None
    try:
        x = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = gradcam_model(x)
            target = predictions[:, class_idx]
        grads = tape.gradient(target, conv_outputs)
        if grads is None:
            return None
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled, axis=-1)
        heatmap = tf.nn.relu(heatmap).numpy()
        vmax    = heatmap.max()
        return heatmap / vmax if vmax > 0 else heatmap
    except Exception:
        return None


def overlay_heatmap(orig_img: Image.Image, heatmap: np.ndarray,
                    alpha: float = 0.45) -> Image.Image:
    orig        = np.array(orig_img.convert("RGB").resize((224, 224)), dtype=float)
    heat_small  = Image.fromarray((heatmap * 255).astype(np.uint8))
    heat_rs     = np.array(heat_small.resize((224, 224), Image.LANCZOS)) / 255.0
    colored     = (cm.jet(heat_rs)[:, :, :3] * 255)
    blended     = np.clip((1 - alpha) * orig + alpha * colored, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ============================================================
#  THEME CSS  (new variables-based system)
# ============================================================
def get_theme_css(theme: str = "light") -> str:
    """Inject CSS custom properties for the chosen theme, then import style.css."""

    if theme == "dark":
        vars_css = """
        :root {
            --bg-primary:        #0f172a;
            --bg-secondary:      #1e293b;
            --bg-card:           #1e293b;
            --bg-card-bio:       #14532d;
            --bg-card-nonbio:    #7c2d12;
            --text-primary:      #f1f5f9;
            --text-secondary:    #94a3b8;
            --text-muted:        #64748b;
            --border-color:      #334155;
            --border-bio:        #166534;
            --border-nonbio:     #c2410c;
            --accent-green:      #22c55e;
            --accent-blue:       #60a5fa;
            --shadow:            rgba(0,0,0,0.35);
            --shadow-md:         rgba(0,0,0,0.55);
            --badge-bg:          #14532d;
            --badge-border:      #166534;
            --badge-text:        #86efac;
            --model-tag-bg:      #334155;
            --model-tag-border:  #475569;
            --model-tag-text:    #94a3b8;
            --stage1-bio-bg:     #14532d;
            --stage1-bio-text:   #86efac;
            --stage1-bio-border: #166534;
            --stage1-nb-bg:      #7c2d12;
            --stage1-nb-text:    #fdba74;
            --stage1-nb-border:  #c2410c;
            --stage2-bg:         #312e81;
            --stage2-text:       #a5b4fc;
            --stage2-border:     #4338ca;
            --stage3-bg:         #1e3a8a;
            --stage3-text:       #93c5fd;
            --cbar-track:        #334155;
            --disposal-bg:       #1e293b;
            --disposal-border:   #334155;
            --disposal-title:    #94a3b8;
            --disposal-text:     #e2e8f0;
            --ood-bg:            #450a0a;
            --ood-border:        #dc2626;
            --ood-text:          #fca5a5;
            --ood-conf-bg:       #7f1d1d;
            --ood-conf-border:   #dc2626;
            --ood-conf-text:     #fca5a5;
        }
        .main .block-container { background-color: #0f172a !important; }
        """
        body_bg = "#0f172a"
    else:
        vars_css = """
        :root {
            --bg-primary:        #f8fafc;
            --bg-secondary:      #f1f5f9;
            --bg-card:           #ffffff;
            --bg-card-bio:       #f0fdf4;
            --bg-card-nonbio:    #fff7ed;
            --text-primary:      #0f172a;
            --text-secondary:    #64748b;
            --text-muted:        #94a3b8;
            --border-color:      #e2e8f0;
            --border-bio:        #86efac;
            --border-nonbio:     #fdba74;
            --accent-green:      #16a34a;
            --accent-blue:       #3b82f6;
            --shadow:            rgba(0,0,0,0.06);
            --shadow-md:         rgba(0,0,0,0.12);
            --badge-bg:          #f0fdf4;
            --badge-border:      #86efac;
            --badge-text:        #16a34a;
            --model-tag-bg:      #f8fafc;
            --model-tag-border:  #e2e8f0;
            --model-tag-text:    #64748b;
            --stage1-bio-bg:     #f0fdf4;
            --stage1-bio-text:   #166534;
            --stage1-bio-border: #bbf7d0;
            --stage1-nb-bg:      #fff7ed;
            --stage1-nb-text:    #9a3412;
            --stage1-nb-border:  #fed7aa;
            --stage2-bg:         #f5f3ff;
            --stage2-text:       #5b21b6;
            --stage2-border:     #ddd6fe;
            --stage3-bg:         #eff6ff;
            --stage3-text:       #1d4ed8;
            --cbar-track:        #f1f5f9;
            --disposal-bg:       #fafafa;
            --disposal-border:   #e2e8f0;
            --disposal-title:    #64748b;
            --disposal-text:     #374151;
            --ood-bg:            #fef2f2;
            --ood-border:        #fca5a5;
            --ood-text:          #991b1b;
            --ood-conf-bg:       #fee2e2;
            --ood-conf-border:   #fca5a5;
            --ood-conf-text:     #b91c1c;
        }
        .main .block-container { background-color: #f8fafc !important; }
        """
        body_bg = "#f8fafc"

    # Read external CSS file if it exists (same directory as app.py)
    css_path = os.path.join(_DIR, "style.css")
    ext_css  = ""
    if os.path.exists(css_path):
        with open(css_path) as fh:
            ext_css = fh.read()

    return f"""
<style>
/* Theme Variables */
{vars_css}

/* App background */
[data-testid="stAppViewContainer"] {{
    background-color: {body_bg};
}}
[data-testid="stAppViewContainer"] > .main {{
    background-color: {body_bg};
}}

/* External stylesheet content */
{ext_css}
</style>
"""


# ============================================================
#  CONFIDENCE GAUGE (SVG)
# ============================================================
def confidence_gauge_html(pct: float) -> str:
    """Return an inline SVG circular gauge for the confidence score."""
    r         = 26
    circ      = 2 * 3.14159 * r
    filled    = circ * (pct / 100)
    remaining = circ - filled
    color     = "#22c55e" if pct >= 70 else ("#f59e0b" if pct >= 40 else "#ef4444")
    return f"""
<div class="conf-gauge-wrap">
  <div class="conf-gauge">
    <svg width="62" height="62" viewBox="0 0 62 62">
      <circle class="conf-gauge-track" cx="31" cy="31" r="{r}"/>
      <circle class="conf-gauge-fill" cx="31" cy="31" r="{r}"
        stroke="{color}"
        stroke-dasharray="{filled:.2f} {remaining:.2f}"
        stroke-dashoffset="0"/>
    </svg>
    <div class="conf-gauge-text">
      <span class="conf-pct-val">{pct:.0f}%</span>
      <span class="conf-pct-lbl">conf</span>
    </div>
  </div>
</div>"""


# ============================================================
#  OOD RENDERER
# ============================================================
def render_ood(max_conf: float, model_label: str) -> None:
    st.markdown(f"""
    <div class="ood-card">
      <div class="ood-icon">🤔</div>
      <p class="ood-title">Unable to Classify</p>
      <p class="ood-body">
        This image doesn't appear to belong to any of the
        <strong>36 waste categories</strong> the model was trained on.<br><br>
        Please provide a clear photo of a waste item — plastic bottles,
        glass jars, cardboard, electronics, food scraps, batteries, clothing, etc.
      </p>
      <span class="ood-conf">max confidence {max_conf*100:.1f}% &nbsp;·&nbsp; {model_label}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  MAIN RESULTS RENDERER
# ============================================================
def render_results(result: dict, orig_img: Image.Image,
                   gradcam_model, show_gradcam: bool) -> None:

    if result["ood"]:
        render_ood(result["max_conf"], MODEL_LABELS[result["model_mode"]])
        return

    s1, s2, s3   = result["stage1"], result["stage2"], result["stage3"]
    meta          = STAGE1_META.get(s1, {"card": "card-nonbio", "icon": "🗑️"})
    s2_emo        = STAGE2_EMOJI.get(s2, "📦")
    mode_lbl      = MODEL_LABELS[result["model_mode"]]
    conf_pct      = result["conf"] * 100

    # ── Prediction Card ──────────────────────────────────────
    st.markdown('<p class="mlabel">Classification Result</p>', unsafe_allow_html=True)
    gauge_html = confidence_gauge_html(conf_pct)
    st.markdown(f"""
    <div class="pred-card {meta['card']}">
      <p class="pred-stage-badge">Stage 3 Result</p>
      <div class="pred-row">
        <div>
          <p class="pred-name">{meta['icon']} {clean_name(s3)}</p>
          <p class="pred-raw">{s3}</p>
          <span class="model-tag">🔬 {mode_lbl}</span>
        </div>
        {gauge_html}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Two sub-columns: Top-5 | Stage Hierarchy ─────────────
    # col_a, col_b = st.columns([1.1, 1])
    col_a, col_b = st.columns([1, 1.6])
    

    with col_a:
        st.markdown('<p class="mlabel">Top 5 Predictions</p>', unsafe_allow_html=True)
        top5  = result["top5"]
        max_p = max(p for _, p in top5) or 1e-6
        bars  = '<div class="cbar-wrap">'
        for i, (cls_name, prob) in enumerate(top5):
            hi   = "hi" if i == 0 else ""
            rank = "●" if i == 0 else f"{i+1}."
            bars += f"""
            <div class="cbar-row">
              <span class="cbar-lbl {hi}" title="{cls_name}">{rank} {clean_name(cls_name)}</span>
              <div class="cbar-track">
                <div class="cbar-fill {hi}" style="width:{prob/max_p*100:.1f}%"></div>
              </div>
              <span class="cbar-pct {hi}">{prob*100:.1f}%</span>
            </div>"""
        bars += "</div>"
        st.markdown(bars, unsafe_allow_html=True)

    with col_b:
        # ── Hierarchical Classification ──────────────────────
        st.markdown('<p class="mlabel">Hierarchical Classification</p>', unsafe_allow_html=True)
        ss1_cls = "ss-1bio" if s1 == "A-Biodegradable" else "ss-1nb"
        st.markdown(f"""
        <div class="stage-flow">
          <div class="stage-step {ss1_cls}">
            <span class="ss-icon">{meta['icon']}</span>
            <p class="ss-label">Stage 1</p>
            <p class="ss-name">{clean_name(s1)}</p>
          </div>
          <div class="stage-step ss-2">
            <span class="ss-icon">{s2_emo}</span>
            <p class="ss-label">Stage 2</p>
            <p class="ss-name">{clean_name(s2)}</p>
          </div>
          <div class="stage-step ss-3">
            <span class="ss-icon">📌</span>
            <p class="ss-label">Stage 3</p>
            <p class="ss-name">{clean_name(s3)}</p>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Disposal Guide + Environmental Impact ────────────────
    guide  = DISPOSAL_GUIDE.get(s2, "Consult your local waste management authority.")
    impact = ENV_IMPACT.get(s2, "")
    st.markdown(f"""
    <div class="disp-card">
      <span class="disp-icon">{s2_emo}</span>
      <div>
        <p class="disp-title">Disposal Guidance · {clean_name(s2)}</p>
        <p class="disp-text">{guide}</p>
        {f'<span class="env-chip">🌍 {impact}</span>' if impact else ""}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Grad-CAM ─────────────────────────────────────────────
    if show_gradcam:
        st.markdown(
            '<div class="gcam-wrap">'
            '<div class="gcam-hdr">🔥 Grad-CAM &mdash; Model Attention Map</div>'
            '<div class="gcam-body">',
            unsafe_allow_html=True)

        gc_key  = f"gc_{result['class_idx']}"
        cam_img = st.session_state.get(gc_key)

        if cam_img is None:
            with st.spinner("Computing Grad-CAM…"):
                img_arr = preprocess_image(orig_img)
                heatmap = compute_gradcam(gradcam_model, img_arr, result["class_idx"])
            if heatmap is not None:
                cam_img = overlay_heatmap(orig_img, heatmap)
                st.session_state[gc_key] = cam_img

        if cam_img is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.image(orig_img.convert("RGB").resize((224, 224)),
                         caption="Original", use_container_width=True)
            with c2:
                st.image(cam_img, caption="Grad-CAM Overlay",
                         use_container_width=True)
            st.markdown(
                '<p class="gcam-note">🔴 Red/Yellow = high attention &nbsp;·&nbsp; 🔵 Blue = low attention</p>',
                unsafe_allow_html=True)
        else:
            st.info("Grad-CAM could not be computed. Verify 'top_activation' layer in backbone.")

        st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
#  SIDEBAR
# ============================================================
def render_sidebar() -> tuple:
    """Render the green sidebar. Returns (model_mode_key, show_gradcam)."""
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sb-brand">
          <div class="sb-logo-row">
            <div class="sb-logo">♻️</div>
            <div>
              <p class="sb-brand-name">WasteVision AI</p>
              <p class="sb-brand-sub">Smart Waste · Better Future</p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # # Navigation items (decorative — main content drives the page)
        # st.markdown("""
        # <div class="sb-nav">
        #   <p class="sb-section-label">Navigation</p>
        #   <div class="sb-nav-item active">
        #     <span class="sb-nav-icon">🏠</span> Home
        #   </div>
        #   <div class="sb-nav-item">
        #     <span class="sb-nav-icon">📷</span> Camera
        #   </div>
        #   <div class="sb-nav-item">
        #     <span class="sb-nav-icon">📁</span> Upload Image
        #   </div>
        #   <div class="sb-nav-item">
        #     <span class="sb-nav-icon">📊</span> History
        #   </div>
        #   <div class="sb-nav-item">
        #     <span class="sb-nav-icon">❓</span> Help & Guide
        #   </div>
        # </div>
        # """, unsafe_allow_html=True)

        # Model Options (white card)
        st.markdown('<div class="sb-model-box">', unsafe_allow_html=True)
        st.markdown('<p class="sb-model-label">Model Options</p>', unsafe_allow_html=True)

        mode_options = list(MODEL_LABELS.values())
        mode_keys    = list(MODEL_LABELS.keys())
        default_idx  = mode_keys.index(st.session_state.get("model_mode", "elm"))
        chosen_label = st.selectbox(
            label="Select Model",
            options=mode_options,
            index=default_idx,
            label_visibility="collapsed",
            key="sb_model_select",
        )
        model_mode = mode_keys[mode_options.index(chosen_label)]
        st.session_state["model_mode"] = model_mode

        show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=False, key="sb_gradcam")
        st.markdown('</div>', unsafe_allow_html=True)

        # Theme toggle
        st.markdown('<div class="sb-theme-box">', unsafe_allow_html=True)
        st.markdown('<p class="sb-theme-label">🎨 Theme</p>', unsafe_allow_html=True)
        theme_choice = st.radio(
            "theme_radio",
            ["Light", "Dark"],
            index=0 if st.session_state.get("theme", "light") == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="sb_theme_radio",
        )
        st.session_state["theme"] = theme_choice.lower()
        st.markdown('</div>', unsafe_allow_html=True)

        # Model info card
        st.markdown("""
        <div class="sb-info">
          <p class="sb-info-title">About Model</p>
          <div class="sb-info-row"><div class="sb-info-dot"></div>
            <p class="sb-info-text">EfficientNetB3 Backbone</p></div>
          <div class="sb-info-row"><div class="sb-info-dot"></div>
            <p class="sb-info-text">7× ELM Ensemble</p></div>
          <div class="sb-info-row"><div class="sb-info-dot"></div>
            <p class="sb-info-text">36 Waste Classes</p></div>
          <div class="sb-info-row"><div class="sb-info-dot"></div>
            <p class="sb-info-text">3-Stage Hierarchy</p></div>
          <div class="sb-info-row"><div class="sb-info-dot"></div>
            <p class="sb-info-text">Grad-CAM Explainability</p></div>
        </div>
        """, unsafe_allow_html=True)

    return model_mode, show_gradcam


# ============================================================
#  MAIN
# ============================================================
def main() -> None:
    # ── Session defaults ──────────────────────────────────────
    for k, v in [("theme", "light"), ("current_img", None),
                 ("img_hash", None), ("result_cache", {}),
                 ("model_mode", "elm")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Apply theme CSS ───────────────────────────────────────
    st.markdown(get_theme_css(st.session_state["theme"]), unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────
    model_mode, show_gradcam = render_sidebar()

    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <div class="wv-header">
      <div class="wv-hdr-logo">♻️</div>
      <div class="wv-hdr-text">
        <p class="wv-hdr-title">Welcome to <span>WasteVision AI</span></p>
        <p class="wv-hdr-sub">AI-Powered Hierarchical Waste Classification System</p>
      </div>
      <div class="wv-hdr-right">
        <div class="wv-badge">
          <div class="wv-badge-dot"></div>
          EfficientNetB3 + ELM &nbsp;·&nbsp; 36 Classes &nbsp;·&nbsp; 3 Stages
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Check model artefacts ─────────────────────────────────
    missing = [p for p in [WEIGHTS_PATH, ELM_PATH, SCALER_PATH, CLASS_IDX_PATH]
               if not os.path.exists(p)]
    if missing:
        st.error("**Missing model artefact files.** "
                 "Create a `weights/` folder next to `app.py` and add:")
        for m in missing:
            st.code(os.path.basename(m))
        st.info("Expected layout:\n```\nweights/\n  best_accuracy.weights.h5\n"
                "  elm_ensemble.pkl\n  feature_scaler.pkl\n  class_indices.json\n```")
        return

    try:
        cnn_model, feat_ext, elm_payload, scaler, class_indices, gradcam_model = \
            load_all_artefacts()
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        return

    # ── Two-column layout ─────────────────────────────────────
    col_left, col_right = st.columns([1, 1.4], gap="large")

    # ─── LEFT: Input ──────────────────────────────────────────
    with col_left:

        # Input method selector (visual only — tabs handle actual logic)
        st.markdown("""
        <p class="mlabel">Select Input Method</p>
        <div class="input-method-grid">
          <div class="input-method-btn active">
            <span class="imb-icon">📷</span>
            <p class="imb-label">Camera</p>
            <p class="imb-sub">Capture from webcam</p>
          </div>
          <div class="input-method-btn">
            <span class="imb-icon">📁</span>
            <p class="imb-label">Upload Image</p>
            <p class="imb-sub">Choose from device</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab_cam, tab_up = st.tabs(["  📷  Webcam", "  📁  Upload"])

        with tab_cam:
            st.markdown('<p class="mlabel">Capture a frame to classify</p>',
                        unsafe_allow_html=True)
            cam_data = st.camera_input(
                label="cam", label_visibility="collapsed",
                help="Click the shutter button — classification runs instantly.",
            )
            if cam_data is not None:
                new_img = Image.open(cam_data).convert("RGB")
                st.session_state["current_img"] = new_img
                for k in [k for k in st.session_state if k.startswith("gc_")]:
                    del st.session_state[k]
                # Compact preview
                st.markdown('<div class="img-preview-card">', unsafe_allow_html=True)
                st.image(new_img.resize((280, 280), Image.LANCZOS),
                         use_container_width=False, width=280)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab_up:
            st.markdown('<p class="mlabel">Browse or drag-and-drop an image</p>',
                        unsafe_allow_html=True)
            uploaded = st.file_uploader(
                label="upload",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                new_img = Image.open(uploaded).convert("RGB")
                st.session_state["current_img"] = new_img
                for k in [k for k in st.session_state if k.startswith("gc_")]:
                    del st.session_state[k]

            # Image preview (only if not from webcam)
            cur = st.session_state.get("current_img")
            if cur is not None:
                st.markdown('<p class="mlabel" style="margin-top:10px">Image Preview</p>',
                            unsafe_allow_html=True)
                st.markdown('<div class="img-preview-card">', unsafe_allow_html=True)
                st.image(cur.resize((280, 280), Image.LANCZOS),
                         use_container_width=False, width=280)
                st.markdown('</div>', unsafe_allow_html=True)

                col_r, col_c = st.columns(2)
                with col_r:
                    if st.button("🔄 Retake / Reset", use_container_width=True):
                        st.session_state["current_img"] = None
                        st.session_state["result_cache"] = {}
                        st.rerun()
                with col_c:
                    if st.button("🗑️ Clear Result", use_container_width=True):
                        st.session_state["result_cache"] = {}
                        st.rerun()

    # ─── RIGHT: Results ───────────────────────────────────────
    with col_right:
        current_img = st.session_state.get("current_img")

        if current_img is None:
            st.markdown("""
            <div class="placeholder">
              <div class="ph-icon">🗑️</div>
              <p class="ph-title">No Image Yet</p>
              <p class="ph-sub">Capture via webcam or upload a photo<br>
              to get an instant 3-stage waste classification.</p>
            </div>""", unsafe_allow_html=True)
        else:
            img_hash  = hash(current_img.tobytes())
            cache_key = (img_hash, model_mode)

            if cache_key not in st.session_state["result_cache"]:
                with st.spinner("Classifying image…"):
                    result = run_inference(
                        current_img, model_mode,
                        cnn_model, feat_ext, elm_payload, scaler, class_indices,
                    )
                st.session_state["result_cache"][cache_key] = result
                if len(st.session_state["result_cache"]) > 9:
                    oldest = next(iter(st.session_state["result_cache"]))
                    del st.session_state["result_cache"][oldest]

            cached = st.session_state["result_cache"].get(cache_key)
            if cached is not None:
                render_results(cached, current_img, gradcam_model, show_gradcam)

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div class="wv-footer">
      <p>WasteVision AI &nbsp;·&nbsp; EfficientNetB3 Backbone &nbsp;·&nbsp;
         2-Phase Fine-tuning &nbsp;·&nbsp; Borderline-SMOTE &nbsp;·&nbsp;
         7× ELM Ensemble &nbsp;·&nbsp; TTA &nbsp;·&nbsp; Grad-CAM</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main(

     
    )
