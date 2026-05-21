import os, re, json, pickle, time
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
    "cnn": "CNN Only (EfficientNetB3)",
    "tta": "CNN + TTA (5-Pass Augmentation)",
    "elm": "CNN + ELM Ensemble (7 × Voting)",
}

# ============================================================
#  STAGE MAPPINGS
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

# ---- Simplified, human-readable disposal guide ----
DISPOSAL_GUIDE = {
    "A_Green Waste":
        "Toss it in your green or organic bin. Even better — compost it at home! "
        "It breaks down naturally and makes great fertiliser for your garden.",
    "B_Recyclable Waste":
        "Flatten cardboard boxes and bundle loose papers. Pop them in your recycling bin "
        "or drop at a local recycling centre. Keep them dry so they can actually be recycled.",
    "C_Glass":
        "Give it a quick rinse, then place in your glass recycling bin. "
        "Try to sort by colour if you can. Glass is 100% recyclable and can be reused forever!",
    "D_Metal":
        "Rinse it out and put in your recycling bin. Scrap yards also accept most metals. "
        "Keeping metal out of landfill saves a huge amount of energy.",
    "E_Polymer (Petrolium Based)":
        "Check the recycling number on the bottom (#1–#7). "
        "Hard plastics go in your recycling bin. Soft plastic bags go to supermarket "
        "drop-off points — not the regular bin.",
    "F_Leather and Fabric":
        "Still wearable? Donate it to a charity shop or give it away. "
        "If it's too worn out, drop it at a textile bank — never the regular bin.",
    "G_Medical Waste":
        "Seal it in a bag and return to a pharmacy or medical waste collection point. "
        "Never throw it in the regular bin — it can be harmful to bin workers and the environment.",
    "H_E Waste":
        "Take it to a certified electronics recycling centre. "
        "It contains toxic materials like lead and mercury that can poison soil and water "
        "if thrown in regular bins.",
    "I_Hazardous Waste":
        "Keep it in the original container and take to a hazardous waste drop-off event or facility. "
        "Never pour it down the drain or put it in the regular bin.",
}

ENVIRONMENTAL_IMPACT = {
    "A_Green Waste":               "Composting organic waste cuts landfill methane emissions and enriches soil naturally.",
    "B_Recyclable Waste":          "Recycling paper saves trees and significantly reduces water consumption.",
    "C_Glass":                     "Recycled glass saves energy and can be recycled endlessly without quality loss.",
    "D_Metal":                     "Recycling metals uses up to 95% less energy than producing new metal from scratch.",
    "E_Polymer (Petrolium Based)": "Recycling plastic bottles saves energy and helps reduce ocean pollution.",
    "F_Leather and Fabric":        "Donating clothes reduces textile waste — one of the world's biggest polluters.",
    "G_Medical Waste":             "Proper disposal stops the spread of infections and protects communities.",
    "H_E Waste":                   "Correct recycling prevents toxic heavy metals from contaminating soil and water.",
    "I_Hazardous Waste":           "Safe disposal protects groundwater and prevents harmful chemical exposure.",
}

STAGE2_EMOJI = {
    "A_Green Waste":               "🌿",
    "B_Recyclable Waste":          "♻️",
    "C_Glass":                     "🫙",
    "D_Metal":                     "🔩",
    "E_Polymer (Petrolium Based)": "🧴",
    "F_Leather and Fabric":        "👕",
    "G_Medical Waste":             "🏥",
    "H_E Waste":                   "💻",
    "I_Hazardous Waste":           "☣️",
}

STAGE1_META = {
    "A-Biodegradable":     {"icon": "🌱", "color": "#16a34a", "bg": "#f0fdf4", "border": "#86efac", "tag_bg": "#dcfce7", "tag_color": "#166534"},
    "B-Non Biodegradable": {"icon": "⚠️", "color": "#ea580c", "bg": "#fff7ed", "border": "#fdba74", "tag_bg": "#ffedd5", "tag_color": "#9a3412"},
}

# ============================================================
#  HELPERS
# ============================================================
_OVERRIDES = {
    "H_E Waste":           "E-Waste",
    "A-Biodegradable":     "Biodegradable",
    "B-Non Biodegradable": "Non-Biodegradable",
    "E_Polymer (Petrolium Based)": "Polymer / Plastic",
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
#  ARTEFACT LOADING
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
#  INFERENCE
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

def infer_cnn(cnn_model, img_array):
    probs = cnn_model.predict(img_array, verbose=0)[0]
    return probs, int(np.argmax(probs))


def infer_tta(cnn_model, img_array, n_tta=5):
    img_np    = img_array[0]
    all_probs = [cnn_model.predict(img_array, verbose=0)[0]]
    for _ in range(n_tta):
        aug   = _TTA_DATAGEN.random_transform(img_np.copy())
        probs = cnn_model.predict(np.expand_dims(aug, 0), verbose=0)[0]
        all_probs.append(probs)
    avg = np.mean(all_probs, axis=0)
    return avg, int(np.argmax(avg))


def infer_elm(cnn_model, feat_ext, elm_payload, scaler, img_array):
    probs       = cnn_model.predict(img_array, verbose=0)[0]
    feat_raw    = feat_ext.predict(img_array, verbose=0)
    feat_scaled = scaler.transform(feat_raw)
    votes = []
    for e in elm_payload:
        H = np.maximum(0.0, feat_scaled @ e["W"] + e["b"])
        votes.append(int(np.argmax(H @ e["beta"], axis=1)[0]))
    elm_idx = int(np.bincount(votes).argmax())
    return probs, elm_idx


def run_inference(img, model_mode, cnn_model, feat_ext,
                  elm_payload, scaler, class_indices):
    img_array = preprocess_image(img)
    t0 = time.perf_counter()

    if model_mode == "cnn":
        probs, class_idx = infer_cnn(cnn_model, img_array)
    elif model_mode == "tta":
        probs, class_idx = infer_tta(cnn_model, img_array, n_tta=5)
    else:
        probs, class_idx = infer_elm(cnn_model, feat_ext, elm_payload, scaler, img_array)

    infer_ms = int((time.perf_counter() - t0) * 1000)
    ood, max_conf, norm_ent = check_ood(probs)

    if ood:
        return {"ood": True, "max_conf": max_conf,
                "norm_ent": norm_ent, "model_mode": model_mode,
                "infer_ms": infer_ms}

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
        "infer_ms":   infer_ms,
    }


# ============================================================
#  GRAD-CAM
# ============================================================
def compute_gradcam(gradcam_model, img_array, class_idx):
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


def overlay_heatmap(orig_img, heatmap, alpha=0.45):
    orig        = np.array(orig_img.convert("RGB").resize((224, 224)), dtype=float)
    heat_small  = Image.fromarray((heatmap * 255).astype(np.uint8))
    heat_rs     = np.array(heat_small.resize((224, 224), Image.LANCZOS)) / 255.0
    colored     = (cm.jet(heat_rs)[:, :, :3] * 255)
    blended     = np.clip((1 - alpha) * orig + alpha * colored, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ============================================================
#  CSS
# ============================================================
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

/* ── Main container ── */
.main .block-container {
    padding: 1rem 1.4rem 0.5rem !important;
    max-width: 100% !important;
}

/* ── Sidebar base ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e8f5e9 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── Sidebar brand block ── */
.sb-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 16px 14px;
    border-bottom: 1px solid #e8f5e9;
}
.sb-logo {
    width: 40px; height: 40px; border-radius: 11px; flex-shrink: 0;
    background: linear-gradient(135deg, #16a34a, #059669);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; box-shadow: 0 2px 10px rgba(22,163,74,.28);
}
.sb-title   { font-size: 1.02rem; font-weight: 700; color: #0f172a; margin: 0; line-height: 1.15; }
.sb-tagline { font-size: .67rem; color: #94a3b8; margin: 1px 0 0; }

/* ── Sidebar nav buttons (override Streamlit) ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    width: 100%;
    background: transparent !important;
    border: none !important;
    border-radius: 9px !important;
    text-align: left !important;
    padding: 9px 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    box-shadow: none !important;
    transition: background .15s, color .15s !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #f0fdf4 !important;
    color: #16a34a !important;
}

/* Active nav item via HTML */
.nav-item-active {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 14px; border-radius: 9px; margin-bottom: 2px;
    background: #f0fdf4; color: #16a34a;
    font-size: .85rem; font-weight: 600;
}
.nav-item-inactive {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 14px; border-radius: 9px; margin-bottom: 2px;
    color: #475569; font-size: .85rem; font-weight: 500;
}
.nav-icon { font-size: .95rem; width: 20px; text-align: center; }

/* ── Sidebar info cards ── */
.sb-card {
    margin: 10px 12px 0;
    padding: 12px 14px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}
.sb-card-title { font-size: .64rem; font-weight: 700; text-transform: uppercase;
                  letter-spacing: .08em; color: #16a34a; margin: 0 0 6px; }
.sb-card-text  { font-size: .73rem; color: #64748b; line-height: 1.55; margin: 0; }

.perf-val { font-size: 1.5rem; font-weight: 700; color: #16a34a; margin: 2px 0 0; line-height: 1; }
.perf-sub { font-size: .65rem; color: #94a3b8; margin: 0; }
.perf-sep { height: 1px; background: #e2e8f0; margin: 8px 0; }
.perf-trophy { font-size: .78rem; font-weight: 700; color: #0f172a; margin: 0 0 4px; }

/* ── Page header ── */
.page-hdr {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 12px;
}
.page-title { font-size: 1.65rem; font-weight: 700; color: #0f172a; margin: 0; line-height: 1.2; }
.page-title span { color: #16a34a; }
.page-sub   { font-size: .78rem; color: #94a3b8; margin: 3px 0 0; }
.acc-badge  {
    background: #f0fdf4; border: 1.5px solid #86efac;
    border-radius: 10px; padding: 8px 16px 9px; text-align: center; flex-shrink: 0;
    min-width: 155px;
}
.acc-badge-lbl { font-size: .65rem; color: #64748b; margin: 0; }
.acc-badge-val { font-size: 1.05rem; font-weight: 700; color: #16a34a; margin: 2px 0 0; }
.acc-badge-sub { font-size: .63rem; color: #94a3b8; margin: 1px 0 0; }

/* ── Input method selector ── */
.input-method-box {
    border: 1.5px solid #e2e8f0; border-radius: 11px;
    padding: 13px 14px; margin-bottom: 11px;
}
.input-method-lbl { font-size: .78rem; font-weight: 700; color: #0f172a; margin: 0 0 9px; }
.im-btn-row { display: flex; gap: 10px; }
.im-btn {
    flex: 1; padding: 14px 10px; border-radius: 9px;
    border: 1.5px solid #e2e8f0; text-align: center; background: white;
}
.im-btn.active { border-color: #16a34a; background: #f0fdf4; }
.im-btn-icon  { font-size: 1.25rem; display: block; margin-bottom: 4px; }
.im-btn-title { font-size: .85rem; font-weight: 600;
                 color: #0f172a; margin: 0; display: block; }
.im-btn.active .im-btn-title { color: #16a34a; }
.im-btn-sub   { font-size: .7rem; color: #94a3b8; margin: 2px 0 0; display: block; }

/* ── Section title ── */
.sec-title { font-size: .82rem; font-weight: 700; color: #0f172a; margin: 0 0 10px; }

/* ── Bordered section card ── */
.section-card {
    border: 1.5px solid #e2e8f0; border-radius: 11px;
    padding: 14px; height: 100%; box-sizing: border-box;
}

/* ── Placeholder ── */
.placeholder {
    min-height: 280px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: #f8fafc; border-radius: 9px;
    border: 2px dashed #cbd5e1; color: #94a3b8; text-align: center; padding: 24px;
}
.ph-icon  { font-size: 2.5rem; margin-bottom: 10px; opacity: .5; }
.ph-title { font-size: .88rem; font-weight: 600; margin: 0 0 4px; color: #64748b; }
.ph-sub   { font-size: .74rem; margin: 0; }

/* ── OOD card ── */
.ood-card {
    border-radius: 11px; background: #fef2f2; border: 1.5px solid #fca5a5;
    padding: 22px; text-align: center; margin-top: 4px;
}
.ood-icon  { font-size: 2.6rem; margin-bottom: 8px; }
.ood-title { font-size: 1rem; font-weight: 700; color: #991b1b; margin: 0 0 8px; }
.ood-body  { font-size: .8rem; color: #7f1d1d; margin: 0; line-height: 1.65; }
.ood-pill  { display: inline-block; margin-top: 12px; background: #fee2e2;
              border: 1px solid #fca5a5; color: #b91c1c; font-size: .71rem;
              font-weight: 600; padding: 3px 13px; border-radius: 20px;
              font-family: 'DM Mono', monospace; }

/* ── Result header card ── */
.res-hdr-card {
    border-radius: 10px; padding: 13px 15px; margin-bottom: 9px;
    border: 1.5px solid; display: flex; align-items: center; gap: 12px;
    position: relative; overflow: hidden;
}
.res-hdr-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.res-bio    { background: #f0fdf4; border-color: #86efac; }
.res-bio::before  { background: linear-gradient(90deg, #16a34a, #34d399); }
.res-nonbio { background: #fff7ed; border-color: #fdba74; }
.res-nonbio::before { background: linear-gradient(90deg, #ea580c, #f97316); }

.res-icon-wrap {
    width: 60px; height: 60px; background: white; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}
.res-info { flex: 1; min-width: 0; }
.res-stage-badge {
    display: inline-block; font-size: .6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    background: rgba(255,255,255,.7); border-radius: 20px;
    padding: 2px 8px; color: #64748b; margin-bottom: 3px;
}
.res-name { font-size: 1.35rem; font-weight: 700; color: #0f172a; margin: 0 0 5px; line-height: 1.2; }
.res-tags { display: flex; gap: 5px; flex-wrap: wrap; }
.res-tag  { font-size: .67rem; font-weight: 600; padding: 2px 9px;
             border-radius: 20px; border: 1px solid; }
.res-conf-label { font-size: .65rem; color: #94a3b8; margin: 5px 0 0; }
.res-ring { flex-shrink: 0; }

/* ── Top-5 predictions ── */
.top5-wrap { border: 1.5px solid #e2e8f0; border-radius: 10px; padding: 12px 13px; }
.top5-row  { display: flex; align-items: center; gap: 7px; margin-bottom: 6px; }
.top5-row:last-child { margin-bottom: 0; }
.top5-num  { font-size: .71rem; font-weight: 700; color: #94a3b8; width: 14px; flex-shrink: 0; }
.top5-lbl  { font-size: .73rem; color: #374151; width: 110px; flex-shrink: 0;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.top5-lbl.hi { font-weight: 700; color: #0f172a; }
.top5-track { flex: 1; height: 5px; background: #f1f5f9; border-radius: 3px; overflow: hidden; }
.top5-fill  { height: 100%; border-radius: 3px; background: #d1fae5; }
.top5-fill.hi { background: #16a34a; }
.top5-pct   { font-size: .69rem; color: #94a3b8; width: 36px; text-align: right;
               flex-shrink: 0; font-family: 'DM Mono', monospace; }
.top5-pct.hi { color: #16a34a; font-weight: 600; }

/* ── Hierarchical classification ── */
.hier-wrap { border: 1.5px solid #e2e8f0; border-radius: 10px; padding: 12px 13px; }
.hier-row  { display: flex; align-items: center; gap: 9px; }
.hier-lbl  { font-size: .7rem; font-weight: 700; color: #94a3b8;
              width: 46px; flex-shrink: 0; }
.hier-chip {
    display: flex; align-items: center; gap: 7px;
    background: #f8fafc; border: 1.5px solid #e2e8f0;
    border-radius: 8px; padding: 7px 12px; flex: 1;
}
.hier-chip-icon { font-size: .9rem; }
.hier-chip-name { font-size: .8rem; font-weight: 600; color: #0f172a; }
.hier-arrow { text-align: center; color: #cbd5e1; font-size: .85rem;
               padding: 3px 0 3px 55px; }

/* ── Disposal guide ── */
.disposal-wrap {
    border: 1.5px solid #e2e8f0; border-radius: 11px;
    padding: 14px 16px;
    display: flex; gap: 14px; align-items: flex-start;
}
.disposal-icon  { font-size: 2rem; flex-shrink: 0; margin-top: 2px; }
.disposal-title { font-size: .82rem; font-weight: 700; color: #0f172a; margin: 0 0 5px; }
.disposal-text  { font-size: .79rem; color: #374151; margin: 0; line-height: 1.65; }

/* ── Environmental impact ── */
.env-card {
    background: #f0fdf4; border: 1.5px solid #86efac;
    border-radius: 11px; padding: 14px 15px; height: 100%; box-sizing: border-box;
}
.env-title { font-size: .78rem; font-weight: 700; color: #16a34a; margin: 0 0 7px; }
.env-text  { font-size: .78rem; color: #166534; margin: 0; line-height: 1.65; }

/* ── Grad-CAM section ── */
.gcam-wrap { border: 1.5px solid #e2e8f0; border-radius: 10px; overflow: hidden; margin-top: 9px; }
.gcam-hdr  { background: #f8fafc; padding: 7px 13px; font-size: .67rem; font-weight: 700;
              letter-spacing: .06em; text-transform: uppercase; color: #64748b;
              border-bottom: 1px solid #e2e8f0; }
.gcam-body { padding: 9px 11px; }
.gcam-note { font-size: .7rem; color: #94a3b8; margin: 4px 0 0;
              text-align: center; font-style: italic; }

/* ── Model tag ── */
.model-tag-pill {
    display: inline-block; margin-top: 6px; font-size: .67rem; font-weight: 600;
    background: #f8fafc; border: 1px solid #e2e8f0; color: #64748b;
    padding: 2px 9px; border-radius: 20px;
}

/* ── Footer ── */
.wv-footer {
    margin-top: 10px; padding-top: 8px;
    border-top: 1px solid #e2e8f0;
    font-size: .62rem; color: #cbd5e1; text-align: center;
}
</style>
"""


# ============================================================
#  CONFIDENCE RING  (SVG)
# ============================================================
def confidence_ring(pct: float, color: str = "#16a34a", size: int = 100) -> str:
    r    = 36
    circ = 2 * 3.14159265 * r
    fill = circ * min(pct / 100, 1.0)
    rest = circ - fill
    return f"""
    <svg viewBox="0 0 88 88" width="{size}" height="{size}" style="display:block;">
      <circle cx="44" cy="44" r="{r}" fill="none" stroke="#e8f5e9" stroke-width="9"/>
      <circle cx="44" cy="44" r="{r}" fill="none" stroke="{color}" stroke-width="9"
        stroke-dasharray="{fill:.2f} {rest:.2f}"
        stroke-linecap="round"
        transform="rotate(-90 44 44)"/>
      <text x="44" y="42" text-anchor="middle" dominant-baseline="middle"
        font-family="DM Sans, sans-serif" font-size="13" font-weight="700"
        fill="#0f172a">{pct:.1f}%</text>
    </svg>"""


# ============================================================
#  RENDER: OOD
# ============================================================
def render_ood(max_conf: float, model_label: str) -> None:
    st.markdown(f"""
    <div class="ood-card">
      <div class="ood-icon">🤔</div>
      <p class="ood-title">Image Not Recognised</p>
      <p class="ood-body">
        This image doesn't match any of the <strong>36 waste categories</strong>
        I was trained on.<br><br>
        Please provide a clear photo of a waste item — plastic bottles, glass jars,
        cardboard, electronics, food scraps, batteries, clothing, etc.
      </p>
      <span class="ood-pill">Max confidence: {max_conf*100:.1f}% &nbsp;·&nbsp; {model_label}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  RENDER: RESULTS
# ============================================================
def render_results(result: dict, orig_img: Image.Image,
                   gradcam_model, show_gradcam: bool) -> None:

    if result["ood"]:
        render_ood(result["max_conf"], MODEL_LABELS[result["model_mode"]])
        return

    s1, s2, s3 = result["stage1"], result["stage2"], result["stage3"]
    meta      = STAGE1_META.get(s1, STAGE1_META["B-Non Biodegradable"])
    s2_emo    = STAGE2_EMOJI.get(s2, "📦")
    conf_pct  = result["conf"] * 100
    mode_lbl  = MODEL_LABELS[result["model_mode"]]
    card_cls  = "res-bio" if s1 == "A-Biodegradable" else "res-nonbio"

    # Stage tags
    tag_style = f'background:{meta["tag_bg"]};border-color:{meta["border"]};color:{meta["tag_color"]};'
    tags_html = f"""
      <span class="res-tag" style="{tag_style}">{clean_name(s2)}</span>
      <span class="res-tag" style="{tag_style}">{clean_name(s3)}</span>"""

    # ── Result header card ──────────────────────────────────
    ring_svg = confidence_ring(conf_pct, meta["color"], size=96)
    st.markdown(f"""
    <div class="res-hdr-card {card_cls}">
      <div class="res-icon-wrap">{s2_emo}</div>
      <div class="res-info">
        <span class="res-stage-badge">Stage 3 Result</span>
        <p class="res-name">{clean_name(s3)}</p>
        <div class="res-tags">{tags_html}</div>
        <p class="res-conf-label">Confidence Score</p>
      </div>
      <div class="res-ring">{ring_svg}</div>
    </div>
    <div class="model-tag-pill">🔬 {mode_lbl}</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:9px'></div>", unsafe_allow_html=True)

    # ── Top-5  |  Hierarchical ──────────────────────────────
    col_t5, col_hr = st.columns([1, 1], gap="small")

    with col_t5:
        st.markdown('<p class="sec-title">Top 5 Predictions</p>', unsafe_allow_html=True)
        top5    = result["top5"]
        max_p   = max(p for _, p in top5) or 1e-6
        rows_html = '<div class="top5-wrap">'
        for i, (cls_name, prob) in enumerate(top5):
            hi   = "hi" if i == 0 else ""
            rows_html += f"""
            <div class="top5-row">
              <span class="top5-num">{i+1}.</span>
              <span class="top5-lbl {hi}" title="{cls_name}">{clean_name(cls_name)}</span>
              <div class="top5-track">
                <div class="top5-fill {hi}" style="width:{prob/max_p*100:.1f}%"></div>
              </div>
              <span class="top5-pct {hi}">{prob*100:.2f}%</span>
            </div>"""
        rows_html += "</div>"
        st.markdown(rows_html, unsafe_allow_html=True)

    with col_hr:
        st.markdown('<p class="sec-title">Hierarchical Classification</p>', unsafe_allow_html=True)
        s1_icon = meta["icon"]
        st.markdown(f"""
        <div class="hier-wrap">
          <div class="hier-row">
            <span class="hier-lbl">Stage 1</span>
            <div class="hier-chip">
              <span class="hier-chip-icon">{s1_icon}</span>
              <span class="hier-chip-name">{clean_name(s1)}</span>
            </div>
          </div>
          <div class="hier-arrow">↓</div>
          <div class="hier-row">
            <span class="hier-lbl">Stage 2</span>
            <div class="hier-chip">
              <span class="hier-chip-icon">{s2_emo}</span>
              <span class="hier-chip-name">{clean_name(s2)}</span>
            </div>
          </div>
          <div class="hier-arrow">↓</div>
          <div class="hier-row">
            <span class="hier-lbl">Stage 3</span>
            <div class="hier-chip">
              <span class="hier-chip-icon">📌</span>
              <span class="hier-chip-name">{clean_name(s3)}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Grad-CAM ────────────────────────────────────────────
    if show_gradcam:
        st.markdown(
            '<div class="gcam-wrap">'
            '<div class="gcam-hdr">🔥 Grad-CAM — What the model focused on</div>'
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
                '<p class="gcam-note">Red / Yellow = high attention &nbsp;·&nbsp; Blue = low attention</p>',
                unsafe_allow_html=True)
        else:
            st.info("Grad-CAM could not be computed.")

        st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
#  MAIN
# ============================================================
def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # ── Session state defaults ─────────────────────────────
    defaults = {
        "current_img":   None,
        "result_cache":  {},
        "model_mode":    "elm",
        "input_method":  "camera",
        "infer_ms":      45,
        "history":       [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── SIDEBAR ────────────────────────────────────────────
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sb-brand">
          <div class="sb-logo">♻️</div>
          <div>
            <p class="sb-title">WasteVision AI</p>
            <p class="sb-tagline">Smart Waste. Better Future.</p>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Navigation — HTML for active item, buttons for clickable ones
        active_nav = st.session_state.get("active_nav", "home")

        st.markdown(f"""
        <div style="padding:0 8px;">
          <div class="{'nav-item-active' if active_nav == 'home' else 'nav-item-inactive'}">
            <span class="nav-icon">🏠</span> Home
          </div>
        </div>""", unsafe_allow_html=True)

        nav_col = st.container()
        with nav_col:
            if st.button("📷  Camera", key="nav_cam"):
                st.session_state["input_method"] = "camera"
                st.session_state["active_nav"]   = "camera"
                st.rerun()
            if st.button("📁  Upload Image", key="nav_upload"):
                st.session_state["input_method"] = "upload"
                st.session_state["active_nav"]   = "upload"
                st.rerun()
            if st.button("📊  History", key="nav_history"):
                st.session_state["active_nav"] = "history"
                st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # About card
        st.markdown("""
        <div class="sb-card">
          <p class="sb-card-title">About</p>
          <p class="sb-card-text">
            AI-powered waste classification system using EfficientNetB3
            and ELM ensemble for hierarchical waste categorisation across 36 classes.
          </p>
        </div>""", unsafe_allow_html=True)

        # Model Performance card
        infer_ms = st.session_state.get("infer_ms", 45)
        st.markdown(f"""
        <div class="sb-card" style="margin-top:8px;">
          <p class="sb-card-title">🏆 Model Performance</p>
          <p class="perf-val">94.33%</p>
          <p class="perf-sub">Accuracy</p>
          <div class="perf-sep"></div>
          <p class="perf-val" style="color:#0f172a;">{infer_ms}ms</p>
          <p class="perf-sub">Inference Time</p>
        </div>""", unsafe_allow_html=True)

    # ── Check for model artefacts ──────────────────────────
    missing = [p for p in [WEIGHTS_PATH, ELM_PATH, SCALER_PATH, CLASS_IDX_PATH]
               if not os.path.exists(p)]
    if missing:
        st.error("**Missing model artefact files.**  "
                 "Create a `weights/` folder next to `app.py` and add the required files.")
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

    # ── HISTORY VIEW ───────────────────────────────────────
    if st.session_state.get("active_nav") == "history":
        st.markdown("""
        <div class="page-hdr">
          <div>
            <p class="page-title">📊 Classification <span>History</span></p>
            <p class="page-sub">Your recent waste classifications this session</p>
          </div>
        </div>""", unsafe_allow_html=True)

        hist = st.session_state.get("history", [])
        if not hist:
            st.markdown("""
            <div class="placeholder">
              <span class="ph-icon">📊</span>
              <p class="ph-title">No history yet</p>
              <p class="ph-sub">Run a classification first to see results here.</p>
            </div>""", unsafe_allow_html=True)
        else:
            for i, h in enumerate(reversed(hist[-10:])):
                s2_emo = STAGE2_EMOJI.get(h["stage2"], "📦")
                st.markdown(f"""
                <div class="disposal-wrap" style="margin-bottom:8px;">
                  <div style="font-size:1.4rem">{s2_emo}</div>
                  <div>
                    <p style="font-size:.78rem;font-weight:700;margin:0 0 2px;color:#0f172a;">
                      {clean_name(h["stage3"])}
                    </p>
                    <p style="font-size:.7rem;color:#64748b;margin:0;">
                      {clean_name(h["stage1"])} → {clean_name(h["stage2"])} &nbsp;·&nbsp;
                      Confidence: {h["conf"]*100:.1f}% &nbsp;·&nbsp; {h["model_mode"].upper()}
                    </p>
                  </div>
                </div>""", unsafe_allow_html=True)

        if st.button("← Back to Home"):
            st.session_state["active_nav"] = "home"
            st.rerun()
        return

    # ── MAIN PAGE ──────────────────────────────────────────

    # Page header
    st.markdown("""
    <div class="page-hdr">
      <div>
        <p class="page-title">Welcome to <span>WasteVision AI</span></p>
        <p class="page-sub">AI-Powered Hierarchical Waste Classification System</p>
      </div>
      <div class="acc-badge">
        <p class="acc-badge-lbl">🧠 EfficientNetB3 + ELM</p>
        <p class="acc-badge-val">94.332% Accuracy</p>
        <p class="acc-badge-sub">36 Classes · 3 Stages</p>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Input method selector ──────────────────────────────
    cam_active  = st.session_state["input_method"] == "camera"
    upl_active  = st.session_state["input_method"] == "upload"
    cam_cls     = "im-btn active" if cam_active  else "im-btn"
    upl_cls     = "im-btn active" if upl_active  else "im-btn"

    st.markdown(f"""
    <div class="input-method-box">
      <p class="input-method-lbl">Select Input Method</p>
      <div class="im-btn-row">
        <div class="{cam_cls}">
          <span class="im-btn-icon">📷</span>
          <span class="im-btn-title">Camera</span>
          <span class="im-btn-sub">Capture image from webcam</span>
        </div>
        <div class="{upl_cls}">
          <span class="im-btn-icon">📁</span>
          <span class="im-btn-title">Upload Image</span>
          <span class="im-btn-sub">Choose image from your device</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Model selector (compact inline) ───────────────────
    with st.expander("⚙️ Model & Options", expanded=False):
        mode_opts   = list(MODEL_LABELS.values())
        mode_keys   = list(MODEL_LABELS.keys())
        default_idx = mode_keys.index(st.session_state["model_mode"])
        chosen_lbl  = st.selectbox("Prediction Model", mode_opts, index=default_idx)
        model_mode  = mode_keys[mode_opts.index(chosen_lbl)]
        st.session_state["model_mode"] = model_mode
        show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=False)

    model_mode   = st.session_state["model_mode"]
    show_gradcam = st.session_state.get("show_gradcam_state", False)

    # Re-read from expander widget (Streamlit doesn't persist toggle across reruns easily)
    # We handle this below through standard widget keys.

    # ── Two-column layout: Image Preview | Results ─────────
    col_img, col_res = st.columns([1, 1.2], gap="medium")

    # ─── LEFT: Image Preview ───────────────────────────────
    with col_img:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="sec-title">Image Preview</p>', unsafe_allow_html=True)

        if st.session_state["input_method"] == "camera":
            cam_data = st.camera_input(
                label="cam", label_visibility="collapsed",
                help="Click the shutter button to capture.",
            )
            if cam_data is not None:
                new_img = Image.open(cam_data).convert("RGB")
                if (st.session_state["current_img"] is None or
                        hash(new_img.tobytes()) != hash(st.session_state["current_img"].tobytes())):
                    st.session_state["current_img"] = new_img
                    for k in [k for k in st.session_state if k.startswith("gc_")]:
                        del st.session_state[k]
        else:
            uploaded = st.file_uploader(
                label="upload", label_visibility="collapsed",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
            )
            if uploaded is not None:
                new_img = Image.open(uploaded).convert("RGB")
                if (st.session_state["current_img"] is None or
                        hash(new_img.tobytes()) != hash(st.session_state["current_img"].tobytes())):
                    st.session_state["current_img"] = new_img
                    for k in [k for k in st.session_state if k.startswith("gc_")]:
                        del st.session_state[k]

            cur = st.session_state.get("current_img")
            if cur is not None:
                st.image(cur, use_container_width=True)
            else:
                st.markdown("""
                <div class="placeholder">
                  <span class="ph-icon">🖼️</span>
                  <p class="ph-title">No image yet</p>
                  <p class="ph-sub">Upload a photo to get started.</p>
                </div>""", unsafe_allow_html=True)

        # Retake / Clear buttons
        if st.session_state.get("current_img") is not None:
            b1, b2 = st.columns(2, gap="small")
            with b1:
                if st.button("🔄  Retake / New", use_container_width=True):
                    st.session_state["current_img"] = None
                    st.session_state["result_cache"] = {}
                    for k in [k for k in st.session_state if k.startswith("gc_")]:
                        del st.session_state[k]
                    st.rerun()
            with b2:
                if st.button("🗑️  Clear", use_container_width=True):
                    st.session_state["current_img"] = None
                    st.session_state["result_cache"] = {}
                    for k in [k for k in st.session_state if k.startswith("gc_")]:
                        del st.session_state[k]
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ─── RIGHT: Classification Results ────────────────────
    with col_res:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="sec-title">Classification Result</p>', unsafe_allow_html=True)

        current_img = st.session_state.get("current_img")

        if current_img is None:
            st.markdown("""
            <div class="placeholder">
              <span class="ph-icon">🗑️</span>
              <p class="ph-title">No image yet</p>
              <p class="ph-sub">Capture via webcam or upload a photo<br>
              to get an instant 3-stage waste classification.</p>
            </div>""", unsafe_allow_html=True)
        else:
            img_hash  = hash(current_img.tobytes())
            cache_key = (img_hash, model_mode)

            if cache_key not in st.session_state["result_cache"]:
                with st.spinner("Classifying…"):
                    result = run_inference(
                        current_img, model_mode,
                        cnn_model, feat_ext, elm_payload, scaler, class_indices,
                    )
                st.session_state["result_cache"][cache_key] = result
                # Update infer_ms in sidebar
                if "infer_ms" in result:
                    st.session_state["infer_ms"] = result["infer_ms"]
                # Save to history
                if not result.get("ood"):
                    st.session_state["history"].append({
                        "stage1": result["stage1"],
                        "stage2": result["stage2"],
                        "stage3": result["stage3"],
                        "conf":   result["conf"],
                        "model_mode": model_mode,
                    })
                # Trim cache
                if len(st.session_state["result_cache"]) > 9:
                    oldest = next(iter(st.session_state["result_cache"]))
                    del st.session_state["result_cache"][oldest]

            cached = st.session_state["result_cache"].get(cache_key)
            if cached is not None:
                # Pass show_gradcam from a dedicated key so it persists
                render_results(cached, current_img, gradcam_model,
                               st.session_state.get("_gradcam", False))

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Grad-CAM toggle (separate widget so it persists) ──
    with st.expander("🔥 Grad-CAM Heatmap", expanded=False):
        gcam_on = st.toggle("Enable Grad-CAM overlay", value=st.session_state.get("_gradcam", False))
        st.session_state["_gradcam"] = gcam_on
        if gcam_on and st.session_state.get("current_img") is not None:
            cached = st.session_state["result_cache"].get(
                (hash(st.session_state["current_img"].tobytes()), model_mode)
            )
            if cached and not cached.get("ood"):
                img_arr = preprocess_image(st.session_state["current_img"])
                gc_key  = f"gc_{cached['class_idx']}"
                cam_img = st.session_state.get(gc_key)
                if cam_img is None:
                    with st.spinner("Computing Grad-CAM…"):
                        heatmap = compute_gradcam(gradcam_model, img_arr, cached["class_idx"])
                    if heatmap is not None:
                        cam_img = overlay_heatmap(st.session_state["current_img"], heatmap)
                        st.session_state[gc_key] = cam_img
                if cam_img is not None:
                    g1, g2 = st.columns(2)
                    with g1:
                        st.image(st.session_state["current_img"]
                                 .convert("RGB").resize((224, 224)),
                                 caption="Original", use_container_width=True)
                    with g2:
                        st.image(cam_img, caption="Grad-CAM Overlay",
                                 use_container_width=True)
                    st.caption("🔴 Red/Yellow = high attention &nbsp;·&nbsp; 🔵 Blue = low attention")
                else:
                    st.info("Grad-CAM unavailable — 'top_activation' layer not found.")

    # ── Disposal Guidance + Environmental Impact ───────────
    current_img = st.session_state.get("current_img")
    cached = None
    if current_img is not None:
        cached = st.session_state["result_cache"].get(
            (hash(current_img.tobytes()), model_mode)
        )

    if cached and not cached.get("ood"):
        s2 = cached["stage2"]
        s2_emo  = STAGE2_EMOJI.get(s2, "📦")
        guide   = DISPOSAL_GUIDE.get(s2, "Consult your local waste management authority.")
        env_txt = ENVIRONMENTAL_IMPACT.get(s2, "Proper waste disposal protects our environment.")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        d_col, e_col = st.columns([1.65, 1], gap="medium")

        with d_col:
            st.markdown(f"""
            <div class="disposal-wrap">
              <div class="disposal-icon">{s2_emo}</div>
              <div>
                <p class="disposal-title">Disposal Guidance</p>
                <p class="disposal-text">{guide}</p>
              </div>
            </div>""", unsafe_allow_html=True)

        with e_col:
            st.markdown(f"""
            <div class="env-card">
              <p class="env-title">🌍 Environmental Impact</p>
              <p class="env-text">{env_txt}</p>
            </div>""", unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────
    st.markdown(
        "<div class='wv-footer'>"
        "WasteVision AI &nbsp;·&nbsp; EfficientNetB3 backbone "
        "&nbsp;·&nbsp; 2-Phase Fine-tuning &nbsp;·&nbsp; "
        "Borderline-SMOTE &nbsp;·&nbsp; 7× ELM Ensemble "
        "&nbsp;·&nbsp; TTA &nbsp;·&nbsp; Grad-CAM"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
