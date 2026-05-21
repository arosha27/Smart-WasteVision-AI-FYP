

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
    page_icon="recycle",
    layout="wide",
    initial_sidebar_state="collapsed",
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

# Out-of-distribution threshold: reject if max-softmax < this
OOD_THRESHOLD = 0.30

MODEL_LABELS = {
    "cnn": "CNN Only  (EfficientNetB3)",
    "tta": "CNN + TTA  (5-Pass Augmentation)",
    "elm": "CNN + ELM Ensemble  (7 x Voting)",
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

STAGE2_ICON = {
    "A_Green Waste": "leaf",
    "B_Recyclable Waste": "recycle",
    "C_Glass": "glass",
    "D_Metal": "wrench",
    "E_Polymer (Petrolium Based)": "bottle",
    "F_Leather and Fabric": "shirt",
    "G_Medical Waste": "hospital",
    "H_E Waste": "computer",
    "I_Hazardous Waste": "warning",
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
#  GRAD-CAM MODEL BUILDER (called inside cache)
# ============================================================
def _build_gradcam_model(cnn_model, backbone):
    """
    Return a Functional model that outputs [last_conv_output, predictions].
    All weights are shared with the original cnn_model - zero extra memory.
    Returns None if construction fails.
    """
    # Locate last spatial (4-D) layer in EfficientNetB3 backbone
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
        # Sub-model: backbone_in -> [last_conv_out, backbone_out]
        backbone_dual = tf.keras.Model(
            inputs  = backbone.inputs,
            outputs = [last_conv.output, backbone.output],
            name    = "backbone_dual",
        )

        # Dense layers from the head (in order: 512, 256, 36)
        dense_layers = [l for l in cnn_model.layers
                        if isinstance(l, tf.keras.layers.Dense)]
        if len(dense_layers) < 3:
            return None

        # Build new Functional model sharing the same layer objects
        gc_inp   = tf.keras.Input(shape=(224, 224, 3), name="gc_input")
        x_prep   = tf.keras.layers.Lambda(efficientnet_preprocess, name="gc_prep")(gc_inp)
        conv_out, bb_out = backbone_dual(x_prep, training=False)

        x        = tf.keras.layers.GlobalAveragePooling2D()(bb_out)
        x        = dense_layers[0](x)                               # 512 units
        x        = tf.keras.layers.Dropout(0.5)(x, training=False)
        x        = dense_layers[1](x)                               # 256 units (feature_vector)
        x        = tf.keras.layers.Dropout(0.4)(x, training=False)
        preds    = dense_layers[2](x)                               # 36 units (softmax)

        return tf.keras.Model(inputs=gc_inp, outputs=[conv_out, preds],
                              name="gradcam_model")
    except Exception:
        return None


# ============================================================
#  ARTEFACT LOADING  (cached once per session)
# ============================================================
@st.cache_resource(show_spinner="Loading WasteVision model...")
def load_all_artefacts():
    # 1. Rebuild architecture (identical to training)
    backbone = EfficientNetB3(input_shape=(224, 224, 3),
                              include_top=False, weights=None)
    backbone.trainable = True  # irrelevant at inference time

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

    # 2. Load trained weights (avoids Lambda/preprocess_input serialisation issues)
    cnn_model.load_weights(WEIGHTS_PATH)

    # 3. ELM ensemble
    with open(ELM_PATH, "rb") as fh:
        elm_payload = pickle.load(fh)

    # 4. Feature scaler + class indices
    scaler = joblib.load(SCALER_PATH)
    with open(CLASS_IDX_PATH) as fh:
        raw = json.load(fh)
    class_indices = {int(k): v for k, v in raw.items()}

    # 5. Grad-CAM model (shared weights, no extra memory)
    gradcam_model = _build_gradcam_model(cnn_model, backbone)

    return cnn_model, feature_extractor, elm_payload, scaler, class_indices, gradcam_model


# ============================================================
#  INFERENCE
# ============================================================
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Return (1, 224, 224, 3) float32 in [0, 255].
    The Lambda layer inside the model calls efficientnet_preprocess internally."""
    arr = np.array(img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS), dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def check_ood(probs: np.ndarray) -> tuple:
    max_conf = float(np.max(probs))
    # Normalised entropy: 0=fully certain, 1=uniform over all classes
    entropy  = -np.sum(probs * np.log(probs + 1e-10))
    norm_ent = float(entropy / np.log(len(probs)))
    ood      = (max_conf < OOD_THRESHOLD) or (norm_ent > 0.92)
    return ood, max_conf, norm_ent


# TTA datagen matches training notebook exactly
_TTA_DATAGEN = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10,
    brightness_range=[0.85, 1.15],
    # No rescale — model handles normalisation internally
)

def infer_cnn(cnn_model, img_array: np.ndarray) -> tuple:
    probs = cnn_model.predict(img_array, verbose=0)[0]
    return probs, int(np.argmax(probs))


def infer_tta(cnn_model, img_array: np.ndarray, n_tta: int = 5) -> tuple:
    """n_tta augmented passes + 1 original, then average probability vectors."""
    img_np    = img_array[0]   # (224, 224, 3)
    all_probs = [cnn_model.predict(img_array, verbose=0)[0]]  # original
    for _ in range(n_tta):
        aug   = _TTA_DATAGEN.random_transform(img_np.copy())
        probs = cnn_model.predict(np.expand_dims(aug, 0), verbose=0)[0]
        all_probs.append(probs)
    avg = np.mean(all_probs, axis=0)
    return avg, int(np.argmax(avg))


def infer_elm(cnn_model, feat_ext, elm_payload, scaler,
              img_array: np.ndarray) -> tuple:
    """CNN softmax for confidence display; ELM majority vote for final class."""
    probs       = cnn_model.predict(img_array, verbose=0)[0]
    feat_raw    = feat_ext.predict(img_array, verbose=0)         # (1, 256)
    feat_scaled = scaler.transform(feat_raw)
    votes = []
    for e in elm_payload:
        H = np.maximum(0.0, feat_scaled @ e["W"] + e["b"])      # relu
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
    else:  # elm
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
#  GRAD-CAM
# ============================================================
def compute_gradcam(gradcam_model, img_array: np.ndarray,
                    class_idx: int) -> np.ndarray:
    """
    Standard Grad-CAM: pool (d_score / d_A_k) -> weight each channel k of A.
    Returns (H, W) float32 heatmap in [0, 1], or None on failure.
    GradientTape differentiates predictions w.r.t. conv_outputs because TF
    records ALL operations on tensors derived from watched Variables within
    the tape context — no explicit tape.watch() needed for intermediate tensors.
    """
    if gradcam_model is None:
        return None
    try:
        x = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = gradcam_model(x)
            target = predictions[:, class_idx]

        grads = tape.gradient(target, conv_outputs)  # (1, H, W, C)
        if grads is None:
            return None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))            # (C,)
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled, axis=-1) # (H, W)
        heatmap = tf.nn.relu(heatmap).numpy()
        vmax    = heatmap.max()
        return heatmap / vmax if vmax > 0 else heatmap
    except Exception:
        return None


def overlay_heatmap(orig_img: Image.Image, heatmap: np.ndarray,
                    alpha: float = 0.45) -> Image.Image:
    """Jet-coloured Grad-CAM heatmap blended over the resized original."""
    orig        = np.array(orig_img.convert("RGB").resize((224, 224)), dtype=float)
    heat_small  = Image.fromarray((heatmap * 255).astype(np.uint8))
    heat_rs     = np.array(heat_small.resize((224, 224), Image.LANCZOS)) / 255.0
    colored     = (cm.jet(heat_rs)[:, :, :3] * 255)
    blended     = np.clip((1 - alpha) * orig + alpha * colored, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ============================================================
#  CSS - Modern UI with Dark/Light Theme Support
# ============================================================
def get_theme_css(theme: str = "light") -> str:
    """Generate CSS based on selected theme (light/dark)."""
    
    if theme == "dark":
        # Dark Theme Colors
        bg_primary = "#0f172a"
        bg_secondary = "#1e293b"
        bg_card = "#334155"
        bg_card_bio = "#14532d"
        bg_card_nonbio = "#7c2d12"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        border_color = "#475569"
        border_bio = "#166534"
        border_nonbio = "#c2410c"
        accent_green = "#22c55e"
        accent_blue = "#3b82f6"
        shadow_color = "rgba(0,0,0,0.4)"
        placeholder_bg = "#1e293b"
        placeholder_border = "#475569"
        gradcam_header = "#1e293b"
        gradcam_border = "#475569"
        ood_bg = "#450a0a"
        ood_border = "#dc2626"
        ood_text = "#fca5a5"
        ood_conf_bg = "#7f1d1d"
        ood_conf_border = "#dc2626"
        ood_conf_text = "#fca5a5"
        model_tag_bg = "#334155"
        model_tag_border = "#475569"
        model_tag_text = "#94a3b8"
        conf_pill_bg = "#1e3a8a"
        conf_pill_text = "#60a5fa"
        stage_1_bio_bg = "#14532d"
        stage_1_bio_text = "#86efac"
        stage_1_bio_border = "#166534"
        stage_1_nb_bg = "#7c2d12"
        stage_1_nb_text = "#fdba74"
        stage_1_nb_border = "#c2410c"
        stage_2_bg = "#312e81"
        stage_2_text = "#a5b4fc"
        stage_2_border = "#4338ca"
        stage_3_bg = "#1e3a8a"
        stage_3_text = "#93c5fd"
        cbar_track = "#334155"
        disposal_bg = "#334155"
        disposal_border = "#475569"
        disposal_title = "#94a3b8"
        disposal_text = "#e2e8f0"
        header_border = "#475569"
        badge_bg = "#14532d"
        badge_border = "#166534"
        badge_text = "#86efac"
    else:
        # Light Theme Colors
        bg_primary = "#ffffff"
        bg_secondary = "#f8fafc"
        bg_card = "#ffffff"
        bg_card_bio = "#f0fdf4"
        bg_card_nonbio = "#fff7ed"
        text_primary = "#0f172a"
        text_secondary = "#64748b"
        text_muted = "#94a3b8"
        border_color = "#e2e8f0"
        border_bio = "#86efac"
        border_nonbio = "#fdba74"
        accent_green = "#16a34a"
        accent_blue = "#3b82f6"
        shadow_color = "rgba(0,0,0,0.1)"
        placeholder_bg = "#f8fafc"
        placeholder_border = "#cbd5e1"
        gradcam_header = "#f8fafc"
        gradcam_border = "#e2e8f0"
        ood_bg = "#fef2f2"
        ood_border = "#fca5a5"
        ood_text = "#991b1b"
        ood_conf_bg = "#fee2e2"
        ood_conf_border = "#fca5a5"
        ood_conf_text = "#b91c1c"
        model_tag_bg = "#f8fafc"
        model_tag_border = "#e2e8f0"
        model_tag_text = "#64748b"
        conf_pill_bg = "#dbeafe"
        conf_pill_text = "#1d4ed8"
        stage_1_bio_bg = "#f0fdf4"
        stage_1_bio_text = "#166534"
        stage_1_bio_border = "#bbf7d0"
        stage_1_nb_bg = "#fff7ed"
        stage_1_nb_text = "#9a3412"
        stage_1_nb_border = "#fed7aa"
        stage_2_bg = "#f5f3ff"
        stage_2_text = "#5b21b6"
        stage_2_border = "#ddd6fe"
        stage_3_bg = "#eff6ff"
        stage_3_text = "#1d4ed8"
        cbar_track = "#f1f5f9"
        disposal_bg = "#fafafa"
        disposal_border = "#e2e8f0"
        disposal_title = "#64748b"
        disposal_text = "#374151"
        header_border = "#e2e8f0"
        badge_bg = "#f0fdf4"
        badge_border = "#86efac"
        badge_text = "#16a34a"
    
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Styles */
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.main .block-container {{
    padding-top: .5rem !important;
    padding-bottom: .3rem !important;
    max-width: 1400px !important;
    background-color: {bg_primary};
}}
#MainMenu, footer, header {{ visibility: hidden; }}

/* Scrollbar Styling */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: {bg_secondary}; }}
::-webkit-scrollbar-thumb {{ background: {border_color}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {text_secondary}; }}

/* Header */
.wv-hdr {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
    background: linear-gradient(135deg, {bg_secondary} 0%, {bg_primary} 100%);
    border-radius: 16px;
    border: 1px solid {border_color};
    box-shadow: 0 4px 20px {shadow_color};
    margin-bottom: 16px;
}}
.wv-logo {{
    width: 48px;
    height: 48px;
    border-radius: 14px;
    flex-shrink: 0;
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
}}
.wv-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {text_primary};
    margin: 0;
    line-height: 1.1;
}}
.wv-sub {{
    font-size: 0.72rem;
    color: {text_secondary};
    margin: 2px 0 0;
    letter-spacing: 0.05em;
    font-weight: 500;
}}
.wv-badge {{
    margin-left: auto;
    background: {badge_bg};
    border: 1px solid {badge_border};
    color: {badge_text};
    font-size: 0.7rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 24px;
    white-space: nowrap;
    box-shadow: 0 2px 8px {shadow_color};
}}

/* Micro labels */
.mlabel {{
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {text_secondary};
    margin: 0 0 8px;
}}

/* Compact Image Preview Container */
.img-preview-container {{
    width: 100%;
    max-width: 280px;
    margin: 0 auto 16px;
    border-radius: 16px;
    overflow: hidden;
    border: 2px solid {border_color};
    box-shadow: 0 4px 16px {shadow_color};
    background: {bg_secondary};
    display: flex;
    align-items: center;
    justify-content: center;
}}
.img-preview-container img {{
    width: 100%;
    height: auto;
    border-radius: 14px;
}}

/* Prediction Card */
.pred-card {{
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
    border: 2px solid;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 16px {shadow_color};
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.pred-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 24px {shadow_color};
}}
.pred-card::after {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
}}
.card-bio {{
    background: {bg_card_bio};
    border-color: {border_bio};
}}
.card-bio::after {{
    background: linear-gradient(90deg, #22c55e, #16a34a);
}}
.card-nonbio {{
    background: {bg_card_nonbio};
    border-color: {border_nonbio};
}}
.card-nonbio::after {{
    background: linear-gradient(90deg, #ea580c, #f97316);
}}
.pred-row {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
}}
.pred-name {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {text_primary};
    margin: 0;
    line-height: 1.2;
}}
.pred-raw {{
    font-size: 0.7rem;
    color: {text_muted};
    margin: 3px 0 0;
    font-family: 'JetBrains Mono', monospace;
}}
.conf-pill {{
    flex-shrink: 0;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 24px;
    background: {conf_pill_bg};
    color: {conf_pill_text};
    white-space: nowrap;
    box-shadow: 0 2px 8px {shadow_color};
}}
.model-tag {{
    display: inline-block;
    margin-top: 8px;
    font-size: 0.68rem;
    font-weight: 600;
    background: {model_tag_bg};
    border: 1px solid {model_tag_border};
    color: {model_tag_text};
    padding: 4px 10px;
    border-radius: 12px;
}}

/* Stage Flow */
.stage-flow {{
    display: flex;
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid {border_color};
    margin-bottom: 12px;
    box-shadow: 0 2px 12px {shadow_color};
}}
.stage-step {{
    flex: 1;
    padding: 10px 14px;
    position: relative;
}}
.stage-step + .stage-step::before {{
    content: "▶";
    position: absolute;
    left: -12px;
    top: 50%;
    transform: translateY(-50%);
    color: {text_muted};
    font-size: 0.6rem;
    z-index: 1;
}}
.ss-label {{
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 4px;
    opacity: 0.7;
}}
.ss-icon {{
    font-size: 1rem;
    margin-bottom: 3px;
    display: block;
}}
.ss-name {{
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0;
    line-height: 1.3;
}}
.ss-1bio {{
    background: {stage_1_bio_bg};
    color: {stage_1_bio_text};
    border-right: 2px solid {stage_1_bio_border};
}}
.ss-1nb {{
    background: {stage_1_nb_bg};
    color: {stage_1_nb_text};
    border-right: 2px solid {stage_1_nb_border};
}}
.ss-2 {{
    background: {stage_2_bg};
    color: {stage_2_text};
    border-right: 2px solid {stage_2_border};
}}
.ss-3 {{
    background: {stage_3_bg};
    color: {stage_3_text};
}}

/* Confidence Bars */
.cbar-wrap {{ margin-bottom: 12px; }}
.cbar-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}}
.cbar-lbl {{
    font-size: 0.74rem;
    color: {text_primary};
    width: 150px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.cbar-lbl.hi {{
    font-weight: 700;
    color: {text_primary};
}}
.cbar-track {{
    flex: 1;
    height: 8px;
    background: {cbar_track};
    border-radius: 6px;
    overflow: hidden;
}}
.cbar-fill {{
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, {accent_blue}, #60a5fa);
    transition: width 0.4s ease;
}}
.cbar-fill.hi {{
    background: linear-gradient(90deg, #22c55e, #16a34a);
}}
.cbar-pct {{
    font-size: 0.7rem;
    color: {text_muted};
    width: 40px;
    text-align: right;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}}
.cbar-pct.hi {{
    color: #22c55e;
    font-weight: 600;
}}

/* Disposal Guide */
.disp-card {{
    background: {disposal_bg};
    border-radius: 14px;
    border: 2px solid {disposal_border};
    padding: 12px 16px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 10px;
    box-shadow: 0 2px 12px {shadow_color};
}}
.disp-icon {{
    font-size: 1.4rem;
    flex-shrink: 0;
    margin-top: 2px;
}}
.disp-title {{
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {disposal_title};
    margin: 0 0 4px;
}}
.disp-text {{
    font-size: 0.8rem;
    color: {disposal_text};
    margin: 0;
    line-height: 1.6;
}}

/* Grad-CAM Section */
.gcam-wrap {{
    border-radius: 14px;
    border: 2px solid {gradcam_border};
    overflow: hidden;
    margin-top: 10px;
    box-shadow: 0 2px 12px {shadow_color};
}}
.gcam-hdr {{
    background: {gradcam_header};
    padding: 10px 16px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {text_secondary};
    border-bottom: 2px solid {gradcam_border};
}}
.gcam-body {{ padding: 12px 14px; }}
.gcam-note {{
    font-size: 0.73rem;
    color: {text_muted};
    margin: 6px 0 0;
    text-align: center;
    font-style: italic;
}}

/* OOD Card */
.ood-card {{
    border-radius: 16px;
    background: {ood_bg};
    border: 2px solid {ood_border};
    padding: 24px 28px;
    text-align: center;
    margin-top: 8px;
    box-shadow: 0 4px 20px {shadow_color};
}}
.ood-icon {{ font-size: 3rem; margin-bottom: 12px; }}
.ood-title {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {ood_text};
    margin: 0 0 12px;
}}
.ood-body {{
    font-size: 0.88rem;
    color: {ood_text};
    margin: 0;
    line-height: 1.7;
}}
.ood-conf {{
    display: inline-block;
    margin-top: 16px;
    background: {ood_conf_bg};
    border: 1px solid {ood_conf_border};
    color: {ood_conf_text};
    font-size: 0.75rem;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 24px;
    font-family: 'JetBrains Mono', monospace;
}}

/* Placeholder */
.placeholder {{
    min-height: 360px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: {placeholder_bg};
    border-radius: 16px;
    border: 2px dashed {placeholder_border};
    color: {text_muted};
    text-align: center;
    padding: 32px;
    box-shadow: 0 2px 12px {shadow_color};
}}
.ph-icon {{ font-size: 3rem; margin-bottom: 12px; opacity: 0.5; }}
.ph-title {{
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 6px;
    color: {text_secondary};
}}
.ph-sub {{ font-size: 0.82rem; margin: 0; line-height: 1.6; }}

/* Sidebar Styling */
.css-1d391kg {{
    background: linear-gradient(180deg, #14532d 0%, #166534 50%, #22c55e 100%) !important;
}}
.css-1d391kg .css-17eq0hr {{
    color: #ffffff !important;
}}
.css-1d391kg .css-1v0mbdj {{
    color: #bbf7d0 !important;
}}
.css-1d391kg .css-1cypcdb {{
    color: #ffffff !important;
}}

/* Streamlit Component Overrides */
.stSelectbox > div > div > div {{
    background-color: {bg_secondary};
    border-color: {border_color};
}}
.stSelectbox [data-testid="stSelectbox"] {{
    background-color: {bg_secondary};
    border-color: {border_color};
    color: {text_primary};
}}
.stToggle [data-testid="stToggleLabel"] {{
    color: {text_primary};
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.85rem;
    font-weight: 600;
    padding: 8px 20px;
    border-radius: 10px 10px 0 0;
    background-color: {bg_secondary};
    color: {text_secondary};
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background-color: {bg_primary};
    color: {text_primary};
    border-bottom: 2px solid {accent_green};
}}
div[data-testid="stCameraInput"] > div {{
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid {border_color};
}}
.stFileUploader > div {{
    border-radius: 14px;
    border: 2px dashed {border_color};
    background-color: {bg_secondary};
}}
.stFileUploader > div:hover {{
    border-color: {accent_green};
}}

/* Button Styling */
.stButton > button {{
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    transition: all 0.2s ease;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(34, 197, 94, 0.4);
}}

/* Info/Success/Error Messages */
.stAlert {{
    border-radius: 12px;
    border: 2px solid {border_color};
    box-shadow: 0 2px 12px {shadow_color};
}}
</style>
"""


# ============================================================
#  RESULT RENDERERS
# ============================================================
def render_ood(max_conf: float, model_label: str) -> None:
    st.markdown(f"""
    <div class="ood-card">
      <div class="ood-icon">🤔</div>
      <p class="ood-title">I can't classify this image</p>
      <p class="ood-body">
        This image doesn't appear to belong to any of the
        <strong>36 waste categories</strong> I was trained on.<br><br>
        Please provide a clear photo of a waste item such as plastic bottles,
        glass jars, cardboard, electronics, food scraps, batteries, clothing, etc.
      </p>
      <span class="ood-conf">max confidence: {max_conf*100:.1f}%
        &nbsp;·&nbsp;{model_label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_results(result: dict, orig_img: Image.Image,
                   gradcam_model, show_gradcam: bool) -> None:

    if result["ood"]:
        render_ood(result["max_conf"], MODEL_LABELS[result["model_mode"]])
        return

    s1, s2, s3 = result["stage1"], result["stage2"], result["stage3"]
    meta    = STAGE1_META.get(s1, {"card": "card-nonbio", "icon": "🗑️"})
    s2_emo  = STAGE2_EMOJI.get(s2, "📦")
    mode_lbl= MODEL_LABELS[result["model_mode"]]
    conf_pct= result["conf"] * 100

    # Predicted item
    st.markdown('<p class="mlabel">Predicted Item</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pred-card {meta['card']}">
      <div class="pred-row">
        <div>
          <p class="pred-name">{meta['icon']} {clean_name(s3)}</p>
          <p class="pred-raw">{s3}</p>
        </div>
        <span class="conf-pill">{conf_pct:.1f}%</span>
      </div>
      <span class="model-tag">&#x1F52C; {mode_lbl}</span>
    </div>""", unsafe_allow_html=True)

    # Stage hierarchy
    st.markdown('<p class="mlabel">Classification Hierarchy</p>', unsafe_allow_html=True)
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
        <span class="ss-icon">&#x1F4CC;</span>
        <p class="ss-label">Stage 3</p>
        <p class="ss-name">{clean_name(s3)}</p>
      </div>
    </div>""", unsafe_allow_html=True)

    # Confidence bars
    st.markdown('<p class="mlabel">Top-5 CNN Probabilities</p>', unsafe_allow_html=True)
    top5  = result["top5"]
    max_p = max(p for _, p in top5) or 1e-6
    bars  = '<div class="cbar-wrap">'
    for i, (cls_name, prob) in enumerate(top5):
        hi   = "hi" if i == 0 else ""
        bars += f"""
        <div class="cbar-row">
          <span class="cbar-lbl {hi}" title="{cls_name}">{clean_name(cls_name)}</span>
          <div class="cbar-track">
            <div class="cbar-fill {hi}" style="width:{prob/max_p*100:.1f}%"></div>
          </div>
          <span class="cbar-pct {hi}">{prob*100:.1f}%</span>
        </div>"""
    bars += "</div>"
    st.markdown(bars, unsafe_allow_html=True)

    # Disposal guide
    guide = DISPOSAL_GUIDE.get(s2, "Consult your local waste management authority.")
    st.markdown(f"""
    <div class="disp-card">
      <span class="disp-icon">{s2_emo}</span>
      <div>
        <p class="disp-title">Disposal Guide &middot; {clean_name(s2)}</p>
        <p class="disp-text">{guide}</p>
      </div>
    </div>""", unsafe_allow_html=True)

    # Grad-CAM section
    if show_gradcam:
        st.markdown(
            '<div class="gcam-wrap">'
            '<div class="gcam-hdr">&#x1F525; Grad-CAM &mdash; What the model focused on</div>'
            '<div class="gcam-body">',
            unsafe_allow_html=True)

        gc_key = f"gc_{result['class_idx']}"
        cam_img = st.session_state.get(gc_key)

        if cam_img is None:
            with st.spinner("Computing Grad-CAM..."):
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
                '<p class="gcam-note">Red/Yellow = high attention &nbsp;&#183;&nbsp; Blue = low attention</p>',
                unsafe_allow_html=True)
        else:
            st.info("Grad-CAM could not be computed. Check that 'top_activation' "
                    "layer exists in the EfficientNetB3 backbone.")

        st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
#  MAIN
# ============================================================
def main() -> None:
    # Initialize theme in session state
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state["theme"]), unsafe_allow_html=True)
    
    # Sidebar with theme toggle
    with st.sidebar:
        st.markdown("""
        <div style="padding: 20px 10px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 1.4rem; font-weight: 700;">⚙️ Settings</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        
        # Theme toggle
        theme_option = st.radio(
            "🎨 Theme",
            ["Light", "Dark"],
            label_visibility="visible",
            horizontal=True,
            key="theme_selector"
        )
        st.session_state["theme"] = theme_option.lower()
        
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        # Model info in sidebar
        st.markdown("""
        <div style="padding: 15px; background: rgba(255,255,255,0.1); border-radius: 12px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="color: white; margin: 0; font-size: 0.85rem; font-weight: 600;">📊 Model Info</p>
            <p style="color: rgba(255,255,255,0.8); margin: 8px 0 0; font-size: 0.75rem; line-height: 1.5;">
                • EfficientNetB3 Backbone<br>
                • 7x ELM Ensemble<br>
                • 36 Waste Classes<br>
                • 3-Stage Hierarchy
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="wv-hdr">
      <div class="wv-logo">&#9851;</div>
      <div>
        <p class="wv-title">WasteVision AI</p>
        <p class="wv-sub">AUTOMATED THREE-STAGE WASTE CLASSIFICATION SYSTEM</p>
      </div>
      <span class="wv-badge">36 Classes &nbsp;&#183;&nbsp; 3 Stages</span>
    </div>""", unsafe_allow_html=True)

    # Check artefacts exist
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

    # Session state defaults
    for k, v in [("current_img", None), ("img_hash", None),
                 ("result_cache", {}), ("model_mode", "elm")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Two-column layout
    col_left, col_right = st.columns([1, 1.3], gap="large")

    # ─── LEFT: Controls + Image Input ────────────────────────
    with col_left:

        # Model selector
        st.markdown('<p class="mlabel">Prediction Model</p>', unsafe_allow_html=True)
        mode_options  = list(MODEL_LABELS.values())
        mode_keys     = list(MODEL_LABELS.keys())
        default_idx   = mode_keys.index(st.session_state["model_mode"])
        chosen_label  = st.selectbox(
            label="model_select",
            options=mode_options,
            index=default_idx,
            label_visibility="collapsed",
        )
        model_mode = mode_keys[mode_options.index(chosen_label)]
        st.session_state["model_mode"] = model_mode

        # Grad-CAM toggle
        show_gradcam = st.toggle("Show Grad-CAM heatmap", value=False)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Input tabs
        tab_cam, tab_up = st.tabs(["  Webcam", "  Upload"])

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
                # Clear Grad-CAM cache when image changes
                for k in [k for k in st.session_state if k.startswith("gc_")]:
                    del st.session_state[k]
                
                # Show compact preview for webcam
                resized = new_img.resize((280, 280), Image.LANCZOS)
                st.markdown('<div class="img-preview-container">', unsafe_allow_html=True)
                st.image(resized, use_container_width=False, width=280)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab_up:
            st.markdown('<p class="mlabel">Browse or drag-and-drop</p>',
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

            # Show compact preview for upload tab
            cur = st.session_state.get("current_img")
            if cur is not None and cam_data is None:
                # Resize image for compact preview
                resized = cur.resize((280, 280), Image.LANCZOS)
                st.markdown('<div class="img-preview-container">', unsafe_allow_html=True)
                st.image(resized, use_container_width=False, width=280)
                st.markdown('</div>', unsafe_allow_html=True)

    # ─── RIGHT: Results ───────────────────────────────────────
    with col_right:
        current_img = st.session_state.get("current_img")

        if current_img is None:
            st.markdown("""
            <div class="placeholder">
              <span class="ph-icon">&#128465;</span>
              <p class="ph-title">No image yet</p>
              <p class="ph-sub">Capture via webcam or upload a photo<br>
              to get an instant 3-stage waste classification.</p>
            </div>""", unsafe_allow_html=True)
        else:
            # Cache key includes image content AND model mode
            img_hash  = hash(current_img.tobytes())
            cache_key = (img_hash, model_mode)

            if cache_key not in st.session_state["result_cache"]:
                with st.spinner("Classifying..."):
                    result = run_inference(
                        current_img, model_mode,
                        cnn_model, feat_ext, elm_payload, scaler, class_indices,
                    )
                st.session_state["result_cache"][cache_key] = result
                # Trim cache to last 9 entries (3 modes x 3 recent images)
                if len(st.session_state["result_cache"]) > 9:
                    oldest = next(iter(st.session_state["result_cache"]))
                    del st.session_state["result_cache"][oldest]

            cached = st.session_state["result_cache"].get(cache_key)
            if cached is not None:
                render_results(cached, current_img, gradcam_model, show_gradcam)

    # Footer
    st.markdown(
        "<hr style='margin:10px 0 5px;border-color:#e2e8f0'>"
        "<p style='font-size:.64rem;color:#cbd5e1;text-align:center;margin:0'>"
        "WasteVision AI &nbsp;&#183;&nbsp; EfficientNetB3 backbone "
        "&nbsp;&#183;&nbsp; 2-Phase Fine-tuning &nbsp;&#183;&nbsp; "
        "Borderline-SMOTE &nbsp;&#183;&nbsp; 7x ELM Ensemble "
        "&nbsp;&#183;&nbsp; TTA &nbsp;&#183;&nbsp; Grad-CAM"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()