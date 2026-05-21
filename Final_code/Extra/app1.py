# # ============================================================
# # IMPORTS
# # ============================================================
# import os, re, json, pickle
# import numpy as np
# import joblib
# import streamlit as st
# from PIL import Image
# import matplotlib.cm as cm

# import tensorflow as tf
# from tensorflow.keras import layers, models, regularizers
# from tensorflow.keras.applications import EfficientNetB3
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # ============================================================
# # CONFIG
# # ============================================================
# st.set_page_config(
#     page_title="WasteVision AI",
#     page_icon="♻",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# DIR = os.path.dirname(os.path.abspath(__file__))

# PATHS = {
#     "weights": os.path.join(DIR, "weights", "best_accuracy.weights.h5"),
#     "elm": os.path.join(DIR, "weights", "elm_ensemble.pkl"),
#     "scaler": os.path.join(DIR, "weights", "feature_scaler.pkl"),
#     "classes": os.path.join(DIR, "weights", "class_indices.json"),
# }

# IMG_SIZE = (224, 224)
# NUM_CLASSES = 36
# OOD_THRESHOLD = 0.30


# # ============================================================
# # LABEL MAPPINGS
# # ============================================================
# STAGE_3_TO_2 = {
#     "A_Foods": "A_Green Waste",
#     "B_Animal Dead Body": "A_Green Waste",
#     "C_Cardboard": "B_Recyclable Waste",
#     "D_Newspaper": "B_Recyclable Waste",
#     "E_Paper Cups": "B_Recyclable Waste",
#     "F_Papers": "B_Recyclable Waste",
#     "G_Brown Glass": "C_Glass",
#     "H_Porcelin": "C_Glass",
#     "I_Green Glass": "C_Glass",
#     "J_White Glass": "C_Glass",
#     "K_Beverage Cans": "D_Metal",
#     "L_Construction Scrap": "D_Metal",
#     "M_Metal Containers": "D_Metal",
#     "N_Plastic Bag": "E_Polymer (Petrolium Based)",
#     "O_Plastic Bottle": "E_Polymer (Petrolium Based)",
#     "Q_Plastic Containers": "E_Polymer (Petrolium Based)",
#     "R_Plastic Cups": "E_Polymer (Petrolium Based)",
#     "S_Tetra Pak": "E_Polymer (Petrolium Based)",
#     "T_Clothes": "F_Leather and Fabric",
#     "U_Shoes": "F_Leather and Fabric",
#     "V_Gloves": "F_Leather and Fabric",
#     "W_Masks": "G_Medical Waste",
#     "Z_G_Battery": "G_Medical Waste",
#     "Z_H_Thermometer": "G_Medical Waste",
#     "X_Bandai": "H_E Waste",
#     "Z_B_Electrical Cables": "H_E Waste",
#     "Z_C_Electronic Chips": "H_E Waste",
#     "Z_D_Laptops": "H_E Waste",
#     "Z_E_Small Appliances": "H_E Waste",
#     "Z_F_Smartphones": "H_E Waste",
#     "Y_Medicine and Medicine Strip": "I_Hazardous Waste",
#     "Z_A_A_Syringe": "I_Hazardous Waste",
#     "Z_A_Diaper": "I_Hazardous Waste",
#     "Z_I_Cigarette Butt": "I_Hazardous Waste",
#     "Z_J_Pesticidebottle": "I_Hazardous Waste",
#     "Z_K_Spray cans": "I_Hazardous Waste",
# }

# STAGE_2_TO_1 = {
#     "A_Green Waste": "A-Biodegradable",
#     "B_Recyclable Waste": "B-Non Biodegradable",
#     "C_Glass": "B-Non Biodegradable",
#     "D_Metal": "B-Non Biodegradable",
#     "E_Polymer (Petrolium Based)": "B-Non Biodegradable",
#     "F_Leather and Fabric": "B-Non Biodegradable",
#     "G_Medical Waste": "B-Non Biodegradable",
#     "H_E Waste": "B-Non Biodegradable",
#     "I_Hazardous Waste": "B-Non Biodegradable",
# }


# # ============================================================
# # UTILITIES
# # ============================================================
# def clean_text(name: str) -> str:
#     return re.sub(r"[_\-]", " ", name).strip()


# def preprocess_image(img: Image.Image) -> np.ndarray:
#     img = img.convert("RGB").resize(IMG_SIZE)
#     return np.expand_dims(np.array(img, dtype=np.float32), 0)


# def check_ood(probs):
#     conf = float(np.max(probs))
#     entropy = -np.sum(probs * np.log(probs + 1e-10))
#     norm_entropy = entropy / np.log(len(probs))
#     return (conf < OOD_THRESHOLD or norm_entropy > 0.92), conf


# # ============================================================
# # MODEL LOADING
# # ============================================================
# @st.cache_resource
# def load_models():
#     backbone = EfficientNetB3(include_top=False, input_shape=(224,224,3))
#     backbone.trainable = False

#     inp = layers.Input((224,224,3))
#     x = layers.Lambda(preprocess_input)(inp)
#     x = backbone(x)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(512, activation="relu")(x)
#     feat = layers.Dense(256, activation="relu", name="feat")(x)
#     out = layers.Dense(NUM_CLASSES, activation="softmax")(feat)

#     model = models.Model(inp, out)
#     feat_model = models.Model(inp, feat)

#     model.load_weights(PATHS["weights"])

#     elm = pickle.load(open(PATHS["elm"], "rb"))
#     scaler = joblib.load(PATHS["scaler"])
#     class_map = json.load(open(PATHS["classes"]))

#     return model, feat_model, elm, scaler, class_map


# # ============================================================
# # INFERENCE
# # ============================================================
# def predict(img, model, feat_model, elm, scaler, class_map):
#     x = preprocess_image(img)
#     probs = model.predict(x, verbose=0)[0]

#     ood, conf = check_ood(probs)
#     if ood:
#         return {"ood": True, "conf": conf}

#     idx = int(np.argmax(probs))
#     stage3 = class_map.get(str(idx), "Unknown")
#     stage2 = STAGE_3_TO_2.get(stage3, "Unknown")
#     stage1 = STAGE_2_TO_1.get(stage2, "Unknown")

#     top5_idx = np.argsort(probs)[::-1][:5]
#     top5 = [(class_map.get(str(i)), float(probs[i])) for i in top5_idx]

#     return {
#         "ood": False,
#         "stage3": stage3,
#         "stage2": stage2,
#         "stage1": stage1,
#         "conf": float(probs[idx]),
#         "top5": top5
#     }


# # ============================================================
# # UI
# # ============================================================
# def main():
#     st.title("♻ WasteVision AI")

#     model, feat_model, elm, scaler, class_map = load_models()

#     img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

#     if not img_file:
#         st.info("Upload image to start")
#         return

#     img = Image.open(img_file)
#     st.image(img, caption="Input Image")

#     result = predict(img, model, feat_model, elm, scaler, class_map)

#     if result["ood"]:
#         st.warning(f"Not a valid waste image (confidence {result['conf']:.2f})")
#         return

#     st.subheader("Prediction")
#     st.write("Stage 3:", result["stage3"])
#     st.write("Stage 2:", result["stage2"])
#     st.write("Stage 1:", result["stage1"])
#     st.write("Confidence:", f"{result['conf']*100:.2f}%")

#     st.subheader("Top 5 Predictions")
#     for label, prob in result["top5"]:
#         st.write(f"{clean_text(label)}: {prob*100:.2f}%")


# # ============================================================
# if __name__ == "__main__":
#     main()

import tensorflow as tf
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import PIL
import joblib

print("TensorFlow:", tf.__version__)
print("Scikit-learn:", sklearn.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pillow:", PIL.__version__)
print("Joblib:", joblib.__version__)
