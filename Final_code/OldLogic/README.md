# WasteVision AI — Streamlit UI

## Folder Structure
```
project/
├── app.py
├── requirements.txt
└── weights/
    ├── best_accuracy.weights.h5   ← from checkpoints/
    ├── elm_ensemble.pkl            ← from checkpoints/
    ├── feature_scaler.pkl          ← from checkpoints/
    └── class_indices.json          ← from checkpoints/
```

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The model rebuilds the **exact same EfficientNetB3 architecture** as training and
  loads weights only (`model.load_weights`), avoiding any preprocess_input / Lambda
  serialisation issues.
- The Lambda layer (`efficientnet_preprocess`) is re-defined in Python at load time,
  so the model correctly normalises raw [0–255] pixel input before the backbone.
- ELM inference: raw feature vector → StandardScaler.transform → 7× ELM heads →
  majority vote → final class index.
- CNN softmax probabilities are shown as top-5 confidence bars; ELM vote is the
  **primary** predicted label.
- Image hash caching ensures the heavy inference pipeline runs only when the image
  actually changes, not on every Streamlit rerun.