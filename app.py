import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ==========================================================
# 🔹 Load Trained Model
# ==========================================================
@st.cache_resource
def load_trained_model():
    model = load_model("garbage_classifier_model.h5")
    return model

model = load_trained_model()

# ==========================================================
# 🔹 Streamlit Page Setup
# ==========================================================
st.set_page_config(
    page_title="Garbage vs Clean Detector",
    page_icon="🗑️",
    layout="wide"
)

st.title("🧹 Garbage vs Clean Classifier")
st.markdown("""
Upload one or more images to check if they contain **Garbage** or are **Clean** areas.
""")

# ==========================================================
# 🔹 Image Upload
# ==========================================================
uploaded_files = st.file_uploader(
    "📤 Upload image(s) for classification...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ==========================================================
# 🔹 Preprocessing Function
# ==========================================================
def preprocess_image(image):
    image = image.resize((128, 128))  # match model training size
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # remove alpha if present
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    return image

# ==========================================================
# 🔹 Prediction Function
# ==========================================================
def predict(image):
    preds = model.predict(image)
    prob = float(preds[0][0])
    label = "🧹 GARBAGE" if prob < 0.5 else "🚮 CLEAN"
    confidence = (1 - prob) * 100 if prob < 0.5 else prob * 100
    return label, confidence

# ==========================================================
# 🔹 Display Results
# ==========================================================
if uploaded_files:
    st.write("🔍 **Analyzing uploaded images...**")

    cols = st.columns(2)  # 2-column grid for results
    col_index = 0

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        processed = preprocess_image(image)
        label, confidence = predict(processed)

        with cols[col_index]:
            st.image(image, caption=f"🖼️ {uploaded_file.name}", use_container_width=True)
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")

            if "CLEAN" in label:
                st.success("✅ Area looks clean!")
            else:
                st.error("⚠️ Garbage detected!")

        col_index = (col_index + 1) % 2
else:
    st.info("Please upload one or more images above to start prediction.")
