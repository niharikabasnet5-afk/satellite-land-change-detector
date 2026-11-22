import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TerraWatch AI",
    page_icon="🛰️",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp { background-color: #0B1220; color: white; }
.block-container { padding-top: 2rem; }
[data-testid="stSidebar"] { background-color: #111B2D; }
h1, h2, h3 { color: #EAF4FF; }
div[data-testid="stMetric"] {
    background-color: #162337; padding: 18px; border-radius: 12px;
    border: 1px solid #263A54;
}
.stButton>button {
    background-color: #00A6D6; color: white; border-radius: 10px;
    height: 50px; border: none; font-weight: bold;
}
.stButton>button:hover { background-color: #008DB8; color: white; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 256
MODEL_PATH = "change_detection_model.keras"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)

# ---------------- FUNCTIONS ----------------
def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image.astype("float32") / 255.0

def detect_changes(image_A, image_B, threshold=0.5):
    A = np.expand_dims(preprocess_image(image_A), axis=0)
    B = np.expand_dims(preprocess_image(image_B), axis=0)
    prediction = model.predict([A, B], verbose=0)
    probability_map = prediction[0, :, :, 0]
    change_mask = (probability_map > threshold).astype(np.uint8)
    return probability_map, change_mask

def calculate_change_area(change_mask):
    changed_pixels = int(np.sum(change_mask == 1))
    total_pixels = int(change_mask.size)
    change_percentage = (changed_pixels / total_pixels) * 100
    return changed_pixels, total_pixels, change_percentage

def create_overlay(image, change_mask):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    overlay = image.copy()
    overlay[change_mask == 1] = [255, 0, 0]
    return cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🛰️ TerraWatch AI")
    st.caption("Satellite Land Monitoring System")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("🏠 Dashboard  \n🛰️ Change Detection  \n📊 Analysis Results  \nℹ️ About Project")
    st.markdown("---")
    st.markdown("### System Status")
    if model_loaded:
        st.success("AI Model Loaded")
    else:
        st.error("Model file not found or could not be loaded")
    st.caption("Powered by Deep Learning")

# ---------------- HEADER ----------------
st.markdown("# 🛰️ TerraWatch AI")
st.markdown("### Satellite Change Detection Dashboard")
st.write("Monitor land transformation using AI-powered satellite image analysis.")
st.markdown("---")

# ---------------- UPLOAD SECTION ----------------
st.markdown("## 🖼️ Satellite Image Comparison")
st.write("Upload two satellite images of the same location captured at different times.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📅 Earlier Satellite Image")
    uploaded_A = st.file_uploader("Upload Previous Image", type=["jpg", "jpeg", "png"], key="image_A")
with col2:
    st.markdown("### 📅 Later Satellite Image")
    uploaded_B = st.file_uploader("Upload Latest Image", type=["jpg", "jpeg", "png"], key="image_B")

threshold = st.slider("Detection Sensitivity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
st.markdown("---")

if not model_loaded:
    st.error("The model could not be loaded. Make sure change_detection_model.keras is included with this app.")
    st.caption(f"Model loading error: {model_error}")

if uploaded_A and uploaded_B and model_loaded:
    image_A = Image.open(uploaded_A).convert("RGB")
    image_B = Image.open(uploaded_B).convert("RGB")
    st.markdown("### 🔍 Ready for Analysis")

    if st.button("🚀 ANALYZE LAND CHANGES", use_container_width=True):
        with st.spinner("Analyzing satellite images..."):
            probability_map, change_mask = detect_changes(image_A, image_B, threshold)
            changed_pixels, total_pixels, change_percentage = calculate_change_area(change_mask)
            overlay = create_overlay(image_B, change_mask)

        st.success("Analysis completed successfully!")
        st.markdown("---")
        st.subheader("📊 Change Analysis Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Changed Pixels", f"{changed_pixels:,}")
        m2.metric("Total Pixels", f"{total_pixels:,}")
        m3.metric("Changed Area", f"{change_percentage:.2f}%")

        st.markdown("---")
        st.subheader("🖼️ Visual Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.image(image_A, caption="Earlier Image", use_container_width=True)
            st.image(change_mask * 255, caption="Detected Change Mask", use_container_width=True)
        with c2:
            st.image(image_B, caption="Later Image", use_container_width=True)
            st.image(overlay, caption="Change Overlay", use_container_width=True)

        st.info("Red regions highlight potential land changes identified by the AI model. These detections should be reviewed before confirming actual encroachment.")
elif not uploaded_A or not uploaded_B:
    st.info("👆 Upload both earlier and later satellite images to begin analysis.")

st.markdown("---")
st.caption("TerraWatch AI | Satellite Land Change & Encroachment Monitoring | Powered by Deep Learning")
