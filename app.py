
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="TerraWatch AI",
    page_icon="🛰️",
    layout="wide"
)

IMG_SIZE = 256
MODEL_PATH = "change_detection_model.keras"

# =========================================================
# CUSTOM DESIGN
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #111B2D;
}
h1, h2, h3 {
    color: #EAF4FF;
}
div[data-testid="stMetric"] {
    background-color: #162337;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263A54;
}
.stButton>button {
    background-color: #00A6D6;
    color: white;
    border-radius: 10px;
    height: 50px;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #008DB8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    model_loaded = True
    model_error = None

except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)

# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(image):

    image = image.convert("RGB")

    image = np.array(image)

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype("float32") / 255.0

    return image

# =========================================================
# CHANGE DETECTION
# =========================================================
def detect_changes(image_A, image_B, threshold=0.5):

    # Prepare images
    A = preprocess_image(image_A)
    B = preprocess_image(image_B)

    # Add batch dimension
    A = np.expand_dims(A, axis=0)
    B = np.expand_dims(B, axis=0)

    # CNN prediction
    prediction = model.predict(
        [A, B],
        verbose=0
    )

    # Extract probability map
    probability_map = prediction[0, :, :, 0]

    # Generate binary change mask
    change_mask = (
        probability_map > threshold
    ).astype(np.uint8)

    # Convert mask to display format
    mask_display = change_mask * 255

    # Calculate changed pixels
    changed_pixels = int(np.sum(change_mask))

    total_pixels = int(change_mask.size)

    change_percentage = (
        changed_pixels / total_pixels
    ) * 100

    # Create red overlay on later image
    later_image = np.array(
        image_B.convert("RGB")
    )

    later_image = cv2.resize(
        later_image,
        (IMG_SIZE, IMG_SIZE)
    )

    overlay = later_image.copy()

    overlay[change_mask == 1] = [255, 0, 0]

    overlay = cv2.addWeighted(
        later_image,
        0.65,
        overlay,
        0.35,
        0
    )

    return (
        probability_map,
        mask_display,
        overlay,
        changed_pixels,
        total_pixels,
        change_percentage
    )

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🛰️ TerraWatch AI")

    st.caption(
        "Satellite Land Monitoring System"
    )

    st.markdown("---")

    st.markdown("### Navigation")

    st.markdown("""
    🏠 Dashboard

    🛰️ Change Detection

    📊 Analysis Results

    ℹ️ About Project
    """)

    st.markdown("---")

    st.markdown("### System Status")

    if model_loaded:
        st.success("AI Model Loaded")
    else:
        st.error("AI Model Failed to Load")

    st.caption("Powered by Deep Learning")

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown("# 🛰️ TerraWatch AI")

st.markdown(
    "### Satellite Land Use Change & Encroachment Detector"
)

st.write(
    "Compare satellite images captured at different times "
    "to identify potential land-use changes."
)

st.markdown("---")

# =========================================================
# MODEL ERROR
# =========================================================
if not model_loaded:

    st.error(
        "The trained CNN model could not be loaded."
    )

    st.code(model_error)

    st.info(
        "Ensure change_detection_model.keras is present "
        "in the same GitHub repository as app.py."
    )

# =========================================================
# IMAGE UPLOAD SECTION
# =========================================================
st.markdown("## 🖼️ Satellite Image Comparison")

st.write(
    "Upload two satellite images of the same location "
    "captured at different times."
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 📅 Earlier Satellite Image")

    uploaded_A = st.file_uploader(
        "Upload Previous Image",
        type=["jpg", "jpeg", "png"],
        key="image_A"
    )

with col2:

    st.markdown("### 📅 Later Satellite Image")

    uploaded_B = st.file_uploader(
        "Upload Latest Image",
        type=["jpg", "jpeg", "png"],
        key="image_B"
    )

# =========================================================
# SENSITIVITY
# =========================================================
threshold = st.slider(
    "Detection Sensitivity",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.markdown("---")

# =========================================================
# ANALYSIS SECTION
# =========================================================
if uploaded_A and uploaded_B and model_loaded:

    image_A = Image.open(
        uploaded_A
    ).convert("RGB")

    image_B = Image.open(
        uploaded_B
    ).convert("RGB")

    st.markdown("### 🔍 Ready for Analysis")

    if st.button(
        "🚀 ANALYZE LAND CHANGES",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analyzing satellite images..."
        ):

            (
                probability_map,
                change_mask,
                overlay,
                changed_pixels,
                total_pixels,
                change_percentage
            ) = detect_changes(
                image_A,
                image_B,
                threshold
            )

        st.success(
            "Analysis completed successfully!"
        )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------
        st.markdown("---")

        st.subheader("📊 Change Analysis Results")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Changed Pixels",
            f"{changed_pixels:,}"
        )

        m2.metric(
            "Total Pixels",
            f"{total_pixels:,}"
        )

        m3.metric(
            "Changed Area",
            f"{change_percentage:.2f}%"
        )

        st.markdown("---")

        # -------------------------------------------------
        # VISUAL RESULTS
        # -------------------------------------------------
        st.subheader("🛰️ Visual Analysis")

        c1, c2 = st.columns(2)

        with c1:

            st.image(
                image_A,
                caption="Earlier Satellite Image",
                use_container_width=True
            )

            st.image(
                change_mask,
                caption="CNN Predicted Binary Change Mask",
                use_container_width=True
            )

        with c2:

            st.image(
                image_B,
                caption="Later Satellite Image",
                use_container_width=True
            )

            st.image(
                overlay,
                caption="CNN Change Overlay",
                use_container_width=True
            )

        # -------------------------------------------------
        # PROBABILITY MAP
        # -------------------------------------------------
        st.markdown("---")

        st.subheader("🧠 Model Confidence Map")

        probability_display = (
            probability_map * 255
        ).clip(0, 255).astype(np.uint8)

        st.image(
            probability_display,
            caption="Change Probability Map",
            use_container_width=True
        )

        st.info(
            "The CNN predicts potential land changes between "
            "the two satellite images. White regions in the "
            "binary mask indicate detected changes. Red regions "
            "in the overlay highlight predicted changes on the "
            "later satellite image. Results should be reviewed "
            "before confirming actual encroachment."
        )

# =========================================================
# UPLOAD MESSAGE
# =========================================================
elif not uploaded_A or not uploaded_B:

    st.info(
        "👆 Upload both earlier and later satellite images "
        "to begin analysis."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "TerraWatch AI | Satellite Land Use Change & "
    "Encroachment Monitoring | Powered by Deep Learning"
)


