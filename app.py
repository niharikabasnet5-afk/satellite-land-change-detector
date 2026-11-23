
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="TerraWatch AI",
    page_icon="🛰️",
    layout="wide"
)

IMG_SIZE = 256

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp { background-color: #0B1220; color: white; }
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: #EAF4FF; }
div[data-testid="stMetric"] {
    background: #162337;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #263A54;
}
.stButton>button {
    background: #00A6D6;
    color: white;
    height: 50px;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("# 🛰️ TerraWatch AI")
st.markdown("### Satellite Land Use Change & Encroachment Detector")
st.write("Compare satellite images captured at different times to identify potential land changes.")
st.markdown("---")

# ---------------- IMAGE PROCESSING ----------------
def prepare_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

def detect_changes(img1, img2, sensitivity):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Reduce small noise
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Calculate pixel differences
    difference = cv2.absdiff(gray1, gray2)

    # Threshold differences
    threshold_value = int(255 * (1 - sensitivity))
    _, mask = cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    changed_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    change_percentage = (changed_pixels / total_pixels) * 100

    # Create red overlay
    overlay = img2.copy()
    overlay[mask > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(img2, 0.65, overlay, 0.35, 0)

    return difference, mask, overlay, changed_pixels, change_percentage

# ---------------- UPLOAD ----------------
st.markdown("## 📷 Satellite Image Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🕒 Earlier Image")
    upload1 = st.file_uploader(
        "Upload the earlier satellite image",
        type=["jpg", "jpeg", "png"],
        key="earlier"
    )

with col2:
    st.markdown("### 🕒 Later Image")
    upload2 = st.file_uploader(
        "Upload the later satellite image",
        type=["jpg", "jpeg", "png"],
        key="later"
    )

sensitivity = st.slider(
    "Change Detection Sensitivity",
    0.1, 0.9, 0.5, 0.05
)

st.markdown("---")

# ---------------- ANALYSIS ----------------
if upload1 and upload2:

    image1 = prepare_image(Image.open(upload1))
    image2 = prepare_image(Image.open(upload2))

    st.markdown("### Images Ready for Comparison")

    if st.button("🚀 ANALYZE LAND CHANGES", use_container_width=True):

        difference, mask, overlay, changed_pixels, change_percentage = detect_changes(
            image1, image2, sensitivity
        )

        st.success("Image comparison completed!")

        # Metrics
        st.markdown("## 📊 Analysis Results")

        m1, m2, m3 = st.columns(3)

        m1.metric("Changed Pixels", f"{changed_pixels:,}")
        m2.metric("Total Pixels", f"{IMG_SIZE * IMG_SIZE:,}")
        m3.metric("Detected Change", f"{change_percentage:.2f}%")

        st.markdown("---")

        # Images
        st.markdown("## 🛰️ Visual Results")

        c1, c2 = st.columns(2)

        with c1:
            st.image(image1, caption="Earlier Satellite Image", use_container_width=True)
            st.image(mask, caption="Binary Change Mask", use_container_width=True)

        with c2:
            st.image(image2, caption="Later Satellite Image", use_container_width=True)
            st.image(overlay, caption="Detected Changes — Red Highlight", use_container_width=True)

        st.info(
            "Red regions indicate visual differences between the two images. "
            "These differences may represent land changes, construction, vegetation changes, "
            "or other environmental variations. This demo uses image processing, "
            "not the trained CNN model."
        )

elif not upload1 or not upload2:
    st.info("👆 Upload both earlier and later satellite images to begin.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("TerraWatch AI | Satellite Land Change Monitoring Demo")

