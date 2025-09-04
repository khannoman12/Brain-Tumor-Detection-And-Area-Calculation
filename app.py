import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# Page setup
st.set_page_config(page_title="Brain Tumor Detection", layout="wide", page_icon="🧠")
st.title("Type of Brain Tumor Detection And Area Calculation")

st.markdown("""
Upload a brain MRI image and the model will:
1. Detect if a brain tumor is present.
2. Segment the tumor mask.
3. Calculate the tumor size as a percentage of the brain area.
""")

# Load model
model = YOLO("best.pt")

# Sidebar - Controls
st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

st.sidebar.divider()
st.sidebar.header("Display Settings")
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
st.sidebar.markdown("---")
show_mask = st.sidebar.checkbox("Show Mask of Tumor", value=True)
st.sidebar.markdown("---")
show_estimation = st.sidebar.checkbox("Show Area Calculation", value=True)

# File Upload
uploaded_image = st.file_uploader("📄 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    with st.spinner("🔍 Processing image..."):
        result = model.predict(source=img_np, conf=confidence)
        res = result[0]

        # Layout
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # Original Image
        with col1:
            st.markdown("### Original MRI Image")
            st.image(image, use_container_width=True)

        # Detection
        if res.masks is None or len(res.boxes) == 0:
            with col2:
                st.markdown("### Tumor Detection")
                st.warning("✅ No tumor detected in the image.")
        else:
            annotated_img = res.plot() if show_boxes else img_np.copy()
            annotated_img_rgb = annotated_img[:, :, ::-1]
            annotated_pil = Image.fromarray(annotated_img_rgb)
            with col2:
                st.markdown("### Detected Brain Tumor")
                st.image(annotated_img_rgb, use_container_width=True)

            # Tumor Mask
            tumor_mask_pil = None
            mask_data = res.masks.data.cpu().numpy()
            combined_mask = np.sum(mask_data, axis=0)
            combined_mask = np.clip(combined_mask, 0, 1)
            tumor_mask_image = (combined_mask * 255).astype(np.uint8)
            tumor_mask_rgb = cv2.cvtColor(tumor_mask_image, cv2.COLOR_GRAY2RGB)
            tumor_mask_pil = Image.fromarray(tumor_mask_rgb)
            if show_mask:
                with col3:
                    st.markdown("### Mask of Tumor")
                    st.image(tumor_mask_rgb, use_container_width=True)

            # Size Estimation
            overlay = img_bgr.copy()
            tumor_contours = []
            overlay_pil = None
            brain_area = h * w

            for seg, box in zip(res.masks.xy, res.boxes):
                contour = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                area = cv2.contourArea(contour)
                tumor_contours.append(contour)
                cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 2)

                tumor_percent = (area / brain_area) * 100 if brain_area > 0 else 0
                label_text = f"{tumor_percent:.2f}%"
                x, y = int(contour[0][0][0]), int(contour[0][0][1]) - 10
                cv2.putText(overlay, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            if show_estimation:
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                overlay_pil = Image.fromarray(overlay_rgb)
                with col4:
                    st.markdown("### Tumor Area Calculation")
                    st.image(overlay_rgb, use_container_width=True)

            # Downloads
            with st.expander("📥 Download Images"):
                st.markdown("Select the image you want to download:")
                col_download1, col_download2, col_download3 = st.columns(3)

                if annotated_pil:
                    with io.BytesIO() as buf1:
                        annotated_pil.save(buf1, format="PNG")
                        col_download1.download_button("Download Detection", data=buf1.getvalue(), file_name="detection.png", mime="image/png")

                if tumor_mask_pil:
                    with io.BytesIO() as buf2:
                        tumor_mask_pil.save(buf2, format="PNG")
                        col_download2.download_button("Download Tumor Mask", data=buf2.getvalue(), file_name="tumor_mask.png", mime="image/png")

                if overlay_pil:
                    with io.BytesIO() as buf3:
                        overlay_pil.save(buf3, format="PNG")
                        col_download3.download_button("Download Size Estimation", data=buf3.getvalue(), file_name="size_estimation.png", mime="image/png")
else:
    st.info("👈 Upload a brain MRI image to start tumor detection.")
