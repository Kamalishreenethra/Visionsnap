import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="Object Detection App", layout="centered")

st.title("🖼️ Upload Image for Object Detection")
st.write("Upload an image and the model will detect objects.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", width=600)

    if st.button("🔍 Detect Objects"):
        results = model(img_array)
        annotated_img = results[0].plot()

        st.image(annotated_img, caption="Detected Objects", width=600)
