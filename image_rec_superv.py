import streamlit as st
from PIL import Image
import supervision as sv
from ultralytics import YOLO
import numpy as np

# import logging
# logging.basicConfig(level=logging.ERROR)

# Title of the web application
st.title('Object Recognition AI')

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image.save('temp.jpeg', 'JPEG')
    image = Image.open('temp.jpeg')
    st.write("")
    st.write("Classifying...")

    model = YOLO('yolov8x.pt')

    result = model(image)[0]
    detections = sv.Detections.from_yolov8(result)

    box_annotator = sv.BoxAnnotator()
    classes = model.model.names


    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]
    annotated_img = box_annotator.annotate(
        scene=np.array(image),
        detections=detections,
        labels=labels
    )

    st.write("Classification complete")
    st.image(annotated_img, caption='Processed Image', use_column_width=True)
