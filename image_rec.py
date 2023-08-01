import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
import io
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
import base64
import os

hf_token = os.getenv('HUGGING_FACE_ACCESS_TOKEN')

# Title of the web application
st.title('Object Recognition AI')

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("")
    st.write("Classifying...")

    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        0
    ]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red")
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]} {round(score.item(), 3)}", fill="red", font=font)

    st.image(image, caption='Processed Image.', use_column_width=True)
