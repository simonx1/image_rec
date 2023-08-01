import streamlit as st
from PIL import Image
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
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # # Convert the image to a byte array
    # byte_arr = io.BytesIO()
    # image.save(byte_arr, format='JPEG')

    # Encode the byte array in base64
    # base64_image = base64.b64encode(byte_arr).decode('utf-8')

    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        0
    ]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    # # Check the status of the response
    # if response.status_code == 200:
    #     # Extract the image from the response
    #     base64_image = response.json()['image']

    #     # Decode the base64 image
    #     decoded_image = base64.b64decode(base64_image)
    #     image = Image.open(io.BytesIO(decoded_image))

    #     # Display the image with boxes and labels
    #     st.image(image, caption='Image with identified objects.', use_column_width=True)
    # else:
    #     st.write('Error:', response.text)
