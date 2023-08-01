import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

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

    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base")

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
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red")
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]} {round(score.item(), 3)}", fill="red", font=font)

    st.image(image, caption='Processed Image.', use_column_width=True)
