import streamlit as st
from PIL import Image
import requests
import io

# Title of the web application
st.title('Object Recognition AI')

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # TODO: Send the image to Huggingface API and get the labels
    # TODO: Display the image with labels
