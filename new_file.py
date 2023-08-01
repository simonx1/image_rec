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
    
    # Convert the image to a byte array
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr = byte_arr.getvalue()

    # Encode the byte array in base64
    base64_image = base64.b64encode(byte_arr).decode('utf-8')

    # Send a POST request to the Huggingface API
    response = requests.post(
        'https://api-inference.huggingface.co/models/your-model-name',
        headers={'Authorization': 'Bearer your-api-key'},
        data=json.dumps({'inputs': base64_image})
    )

    # Check the status of the response
    if response.status_code == 200:
        # Extract the labels from the response
        labels = response.json()[0]['label']

        # Display the labels
        st.write('Labels:', ', '.join(labels))
    else:
        st.write('Error:', response.text)
