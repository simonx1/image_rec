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
        # Extract the image from the response
        base64_image = response.json()['image']

        # Decode the base64 image
        decoded_image = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(decoded_image))

        # Display the image with boxes and labels
        st.image(image, caption='Image with identified objects.', use_column_width=True)
    else:
        st.write('Error:', response.text)
