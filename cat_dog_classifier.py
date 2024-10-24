import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image

# Load your pre-trained model
@st.cache_resource
def load_model():
    model = keras.models.load_model('cat_dog_model.h5')  # Ensure the file is in the same directory
    return model

model = load_model()  # Load the model using the updated caching function

# Preprocess the image before making predictions
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (256, 256))  # Resize the image to 256x256
    image = image / 255.0  # Normalize the image
    image = np.reshape(image, (1, 256, 256, 3))  # Reshape for model input
    return image

# Make predictions using the model
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit App Layout
st.title("Cat vs Dog Classifier")
st.write("Upload an image, and the model will classify it as either a cat or a dog.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Predict the class of the image
    prediction = predict(image)

    # Display the result
    if prediction < 0.5:
        st.write("It's a **cat**!")
    else:
        st.write("It's a **dog**!")
