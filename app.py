import streamlit as st
import tensorflow
import numpy as np
from PIL import Image

# Set title and page layout
st.title("CovidXRayNet: Detecting COVID-19 from Chest X-rays")
st.markdown("---")

# Load the trained model
def load_model():
    model = tensorflow.keras.models.load_model("my_model.h5")
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = Image.open(image).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image, model):
    predictions = model.predict(image)
    class_names = ['COVID-19', 'Normal']
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Sidebar - Upload image
st.sidebar.title("Upload X-ray Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.sidebar.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)

    # Make prediction on the uploaded image
    if st.sidebar.button("Predict"):
        image_data = preprocess_image(uploaded_file)
        prediction = predict(image_data, model)
        st.write("Prediction:", prediction)
