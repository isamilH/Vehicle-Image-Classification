import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load the pre-trained model
MODEL_PATH = 'vehicle_classification_model.h5'
model = load_model(MODEL_PATH)

# Define the input image size for the model
IMG_SIZE = (150, 150)

# Title of the application
st.title("Vehicle Image Classification")
st.write("Upload an image of a vehicle to classify its type.")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    try:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocess the image
        img = load_img(temp_file_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        # Define class labels (modify these to match your dataset's classes)
        class_labels = ["Car", "Truck", "Bus", "Motorbike", "Bicycle", "SUV", "Van"]

        # Display the result
        st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # Clean up the temporary file
        os.remove(temp_file_path)

    except Exception as e:
        st.error(f"Error: {e}")
