import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import os

# Get absolute paths
absolute_path = os.getcwd()
full_path = os.path.join(absolute_path, "Notebook")

# Sidebar configuration
st.sidebar.title("🐶🐱 Cats & Dogs Classifier")
st.sidebar.markdown("""
    This application uses a pre-trained deep learning model to classify images as either a **cat** or a **dog**. 
    Upload an image of a cat or a dog, and see what the model predicts!
""")

# Load Model
@st.cache_resource
def load_model():
    hub_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"  # Replace with the correct URL
    feature_extractor_layer = hub.KerasLayer(hub_url, input_shape=(224, 224, 3), trainable=False)
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    try:
        model.load_weights(full_path + '/my_model.hdf5')
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading model weights: {str(e)}")

    return model

model = None
try:
    model = load_model()
except Exception as e:
    st.sidebar.warning("⚠️ Model failed to load properly due to compatibility issues. Predictions are currently unavailable.")

# Page Title
st.title("🐾 Dogs and Cats Image Classification 🐾")

# Improved layout with columns
col1, col2 = st.columns(2)

# Upload Image section
with col1:
    st.header("📤 Upload Your Image")
    file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# Function to preprocess and predict the image
def import_and_predict(image_data, model):
    if model is None:
        return None
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.array(image)
    img = img.astype(np.float32)
    img_reshape = img[np.newaxis, ...]
    img_scale = img_reshape / 255.0
    prediction = model.predict(img_scale)
    return prediction

# Show Image and Prediction Result
if file is not None:
    with col1:
        st.image(file, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.header("🧠 Model Prediction")
        if model is None:
            st.warning("The model is currently unavailable due to compatibility issues. We are working on a solution.")
        else:
            try:
                with st.spinner("Predicting..."):
                    image = Image.open(file)
                    predictions = import_and_predict(image, model)
                    class_names = ["Cat", "Dog"]
                    result = class_names[np.argmax(predictions)]
                    string = f"What you see here is a **{result}**"
                    st.success(string)
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

