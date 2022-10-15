import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import os

absolute_path = os.path.dirname(__file__)

full_path = os.path.join(absolute_path, "Notebook")

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(full_path + '/my_model.hdf5', custom_objects={'KerasLayer': hub.KerasLayer})
    return model


model = load_model()
st.title("""
      Dogs and Cats Classification
""")
st.subheader("Choose a Image with Cat or Dog")

file = st.file_uploader("Upload your Image", type=['jpg', 'png'])


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.array(image)
    img_reshape = img[np.newaxis]
    img_scale = img_reshape / 255
    prediction = model.predict(img_scale)
    return prediction


st.markdown("<h4 style='text-align:left; color:gray'> Prediction </h4>", unsafe_allow_html=True)

if file is not None:
    image = Image.open(file)
    st.image(image, width=None)
    predictions = import_and_predict(image, model)
    class_names = ["Cat", 'Dog']
    string = "What you see here is a " + class_names[np.argmax(predictions)]
    st.success(string)
