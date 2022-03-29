# Model Deployment using All in One Method

import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Milestone 2",
    page_icon= "🍂" ,
    layout="centered", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Image Classification with CNN",
    }
)

st.markdown("<h1 style='text-align: center; color: black;'>Weather Image Classification</h1>", unsafe_allow_html=True)
st.write("This is an image classification web app to predict weather class (cloudy, rain, shine, or sunrise)")

weather = Image.open('weather.png')
st.image(weather)

def classification_image(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

uploaded_file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# Button to Predict
button = st.button("Click Here to Classify")

if button:
    if uploaded_file is None:
        st.write("Please upload an image file!")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file)
            prediction = classification_image(image, 'final_model.h5')
            if prediction == 0:
                st.image(image, caption='Cloudy Image', use_column_width=True)
                st.write("The image is classified as cloudy ☁️")
            if prediction == 1:
                st.image(image, caption='Rain Image', use_column_width=True)
                st.write("The image is classified as rain 🌧️")
            if prediction == 2:
                st.image(image, caption='Shine Image', use_column_width=True)
                st.write("The image is classified as shine 🌞")
            if prediction == 3:
                st.image(image, caption='Sunrise Image', use_column_width=True)
                st.write("The image is classified as sunrise 🌅")
        with col2:
            pass