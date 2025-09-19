import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


model = tf.keras.models.load_model("digit_recognizer.h5")

st.title("📸 Handwritten Digit Recognizer")
st.write("Upload an image or use your webcam to recognize digits (0-9).")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

camera_image = st.camera_input("Take a picture of a digit")

def preprocess_image(img):
    img = img.convert("L")              
    img = img.resize((28, 28))          
    img_array = np.array(img) / 255.0  
    img_array = img_array.reshape(1, 28, 28, 1)  
    return img_array

img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file)
elif camera_image is not None:
    img = Image.open(camera_image)

if img:
    st.image(img, caption="Input Image", width=150)
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    pred_label = np.argmax(prediction, axis=1)[0]
    st.subheader(f"✅ Predicted Digit: {pred_label}")
