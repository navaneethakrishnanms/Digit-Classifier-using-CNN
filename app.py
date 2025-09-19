import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_recognizer.h5")

# Preprocess function
def preprocess_and_predict(img):
    img = img.convert("L")                # grayscale
    img = img.resize((28, 28))            # resize to 28x28
    img_array = np.array(img) / 255.0     # normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    
    prediction = model.predict(img_array)
    pred_label = np.argmax(prediction, axis=1)[0]
    return f"Predicted Digit: {pred_label}"


demo = gr.Interface(
    fn=preprocess_and_predict,
    inputs=gr.Image(type="pil", label="Upload or Capture Digit"),
    outputs=gr.Label(label="Prediction"),
    title="📸 Handwritten Digit Recognizer",
    description="Upload an image or take a picture of a handwritten digit (0-9)."
)

if __name__ == "__main__":
    demo.launch()

