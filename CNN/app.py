import numpy as np 
import pandas as pd 
import tensorflow as tf
import streamlit as st 
from PIL import Image

model=tf.keras.models.load_model('fashion_mnist_cnn.h5')
class_names = [
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]
st.title("👕 Fashion MNIST Classifier")
st.write("Upload a clothing image and let the CNN classify it.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image,
             caption="Uploaded Image",
             use_container_width=True)
        
        image = image.convert("L")     
        image = image.resize((28,28))   

        img_array = np.array(image)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        prediction = tf.nn.softmax(prediction)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        st.success(
              f"Prediction: {class_names[predicted_class]}"
        )

        st.write(
              f"Confidence: {confidence*100:.2f}%"
        )