import tensorflow as tf 
import numpy as np 
import pandas as pd 
import streamlit as st 
from PIL import Image
model=tf.keras.models.load_model('cat_dog_model.h5')

st.title("Cat Vs Dog Identification")
st.write("Upload the image to predicat the dog vs cat")

uploaded_file=st.file_uploader(
    "image uploaded",
    type=['jpg', 'png', 'jpeg']
)
class_names = ["Cat", "Dog"]
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image=image,
        use_container_width=True
    )
    image=image.convert('RGB')
    image = image.resize((150, 150))

    img_array=np.array(image)
    img_array=img_array/255

    img_array=img_array.reshape(1,150, 150, 3)

    prediction=model.predict(img_array)

    class_predict=np.argmax(prediction)
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = "Dog"
    else:
        result = "Cat"

    st.success(f"Prediction: {result}")