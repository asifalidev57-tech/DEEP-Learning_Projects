import pandas as pd 
import numpy as np 
import streamlit as st
import pickle
import tensorflow as tf 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
model=tf.keras.models.load_model('sms_spam_model.h5')
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("SMS Spam , Harm Prediction Model")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    sequence = tokenizer.texts_to_sequences([words])
    padd=pad_sequences(sequence, maxlen=500)
    return padd



user_input=st.text_area("SMS Spam Harm prediction")
if st.button("classify"):
    preprocess=preprocess_text(user_input)
    prediction=model.predict(preprocess)
    if prediction[0][0] > 0.5:
       result = "Spam"
    else:
       result = "Ham"
    st.write(f'Sentiment: {result}')
    st.write(f'prediction score: {prediction[0][0]}')
