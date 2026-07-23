import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load model
model = tf.keras.models.load_model("encoder_decoder.h5")

# Load tokenizers
with open("tokenizer_eng.pkl", "rb") as f:
    tokenizer_eng = pickle.load(f)

with open("tokenizer_urd.pkl", "rb") as f:
    tokenizer_urd = pickle.load(f)

max_eng_len = 1000      # Replace with your max English length
max_urdu_len = 1000     # Replace with your max Urdu length

# Reverse Urdu dictionary
index_word = {v: k for k, v in tokenizer_urd.word_index.items()}

st.title(" English →  Urdu Translator")

text = st.text_input("Enter English Sentence")

def translate(sentence):
    # Encoder input
    seq = tokenizer_eng.texts_to_sequences([sentence.lower()])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding="pre")

    # Start token
    start = tokenizer_urd.word_index["<sos>"]

    decoder_input = np.zeros((1, max_urdu_len - 1), dtype=np.int32)
    decoder_input[0, 0] = start

    translated = []

    for i in range(max_urdu_len - 1):

        pred = model.predict([seq, decoder_input], verbose=0)

        token = np.argmax(pred[0, i])

        word = index_word.get(token, "")

        if word == "<eos>" or word == "":
            break

        translated.append(word)

        if i + 1 < max_urdu_len - 1:
            decoder_input[0, i + 1] = token

    return " ".join(translated)


if st.button("Translate"):
    result = translate(text)
    st.success(result)