import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('C:/Users/USER/OneDrive/Desktop/deeplearning/deep-learning/RNN/results/result/spam_model.keras')

# Function to preprocess input using the tokenizer from the training phase
def preprocess_text(text, tokenizer, max_length=10):
    # Tokenize the input text
    input_seq = tokenizer.texts_to_sequences([text])
    # Pad the sequence
    padded_input = pad_sequences(input_seq, maxlen=max_length, padding='post')
    return padded_input

# Streamlit app interface
st.title("Spam Detection using RNN")

# Input message
input_message = st.text_area("Enter a message to classify:", "")

if st.button('Predict'):
    if input_message:
        # Initialize and fit the tokenizer inside the Streamlit app
        tokeniser = tf.keras.preprocessing.text.Tokenizer()
        # Sample data to fit the tokenizer (using the training data approach)
        X_train_sample = ['Sample message', 'Another sample message', 'One more message for testing']
        tokeniser.fit_on_texts(X_train_sample)

        # Preprocess the input message
        padded_input = preprocess_text(input_message, tokeniser)

        # Make prediction
        prediction = model.predict(padded_input)
        result = 'Spam' if prediction > 0.5 else 'Not Spam'

        # Display result
        st.write(f'This message is: **{result}**')
    else:
        st.write("Please enter a message to classify.")
