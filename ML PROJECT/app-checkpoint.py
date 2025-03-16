import streamlit as st
st.title('Welcome to  Language Detection App')
import pickle 
import streamlit as st
import numpy as np
import pandas as pd



import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the trained ML model
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Load the trained CountVectorizer
with open("count_vectorizer.pkl", "rb") as file:
    loaded_vectorizer = pickle.load(file)

def prediction(input_data):
    """Predicts the language based on the input text."""
    # Ensure input_data is a string
    if not isinstance(input_data, str):
        input_data = str(input_data)

    # Transform input using the trained CountVectorizer
    vectorized_form = loaded_vectorizer.transform([input_data]).toarray()

    # Predict using the loaded model
    pred = loaded_model.predict(vectorized_form)  # Use vectorized_form instead of arr2

    return f'The text "{input_data}" is of {pred[0]} language type.'

def main():
    """Streamlit UI for language detection."""

    
    st.header("Please Input Your Text")
    input_text = st.text_input("Enter text:", value="Hii, How are you?")  # Changed "Input" to "input_text"
    
    st.write("### User Input:")
    st.write(input_text)  # Directly displaying input instead of DataFrame
    
    if st.button("Detect Language"):
        result = prediction(input_text)  # Pass raw text instead of a DataFrame
        st.success(result)

if __name__ == "__main__":
    main()




