import streamlit as st
import generation

# Title of the chatbot UI
st.title("Suvidha")

# Instructions
st.write("Welcome to Suvidha! Please type your query below.")

# Create an input box for user to type their query
user_input = st.text_input("Please enter your Query:")

# Display a response once the user enters a query
if user_input:
    # Simulating the chatbot response (this can be connected to an actual chatbot or NLP model later)
    response = generation.generate_response(user_input)
    
    # Display the response
    st.write(response)