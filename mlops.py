import streamlit as st

# Create a title and header
st.title("My Streamlit App")
st.header("Welcome!")

# Add a text input field
name = st.text_input("Enter your name", "")

# Display a button
if st.button("Submit"):
    st.write(f"Hello, {name}!")

