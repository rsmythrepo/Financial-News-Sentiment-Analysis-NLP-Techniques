import streamlit as st
from transformers import pipeline

# Load the category tagger model
pipe = pipeline("text-classification", 
                model="nickmuchi/finbert-tone-finetuned-finance-topic-classification", 
                token='my_token')

# Define the Streamlit app
def main():
    # Set title and description
    st.title("Finance Topic Classification")
    st.write("Enter a text to classify its finance-related topic.")

    # Create an input text box for user input
    user_input = st.text_input("Enter the text:")

    # When the user submits the text
    if st.button("Classify"):
        # Check if the input is not empty
        if user_input.strip():
            # Classify the input text
            result = pipe(user_input)

            # Display the result
            st.write("Classification Result:")
            st.write(result)
        else:
            st.write("Please enter a text to classify.")

# Run the app
if __name__ == "__main__":
    main()
