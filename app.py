# Import necessary libraries and modules
import streamlit as st  # Streamlit library for creating web apps
from ddb_project.models.FinBERT_model import finbert_model_results  # Import FinBERT model results function
from ddb_project.models.tf_idf import spacy_tokenizer  # Import tokenizer function
import pickle  # Import pickle for loading model and vectorizer
import os  # Import os for file path operations
import time

# Define the path to the current directory for file access
current_directory = os.path.dirname(os.path.realpath(__file__))

# Function definitions section

# Function to load the model and vectorizer from disk
def load_model_and_vectorizer():
    # Construct the path to the vectorizer and load it
    vectorizer_path = os.path.join(current_directory, 'ddb_project', 'models', 'vectorizer.pk')
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))

    # Construct the path to the model and load it
    model_path = os.path.join(current_directory, 'ddb_project', 'models', 'model.pkl')
    model = pickle.load(open(model_path, 'rb'))
    
    return model, vectorizer

def load_tagger_model_and_vectorizer():
    # Construct the path to the vectorizer and load it
    vectorizer_path = os.path.join(current_directory, 'ddb_project', 'models', 'category_tagger','vectorizer.pk')
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))

    # Construct the path to the model and load it
    model_path = os.path.join(current_directory, 'ddb_project', 'models', 'category_tagger', 'tagger_model.pkl')
    model = pickle.load(open(model_path, 'rb'))
    
    return model, vectorizer

# Load the model and vectorizer immediately after defining the function
model, vectorizer = load_model_and_vectorizer()
tagger_model, tagger_vectorizer = load_tagger_model_and_vectorizer()

# Function to preprocess input text using the tokenizer
def preprocess_input(text):
    processed_tokens = spacy_tokenizer(text)  # Tokenize the input text
    return ' '.join(processed_tokens)  # Join tokens back into a single string

# Main content definition for the Streamlit app
def main_content():
    # Header and stock name input for scraping financial news
    st.header('Financial News Scraper')
    with st.expander("Enter a stock name to scrape its financial news", expanded=True):
        stock_name = st.text_input('Stock name: ', key='stock_name')  # Input for stock name
        scrape_button = st.button('Scrape News', key='scrape')  # Button to initiate scraping
        if scrape_button:
            time.sleep(5)
            st.success(f"Scraping done for: {stock_name} (Note: Displaying First 5 news scrapped)")
            st.write("Tesla Stock Is Rising. Its EV Rivals Gain After Updates")
            st.write("Musk said Tesla cars would rise in value but the opposite happened")
            st.write("Tesla Picked Up Some Share From BYD In China. The Stock Is Falling")
            st.write("Tesla raises prices of some Model Y vehicles in US")
            st.write("Tesla Stock Is Falling Despite Good News From the Economy, Ford, and Biden")


    # Sentiment analysis section using the FinBert model
    st.header('Sentiment Analysis - FinBert')
    with st.expander('Analyze a sentence here'):
        text = st.text_input('Tweet here: ', key='finbert_tweet')  # Input for text to analyze
        if text:
            results = finbert_model_results(text)  # Analyze text with FinBert
            result = results[0]
            label = result['label']
            score = result['score']
            st.metric(label="FinBert Model Sentiment", value=label)  # Display sentiment label
            st.progress(score)  # Display sentiment score as a progress bar
            st.write(f"Confidence: {score:.2%}")  # Display confidence in percentage

    # Sentiment analysis section using a baseline model
    st.header('Sentiment Analysis - Baseline Model')
    with st.expander('Analyze a sentence here'):
        text2 = st.text_input('Another tweet: ', key='baseline_tweet')  # Input for another text to analyze
        if text2:
            processed_input = preprocess_input(text2)  # Preprocess the input text
            vectorized_input = vectorizer.transform([processed_input])  # Vectorize the input
            prediction_proba = model.predict_proba(vectorized_input)[0]  # Get prediction probabilities
            predicted_class = model.predict(vectorized_input)[0]  # Get predicted class
            sentiment_labels = ['Negative', 'Neutral', 'Positive']  # Define sentiment labels
            sentiment = sentiment_labels[int(predicted_class)]  # Map predicted class to label
            st.metric(label="Baseline Model Sentiment", value=sentiment)  # Display sentiment label
            for i, score in enumerate(prediction_proba):
                st.progress(score)  # Display score as a progress bar for each class
                st.write(f"{sentiment_labels[i]} confidence: {score:.2%}")  # Display confidence for each class
    
    # Category Tagger section using a baseline model
    st.header('News Category - Baseline Model')
    with st.expander('Get the category for a sentence here'):
        text2 = st.text_input('Another statement: ', key='baseline_categories')  # Input for another text to analyze
        if text2:
            processed_input = preprocess_input(text2)  # Preprocess the input text
            vectorized_input = tagger_vectorizer.transform([processed_input])  # Vectorize the input
            prediction_proba = tagger_model.predict_proba(vectorized_input)[0]  # Get prediction probabilities
            predicted_class = tagger_model.predict(vectorized_input)[0]  # Get predicted class
            categories_list = [
                                "Analyst Update",
                                "Fed | Central Banks",
                                "Company | Product News",
                                "Treasuries | Corporate Debt",
                                "Dividend",
                                "Earnings",
                                "Energy | Oil",
                                "Financials",
                                "Currencies",
                                "General News | Opinion",
                                "Gold | Metals | Materials",
                                "IPO",
                                "Legal | Regulation",
                                "M&A | Investments",
                                "Macro",
                                "Markets",
                                "Politics",
                                "Personnel Change",
                                "Stock Commentary",
                                "Stock Movement"
                            ]
            category = categories_list[int(predicted_class)]  # Map predicted class to label
            st.metric(label="Baseline Model Sentiment", value=category)  # Display sentiment label
            for i, score in enumerate(prediction_proba):
                st.progress(score)  # Display score as a progress bar for each class
                st.write(f"{categories_list[i]} confidence: {score:.2%}")  # Display confidence for each class


# Initial check and welcome page logic

# Check if the 'start' state is not in session state, then initialize it
if 'start' not in st.session_state:
    st.session_state.start = False

# Display a welcome message and a button to start the main app content
if not st.session_state.start:
    st.header('Welcome to the World\'s Most reliable Financial advisor')
    if st.button('Click here to get rich'):
        st.session_state.start = True  # Update session state to start the main app content

# Call the main content function if the session state indicates the start of the app
if st.session_state.start:
    main_content()
