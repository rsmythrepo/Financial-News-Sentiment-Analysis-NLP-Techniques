import streamlit as st
from ddb_project.models.FinBERT_model import finbert_model_results
import pickle
import os

# Although it looks unused, it is being used by the vectorizer - do not remove
from ddb_project.models.tf_idf import spacy_tokenizer

current_directory = os.path.dirname(os.path.realpath(__file__))

# Load the Vectorizer
vectorizer_path = os.path.join(current_directory, 'ddb_project', 'models', 'vectorizer.pk')
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Load the Baseline model - logistic_regression_model.py - sklearn
model_path = os.path.join(current_directory, 'ddb_project', 'models', 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

# Test Finbert models with input field
st.header('Sentiment Analysis - FinBert')
with st.expander('Analyze a sentence here'):
    text = st.text_input('Tweet here: ')
    if text:
        results = finbert_model_results(text)
        result = results[0]
        label = result['label']
        score = result['score']
        st.write('Tweet:', text)
        st.write('Results: ', label)
        st.write('Score: ', score)


# Test our baseline model with input field
st.header('Sentiment Analysis - Baseline Model')
with st.expander('Analyze a sentence here'):
    text2 = st.text_input('Another tweet: ')
    if text2:
        # vectorize and predict
        list = []
        list.append(text2)
        test = vectorizer.transform(list)
        test_pred = model.predict_proba(test)

        # Display sentence
        st.write('Tweet:', text2)

        class_labels = model.classes_
        for i, prob in enumerate(test_pred):
            for j, labelled in enumerate(class_labels):
                if labelled == 0.0:
                    label = 'negative'
                elif labelled == 1.0:
                    label = 'neutral'
                else:
                    label = 'positive'
                # Display each label and its probability
                st.write('Result: ', label)
                st.write('Score: ', prob[j])