import streamlit as st
from ddb_project.models.FinBERT_model import finbert_model_results

# Streamlit test example
st.header('Sentiment Analysis - FinBert')
with st.expander('Analyze Text'):
    text = st.text_input('Tweet here: ')
    if text:
        results = finbert_model_results(text)
        result = results[0]
        label = result['label']
        score = result['score']
        st.write('Tweet:', text)
        st.write('Results: ', label)
        st.write('Score: ', score)


st.header('Sentiment Analysis - Baseline Model')
with st.expander('Analyze Text'):
    text2 = st.text_input('Another tweet: ')
    if text2:
        results = finbert_model_results(text2)
        result  = results[0]
        label = result['label']
        score = result['score']
        st.write('Tweet:', text2)
        st.write('Results: ', label)
        st.write('Score: ', score)