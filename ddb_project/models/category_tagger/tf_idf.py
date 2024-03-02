import numpy as np
import pandas as pd
import spacy
import string
string.punctuation
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS
stop_words = spacy.lang.en.stop_words.STOP_WORDS
np.random.seed(42)
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


'''Task 2.1:
Creating a Tokenizer and Vectorizer 
Saving the Vectorizer
'''

def get_data():
    df = pd.read_csv("../../data/processed/financial_phrasebank/sentences_allagree_processed_ver1.2.csv")
    return df

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    punctuations = string.punctuation
    nlp = spacy.load('en_core_web_sm')
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    mytokens = nlp(sentence)
    mytokens = [ word.lemma_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    return mytokens

if __name__ == '__main__':

    # Get data
    df = get_data()
    df1 = df.dropna()

    # Using TF_IDF as our model:
    tfvectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
    tf_idf = tfvectorizer.fit_transform(df1['entities'].to_list())
    tf_idf = tf_idf.toarray()

    # Save the vectorizer
    with open('vectorizer.pk', 'wb') as fin:
        pickle.dump(tfvectorizer, fin)

    # Save the vectorized corpus to a file for observing the output
    df_vec = pd.DataFrame(data=tf_idf, columns=tfvectorizer.get_feature_names_out())
    df_vec['labels'] = df1['label']
    df_vec.to_csv("../../data/processed/financial_phrasebank/vectorized_entities.csv")




