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


'''Task 2.1:
2.1 Let´s start with simple baseline (at your own choice).
For example, build a logistic regression model based on pre-trained word embeddings or TF-IDF vectors
of the financial news corpus **
i.e.Build a baseline model with Financial Phrasebank dataset.
What are the limitations of these baseline models?
'''

def get_data():
    df = pd.read_csv("../../data/processed/financial_phrasebank/sentences_50agree_entities.csv")
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

def get_vectorized_dataframe(df):

    vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
    count_matrix_tf = vectorizer.fit_transform(df['docs'].to_list())
    count_array_tf = count_matrix_tf.toarray()
    df_vec = pd.DataFrame(data=count_array_tf, columns=vectorizer.get_feature_names_out())

    return df_vec


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Get data
    df = get_data()
    print(df.head())
    print(df.shape)

    #print(df['entities'].to_list())
    #df1 = df[df.isna().any(axis=1)]

    # TODO get input with no NaNs
    df1 = df.dropna()
    print(df1.shape)

    # Using TF_IDF as our model:
    tfvectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
    tf_idf = tfvectorizer.fit_transform(df1['entities'].to_list())
    tf_idf = tf_idf.toarray()

    df_vec = pd.DataFrame(data=tf_idf, columns=tfvectorizer.get_feature_names_out())
    df_vec.to_csv("../../data/processed/financial_phrasebank/vectorized.csv")

    print(tf_idf.shape)
    print(df_vec.head())