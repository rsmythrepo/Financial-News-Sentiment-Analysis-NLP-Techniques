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

np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


'''Task 2.1:
2.1 Let´s start with simple baseline (at your own choice).
For example, build a logistic regression model based on pre-trained word embeddings or TF-IDF vectors
of the financial news corpus **
i.e.Build a baseline model with Financial Phrasebank dataset.
What are the limitations of these baseline models?
'''

'''Class Notes: 
White box solution to show the probabilities of the testing set
'40% likely that you have cancer'

- augment dataset (sample minority)
- re-weight the loss function 
- ensemble methods 
- conformal prediction (new, advanced)
'''

def get_data():
    df = pd.read_csv("../../data/processed/financial_phrasebank/vectorized_entities.csv", index_col=0)
    return df

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

if __name__ == '__main__':

    # Get data
    df = get_data()
    df = df.dropna()

    # Get features
    X_fin = df.drop('labels', axis=1)

    # Get labels
    Y_fin = df['labels']

    # Train Test Split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_fin, Y_fin, random_state=1)

    # Fit to the Logistic Regression model
    model = LogisticRegression().fit(Xtrain, ytrain)

    # Predict on new data
    y_predictions = model.predict_proba(Xtest)
    print('Predictions')
    print(y_predictions)

    # TODO output probabilities of input
    # Print probabilities for each class - Test
    class_labels = model.classes_
    for i, prob in enumerate(y_predictions):
        print("Sentence:", Xtest.iloc[i])
        print("Predicted probabilities:")
        for j, label in enumerate(class_labels):
            print(f"{label}: {prob[j]}")
        print()

    # Check accuracy score - 82%
    print('Accuracy Score')
    accuracy = accuracy_score(ytest, model.predict(Xtest))
    print("Accuracy:", accuracy)













