import numpy as np
import pandas as pd
import spacy
import string

string.punctuation
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS

stop_words = spacy.lang.en.stop_words.STOP_WORDS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, \
    roc_curve, roc_auc_score
import pickle

'''sklearn model - Logistic Regression'''

'''Task 2.1:
2.1 Let´s start with simple baseline (at your own choice).
For example, build a logistic regression model based on pre-trained word embeddings or TF-IDF vectors
of the financial news corpus **
i.e.Build a baseline model with Financial Phrasebank dataset.
What are the limitations of these baseline models?
'''


def get_data():
    df = pd.read_csv("../../data/processed/financial_phrasebank/vectorized_entities.csv", index_col=0)
    return df


def spacy_tokenizer(sentence):
    punctuations = string.punctuation
    nlp = spacy.load('en_core_web_sm')
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    mytokens = nlp(sentence)
    mytokens = [word.lemma_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens


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

    # Training the model
    model = LogisticRegression(verbose=True)
    model.fit(Xtrain, ytrain)

    # Save the model with pickle
    pickle_out = open("model.pkl", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

    # Predict on new data
    test_predictions = model.predict_proba(Xtest[:5])
    print('Predictions')
    print(test_predictions)

    # Print probabilities for each class - Test
    class_labels = model.classes_
    for i, prob in enumerate(test_predictions):
        words = []
        for column in Xtest.columns:
            if Xtest.iloc[i][column] > 0.0:
                words.append(column)
        sentence = " ".join(words)
        print("Sentence:", sentence)
        print("Predicted probabilities:")
        for j, labelled in enumerate(class_labels):
            if labelled == 0.0:
                label = 'negative'
            elif labelled == 1.0:
                label = 'neutral'
            else:
                label = 'positive'
            print(f"{label}: {prob[j]}")
        print()

    # Check accuracy score - 82%
    print(confusion_matrix(ytest, model.predict(Xtest)))
    print(classification_report(ytest, model.predict(Xtest)))
    print("Accuracy:", accuracy_score(ytest, model.predict(Xtest)))
    print("Recall Score:", recall_score(ytest, model.predict(Xtest), average='macro'))
    print("Precision Score:", precision_score(ytest, model.predict(Xtest), average='macro'))

