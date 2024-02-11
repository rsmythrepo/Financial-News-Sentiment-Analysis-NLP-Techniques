import numpy as np
import pandas as pd

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

def get_data():
    df_labels = pd.read_csv("../../data/processed/financial_phrasebank/sentences_50agree_entities.csv")
    df_labels = df_labels.dropna()
    df = pd.read_csv("../../data/processed/financial_phrasebank/vectorized.csv", index_col=0)
    df['label'] = df_labels['label']
    return df


if __name__ == '__main__':

    # Get data
    df = get_data()

    df1 = df[df.isna().any(axis=1)]
    print(df1)

    # TODO get rid of NaNs in the processed file
    df = df.dropna()
    print(df.shape)

    # Get features
    X_fin = df.drop('label', axis=1)
    print('X shape')
    print(X_fin.shape)

    # Get labels
    Y_fin = df['label']
    print('Y shape')
    print(Y_fin.shape)

    # Train Test Split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_fin, Y_fin, random_state=1)


    # Fit to the Logistic Regression model
    model = LogisticRegression().fit(Xtrain, ytrain)

    # Predict on new data
    y_predictions = model.predict_proba(Xtest)
    print('Predictions')
    print(y_predictions)

    # Check accuracy score
    print('Accuracy Score')
    #score = accuracy_score(ytest,y_predictions)
    #print(score)





