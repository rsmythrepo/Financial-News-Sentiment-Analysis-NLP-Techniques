from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import scipy.special
import torch

'''Task 2.2:
Run inference and compute performance metrics 
to get a sense of how the specialized pre-trained model fares against 
the baseline model.'''

def get_data():
    df_balanced = pd.read_csv("../../data/processed/financial_phrasebank/sentences_allagree_processed_ver1.2.csv", index_col=0)
    df_balanced = df_balanced.dropna()
    df_balanced = df_balanced.reset_index()
    df_unbalanced = pd.read_csv("../../data/processed/financial_phrasebank/sentences_allagree_processed_ver2_balanced.csv", index_col=0)
    df_unbalanced = df_unbalanced.dropna()
    df_unbalanced = df_unbalanced.reset_index()
    return df_balanced, df_unbalanced

    # Get features and labels
    X_fin = df.drop('label', axis=1)['entities'].astype(str).tolist()  # Convert to string and then to list
    Y_fin = df['label'].to_list()
    label_mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
    # Convert numeric labels to text labels
    Y_fin = [label_mapping[label] for label in Y_fin]

finbert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

 def finbert_model_results(sentence):
     return nlp(sentence)

def finbert_model_results_2(sentences):
    # Initialize lists to store predicted labels
    predicted_labels = []

    # Iterate over each sentence
    for sentence in sentences:
        # Get prediction for the sentence
        result = nlp(sentence)
        predicted_label = result[0]['label']
        predicted_labels.append(predicted_label)

    return predicted_labels

if __name__ == '__main__':
    dfB, dfU = get_data()

    # Balanced
    X_fin_B = dfB.drop('label', axis=1)['entities'].astype(str).tolist()  # Convert to string and then to list
    Y_fin_B = dfB['label'].to_list()
    label_mapping_B = {2: 'positive', 1: 'neutral', 0: 'negative'}
    # Convert numeric labels to text labels
    Y_fin_B = [label_mapping_B[label] for label in Y_fin_B]

    # Get predictions for X_fin data
    predicted_labels_B = finbert_model_results_2(X_fin_B)

    # Generate classification report
    classification_rep_B = classification_report(Y_fin_B, predicted_labels_B)

    # Print the classification report
    print(classification_rep_B)

    # Unbalanced
    # Get features and labels
    X_fin_U = dfU.drop('label', axis=1)['entities'].astype(str).tolist()  # Convert to string and then to list
    Y_fin_U = dfU['label'].to_list()
    label_mapping_U = {2: 'positive', 1: 'neutral', 0: 'negative'}
    # Convert numeric labels to text labels
    Y_fin_U = [label_mapping_U[label] for label in Y_fin_U]

    # Get predictions for X_fin data
    predicted_labels_U = finbert_model_results_2(X_fin_U)

    # Generate classification report
    classification_rep_U = classification_report(Y_fin_U, predicted_labels_U)

    # Print the classification report
    print(classification_rep_U)








