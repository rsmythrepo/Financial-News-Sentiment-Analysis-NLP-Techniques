import spacy
import string
string.punctuation
nlp = spacy.load("en_core_web_sm")
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

''' pytorch '''

# Assuming you have already loaded and preprocessed your data into X (features) and y (labels)
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


if __name__ == '__main__':

    # Get data
    df = get_data()
    df = df.dropna()

    # Get features
    X = df.drop('labels', axis=1)
    print('X shape')
    print(X.shape)

    # Get labels
    y = df['labels']
    print('Y shape')
    print(y.shape)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)


    # Define logistic regression model
    class LogisticRegression(nn.Module):
        def __init__(self, input_size, output_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, output_size)

        def forward(self, x):
            out = self.linear(x)
            return out


    # Initialize model
    input_size = X_train.shape[1]
    output_size = 3
    model = LogisticRegression(input_size, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 600
    batch_size = 64

    for epoch in range(num_epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            inputs = X_train_tensor[i:i + batch_size]
            labels = y_train_tensor[i:i + batch_size]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Accuracy: {accuracy:.4f}')
