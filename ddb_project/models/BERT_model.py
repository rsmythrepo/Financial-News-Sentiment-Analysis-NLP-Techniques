from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer
import torch
import pandas as pd
import pickle


'''Task 2.3:
Fine-tune a pre-trained model such a base BERT model on a small labeled dataset.

ALERT CPU CAPACITY ISSUES - Need to use google collab

Finally: Compare the baseline model with the fine-tuned bert model
'''


'''Example Dataset'''

# Define our example dataset
data = {
    'text': [
        'I love this restaurant, the food is great!',
        'This movie is terrible, I would not recommend it.',
        'I am having a wonderful day!',
        'The service was horrible, I am never going back to that place.',
        'This is a beautiful piece of art.',
        'That is an ugly car.'
    ],
    'label': [0, 1, 0, 1, 0, 1]  # 0 indicates a positive sentiment, 1 indicates a negative sentiment
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)


'''Loading & Preprocessing the Data'''

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Split the dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Convert the texts and labels into tensors
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)
train_labels = torch.tensor(train_labels.tolist())
val_labels = torch.tensor(val_labels.tolist())


'''Fine-tuning the BERT Model'''


