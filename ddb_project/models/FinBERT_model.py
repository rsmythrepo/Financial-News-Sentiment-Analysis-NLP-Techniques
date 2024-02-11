from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

'''Task 2.2:
Run inference and compute performance metrics 
to get a sense of how the specialized pre-trained model fares against 
the baseline model.'''

finbert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

sentences = ["there is a shortage of capital, and we need extra financing",
             "growth is strong and we have plenty of liquidity",
             "there are doubts about our finances",
             "profits are flat"]

results = nlp(sentences)
print(results)

# TODO Compute performance metrics & compare with logistical regression
# Accuracy: The fraction of times the model makes a correct prediction as compared to the total predictions it makes.
# Precision: The percent of true positives identified given all positive cases.
# & Recall: The percent of true positives versus combined true and false positives.



