from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

'''Task 2.2:
Run inference and compute performance metrics 
to get a sense of how the specialized pre-trained model fares against 
the baseline model.'''

finbert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def finbert_model_results(sentence):
    return nlp(sentence)


# TODO Compute performance metrics




