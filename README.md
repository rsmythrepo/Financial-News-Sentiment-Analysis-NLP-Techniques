
An interface allowing for a comparision of a pre-trained FinBert model and a baseline model trained by our team.


To run the app:

Step 1: run tf_idf.py
- This will create the vectorized entities and save it to
data > processed > financial_phrasebank > vectorized_entities.csv
- This script also saves the vectorizer with pickle.


Step 2: run logistic_regression_model.py
- This script will train a logisitic regression model with the 
vectorized entities and save the model with pickle.

Step 3: 'streamlit run app.py'
- This will deploy the streamlit app 
