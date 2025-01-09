import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download stopwords and lemmatizer resources if not already done
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained models and vectorizer
with open('lr_model.pkl', 'rb') as f:
    LR = pickle.load(f)
with open('dt_model.pkl', 'rb') as f:
    DT = pickle.load(f)
with open('gb_model.pkl', 'rb') as f:
    GB = pickle.load(f)
with open('rf_model.pkl', 'rb') as f:
    RF = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorization = pickle.load(f)

# Define the clean_text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Function to convert label output
def output_label(n):
    return "Real News" if n == 1 else "Fake News"

# Define the manual_testing function
def manual_testing(news):
    cleaned_news = clean_text(news)
    transformed_news = vectorization.transform([cleaned_news])
    
    pred_LR = LR.predict(transformed_news)
    pred_DT = DT.predict(transformed_news)
    pred_GB = GB.predict(transformed_news)
    pred_RF = RF.predict(transformed_news)
    
    print("\nPredictions:")
    print(f"Logistic Regression: {output_label(pred_LR[0])}")
    print(f"Decision Tree: {output_label(pred_DT[0])}")
    print(f"Gradient Boosting: {output_label(pred_GB[0])}")
    print(f"Random Forest: {output_label(pred_RF[0])}")

# Prompt for user input
if __name__ == "__main__":
    news_input = input("Enter the news text: ")
    manual_testing(news_input)


# NOTE: 
# Always make sure you paste the text inside DOUBLE QUOTES