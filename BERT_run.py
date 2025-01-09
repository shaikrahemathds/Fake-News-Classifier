import torch
import re
import nltk
from transformers import BertForSequenceClassification, BertTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure stopwords and lemmatizer are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and tokenizer from the downloaded folder
model = BertForSequenceClassification.from_pretrained('./saved_model_bert')
tokenizer = BertTokenizer.from_pretrained('./saved_model_bert')

# Set model to evaluation mode
model.eval()

# Ensure the model uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU (or CPU if GPU is not available

# Define the preprocessing function as you did in Colab
def clean_text(text):
    # 1. Convert text to lowercase
    text = text.lower()
    
    # 2. Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Remove special characters, numbers, and keep only alphabets and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # 6. Lemmatization (to get the root form of words)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Function to predict
def predict(text):
    cleaned_text = clean_text(text)  # Clean the input text

    # Tokenize the cleaned text
    inputs = tokenizer(cleaned_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

    # Make prediction (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label (0 or 1)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Interpret the result
    return "Fake News" if predicted_class == 0 else "Real News"

# Example of using the model
user_input = input("Enter the news text: ")
result = predict(user_input)
print(f"Prediction: {result}")
