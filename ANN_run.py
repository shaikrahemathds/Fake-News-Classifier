# Import necessary libraries
import torch
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Define the ANN model class (same as the one used for training)
class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.linear1 = torch.nn.Linear(10000, 5000)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(5000, 1000)
        self.relu2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(1000, 200)
        self.relu3 = torch.nn.ReLU()

        self.linear4 = torch.nn.Linear(200, 20)
        self.relu4 = torch.nn.ReLU()

        self.linear5 = torch.nn.Linear(20, 2)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)

        out = self.linear2(out)
        out = self.relu2(out)

        out = self.linear3(out)
        out = self.relu3(out)

        out = self.linear4(out)
        out = self.relu4(out)

        out = self.linear5(out)
        return out

# Function to clean and preprocess the text input
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Function to preprocess the input text (title and text)
def preprocess_input(text, vectorizer_text):
    cleaned_text = clean_text(text)

    new_text = [cleaned_text]

    text_matrix = vectorizer_text.transform(new_text).toarray()

    input_vector = text_matrix

    input_tensor = torch.Tensor(input_vector)
    
    return input_tensor

# Load the saved vectorizers (same ones used for training)
import joblib
vectorizer_text = joblib.load('vectorizer_text.pkl')

# Load the saved model
model = ANN()
model.load_state_dict(torch.load("ann_model.pth"))
model.eval()  # Set the model to evaluation mode

# Example to take user input for title and text
raw_text = input("Enter the content of the article: ")  # Take input for text

# Preprocess the input
input_tensor = preprocess_input(raw_text, vectorizer_text)

# Make the prediction
with torch.no_grad():  # Disable gradient computation for inference
    output = model(input_tensor)  # Forward pass through the model

# Get the predicted class (0 or 1)
predicted_class = torch.max(output, 1)[1].item()

# Interpret the prediction
if predicted_class == 1:
    print("The news is real.")
else:
    print("The news is fake.")
