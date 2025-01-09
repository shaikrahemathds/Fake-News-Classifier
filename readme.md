# Fake News Classifier Project 📰🧠🏆

This project tackles the challenge of identifying fake news articles using machine learning 💪. We compared various models to find the champion for this task!

---

## 🔍 Key Objectives:
- **Evaluate an array of machine learning models**, including traditional (Logistic Regression, Decision Tree, Random Forest) and deep learning models (LSTM, BERT).
- **Uncover the model** that reigns supreme in accurately classifying fake news articles.

---

## 🏆 Project Highlights:
- **Model Extravaganza**: We put seven different machine learning models through their paces on a fitting dataset.
- **BERT Takes the Crown**: BERT emerged victorious, achieving an outstanding accuracy of **97.42%** and demonstrating exceptional precision and recall.
- **In-Depth Analysis**: This README serves as a comprehensive guide, providing a clear breakdown of the project, model evaluation results, valuable insights, and the justification for choosing BERT as the top performer.

---

## Project Structure:
.
├── requirements.txt        # Python dependencies
├── preprocessing_only_text.ipynb  # Preprocessing the dataset
├── standard_ml_model.ipynb      # Training standard ML models
├── ANN.ipynb                   # Training the ANN model
├── BERT.ipynb                  # Training the BERT model
├── LSTM.ipynb                  # LSTM model for classification
├── standard_ml_model.py       # Running predictions with ML models
├── ANN_model.py               # Running predictions with ANN model
├── BERT_model.py              # Running predictions with BERT model


---

---

## 🚀 Getting Started:

### 1. **Clone the Repository:**
```bash
git clone <repository-url>
cd <repository-directory>

### 2. **Install Dependencies:**
```bash
pip install -r requirements.txt

### 3. Download dataset:
[Dataset](https://drive.google.com/file/d/1ZKVzTnCE-U5uMkopcBsPNj0LFtPTX3z4/view?usp=sharing )

### 4. Code Exploration Journey:

- **preprocessing_only_text.ipynb**: Cleans and prepares the dataset for training.
- **standard_ml_model.ipynb**: Trains traditional machine learning models.
- **standard_ml_model.py**: Executes predictions using the trained standard ML models.
- **ANN.ipynb, BERT.ipynb, LSTM.ipynb**: Train ANN, BERT, and LSTM models respectively.
- **ANN_model.py, BERT_model.py**: Run predictions using the trained ANN and BERT models respectively.


---

## Models Evaluated 🧑‍💻

The following machine learning models were trained and evaluated for the given dataset:

### 1. **Logistic Regression** 📈
| Metric     | Value   |
|------------|---------|
| Accuracy   | 94.92%  |
| Precision  | 0.95    |
| Recall     | 0.96    |
| F1-Score   | 0.95    |

---

### 2. **Decision Tree Classifier** 🌳
| Metric     | Value   |
|------------|---------|
| Accuracy   | 90.40%  |
| Precision  | 0.95    |
| Recall     | 0.96    |
| F1-Score   | 0.95    |

---

### 3. **Gradient Boosting Classifier** 🔥
| Metric     | Value   |
|------------|---------|
| Accuracy   | 92.90%  |
| Precision  | 0.95    |
| Recall     | 0.92    |
| F1-Score   | 0.93    |

---

### 4. **Random Forest Classifier** 🌲
| Metric     | Value   |
|------------|---------|
| Accuracy   | 91.22%  |
| Precision  | 0.93    |
| Recall     | 0.87    |
| F1-Score   | 0.90    |

---

### 5. **Artificial Neural Network (ANN)** 🤖
| Metric     | Value   |
|------------|---------|
| Accuracy   | 91.52%  |
| Precision  | 0.93    |
| Recall     | 0.87    |
| F1-Score   | 0.90    |

---

### 6. **Long Short-Term Memory (LSTM)** 🧠
| Metric     | Value   |
|------------|---------|
| Accuracy   | 94.12%  |
| Precision  | 0.91    |
| Recall     | 0.96    |
| F1-Score   | 0.94    |

---

### 7. **BERT** 🔥
| Metric     | Value   |
|------------|---------|
| Accuracy   | 97.42%  |
| Precision  | 0.9759  |
| Recall     | 0.9665  |
| F1-Score   | 0.9712  |

---

## Key Insights 📊
- **BERT** outperformed all other models with the highest accuracy (**97.42%**) and F1-score (**0.9712**), demonstrating superior precision and recall, making it the best candidate for this classification task.
- **Logistic Regression** provided a strong baseline performance with an accuracy of **94.92%**, showing balanced precision and recall across both classes.
- **LSTM** performed well with **94.12%** accuracy and high recall, making it suitable for applications prioritizing minimal false negatives.
- **Gradient Boosting Classifier** achieved competitive results (**92.90%** accuracy), though slightly lower than LSTM and Logistic Regression.
- **ANN, Random Forest**, and **Decision Tree** models exhibited slightly lower performance, indicating they may not be as robust for this dataset.

---

## Final Selected Model: **BERT** 🎯

### Reasons:
- **Highest overall accuracy (97.42%)** and **F1-score (0.9712)**, indicating excellent performance in both precision and recall.
- Superior handling of complex patterns in text data.
- Outperformed both traditional machine learning models and deep learning alternatives.

---

## Recommendations 🏆:
- **Deploy BERT** for production, considering its high performance and robustness.
- For real-time applications, monitor latency and optimize hardware resources due to BERT’s computational requirements.
- Continue experimenting with hyperparameter tuning to further improve BERT’s performance.

---

This project equips you with the knowledge and tools to build a powerful fake news classification system using machine learning. Dive into the code, experiment with different models, and be part of the solution to stop the spread of misinformation!

---

## 📞 Contact Information:
- LinkedIn: [Rahemath](https://www.linkedin.com/in/rahemath/)
- Email: [shaikrahemathds@gmail.com](mailto:shaikrahemathds@gmail.com)