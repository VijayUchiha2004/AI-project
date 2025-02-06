# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
def load_data():
    data = pd.read_csv('expenses.csv')
    data.dropna(inplace=True)
    data['Description'] = data['Description'].str.lower()
    return data

# Train Model
def train_model(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Description'])
    y = data['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy
