import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

print(df.head())

# Keep useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert label to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

msg = ["Congratulations! You won a free lottery"]

msg_vec = vectorizer.transform(msg)

prediction = model.predict(msg_vec)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")
