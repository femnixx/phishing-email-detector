import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "phishing_email.csv")

df = pd.read_csv(file_path)

X_train, X_test, y_train, y_test = train_test_split(
    df['text_combined'],
    df['label'],
    test_size=0.2
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = model.score(X_test_vec, y_test)

print(f"AI logic accuracy is: {accuracy * 100:.2f}% accurate.")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))
