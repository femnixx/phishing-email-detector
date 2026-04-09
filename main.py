import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "phishing_email.csv")


MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Loading saved model...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model loaded")
else:
    print("Training model...")
    df = pd.read_csv("phishing_email.csv")
    df = df.dropna(subset=['text_combined'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['text_combined'],
        df['label'],
        test_size=0.2,
        random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Model trained and saved!")

    y_pred = model.predict(X_test_vec)
    accuracy = model.score(X_test_vec, y_test)
    print(f"Accuracy of model: {accuracy*100:.2f}%")
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

# --------------------------------------------------------------------------------

def check_email(my_text): 
    my_vec = vectorizer.transform([my_text])

    guess = model.predict(my_vec)
    prob = model.predict_proba(my_vec)[0]

    status = "PHISHING" if guess[0] == 1 else "SAFE"
    print(f"\nResult: {status} ({prob[1]*100:.2f}% Phishing Probability)")

# print("\n\n--- Email Check Test ---")
# check_email("Hey, are we still meeting for coffee at 3pm?")
# check_email("URGENT: Your account has been locked. Click here to verify now!")
# check_email("Hey, this is the CEO. Your email is compromised. Follow this link to re-enter the database")

# --------------------------------------------------------------------------------
