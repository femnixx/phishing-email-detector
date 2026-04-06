import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "phishing_email.csv")

df = pd.read_csv(file_path)

df['totla_text'] = df['subje']