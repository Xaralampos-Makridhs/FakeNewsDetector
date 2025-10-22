import pandas as pd
from bs4 import BeautifulSoup
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bs4 import MarkupResemblesLocatorWarning

# ---- Ignore BS4 warnings ----
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ---- Text Cleaning Function ----
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()  # remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)           # remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)           # remove special chars
    text = text.lower()                                   # lowercase
    return text

# ---- Load Datasets ----
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

# print("---- FAKE Dataset ----")
# print(df_fake.head(3))
# print(df_fake.columns)
# print(df_fake.dtypes)
# print("Duplicate FAKE rows:", df_fake.duplicated(subset=['text']).sum())
# print("Unique Subjects FAKE:", df_fake['subject'].unique())

# print("---- TRUE Dataset ----")
# print(df_true.head(3))
# print(df_true.columns)
# print(df_true.dtypes)
# print("Duplicate TRUE rows:", df_true.duplicated(subset=['text']).sum())
# print("Unique Subjects TRUE:", df_true['subject'].unique())

# ---- Add Labels ----
df_fake['label'] = 'FAKE'
df_true['label'] = 'TRUE'

# ---- Remove duplicates ----
df_fake.drop_duplicates(subset=['text'], keep='first', inplace=True)
df_true.drop_duplicates(subset=['text'], keep='first', inplace=True)

# ---- Combine datasets ----
df_combined = pd.concat([df_true, df_fake], axis=0)
df_combined = df_combined.sample(frac=1).reset_index(drop=True)
# print("---- Combined Dataset ----")
# print(df_combined.head(5))
# print("Total rows:", len(df_combined))

# ---- Clean text ----
df_combined['clean_text'] = df_combined['text'].apply(clean_text)
# print("---- Sample Cleaned Text ----")
# print(df_combined[['text','clean_text']].head(5))

# ---- Train/Test Split ----
X = df_combined['clean_text']
y = df_combined['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# print("---- Train/Test Split ----")
# print("Training set size:", X_train.shape[0])
# print("Test set size:", X_test.shape[0])

#Accuracy: κοντά στο 1, αλλά ελέγχεις και precision/recall γιατί οι κλάσεις μπορεί να μην είναι ισορροπημένες.

# ---- TF-IDF Vectorization ----
vectorizer = TfidfVectorizer(
    max_features=5000,        # περιορισμος λεξεων
    ngram_range=(1,3),        # 1gram--> "this is fake news" → ["this", "is", "fake", "news"]
                               # 2gram--> "this is fake news" → ["this is", "is fake", "fake news"]
    stop_words='english'      # αφαιρεση συχνων λεξεων
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # σωστό transform, όχι fit_transform

# print("---- TF-IDF Features ----")
# print("X_train shape:", X_train_tfidf.shape)
# print("X_test shape:", X_test_tfidf.shape)

#Classification report: όλα metrics >0.9 είναι εξαιρετικό, αλλά >0.8 είναι αποδεκτό για text classification με μεγάλο dataset.

# ---- Logistic Regression Model ----
model = LogisticRegression(solver='liblinear')
y_train = y_train.squeeze()
y_test = y_test.squeeze()
model.fit(X_train_tfidf, y_train)
y_prediction = model.predict(X_test_tfidf)
print("---- Model Trained ----")

#Confusion matrix: διαγώνιοι μεγάλοι, off-diagonals μικροί.

# ---- Metrics ----
print("---- Model Evaluation ----")
print("Accuracy Score:", accuracy_score(y_test, y_prediction))
print("\nClassification Report:\n", classification_report(y_test, y_prediction))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_prediction))

# --- TEST ---
text_test=input('Enter the testing text: \n')
cleaned_text=clean_text(text_test)
text_tfidf=vectorizer.transform([cleaned_text])
prediction=model.predict(text_tfidf)
probabilities=model.predict_proba(text_tfidf)

print("\nPredicted label:", prediction[0])
print("Probabilities for each class:", probabilities[0])