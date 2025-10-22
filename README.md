# FakeNewsDetector

This repository contains a simple Machine Learning model for detecting fake news using TF-IDF features and a Logistic Regression classifier. The main script is `fakenewsdetector.py` (see file in the repository for implementation details).

Important: the model is trained locally each time you run the script; there is no pre-saved trained model in this repository.

## Datasets

This project uses the "Fake and Real News Dataset" from Kaggle. Download and place the following CSV files in the repository root (or update the paths in the script):

- Fake.csv  
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv

- True.csv  
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv

## What this is

This is a Machine Learning pipeline that:

- Loads the two CSV datasets (Fake and True news).
- Labels examples (`FAKE` / `TRUE`) and removes duplicates.
- Cleans the article text (removes HTML, URLs, special characters, lowercases).
- Converts text to TF-IDF features (max_features=5000, ngram_range=(1,3), English stop words).
- Splits the data into train and test (80/20, stratified).
- Trains a Logistic Regression classifier (solver='liblinear').
- Evaluates the model (accuracy, classification report, confusion matrix).
- Allows interactive testing: you can input a text sample and get the predicted label and class probabilities.

The code file implementing this pipeline is `fakenewsdetector.py`.

## Requirements

- Python 3.8+
- pandas
- beautifulsoup4
- scikit-learn

## Important notes and limitations

- Model performance depends on preprocessing and dataset balance. Always inspect precision and recall per class, especially when classes are imbalanced.
- The script removes HTML and URLs and lowercases text but does not apply stemming/lemmatization.
- No model persistence is implemented — training occurs each run.
- The CSVs must include a `text` column, as in the Kaggle dataset.
