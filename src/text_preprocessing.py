# src/text_processing.py

import pandas as pd
import re
import string
import spacy
from nltk.corpus import stopwords

# Initialisation de SpaCy et des mots vides
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Nettoie le texte brut."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'@\w+', '', text)  # Supprime les mentions
        text = text.translate(str.maketrans('', '', string.punctuation))  # Supprime la ponctuation
        text = re.sub(r'\d+', '', text)  # Supprime les nombres
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Nettoie les espaces inutiles
        return text
    return text

def tokenize_and_remove_stopwords(text):
    """Tokenise et supprime les mots vides."""
    if isinstance(text, str):
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return tokens
    return text

def lemmatize_text(tokens):
    """Effectue la lemmatisation sur une liste de tokens."""
    if isinstance(tokens, list):
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
    return tokens

