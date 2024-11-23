# src/text_processing.py

import pandas as pd
import re
import string
import spacy
from nltk.corpus import stopwords

# IInitialization of SpaCy and stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans the raw text"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'@\w+', '', text)  # Removes mentions
        text = text.translate(str.maketrans('', '', string.punctuation))  # Removes punctuation
        text = re.sub(r'\d+', '', text)  # Removes numbers
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Cleans unnecessary spaces
        return text
    return text

def tokenize_and_remove_stopwords(text):
    """Tokenizes and removes stopwords"""
    if isinstance(text, str):
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return tokens
    return text

def lemmatize_text(tokens):
    """Performs lemmatization on a list of tokens"""
    if isinstance(tokens, list):
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
    return tokens


from collections import Counter

def identify_rare_words(tokens_column, threshold=1):
    """
    Identifies rare words in a column of tokens.

    Args:
        tokens_column (pd.Series): Column containing lists of tokens.
        threshold (int): Maximum frequency for a word to be considered rare.

    Returns:
        list: List of rare words.
    """
    # Count word frequencies
    all_tokens = [token for tokens in tokens_column if isinstance(tokens, list) for token in tokens]
    token_counts = Counter(all_tokens)
    
    # Filter rare words
    rare_words = [word for word, freq in token_counts.items() if freq <= threshold]
    
    return rare_words


def remove_rare_words(tokens_column, rare_words):
    """
    Removes rare words from a column of tokens.

    Args:
        tokens_column (pd.Series): Column containing lists of tokens.
        rare_words (list): List of rare words to be removed.

    Returns:
        pd.Series: Column with rare words removed.
    """
    return tokens_column.apply(lambda tokens: [word for word in tokens if word not in rare_words] if isinstance(tokens, list) else tokens)



from textblob import TextBlob

def identify_misspelled_words(tokens_column):
    """
    Identifies misspelled words in a column of tokens.

    Args:
        tokens_column (pd.Series): Column containing lists of tokens.

    Returns:
        set: Set of misspelled words.
    """
    misspelled_words = set()
    
    for tokens in tokens_column:
        if isinstance(tokens, list):
            for word in tokens:
                blob = TextBlob(word)
                if not blob.correct() == word:  # Check if the corrected word is different
                    misspelled_words.add(word)
    
    return misspelled_words

def correct_spelling(tokens_column):
    """
    Corrects the spelling of words in a column of tokens.

    Args:
        tokens_column (pd.Series): Column containing lists of tokens.

    Returns:
        pd.Series: Column with corrected words.
    """
    def correct_tokens(tokens):
        if isinstance(tokens, list):
            corrected_tokens = []
            for word in tokens:
                blob = TextBlob(word)
                corrected_tokens.append(str(blob.correct()))  # Correct each word
            return corrected_tokens
        return tokens

    return tokens_column.apply(correct_tokens)


import subprocess
import sys
import os
import pickle

def call_gensim_bigram_trigram(tokens_column,script_path):

    # Save tokens_column to a temporary file
    input_file = 'tokens_column.pkl'
    output_file = 'enriched_tokens.pkl'

    with open(input_file, 'wb') as f:
        pickle.dump(tokens_column, f)
    # Build the path to the Python interpreter of 'gensim_env'
    gensim_python = os.path.join('C:/Miniconda3/envs/gensim_env', 'Scripts', 'python.exe')
    
    # Construct the command to execute 'gensim_functions.py' with 'gensim_env'
    command = [gensim_python, script_path, input_file, output_file]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Erreur retournée par le script :")
        print(e.stderr)  # Display the standard error
        print(e.stdout)  # Display the standard output (if available)

    # Load the results from the output file
    with open(output_file, 'rb') as f:
        enriched_tokens = pickle.load(f)

    # Return the results
    return enriched_tokens
    

