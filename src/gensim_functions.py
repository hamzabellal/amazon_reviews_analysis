from gensim.models import Phrases
from gensim.models.phrases import Phraser
import pandas
import pickle

def generate_bigrams_and_trigrams(tokens_column, min_count=5, threshold=10):
    """
    Generates bigrams and trigrams from a column of tokens.
    
    Args:
        tokens_column (pd.Series): Column containing lists of tokens.
        min_count (int): Minimum frequency for a bigram/trigram to be retained.
        threshold (int): Sensitivity threshold for forming n-grams.
        
    Returns:
        pd.Series: Column containing tokens enriched with bigrams/trigrams."""
    
    # Prepare the data for Gensim
    sentences = tokens_column.tolist()

    # Create a model for bigrams
    bigram_model = Phrases(sentences, min_count=min_count, threshold=threshold)
    bigram_phraser = Phraser(bigram_model)

    # Create a model for trigrams based on the bigrams
    trigram_model = Phrases(bigram_phraser[sentences], threshold=threshold)
    trigram_phraser = Phraser(trigram_model)

    # Apply bigrams and trigrams to the tokens
    enriched_tokens = [trigram_phraser[bigram_phraser[sentence]] for sentence in sentences]
    
    return enriched_tokens

if __name__ == '__main__':
    import sys

    # Load the file containing tokens_column
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'rb') as f:
        tokens_column = pickle.load(f)

    # Generate bigrams and trigrams
    enriched_tokens = generate_bigrams_and_trigrams(tokens_column)

    # Save the results
    with open(output_file, 'wb') as f:
        pickle.dump(enriched_tokens, f)