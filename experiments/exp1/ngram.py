from collections import Counter
from itertools import islice

def ngram_diversity(token_lists, n=2):
    """
    Calculate diversity of token sequences using n-gram analysis.

    Args:
        token_lists (list of list of int): List of token sequences
        n (int): n-gram size (default: 2 for bigrams)

    Returns:
        float: Diversity score between 0 and 1
    """
    if not token_lists:
        return 0.0

    # Generate all n-grams
    ngrams = []
    #print(len(token_lists))
    #for tokens in token_lists:
    #    print(len(tokens))
    #     print(tokens)

    for tokens in token_lists:
        ngrams.extend([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    # Calculate unique n-grams ratio
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))

    print(total_ngrams, unique_ngrams)

    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0