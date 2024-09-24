def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """

    # Padding the sequence with 'START' and 'STOP' tokens
    if n != 1:
        padded_sequence = ['START'] * (n - 1) + sequence + ['STOP']
    else:
        padded_sequence = ['START'] * (n) + sequence + ['STOP']
    
    # Generate n-grams as tuples
    ngrams = []
    for i in range(len(padded_sequence) - n + 1):
        ngram = tuple(padded_sequence[i:i+n])
        ngrams.append(ngram)
    
    # ngrams = [tuple(padded_sequence[i:i+n]) for i in range(len(padded_sequence) - n + 1)]

    return ngrams


print(get_ngrams(["natural","language","processing"],1))
print(get_ngrams(["natural","language","processing"],2))
print(get_ngrams(["natural","language","processing"],3))


# For example: 

# >>> get_ngrams(["natural","language","processing"],1)
# [('START',), ('natural',), ('language',), ('processing',), ('STOP',)]
# >>> get_ngrams(["natural","language","processing"],2)
# ('START', 'natural'), ('natural', 'language'), ('language', 'processing'), ('processing', 'STOP')]
# >>> get_ngrams(["natural","language","processing"],3)
# [('START', 'START', 'natural'), ('START', 'natural', 'language'), ('natural', 'language', 'processing'), ('language', 'processing', 'STOP')]