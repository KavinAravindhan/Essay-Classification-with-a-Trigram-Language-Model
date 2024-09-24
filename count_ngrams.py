import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



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

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        
        # Iterate through each sentence in the corpus
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)  
            bigrams = get_ngrams(sentence, 2)   
            trigrams = get_ngrams(sentence, 3)
            
            for unigram in unigrams:
                self.unigramcounts[unigram] += 1
            
            for bigram in bigrams:
                self.bigramcounts[bigram] += 1

            for trigram in trigrams:
                self.trigramcounts[trigram] += 1

        self.unigramcounts['START',] = 0
        # print("Dictionary unigramcounts - START key")
        # print(self.unigramcounts['START',])

        return

if __name__ == "__main__":

    model = TrigramModel("brown_train.txt") 

    print("n-gram counts")
    print(model.trigramcounts[('START','START','the')])
    print(model.bigramcounts[('START','the')])
    print(model.unigramcounts[('the',)])

# Counting n-grams

# Now it's your turn again. In this step, you will implement the method count_ngramsthat should count the occurrence frequencies for ngrams in the corpus. The method already creates three instance variables of TrigramModel, which store the unigram, bigram, and trigram counts in the corpus. Each variable is a dictionary (a hash map) that maps the n-gram to its count in the corpus. 
# For example, after populating these dictionaries, we want to be able to query

# >>> model.trigramcounts[('START','START','the')]
# 5478
# >>> model.bigramcounts[('START','the')]
# 5478
# >>> model.unigramcounts[('the',)]
# 61428