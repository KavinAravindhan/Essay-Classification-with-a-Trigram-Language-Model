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

        # Storing the total number of unigrams (total word count) for efficiency
        self.total_unigrams = sum(self.unigramcounts.values())


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

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        # Count of the trigram (u, w, v)
        trigram_count = self.trigramcounts.get(trigram, 0)
        
        # Count of the bigram (u, w) which is the context
        bigram = trigram[:2]
        bigram_count = self.bigramcounts.get(bigram, 0)

        # print("Trigram Count")
        # print(trigram, trigram_count)

        # print("Bigram Count")
        # print(bigram, bigram_count)

        # print("Lexicon Count")
        # print(len(self.lexicon))
        
        if bigram_count > 0:
            # P(v | u, w) = count(u, w, v) / count(u, w)
            return trigram_count / bigram_count
        else:
            # If the bigram context is unseen, return uniform distribution over the vocabulary
            return 1 / len(self.lexicon)


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # Count of the bigram (u, w)
        bigram_count = self.bigramcounts.get(bigram, 0)
        
        # Count of the unigram (u), which is the context
        unigram = bigram[:1]
        unigram_count = self.unigramcounts.get(unigram, 0)

        # print("Bigram Count")
        # print(bigram, bigram_count)

        # print("Unigram Count")
        # print(unigram, unigram_count)

        # print("Lexicon Count")
        # print(len(self.lexicon))
        
        if unigram_count > 0:
            # P(w | u) = count(u, w) / count(u)
            return bigram_count / unigram_count
        else:
            # If the unigram context is unseen, return uniform distribution over the vocabulary
            return 1 / len(self.lexicon)


    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # Total number of unigrams (words) in the corpus
        # total_unigrams = sum(self.unigramcounts.values())
        # print(total_unigrams)
        
        # Count of the unigram (w)
        unigram_count = self.unigramcounts.get(unigram, 0)

        # print("Unigram Count")
        # print(unigram, unigram_count)

        # print("Total Unigrams - Word Count")
        # print(self.total_unigrams)

        # print("Lexicon Count")
        # print(len(self.lexicon))
        
        # P(w) = count(w) / total number of unigrams
        if self.total_unigrams > 0:
            return unigram_count / self.total_unigrams
        else:
            return 0.0

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it. 

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        # Extract individual words from the trigram
        # w1, w2, w3 = trigram[0:1], trigram[1:], trigram[:-1]
        w1, w2, w3 = trigram

        # print("w1", w1)
        # print("w2", w2)
        # print("w3", w3)
        
        # Compute the raw probabilities
        trigram_prob = self.raw_trigram_probability(trigram)
        bigram_prob = self.raw_bigram_probability((w2, w3))
        unigram_prob = self.raw_unigram_probability((w3,))
        
        # Apply linear interpolation
        smoothed_prob = (lambda1 * trigram_prob) + (lambda2 * bigram_prob) + (lambda3 * unigram_prob)
        
        return smoothed_prob


if __name__ == "__main__":

    model = TrigramModel("brown_train.txt") 

    print("n-gram counts")
    print(model.trigramcounts[('START','START','the')])
    print(model.bigramcounts[('START','the')])
    print(model.unigramcounts[('the',)])

    print("\n")

    print("raw probab")
    print(model.raw_trigram_probability(('START','START','the')))
    print(model.raw_bigram_probability(('START','the')))
    print(model.raw_unigram_probability(('the',)))

    print("\n")

    print("Smoothing")
    print(model.smoothed_trigram_probability(('START','START','the')))

