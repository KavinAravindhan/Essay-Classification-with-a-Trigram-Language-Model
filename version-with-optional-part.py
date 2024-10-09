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

        # To store the number of sentences
        self.num_sentences = 0 
        
        # Iterate through each sentence in the corpus
        for sentence in corpus:
            self.num_sentences += 1  # Increment sentence count for each sentence

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
            # return 1 / len(self.lexicon)

            # Updating as per the prof's instruction
            # If you get a bigram (START, the) you calculate the probability as count(START, the) / #num_sentences
            if bigram[0] == 'START':
                return bigram_count / self.num_sentences
            else:
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
    
    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = []
        previous_tokens = ("START", "START")  # Start with the initial context

        for _ in range(t):
            # Get all possible next words based on the current context
            next_word_probs = self.get_next_word_probabilities(previous_tokens)

            # If no next words are found, break the loop
            if not next_word_probs:
                break

            # Draw a random word from the distribution
            words, probabilities = zip(*next_word_probs.items())
            next_word = random.choices(words, probabilities)[0]

            # Add the selected word to the result
            result.append(next_word)

            # Update the previous tokens
            previous_tokens = (previous_tokens[1], next_word)

            # Stop generating if we hit the "STOP" token
            if next_word == "STOP":
                break

        return result

    def get_next_word_probabilities(self, previous_tokens):
        """
        Given the previous two tokens, return a dictionary of next words and their raw trigram probabilities.
        """
        next_word_probs = {}
        for word in self.lexicon:  # Iterate through the vocabulary
            trigram = (previous_tokens[0], previous_tokens[1], word)
            prob = self.raw_trigram_probability(trigram)
            if prob > 0:  # Only include words with a non-zero probability
                next_word_probs[word] = prob
        return next_word_probs 
    

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
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        total_log_prob = 0.0
        
        # Get the trigrams for the sentence
        trigrams = get_ngrams(sentence, 3)
        
        # Iterate through each trigram in the sentence
        for trigram in trigrams:
            # print(trigram, self.trigramcounts.get(trigram, 0))
            # print(f"Trigram: {trigram}, Smoothed Probability: {self.smoothed_trigram_probability(trigram)}")

            # Get the smoothed trigram probability
            prob = self.smoothed_trigram_probability(trigram)
            
            # If the probability is 0, the log probability would be negative infinity.
            # To avoid math domain error, handle that case explicitly
            if prob == 0:
                return float("-inf")
            
            # Convert the probability into log2 space and add to the total log probability
            total_log_prob += math.log2(prob)
        
        return total_log_prob
    
    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
    
        log_prob_sum = 0.0  # Sum of log probabilities of sentences
        total_words = 0      # Total number of word tokens

        # Iterate through each sentence in the corpus
        for sentence in corpus:
            # Get log probability of the sentence
            log_prob_sum += self.sentence_logprob(sentence)

            # print("Sentence")
            # print(sentence)
            
            # Count the number of word tokens
            # total_words += len(sentence) # START, STOP should not be counted
            total_words += len(sentence) + 1 # START should not be counted, STOP should be included

        # Compute perplexity
        if total_words == 0:
            return float('inf')  # Avoid division by zero if there are no words
        
        avg_log_prob = log_prob_sum / total_words  # Compute l
        perplexity_value = 2 ** (-avg_log_prob)  # Perplexity is 2^(-l)
        
        return perplexity_value



def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    """
    COMPLETE THIS METHOD (PART 7)
    Trains two trigram models on different training data (high vs low skill essays).
    Then, it computes the perplexity of each test essay using both models.
    The model with lower perplexity is used to predict the class of the essay.
    """
    
    # Train the first model (high skill essays)
    model1 = TrigramModel(training_file1)
    
    # Train the second model (low skill essays)
    model2 = TrigramModel(training_file2)
    
    total = 0  # Total number of test essays
    correct = 0  # Number of correct predictions

    # Evaluate test essays from testdir1 (high skill)
    for f in os.listdir(testdir1):
        # Compute perplexity for the essay using both models
        pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        
        # If model1 has lower perplexity, it's correct (since it's a high skill essay)
        if pp_model1 < pp_model2:
            correct += 1
        
        total += 1  # Count this essay as evaluated

    # Evaluate test essays from testdir2 (low skill)
    for f in os.listdir(testdir2):
        # Compute perplexity for the essay using both models
        pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        
        # If model2 has lower perplexity, it's correct (since it's a low skill essay)
        if pp_model2 < pp_model1:
            correct += 1
        
        total += 1  # Count this essay as evaluated

    # Calculate and return accuracy (correct predictions / total predictions)
    # accuracy = correct / total if total > 0 else 0.0
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0.0
    return accuracy




if __name__ == "__main__":

    model = TrigramModel('hw1_data/brown_train.txt') 

    # print("n-gram counts")
    # print(model.trigramcounts[('START','START','the')])
    # print(model.bigramcounts[('START','the')])
    # print(model.unigramcounts[('the',)])
    # # print(model.trigramcounts[('It','urged','that')])

    # print("\n")

    # print("raw probab")
    # print("Using bigram (START, the) / number_of_sentences - For Bigram Probability")
    # print(model.raw_trigram_probability(('START','START','the')))
    # print(model.raw_bigram_probability(('START','the')))
    # print(model.raw_unigram_probability(('the',)))
    # # print(model.raw_trigram_probability(('It','urged','that')))

    print("\n")

    print("Generate Sentence Function")
    sentence = model.generate_sentence(t=20)
    print(sentence)

    print("\n")

    # print("Smoothing")
    # print(model.smoothed_trigram_probability(('START','START','the')))

    # print("\n")

    # print("Sentence Probability")
    # print(model.sentence_logprob((['START','START','the'])))

    # # print(model.sentence_logprob((['It','urged','that'])))
    # # It urged that the city

    # print("\n")

    # dev_corpus = corpus_reader('hw1_data/brown_train.txt', model.lexicon)
    # print("Perplexity - Train")
    # print(model.perplexity(dev_corpus))

    # print("\n")

    # dev_corpus = corpus_reader('hw1_data/brown_test.txt', model.lexicon)
    # print("Perplexity - Test")
    # print(model.perplexity(dev_corpus))

    # print("\n")

    # print("Essay scoring experiment")
    # acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', 'hw1_data/ets_toefl_data/test_high', 'hw1_data/ets_toefl_data/test_low')
    # print(acc)