# Group member: Monika Zeng, AJ Aizpurua

import math, re, random, string

def split_into_words(text):
    """
    :param text: input string
    :return: tokenized string
    """
    text = text.lower()
    for punct in string.punctuation:
       text = text.replace(punct, ' '+punct+' ')
    split_words = text.split()

    return split_words

def get_ngrams(ngram_size, tokens):
    """
    :param ngram_size: n-gram size
    :param tokens: tokenized string
    :return: list of ngrams
    ngrams are in tuple form: ((previous words), target word)
    """

    tag = 'START_TOKEN'
    list_tup = []
    ngram_list = []
    
    for i in range(ngram_size-1):
        list_tup.append(tag)
    tup = tuple(list_tup)
    
    ngram_list.append((tup, tokens[0]))
    for i, token in enumerate(tokens[0:len(tokens)-1]):
        list_next = list_tup[1:]
        list_next.append(token)
        tup_next = tuple(list_next)
        list_tup = list_next
        ngram_list.append((tup_next, tokens[i+1]))  
        

    return ngram_list
