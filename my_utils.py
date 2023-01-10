import nltk, re, pprint
import math, re, random, string


# preprocessing text to remove all numbers
def preprocessing(text:str):
    text = re.sub(r'\d', '', text)
    text = re.sub(r'--', '', text)
    
    return text

def remove_stopwords(x, removal_list):
    y = []
    for pair in x:
        count = 1
        for word in pair:
            if word in removal_list:
                count = 0
        if (count==1):
            y.append(pair)
    return (y)

def unigram_stopwords(x, removal_list):
    y = []
    for word in x:
        count = 1
        if word in removal_list:
            count = 0
        if (count == 1):
            y.append(word)
    return (y)

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
