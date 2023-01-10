import re
from collections import Counter
from lib2to3.pgen2 import token
import numpy as np
from scipy.spatial import distance
import time, os


def tokenize_text(text):
    lowercase_text = text.lower()
    # split_words = re.split("\W+", lowercase_text)
    split_words = re.split(" ", lowercase_text)
    uniq_words = Counter(split_words)
    return uniq_words

def preprocessing(text:str):
    text = text.replace('_', '')
    text = text.replace('\n','')
    text = re.sub(r'\d', '', text)
    text = re.sub(r'--', '', text)
    text = re.sub(r'([a-zA-Z])([—;,.!““”’()\[\]"-:?\'])', r'\1 \2', text)
    text = re.sub(r'([—;,.!““”’()\[\]"-:?\'])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([—;,.!““”’()\[\]"-:?\'])([—;,.!““”’()\[\]"-:?\'])', r'\1 \2', text)
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