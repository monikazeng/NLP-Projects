# This is the program for Homer GloVe Embeddings

import numpy as np
from scipy.spatial import distance
from collections import Counter
from my_util import tokenize_text, preprocessing
import pandas as pd


def load_glove_embeddings(filename):
    # returns a dictionary of words with their corresponding embedding
    with open(filename, 'r', encoding='utf-8') as file:
        model = {}
        for line in file:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    return model


def homer_model(tokens, model):
    homer = {}
    skipped = []
    top3k = []
    for token in tokens:
        if token in model.keys():
            homer[token] = model[token]
            if (len(top3k)<3000):
                top3k.append(token)
        else:
            skipped.append(token)
    return homer, skipped, top3k     


def save_model(model):
    converted_model = pd.DataFrame.from_dict(model)
    converted_model = converted_model.T
    converted_model.to_csv('./data/HomerVe.csv', sep = " ", header=False)


def save_list(words_list, filename):
    with open(filename, 'w') as f:
        # with open('./data/skipped_words.txt', 'w') as f:
        for word in words_list:
            f.write(f"{word}\n")
            

def find_nearest_neighbor(model, word_vector):
    max = 0
    similarity = 0
    for key in model.keys():
        similarity = compute_cosine_similarity(word_vector, model[key])
        if similarity != 1:    
            if similarity > max:
                max = similarity
                neighbor = key
    # print('the cosine similarity score: ' + str(max))
    return neighbor


def find_nearest_neighbors(model, word_vector, k):
    neighbors = []
    distances = Counter()
    for word in model:
        similarity = compute_cosine_similarity(word_vector, model[word])
        if similarity != 1:
            distances[word] = similarity
    neighbors = list(zip(*distances.most_common(k)))[0]

    return neighbors


# def find_nearest_neighbors(model, word_vector, k):
#     neighbors = []
#     tup = () 
#     curr = 0
#     for key in model.keys(): 
#         curr = compute_cosine_similarity(word_vector, model[key])
#         if curr != 1:
#             tup = (key, curr)
#             neighbors.append(tup)
#     return sorted(neighbors, key=lambda x:x[1], reverse=True)[:k]        


def compute_cosine_similarity(word_vector1, word_vector2):
    similarity = 1 - distance.cosine(word_vector1, word_vector2)
    return similarity




def main():

    try:
        homer_embeddings = load_glove_embeddings('./data/HomerVe.csv')
    except:
        glove_pathname = '/deac/csc/classes/csc391/data/glove.6B.300d.txt'
        glove_embeddings = load_glove_embeddings(glove_pathname)

        myCorpus = open('./data/HOMER.txt', 'r').read()
        myCorpus = preprocessing(myCorpus)
        tokens = tokenize_text(myCorpus)

        model, skipped_list, top3k = homer_model(tokens, glove_embeddings)
        save_model(model)
        skipped_list = Counter(skipped_list)
        skipped_list = sorted(skipped_list, key=skipped_list.get, reverse=True)
        save_list(skipped_list, './output/skipped_words.txt')
        save_list(top3k, './data/vocab_top3k.txt')
        homer_embeddings = load_glove_embeddings('./data/HomerVe.csv')


    # Example Query
    query = homer_embeddings['agamemnon']
    print(find_nearest_neighbors(homer_embeddings, query, 10))
    # Output: ('menelaus', 'clytemnestra', 'orestes', 'odysseus', 'priam', 'iphigenia', 'theseus', 'atreus', 'creon', 'aegisthus')

    query = homer_embeddings['thebes']
    print(find_nearest_neighbors(homer_embeddings, query, 10))
    # Output: ('corinth', 'assyria', 'theban', 'mycenae', 'orchomenus', 'creon', 'athenians', 'sparta', 'thebans', 'romans')

    query = homer_embeddings['queen'] - homer_embeddings['king'] + homer_embeddings['man']
    print(find_nearest_neighbor(homer_embeddings, query))
    # Output: woman
    

    # Query (taking user input)
    word = input('\nInput the word you want to query for: ').lower()
    if word in homer_embeddings:
        query = homer_embeddings[word]
        print(find_nearest_neighbors(homer_embeddings, query, 10))
    else:
        print('word is not in corpus') 

    print('\nInput the analogy (e.g. king is to queen as man is to woman) you want to query for')
    triwords = input('input format - (e.g. king queen man) three words in one line with space in between them: ').lower()
    triwords = triwords.split()
    try:
        query = (homer_embeddings[triwords[1]] - homer_embeddings[triwords[0]] + homer_embeddings[triwords[2]])
        print(find_nearest_neighbor(homer_embeddings, query))
    except:
        print('one of the words does not exist in the corpus')

    

if __name__ == "__main__":
    main()