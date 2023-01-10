# This is the program for Word2Vec Embeddings


import nltk
import os, string, time, random
import gensim
from nltk import word_tokenize
from nltk.corpus import stopwords
from my_util import preprocessing, remove_stopwords, unigram_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from gensim import utils
from gensim.test.utils import datapath
nltk.download('punkt')
nltk.download('stopwords')
# import tempfile


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        corpus_path = datapath('/deac/csc/classes/csc391/aizpa316/mini_project_5/data/HOMER.txt')
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)


#Create model function if mdoel has not been created yet or cannot be loaded 
def createModel():
    sentences = MyCorpus()
    model = gensim.models.Word2Vec(sentences=sentences)
    total_examples=model.corpus_count
    # print(total_examples)

    #retrains neural nettwork on our corpus 
    model.train(sentences, total_words=None, word_count=0, total_examples=model.corpus_count, queue_factor=2, report_delay=1.0, epochs=model.epochs)

    #saves model 
    model.save("./data/Homer2vec")   


def load_word2vec_embeddings(filename):
    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    return model

def find_nearest_neighbor(model, word_vector):
    neighbor = []
    neighbor = model.wv.similar_by_vector(word_vector, topn=2)
    neighbor = [val[0] for val in neighbor[1:]]
    
    return neighbor[0]

def find_nearest_neighbors(model, word_vector, k):
    neighbors = []
    neighbors = model.wv.similar_by_vector(word_vector, topn=k)
    neighbors = [val[0] for val in neighbors[1:]]
    return neighbors


def save_list(words_list, filename):
    with open(filename, 'w') as f:
        for word in words_list:
            f.write(f"{word}\n")


#Iterates through model and finds the most related words given a similarity threshhold 
# for words in model.wv.index_to_key:      
def topSim_Corp1(model,vec_val):
    words = []
    temp1 = (int)(len(model.wv.index_to_key)/4)    #splits corpus so it doesn't take a million years to run 

    for k in range(0,temp1):
        for j in range(k + 1, len(model.wv.index_to_key)):
            word1 = model.wv.index_to_key[k]
            word2 = model.wv.index_to_key[j]
            vec = model.wv.similarity(word1,word2)
            if(vec>=vec_val):
                tup = (word1,word2,vec)
                words.append(tup)
    return words

#Iterates through model and finds the most related words given a similarity threshhold 
# for words in model.wv.index_to_key:      

def topSim_Corp2(model,vec_val):
    words = []
    temp1 = (int)(len(model.wv.index_to_key)/4)    #splits corpus so it doesn't take a million years to run 

    for k in range(temp1,temp1*2):
        for j in range(k + 1, len(model.wv.index_to_key)):
            word1 = model.wv.index_to_key[k]
            word2 = model.wv.index_to_key[j]
            vec = model.wv.similarity(word1,word2)
            if(vec>=vec_val):
                tup = (word1,word2,vec)
                words.append(tup)
    return words


def topSim_Corp3(model,vec_val):
    words = []
    temp1 = (int)(len(model.wv.index_to_key)/4)    #splits corpus so it doesn't take a million years to run 

    for k in range(temp1*2,temp1*3):
        for j in range(k + 1, len(model.wv.index_to_key)):
            word1 = model.wv.index_to_key[k]
            word2 = model.wv.index_to_key[j]
            vec = model.wv.similarity(word1,word2)
            if(vec>=vec_val):
                tup = (word1,word2,vec)
                words.append(tup)
    return words


def topSim_Corp4(model,vec_val):
    words = []
    temp1 = (int)(len(model.wv.index_to_key)/4)    #splits corpus so it doesn't take a million years to run 

    for k in range(temp1*3,temp1*4):
        for j in range(k + 1, len(model.wv.index_to_key)):
            word1 = model.wv.index_to_key[k]
            word2 = model.wv.index_to_key[j]
            vec = model.wv.similarity(word1,word2)
            if(vec>=vec_val):
                tup = (word1,word2,vec)
                words.append(tup)
    return words


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 150)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.savefig('graph.png')



def main():

    try:
        model = gensim.models.Word2Vec.load("./data/Homer2vec")   #loads in model 
    except: 
        createModel()
        model = gensim.models.Word2Vec.load("./data/Homer2vec")


    # Example Query
    query = model.wv['agamemnon']
    print(find_nearest_neighbors(model, query, 10))
    # Output: ['idomeneus', 'aias', 'atreides', 'aeneas', 'menelaus', 'glaucus', 'ajax', 'diomed', 'alcinous']

    query = model.wv['thebes']
    print(find_nearest_neighbors(model, query, 10))
    # Output: ['pylos', 'thebe', 'crete', 'gated', 'palace', 'court', 'lemnos', 'seasons', 'elis']

    query = model.wv['queen'] - model.wv['king'] + model.wv['man']
    print(find_nearest_neighbor(model, query))
    # Output: woman
    

    # Query (taking user input)
    word = input('\nInput the word you want to query for: ').lower()
    if word in model.wv:
        query = model.wv[word]
        print(find_nearest_neighbors(model, query, 10))
    else:
        print('word is not in corpus') 

    print('\nInput the analogy (e.g. king is to queen as man is to woman) you want to query for')
    triwords = input('input format - (e.g. king queen man) three words in one line with space in between them: ').lower()
    triwords = triwords.split()
    try:
        query = (model.wv[triwords[1]] - model.wv[triwords[0]] + model.wv[triwords[2]])
        print(find_nearest_neighbor(model, query))
    except:
        print('one of the words does not exist in the corpus')


    # saves top word pairs that are above the cosine similarity threshold 
    # threshold being 0.90 
    # the output is saved to ./output/top_cos_list.txt

    words_list = topSim_Corp1(model,0.90)
    words_list.sort(key = lambda x: x[2])
    words_list.reverse() 
    # print(word_list[0:5],"\n")

    words_list2 = topSim_Corp2(model,0.90)  
    words_list2.sort(key = lambda x: x[2])
    words_list2.reverse() 
    # print(words_list2[0:5],"\n")

    words_list3 = topSim_Corp3(model,0.90)  
    words_list3.sort(key = lambda x: x[2])
    words_list3.reverse() 
    # print(words_list3[0:5],"\n")

    words_list4 = topSim_Corp4(model,0.90)  
    words_list4.sort(key = lambda x: x[2])
    words_list4.reverse() 
    # print(words_list4[0:5],"\n")

    words_list.extend(words_list2)
    words_list.extend(words_list3)
    words_list.extend(words_list4)

    words_list.sort(key = lambda x: x[2])
    words_list.reverse() 

    save_list(words_list,"./output/top_cos_list.txt")

    print(words_list)


    # graph
    # x_vals, y_vals, labels = reduce_dimensions(model)
    # plot_with_matplotlib(x_vals, y_vals, labels)



if __name__ == "__main__":
    main()



#ackknowledgements 

#https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/

#https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#training-your-own-model

#https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-download-auto-examples-tutorials-run-word2vec-py

#https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html?highlight=word2vec



#In the common analogy-solving case, of two positive and 
#one negative examples, this method is equivalent to the “3CosMul” objective 
#(equation (4)) of Levy and Goldberg.

#https://tedboy.github.io/nlps/generated/generated/gensim.models.Doc2Vec.most_similar_cosmul.html


# to run on cluster, make sure to have slurm file and execute following command, ask for above 16gb of memory to prevent your program from halting 
#sbatch /deac/csc/classes/csc391/{USERNAME}/mini_project_5/mini_5.slurm 