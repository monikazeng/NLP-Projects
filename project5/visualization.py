# Visualizing GloVe Word Embeddings
#
# Uses t-SNE from sklearn.manifold followed by adjustText (https://github.com/Phlya/adjustText).
#
# For the similarity metric for t-SNE, I used cosine similarity since it tends to produce 
# better visualizations than Euclidean.
#
# The code below runs t-SNE on the GloVe (homer specific model) embeddings, 
# plots the homer corpus's top 3K words, then uses 
# adjustText to spread out the text labels so they are more readable.
# 
# The files containing the embeddings (HomerVe.csv) and vocabulary 
# (vocab_top3k.txt) are provided in the data directory
#
# The output generated visualization files are in pdf format in the output directory: 
#  visualization-glove-homer-plot3k-adj.pdf and visualization-glove-homer-plot3k-noadj.pdf
#
# Reference: 
#   https://home.ttic.edu/~kgimpel/wordembviz/wordembviz.html
#   https://github.com/Phlya/adjustText


import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from adjustText import adjust_text


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


def loadQueryWordsAsSet(filename):
    print("Loading query words (as a set) from file", filename)
    f = open(filename,'r')
    queryWords = set()
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        queryWords.add(word)
    print("Done. ",len(queryWords)," query words loaded!")
    return queryWords    



def main():

    random.seed()
    # We'll be generating big figures which helps in distinguishing nearby words.
    plt.rcParams['figure.figsize'] = [100, 60]

    # We'll use the top 25K most common words in the GloVe embeddings.
    gembs = load_glove_embeddings('./data/HomerVe.csv')

    # Get the words in the order in which they are specified in the embeddings dictionary.
    words = np.array(list(gembs.keys()))
    # Create the matrix of vectors for running t-SNE by including row vectors for each embedding.
    gX = np.array([gembs[word] for word in words])
    # shape should be (25000, 300)
    gX.shape

    # Run t-SNE on the embeddings using cosine similarity as the similarity metric and with at most 2000 iterations
    mytsne = TSNE(n_components=2,early_exaggeration=12,verbose=2,metric='cosine',init='pca',n_iter=2000)
    gX_tsne = mytsne.fit_transform(gX)

    # Load the words to plot. We only use 3000 so that we can more easily distinguish the words 
    # visually, but note that we used many more embeddings when running t-SNE above, which helps 
    # us learn a better projection. 
    wordsToPlot = loadQueryWordsAsSet("./data/vocab_top3k.txt")

    fig = plt.figure()
    alltexts = list()
    # Go through all positions and words in words array.
    for i, word in enumerate(words):
        # Only plot if the current word is a word we want to plot.
        if (word in wordsToPlot):
            # Place an invisible point.
            plt.scatter(gX_tsne[i,0], gX_tsne[i,1], s=0)
            # Create a text element at that point.
            currtext = plt.text(gX_tsne[i,0], gX_tsne[i,1], word, family='sans-serif')
            # Store the text element.
            alltexts.append(currtext)
    
    # Save a pdf of the visualization before we run adjustText.
    plt.savefig('./output/visualization-glove-homer-plot3k-noadj.pdf', format='pdf')
    # Run adjust_text on the text elements (note: this may take a very long time).
    print('now running adjust_text...')
    # Note: using autoalign=True tends to give better results in my experience, but takes much longer.
    #numiters = adjust_text(alltexts, autoalign=True, lim=200)
    #numiters = adjust_text(alltexts, autoalign=True, lim=20, save_steps=True, add_step_numbers=False, save_prefix='wordembviz-glove-tsne25k-plot3k-autoalign-step', save_format='pdf')
    #numiters = adjust_text(alltexts, autoalign=False, lim=20, save_steps=True, add_step_numbers=False, save_prefix='wordembviz-glove-tsne25k-plot3k-step', save_format='pdf')
    numiters = adjust_text(alltexts, autoalign=False, lim=200)
    print('done adjust_text, num iterations: ', numiters)
    plt.savefig('./output/visualization-glove-homer-plot3k-adj.pdf', format='pdf')
    plt.show


if __name__ == "__main__":
    main()







