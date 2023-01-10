I build the Homer GloVe model based on the big text corpus "HOMER.txt" (under data directory), which incorporated a few English translations of Iliad and Odyssey. 
For the Homer_GloVe embeddings, I extracted 17659 unique words from the 40,000 original model of 300 dimension and as well as a list of 37043 skipped words. 
The Homer_GloVe model is saved in the output directory as "HomerVe.csv" and the skipped words list are outputted to a file called "skipped_words.txt."

For Word2Vec embeddings, I retrained the model base on our input corpus "HOMER.txt" and 131132 embeddings are extracted. 
The Word2Vec model is saved in the output directory as "Homer2vec."


I compared some sample query results between GloVe (300d) and Word2Vec (300d). The query results are:
    for word 'agamemnon':
    the top 10 most similar words by Homer_GloVe are: 'menelaus', 'clytemnestra', 'orestes', 'odysseus', 'priam', 'iphigenia', 'theseus', 'atreus', 'creon', 'aegisthus'
    the top 10 most similar words by Word2Vec are: 'idomeneus', 'aias', 'atreides', 'aeneas', 'menelaus', 'glaucus', 'ajax', 'diomed', 'alcinous'

    for word 'thebes':
    the top 10 most similar words by Homer_GloVe are: 'corinth', 'assyria', 'theban', 'mycenae', 'orchomenus', 'creon', 'athenians', 'sparta', 'thebans', 'romans'
    the top 10 most similar words by Word2Vec are: 'pylos', 'thebe', 'crete', 'gated', 'palace', 'court', 'lemnos', 'seasons', 'elis'

    for analogy: 'king is to queen as man is to woman' I query for king is to queen as man is to __, where I should expect 'woman'
    both of our models outputted 'woman' for the analogy query

    For the words query however, I have gotten very different results as shown above. This is interesting because although the words in context all makes sense, but
    in my opinion, GloVe embeddings did a better job based on my understanding of the book context. 'Agamemnon' is the leader of Greece in the trojan war, and his co-leader
    is his brother Menelaus, and his wife is Clytemnestra. They also had Odysseus and Ajax on their team. These related words are reflected in the GloVe embeddings.
    For the word 'thebes,' the cities Corinth, Mycenae and Sparta are closeby cities to Thebes and they are also reflected more in the GloVe embeddings.


I'm also able to take in query from user input, including singular words and word analogies. 

For the Homer_GloVe embeddings, I generated a visualization for the model in the output directory called "visualization-glove-homer-plot3k-noadj", where the top 3000 words
are plotted in clusterings, where words that are related are closeby to each other. The file is in pdf format and since it's a very large file, I need to zoom in to examine
each cluster. I also attached some screenshots of the zoomed in versions in the output directory called "GloVe Vis Zoomin 1-4" and GloVe Vis 1 is a screenshot of the overall
shape of our embedding models.


For the Word2Vec embeddings, I found the top 1500 most similar word pairs of the corpus. 

I created a function that finds pairings of words with a cosine simliarity above a given threshold that I choose. I ran into a few issues when running the code 
initially for the whole corpus. First, I tried running through all possible comparisons for our entire vocab with the cluster. The cluster ended up freezing as I didn't issue
enough memory to the slurm job. Even with 32gb I couldn't get the job to finish outputting our word pairings. What I ended up doing was splitting our corpus into fourths and running it 
across four functions. This way the code outputted at a managable speed and without any memory hiccups. The output of each function is a list of all the word pairings that passed the 
cosine threshold I defined as an argument to the function. Then I extend all four lists into one big list and save the output to a an output file called top_cos_list.txt. For our tests, I chose a cosine 
threshhold of 0.90. Therefore the final output is all word pairings that were above a 0.90 cosine similarity within our corpus. 

I also printed out a visualization for the Word2vec embeddings. Words closer in similarity are grouped closer together in the graph. 
For instance, meadow and harvest are very close together in the graph, suggesting a strong similiarity to each other. I noticed that 
there is a lot of words clustered at the top/middle of the graph and the data points began to spread out at the bottom of the graph. 



Acknowledgements 

https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/

https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#training-your-own-model

https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-download-auto-examples-tutorials-run-word2vec-py

https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html?highlight=word2vec

https://tedboy.github.io/nlps/generated/generated/gensim.models.Doc2Vec.most_similar_cosmul.html

https://home.ttic.edu/~kgimpel/wordembviz/wordembviz.html

https://github.com/Phlya/adjustText

Text Referances 

Butcher & Lang

https://www.gutenberg.org/files/1728/1728-0.txt

George Chapman

https://www.gutenberg.org/files/48895/48895-0.txt

Butler

https://www.gutenberg.org/cache/epub/1727/pg1727.txt https://www.gutenberg.org/cache/epub/2199/pg2199.txt

Andrew Lang

https://www.gutenberg.org/cache/epub/7972/pg7972.txt https://www.gutenberg.org/cache/epub/3059/pg3059.txt

Iakovos Polylas

https://www.gutenberg.org/cache/epub/55184/pg55184.txt

Alexandros Pallis

https://www.gutenberg.org/files/36248/36248-0.txt

Jack London

https://www.gutenberg.org/cache/epub/55184/pg55184.txt

Alexander Pope

https://www.gutenberg.org/files/3160/3160-0.txt https://www.gutenberg.org/cache/epub/6130/pg6130.txt

William Cowper

https://www.gutenberg.org/files/24269/24269-0.txt https://www.gutenberg.org/cache/epub/16452/pg16452.txt https://www.gutenberg.org/cache/epub/51355/pg51355.txt

William Lucas Collins

https://www.gutenberg.org/files/59306/59306-0.txt https://www.gutenberg.org/files/59306/59306-0.txt

Leconte de Lisle

https://www.gutenberg.org/cache/epub/14285/pg14285.txt

Edward, Earl of Derby

https://www.gutenberg.org/files/6150/6150-0.txt

Luis Segalá y Estalella

https://www.gutenberg.org/files/57654/57654-0.txt



