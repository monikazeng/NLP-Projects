import nltk, re, pprint

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