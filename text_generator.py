# Group member: Monika Zeng, AJ Aizpurua

import sys, string, random, time
import my_utils as my_utils
import re
import readability
import difflib
import textstat

class NgramModel(object):
    def __init__(self, n):
        self.ngram_size = n

        # dictionary that keeps list of candidate words given history
        self.history = {}

        # keeps track of how many times ngram has appeared in the text before
        self.ngram_counter = {}

    def update(self, text: str) -> None:
        """
        update ngram model
        :param text: input text
        """
        n = self.ngram_size
        ngrams = my_utils.get_ngrams(n, my_utils.split_into_words(text))
        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1.0
            else:
                self.ngram_counter[ngram] = 1.0

            prev_words, target_word = ngram
            if prev_words in self.history:
                self.history[prev_words].append(target_word)
            else:
                self.history[prev_words] = [target_word]

    def get_token_probability(self, history, token):
        """
        calculate probability of a candidate token to be generated given a history
        :return: conditional probability
        """
        try:
            count_of_token = self.ngram_counter[(history, token)]
            count_of_history = float(len(self.history[history]))
            result = count_of_token / count_of_history

        except KeyError:
            result = 0.0
        return result

    def random_token(self, history):
        """
        given a history, "semi-randomly" select the next word to append in a sequence
        :param history:
        :return:
        """
        r = random.random()
        map_to_probs = {}
        token_of_interest = self.history[history]
        for token in token_of_interest:
            map_to_probs[token] = self.get_token_probability(history, token)

        accumulator = 0.0
        for token in sorted(map_to_probs):
            accumulator += map_to_probs[token]
            if accumulator > r:
                return token

    def generate_text(self, token_count: int):
        """
        :param token_count: number of words to be produced
        :return: generated text
        """
        
        n = self.ngram_size
        history_queue = (n - 1) * ['START_TOKEN']
        result = []
        counter = 0
        upper_bound = random.randint(9, 15)
        
        for i in range(token_count):
            obj = self.random_token(tuple(history_queue))
            result.append(obj)
            counter = counter + 1
            
            # if obj == '.' and counter < upper_bound:
            #     history_queue = (n - 1) * ['START_TOKEN']
            #     continue
            
            if obj == '.' and counter < upper_bound:
                result[-1] = ' '
                history_queue = (n - 1) * ['START_TOKEN']
                continue
            
            if n > 1:
                history_queue.pop(0)
                if obj == '.':
                    history_queue = (n - 1) * ['START_TOKEN']
                elif counter >= upper_bound:
                    history_queue = (n - 1) * ['START_TOKEN']
                    # result[-1] = '.'
                    result.append('.')
                    counter = 0
                    upper_bound = random.randint(9, 15)
                    continue
                else:
                    history_queue.append(obj)
        return ' '.join(result)
    
    
    def process_text(self, text: str):
    
        text = re.sub(r'\s+', ' ', text)
        text = text[0].upper() + text[1:]
        text = re.sub('[”“"’‘-]', '', text)
        text = text.replace('_', '')
        text = text.replace(" i ", " I ")
        text = re.sub(r'(?<=[\.,;?!]\s)(\.)', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\s)(\W)(\s)', lambda x: x.group()[1]+x.group()[2], text)
        text = re. sub(r'(?<=\.\s)(\W)','', text)
        text = re.sub(r'\s+', ' ', text)
        processed_text = re.sub(r'(?<=[a-z][\.?!]\s)([a-z])', lambda x: x[1].upper(), text)
        
        if processed_text[-1] != '.' or processed_text[-1] != '!':
            processed_text = re.sub(r'.$','.', processed_text)
        
        return processed_text
    
def preprocessing(text:str):
    text = re.sub(r'\d', '', text)
    processed_text = re.sub(r'[*\[\]\(\)]', '', text)
    return processed_text
    
def train_ngram_model(ngram_size, file_path):
    model = NgramModel(ngram_size)
    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        text = f.read()
        text = preprocessing(text)
        text = text.split('.')
        for sentence in text:
            # add back the fullstop
            sentence += '.'
            model.update(sentence)
    return model
    

def generate(size, path):
    random.seed(0)
    ngram_size = size
    file_path = path
    start = time.time()
    model = train_ngram_model(ngram_size, file_path)
    
    print (f'Language Model creating time: {time.time() - start}')
    start = time.time() 
    print(f'{"="*50}\nGenerated text:')
    text = model.generate_text(500)
    processed_text = model.process_text(text)
    print(processed_text)
    
    print("\n")
    #reading score from 0-100, 100 being the most readable
    resultsOne = textstat.flesch_reading_ease(processed_text)    
    resultsTwo =  textstat.flesch_kincaid_grade(processed_text) #Reading at x grade score
    #results = readability.getmeasures(processed_text, lang='en')
    #print(results['readability grades']['FleschReadingEase'])
    print(f'Readability Score (0-100) = {resultsOne}')
    print(f'Grade Level Score = {resultsTwo}')
    
    print(f'{"="*50}')
