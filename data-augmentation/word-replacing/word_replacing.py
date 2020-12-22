import sys
import os

root_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

bert_path = os.path.join(root_path, 'base-model/bert/')
sys.path.append(bert_path)

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

split_path = os.path.join(root_path, 'dataset/split')

curr_path = os.path.dirname(os.path.abspath(__file__))
from reader import Reader
from bert_multilabel import BERTMultilabel

import gensim
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
import csv
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from random import shuffle

def f1_evaluate(true, pred):
    for p in pred:
        for i in range(len(p)):
            if p[i] >= 0.5:
                p[i] = 1
            else:
                p[i] = 0

    score = f1_score(true, pred, average = 'macro')
    label = f1_score(true, pred, average = None)
    print(score)
    print(label)
    
    return score

class WordReplacer():
    def __init__(self, split_path, model):
        self.reader = Reader()
        self.model = model
        self.split_path = split_path
        self.data = [' '.join(d[0].split(' </br> ')).strip() for d in self.reader.read_from_split(split_path)]
    
    def calc_tfidf(self):
        vect = TfidfVectorizer()
        tfidf_matrix = vect.fit_transform(self.data)

        df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
        return df

    def generate_data(self):
        df = self.calc_tfidf()
        stanzas = self.reader.read_from_split(self.split_path)
        result = []

        for j in range(len(stanzas)):
            s = stanzas[j]
            new_stanza = []
            new_data = ''

            lines = s[0].split(' </br> ')

            for l in lines:
                words = l.split(' ')
                values = []
                for w in words:
                    try:
                        values.append((w, df.at[j, w]))
                    except:
                        continue
                values.sort(key = lambda value: value[1])
                for i in range(len(values)):
                    to_be_replaced = values[i][0]
                    try:
                        new_word = self.model.most_similar(positive = [to_be_replaced], topn = 1)[0][0]
                        break
                    except:
                        continue
                try:
                    words[words.index(to_be_replaced)] = new_word
                    new_line = ' '.join(words).strip()
                except:
                    new_line = l
                new_stanza.append(new_line)
            for l in new_stanza:
                new_data = new_data + ' </br> ' + l
            new_data = new_data[7:].lower()
            result.append([new_data, s[1]])
        return result + stanzas
    
    def save_data(self, save_path):
        data = self.generate_data()
        shuffle(data)
        with open(save_path, 'w', encoding='utf-8', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerows(data)

# wr = WordReplacer(split_path + '/split_0.tsv', model)

# save_path = curr_path + '/test.tsv'
# wr.save_data(save_path)

# Train classification models
# Training args
args = {"reprocess_input_data": True, 
    "overwrite_output_dir": True, 
    "num_train_epochs": 30, 
    'fp16': False,
    "use_early_stopping": True,
    'learning_rate': 4e-5,
    'evaluate_during_training' : True,
    'early_stopping_metric' : 'f1_macro',
    'early_stopping_metric_minimize': False,
    'save_model_every_epoch' : False, 
    'train_batch_size' : 1,
    'manual_seed' : 1
}

# Read the splits

r = Reader()

split_0_replaced = r.read_from_split(curr_path + '/split_0_replaced.tsv')
split_1_replaced = r.read_from_split(curr_path + '/split_1_replaced.tsv')

split_0 = r.read_from_split(split_path + '/split_0.tsv')
split_1 = r.read_from_split(split_path + '/split_1.tsv')

bm = BERTMultilabel()

base_model_path = 'bert-large-uncased'

# Train 01 model
train = split_0_replaced
test = split_1

model = bm.train_model(train, args, base_model_path, test)
