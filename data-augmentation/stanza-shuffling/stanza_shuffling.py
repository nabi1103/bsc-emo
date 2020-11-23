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
import csv
import pandas as pd

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

class StanzaShuffler():
    def __init__(self, split_path):
        self.path = split_path
        self.reader = Reader()
        self.data = [[d[0].split(' </br> '), d[1]] for d in self.reader.read_from_split(split_path)]

    def shuffle_data(self):
        result = []
        for stanza in self.data:
            tmp = stanza[0]
            shuffle(tmp)
            lines = ''
            for line in tmp:
                lines = lines + ' </br> ' + line
            lines = lines[7:]
            result.append([lines, stanza[1]])            
        return result + self.reader.read_from_split(self.path)

    def save_data(self, save_path):
        data = self.shuffle_data()
        shuffle(data)
        with open(save_path, 'w', encoding='utf-8', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerows(data)

## Train classification models
## Training args
# args = {"reprocess_input_data": True, 
#     "overwrite_output_dir": True, 
#     "num_train_epochs": 30, 
#     'fp16': False,
#     "use_early_stopping": True,
#     'learning_rate': 4e-5,
#     'evaluate_during_training' : True,
#     'early_stopping_metric' : 'f1_macro',
#     'early_stopping_metric_minimize': False,
#     'save_model_every_epoch' : False, 
#     'train_batch_size' : 1,
#     'manual_seed' : 1
# }

## Read the splits

# r = Reader()

# split_0_shuffled = r.read_from_split(curr_path + '/split_0_shuffled.tsv')
# split_1_shuffled = r.read_from_split(curr_path + '/split_1_shuffled.tsv')

# split_0 = r.read_from_split(split_path + '/split_0.tsv')
# split_1 = r.read_from_split(split_path + '/split_1.tsv')

# bm = BERTMultilabel()

# base_model_path = 'bert-large-uncased'

## Train 01 model
# train = split_0_btl_fix
# test = split_1

# model = bm.train_model(train, args, base_model_path, test)