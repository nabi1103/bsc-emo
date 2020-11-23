from string import punctuation
from random import shuffle
import sys
import os
import random

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

class Oversampler:
    def __init__(self, path):
        self.reader = Reader()
        self.data = [self.reader.assign_label(s) for s in self.reader.read_from_split(path)]
    
    def get_label_count(self):
        return self.reader.get_label_count(self.data)

    def get_average_label_count(self):
        temp = self.get_label_count()
        return sum(temp) / len(temp)

    def random_select(self, index, avoid):
        avoided = []
        for a in avoid:
            temp = [stanza for stanza in self.data if stanza[1][a] == 1]
            for s in temp:
                if s not in avoided:
                    avoided.append(s)
        cand = [stanza for stanza in self.data if stanza[1][index] == 1 and stanza not in avoided]
        if len(cand) == 0:
            cand = [stanza for stanza in self.data if stanza[1][index] == 1]
        return random.choice(cand)

    def oversampling(self):
        count = self.get_label_count()
        average = self.get_average_label_count()
        avoid = []
        result = []

        for i in range(len(count)):
            if count[i] >= average:
                avoid.append(i)

        for i in range(len(count)):
            if count[i] >= average:
                continue
            while(count[i] < average):
                result.append(self.random_select(i, avoid))
                count[i] += 1

        return self.data + result
    
    def splitify(self):
        labels = ['suspense', 'awe/sublime', 'sadness', 'annoyance',
                  'uneasiness', 'beauty/joy', 'vitality', 'humor']

        data = self.oversampling()

        result = []

        for d in data:
            text_label = ''
            for i in range(len(labels)):
                if d[1][i] == 1:
                    text_label = text_label + labels[i] + ', '
            new_d = [d[0], text_label[:-2]]
            result.append(new_d)
        shuffle(result)
        return result

# path_0 = split_path + '/split_0.tsv'
# path_1 = split_path + '/split_1.tsv'

# ovs = Oversampler(path_0)

# ovs_data = ovs.splitify()
# save_path = curr_path + '/test.tsv'

# with open(save_path, 'w', encoding='utf-8', newline='') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     tsv_writer.writerows(ovs_data)

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

# split_0_ovs = r.read_from_split(curr_path + '/split_0_ovs.tsv')
# split_1_ovs = r.read_from_split(curr_path + '/split_1_ovs.tsv')

# split_0 = r.read_from_split(split_path + '/split_0.tsv')
# split_1 = r.read_from_split(split_path + '/split_1.tsv')

# bm = BERTMultilabel()

# base_model_path = 'bert-large-uncased'

## Train 01 model
# train = split_0_ovs
# test = split_1

# model = bm.train_model(train, args, base_model_path, test)
