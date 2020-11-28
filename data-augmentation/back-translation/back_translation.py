import os
import sys

root_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

bert_path = os.path.join(root_path, 'base-model/bert/')
sys.path.append(bert_path)

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

split_path = os.path.join(root_path, 'dataset/split')

curr_path = os.path.dirname(os.path.abspath(__file__))

from googletrans import Translator
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

class BackTranslator:
    def __init__(self, src, dest):
        self.worker = Translator()
        self.src = src
        self.dest = dest

    def trans_from_split(self, data):
        result = []

        stanzas = [d[0].split(' </br> ') for d in data]
        labels = d[1]

        for i in range(len(stanzas)):
            translation = self.back_trans(stanzas[i])

            # for line in translation:
            #     # print(line)
            #     translation[translation.index(line)] = self.worker.translate(line, src= self.dest, dest=self.src).text

            lines = ''
            for line in translation:
                lines = lines + ' </br> ' + line.lower()
            lines = lines[7:]
            result = [lines, labels]
            print(result)
        return result

    def forward_trans(self, text):
        result = []
        temp = self.worker.translate(text, src = self.src, dest=self.dest)
        for line in temp:
            result.append(line.text)
        return result

    def back_trans(self, text):
        result = []
        f_t = self.forward_trans(text)
        temp = self.worker.translate(f_t, src= self.dest, dest=self.src)
        for line in temp:
            text = line.text
            # print(text)
            result.append(text)
        return result

## Back-translating and save to file

# r = Reader()

# data = r.read_from_split(split_path + '/split_0.tsv')

# text = [d[0] for d in data]
# label = [d[1] for d in data]

# result = []

# for d in data:
#     temp = [d]
#     while True:
#         try:
#             b = BackTranslator('en', 'fr')
#             trans = b.trans_from_split(temp)
#             result.append(trans)
#             break
#         except Exception as e:
#             b = BackTranslator('en', 'fr')

# result = result + data
# save_path = curr_path + '/test.tsv'
# with open(save_path, 'w', encoding='utf-8', newline='') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     tsv_writer.writerows(result)


# Train classification models
# Training args
#args = {"reprocess_input_data": True, 
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

# Read the splits

#r = Reader()
#
#split_0_btl_fix = r.read_from_split(curr_path + '/split_0_btl_fix.tsv')
#split_1_btl_fix = r.read_from_split(curr_path + '/split_1_btl_fix.tsv')
#
#split_0 = r.read_from_split(split_path + '/split_0.tsv')
#split_1 = r.read_from_split(split_path + '/split_1.tsv')
#
#bm = BERTMultilabel()
#
#base_model_path = 'bert-large-uncased'

# Train 01 model
#train = split_0_btl_fix
#test = split_1
#
#model = bm.train_model(train, args, base_model_path, test)
