from sklearn.metrics import f1_score
from simpletransformers.classification import MultiLabelClassificationModel

import sys
import os

import pandas as pd
import datetime
import csv
import numpy as np

root_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

split_path = os.path.join(root_path, 'dataset/split')

result_path = os.path.join(root_path, 'result/base-model/')

curr_path = os.path.dirname(os.path.abspath(__file__))

from reader import Reader

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

class BERTMultilabel:
    def __init__(self):
        self.dummy = None
        self.reader = Reader()

    def train_model(self, training_data, training_args, base_model_path, test_data):
        model = MultiLabelClassificationModel(
            "bert",
            base_model_path,
            num_labels=8,
            use_cuda = False, # Highly recommended to set use_cuda = True to ultilize GPU (if available) for training
            args = training_args,
        )

        temp = [self.reader.assign_label(s) for s in training_data]

        test_data = [self.reader.assign_label(s) for s in test_data]
        test_text = [s[0] for s in test_data]
        test_label = [s[1] for s in test_data]

        eval_df = pd.DataFrame(test_data, columns=['text', 'labels'])

        model.train_model(pd.DataFrame(temp, columns=['text', 'labels']), eval_df = eval_df, f1_macro = f1_evaluate)

        return model

    def test_model(self, model, test_data):
        if model == None:
            print('No model found')
            return

        test_data = [self.reader.assign_label(s) for s in test_data]
        test_text = [s[0] for s in test_data]
        test_label = [s[1] for s in test_data]

        predictions, dummy = model.predict(test_text)
        pred = pd.DataFrame(predictions)
        pd_label = pd.DataFrame(test_label)

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        f1_macro = f1_score(pd_label, pred, average='macro')
        f1_all = f1_score(pd_label, pred, average=None)

        print(f1_macro)
        print(f1_all)

        return f1_macro, f1_all


# # Training args
# args = {"reprocess_input_data": True,
# "overwrite_output_dir": True, 
# "num_train_epochs": 30, 
# 'fp16': False,
# "use_early_stopping": True,
# 'learning_rate': 4e-5,
# 'evaluate_during_training' : True,
# 'early_stopping_metric' : 'f1_macro',
# 'early_stopping_metric_minimize': False,
# 'save_model_every_epoch' : False, 
# 'train_batch_size' : 8
# }

# # Read the splits

# r = Reader()

# split_0 = r.read_from_split(split_path + '/split_0.tsv')
# split_1 = r.read_from_split(split_path + '/split_1.tsv')

# base_model_path = 'bert-large-uncased'

# bm = BERTMultilabel()

# # Train 01 base model
# train = split_0
# test = split_1

# model = bm.train_model(train, args, base_model_path, test)