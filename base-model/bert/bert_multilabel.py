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

lex_path = os.path.join(root_path, 'dataset/vad-lexicon')
split_path = os.path.join(root_path, 'dataset/split')

result_path = os.path.join(root_path, 'result/base-model/')

from reader import Reader
r = Reader()
split_0 = r.read_from_split(split_path + '/split_0.tsv')
split_1 = r.read_from_split(split_path + '/split_1.tsv')


class BERTMultilabel:
    def __init__(self):
        self.dummy = None
        self.reader = Reader()

    def train_model(self, num_epochs, training_data):
        model = MultiLabelClassificationModel(
            "bert",
            "bert-base-multilingual-cased",
            num_labels=8,
            args={"reprocess_input_data": True, "overwrite_output_dir": True,
                  "num_train_epochs": num_epochs, 'fp16': False},
        )

        temp = [self.reader.assign_label(s) for s in training_data]

        model.train_model(pd.DataFrame(temp))

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

        with open(result_path + 'bert_multilable_' + str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_' + '10.tsv', 'wt', encoding='utf-8', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['Name', 'Score'])
            tsv_writer.writerow(['f1_macro', str(f1_macro)])
            tsv_writer.writerow(['f1_all', str(f1_all)])
        return
