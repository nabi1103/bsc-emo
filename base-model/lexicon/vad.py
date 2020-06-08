import csv
import datetime
import numpy as np
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from reader import Reader
import sys
import os

root_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

lex_path = os.path.join(root_path, 'dataset/vad-lexicon')
split_path = os.path.join(root_path, 'dataset/split')

result_path = os.path.join(root_path, 'result/base-model/')

r = Reader()

split_0 = r.read_from_split(split_path + '/split_0.tsv')
split_1 = r.read_from_split(split_path + '/split_1.tsv')
lex = r.read_from_txt(lex_path + '/NRC-VAD-Lexicon.txt')[1:]

fast_dict = [e[0] for e in lex]


class VAD:
    def __init__(self):
        self.dummy = None

    def get_value(self, stanza):
        text = " ".join([line.strip() for line in stanza[0].split('</br>')])
        [v, a, d] = [0, 0, 0]
        count = 0
        for word in text.split(' '):
            if word in fast_dict:
                count += 1
                idx = fast_dict.index(word)
                v = v + float(lex[idx][1])
                a = a + float(lex[idx][2])
                d = d + float(lex[idx][3])

        return ([v/count, a/count, d/count], r.assign_label(stanza)[1])

    def train_model(self, train):
        data = [self.get_value(s) for s in train]

        X_train = np.array([d[0] for d in data])
        y_train = np.array([d[1] for d in data])

        model = BinaryRelevance(classifier=SVC(), require_dense=[False, True])
        model.fit(X_train, y_train)

        return model

    def test_model(self, model, test_data):
        if model == None:
            print('No model found')
            return

        test_data = [vad.get_value(s) for s in test_data]
        X_test = np.array([s[0] for s in test_data])
        y_test = np.array([s[1] for s in test_data])

        y_pred = model.predict(X_test)

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_all = f1_score(y_test, y_pred, average=None)

        print(f1_macro)
        print(f1_all)

        with open(result_path + 'vad_' + str(datetime.datetime.now().strftime("%Y-%m-%d ")) + '_10.tsv', 'wt', encoding='utf-8', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['Name', 'Score'])
            tsv_writer.writerow(['f1_macro', str(f1_macro)])
            tsv_writer.writerow(['f1_all', str(f1_all)])
        return


# Testing code
vad = VAD()

model = vad.train_model(split_1)
vad.test_model(model, split_0)
