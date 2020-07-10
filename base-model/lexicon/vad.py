import csv
import datetime
import numpy as np

from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

lex_path = os.path.join(root_path, 'dataset/vad-lexicon')
split_path = os.path.join(root_path, 'dataset/split')

result_path = os.path.join(root_path, 'result/base-model/')

from reader import Reader
from oversample import Oversampler

r = Reader()

split_0 = r.read_from_split(split_path + '/split_0.tsv')
split_1 = r.read_from_split(split_path + '/split_1.tsv')

ovs = Oversampler(split_path + '/split_0.tsv')
split_0_ovs = ovs.oversampling()

ovs = Oversampler(split_path + '/split_1.tsv')
split_1_ovs = ovs.oversampling()

lex = r.read_from_txt(lex_path + '/NRC-VAD-Lexicon.txt')[1:]

fast_dict = [e[0] for e in lex]

class VAD:
    def __init__(self):
        self.dummy = None

    def get_value(self, stanza, ovs):
        text = " ".join([line.strip() for line in stanza[0].split('</br>')])
        [v, a, d, max_v ,max_a, max_d, min_v, min_a, min_d] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        count = 0
        for word in text.split(' '):
            if word in fast_dict:
                idx = fast_dict.index(word)
                value = [float(v) for v in lex[idx][1:]]
                v = v + value[0]
                a = a + value[1]
                d = d + value[2]

                if v*a*d == 0:
                    continue

                if value[0] > max_v or max_v == 0:
                    max_v = value[0]

                if value[1] > max_a or max_a == 0:
                    max_a = value[1]

                if value[2] > max_d or max_d == 0:
                    max_d = value[2]
                
                if value[0] < min_v or min_v == 0:
                    min_v = value[0]

                if value[1] < min_a or min_a == 0:
                    min_a = value[1]
                    
                if value[2] < min_d or min_d == 0:
                    min_d = value[2]

                count += 1
        
        if ovs:
            return ([v/count, a/count, d/count, max_v ,max_a, max_d, min_v, min_a, min_d], stanza[1])

        return ([v/count, a/count, d/count, max_v ,max_a, max_d, min_v, min_a, min_d], r.assign_label(stanza)[1])

    def train_model(self, train):
        data = [self.get_value(s, True) for s in train]

        X_train = np.array([d[0] for d in data])
        y_train = np.array([d[1] for d in data])

        model = BinaryRelevance(classifier=SVC(probability=True, class_weight='balanced', break_ties=True), require_dense=[False, True])
        # model = BinaryRelevance(classifier=LogisticRegression(class_weight='balanced', solver='lbfgs'), require_dense=[False, True])

        model.fit(X_train, y_train)

        return model

    def test_model(self, model, test_data, save_result):
        if model == None:
            print('No model found')
            return

        test_data = [vad.get_value(s, False) for s in test_data]
        X_test = np.array([s[0] for s in test_data])
        y_test = np.array([s[1] for s in test_data])

        y_pred = model.predict(X_test)

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_all = f1_score(y_test, y_pred, average=None)

        print(f1_macro)
        print(f1_all)

        if save_result == True:
            with open(result_path + 'vad_ovs_' + str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_' + '10.tsv', 'wt', encoding='utf-8', newline='') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(['Name', 'Score'])
                tsv_writer.writerow(['f1_macro', str(f1_macro)])
                tsv_writer.writerow(['f1_all', str(f1_all)])
        return


# Testing code
vad = VAD()

model = vad.train_model(split_1_ovs)
vad.test_model(model, split_0, True)
