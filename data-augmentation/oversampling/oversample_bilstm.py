from string import punctuation
from random import shuffle

import csv
import sys
import os
import random

path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(1, path)
bilstm_data_path = path + '/base-model/bilstm/data'

from preprocess.reader import Reader

class BiLSTMOversampler:
    def __init__(self):
        self.reader = Reader()
        self.data = []

    def init_data(self, path):
        for p in path:
            raw = self.reader.read_from_txt(p)

            temp_0 = []
            stanza = []
            labels = ['suspense', 'awe/sublime', 'sadness', 'annoyance',
                        'uneasiness', 'beauty/joy', 'vitality', 'humor']
            for line in raw:
                if line == []:
                    if stanza != []:
                        temp_0.append(stanza)
                        stanza = []
                if len(line) > 1 and line[1] != '':
                    stanza.append(line)
            if stanza != []:
                temp_0.append(stanza)
            temp_1 = []
            for s in temp_0:
                vector = [0, 0, 0, 0, 0, 0, 0, 0]
                lst_label = [l[1].lower() for l in s]

                for l in lst_label:
                    idx = labels.index(l)
                    vector[idx] = 1
                temp_1.append(s + [vector])
            self.data += temp_1

    def get_label_count(self):
        if self.data == []:
            print('no data')
            return
        vector = [0, 0, 0, 0, 0, 0, 0, 0]
        for stanza in self.data:
            for i in range(8):
                vector[i] = vector[i] + stanza[-1][i]
        return vector

    def get_average_label_count(self):
        if self.data == []:
            print('no data')
            return
        
        temp = self.get_label_count()
        return sum(temp) / len(temp)

    def random_select(self, index, avoid):
        if self.data == []:
            print('no data')
            return
        avoided = []
        for a in avoid:
            temp = [stanza for stanza in self.data if stanza[-1][a] == 1]
            for s in temp:
                if s not in avoided:
                    avoided.append(s)
        cand = [stanza for stanza in self.data if stanza[-1][index] == 1 and stanza not in avoided]
        if len(cand) == 0:
            cand = [stanza for stanza in self.data if stanza[-1][index] == 1]
        return random.choice(cand)

    def oversampling(self):
        if self.data == []:
            print('no data')
            return
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
        self.data = self.data + result

    def get_data(self):
        if self.data == []:
            print('no data')
            return
        
        return self.data

    def get_data_simple_label(self):
        result = []
        for s in self.data:
            result.append(s[0:-1])
        return result

"""
train_path = bilstm_data_path + '/emo_10/train.txt'

save_path = bilstm_data_path + '/emo_10/train_ovs.txt'

biovs = BiLSTMOversampler()

biovs.init_data([train_path])

print(biovs.get_label_count(), biovs.get_average_label_count())

biovs.oversampling()
new_data = biovs.get_data_simple_label()

print(biovs.get_label_count(), biovs.get_average_label_count())

with open(save_path, 'w', encoding = 'utf-8') as out_file:
    for s in new_data:
        for l in s:
            if len(l) > 1:
                out_file.write(l[0] + '\t' + l[1] + '\n')
        out_file.write('\n')
"""