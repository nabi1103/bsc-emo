from string import punctuation
from random import shuffle
import csv
import sys
import os

path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(1, path)

tsv_path = path + '/dataset/po-emo/english.tsv'
split_path = path + '/dataset/split/'


class Reader:
    def __init__(self):
        self.dummy = None

    def get_stanza_w_info(self, stanza):
        text = [" ".join(line[0].translate(str.maketrans(
            punctuation, ' '*len(punctuation))).lower().split()) for line in stanza]
        lines = ''

        for line in text:
            lines = lines + ' </br> ' + line

        lines = lines[7:]

        label_1 = []
        label_2 = []

        label_1 = [line[1].split('---') for line in stanza]
        label_1 = [item.strip().strip('\t').lower()
                   for sublist in label_1 for item in sublist]

        label_2 = [line[2].split('---') for line in stanza if len(line) > 2]
        label_2 = [item.strip().strip('\t').lower()
                   for sublist in label_2 for item in sublist]

        labels = list(set(label_1 + label_2))
        if 'nostalgia' in labels:
            labels.remove('nostalgia')

        labels = [label.replace(' ', '') for label in labels]
        labels_text = ''
        for l in labels:
            labels_text = labels_text + ", " + l

        return [lines, labels_text[2:]]

    def read_from_tsv(self, path):
        data = []
        count = 1

        with open(path, encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            raw = [row for row in reader]
            stanza = []

            for i in range(len(raw)):
                if len(raw[i]) > 1 and raw[i][1] == '':
                    count = count + 1
                if len(raw[i]) > 1 and raw[i][1] != '':
                    stanza.append(raw[i])
                if raw[i] == []:
                    data.append(stanza)
                    stanza = []

        data = [self.get_stanza_w_info(s) for s in data if s != []]
        # shuffle(data)

        return count, data

    def read_from_split(self, path):
        count = 1

        with open(path, encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            data = [row for row in reader]

        return data

    def read_from_txt(self, path):
        count = 1

        with open(path, encoding='utf-8') as txtfile:
            reader = csv.reader(txtfile, delimiter='\t')
            data = [row for row in reader]

        return data

    def assign_label(self, stanza):
        labels = ['suspense', 'awe/sublime', 'sadness', 'annoyance',
                  'uneasiness', 'beauty/joy', 'vitality', 'humor']
        vector = [0, 0, 0, 0, 0, 0, 0, 0]
        lst_label = [l.strip() for l in stanza[1].split(',')]
        for l in lst_label:
            idx = labels.index(l)
            vector[idx] = 1
        return [stanza[0], vector]

    def get_label_count(self, data):
        vector = [0, 0, 0, 0, 0, 0, 0, 0]
        for stanza in data:
            for i in range(8):
                vector[i] = vector[i] + stanza[1][i]
        return vector


"""
r = Reader()
data = r.read_from_split('F:/[Uni]/Thesis/bsc-emo/dataset/split/split_1.tsv')

#count, data = r.read_from_tsv(tsv_path)

data = [r.assign_label(s) for s in data]

print(len(data))
print(r.get_label_count(data))

data_0 = data[:int(len(data)*0.5)]
data_1 = data[int(len(data)*0.5):]

d = [data_0, data_1]

for i in range(len(d)):
    name = split_path + 'split_'+ str(i) +'.tsv'
    with open(name, 'wt', encoding='utf-8', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(d[i])
"""
