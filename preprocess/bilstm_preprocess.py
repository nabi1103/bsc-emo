from string import punctuation
from random import shuffle

import csv
import sys
import os

import re
import xml.etree.ElementTree as ET

path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(1, path)

tsv_path = path + '/dataset/po-emo/english.tsv'
tsv_path_de = path + '/dataset/po-emo/emotion.german.tsv'
split_path = path + '/dataset/split/'

path_0 = split_path + 'split_0.tsv'
path_1 = split_path + 'split_1.tsv'

from reader import Reader

class BiReader():
    def __init__(self):
        self.reader = Reader()

        c, t = self.reader.read_from_tsv(tsv_path)
        self.en_data = t

        c, t = self.reader.read_from_tsv_de(tsv_path_de)
        self.de_data = t

    def get_stanza_w_info_bilstm(self, stanza):
        result = []
        for l in stanza:
            text = " ".join(l[0].translate(str.maketrans(punctuation, ' '*len(punctuation))).split())
            text = text.replace(' ', '_')

            label_1 = l[1].split('---')
            label_1 = [item.strip().strip('\t').replace(' ', '') for item in label_1]
            if len(l) > 2:
                label_2 = l[2].split('---') 
                label_2 = [item.strip().strip('\t').replace(' ', '') for item in label_2]

            labels = list(set(label_1 + label_2))
            if 'Nostalgia' in labels:
                labels.remove('Nostalgia')
            label = labels[0]
            line = [text, label]
            result.append(line)
        return result

    def get_data_bilstm(self):
        en_bilstm = []
        de_bilstm = []

        with open(tsv_path, encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            raw = [row for row in reader]
            stanza = []

            count = 0

            for i in range(len(raw)):
                if len(raw[i]) > 1 and raw[i][1] == '':
                    count = count + 1
                if len(raw[i]) > 1 and raw[i][1] != '':
                    stanza.append(raw[i])
                if raw[i] == []:
                    en_bilstm.append(stanza)
                    stanza = []

        en_bilstm = [self.get_stanza_w_info_bilstm(s) for s in en_bilstm if s!=[]]

        with open(tsv_path_de, encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            raw = [row for row in reader]
            stanza = []

            count = 0

            for i in range(len(raw)):
                if len(raw[i]) > 1 and raw[i][1] == '':
                    count = count + 1
                if len(raw[i]) > 1 and raw[i][1] != '':
                    stanza.append(raw[i])
                if raw[i] == []:
                    de_bilstm.append(stanza)
                    stanza = []

        de_bilstm = [self.get_stanza_w_info_bilstm(s) for s in de_bilstm if s!=[]]

        return en_bilstm, de_bilstm

    def split_match(self):
        
        en_bilstm, de_bilstm = self.get_data_bilstm()

        split_data_0 = [self.reader.assign_label(s) for s in self.reader.read_from_split(path_0)]
        split_data_1 = [self.reader.assign_label(s) for s in self.reader.read_from_split(path_1)]

        temp = [self.reader.assign_label(s) for s in self.en_data]

        split_0 = []
        split_1 = []

        for s in split_data_0:
            split_0.append(temp.index(s))

        for s in split_data_1:
            split_1.append(temp.index(s))
            
        split_0 = [en_bilstm[i] for i in split_0]
        split_1 = [en_bilstm[i] for i in split_1]

        return split_0, split_1, de_bilstm

    def save_training_data(self, identifier):
        split_0, split_1, de_bilstm = self.split_match()

        with open('de_bilstm.txt', 'w', encoding = 'utf-8') as out_file:
            for stanza in de_bilstm:
                for l in stanza:
                    if len(l) > 1:
                        out_file.write(l[0] + '\t' + l[1] + '\n')
                    else: 
                        out_file.write('\n')
                out_file.write('\n')
        
        if identifier == '01':
            train_data = split_0 + de_bilstm
            shuffle(train_data)

            train = train_data[:int(0.8*len(train_data))]
            dev = train_data[int(0.8*len(train_data)):]
            test = split_1

            with open('train.txt', 'w', encoding = 'utf-8') as out_file:
                for stanza in train:
                    for l in stanza:
                        if len(l) > 1:
                            out_file.write(l[0] + '\t' + l[1] + '\n')
                        else: 
                            out_file.write('\n')
                    out_file.write('\n')

            with open('dev.txt', 'w', encoding = 'utf-8') as out_file:
                for stanza in dev:
                    for l in stanza:
                        if len(l) > 1:
                            out_file.write(l[0] + '\t' + l[1] + '\n')
                        else: 
                            out_file.write('\n')
                    out_file.write('\n')

            with open('test.txt', 'w', encoding = 'utf-8') as out_file:
                for stanza in test:
                    for l in stanza:
                        if len(l) > 1:
                            out_file.write(l[0] + '\t' + l[1] + '\n')
                        else: 
                            out_file.write('\n')
                    out_file.write('\n') 

        if identifier == '10':
            train_data = split_1 + de_bilstm
            shuffle(train_data)

            train = train_data[:int(0.8*len(train_data))]
            dev = train_data[int(0.8*len(train_data)):]
            test = split_0

            with open('train.txt', 'w', encoding = 'utf-8') as out_file:
                for stanza in train:
                    for l in stanza:
                        if len(l) > 1:
                            out_file.write(l[0] + '\t' + l[1] + '\n')
                        else: 
                            out_file.write('\n')
                    out_file.write('\n')

            with open('dev.txt', 'w', encoding = 'utf-8') as out_file:
                for stanza in dev:
                    for l in stanza:
                        if len(l) > 1:
                            out_file.write(l[0] + '\t' + l[1] + '\n')
                        else: 
                            out_file.write('\n')
                    out_file.write('\n')

            with open('test.txt', 'w', encoding = 'utf-8') as out_file:
                for stanza in test:
                    for l in stanza:
                        if len(l) > 1:
                            out_file.write(l[0] + '\t' + l[1] + '\n')
                        else: 
                            out_file.write('\n')
                    out_file.write('\n')                                             

# bi = BiReader()
# bi.save_training_data('10')
