from string import punctuation
from random import shuffle
import csv
import sys
import os
import random

path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(1, path)

tsv_path = path + '/dataset/po-emo/english.tsv'
split_path = path + '/dataset/split/'

from preprocess.reader import Reader

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