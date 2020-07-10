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

    def read_gutenberg_poem(self, path):
        try:
            root = ET.parse(path).getroot()
        except:
            print(path)
            return []
        poem = []

        for lg in root.iter('{http://www.tei-c.org/ns/1.0}lg'):
            # Look for stanza
            if (lg.get('type') == 'stanza'):
                stanza = []
                prev_stanza = []
                sroot = lg
                # Read every line
                for l in sroot.iter('{http://www.tei-c.org/ns/1.0}l'):
                    # Ignore empty lines and lines with only special characters or numbers
                    if (l.text != "" and re.search('[a-zA-Z]', str(l.text))):
                        line = str(l.text)
                        # Strip off leading or trailing spaces
                        line = line.strip()
                        # Remove redundant spaces in between
                        line = line.replace(" ,", ",")
                        line = line.replace(" ?", "?")
                        line = line.replace(" .", ".")
                        line = line.replace(" !", "!")
                        line = line.replace(" :", ":")
                        line = line.replace(" ;", ";")
                        line = line.replace(" -", "-")
                        line = line.replace(" '", "'")
                        line = line.replace("' ", "'")
                        line = line.replace("\u2018 ", "\u2018").replace("\x91 ", "\x91")
                        line = line.replace(" \u2019", "\u2019").replace(" \x92", "\x92")
                        line = line.replace("\u201C ", "\u201C").replace("\x93 ", "\x93")
                        line = line.replace(" \u201D", "\u201D").replace(" \x94", "\x94")

                        line = line.replace(" [= e ]", "e")
                        line = line.replace(" [= o ]", "o")
                        line = line.replace(" [= a ]", "a")
                        line = line.replace(" [= u ]", "u")
                        line = line.replace(" [= i ]", "i")
                        line = line.replace("=", "")
                        line = line.replace("\u2019 d", "\u2019d")
                        line = line.replace("\x92 d", "\x92d")
                        line = re.sub("\{ [0-9]+ \}", "", line)
                        line = re.sub("\[ [0-9]+ \]", "", line)
                        line = re.sub("\( [0-9]+ \)", "", line)
                        line = re.sub("\{ [a-zA-Z] \}", "", line)
                        line = re.sub("\[ [a-zA-Z] \]", "", line)
                        line = re.sub("\( [a-zA-Z] \)", "", line)
                        line = re.sub("\{ [a-zA-Z][a-zA-Z] \}", "", line)
                        line = re.sub("\[ [a-zA-Z][a-zA-Z] \]", "", line)
                        line = re.sub("\( [a-zA-Z][a-zA-Z] \)", "", line)
                        line = line.replace(" n't", "n't")
                        line = line.replace(" n\x92t", "n\x92t")
                        line = line.replace(" n\u2019t", "n\u2019t")
                        line = line.replace(" 's ", "'s ")
                        line = line.replace(" 'm ", "'m ")
                        line = line.replace(" 've ", "'ve ")
                        line = line.replace(" \u2018s ", "\u2018s ")
                        line = line.replace(" \u2018m ", "\u2018m ")
                        line = line.replace(" \u2018ve ", "\u2018ve ")
                        line = line.replace(" \x91s ", "\x91s ")
                        line = line.replace(" \x91m ", "\x91m ")
                        line = line.replace(" \x91ve ", "\x91ve ")
                        line = line.replace("( ", "").replace(" )", "")
                        line = line.replace("{ ", "").replace(" }", "")
                        line = line.replace("[ ", "").replace(" ]", "")
                        line = line.replace("(", "").replace(")", "")
                        line = line.replace("{", "").replace("}", "")
                        line = line.replace("[", "").replace("]", "")
                        line = re.sub(" +", " ", line)
                        line = line.strip()
                        if len(line) > 1:
                            c = line[-1]
                            while not (
                                    c.isalpha() or c == "?" or c == "!" or c == "\u2019" or c == "\x92" or c == "\u201D" or c == "\x94"):
                                line = line[:-1]
                                c = line[-1]
                            line = line.strip()
                        else:
                            continue

                        stanza.append(line.lower())

                if (len(stanza) > 3 and len(stanza) < 11):
                    poem.append(stanza)
                    stanza = []
        return poem

    def read_from_emmood(self, path):
        with open(path, encoding='utf-8') as txtfile:
            reader = csv.reader(txtfile, delimiter='\t')
            data = [row for row in reader]

        return data

    def assign_label_tales(self, line):
        labels = ['A', 'D', 'F', 'H',
                  'N', 'Sa', 'Su+', 'Su-']
        vector = [0, 0, 0, 0, 0, 0, 0, 0]
        lst_label = [l.strip() for l in line[1].split(':')]
        for l in lst_label:
            idx = labels.index(l)
            vector[idx] = 1
        return [line[0], vector]

    def get_all_in_dir(self, path, filetype):
        result = []
        for filename in os.listdir(path):
            if filename.endswith(filetype):
                r = os.path.join(path, filename)
                result.append(r)
        return result

# data = []
# reader = Reader()

# folders = ['F:/[Uni]/Thesis/[Misc]/Code/tales-emotion/Grimms/emmood',
#             'F:/[Uni]/Thesis/[Misc]/Code/tales-emotion/HCAndersen/emmood',
#             'F:/[Uni]/Thesis/[Misc]/Code/tales-emotion/Potter/emmood']

# for folder in folders:
#     r = Reader()
#     files = r.get_all_in_dir(folder, '.emmood')

#     for f in files:
#         for s in r.read_from_emmood(f):
#             data.append([s[3].lower() + ' </br>', s[1]])

# with open('F:/[Uni]/Thesis/[Misc]/Code/tales-emotion/preprocessed-tales.txt', 'w', encoding='utf-8', newline='') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     tsv_writer.writerows(data)
