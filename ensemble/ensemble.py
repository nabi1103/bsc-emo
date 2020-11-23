import sys 
import os

import pandas as pd
import datetime
import csv
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from random import shuffle
from sklearn.metrics import f1_score

import math
import matplotlib.pyplot as plt

root_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

split_path = os.path.join(root_path, 'dataset/split')

curr_path = os.path.dirname(os.path.abspath(__file__))

from reader import Reader

r = Reader()

split_0 = r.read_from_split(split_path + '/split_0.tsv')
split_1 = r.read_from_split(split_path + '/split_1.tsv')

test_data_0 = [r.assign_label(s) for s in split_0]
test_text_0 = [s[0] for s in test_data_0]
test_label_0 = [s[1] for s in test_data_0]

test_data_1 = [r.assign_label(s) for s in split_1]
test_text_1 = [s[0] for s in test_data_1]
test_label_1 = [s[1] for s in test_data_1]

case_0 = [test_text_0, test_label_0]
case_1 = [test_text_1, test_label_1]

predictions_path_01 = curr_path + '/predictions/01'
predictions_path_10 = curr_path + '/predictions/10'

raw_path_01 = curr_path + '/raws/01'
raw_path_10 = curr_path + '/raws/10'

def get_label_count(data):
        vector = [0, 0, 0, 0, 0, 0, 0, 0]
        for stanza in data:
            for i in range(8):
                vector[i] = vector[i] + stanza[i]
        return vector

def get_all_in_dir(path, filetype):
    result = []
    for filename in os.listdir(path):
        if filename.endswith(filetype):
            r = os.path.join(path, filename)
            result.append(r)
    return result

def get_name(path):
    name = ''
    path = path.replace('\\', '/')
    for w in path.split('/')[-1].split('_'):
        try:
            int(w)
            break
        except:
            name = name + '_' + w
    return name[1:]

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def get_top_value(values, limit):
    temp = []
    for i in range(len(values)):
        temp.append(values[i])
    temp.sort(reverse=True)
    top = temp[:limit]
    for i in range(len(values)):
        if values[i] not in top:
            values[i] = 0
    return values

def read_csv(path, mode):
    if mode not in ['int', 'float']:
        print('Wrong mode')
        return
    with open(path) as csv_file:
        result = []
        reader = csv.reader(csv_file, delimiter='\t')
        for row in reader:
            if mode == 'int':
                for i in range(len(row)):
                    row[i] = int(row[i])
                result.append(row)
            if mode == 'float':
                for i in range(len(row)):
                    row[i] = float(row[i])
                result.append(row)
    return [result, get_name(path)]

def pairwise_ensemble(label_0, label_1, mode, test_case):
    if mode not in ['and', 'or', 'avg']:
        print('Wrong mode')
        return
    if len(label_0) != len(label_1):
        print(len(label_0))
        print(len(label_1))
        return
    name_0 = label_0[1]
    name_1 = label_1[1]

    label_0 = label_0[0]
    label_1 = label_1[0]

    if mode == 'and':
        temp = []
        for i in range(len(label_0)):
            t = []
            for j in range(len(label_0[i])):
                t.append(label_0[i][j] & label_1[i][j])
            temp.append(t)
        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2), name_0, name_1

    if mode == 'or':
        temp = []
        for i in range(len(label_0)):
            t = []
            for j in range(len(label_0[i])):
                t.append(label_0[i][j] | label_1[i][j])
            temp.append(t)
        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2), name_0, name_1

    if mode == 'avg':
        temp = []
        for i in range(len(label_0)):
            t = []
            for j in range(len(label_0[i])):
                avg = (label_0[i][j] + label_1[i][j]) / 2
                if avg >= 0.5:
                    t.append(1)
                else:
                    t.append(0)
            temp.append(t)
        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2), name_0, name_1

def triplewise_ensemble(label_0, label_1, label_2, mode, test_case):
    if mode not in ['and', 'or', 'avg']:
        print('Wrong mode')
        return
    if len(label_0) != len(label_1) or len(label_0) != len(label_2):
        print(len(label_0))
        print(len(label_1))
        print(len(label_2))
        return

    name_0 = label_0[1]
    name_1 = label_1[1]
    name_2 = label_2[1]

    label_0 = label_0[0]
    label_1 = label_1[0]
    label_2 = label_2[0]

    if mode == 'and':
        temp = []
        for i in range(len(label_0)):
            t = []
            for j in range(len(label_0[i])):
                t.append(label_0[i][j] & label_1[i][j] & label_2[i][j])
            temp.append(t)
        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2), name_0, name_1, name_2

    if mode == 'or':
        temp = []
        for i in range(len(label_0)):
            t = []
            for j in range(len(label_0[i])):
                t.append(label_0[i][j] | label_1[i][j] | label_2[i][j])
            temp.append(t)
        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2), name_0, name_1, name_2

    if mode == 'avg':
        temp = []
        for i in range(len(label_0)):
            t = []
            for j in range(len(label_0[i])):
                avg = (label_0[i][j] + label_1[i][j] + label_2[i][j]) / 3
                if avg >= 0.5:
                    t.append(1)
                else:
                    t.append(0)
            temp.append(t)
        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2), name_0, name_1, name_2

def get_pairwise_value(prediction_folder, mode, case):

    cell_value = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(prediction_folder)):
        temp = []
        for j in range(len(prediction_folder)):
            f1_macro, name_0, name_1 = pairwise_ensemble(prediction_folder[i], prediction_folder[j], mode, case)
            temp.append(f1_macro)
        cell_value[i] = temp

    return cell_value

def get_triplewise_value(prediction_folder, mode, case):
    triple = []
    result = []
    for a in prediction_folder:
        for b in prediction_folder:
            for c in prediction_folder:
                if a != b and b !=c and a != c:
                    temp = [a, b, c]
                    # temp.sort(reverse = True)
                    if temp not in triple:
                        triple.append(temp)

    for t in triple:
        f1_score, name_0, name_1, name_2 = triplewise_ensemble(t[0], t[1], t[2], mode, case)
        f1_score = round_half_up(f1_score, 2)
        result.append([f1_score, name_0 + ' x ' + name_1 + ' x ' + name_2])
    return result

def all_ensemble(labels, mode, test_case):
    if mode not in ['and', 'or', 'avg', 'majority', '1', '2', '3', '4', '5']:
        print('Wrong mode')
        return

    len_check = len(labels[0])

    for l in labels:
        if len(l) != len_check:
            return

    labels = [l[0] for l in labels]
        
    label_0 = labels[0]
    label_1 = labels[1]
    label_2 = labels[2]
    label_3 = labels[3]
    label_4 = labels[4]
    label_5 = labels[5]
    label_6 = labels[6]
    label_7 = labels[7]
    label_8 = labels[8]
    label_9 = labels[9]

    if mode == 'and':
        temp = []
        for i in range(len(labels[0])):
            t = []
            for j in range(len(label_0[i])):
                t.append(label_0[i][j] & label_1[i][j] & label_2[i][j] & label_3[i][j] & label_4[i][j] & label_5[i][j] & label_6[i][j] & label_7[i][j] & label_8[i][j] & label_9[i][j])
            temp.append(t)

        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2)
    
    if mode == 'or':
        temp = []
        for i in range(len(labels[0])):
            t = []
            for j in range(len(label_0[i])):
                t.append(label_0[i][j] | label_1[i][j] | label_2[i][j] | label_3[i][j] | label_4[i][j] | label_5[i][j] | label_6[i][j] | label_7[i][j] | label_8[i][j] | label_9[i][j])
            temp.append(t)

        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2)

    if mode == 'avg':
        temp = []
        for i in range(len(label_0)):
            t = []
            for j in range(len(label_0[i])):
                avg = (label_0[i][j] + label_1[i][j] + label_2[i][j] + label_3[i][j] + label_4[i][j] + label_5[i][j] + label_6[i][j] + label_7[i][j] + label_8[i][j] + label_9[i][j]) / 10
                if avg >= 0.5:
                    t.append(1)
                else:
                    t.append(0)
            temp.append(t)
        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2)

    if mode == 'majority':
        temp = []
        for i in range(len(label_0)):
            t = []
            for j in range(len(label_0[i])):
                sum = (label_0[i][j] + label_1[i][j] + label_2[i][j] + label_3[i][j] + label_4[i][j] + label_5[i][j] + label_6[i][j] + label_7[i][j] + label_8[i][j] + label_9[i][j])
                if sum >= 5:
                    t.append(1)
                else:
                    t.append(0)
            temp.append(t)
        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2)

    if mode in ['1', '2', '3', '4']:
        limit = int(mode)
        temp = []
        for i in range(len(labels[0])):
            t = []
            for j in range(len(label_0[i])):
                t.append(label_0[i][j] + label_1[i][j] + label_2[i][j] + label_3[i][j] + label_4[i][j] + label_5[i][j] + label_6[i][j] + label_7[i][j] + label_8[i][j] + label_9[i][j])
            t = get_top_value(t, limit)
            for i in range(len(t)):
                if t[i] > 0:
                    t[i] = 1
            temp.append(t)

        f1_macro = f1_score(test_case[1], temp, average = 'macro')
        return round_half_up(f1_macro, 2)

# Get output folders

prediction_folder_01 = [read_csv(f, 'int') for f in get_all_in_dir(predictions_path_01, 'csv')]
prediction_folder_10 = [read_csv(f, 'int') for f in get_all_in_dir(predictions_path_10, 'csv')]

raw_folder_01 = [read_csv(f, 'float') for f in get_all_in_dir(raw_path_01, 'csv')]
raw_folder_10 = [read_csv(f, 'float') for f in get_all_in_dir(raw_path_10, 'csv')]

# # Pairwise ensemble

# labels = [f[1] for f in prediction_folder_01]
# col_labels = [f.replace('_', '\n') for f in labels]
# cell_text_01 = get_pairwise_value(prediction_folder_01, mode = 'or', case = case_1)  
# cell_text_10 = get_pairwise_value(prediction_folder_10, mode = 'or', case = case_0)  

# cell_text = []

# for i in range(len(cell_text_01)):
#     temp_row = []
#     for j in range(len(cell_text_01[i])):
#         avg = round_half_up((cell_text_01[i][j] + cell_text_10[i][j]) / 2, 2)
#         temp_row.append(avg)
#     cell_text.append(temp_row)
    
# fig, ax = plt.subplots() 
# ax.set_axis_off() 
# table = ax.table( 
#     cellText = cell_text,  
#     rowLabels = labels,  
#     colLabels = col_labels,
#     colWidths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
#     rowColours =["palegreen"] * 10,  
#     colColours =["palegreen"] * 10, 
#     cellLoc ='center',  
#     loc ='center')
# table.auto_set_font_size(False)
# table.set_fontsize(25)   
# table.scale(7, 7)

# length = len(cell_text[0])
# sum = 0
# count = 0
# best = -1
# base = 0.37

# for i in range(length):
#     temp = cell_text[i]
#     if i == 0:
#         continue
#     for j in range(i):
#         if best < temp[j]:
#             best = temp[j]
#         sum = sum + (temp[j] - base)
#         count = count + 1

# r = round_half_up(sum/count,2)
# print(r)
# print(best)

# # Ensemble of triple of models

# triple_0 = get_triplewise_value(raw_folder_01, 'avg', case_1)
# triple_0.sort(key = lambda x: x[1])
# triple_1 = get_triplewise_value(raw_folder_10, 'avg', case_0)
# triple_1.sort(key = lambda x: x[1])

# triple_avg = []

# count = 0
# sum = 0
# base = 0.37

# for t_0 in triple_0:
#     for t_1 in triple_1:
#         if t_0[1] != t_1[1]:
#             continue
#         value = round_half_up((t_0[0] + t_1[0]) / 2, 2)
#         iden = t_0[1].split(' x ')
#         iden.sort()
#         result = [value, iden]
#         if result not in triple_avg:
#             triple_avg.append(result)
#             count = count + 1
#             sum = sum + (value - base)

# triple_avg.sort(key = lambda x : x[0], reverse=True)

# r = round_half_up(sum/count,2)
# print(r)

# for r in triple_avg[:10]:
#     result = r[0]
#     line = r[1][0].replace('_', '\\_') + ' & ' + r[1][1].replace('_', '\\_') + ' & ' + r[1][2].replace('_', '\\_')
#     print(str(r[0]) + ' & ' + line + '\\\\')
#     print('\\hline')

# # All ensemble

# mode = ['and', 'or', 'majority', '2', '3', '4']
# base_01 = 0.38
# base_10 = 0.35
# base = 0.37

# for m in mode:
#     score_01 = all_ensemble(prediction_folder_01, mode = m, test_case = case_1)
#     score_10 = all_ensemble(prediction_folder_10, mode = m, test_case = case_0)
#     sum = score_01 + score_10
#     delta_01 = round_half_up(score_01 - base_01, 2)
#     delta_10 = round_half_up(score_10 - base_10, 2)
#     delta_base = round_half_up(round_half_up(sum/2,  2) - base, 2)
#     print(m.upper() + ' & ' + str(score_01) + '(' + str(delta_01) + ')' +' & '+ str(score_10) + '(' + str(delta_10) + ')' + ' & '+  str(round_half_up(sum/2,  2)) + '(' + str(delta_base) + ')' +' \\\\')

# score_01 = all_ensemble(raw_folder_01, mode = 'avg', test_case = case_1)
# score_10 = all_ensemble(raw_folder_10, mode = 'avg', test_case = case_0)
# sum = score_01 + score_10
# print('AVERAGE' + ' & ' + str(score_01) + '(' + str(delta_01) + ')' +' & '+ str(score_10) + '(' + str(delta_10) + ')' + ' & '+  str(round_half_up(sum/2,  2)) + '(' + str(delta_base) + ')' +' \\\\')