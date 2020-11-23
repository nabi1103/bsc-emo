import sys 
import os

import pandas as pd
import datetime
import csv
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from random import shuffle
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = list(stopwords.words('english')) + ['br']

# from simpletransformers.classification import MultiLabelClassificationModel
# from lime.lime_text import LimeTextExplainer

root_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

split_path = os.path.join(root_path, 'dataset/split')

curr_path = os.path.dirname(os.path.abspath(__file__))

from collections import Counter
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

class Predictor:
    def __init__(self, model):
        self.model = model
    def predict_proba(self, texts):
        results = []
        for text in texts:
            preds, raw_outputs = self.model.predict([text])
            results.append(raw_outputs[0])

        ress = [res for res in results]
        results_array = np.array(ress)
        
        return results_array

def format_explainer(exp):
    temp = exp.predict_proba.tolist()
    predicted = []
    explaination = []
    for i in range(len(temp)):
        if temp[i] >= 0.5:
            predicted.append(i)
    for p in predicted:
        explaination.append([p, exp.as_list(label = p)])
    return explaination

def compile_explaination(explaination_list):
    temp = []
    result = []
    for exp in explaination_list:
        for e in exp:
            temp.append(e)
    sub_0, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7 = [], [], [], [], [], [], [], []
    subs = [sub_0, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7]
    for i in range(len(subs)):
        for e in temp:
            if e[0] == i:
                subs[i].append(e[1])
    for s in subs:
        temp = []
        for e in s:
            for t in e:
                temp.append(t)
        temp.sort(key = lambda x: x[1], reverse = True)
        result.append(temp)
    return result

def read_json(f):
    with open(f) as json_file:
        result = []
        data = json.load(json_file)
        for d in data:
            d.sort(key = lambda x: x[1], reverse = True)
            result.append(d)
    return result

def compile_feature(prediction_folder):
    sub_0, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7 = [], [], [], [], [], [], [], []
    subs = [sub_0, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7]
    result = []

    for pred in prediction_folder:
        for i in range(len(subs)):
            for e in pred[i]:
                subs[i].append(e)
    
    for sub in subs:
        sub_dict = {}
        for s in sub:
            # print(s)
            if s[0] != 'br' and s[0] not in stopwords and s[1] > 0:
                if s[0] not in sub_dict:
                    sub_dict[s[0]] = 1
                else:
                    sub_dict[s[0]] += 1
        sub_dict = {k: v for k, v in sorted(sub_dict.items(), key=lambda item: item[1], reverse = True)}
        result.append(sub_dict)
    return result

def add_value_labels(ax, spacing):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = spacing
        va = 'bottom'
        if y_value < 0:
            space *= -1
            va = 'top'

        label = "{:.0f}".format(y_value)

        ax.annotate(
            label,                     
            (x_value, y_value),         
            xytext=(0, space),         
            textcoords="offset points", 
            ha='center',                
            va=va)

def visualize_feature(feature_dict, label):
    keys = []
    values = []
    name = 'Most common features for '+ label + ' label'

    for key, value in feature_dict.items():
        keys.append(key)
        values.append(value)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(keys[:5], values[:5])
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Features')

    # ax.set_title(name, y =-0.15)
    add_value_labels(ax, spacing = 1)
    if '/' in name:
        name = name.replace('/', '_')
    save_path = curr_path + '/plot/' + name + '.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return

def merge_dict(d1: dict, d2: dict):
    return Counter(d1) + Counter(d2)

class_names = ['suspense', 'awe/sublime', 'sadness', 'annoyance', 'uneasiness', 'beauty/joy', 'vitality', 'humor']

# model_path =  # Add model path here for LIME explanation

# model = MultiLabelClassificationModel(
#     'bert',
#     model_path,
#     num_labels = 8,
#     use_cuda = True,
#     args = {
#         'silent' : True,
#     }
# )

# p = Predictor(model)

# explainations = []
# explainer = LimeTextExplainer(class_names=class_names)
# count = 0

# for text in test_text_0:
#     exp = explainer.explain_instance(text, p.predict_proba, num_features = 5, labels = (0,1,2,3,4,5,6,7), num_samples = 50)
#     explainations.append(format_explainer(exp))
#     count = count + 1
# print(count)

# exps = compile_explaination(explainations)

# save_name = ' ' # Save name for output .json file

# with open(save_name + '_lime.json', 'w') as fp:
#     json.dump(exps, fp)


# json_path = curr_path + '/json'
# folder = r.get_all_in_dir(json_path, '.json')

# pred_01 = [f for f in folder if '01' in f]
# pred_01.sort()
# pred_01 = [read_json(f) for f in pred_01]

# pred_10 = [f for f in folder if '10' in f]
# pred_10.sort()
# pred_10 = [read_json(f) for f in pred_10]

# top_feature_01 = compile_feature(pred_01)
# top_feature_10 = compile_feature(pred_10)

# features = []

# for i in range(len(top_feature_01)):
#     temp = merge_dict(top_feature_01[i], top_feature_10[i])
#     temp = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1], reverse = True)}
#     features.append(temp)

# for i in range(len(features)):
#     visualize_feature(features[i], class_names[i])