import sys 
import os

root_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

bert_path = os.path.join(root_path, 'base-model/bert/')
sys.path.append(bert_path)

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

split_path = os.path.join(root_path, 'dataset/split')
de_path = os.path.join(root_path, 'dataset/po-emo')

curr_path = os.path.dirname(os.path.abspath(__file__))

from bert_multilabel import BERTMultilabel
from reader import Reader
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd

# Train a fairy tales emotion classifier 

def f1_evaluate(true, pred):
    for p in pred:
        for i in range(len(p)):
            if p[i] >= 0.5:
                p[i] = 1
            else:
                p[i] = 0

    score = f1_score(true, pred, average = 'macro')
    label = f1_score(true, pred, average = None)
    print(score)
    print(label)
    
    return score

r = Reader()

de = r.read_from_tsv_de(de_path + '/emotion.german.tsv')[1]

train_de = de[:int(0.8*len(de))]
test_de = de[int(0.8*len(de)):]

args = {"reprocess_input_data": True, 
    "overwrite_output_dir": True, 
    "num_train_epochs": 35, 
    'fp16': False,
    "use_early_stopping": True,
    'learning_rate': 4e-5,
    'evaluate_during_training' : True,
    'early_stopping_metric' : 'f1_macro',
    'early_stopping_metric_minimize': False,
    'save_model_every_epoch' : False, 
    'train_batch_size' : 1,
    'manual_seed' : 1
}

bm = BERTMultilabel()

model = bm.train_model(training_data = train, training_args = args, base_model_path = 'bert-base-multilingual-cased', eval_data = test)

## Train an emotion classifier with the fine-tuned model from above

# args = {"reprocess_input_data": True, 
#     "overwrite_output_dir": True, 
#     "num_train_epochs": 30, 
#     'fp16': False,
#     "use_early_stopping": True,
#     'learning_rate': 4e-5,
#     'evaluate_during_training' : True,
#     'early_stopping_metric' : 'f1_macro',
#     'early_stopping_metric_minimize': False,
#     'save_model_every_epoch' : False, 
#     'train_batch_size' : 1,
# }

# split_0 = r.read_from_split(split_path + '/split_0.tsv')
# split_1 = r.read_from_split(split_path + '/split_1.tsv')

# bm = BERTMultilabel()

# train = split_0
# test = split_1

## Either download the fine-tuned model or fine-tune from scratch 

# finetuned_model_path = curr_path + '/outputs/'

# model = bm.train_model(training_data = train, training_args = args, base_model_path = finetuned_model_path, eval_data = test)