import sys 
import os

root_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

bert_path = os.path.join(root_path, 'base-model/bert/')
sys.path.append(bert_path)

prep_path = os.path.join(root_path, 'preprocess')
sys.path.append(prep_path)

split_path = os.path.join(root_path, 'dataset/split')

curr_path = os.path.dirname(os.path.abspath(__file__))

from bert_multilabel import BERTMultilabel
from reader import Reader
from simpletransformers.classification import MultiLabelClassificationModel
from meter_preprocess import MeterPreprocess
import pandas as pd

# Train a meter classification model

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
mp = MeterPreprocess()

train_meter = r.read_from_txt(curr_path + '/meter-train.txt')
test_meter = r.read_from_txt(curr_path + '/meter-test.txt')

model = MultiLabelClassificationModel(
    "bert",
    'bert-large-uncased',
    num_labels = 8,
    args = {"reprocess_input_data": True, 
        "overwrite_output_dir": True, 
        "num_train_epochs": 30, 
        'fp16': False, 
        'learning_rate': 4e-5,
        'early_stopping_metric_minimize': False,
        'save_model_every_epoch' : False, 
        'train_batch_size' : 16,
        },
    use_cuda = False
)

temp = [mp.assign_label_meter(s) for s in train_meter]

model.train_model(pd.DataFrame(temp, columns=['text', 'labels']))

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

# r = Reader()

# split_0 = r.read_from_split(split_path + '/split_0.tsv')
# split_1 = r.read_from_split(split_path + '/split_1.tsv')

# bm = BERTMultilabel()

# train = split_0
# test = split_1

## Either download the fine-tuned model or fine-tune from scratch 

# finetuned_model_path =  curr_path + '/outputs/'

# model = bm.train_model(training_data = train, training_args = args, base_model_path = finetuned_model_path, eval_data = test)
