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
from simpletransformers.language_modeling import LanguageModelingModel
from reader import Reader
import pandas as pd

# Fine-tuning BERT-large with poetry 

finetune_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    'fp16': False,
    "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>", '</br>'],
    "evaluate_during_training" : True,
    'save_model_every_epoch': False,
    'num_train_epochs': 10,
    'train_batch_size' : 16,
}

model = LanguageModelingModel('bert', 'bert-large-uncased', args = finetune_args, use_cuda = False)

model.train_model(curr_path + "/raw-poetry-train.txt", eval_file = curr_path + '/raw-poetry-test.txt')

## Train an emotion classifier with the fine-tuned model from above

# def f1_evaluate(true, pred):
#     for p in pred:
#         for i in range(len(p)):
#             if p[i] >= 0.5:
#                 p[i] = 1
#             else:
#                 p[i] = 0

#     score = f1_score(true, pred, average = 'macro')
#     label = f1_score(true, pred, average = None)
#     print(score)
#     print(label)
    
#     return score

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
