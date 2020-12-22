import sys 
import os
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from simpletransformers.classification import MultiLabelClassificationModel

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

# Prepare split_0 for testing
test_data_0 = [r.assign_label(s) for s in split_0]
test_text_0 = [s[0] for s in test_data_0]
test_label_0 = [s[1] for s in test_data_0]

# Prepare split_1 for testing
test_data_1 = [r.assign_label(s) for s in split_1]
test_text_1 = [s[0] for s in test_data_1]
test_label_1 = [s[1] for s in test_data_1]

model_path =  curr_path + "/model/"

model = MultiLabelClassificationModel(
    'bert',
    model_path,
    num_labels = 8,
    use_cuda = True,
    args = {
        'silent' : True,
    }
)
## Edit the following lines to change the splits usd for testing
## The example below shows how to test with split_1 as the test set. split_1 has 2 components: test_text_1 for the raw text, and test_label_1 for the annotated label. The components have already been initiated above
predictions, dummy = model.predict(test_text_1) # Load the raw text and predict using the model
pred = pd.DataFrame(predictions) # Format the label predictions
pd_label = pd.DataFrame(test_label_1) # Load the annotated labels
## End editing

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

f1_macro = f1_score(pd_label, pred, average = 'macro') # Compare the predicted labels and annotated labels, then calculate the F1-macro score
f1_all = f1_score(pd_label, pred, average = None) # Compare the predicted labels and annotated labels, then calculate F1-macro score for each class

print(f1_macro)
print(f1_all) # The labels are as follow : ['suspense', 'awe/sublime', 'sadness', 'annoyance', 'uneasiness', 'beauty/joy', 'vitality', 'humor']