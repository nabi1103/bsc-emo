import nltk
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
from sklearn.metrics import f1_score
import datetime
import csv
import sys

if len(sys.argv) < 3:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]

with open(inputPath, 'r') as f:
    data = f.readlines()

label = []
temp = []
temp_label = []
agg_label = []

data = [d.strip().split('\t') for d in data]
text = [d[0] for d in data]

for d in data:
  if len(d) > 1:
    temp_label.append(d[1])
  else:
    temp_label.append(d[0])

stanza_label = []
for l in temp_label:
    if l != '':
        stanza_label.append(l)
    else:
        label.append(stanza_label)
        stanza_label = []

for l in label:
  agg_label.append(list(set(l)))

stanza = ''
for line in text:
    if line != '':
        stanza += (line + ' ')
    else:
        temp.append(stanza)
        stanza = ''

lstmModel = BiLSTM.loadModel(modelPath)

sentences = [{'tokens': sent.split(' ')[:-1]} for sent in temp]
addCharInformation(sentences)
addCasingInformation(sentences)
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
tags = lstmModel.tagSentences(dataMatrix)

pred = []
tagged_stanza = []
agg = []

for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    stanzaLines = []
    stanzaLabel = []
    for tokenIdx in range(len(tokens)):
      tokenTags = []
      for modelName in sorted(tags.keys()):
        tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])
        stanzaLabel.append(tokenTags)
        stanzaLines.append("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    tagged_stanza.append(stanzaLines)
    pred.append(stanzaLabel)

for s in tagged_stanza:
  for l in s:
    print(l)
  print('\n')

for stanza in pred:
  temp_agg = []
  for l in stanza:
    temp_agg.append(l[0])
  agg.append(list(set(temp_agg)))

def vectorize(labels):
  label_name = ['Suspense', 'Awe/Sublime', 'Sadness', 'Annoyance', 'Uneasiness', 'Beauty/Joy', 'Vitality', 'Humor']
  vector = [0, 0, 0, 0, 0, 0, 0, 0]
  for l in labels:
    idx = label_name.index(l)
    vector[idx] = 1
  return vector

agg_label = [vectorize(l) for l in agg_label]
agg = [vectorize(l) for l in agg]


f1_macro = f1_score(agg_label, agg, average='macro')
f1_all = f1_score(agg_label, agg, average=None)

print(f1_macro)
print(f1_all)
"""
with open('bilstm_cnn_crf_' + str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_' + '10.tsv', 'wt', encoding='utf-8', newline='') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['Name', 'Score'])
    tsv_writer.writerow(['f1_macro', str(f1_macro)])
    tsv_writer.writerow(['f1_all', str(f1_all)])
"""