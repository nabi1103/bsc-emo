from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'emo_10':                                #Name of the dataset
        {'columns': {0:'tokens', 1:'label'},  #
         'label': 'label',                   #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}


# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
embeddingsPath = 'laser-embedding.tsv'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)

######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [1024], 'dropout': 0, 'featureNames': ['tokens'], 'miniBatchSize' : 8, 'earlyStopping' : 10, 
          'charEmbeddings' : 'lstm', 'charEmbeddingsSize' : 500, 'maxCharLength': 500, 'charLSTMSize': 50}

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/emo_01_results_laser_0.csv') #Path to store performance scores for dev / test
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" #Path to store models
model.fit(epochs=50)