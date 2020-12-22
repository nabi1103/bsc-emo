# Aesthetic Emotion Classification in Poetry from Small Data

This repository contains the Python implemetation for the base models and experiments described in the Bachelor Thesis of Nguyen, Thanh Tung Linh from the Department of Computer Science, TU Darmstadt. The thesis is supervised by [Dr. rer. pol. Steffen Eger](https://www.informatik.tu-darmstadt.de/aiphes/aiphes/people_7/mitarbeiter_4_detailseite_72000.en.jsp).

In this repository, we provide the following codes:

* Preprocessing codes for the dataset.
* Training codes for the models.
* Data transformation codes for the experiments (if required).

(Most of) The trained models that we trained and reported in the thesis can be found and downloaded [here](https://drive.google.com/drive/folders/1UIk5FwmX4iY8DT049m1DzgKDm4vynoWN?usp=sharing).


# Install requirements

The code is tested with Python 3.6

``` 
pip install - r requirements.txt
```

Setup virtual environment (optional) and install requirements:
``` 
virtualenv --system-site-packages -p python.6 venv
source venv/bin/activate
venv/bin/pip3 install -r requirements.txt
```

# Running the codes

## Training Baseline Models

### VAD-lexicon based model

Change the working directory

```
cd base-model/lexicon
```

To train the model, look into `vad.py` and uncomment the codes from line 109 to line 120. Run the code with
```
python vad.py
```

### BiLSTM-CNN-CRF model 

This implementation is based on https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf

Install additional requirements. It is recommended to setup a different venv to run this experiment separately from other experiments due to conflicting requirements.

```
venv/bin/pip3 install -r base-model/bilstm/requirements.txt
```
or 

```
pip install base-model/bilstm/requirements.txt
```

If there is an error with Tensorflow, downgrade it to 1.5.0 should help

```
pip install tensorflow==1.5.0
```

Change the working directory

```
cd base-model/bilstm
```

Train the model

```
python train_emo.py
```

### BERT-based implementation

This implementation is based on https://github.com/ThilinaRajapakse/simpletransformers

Change the working directory

```
cd base-model/bert
```

To train the model, look into `bert_multilabel.py` and uncomment the codes from line 91 to line 120. It is recommended to use GPU to train. Enable training with GPU by changing `use_cuda = False` to `True` at line 51. 

In order to run the training code, a base model of BERT-large (about 1.34 GB in size) needs to be downloaded. This is done automatically.

Train model.
```
python bert_multilabel.py
```

After training, make sure to comment out the code from line 91 to 120 again since we import this class again to train models in other experiments.

## Transfer Learning Experiments

### Poetry fine-tuning

This was used in the Poetry Fine-tuning experiment (Section 5.1.1)

Change the working directory

```
cd transfer-learning/poetry-finetuning
```

The training code for this experiment has two parts, which need to be executed in order:
1. Pre-train (or fine-tune) BERT-large with raw poetry
2. Train the classification models using the model trained in Step 1

The first part can be executed by running the code

```
python poetry_finetuning.py
```

After the training has completed, the resulting model can be found in `outputs` under the same working directory. To ensure reproducibility, we provide our poetry fine-tuned model, which should be downloaded [here](https://drive.google.com/drive/folders/1-fmFdqen7tH-mYWIN2prUNURihEAd_Te?usp=sharing) and extracted to the `outputs` folder. If done correctly, the folder structure should look like this

```
bsc-emo/transfer-learning/poetry-finetuning/outputs/
                                                    |--------pytorch_model.bin
                                                    |--------config.json
                                                    |--------vocab.txt
                                                    |--------training_args.bin
                                                    |--------.....

```
After first part is completed, comment out the first part (line 24 to 37), uncomment the second part (line 41 to 83) in `poetry_finetuning.py` and run the code again

```
python poetry_finetuning.py
```

### Emotion fine-tuning

This was used in the Emotion Fine-tuning experiment (Section 5.1.2)

Change the working directory

```
cd transfer-learning/emotion-finetuning
```

The training code for this experiment has two parts, which need to be executed in order:
1. Fine-tune BERT-large with emotion classification task
2. Train the classification models using the model trained in Step 1

The first part can be executed by running the code

```
python emotion_finetuning.py
```

After the training has completed, the resulting model can be found in `outputs` under the same working directory. To ensure reproducibility, we provide our emotion fine-tuned model, which should be downloaded [here](https://drive.google.com/drive/folders/1-FzjoLAsmLWPVgpW6ri27FW3huM_bqnO?usp=sharing) and extracted to the `outputs` folder. If done correctly, the folder structure should look like this

```
bsc-emo/transfer-learning/emotion-finetuning/outputs/
                                                    |--------pytorch_model.bin
                                                    |--------config.json
                                                    |--------vocab.txt
                                                    |--------training_args.bin
                                                    |--------.....

```
After the first part is completed, comment out the first part (line 24 to 61), uncomment the second part (65 to 83) in `emotion_finetuning.py` and run the code again

```
python emotion_finetuning.py
```

### Meter fine-tuning

This was used in the Meter Fine-tuning experiment (Section 5.1.3)

Change the working directory

```
cd transfer-learning/meter-finetuning
```

The training code for this experiment has two parts, which need to be executed in order:
1. Fine-tune BERT-large with meter classification task
2. Train the classification models using the model trained in Step 1

The first part can be executed by running the code

```
python meter_finetuning.py
```

After the training has completed, the resulting model can be found in `outputs` under the same working directory. To ensure reproducibility, we provide our meter fine-tuned model, which should be downloaded [here](https://drive.google.com/drive/folders/1x5yJ40PPBlcy88P8jUCJ2DrgZWWKTOB6?usp=sharing) and extracted to the `outputs` folder. If done correctly, the folder structure should look like this

```
bsc-emo/transfer-learning/meter-finetuning/outputs/
                                                    |--------pytorch_model.bin
                                                    |--------config.json
                                                    |--------vocab.txt
                                                    |--------training_args.bin
                                                    |--------.....

```
After the first part is completed, comment out the first part (line 25 to 64), uncomment the second part (line 68 to 97) in `meter_finetuning.py` and run the code again

```
python meter_finetuning.py
```

## Data Augmentation Experiments

### Back-translation

This was used in the Back-translation experiment (Section 5.2.2)

**IMPORTANT** The Translation API is currently having some problem (https://github.com/ssut/py-googletrans/issues/264). However, the back-translated text using an older version of this API is still available for training in the working directory.

Change the working directory

```
cd data-augmentation/back-translation
```

The augmented data is already available in the folder. Train the model by running

```
python back_translation.py
```

### Oversampling

This was used in the Oversampling experiment (Section 5.2.1)

Change the working directory

```
cd data-augmentation/oversampling
```

The augmented data is already available in the folder. Train the model by running

```
python oversample.py
```

### Stanza Shuffling

This was used in the Stanza Shuffling experiment (Section 5.2.4)

Change the working directory

```
cd data-augmentation/stanza-shuffling
```

The augmented data is already available in the folder. Train the model by running

```
python stanza_shuffling.py
```

### Words Replacement

This was used in the Words Replacement experiment (Section 5.2.3)

**NOTE** Running the word replacing functions require large amount of memory available (>10GB).

Change the working directory

```
cd data-augmentation/word-replacing
```

The augmented data is already available in the folder. Train the model by running

```
python word_replacing.py
```

## Cross-lingual Experiments

### Baseline multilingual BERT (m-BERT) model

This was used to train the baseline m-BERT model (Section 5.3.1)

Change the working directory

```
cd cross-lingual/baseline-mbert
```

Train the model

```
python baseline_mbert.py
```

### Fine-tune m-BERT with German poetry

This was used in the m-BERT Fine-tuning experiment (Section 5.3.4)

Change the working directory

```
cd cross-lingual/baseline-mbert
```

The training code for this experiment has two parts, which need to be executed in order:
1. Fine-tune m-BERT with raw German data
2. Train the classification models using the model trained in Step 1

The first part can be executed by running the code

```
python finetune_mbert.py
```

After the training has completed, the resulting model can be found in `outputs` under the same working directory. To ensure reproducibility, we provide our German fine-tuned model, which should be downloaded [here](https://drive.google.com/drive/folders/103HdxRb7LH-tHv27w4EpvlFvO6sxeIZd?usp=sharing) and extracted to the `outputs` folder. If done correctly, the folder structure should look like this

```
bsc-emo/cross-lingual/finetune-mbert/outputs/
                                            |--------pytorch_model.bin
                                            |--------config.json
                                            |--------vocab.txt
                                            |--------training_args.bin
                                            |--------.....

```
After the first part is completed, comment out the first part (line 25 to 38), uncomment the second part (line 42 to 88) in `finetune_mbert.py` and run the code again

```
python finetune_mbert.py
```

### Intermediate task training with German poetry emotion classification

This was used in the m-BERT intermediate task training experiment (Section 5.3.5)

Change the working directory

```
cd cross-lingual/baseline-mbert-cls
```

The training code for this experiment has two parts, which need to be executed in order:
1. Fine-tune m-BERT with German poetry emotion classification
2. Train the classification models using the model trained in Step 1

The first part can be executed by running the code

```
python finetune_mbert_cls.py
```

After the training has completed, the resulting model can be found in `outputs` under the same working directory. To ensure reproducibility, we provide our German poetry emotion classification fine-tuned model, which should be downloaded [here](https://drive.google.com/drive/folders/1fho1TGenK946yJjPppBtTwDtIt8FYA9h?usp=sharing) and extracted to the `outputs` folder. If done correctly, the folder structure should look like this

```
bsc-emo/cross-lingual/finetune-mbert-cls/outputs/
                                                |--------pytorch_model.bin
                                                |--------config.json
                                                |--------vocab.txt
                                                |--------training_args.bin
                                                |--------.....

```
After the first part is completed, comment out the first part (line 25 to 63), uncomment the second part (line 67 to 92) in `finetune_mbert_cls.py` and run the code again

```
python finetune_mbert_cls.py
```

## Ensemble

Change the working directory

```
cd ensemble
```

We provide the binary predictions and the raw output probabilities from all of the models listed in Section 5.4 in the folder `predictions` and `raws`, respectively. 

The code to run the ensembles is commented out in `ensemble.py`. Simply uncomment them and run again with

```
python ensemble.py
```

should provide identical results to the ones reported in the thesis.

In case there's an error with matplotlib, reinstalling Pillow might have

```
pip install --upgrade --force-reinstall pillow
```


## LIME

Change the working directory

```
cd analysis
```

Since LIME needs all the trained classification models to make explanations, we provide all the LIME explanations from our models in `json` folder for quick assessment. Running the code will visualize these explanations. 

```
python lime_analysis.py
```