# BERT-COIE
A BERT based Chinese Open Information Extraction Method

## Requirements
```
python==3.6.5
Tensorflow>=1.12.0
pyltp==0.2.1
```

## Usage:
### Step 1: Download the BERT pre-trained checkpoint
Download the BERT-base Chinese checkpoints from https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip 

to ./bert-model/
### Step 2: Download the LTP data
Download the LTP model ltp_data_v3.4.0.zip from http://ltp.ai/download.html and unzip it
### Step 3: Ready your dataset
One row for each piece of data, and a piece of data should be formed in json format.

It should contain a field called "natural", which is the original sentecen, and a field called "tag_seq" which is the tag sequence of

the sentence, one tag to one Chinese character. The tag scheme is "BIO" scheme. B-E1 for the begining of the argument 1, I-E1 for the 

following words of argument 1. Similiar for E2(argument 2) and R(relation). 

The full data set should be put into ./data/

Then use add_features/additional_features.py to add the POS and DP features
```
python add_features/additional_features.py -sp -rf data/saoke.json -train data/train.json -test data/test.json -ltp ../ltp_data_v3.4.0
```
### Step 4: Training and testing
run BERT_COIE.py to train and test 
### Step 5: Post-processing and get P, R, F1
run utils/post_process.py
```
python post_processing.py -data_dir <path/to/the/testset> -output_dir <path/to/the_output_dir_of_the_model>
```
Updating in succession
