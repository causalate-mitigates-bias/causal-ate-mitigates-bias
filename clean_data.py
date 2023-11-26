import pandas as pd
import numpy as np
import time
import csv, re, emoji, time, pandas as pd, numpy as np
from collections import defaultdict
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase
from word_utils import normalizeSentence, read_data_csv, read_text
from keras_preprocessing.sequence import pad_sequences
# from getHatebertScores import getHBScores, getListOfModelConfigs

import nltk, string
from nltk.tokenize import word_tokenize
from transformers import pipeline
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset


def clean_data(filedir="datasets/", filename="gao", filetype=".csv", index_column_name="id", labels_column="labels",
               text_column="text", words_to_filter=None, normalizer_sequence=None):
    if normalizer_sequence is None:
        normalizer_sequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    if words_to_filter is None:
        words_to_filter = ["@USER"]

    filepath = filedir + filename + filetype
    inputDF = pd.read_csv(filepath)
    inputDF = inputDF.set_index(index_column_name)
    inputDF[labels_column] = inputDF[labels_column].astype(bool)
    # Use str.replace() to remove the specified words from the 'text' column
    for word in words_to_filter:
        inputDF[text_column] = inputDF[text_column].str.replace(word, '', case=False)
    inputDF = normalize_sentence(inputDF,text_column=text_column,normalizer_sequence=normalizer_sequence)
    return inputDF

def normalize_sentence(inputDF,text_column,normalizer_sequence=None):
    if normalizer_sequence is None:
        normalizer_sequence = [NFD(), StripAccents(), Strip(), Lowercase()]

    inputDF['normalizedSentence'] = [normalizeSentence(sentence, normalizer_sequence) for sentence in inputDF[text_column]]
    inputDF[text_column] = inputDF['normalizedSentence']
    inputDF.drop('normalizedSentence', axis=1, inplace=True)
    inputDF = inputDF[inputDF[text_column] !='']
    return inputDF


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    # Set a random seed for reproducability
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    gao = clean_data(filename='gao')
    print("gao.columns =", gao.columns)
    print("gao.head(5) =", gao.head(5))


    zampieri = clean_data(filename='zampieri')
    print("zampieri.columns =", zampieri.columns)
    print("zampieri.head(5) =", zampieri.head(5))


    # Calculate and print the total time taken to run the code
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
