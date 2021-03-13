from __future__ import unicode_literals
from hazm import *
import pandas as pd
import xlrd
import re
import csv
from csv import reader
import numpy as np
from parsivar import Tokenizer
import nltk
from nltk.tokenize import word_tokenize as nltk_word_tokenizer
from transformers import BertTokenizer, BertConfig

########################  Functions ############################

def hazm_tokenizer(string):
  return word_tokenize(string)

def parsivar_tokenizer(string):
  return parsivarTokenizer.tokenize_words(string)

def nltk_tokenizer(string):
  return nltk_word_tokenizer(string)

def transformers_tokenizer(string):
  intermediate = berttokenizer.tokenize(string)
  intermediate = '||'.join(intermediate)
  intermediate = intermediate.replace('||##', '')
  result = intermediate.split('||')
  return result

def transformers_tokenizer_compound(string):
  intermediate = berttokenizer.tokenize(string)
  intermediate = '||'.join(intermediate)
  intermediate = intermediate.replace('||##', '÷')
  result = intermediate.split('||')
  return result

def token_evaluation_compounds(actual_serie, text_serie, tokenizer):
  num_total_compounds = 0
  num_predict_compounds = 0
  actual_tokens_list = actual_serie.apply(lambda string:[item.strip() for item in string.split('=')]).to_list()
  actual_compounds_list = []
  for tokens in actual_tokens_list:
    compounds_list = []
    for token in tokens:
      if '÷' in token:
        compounds_list.append(token.strip().replace('÷', ' '))
        num_total_compounds += 1
    if len(compounds_list) >= 1:
      actual_compounds_list.append(compounds_list)
    else:
      actual_compounds_list.append('no compound')

  prediction_tokens_list = text_serie.apply(tokenizer).to_list()
  precision, recall, fmeasure = [], [], []
  temp_fmeasure = 0
  for index in range(len(actual_compounds_list)):
    print("\nactual compound: ",actual_compounds_list[index])
    print("prediction tokens: ",prediction_tokens_list[index])
    temp_recall = len([item for item in actual_compounds_list[index] if item in prediction_tokens_list[index]])/len(actual_compounds_list[index])
    temp_precision = len([item for item in prediction_tokens_list[index] if item in actual_compounds_list[index]])/len(prediction_tokens_list[index])
    try:
      temp_fmeasure = 2*(temp_recall*temp_precision)/(temp_precision+temp_recall)
    except:
      pass
    precision.append(temp_precision); recall.append(temp_recall); fmeasure.append(temp_fmeasure)
  precision, recall, fmeasure = np.mean(precision), np.mean(recall), np.mean(fmeasure)
  print(f'Precision: {precision}, Recall: {recall}, F-measure: {fmeasure}')

def token_evaluation_compounds_transformers(actual_serie, text_serie, tokenizer):
  actual_tokens_list = actual_serie.apply(lambda string:[item.strip() for item in string.split('=')]).to_list()
  actual_compounds_list = []
  for tokens in actual_tokens_list:
    compounds_list = []
    for token in tokens:
      if '÷' in token:
        compounds_list.append(token)
    if len(compounds_list) >= 1:
      actual_compounds_list.append(compounds_list)
    else:
      actual_compounds_list.append('no compound')

  prediction_tokens_list = text_serie.apply(tokenizer).to_list()
  precision, recall, fmeasure = [], [], []
  temp_fmeasure = 0
  for index in range(len(actual_compounds_list)):
    print("\nactual compound: ",actual_compounds_list[index])
    print("prediction tokens: ",prediction_tokens_list[index])
    temp_recall = len([item for item in actual_compounds_list[index] if item in prediction_tokens_list[index]])/len(actual_compounds_list[index])
    temp_precision = len([item for item in prediction_tokens_list[index] if item in actual_compounds_list[index]])/len(prediction_tokens_list[index])
    try:
      temp_fmeasure = 2*(temp_recall*temp_precision)/(temp_precision+temp_recall)
    except:
      pass
    precision.append(temp_precision); recall.append(temp_recall); fmeasure.append(temp_fmeasure)
  precision, recall, fmeasure = np.mean(precision), np.mean(recall), np.mean(fmeasure)
  print(f'Precision: {precision}, Recall: {recall}, F-measure: {fmeasure}')

def token_evaluation(actual_serie, text_serie, tokenizer):
  actual_tokens_list = actual_serie.apply(lambda string:[item.strip().replace('÷', '') for item in string.split('=')]).to_list()
  prediction_tokens_list = text_serie.apply(tokenizer).to_list()
  precision, recall, fmeasure = [], [], []
  for index in range(len(actual_tokens_list)):
    temp_recall = len([item for item in actual_tokens_list[index] if item in prediction_tokens_list[index]])/len(actual_tokens_list[index])
    temp_precision = len([item for item in prediction_tokens_list[index] if item in actual_tokens_list[index]])/len(prediction_tokens_list[index])
    try:
      temp_fmeasure = 2*(temp_recall*temp_precision)/(temp_precision+temp_recall)
    except:
      pass
    precision.append(temp_precision); recall.append(temp_recall); fmeasure.append(temp_fmeasure)
  precision, recall, fmeasure = np.mean(precision), np.mean(recall), np.mean(fmeasure)
  print(f'Precision: {precision}, Recall: {recall}, F-measure: {fmeasure}')

################################# Performance #####################################

berttokenizer = BertTokenizer.from_pretrained('HooshvareLab/bert-fa-base-uncased', do_lower_case=False)
parsivarTokenizer = Tokenizer()

df = pd.read_excel('Tokenizer.xlsx', header=None)
df = df.iloc[:156, :2]
df.columns = ['text', 'label']

token_evaluation(df['label'], df['text'], hazm_tokenizer)

token_evaluation(df['label'], df['text'], parsivar_tokenizer)

token_evaluation(df['label'], df['text'], nltk_tokenizer)

token_evaluation(df['label'], df['text'], transformers_tokenizer)

#evaluation compounds
token_evaluation_compounds(df['label'], df['text'], hazm_tokenizer)

#evaluation compounds
token_evaluation_compounds(df['label'], df['text'], parsivar_tokenizer)

#evaluation compounds
token_evaluation_compounds(df['label'], df['text'], nltk_tokenizer)

#evaluation compounds
token_evaluation_compounds_transformers(df['label'], df['text'], transformers_tokenizer_compound)