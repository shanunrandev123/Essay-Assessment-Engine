import numpy as np
import pandas as pd
import warnings
import logging
import os
import shutil
import json
import string
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
# from transformers import DataCollatorWithPadding
# from transformers import TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error
import torch
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
import spacy
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade


from nltk.sentiment.vader import SentimentIntensityAnalyzer

import textstat


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')


warnings.simplefilter("ignore")




df = pd.read_csv(r"C:\Users\Asus\Downloads\train.csv (6)\train.csv")

df = df[:100]


sid = SentimentIntensityAnalyzer()

df['sentences'] = df['full_text'].apply(lambda x: sent_tokenize(x))

df['words'] = df['full_text'].apply(lambda x : word_tokenize(x))


df['avg_word_len'] = df['words'].apply(lambda x : np.mean([len(w) for w in x]))

df['total_sentences'] = df['sentences'].apply(lambda x : len(x))

df['paragraph_count'] = df['full_text'].apply(lambda x : x.count('\n\n')) + 1

df['unique_words'] = df['words'].apply(lambda x : len(list(set(x))))


df['total_words'] = df['words'].apply(lambda x : len(x))

df['type_token_ratio'] = df['unique_words'] / df['total_words']

df['avg_sentence_len'] = df['sentences'].apply(lambda x : np.mean([len(sent.split(',')) for sent in x]))

df['Noun_count'] = df['words'].apply(lambda x : len([w for w in x if nltk.pos_tag([w])[0][1] in ['NN','NNP','NNPS','NNS']]))

df['Verb_count'] = df['words'].apply(lambda x : len([w for w in x if nltk.pos_tag([w])[0][1] in ['VB','VBD','VBG','VBN','VBP','VBZ']]))


df['Adj_count'] = df['words'].apply(lambda x : len([w for w in x if nltk.pos_tag([w])[0][1] in ['JJ','JJR','JJS']]))

df['pronoun_count'] = df['words'].apply(lambda x : len([w for w in x if nltk.pos_tag([w])[0][1] in ['PRP','PRP$','WP','WP$']]))

df['word_to_sent_ratio'] = df['total_words'] / df['total_sentences']


#flesch_reading_ease  - evaluates the readability of English text and assess how easy or difficult is it to read text

df['flesch_reading_index'] = df['full_text'].apply(lambda x : flesch_reading_ease(x))

#smog index - estimates the years of education needed to understand a piece of writing

df['smog_index'] = df['full_text'].apply(lambda x : textstat.smog_index(x))



df['sentiment_scores'] = df['full_text'].apply(lambda x : sid.polarity_scores(x))








print(df.head(10))





# def striphtml(data):
#     p = re.compile(r'<.*?>')
#     return p.sub('', data)




# def remove_punctuation(text):
#     # Define a regular expression pattern to match punctuation
#     pattern = r'[^\w\s]'  # Matches anything that is not a word character or whitespace

#     # Use re.sub() to replace matched patterns with an empty string
#     cleaned_text = re.sub(pattern, '', text)

#     return cleaned_text




# # Define a dictionary of contractions and their expanded forms
# contraction_dict = {
#     "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
#     "could've": "could have",
#     "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
#     "don't": "do not", "hadn't": "had not",
#     "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
#     "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
#     "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
#     "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have",
#     "isn't": "is not",
#     "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
#     "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
#     "mightn't've": "might not have",
#     "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
#     "needn't": "need not", "needn't've": "need not have",
#     "o'clock": "of the clock",
#     "oughtn't": "ought not", "oughtn't've": "ought not have",
#     "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
#     "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
#     "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
#     "so've": "so have", "so's": "so is",
#     "that'd've": "that would have", "that's": "that is",
#     "there'd've": "there would have", "there's": "there is",
#     "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
#     "they've": "they have",
#     "to've": "to have", "wasn't": "was not", "weren't": "were not",
#     "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
#     "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
#     "what've": "what have",
#     "when's": "when is", "when've": "when have",
#     "where'd": "where did", "where's": "where is", "where've": "where have",
#     "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
#     "why've": "why have",
#     "will've": "will have", "won't": "will not", "won't've": "will not have",
#     "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
#     "y'all": "you all", "y'alls": "you alls", "y'all'd": "you all would", "y'all'd've": "you all would have",
#     "y'all're": "you all are",
#     "y'all've": "you all have", "you'd": "you had", "you'd've": "you would have", "you'll": "you you will",
#     "you'll've": "you you will have",
#     "you're": "you are", "you've": "you have"
# }





# def expand_contractions(text, contraction_dict):
#     # Define a regular expression pattern to match contractions
#     pattern = re.compile(r'\b(' + '|'.join(contraction_dict.keys()) + r')\b')

#     # Function to replace each matched contraction with its expanded form
#     def replace(match):
#         return contraction_dict[match.group(0)]

#     # Use re.sub() to substitute each contraction with its expanded form
#     expanded_text = pattern.sub(replace, text)

#     return expanded_text




# def preprocess(text):
#     text = text.lower()

#     text = striphtml(text)

#     text = re.sub('@\w+', '', text)

#     text = re.sub("\d+", '', text)

#     text = re.sub(r'http\S+', '', text)

#     text = re.sub(r"\s+", " ", text)

#     text = re.sub(r"\.+", ".", text)

#     text = re.sub(r"\,+", ",", text)

#     text = text.strip()

#     return text





