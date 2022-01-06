import pandas as pd 
import numpy as np
import re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import regexp_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#import gensim
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
#sp = spacy.load('en_core_web_sm')
stopwords_nltk = list(stopwords.words('english'))
#stopwords_spacy = list(sp.Defaults.stop_words)
#stopwords_gensim = list(gensim.parsing.preprocessing.STOPWORDS)
all_stopwords = []
all_stopwords.extend(stopwords_nltk)
#all_stopwords.extend(stopwords_spacy)
#all_stopwords.extend(stopwords_gensim)
# all unique stop words
all_stopwords = list(set(all_stopwords))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split
from flaml import AutoML
#from evalml import AutoMLSearch
from sklearn.metrics import accuracy_score
